"""Entrena el modelo SAKT (knowledge tracing) con las interacciones reales y
exporta un checkpoint compatible con `src/domain/services/modelo_sakt.py`.

Entrada (env DATA_FILE, JSON): lista de
    {"estudiante_id": str, "concepto": str, "correcta": bool, "orden": str}
(la entrega el endpoint interno GET /internal/training-data de ms-trazabilidad).

Salida (env OUT_FILE, default model.pth): checkpoint dict con las claves que
espera el loader: n_skills, seq_len, emb_size, n_heads, dropout, n_layers,
concept_index, model_state_dict.

Diseñado para correr en CI (GitHub Actions). El formato de secuencia (q, r, qry)
es idéntico al de inferencia en modelo_sakt.py.
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import torch
from torch import nn


def _auc(scores: list[float], labels: list[int]) -> float | None:
    """AUC-ROC por rangos (Mann-Whitney U). Devuelve None si no hay ambas clases."""
    pares = sorted(zip(scores, labels), key=lambda x: x[0])
    suma_rangos_pos = 0.0
    n_pos = 0
    for i, (_, label) in enumerate(pares, start=1):
        if label == 1:
            suma_rangos_pos += i
            n_pos += 1
    n_neg = len(pares) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    return (suma_rangos_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# Hiperparámetros (overridibles por env). El modelo es chico → entrena en CPU.
SEQ_LEN = int(os.environ.get("SEQ_LEN", "64"))
EMB_SIZE = int(os.environ.get("EMB_SIZE", "64"))
N_HEADS = int(os.environ.get("N_HEADS", "4"))
DROPOUT = float(os.environ.get("DROPOUT", "0.2"))
N_LAYERS = int(os.environ.get("N_LAYERS", "2"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
LR = float(os.environ.get("LR", "1e-3"))
BATCH = int(os.environ.get("BATCH", "16"))

DATA_FILE = os.environ.get("DATA_FILE", "training_data.json")
OUT_FILE = os.environ.get("OUT_FILE", "model.pth")


def cargar_secuencias(path: str):
    """Agrupa las interacciones por estudiante y las ordena temporalmente."""
    data = json.load(open(path))
    por_est = defaultdict(list)
    for r in data:
        por_est[r["estudiante_id"]].append(
            (str(r.get("orden", "")), str(r["concepto"]), bool(r["correcta"]))
        )
    seqs = []
    for est, filas in por_est.items():
        filas.sort(key=lambda x: x[0])  # por fecha
        seqs.append([(c, 1 if ok else 0) for _, c, ok in filas])
    return seqs


def construir_indice(seqs):
    conceptos = sorted({c for s in seqs for c, _ in s})
    return {c: i for i, c in enumerate(conceptos)}


def to_tensors(seq, concept_index):
    """Replica el formato de modelo_sakt.py: q=pasado, r=respuestas, qry=shift."""
    idxs = [concept_index[c] for c, _ in seq]
    res = [r for _, r in seq]
    idxs, res = idxs[-SEQ_LEN:], res[-SEQ_LEN:]
    L = len(idxs)
    if L < 2:
        return None
    q, r, qry, tgt = idxs[:-1], res[:-1], idxs[1:], res[1:]
    pad = SEQ_LEN - (L - 1)
    mask = [0] * pad + [1] * (L - 1)
    return (
        [0] * pad + q,
        [0] * pad + r,
        [0] * pad + qry,
        [0] * pad + tgt,
        mask,
    )


def main():
    seqs = cargar_secuencias(DATA_FILE)
    concept_index = construir_indice(seqs)
    n_skills = len(concept_index)
    if n_skills < 2:
        raise SystemExit(f"Datos insuficientes: solo {n_skills} conceptos.")

    muestras = [t for s in seqs if (t := to_tensors(s, concept_index))]
    if len(muestras) < 2:
        raise SystemExit(f"Muestras insuficientes: {len(muestras)}.")
    print(
        f"Entrenando SAKT | estudiantes={len(seqs)} conceptos={n_skills} muestras={len(muestras)}"
    )

    # Monkeypatch turtle (pyKT lo importa) para entornos sin display.
    import sys
    import types as _t

    if "turtle" not in sys.modules:
        m = _t.ModuleType("turtle")
        m.forward = lambda *a, **k: None
        sys.modules["turtle"] = m
    from pykt.models.sakt import SAKT

    model = SAKT(
        num_c=n_skills,
        seq_len=SEQ_LEN,
        emb_size=EMB_SIZE,
        num_attn_heads=N_HEADS,
        dropout=DROPOUT,
        num_en=N_LAYERS,
        emb_type="qid",
    )
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss(reduction="none")

    qs = torch.LongTensor([m[0] for m in muestras])
    rs = torch.LongTensor([m[1] for m in muestras])
    qrys = torch.LongTensor([m[2] for m in muestras])
    tgts = torch.FloatTensor([m[3] for m in muestras])
    masks = torch.FloatTensor([m[4] for m in muestras])

    # Split entrenamiento/validación (holdout 15%) para reportar un AUC honesto.
    # Con pocas muestras se entrena con todo y el AUC queda en None.
    n = len(muestras)
    torch.manual_seed(42)
    perm_split = torch.randperm(n)
    n_val = n // 7 if n >= 8 else 0
    val_idx = perm_split[:n_val]
    train_idx = perm_split[n_val:]

    n_train = len(train_idx)
    for ep in range(EPOCHS):
        perm = train_idx[torch.randperm(n_train)]
        total = 0.0
        for i in range(0, n_train, BATCH):
            b = perm[i : i + BATCH]
            opt.zero_grad()
            out = model(qs[b], rs[b], qrys[b])  # pyKT SAKT aplica sigmoid → probs
            out = out.view(tgts[b].shape)
            loss = bce(out, tgts[b]) * masks[b]
            loss = loss.sum() / masks[b].sum().clamp(min=1)
            loss.backward()
            opt.step()
            total += loss.item()
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:>3} | loss {total / max(1, n_train // BATCH):.4f}")

    model.eval()

    # Evaluación: AUC sobre las posiciones válidas del holdout de validación.
    test_auc = None
    try:
        if n_val > 0:
            with torch.no_grad():
                out_v = model(qs[val_idx], rs[val_idx], qrys[val_idx]).view(
                    tgts[val_idx].shape
                )
            sel = masks[val_idx] == 1
            test_auc = _auc(out_v[sel].tolist(), [int(t) for t in tgts[val_idx][sel]])
        if test_auc is not None:
            print(f"  AUC validación = {test_auc:.4f} (n_val={n_val})")
    except Exception as e:  # noqa: BLE001 — el AUC es informativo, no debe romper el train
        print(f"  (no se pudo calcular AUC: {e})")

    torch.save(
        {
            "n_skills": n_skills,
            "seq_len": SEQ_LEN,
            "emb_size": EMB_SIZE,
            "n_heads": N_HEADS,
            "dropout": DROPOUT,
            "n_layers": N_LAYERS,
            "learning_rate": LR,
            "epochs": EPOCHS,
            "test_auc": test_auc,
            "n_estudiantes": len(seqs),
            "n_muestras": n,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "concept_index": concept_index,
            "model_state_dict": model.state_dict(),
        },
        OUT_FILE,
    )
    print(
        f"✓ Checkpoint guardado en {OUT_FILE} (n_skills={n_skills}, seq_len={SEQ_LEN})"
    )


if __name__ == "__main__":
    main()
