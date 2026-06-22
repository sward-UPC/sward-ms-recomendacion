"""Variante EXPERIMENTAL del entrenamiento SAKT que incorpora el FORMATO del
recurso (`tipo_recurso`) como parte de la unidad de conocimiento ("skill").

CONTRIBUCIÓN DE MODELADO (tesis)
--------------------------------
El SAKT de producción (``train_sakt.py``) modela cada interacción como el par
``(concepto, acierto)``. Con ello el modelo NO distingue cómo practicó el
estudiante un mismo concepto: "AVL leído" y "AVL practicado" colapsan al mismo
skill. Esta variante define el skill como el PAR COMPUESTO
``(concepto, tipo_recurso)``. Así el SAKT puede aprender patrones del estilo
"este alumno consolida mejor cuando practica que cuando lee", porque cada
combinación concepto×formato ocupa su propia fila de embedding y su propia
dinámica de atención.

Es una variante DELIBERADAMENTE separada del pipeline de producción: NO modifica
``train_sakt.py`` ni ``modelo_sakt.py``. El checkpoint que genera es compatible
en estructura con el loader actual (mismas claves), pero añade:
    - ``formato_aware: True``      → bandera para que el loader/inferencia sepa
                                     que ``concept_index`` está sobre pares.
    - ``concept_index``            → dict ``"concepto||formato" -> idx`` (compuesto).
    - ``formato_index``            → dict ``formato -> idx`` (catálogo de formatos
                                     observados; informativo / para análisis).
    - ``sep_token``                → separador usado al serializar el par.

DEPENDENCIA DE DATOS (pendiente en el pipeline)
-----------------------------------------------
La entrada actual de ``/internal/training-data`` (ms-trazabilidad) entrega
``{estudiante_id, concepto, correcta, orden}`` y NO incluye ``tipo_recurso``.
Para que esta variante use formato REAL, el export debe añadir ``tipo_recurso``
por interacción (el dato ya existe en ms-trazabilidad). Mientras tanto, las
interacciones sin formato caen en el formato ``"generico"`` (degradación segura:
equivale, para esos casos, al comportamiento de un único skill por concepto).

NO ejecutar localmente sin pyKT instalado. Validado con ``py_compile`` y tests
puros sobre las funciones de indexado.
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constantes de modelado
# ---------------------------------------------------------------------------

# Separador para serializar el par (concepto, formato) en una sola clave string.
# Se elige una secuencia improbable dentro de un nombre de concepto/formato.
SEP_TOKEN = "||"

# Formato por defecto cuando la interacción no trae ``tipo_recurso``. Mantener
# este valor estable: forma parte de las claves del índice del checkpoint.
FORMATO_GENERICO = "generico"


# ===========================================================================
# FUNCIONES PURAS (sin torch / sin pyKT) — testeables en memoria
# ===========================================================================


def normalizar_formato(tipo_recurso) -> str:
    """Normaliza el formato de un recurso a una etiqueta canónica.

    Reglas:
      - ``None`` / vacío / solo espacios → ``FORMATO_GENERICO``.
      - en otro caso, se hace ``str``, ``strip`` y ``lower`` para evitar que
        "Quiz", "quiz" y " quiz " sean formatos distintos.

    Es pura: no toca disco ni estado global. Se aísla para poder testear el
    mapeo sin depender del resto del pipeline.
    """
    if tipo_recurso is None:
        return FORMATO_GENERICO
    etiqueta = str(tipo_recurso).strip().lower()
    return etiqueta if etiqueta else FORMATO_GENERICO


def skill_compuesto(concepto, tipo_recurso, sep: str = SEP_TOKEN) -> str:
    """Construye la clave de skill compuesto ``"concepto<sep>formato"``.

    El concepto se normaliza solo con ``str`` (se respeta su capitalización,
    porque del lado de inferencia el concepto llega tal cual de Moodle); el
    formato se normaliza con :func:`normalizar_formato`.
    """
    return f"{str(concepto)}{sep}{normalizar_formato(tipo_recurso)}"


def cargar_secuencias(data):
    """Agrupa interacciones por estudiante y las ordena temporalmente.

    ``data`` es una lista de dicts con al menos ``estudiante_id``, ``concepto``
    y ``correcta``; opcionalmente ``orden`` (fecha/posición para ordenar) y
    ``tipo_recurso`` (formato). Devuelve una lista de secuencias, cada una una
    lista de ``(skill_compuesto, acierto_int)`` ordenada por ``orden``.

    Acepta ``data`` ya deserializado (lista) para poder testear en memoria.
    """
    por_est = defaultdict(list)
    for r in data:
        formato = r.get("tipo_recurso")  # puede no venir → generico
        clave = skill_compuesto(r["concepto"], formato)
        por_est[r["estudiante_id"]].append(
            (str(r.get("orden", "")), clave, bool(r["correcta"]))
        )
    seqs = []
    for _, filas in por_est.items():
        filas.sort(key=lambda x: x[0])  # por fecha/orden
        seqs.append([(c, 1 if ok else 0) for _, c, ok in filas])
    return seqs


def construir_indice(seqs):
    """Índice de skills compuestos ``"concepto||formato" -> idx`` (ordenado).

    El orden alfabético hace el índice determinista entre corridas (importante
    para reproducibilidad de la tesis y para comparar checkpoints).
    """
    skills = sorted({c for s in seqs for c, _ in s})
    return {c: i for i, c in enumerate(skills)}


def construir_indice_formatos(concept_index, sep: str = SEP_TOKEN):
    """Catálogo informativo ``formato -> idx`` extraído de las claves compuestas.

    No participa de la red (el modelo solo ve el índice compuesto); sirve para
    el panel admin y para análisis post-hoc (cuántos formatos vio el modelo).
    """
    formatos = sorted({clave.split(sep, 1)[1] for clave in concept_index})
    return {f: i for i, f in enumerate(formatos)}


def cargar_secuencias_desde_archivo(path: str):
    """Wrapper de :func:`cargar_secuencias` que lee el JSON de ``path``."""
    with open(path) as fh:
        return cargar_secuencias(json.load(fh))


# ===========================================================================
# MÉTRICA (idéntica a train_sakt.py para que el AUC sea comparable)
# ===========================================================================


def _auc(scores: list[float], labels: list[int]) -> float | None:
    """AUC-ROC por rangos (Mann-Whitney U). None si no hay ambas clases."""
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


# ---------------------------------------------------------------------------
# Hiperparámetros (overridibles por env) — mismos defaults que producción.
# ---------------------------------------------------------------------------
SEQ_LEN = int(os.environ.get("SEQ_LEN", "64"))
EMB_SIZE = int(os.environ.get("EMB_SIZE", "64"))
N_HEADS = int(os.environ.get("N_HEADS", "4"))
DROPOUT = float(os.environ.get("DROPOUT", "0.2"))
N_LAYERS = int(os.environ.get("N_LAYERS", "2"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
LR = float(os.environ.get("LR", "1e-3"))
BATCH = int(os.environ.get("BATCH", "16"))

DATA_FILE = os.environ.get("DATA_FILE", "training_data.json")
OUT_FILE = os.environ.get("OUT_FILE", "model_formato.pth")


def to_tensors(seq, concept_index):
    """Replica el formato de modelo_sakt.py: q=pasado, r=respuestas, qry=shift.

    ``seq`` ya es una lista de ``(skill_compuesto, acierto)`` y
    ``concept_index`` mapea skills compuestos → idx. La lógica de padding/shift
    es idéntica a la de producción; lo único que cambia es el alfabeto de skills.
    """
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
    import torch
    from torch import nn

    seqs = cargar_secuencias_desde_archivo(DATA_FILE)
    concept_index = construir_indice(seqs)
    formato_index = construir_indice_formatos(concept_index)
    n_skills = len(concept_index)
    if n_skills < 2:
        raise SystemExit(f"Datos insuficientes: solo {n_skills} skills compuestos.")

    muestras = [t for s in seqs if (t := to_tensors(s, concept_index))]
    if len(muestras) < 2:
        raise SystemExit(f"Muestras insuficientes: {len(muestras)}.")
    print(
        f"Entrenando SAKT (format-aware) | estudiantes={len(seqs)} "
        f"skills_compuestos={n_skills} formatos={len(formato_index)} "
        f"muestras={len(muestras)}"
    )

    # Monkeypatch turtle (pyKT lo importa) para entornos sin display.
    import sys
    import types as _t

    if "turtle" not in sys.modules:
        m = _t.ModuleType("turtle")
        m.forward = lambda *a, **k: None
        sys.modules["turtle"] = m
    from pykt.models.sakt import SAKT

    # num_c = nº de skills compuestos (concepto×formato). Es la ÚNICA diferencia
    # estructural relevante frente a producción: el espacio de skills es mayor.
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

    # Split entrenamiento/validación (holdout ~15%) para un AUC honesto.
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
            out = model(qs[b], rs[b], qrys[b]).view(tgts[b].shape)
            loss = bce(out, tgts[b]) * masks[b]
            loss = loss.sum() / masks[b].sum().clamp(min=1)
            loss.backward()
            opt.step()
            total += loss.item()
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:>3} | loss {total / max(1, n_train // BATCH):.4f}")

    model.eval()

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
    except Exception as e:  # noqa: BLE001 — el AUC es informativo
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
            # --- claves específicas de la variante format-aware ---
            "formato_aware": True,
            "sep_token": SEP_TOKEN,
            "concept_index": concept_index,  # compuesto: "concepto||formato" -> idx
            "formato_index": formato_index,  # catálogo informativo formato -> idx
            "model_state_dict": model.state_dict(),
        },
        OUT_FILE,
    )
    print(
        f"✓ Checkpoint (format-aware) guardado en {OUT_FILE} "
        f"(n_skills={n_skills}, formatos={len(formato_index)}, seq_len={SEQ_LEN})"
    )


if __name__ == "__main__":
    main()
