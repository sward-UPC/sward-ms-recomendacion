import logging
import os
import sys
import tempfile
import types as _types
from datetime import timezone

from src.domain.entities.prediccion_kt import PrediccionKT
from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion

logger = logging.getLogger(__name__)


def _mock_turtle() -> None:
    """pyKT bug: qdkt.py importa turtle (requiere Tk). Mock antes de que pyKT cargue."""
    if "turtle" not in sys.modules:
        _mock = _types.ModuleType("turtle")
        _mock.forward = lambda *a, **kw: None
        sys.modules["turtle"] = _mock


def leer_info_modelo() -> dict:
    """Metadata REAL del modelo entrenado, leída del artefacto en S3 (no hardcode).

    Fuente de verdad para el panel admin: la fecha de último reentrenamiento sale
    del `LastModified` del objeto en S3 y los hiperparámetros/métricas del propio
    checkpoint (consistentes con el modelo desplegado). No construye la red SAKT;
    solo abre el dict del checkpoint, así que es liviano.
    """
    import boto3
    import torch

    from src.infrastructure.config.settings import settings

    s3 = boto3.client("s3", region_name=settings.aws_region)
    head = s3.head_object(
        Bucket=settings.aws_s3_model_bucket, Key=settings.sakt_model_s3_key
    )
    actualizado_en = head["LastModified"].astimezone(timezone.utc).isoformat()
    tam_bytes = int(head["ContentLength"])

    # Cache local; re-descarga si cambió el tamaño (p.ej. tras un reentrenamiento).
    local_path = os.path.join(tempfile.gettempdir(), "sakt_info.pth")
    if not os.path.exists(local_path) or os.path.getsize(local_path) != tam_bytes:
        s3.download_file(
            settings.aws_s3_model_bucket, settings.sakt_model_s3_key, local_path
        )

    ckpt = torch.load(local_path, map_location="cpu", weights_only=True)
    concept_index = ckpt.get("concept_index") or {}
    return {
        "n_skills": ckpt.get("n_skills"),
        "n_conceptos": len(concept_index) or ckpt.get("n_skills"),
        "seq_len": ckpt.get("seq_len"),
        "emb_size": ckpt.get("emb_size"),
        "n_heads": ckpt.get("n_heads"),
        "n_layers": ckpt.get("n_layers"),
        "dropout": ckpt.get("dropout"),
        "learning_rate": ckpt.get("learning_rate"),
        "epochs": ckpt.get("epochs"),
        "test_auc": ckpt.get("test_auc"),
        "n_estudiantes": ckpt.get("n_estudiantes"),
        "n_muestras": ckpt.get("n_muestras"),
        "entrenado_en": ckpt.get("trained_at") or actualizado_en,
        "actualizado_en": actualizado_en,
        "tam_bytes": tam_bytes,
    }


class ModeloSAKT:
    """SAKT wrapper. Mock en development, real en production cargando desde S3."""

    def __init__(self, version: str = "v1.0", mock: bool = True):
        self.version = version
        self._mock = mock
        self._model = None
        self._seq_len = 200
        # Mapa concepto(str)→índice entero del modelo. Vacío para modelos legacy
        # (assist2015, cuyos conceptos ya son ints); poblado para modelos Moodle.
        self._concept_index: dict[str, int] = {}
        if not mock:
            self._cargar_modelo()

    def _cargar_modelo(self) -> None:
        try:
            import torch
            import boto3

            from src.infrastructure.config.settings import settings

            s3 = boto3.client("s3", region_name=settings.aws_region)
            local_path = os.path.join(tempfile.gettempdir(), f"sakt_{self.version}.pth")
            if not os.path.exists(local_path):
                logger.info(
                    "Descargando modelo desde S3 | key=%s", settings.sakt_model_s3_key
                )
                s3.download_file(
                    settings.aws_s3_model_bucket, settings.sakt_model_s3_key, local_path
                )

            # El checkpoint contiene solo tensores y primitivas Python → weights_only=True es seguro
            checkpoint = torch.load(local_path, map_location="cpu", weights_only=True)

            n_skills = checkpoint["n_skills"]
            seq_len = checkpoint["seq_len"]
            emb_size = checkpoint["emb_size"]
            n_heads = checkpoint["n_heads"]
            dropout = checkpoint["dropout"]
            n_layers = checkpoint["n_layers"]
            self._seq_len = seq_len
            # Índice de conceptos del modelo (si fue entrenado con conceptos Moodle).
            self._concept_index = checkpoint.get("concept_index", {}) or {}

            _mock_turtle()

            import pykt.models.utils as _pykt_utils
            from pykt.models.sakt import SAKT, Blocks
            from pykt.models.utils import ut_mask

            _pykt_utils.device = "cpu"

            # Monkey-patch del bloque de atención para CAPTURAR los pesos (el forward
            # de pyKT los descarta). Idéntico al stock salvo que guarda _last_attn.
            def _blocks_forward_capture(self, q=None, k=None, v=None):

                q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
                causal_mask = ut_mask(seq_len=k.shape[0])
                attn_emb, attn_w = self.attn(
                    q, k, v, attn_mask=causal_mask, need_weights=True
                )
                self._last_attn = attn_w.detach()  # (batch, tgt_len, src_len)
                attn_emb = self.attn_dropout(attn_emb)
                attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)
                attn_emb = self.attn_layer_norm(q + attn_emb)
                emb = self.FFN(attn_emb)
                emb = self.FFN_dropout(emb)
                emb = self.FFN_layer_norm(attn_emb + emb)
                return emb

            Blocks.forward = _blocks_forward_capture

            model = SAKT(
                num_c=n_skills,
                seq_len=seq_len,
                emb_size=emb_size,
                num_attn_heads=n_heads,
                dropout=dropout,
                num_en=n_layers,
                emb_type="qid",
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self._model = model
            logger.info(
                "Modelo SAKT cargado | version=%s n_skills=%d seq_len=%d emb_size=%d",
                self.version,
                n_skills,
                seq_len,
                emb_size,
            )
            # print() para que quede visible en CloudWatch (el logger custom no
            # siempre se captura; uvicorn sí captura stdout).
            print(
                f"[SAKT] Modelo REAL cargado desde S3 | n_skills={n_skills} seq_len={seq_len}",
                flush=True,
            )
        except Exception as e:
            logger.error("Error cargando SAKT, usando mock: %s", e)
            print(f"[SAKT] FALLBACK A MOCK (no se pudo cargar de S3): {e}", flush=True)
            self._mock = True

    def predecir_dominio(self, secuencia: SecuenciaInteraccion) -> PrediccionKT:
        if self._mock or not secuencia.concepto_ids:
            return self._mock_prediccion(secuencia)
        return self._real_prediccion(secuencia)

    def _mock_prediccion(self, secuencia: SecuenciaInteraccion) -> PrediccionKT:
        if not secuencia.respuestas_correctas:
            prob = 0.5
        else:
            prob = sum(secuencia.respuestas_correctas) / len(
                secuencia.respuestas_correctas
            )
        n = len(secuencia.concepto_ids)
        pesos = [1.0 / n if n > 0 else 0.0] * n
        return PrediccionKT(
            estudiante_id=secuencia.estudiante_id,
            curso_id=secuencia.curso_id,
            probabilidad_dominio=round(prob, 4),
            confianza=0.7 if n >= 5 else 0.4,
            pesos_atencion=pesos,
        )

    def _real_prediccion(self, secuencia: SecuenciaInteraccion) -> PrediccionKT:
        try:
            import torch

            # Mapear conceptos→índices enteros que entiende el modelo:
            #  - con concept_index (modelo Moodle): traduce la sección; omite desconocidos.
            #  - sin índice (legacy assist2015): los conceptos ya son string-ints.
            pares: list[tuple[int, int]] = []
            for concepto, correcta in zip(
                secuencia.concepto_ids, secuencia.respuestas_correctas
            ):
                if self._concept_index:
                    idx = self._concept_index.get(str(concepto))
                    if idx is None:
                        continue
                elif str(concepto).lstrip("-").isdigit():
                    idx = int(concepto)
                else:
                    continue
                pares.append((idx, 1 if correcta else 0))

            if len(pares) < 2:
                return self._mock_prediccion(secuencia)
            concepts = [p[0] for p in pares]
            responses = [p[1] for p in pares]

            seq_len = self._seq_len
            concepts = concepts[-seq_len:]
            responses = responses[-seq_len:]
            L = len(concepts)

            # SAKT input format (igual que train.py):
            #   q   = past concepts [0..L-2]
            #   r   = past responses [0..L-2]
            #   qry = shifted concepts [1..L-1]  ← output[-1] = P(correct para concepts[-1])
            q = concepts[:-1]
            r = responses[:-1]
            qry = concepts[1:]

            pad = seq_len - (L - 1)
            q_t = torch.LongTensor([[0] * pad + q])
            r_t = torch.LongTensor([[0] * pad + r])
            qry_t = torch.LongTensor([[0] * pad + qry])

            with torch.no_grad():
                # pyKT SAKT.forward(q, r, qry) → ya aplica sigmoid internamente
                out = self._model(q_t, r_t, qry_t)
                prob = float(out[0, -1].item())

            # Pesos de atención REALES: cuánto atendió el último paso a cada
            # interacción pasada (último bloque). Fallback a uniforme si falla.
            pesos = [1.0 / (L - 1)] * (L - 1)
            try:
                attn = self._model.blocks[-1]._last_attn  # (1, seq_len, seq_len)
                fila = attn[0, -1, -(L - 1) :].tolist()
                total = sum(fila)
                if total > 0:
                    pesos = [round(w / total, 4) for w in fila]
            except Exception as e:
                logger.warning("No se pudo extraer atención real: %s", e)
            return PrediccionKT(
                estudiante_id=secuencia.estudiante_id,
                curso_id=secuencia.curso_id,
                probabilidad_dominio=round(min(max(prob, 0.0), 1.0), 4),
                confianza=0.85,
                pesos_atencion=pesos,
            )
        except Exception as e:
            logger.error("Inferencia real falló, usando mock: %s", e)
            return self._mock_prediccion(secuencia)
