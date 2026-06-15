import logging
import os
import sys
import tempfile
import types as _types

from src.domain.entities.prediccion_kt import PrediccionKT
from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion

logger = logging.getLogger(__name__)


def _mock_turtle() -> None:
    """pyKT bug: qdkt.py importa turtle (requiere Tk). Mock antes de que pyKT cargue."""
    if "turtle" not in sys.modules:
        _mock = _types.ModuleType("turtle")
        _mock.forward = lambda *a, **kw: None
        sys.modules["turtle"] = _mock


class ModeloSAKT:
    """SAKT wrapper. Mock en development, real en production cargando desde S3."""

    def __init__(self, version: str = "v1.0", mock: bool = True):
        self.version = version
        self._mock = mock
        self._model = None
        self._seq_len = 200
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

            _mock_turtle()

            import pykt.models.utils as _pykt_utils
            from pykt.models.sakt import SAKT

            _pykt_utils.device = "cpu"

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
        except Exception as e:
            logger.error("Error cargando SAKT, usando mock: %s", e)
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

            # concepto_ids son string-ints de ASSISTments skill IDs (ej. "42", "17")
            concepts = [int(c) for c in secuencia.concepto_ids]
            responses = [1 if r else 0 for r in secuencia.respuestas_correctas]

            if len(concepts) < 2:
                return self._mock_prediccion(secuencia)

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

            pesos = [1.0 / (L - 1)] * (L - 1)
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
