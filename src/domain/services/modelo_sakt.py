import logging
import os

from src.domain.entities.prediccion_kt import PrediccionKT
from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion

logger = logging.getLogger(__name__)


class ModeloSAKT:
    """SAKT wrapper. Mock en development, real en production cargando desde S3."""

    def __init__(self, version: str = "v1.0", mock: bool = True):
        self.version = version
        self._mock = mock
        self._model = None
        if not mock:
            self._cargar_modelo()

    def _cargar_modelo(self) -> None:
        try:
            import boto3
            import torch

            from src.infrastructure.config.settings import settings

            s3 = boto3.client("s3", region_name=settings.aws_region)
            local_path = f"/tmp/sakt_{self.version}.pth"
            if not os.path.exists(local_path):
                logger.info(
                    "Descargando modelo desde S3 | key=%s", settings.sakt_model_s3_key
                )
                s3.download_file(
                    settings.aws_s3_model_bucket, settings.sakt_model_s3_key, local_path
                )
            self._model = torch.load(local_path, map_location="cpu")
            logger.info("Modelo SAKT cargado | version=%s", self.version)
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

            x, y = secuencia.to_vectors()
            with torch.no_grad():
                t = torch.tensor([list(zip(x, y))], dtype=torch.float32)
                out = self._model(t)
                prob = float(torch.sigmoid(out[:, -1]).item())
            return PrediccionKT(
                estudiante_id=secuencia.estudiante_id,
                curso_id=secuencia.curso_id,
                probabilidad_dominio=round(prob, 4),
                confianza=0.85,
            )
        except Exception as e:
            logger.error("Inferencia real falló, usando mock: %s", e)
            return self._mock_prediccion(secuencia)
