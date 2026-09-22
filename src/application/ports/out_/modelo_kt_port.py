from abc import ABC, abstractmethod

from src.domain.entities.prediccion_kt import PrediccionKT
from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion


class ModeloKTPort(ABC):
    """Puerto de salida del núcleo hacia el modelo de Knowledge Tracing (SAKT).

    El dominio/aplicación solo conoce esta interfaz pura: estima el dominio de un
    concepto a partir de la secuencia de interacciones del estudiante y, opcionalmente,
    expone la metadata del artefacto entrenado. Toda la dependencia de ML/infra
    (torch/pyKT/boto3/S3) vive en los adaptadores que la implementan.
    """

    @abstractmethod
    def predecir_dominio(self, secuencia: SecuenciaInteraccion) -> PrediccionKT:
        """Estima la probabilidad de dominio y los pesos de atención del último paso."""
        ...

    def predecir_dominio_por_concepto(
        self, secuencia: SecuenciaInteraccion, conceptos: list[str]
    ) -> dict[str, float]:
        """Probabilidad de acertar el próximo intento de cada concepto, dada toda la
        historia del estudiante.

        Devuelve solo los conceptos que el modelo pudo estimar. Vacío cuando no
        puede (modelo simulado, conceptos que no conoce, historia insuficiente):
        en ese caso quien llama no debe atribuirle ninguna cifra al modelo.
        """
        return {}

    @abstractmethod
    def leer_info(self) -> dict:
        """Metadata real del modelo entrenado (hiperparámetros, métricas, fechas)."""
        ...
