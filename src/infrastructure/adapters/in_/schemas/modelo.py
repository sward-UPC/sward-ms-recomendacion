"""Contrato HTTP de la metadata real del modelo SAKT (endpoint interno s2s)."""

from pydantic import BaseModel, ConfigDict, Field


class ModelInfoResponse(BaseModel):
    """Metadata REAL del artefacto SAKT entrenado en S3, para el panel admin.

    Los hiperparámetros y métricas salen del checkpoint (``.get(...)``), por lo que
    pueden ser ``None`` si el artefacto no los trae. ``mock`` indica si ESTE entorno
    corre el modelo simulado, aunque la metadata describe igual el modelo en S3.
    """

    model_config = ConfigDict(extra="forbid")

    mock: bool = Field(..., description="True si este entorno corre el modelo simulado")
    n_skills: int | None = Field(
        default=None, description="Número de skills del modelo"
    )
    n_conceptos: int | None = Field(
        default=None, description="Número de conceptos (secciones de Moodle)"
    )
    seq_len: int | None = Field(default=None, description="Longitud de secuencia")
    emb_size: int | None = Field(default=None, description="Tamaño de embedding")
    n_heads: int | None = Field(default=None, description="Cabezas de atención")
    n_layers: int | None = Field(default=None, description="Número de capas")
    dropout: float | None = Field(default=None, description="Tasa de dropout")
    learning_rate: float | None = Field(default=None, description="Learning rate")
    epochs: int | None = Field(default=None, description="Épocas de entrenamiento")
    test_auc: float | None = Field(default=None, description="AUC en el set de test")
    n_estudiantes: int | None = Field(
        default=None, description="Estudiantes en el dataset de entrenamiento"
    )
    n_muestras: int | None = Field(
        default=None, description="Muestras en el dataset de entrenamiento"
    )
    entrenado_en: str | None = Field(
        default=None, description="Timestamp ISO 8601 del último entrenamiento"
    )
    actualizado_en: str | None = Field(
        default=None, description="Timestamp ISO 8601 del artefacto en S3"
    )
    tam_bytes: int | None = Field(
        default=None, description="Tamaño del artefacto en bytes"
    )
