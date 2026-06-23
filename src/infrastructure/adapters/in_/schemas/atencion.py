"""Contratos HTTP del heatmap de atención del SAKT."""

from pydantic import BaseModel, ConfigDict, Field


class PuntoAtencionResponse(BaseModel):
    """Interacción pasada con el peso de atención que SAKT le asignó."""

    model_config = ConfigDict(extra="forbid")

    concepto: str = Field(..., description="Concepto/sección de la interacción")
    acierto: bool = Field(..., description="Si la respuesta fue correcta")
    peso: float = Field(..., ge=0.0, le=1.0, description="Peso de atención [0,1]")


class AtencionResponse(BaseModel):
    """Heatmap de atención del SAKT para un estudiante."""

    model_config = ConfigDict(extra="forbid")

    probabilidad_dominio: float = Field(..., ge=0.0, le=1.0)
    puntos: list[PuntoAtencionResponse]
