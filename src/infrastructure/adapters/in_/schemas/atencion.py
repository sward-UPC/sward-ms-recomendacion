"""Contratos HTTP del heatmap de atención del SAKT."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PuntoAtencionResponse(BaseModel):
    """Interacción pasada con el peso de atención que SAKT le asignó."""

    model_config = ConfigDict(extra="forbid")

    concepto: str = Field(..., description="Concepto/sección de la interacción")
    acierto: bool = Field(..., description="Si la respuesta fue correcta")
    peso: float = Field(..., ge=0.0, le=1.0, description="Peso de atención [0,1]")
    suficiente: bool = Field(
        False,
        description=(
            "True si la verificación comprobó que esta interacción basta para "
            "llegar a la predicción. Solo puede ser True con suficiencia verificada."
        ),
    )


class ContrafactualResponse(BaseModel):
    """Qué pasaría si la interacción más atendida hubiera salido al revés."""

    model_config = ConfigDict(extra="forbid")

    concepto: str
    acierto_original: bool
    probabilidad_original: float = Field(..., ge=0.0, le=1.0)
    probabilidad_contrafactual: float = Field(..., ge=0.0, le=1.0)


class FidelidadResponse(BaseModel):
    """Veredicto de la verificación de fidelidad para esta predicción.

    Le dice a quien presenta la explicación qué puede afirmar y qué no. La
    atención de SAKT, medida con ERASER, resultó suficiente pero no necesaria:
    con `verificada` en True se puede decir que las interacciones indicadas
    «bastan para llegar a esta conclusión», nunca que la causan. El contrafactual
    solo viene si además se verificó necesidad.
    """

    model_config = ConfigDict(extra="forbid")

    criterio: str = Field(..., description='"suficiencia" o "exhaustividad"')
    verificada: bool = Field(
        ..., description="Si la atención superó al azar según el criterio configurado"
    )
    motivo: str = Field(
        ...,
        description=(
            "fiel | no_supera_azar | pocas_interacciones | "
            "verificacion_desactivada | error_verificacion"
        ),
    )
    confianza: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fracción de comparaciones al azar que la atención superó",
    )
    umbral: float = Field(..., ge=0.0, le=1.0)
    n_comparaciones: int = Field(
        ..., ge=0, description="Comparaciones al azar realizadas en la prueba decisiva"
    )
    conceptos_suficientes: list[str] = Field(
        default_factory=list,
        description="Conceptos de las interacciones que bastan para la predicción",
    )
    es_necesaria: bool = Field(
        False, description="Si también se verificó que quitarlas cambia la predicción"
    )
    contrafactual: ContrafactualResponse | None = None


class AtencionResponse(BaseModel):
    """Heatmap de atención del SAKT para un estudiante."""

    model_config = ConfigDict(extra="forbid")

    probabilidad_dominio: float = Field(..., ge=0.0, le=1.0)
    puntos: list[PuntoAtencionResponse]
    fidelidad: FidelidadResponse | None = None
    fuente: Literal["modelo", "promedio"] = Field(
        "modelo",
        description=(
            "De dónde sale la cifra: «modelo» si la produjo el SAKT entrenado; "
            "«promedio» si el modelo no pudo (conceptos que no conoce, historia "
            "muy corta) y es el promedio de aciertos con atención uniforme, que "
            "no debe presentarse como una estimación del modelo."
        ),
    )
