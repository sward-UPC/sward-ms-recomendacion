"""Contratos HTTP de generación de material de estudio (LLM + video real)."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class GenerarMaterialRequest(BaseModel):
    """Solicitud para generar material de estudio sobre el concepto débil."""

    model_config = ConfigDict(extra="forbid")

    estudianteId: UUID = Field(
        ...,
        description="UUID del estudiante",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )
    cursoId: UUID = Field(
        ...,
        description="UUID del curso",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440001"},
    )
    refrescar: bool = Field(
        default=False,
        description="Si True, ignora el cache y genera material fresco (Generar más)",
    )
    formatoPreferido: str | None = Field(
        default=None,
        description="Formato en que el alumno rinde mejor; el LLM lo enfatiza",
    )
    evitarConcepto: str | None = Field(
        default=None,
        description="Concepto ya mostrado a evitar (Generar más rota al siguiente débil)",
    )


class MaterialResponse(BaseModel):
    """Set de recursos educativos tipados (o fallback vacío si el LLM no está activo).

    ``recursos`` es una lista de objetos tipados por su campo ``tipo``:
    - ``quiz``: ``{tipo, titulo, preguntas: [{enunciado, opciones[4], correcta, explicacion}]}``
    - ``lectura``: ``{tipo, titulo, contenido}``
    - ``practica``: ``{tipo, titulo, ejercicios: [{enunciado, solucion}]}``
    - ``video``: ``{tipo, titulo, video_id, url, query}``
    """

    model_config = ConfigDict(extra="forbid")

    disponible: bool = Field(
        ...,
        description="True si el LLM generó material; False en el fallback",
    )
    concepto: str | None = Field(
        ...,
        description="Concepto más débil reforzado (None si no hay interacciones)",
    )
    recursos: list[dict] = Field(
        default_factory=list,
        description="Recursos educativos tipados (quiz, lectura, practica, video)",
    )
    dominio: int | None = Field(
        default=None,
        description="Dominio estimado por el SAKT en el concepto (0-100), para el XAI",
    )
