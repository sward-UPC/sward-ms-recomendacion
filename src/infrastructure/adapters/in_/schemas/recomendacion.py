"""Contratos HTTP de generación, listado y completado de recomendaciones."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ItemRecomendadoResponse(BaseModel):
    """Representa un recurso recomendado en la respuesta de generación de recomendaciones."""

    model_config = ConfigDict(extra="forbid")

    recurso_id: str = Field(
        ...,
        description="UUID del recurso recomendado",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )
    titulo: str = Field(
        ...,
        max_length=255,
        description="Título del recurso",
        json_schema_extra={"example": "Introducción a Algoritmos"},
    )
    tipo: str = Field(
        ...,
        description="Tipo de recurso educativo",
        json_schema_extra={
            "pattern": "^(video|lectura|ejercicio|quiz|presentacion)$",
            "example": "video",
        },
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Puntuación de relevancia [0, 1]",
        json_schema_extra={"example": 0.92},
    )
    orden: int = Field(
        ...,
        ge=1,
        description="Posición en el ranking de recomendación",
        json_schema_extra={"example": 1},
    )
    motivo: str = Field(
        "",
        max_length=512,
        description="Explicación breve de por qué se recomienda este recurso",
        json_schema_extra={
            "example": "Alinea con tu dominio actual (0.87) y cubre conceptos faltantes"
        },
    )
    url: str = Field(
        "",
        description="URL del recurso recomendado",
        json_schema_extra={"example": "https://lms.example.com/mod/page/view.php?id=1"},
    )


class GenerarRecomendacionRequest(BaseModel):
    """Solicitud para generar recomendaciones personalizadas."""

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


class GenerarRecomendacionResponse(BaseModel):
    """Respuesta con recomendaciones generadas."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="UUID de la recomendación",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440002"},
    )
    estado: str = Field(
        ...,
        description="Estado actual de la recomendación",
        json_schema_extra={"enum": ["pendiente", "completado"], "example": "pendiente"},
    )
    items: list[ItemRecomendadoResponse] = Field(
        ...,
        description="Lista de recursos recomendados ordenados por relevancia",
    )


class ListarRecomendacionesResponse(BaseModel):
    """Resumen de una recomendación en la listado."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="UUID de la recomendación",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440002"},
    )
    estado: str = Field(
        ...,
        description="Estado actual de la recomendación",
        json_schema_extra={"example": "pendiente"},
    )
    generada_en: str = Field(
        ...,
        description="Timestamp ISO 8601 de generación",
        json_schema_extra={"example": "2025-05-31T14:30:00Z"},
    )
    items_count: int = Field(
        ...,
        ge=0,
        description="Cantidad de recursos recomendados",
        json_schema_extra={"example": 5},
    )


class CompletarRecomendacionResponse(BaseModel):
    """Respuesta al completar una recomendación."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="UUID de la recomendación",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440002"},
    )
    estado: str = Field(
        ...,
        description="Nuevo estado de la recomendación",
        json_schema_extra={"example": "completado"},
    )
