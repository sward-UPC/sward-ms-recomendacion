from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from src.application.use_cases.completar_recomendacion import (
    CompletarRecomendacionCommand,
    CompletarRecomendacionUseCase,
)
from src.application.use_cases.consultar_recomendacion import (
    ConsultarRecomendacionCommand,
    ConsultarRecomendacionUseCase,
)
from src.application.use_cases.generar_recomendacion import (
    GenerarRecomendacionCommand,
    GenerarRecomendacionUseCase,
)
from src.infrastructure.dependencies import (
    get_completar_uc,
    get_consultar_uc,
    get_generar_uc,
    require_jwt,
)

router = APIRouter(prefix="/recommendations", tags=["Recomendación"])


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
        ...,
        max_length=512,
        description="Explicación breve de por qué se recomienda este recurso",
        json_schema_extra={
            "example": "Alinea con tu dominio actual (0.87) y cubre conceptos faltantes"
        },
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


@router.post(
    "/generate",
    status_code=status.HTTP_201_CREATED,
    response_model=GenerarRecomendacionResponse,
    responses={
        201: {"description": "Recomendaciones generadas exitosamente"},
        400: {"description": "Datos de entrada inválidos"},
        401: {"description": "No autorizado - requiere autenticación JWT"},
        404: {"description": "Estudiante o curso no encontrado"},
        500: {"description": "Error interno del servidor"},
    },
)
async def generar(
    body: GenerarRecomendacionRequest,
    uc: GenerarRecomendacionUseCase = Depends(get_generar_uc),
    payload: dict = Depends(require_jwt),
):
    """
    Genera recomendaciones personalizadas para un estudiante en un curso.

    **Flujo:**
    1. Valida que el estudiante y curso existan
    2. Ejecuta el modelo SAKT para calcular déficit de conocimiento
    3. Clasifica recursos educativos por relevancia
    4. Retorna top N recursos ordenados por score

    **SLA:** ≤2 segundos

    **Autenticación:** Bearer JWT (requerido)
    """
    # Un estudiante solo genera sus propias recomendaciones (toma su UUID del JWT);
    # docente/admin pueden generar para un estudiante indicado en el body.
    estudiante_id = (
        UUID(payload["sub"]) if payload.get("rol") == "estudiante" else body.estudianteId
    )
    try:
        rec = await uc.execute(
            GenerarRecomendacionCommand(
                estudiante_id=estudiante_id, curso_id=body.cursoId
            )
        )
        return GenerarRecomendacionResponse(
            id=str(rec.id),
            estado=str(rec.estado),
            items=[
                ItemRecomendadoResponse(
                    recurso_id=str(i.recurso_id),
                    titulo=i.titulo,
                    tipo=i.tipo,
                    score=i.score,
                    orden=i.orden,
                    motivo=i.motivo,
                )
                for i in rec.items
            ],
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "",
    response_model=list[ListarRecomendacionesResponse],
    responses={
        200: {"description": "Listado de recomendaciones obtenido exitosamente"},
        401: {"description": "No autorizado - requiere autenticación JWT"},
        404: {"description": "Estudiante no encontrado"},
        500: {"description": "Error interno del servidor"},
    },
)
async def listar(
    estudianteId: UUID = Query(
        ...,
        description="UUID del estudiante",
    ),
    uc: ConsultarRecomendacionUseCase = Depends(get_consultar_uc),
    payload: dict = Depends(require_jwt),
):
    """
    Lista todas las recomendaciones generadas para un estudiante.

    **Flujo:**
    1. Recupera el historial de recomendaciones del estudiante
    2. Retorna resúmenes ordenados por fecha (más reciente primero)

    **SLA:** ≤1 segundo

    **Autenticación:** Bearer JWT (requerido)
    """
    # El estudiante solo ve sus propias recomendaciones (UUID del JWT).
    if payload.get("rol") == "estudiante":
        estudianteId = UUID(payload["sub"])
    try:
        recs = await uc.execute(
            ConsultarRecomendacionCommand(estudiante_id=estudianteId)
        )
        return [
            ListarRecomendacionesResponse(
                id=str(r.id),
                estado=str(r.estado),
                generada_en=r.generada_en.isoformat(),
                items_count=len(r.items),
            )
            for r in recs
        ]
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.patch(
    "/{rec_id}/complete",
    response_model=CompletarRecomendacionResponse,
    responses={
        200: {"description": "Recomendación completada exitosamente"},
        401: {"description": "No autorizado - requiere autenticación JWT"},
        404: {"description": "Recomendación no encontrada"},
        500: {"description": "Error interno del servidor"},
    },
)
async def completar(
    rec_id: UUID,
    uc: CompletarRecomendacionUseCase = Depends(get_completar_uc),
    _payload: dict = Depends(require_jwt),
):
    """
    Marca una recomendación como completada por el estudiante.

    **Flujo:**
    1. Verifica que la recomendación existe y está en estado PENDIENTE
    2. Actualiza el estado a COMPLETADO
    3. Registra timestamp de completación

    **SLA:** ≤500ms

    **Autenticación:** Bearer JWT (requerido)
    """
    try:
        await uc.execute(CompletarRecomendacionCommand(recomendacion_id=rec_id))
        return CompletarRecomendacionResponse(
            id=str(rec_id),
            estado="completado",
        )
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
