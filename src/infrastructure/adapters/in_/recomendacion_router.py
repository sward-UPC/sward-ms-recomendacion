from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from src.application.use_cases.completar_recomendacion import (
    CompletarRecomendacionCommand,
    CompletarRecomendacionUseCase,
)
from src.application.use_cases.consultar_atencion import ConsultarAtencionUseCase
from src.application.use_cases.consultar_recomendacion import (
    ConsultarRecomendacionCommand,
    ConsultarRecomendacionUseCase,
)
from src.application.use_cases.generar_material import (
    GenerarMaterialCommand,
    GenerarMaterialUseCase,
)
from src.application.use_cases.generar_recomendacion import (
    GenerarRecomendacionCommand,
    GenerarRecomendacionUseCase,
)
from src.application.use_cases.verificar_ejercicio import (
    VerificarEjercicioCommand,
    VerificarEjercicioUseCase,
)
from src.domain.services.modelo_sakt import leer_info_modelo
from src.infrastructure.config.settings import settings
from src.infrastructure.dependencies import (
    get_atencion_uc,
    get_completar_uc,
    get_consultar_uc,
    get_generar_uc,
    get_material_uc,
    get_verificar_ejercicio_uc,
    require_jwt,
    require_service_key,
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
        UUID(payload["sub"])
        if payload.get("rol") == "estudiante"
        else body.estudianteId
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
                    url=i.url,
                )
                for i in rec.items
            ],
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


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


@router.post(
    "/material",
    response_model=MaterialResponse,
    responses={
        200: {"description": "Material generado, o fallback si el LLM no está activo"},
        401: {"description": "No autorizado - requiere autenticación JWT"},
    },
)
async def generar_material(
    body: GenerarMaterialRequest,
    uc: GenerarMaterialUseCase = Depends(get_material_uc),
    payload: dict = Depends(require_jwt),
):
    """Genera un set de recursos educativos tipados (LLM + video real de YouTube).

    El LLM produce un quiz, una mini-lección y una práctica para reforzar el
    concepto débil; YouTube aporta un video real. Un estudiante usa su propio JWT;
    docente/admin pueden indicar otro en el body. Todo es best-effort: si no hay
    clave o la llamada falla, devuelve ``disponible: false`` con ``recursos: []``
    (status 200, nunca rompe).
    """
    estudiante_id = (
        UUID(payload["sub"])
        if payload.get("rol") == "estudiante"
        else body.estudianteId
    )
    material = await uc.execute(
        GenerarMaterialCommand(
            estudiante_id=estudiante_id,
            curso_id=body.cursoId,
            refrescar=body.refrescar,
            formato_preferido=body.formatoPreferido,
            evitar_concepto=body.evitarConcepto,
        )
    )
    return MaterialResponse(
        disponible=material.disponible,
        concepto=material.concepto,
        recursos=material.recursos,
        dominio=material.dominio,
    )


class VerificarEjercicioRequest(BaseModel):
    enunciado: str = Field(..., description="Enunciado del ejercicio")
    solucion: str = Field(..., description="Solución de referencia del ejercicio")
    respuesta: str = Field(..., description="Respuesta escrita por el estudiante")


class VerificacionResponse(BaseModel):
    aprobado: bool = Field(
        ..., description="True si la IA da por correcta la respuesta"
    )
    feedback: str = Field(
        ..., description="Comentario breve y alentador para el alumno"
    )


@router.post(
    "/verify-exercise",
    response_model=VerificacionResponse,
    responses={
        200: {"description": "Evaluación de la respuesta (aprobado + feedback)"},
        401: {"description": "No autorizado - requiere autenticación JWT"},
    },
)
async def verificar_ejercicio(
    body: VerificarEjercicioRequest,
    uc: VerificarEjercicioUseCase = Depends(get_verificar_ejercicio_uc),
    payload: dict = Depends(require_jwt),
):
    """Evalúa con un LLM si la respuesta del estudiante a un ejercicio es correcta.

    Permite que el alumno resuelva la práctica dentro de la plataforma: escribe su
    respuesta y la IA la aprueba (o da una pista). Best-effort: si el LLM falla,
    devuelve ``aprobado: false`` con un mensaje amable (status 200, nunca rompe).
    """
    r = await uc.execute(
        VerificarEjercicioCommand(
            enunciado=body.enunciado,
            solucion=body.solucion,
            respuesta=body.respuesta,
        )
    )
    return VerificacionResponse(aprobado=r.aprobado, feedback=r.feedback)


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


@router.get(
    "/attention",
    response_model=AtencionResponse,
    responses={
        200: {"description": "Heatmap de atención obtenido"},
        401: {"description": "No autorizado - requiere autenticación JWT"},
    },
)
async def atencion(
    estudianteId: UUID = Query(..., description="UUID del estudiante"),
    cursoId: UUID = Query(..., description="UUID del curso"),
    uc: ConsultarAtencionUseCase = Depends(get_atencion_uc),
    payload: dict = Depends(require_jwt),
):
    """Pesos de atención reales del SAKT sobre las interacciones del estudiante.

    Un estudiante solo ve los suyos (UUID del JWT); docente/admin pueden indicar otro.
    """
    if payload.get("rol") == "estudiante":
        estudianteId = UUID(payload["sub"])
    res = await uc.execute(estudianteId, cursoId)
    return AtencionResponse(
        probabilidad_dominio=res.probabilidad_dominio,
        puntos=[
            PuntoAtencionResponse(concepto=p.concepto, acierto=p.acierto, peso=p.peso)
            for p in res.puntos
        ],
    )


@router.get(
    "/internal/model-info",
    summary="Metadata real del modelo SAKT (s2s)",
    responses={
        200: {"description": "Metadata del modelo entrenado"},
        401: {"description": "Service-key inválida"},
        503: {"description": "No se pudo leer el modelo desde S3"},
    },
    include_in_schema=False,
)
async def model_info(_auth: None = Depends(require_service_key)):
    """Metadata REAL del modelo (fecha de entreno, hiperparámetros, AUC).

    Consumido s2s por ms-usuarios para el panel admin. La lee del artefacto en S3,
    así que refleja siempre el modelo desplegado (no valores hardcodeados).
    `mock=true` indica que ESTE entorno corre el modelo simulado (desarrollo),
    aunque la metadata describe igual el artefacto entrenado en S3.
    """
    es_mock = settings.environment == "development"
    try:
        info = leer_info_modelo()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No se pudo leer el modelo desde S3: {exc}",
        ) from exc
    return {"mock": es_mock, **info}
