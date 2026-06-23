from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from src.application.use_cases.completar_recomendacion import (
    CompletarRecomendacionCommand,
    CompletarRecomendacionUseCase,
)
from src.application.use_cases.consultar_atencion import ConsultarAtencionUseCase
from src.application.use_cases.consultar_modelo_info import ConsultarModeloInfoUseCase
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
from src.infrastructure.adapters.in_.schemas import (
    AtencionResponse,
    CompletarRecomendacionResponse,
    GenerarMaterialRequest,
    GenerarRecomendacionRequest,
    GenerarRecomendacionResponse,
    ItemRecomendadoResponse,
    ListarRecomendacionesResponse,
    MaterialResponse,
    ModelInfoResponse,
    PuntoAtencionResponse,
    VerificacionResponse,
    VerificarEjercicioRequest,
)
from src.infrastructure.dependencies import (
    get_atencion_uc,
    get_completar_uc,
    get_consultar_uc,
    get_generar_uc,
    get_material_uc,
    get_modelo_info_uc,
    get_verificar_ejercicio_uc,
    require_jwt,
    require_service_key,
)

router = APIRouter(prefix="/recommendations", tags=["Recomendación"])


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
    rec = await uc.execute(
        GenerarRecomendacionCommand(estudiante_id=estudiante_id, curso_id=body.cursoId)
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
    recs = await uc.execute(ConsultarRecomendacionCommand(estudiante_id=estudianteId))
    return [
        ListarRecomendacionesResponse(
            id=str(r.id),
            estado=str(r.estado),
            generada_en=r.generada_en.isoformat(),
            items_count=len(r.items),
        )
        for r in recs
    ]


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
    await uc.execute(CompletarRecomendacionCommand(recomendacion_id=rec_id))
    return CompletarRecomendacionResponse(
        id=str(rec_id),
        estado="completado",
    )


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
    response_model=ModelInfoResponse,
    responses={
        200: {"description": "Metadata del modelo entrenado"},
        401: {"description": "Service-key inválida"},
        503: {"description": "No se pudo leer el modelo desde S3"},
    },
    include_in_schema=False,
)
async def model_info(
    _auth: None = Depends(require_service_key),
    uc: ConsultarModeloInfoUseCase = Depends(get_modelo_info_uc),
):
    """Metadata REAL del modelo (fecha de entreno, hiperparámetros, AUC).

    Consumido s2s por ms-usuarios para el panel admin. La lee del artefacto en S3,
    así que refleja siempre el modelo desplegado (no valores hardcodeados).
    `mock=true` indica que ESTE entorno corre el modelo simulado (desarrollo),
    aunque la metadata describe igual el artefacto entrenado en S3.
    """
    return uc.execute()
