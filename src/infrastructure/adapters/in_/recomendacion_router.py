from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

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


class GenerarRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    estudianteId: UUID
    cursoId: UUID


@router.post("/generate", status_code=201)
async def generar(
    body: GenerarRequest,
    uc: GenerarRecomendacionUseCase = Depends(get_generar_uc),
    _payload: dict = Depends(require_jwt),
):
    rec = await uc.execute(
        GenerarRecomendacionCommand(
            estudiante_id=body.estudianteId, curso_id=body.cursoId
        )
    )
    return {
        "id": str(rec.id),
        "estado": rec.estado,
        "items": [
            {
                "recurso_id": i.recurso_id,
                "titulo": i.titulo,
                "tipo": i.tipo,
                "score": i.score,
                "orden": i.orden,
                "motivo": i.motivo,
            }
            for i in rec.items
        ],
    }


@router.get("")
async def listar(
    estudianteId: UUID,
    uc: ConsultarRecomendacionUseCase = Depends(get_consultar_uc),
    _payload: dict = Depends(require_jwt),
):
    recs = await uc.execute(ConsultarRecomendacionCommand(estudiante_id=estudianteId))
    return [
        {
            "id": str(r.id),
            "estado": r.estado,
            "generada_en": r.generada_en.isoformat(),
            "items_count": len(r.items),
        }
        for r in recs
    ]


@router.patch("/{rec_id}/complete")
async def completar(
    rec_id: UUID,
    uc: CompletarRecomendacionUseCase = Depends(get_completar_uc),
    _payload: dict = Depends(require_jwt),
):
    await uc.execute(CompletarRecomendacionCommand(recomendacion_id=rec_id))
    return {"id": str(rec_id), "estado": "completado"}
