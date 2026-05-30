from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.application.use_cases.consultar_recomendacion import (
    ConsultarRecomendacionCommand,
    ConsultarRecomendacionUseCase,
)
from src.application.use_cases.generar_recomendacion import (
    GenerarRecomendacionCommand,
    GenerarRecomendacionUseCase,
)
from src.domain.services.modelo_sakt import ModeloSAKT
from src.infrastructure.adapters.out_.cursos_rest_adapter import CursosRestAdapter
from src.infrastructure.adapters.out_.eventbridge_adapter import EventBridgeAdapter
from src.infrastructure.adapters.out_.recomendacion_postgres_adapter import (
    RecomendacionPostgresAdapter,
)
from src.infrastructure.adapters.out_.trazabilidad_rest_adapter import (
    TrazabilidadRestAdapter,
)
from src.infrastructure.adapters.out_.xai_rest_adapter import XaiRestAdapter
from src.infrastructure.config.settings import settings
from src.infrastructure.db.database import get_session

router = APIRouter(prefix="/recommendations", tags=["Recomendación"])


def _modelo() -> ModeloSAKT:
    return ModeloSAKT(mock=(settings.environment == "development"))


class GenerarRequest(BaseModel):
    estudianteId: UUID
    cursoId: UUID


@router.post("/generate", status_code=201)
async def generar(body: GenerarRequest, session: AsyncSession = Depends(get_session)):
    uc = GenerarRecomendacionUseCase(
        trazabilidad=TrazabilidadRestAdapter(),
        cursos=CursosRestAdapter(),
        xai=XaiRestAdapter(),
        repo=RecomendacionPostgresAdapter(session),
        event_publisher=EventBridgeAdapter(),
        modelo=_modelo(),
    )
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
async def listar(estudianteId: UUID, session: AsyncSession = Depends(get_session)):
    uc = ConsultarRecomendacionUseCase(RecomendacionPostgresAdapter(session))
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
async def completar(rec_id: UUID, session: AsyncSession = Depends(get_session)):
    await RecomendacionPostgresAdapter(session).update_estado(rec_id, "completado")
    return {"id": str(rec_id), "estado": "completado"}
