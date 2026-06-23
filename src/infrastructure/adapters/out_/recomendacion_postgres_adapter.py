import json
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.entities.recomendacion import ItemRecomendado, Recomendacion
from src.application.ports.out_.recomendacion_repository_port import (
    RecomendacionRepositoryPort,
)
from src.domain.value_objects.estado_recomendacion import EstadoRecomendacion
from src.infrastructure.db.models.recomendacion_models import RecomendacionModel


class RecomendacionPostgresAdapter(RecomendacionRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._s = session

    async def save(self, r: Recomendacion) -> Recomendacion:
        items_json = json.dumps(
            [
                {
                    "recurso_id": i.recurso_id,
                    "titulo": i.titulo,
                    "tipo": i.tipo,
                    "score": i.score,
                    "orden": i.orden,
                    "motivo": i.motivo,
                    "url": i.url,
                }
                for i in r.items
            ]
        )
        m = RecomendacionModel(
            id=r.id,
            estudiante_id=r.estudiante_id,
            curso_id=r.curso_id,
            estado=r.estado.value,
            items_json=items_json,
            generada_en=r.generada_en,
        )
        self._s.add(m)
        await self._s.flush()
        return r

    async def find_by_id(self, id: UUID) -> Recomendacion | None:
        res = await self._s.execute(
            select(RecomendacionModel).where(RecomendacionModel.id == id)
        )
        m = res.scalar_one_or_none()
        return _to_entity(m) if m else None

    async def find_by_estudiante(
        self, estudiante_id: UUID, limit: int = 20
    ) -> list[Recomendacion]:
        res = await self._s.execute(
            select(RecomendacionModel)
            .where(RecomendacionModel.estudiante_id == estudiante_id)
            .order_by(RecomendacionModel.generada_en.desc())
            .limit(limit)
        )
        return [_to_entity(m) for m in res.scalars().all()]

    async def update_estado(self, id: UUID, estado: str) -> None:
        await self._s.execute(
            update(RecomendacionModel)
            .where(RecomendacionModel.id == id)
            .values(estado=estado)
        )


def _to_entity(m: RecomendacionModel) -> Recomendacion:
    items = [ItemRecomendado(**i) for i in json.loads(m.items_json or "[]")]
    return Recomendacion(
        id=m.id,
        estudiante_id=m.estudiante_id,
        curso_id=m.curso_id,
        estado=EstadoRecomendacion(m.estado),
        items=items,
        generada_en=m.generada_en,
    )
