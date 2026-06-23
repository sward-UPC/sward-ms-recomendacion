from dataclasses import dataclass
from uuid import UUID

from src.domain.entities.recomendacion import Recomendacion
from src.application.ports.out_.recomendacion_repository_port import (
    RecomendacionRepositoryPort,
)


@dataclass
class ConsultarRecomendacionCommand:
    estudiante_id: UUID


class ConsultarRecomendacionUseCase:
    def __init__(self, repo: RecomendacionRepositoryPort):
        self._repo = repo

    async def execute(self, cmd: ConsultarRecomendacionCommand) -> list[Recomendacion]:
        return await self._repo.find_by_estudiante(cmd.estudiante_id)
