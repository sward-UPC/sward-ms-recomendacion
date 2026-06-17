from dataclasses import dataclass
from uuid import UUID

from src.domain.ports.out_.recomendacion_repository_port import (
    RecomendacionRepositoryPort,
)
from src.domain.value_objects.estado_recomendacion import EstadoRecomendacion


@dataclass
class CompletarRecomendacionCommand:
    recomendacion_id: UUID


class CompletarRecomendacionUseCase:
    """Marca una recomendación como completada por el estudiante."""

    def __init__(self, repo: RecomendacionRepositoryPort):
        self._repo = repo

    async def execute(self, cmd: CompletarRecomendacionCommand) -> None:
        existing = await self._repo.find_by_id(cmd.recomendacion_id)
        if existing is None:
            raise KeyError(f"Recomendación {cmd.recomendacion_id} no encontrada")
        await self._repo.update_estado(
            cmd.recomendacion_id, EstadoRecomendacion.COMPLETADO.value
        )
