from abc import ABC, abstractmethod
from uuid import UUID

from src.domain.entities.recomendacion import Recomendacion


class RecomendacionRepositoryPort(ABC):
    @abstractmethod
    async def save(self, r: Recomendacion) -> Recomendacion: ...

    @abstractmethod
    async def find_by_id(self, id: UUID) -> Recomendacion | None: ...

    @abstractmethod
    async def find_by_estudiante(
        self, estudiante_id: UUID, limit: int = 20
    ) -> list[Recomendacion]: ...

    @abstractmethod
    async def update_estado(self, id: UUID, estado: str) -> None: ...
