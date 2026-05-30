from abc import ABC, abstractmethod
from uuid import UUID


class CursosClientPort(ABC):
    @abstractmethod
    async def obtener_recursos_candidatos(
        self, curso_id: UUID, limit: int = 10
    ) -> list[dict]: ...
