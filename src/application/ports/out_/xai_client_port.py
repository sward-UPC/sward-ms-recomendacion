from abc import ABC, abstractmethod
from uuid import UUID


class XaiClientPort(ABC):
    @abstractmethod
    async def generar_explicacion(
        self, recomendacion_id: UUID, pesos_atencion: list[float]
    ) -> dict: ...
