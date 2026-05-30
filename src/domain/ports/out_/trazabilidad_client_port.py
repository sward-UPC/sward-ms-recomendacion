from abc import ABC, abstractmethod
from uuid import UUID

from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion


class TrazabilidadClientPort(ABC):
    @abstractmethod
    async def obtener_secuencia(
        self, estudiante_id: UUID, curso_id: UUID
    ) -> SecuenciaInteraccion: ...
