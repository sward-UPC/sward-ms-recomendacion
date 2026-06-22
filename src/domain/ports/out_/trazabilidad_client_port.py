from abc import ABC, abstractmethod
from uuid import UUID

from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion


class TrazabilidadClientPort(ABC):
    @abstractmethod
    async def obtener_secuencia(
        self, estudiante_id: UUID, curso_id: UUID
    ) -> SecuenciaInteraccion: ...

    @abstractmethod
    async def obtener_preferencias(
        self, estudiante_id: UUID, curso_id: UUID
    ) -> dict | None:
        """Preferencia de formato del estudiante (rendimiento + engagement).

        Devuelve el dict de ms-trazabilidad (``tipo_fuerte``, ``por_tipo``,
        ``formato_mas_consumido``…) o ``None`` si no hay señal / falla.
        """
        ...
