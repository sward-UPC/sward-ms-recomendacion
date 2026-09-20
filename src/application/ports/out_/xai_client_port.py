from abc import ABC, abstractmethod
from uuid import UUID

from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion


class XaiClientPort(ABC):
    @abstractmethod
    async def generar_explicacion(
        self,
        recomendacion_id: UUID,
        pesos_atencion: list[float],
        secuencia: SecuenciaInteraccion,
    ) -> dict:
        """Registra en ms-xai la explicacion de una recomendacion.

        Cada peso se ancla a la interaccion pasada que lo recibio. Devuelve el
        cuerpo de ms-xai, o {} si no pudo registrarse.
        """
