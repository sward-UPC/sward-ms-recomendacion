from abc import ABC, abstractmethod
from uuid import UUID


class AsignacionNoDisponibleError(Exception):
    """No se pudo saber quién dicta el curso: servicio caído, tiempo agotado o
    una respuesta que no se entiende. Quien la recibe no debe entregar datos."""


class CursosClientPort(ABC):
    @abstractmethod
    async def obtener_recursos_candidatos(
        self, curso_id: UUID, limit: int = 10, seccion: str | None = None
    ) -> list[dict]: ...

    @abstractmethod
    async def docente_del_curso(self, curso_id: UUID, token: str) -> UUID | None:
        """UUID del docente asignado al curso.

        Devuelve None si el curso no existe o no tiene docente asignado. Lanza
        ``AsignacionNoDisponibleError`` si la consulta no se pudo hacer, para que
        quien llama no confunda «no lo sé» con «no lo dicta».
        """
        ...
