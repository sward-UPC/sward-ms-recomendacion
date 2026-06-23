from abc import ABC, abstractmethod


class YoutubeClientPort(ABC):
    """Puerto para buscar un video real en YouTube (best-effort)."""

    @abstractmethod
    async def buscar_video(self, query: str) -> dict | None:
        """Devuelve el primer video que coincide con ``query``, o None.

        Nunca debe lanzar: ante cualquier fallo (sin clave, error de red,
        respuesta vacía) retorna None para no romper el flujo del caso de uso.
        La forma del dict es::

            {"video_id": "abc123", "titulo": "...", "url": "https://..."}
        """
        ...
