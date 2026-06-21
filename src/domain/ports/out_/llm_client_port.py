from abc import ABC, abstractmethod


class LlmClientPort(ABC):
    """Puerto para un cliente LLM generativo (best-effort)."""

    @abstractmethod
    async def generar_texto(self, prompt: str) -> str | None:
        """Devuelve el texto generado por el LLM, o None si no está disponible.

        Nunca debe lanzar: ante cualquier fallo (sin clave, error de red,
        respuesta inválida) retorna None para no romper el flujo del caso de uso.
        """
        ...
