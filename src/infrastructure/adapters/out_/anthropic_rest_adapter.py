import logging

import httpx

from src.domain.ports.out_.llm_client_port import LlmClientPort
from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"


class AnthropicRestAdapter(LlmClientPort):
    """Cliente del API de mensajes de Anthropic vía httpx (sin SDK).

    Best-effort: si no hay clave configurada o la llamada falla, retorna None.
    """

    async def generar_texto(self, prompt: str) -> str | None:
        if not settings.anthropic_api_key:
            return None
        headers = {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": settings.anthropic_model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(
                    ANTHROPIC_MESSAGES_URL, json=body, headers=headers
                )
            if r.status_code != 200:
                logger.warning("Anthropic respondió %s", r.status_code)
                return None
            data = r.json()
            return data["content"][0]["text"]
        except Exception as e:  # best-effort: cualquier fallo => None
            logger.warning("Llamada a Anthropic falló: %s", e)
            return None
