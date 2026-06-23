import asyncio
import logging

import boto3

from src.application.ports.out_.llm_client_port import LlmClientPort
from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


class BedrockLlmAdapter(LlmClientPort):
    """Cliente LLM vía AWS Bedrock (Converse API).

    Usa las credenciales IAM del task role de ECS — NO requiere API key. El
    modelo Claude debe estar habilitado en Bedrock (Model access) para la región.
    Best-effort: cualquier fallo (modelo no habilitado, sin permisos, etc.) => None.
    """

    async def generar_texto(self, prompt: str) -> str | None:
        try:
            return await asyncio.to_thread(self._invoke, prompt)
        except Exception as e:  # best-effort: cualquier fallo => None
            # print además del logger: el logger custom no se captura en CloudWatch.
            print(f"[BEDROCK] falló: {e}", flush=True)
            logger.warning("Bedrock falló: %s", e)
            return None

    def _invoke(self, prompt: str) -> str | None:
        client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        resp = client.converse(
            modelId=settings.bedrock_model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={
                "maxTokens": settings.bedrock_max_tokens,
                "temperature": 0.4,
            },
        )
        stop = resp.get("stopReason")
        if stop == "max_tokens":
            print("[BEDROCK] respuesta truncada por max_tokens", flush=True)
        return resp["output"]["message"]["content"][0]["text"]
