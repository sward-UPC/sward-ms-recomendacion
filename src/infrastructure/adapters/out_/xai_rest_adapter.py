from uuid import UUID

import httpx

from src.domain.ports.out_.xai_client_port import XaiClientPort
from src.infrastructure.config.settings import settings


class XaiRestAdapter(XaiClientPort):
    async def generar_explicacion(
        self, recomendacion_id: UUID, pesos_atencion: list[float]
    ) -> dict:
        if settings.environment == "development":
            return {"status": "mock"}
        headers = {"X-Service-Key": settings.service_key} if settings.service_key else {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                f"{settings.xai_service_url}/xai/explain",
                json={
                    "recomendacion_id": str(recomendacion_id),
                    "pesos_atencion": pesos_atencion,
                },
                headers=headers,
            )
            return r.json() if r.status_code == 200 else {}
