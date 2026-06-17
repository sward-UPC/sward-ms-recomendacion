from uuid import UUID

import httpx

from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion
from src.domain.ports.out_.trazabilidad_client_port import TrazabilidadClientPort
from src.infrastructure.config.settings import settings


class TrazabilidadRestAdapter(TrazabilidadClientPort):
    async def obtener_secuencia(
        self, estudiante_id: UUID, curso_id: UUID
    ) -> SecuenciaInteraccion:
        if settings.environment == "development":
            return SecuenciaInteraccion(
                estudiante_id=estudiante_id,
                curso_id=curso_id,
                concepto_ids=["c1", "c2", "c1", "c3"],
                respuestas_correctas=[True, False, True, True],
            )
        headers = (
            {"X-Service-Key": settings.service_key} if settings.service_key else {}
        )
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{settings.trazabilidad_service_url}/internal/students/{estudiante_id}/interactions",
                params={"courseId": str(curso_id), "limit": 50},
                headers=headers,
            )
            items = r.json() if r.status_code == 200 else []
        return SecuenciaInteraccion(
            estudiante_id=estudiante_id,
            curso_id=curso_id,
            concepto_ids=[str(i.get("actividad_id", "")) for i in items],
            respuestas_correctas=[
                i.get("tipo", "").upper() == "COMPLETADO" for i in items
            ],
        )
