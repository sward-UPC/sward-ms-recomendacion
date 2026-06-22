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
        # Concepto = sección Moodle (concept_id); corrección real (is_correct).
        # Se descartan interacciones sin concepto (no aportan a la secuencia KT).
        con_concepto = [i for i in items if i.get("concept_id")]
        return SecuenciaInteraccion(
            estudiante_id=estudiante_id,
            curso_id=curso_id,
            concepto_ids=[str(i["concept_id"]) for i in con_concepto],
            respuestas_correctas=[bool(i.get("is_correct")) for i in con_concepto],
        )

    async def obtener_preferencias(
        self, estudiante_id: UUID, curso_id: UUID
    ) -> dict | None:
        # En dev no hay señal real de preferencia (secuencia mockeada) → None.
        if settings.environment == "development":
            return None
        headers = (
            {"X-Service-Key": settings.service_key} if settings.service_key else {}
        )
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    f"{settings.trazabilidad_service_url}"
                    f"/internal/students/{estudiante_id}/preferences",
                    params={"courseId": str(curso_id)},
                    headers=headers,
                )
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None  # best-effort: la preferencia no debe romper la recomendación
