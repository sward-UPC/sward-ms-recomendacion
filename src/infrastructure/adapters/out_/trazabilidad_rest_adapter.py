from datetime import datetime, timezone
from uuid import UUID

import httpx

from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion
from src.application.ports.out_.trazabilidad_client_port import TrazabilidadClientPort
from src.infrastructure.config.settings import settings


def _cronologico(items: list[dict]) -> list[dict]:
    """Ordena las interacciones de la más antigua a la más reciente.

    ms-trazabilidad las devuelve de la más reciente a la más antigua (las 50
    últimas), y el modelo se entrenó con las secuencias en orden cronológico:
    sin reordenarlas, lo que el modelo toma como la interacción más reciente es
    en realidad la más antigua, y la predicción y la verificación de la
    explicación se calculan sobre la historia al revés.
    """

    def clave(item: dict) -> datetime:
        try:
            fecha = datetime.fromisoformat(item["fecha"])
        except (KeyError, TypeError, ValueError):
            return datetime.min.replace(tzinfo=timezone.utc)
        return fecha if fecha.tzinfo else fecha.replace(tzinfo=timezone.utc)

    return sorted(items, key=clave)


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
            items = _cronologico(r.json()) if r.status_code == 200 else []
        # Concepto = sección Moodle (concept_id); corrección real (is_correct).
        # Se descartan interacciones sin concepto (no aportan a la secuencia KT).
        con_concepto = [i for i in items if i.get("concept_id")]
        return SecuenciaInteraccion(
            estudiante_id=estudiante_id,
            curso_id=curso_id,
            concepto_ids=[str(i["concept_id"]) for i in con_concepto],
            respuestas_correctas=[bool(i.get("is_correct")) for i in con_concepto],
            interaccion_ids=[str(i.get("id") or "") for i in con_concepto],
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
