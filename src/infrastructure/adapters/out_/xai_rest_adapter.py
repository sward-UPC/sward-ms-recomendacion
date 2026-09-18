import logging
from uuid import UUID

import httpx

from src.application.ports.out_.xai_client_port import XaiClientPort
from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion
from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


def construir_pesos(
    pesos_atencion: list[float], secuencia: SecuenciaInteraccion
) -> list[dict]:
    """Pesos en el formato de `POST /xai/explain`.

    ms-xai no acepta una lista de numeros: exige, por cada peso, la interaccion a
    la que corresponde y su concepto, y rechaza campos extra (422). La alineacion
    es la misma que usa el heatmap de /attention: el peso i corresponde a la
    interaccion pasada i, y zip recorta al menor.

    Si trazabilidad no entrego el id de una interaccion, se usa una referencia
    posicional estable (`<secuencia>:<i>`) para no perder el peso.
    """
    ids = secuencia.interaccion_ids
    return [
        {
            "interaccion_referencia_id": (ids[i] if i < len(ids) and ids[i] else f"{secuencia.id}:{i}"),
            "peso": round(float(peso), 6),
            "concepto": str(concepto),
        }
        for i, (peso, concepto) in enumerate(zip(pesos_atencion, secuencia.concepto_ids))
    ]


class XaiRestAdapter(XaiClientPort):
    async def generar_explicacion(
        self,
        recomendacion_id: UUID,
        pesos_atencion: list[float],
        secuencia: SecuenciaInteraccion,
    ) -> dict:
        if settings.environment == "development":
            return {"status": "mock"}
        pesos = construir_pesos(pesos_atencion, secuencia)
        if not pesos:
            return {}
        headers = (
            {"X-Service-Key": settings.service_key} if settings.service_key else {}
        )
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                f"{settings.xai_service_url}/xai/explain",
                json={"recomendacion_id": str(recomendacion_id), "pesos_atencion": pesos},
                headers=headers,
            )
        # ms-xai responde 201 Created. Tratar solo el 200 como exito descartaba
        # cada explicacion registrada correctamente.
        if 200 <= r.status_code < 300:
            return r.json()
        logger.warning(
            "ms-xai rechazo la explicacion de %s: HTTP %s %s",
            recomendacion_id,
            r.status_code,
            r.text[:300],
        )
        return {}
