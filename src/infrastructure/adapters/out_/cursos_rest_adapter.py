from uuid import UUID

import httpx

from src.application.ports.out_.cursos_client_port import (
    AsignacionNoDisponibleError,
    CursosClientPort,
)
from src.infrastructure.config.settings import settings

MOCK_RESOURCES = [
    {
        "id": "r1",
        "titulo": "Intro a Algoritmos",
        "tipo": "video",
        "nivel_dificultad": "basico",
        "url": "",
    },
    {
        "id": "r2",
        "titulo": "Listas Enlazadas",
        "tipo": "lectura",
        "nivel_dificultad": "intermedio",
        "url": "",
    },
    {
        "id": "r3",
        "titulo": "Árboles Binarios",
        "tipo": "ejercicio",
        "nivel_dificultad": "intermedio",
        "url": "",
    },
    {
        "id": "r4",
        "titulo": "Grafos Avanzados",
        "tipo": "quiz",
        "nivel_dificultad": "avanzado",
        "url": "",
    },
    {
        "id": "r5",
        "titulo": "Complejidad Algorítmica",
        "tipo": "lectura",
        "nivel_dificultad": "avanzado",
        "url": "",
    },
]


class CursosRestAdapter(CursosClientPort):
    async def obtener_recursos_candidatos(
        self, curso_id: UUID, limit: int = 10, seccion: str | None = None
    ) -> list[dict]:
        if settings.environment == "development":
            # El mock ignora la sección.
            return MOCK_RESOURCES[:limit]
        headers = (
            {"X-Service-Key": settings.service_key} if settings.service_key else {}
        )
        params: dict = {"courseId": str(curso_id), "limit": limit}
        if seccion:
            params["seccion"] = seccion
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{settings.cursos_service_url}/resources/candidates",
                params=params,
                headers=headers,
            )
            return r.json() if r.status_code == 200 else []

    async def docente_del_curso(self, curso_id: UUID, token: str) -> UUID | None:
        """Quién dicta el curso: la columna ``docente_id``, que pone el administrador.

        ``GET /courses/{id}`` solo acepta el JWT de un usuario —no hay ruta
        interna con service-key para leer cursos—, así que se reenvía el token
        de quien hizo la petición, que ya se validó aquí.

        A diferencia de los recursos candidatos, en desarrollo no hay simulación:
        inventar una asignación abriría o cerraría la puerta a ciegas, y
        ``ENVIRONMENT`` vale «development» por defecto. Sin ms-cursos-recursos el
        docente recibe un 503; el administrador no depende de esta consulta.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    f"{settings.cursos_service_url}/courses/{curso_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
        except httpx.HTTPError as exc:
            raise AsignacionNoDisponibleError(
                f"ms-cursos-recursos no respondió: {type(exc).__name__}"
            ) from exc
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            raise AsignacionNoDisponibleError(
                f"ms-cursos-recursos respondió HTTP {r.status_code}"
            )
        try:
            docente_id = r.json().get("docente_id")
            return UUID(str(docente_id)) if docente_id else None
        except (ValueError, TypeError, AttributeError) as exc:
            raise AsignacionNoDisponibleError(
                "ms-cursos-recursos devolvió un curso que no se entiende"
            ) from exc
