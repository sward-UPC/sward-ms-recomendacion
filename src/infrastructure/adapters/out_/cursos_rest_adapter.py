from uuid import UUID

import httpx

from src.domain.ports.out_.cursos_client_port import CursosClientPort
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
        self, curso_id: UUID, limit: int = 10
    ) -> list[dict]:
        if settings.environment == "development":
            return MOCK_RESOURCES[:limit]
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{settings.cursos_service_url}/resources/candidates",
                params={"courseId": str(curso_id), "limit": limit},
            )
            return r.json() if r.status_code == 200 else []
