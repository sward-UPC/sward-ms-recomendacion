import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from src.application.use_cases.generar_recomendacion import (
    GenerarRecomendacionCommand,
    GenerarRecomendacionUseCase,
)
from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion
from src.domain.services.modelo_sakt import ModeloSAKT

CANDIDATOS = [
    {
        "id": "r1",
        "titulo": "Intro",
        "tipo": "video",
        "nivel_dificultad": "basico",
        "url": "",
    },
    {
        "id": "r2",
        "titulo": "Medio",
        "tipo": "lectura",
        "nivel_dificultad": "intermedio",
        "url": "",
    },
    {
        "id": "r3",
        "titulo": "Avanzado",
        "tipo": "ejercicio",
        "nivel_dificultad": "avanzado",
        "url": "",
    },
]


@pytest.fixture
def use_case():
    trazabilidad = AsyncMock()
    trazabilidad.obtener_secuencia.return_value = SecuenciaInteraccion(
        estudiante_id=uuid4(),
        curso_id=uuid4(),
        concepto_ids=["c1"],
        respuestas_correctas=[True],
    )
    cursos = AsyncMock()
    cursos.obtener_recursos_candidatos.return_value = CANDIDATOS
    xai = AsyncMock()
    xai.generar_explicacion.return_value = {}
    repo = AsyncMock()
    repo.save.side_effect = lambda r: r
    return GenerarRecomendacionUseCase(
        trazabilidad, cursos, xai, repo, MagicMock(), ModeloSAKT(mock=True)
    )


@pytest.mark.asyncio
async def test_genera_minimo_3_items(use_case):
    rec = await use_case.execute(
        GenerarRecomendacionCommand(estudiante_id=uuid4(), curso_id=uuid4())
    )
    assert len(rec.items) >= 3


@pytest.mark.asyncio
async def test_items_tienen_motivo(use_case):
    rec = await use_case.execute(
        GenerarRecomendacionCommand(estudiante_id=uuid4(), curso_id=uuid4())
    )
    assert all(item.motivo for item in rec.items)


@pytest.mark.asyncio
async def test_publica_evento(use_case):
    await use_case.execute(
        GenerarRecomendacionCommand(estudiante_id=uuid4(), curso_id=uuid4())
    )
    use_case._event_publisher.publish.assert_called_once()


@pytest.mark.asyncio
async def test_xai_falla_no_bloquea(use_case):
    use_case._xai.generar_explicacion.side_effect = Exception("XAI no disponible")
    rec = await use_case.execute(
        GenerarRecomendacionCommand(estudiante_id=uuid4(), curso_id=uuid4())
    )
    assert len(rec.items) >= 3
