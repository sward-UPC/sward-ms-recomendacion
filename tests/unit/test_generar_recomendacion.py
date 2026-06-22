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
    trazabilidad.obtener_preferencias.return_value = None
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


@pytest.mark.asyncio
async def test_excluye_foros(use_case):
    use_case._cursos.obtener_recursos_candidatos.return_value = [
        {"id": "f1", "titulo": "Foro de dudas", "tipo": "foro", "url": ""},
        {"id": "f2", "titulo": "Announcements", "tipo": "lectura", "url": ""},
        {"id": "v1", "titulo": "Video clase", "tipo": "video", "url": "u"},
    ]
    rec = await use_case.execute(
        GenerarRecomendacionCommand(estudiante_id=uuid4(), curso_id=uuid4())
    )
    titulos = [i.titulo for i in rec.items]
    assert "Foro de dudas" not in titulos
    assert "Announcements" not in titulos
    assert "Video clase" in titulos


@pytest.mark.asyncio
async def test_prioriza_estudio_sobre_practica(use_case):
    use_case._cursos.obtener_recursos_candidatos.return_value = [
        {"id": "e1", "titulo": "Quiz", "tipo": "quiz", "url": ""},
        {"id": "l1", "titulo": "Lectura", "tipo": "lectura", "url": ""},
    ]
    rec = await use_case.execute(
        GenerarRecomendacionCommand(estudiante_id=uuid4(), curso_id=uuid4())
    )
    assert rec.items[0].tipo == "lectura"


@pytest.mark.asyncio
async def test_targetea_concepto_debil_y_salta_si_vacio():
    trazabilidad = AsyncMock()
    trazabilidad.obtener_preferencias.return_value = None
    trazabilidad.obtener_secuencia.return_value = SecuenciaInteraccion(
        estudiante_id=uuid4(),
        curso_id=uuid4(),
        concepto_ids=["grafos", "grafos", "listas"],
        respuestas_correctas=[False, False, True],
    )
    cursos = AsyncMock()
    # grafos (más débil) sin recursos -> salta al siguiente concepto débil (listas).
    cursos.obtener_recursos_candidatos.side_effect = [[], CANDIDATOS]
    xai = AsyncMock()
    repo = AsyncMock()
    repo.save.side_effect = lambda r: r
    uc = GenerarRecomendacionUseCase(
        trazabilidad, cursos, xai, repo, MagicMock(), ModeloSAKT(mock=True)
    )
    rec = await uc.execute(
        GenerarRecomendacionCommand(estudiante_id=uuid4(), curso_id=uuid4())
    )
    # El concepto más débil (grafos: 0% aciertos) se consulta primero como sección.
    primera = cursos.obtener_recursos_candidatos.call_args_list[0]
    assert primera.kwargs.get("seccion") == "grafos"
    # Como grafos vino vacío, los items provienen del siguiente débil: listas.
    assert len(rec.items) >= 1
    assert all("listas" in i.motivo for i in rec.items)


@pytest.mark.asyncio
async def test_cubre_varios_conceptos_debiles():
    """Con varios conceptos débiles, la recomendación cubre más de uno (no solo 1)."""
    trazabilidad = AsyncMock()
    trazabilidad.obtener_preferencias.return_value = None
    trazabilidad.obtener_secuencia.return_value = SecuenciaInteraccion(
        estudiante_id=uuid4(),
        curso_id=uuid4(),
        concepto_ids=["grafos", "listas", "arboles"],
        respuestas_correctas=[False, False, False],
    )
    cursos = AsyncMock()
    cursos.obtener_recursos_candidatos.return_value = CANDIDATOS
    repo = AsyncMock()
    repo.save.side_effect = lambda r: r
    uc = GenerarRecomendacionUseCase(
        trazabilidad, cursos, AsyncMock(), repo, MagicMock(), ModeloSAKT(mock=True)
    )
    rec = await uc.execute(
        GenerarRecomendacionCommand(estudiante_id=uuid4(), curso_id=uuid4())
    )
    conceptos_cubiertos = {
        c
        for c in ("grafos", "listas", "arboles")
        if any(c in i.motivo for i in rec.items)
    }
    assert len(conceptos_cubiertos) >= 2
