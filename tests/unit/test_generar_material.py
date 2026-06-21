from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.application.use_cases.generar_material import (
    GenerarMaterialCommand,
    GenerarMaterialUseCase,
)
from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion


def _build_uc(llm_texto, video=None):
    trazabilidad = AsyncMock()
    trazabilidad.obtener_secuencia.return_value = SecuenciaInteraccion(
        estudiante_id=uuid4(),
        curso_id=uuid4(),
        concepto_ids=["grafos", "grafos", "listas"],
        respuestas_correctas=[False, False, True],
    )
    cursos = AsyncMock()
    cursos.obtener_recursos_candidatos.return_value = [
        {"id": "r1", "titulo": "Grafos 101", "tipo": "lectura"},
    ]
    llm = AsyncMock()
    llm.generar_texto.return_value = llm_texto
    youtube = AsyncMock()
    youtube.buscar_video.return_value = video
    uc = GenerarMaterialUseCase(trazabilidad, cursos, llm, youtube)
    return uc, cursos, youtube


@pytest.mark.asyncio
async def test_sin_api_key_devuelve_no_disponible():
    # El adapter sin clave retorna None -> material no disponible, pero no rompe.
    uc, cursos, _ = _build_uc(llm_texto=None)
    material = await uc.execute(
        GenerarMaterialCommand(estudiante_id=uuid4(), curso_id=uuid4())
    )
    assert material.disponible is False
    assert material.concepto == "grafos"  # concepto más débil (0% aciertos)
    assert material.recursos == []
    # Se piden recursos de la sección del concepto débil.
    assert cursos.obtener_recursos_candidatos.call_args.kwargs["seccion"] == "grafos"


@pytest.mark.asyncio
async def test_con_llm_parsea_json_envuelto_en_fences_y_arma_recursos():
    texto = (
        "Aquí tienes:\n```json\n"
        "{"
        '"quiz": {"titulo": "Quiz de grafos", "preguntas": ['
        '{"enunciado": "¿Qué es un nodo?", "opciones": ["a", "b", "c", "d"], '
        '"correcta": 1, "explicacion": "Un nodo es..."}]}, '
        '"lectura": {"titulo": "Grafos", "contenido": "Un grafo es un conjunto..."}, '
        '"practica": {"titulo": "Ejercicios", "ejercicios": ['
        '{"enunciado": "Dibuja un grafo", "solucion": "..."}]}, '
        '"video_query": "introducción a grafos"'
        "}"
        "\n```"
    )
    video = {
        "video_id": "abc123",
        "titulo": "Grafos para principiantes",
        "url": "https://www.youtube.com/watch?v=abc123",
    }
    uc, _, youtube = _build_uc(llm_texto=texto, video=video)
    material = await uc.execute(
        GenerarMaterialCommand(estudiante_id=uuid4(), curso_id=uuid4())
    )

    assert material.disponible is True
    assert material.concepto == "grafos"

    por_tipo = {r["tipo"]: r for r in material.recursos}
    assert set(por_tipo) == {"quiz", "lectura", "practica", "video"}

    quiz = por_tipo["quiz"]
    assert quiz["titulo"] == "Quiz de grafos"
    assert quiz["preguntas"][0]["enunciado"] == "¿Qué es un nodo?"
    assert quiz["preguntas"][0]["opciones"] == ["a", "b", "c", "d"]
    assert quiz["preguntas"][0]["correcta"] == 1
    assert quiz["preguntas"][0]["explicacion"] == "Un nodo es..."

    assert por_tipo["lectura"]["contenido"] == "Un grafo es un conjunto..."
    assert por_tipo["practica"]["ejercicios"][0]["solucion"] == "..."

    video_recurso = por_tipo["video"]
    assert video_recurso["video_id"] == "abc123"
    assert video_recurso["url"] == "https://www.youtube.com/watch?v=abc123"
    assert video_recurso["query"] == "introducción a grafos"
    # La query del LLM se usa para buscar el video.
    youtube.buscar_video.assert_awaited_once_with("introducción a grafos")


@pytest.mark.asyncio
async def test_sin_video_omite_recurso_video():
    texto = (
        "{"
        '"quiz": {"titulo": "Q", "preguntas": ['
        '{"enunciado": "e", "opciones": ["a", "b", "c", "d"], '
        '"correcta": 0, "explicacion": "x"}]}, '
        '"lectura": {"titulo": "L", "contenido": "texto"}, '
        '"practica": {"titulo": "P", "ejercicios": ['
        '{"enunciado": "e", "solucion": "s"}]}, '
        '"video_query": "grafos"'
        "}"
    )
    # YouTube best-effort devuelve None -> no hay recurso de video.
    uc, _, _ = _build_uc(llm_texto=texto, video=None)
    material = await uc.execute(
        GenerarMaterialCommand(estudiante_id=uuid4(), curso_id=uuid4())
    )
    assert material.disponible is True
    tipos = {r["tipo"] for r in material.recursos}
    assert tipos == {"quiz", "lectura", "practica"}
    assert "video" not in tipos
