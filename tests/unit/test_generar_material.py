from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.application.use_cases.generar_material import (
    GenerarMaterialCommand,
    GenerarMaterialUseCase,
)
from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion


def _build_uc(llm_texto):
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
    uc = GenerarMaterialUseCase(trazabilidad, cursos, llm)
    return uc, cursos


@pytest.mark.asyncio
async def test_sin_api_key_devuelve_no_disponible():
    # El adapter sin clave retorna None -> material no disponible, pero no rompe.
    uc, cursos = _build_uc(llm_texto=None)
    material = await uc.execute(
        GenerarMaterialCommand(estudiante_id=uuid4(), curso_id=uuid4())
    )
    assert material.disponible is False
    assert material.concepto == "grafos"  # concepto más débil (0% aciertos)
    assert material.resumen == ""
    assert material.puntos_clave == []
    assert material.preguntas == []
    # Se piden recursos de la sección del concepto débil.
    assert cursos.obtener_recursos_candidatos.call_args.kwargs["seccion"] == "grafos"


@pytest.mark.asyncio
async def test_con_llm_parsea_json_envuelto_en_fences():
    texto = (
        "Aquí tienes:\n```json\n"
        '{"resumen": "Un grafo es...", "puntos_clave": ["a", "b"], '
        '"preguntas": [{"pregunta": "¿Qué es un nodo?", "respuesta": "..."}]}'
        "\n```"
    )
    uc, _ = _build_uc(llm_texto=texto)
    material = await uc.execute(
        GenerarMaterialCommand(estudiante_id=uuid4(), curso_id=uuid4())
    )
    assert material.disponible is True
    assert material.concepto == "grafos"
    assert material.resumen == "Un grafo es..."
    assert material.puntos_clave == ["a", "b"]
    assert material.preguntas[0]["pregunta"] == "¿Qué es un nodo?"
