"""Defecto 24 (HU-040, esc. 3; CP102, grupo 10).

A un estudiante sin actividades calificadas el motivo le decía «llevas 50% de
aciertos en el curso». Ese 50 % es el valor que el modelo usa por defecto cuando
no hay historial, no un dato del estudiante. El motivo no debe afirmar ningún
porcentaje: los recursos son su punto de partida en el curso.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.application.use_cases.generar_recomendacion import (
    GenerarRecomendacionCommand,
    GenerarRecomendacionUseCase,
)
from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion
from src.infrastructure.adapters.out_.modelo_kt_mock_adapter import ModeloKtMockAdapter

CANDIDATOS = [
    {"id": "r1", "titulo": "Video", "tipo": "video", "nivel_dificultad": "basico"},
    {"id": "r2", "titulo": "Lectura", "tipo": "lectura", "nivel_dificultad": "basico"},
    {"id": "r3", "titulo": "Quiz", "tipo": "quiz", "nivel_dificultad": "intermedio"},
]


def _caso(concepto_ids, respuestas):
    trazabilidad = AsyncMock()
    trazabilidad.obtener_preferencias.return_value = None
    trazabilidad.obtener_secuencia.return_value = SecuenciaInteraccion(
        estudiante_id=uuid4(),
        curso_id=uuid4(),
        concepto_ids=concepto_ids,
        respuestas_correctas=respuestas,
    )
    cursos = AsyncMock()
    cursos.obtener_recursos_candidatos.return_value = CANDIDATOS
    repo = AsyncMock()
    repo.save.side_effect = lambda r: r
    return GenerarRecomendacionUseCase(
        trazabilidad, cursos, AsyncMock(), repo, MagicMock(), ModeloKtMockAdapter()
    )


@pytest.mark.asyncio
async def test_sin_historial_el_motivo_no_afirma_un_porcentaje():
    rec = await _caso([], []).execute(
        GenerarRecomendacionCommand(estudiante_id=uuid4(), curso_id=uuid4())
    )
    assert rec.items, "sin historial igual debe recomendar recursos del curso"
    for item in rec.items:
        assert "%" not in item.motivo, item.motivo
        assert "aciertos" not in item.motivo, item.motivo
        assert "punto de partida" in item.motivo.lower(), item.motivo


@pytest.mark.asyncio
async def test_con_historial_el_motivo_sigue_dando_su_cifra():
    rec = await _caso(["c1", "c1"], [True, False]).execute(
        GenerarRecomendacionCommand(estudiante_id=uuid4(), curso_id=uuid4())
    )
    assert rec.items
    assert all("50%" in i.motivo for i in rec.items)
