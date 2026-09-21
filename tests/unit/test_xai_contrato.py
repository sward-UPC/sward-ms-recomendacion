"""Contrato con ms-xai y frescura de la recomendación (defecto 16).

ms-xai exige, por cada peso de atención, la interacción a la que corresponde y
su concepto, rechaza campos extra (422) y responde 201 al registrar. Antes se
enviaba una lista de números, se trataba el 201 como fallo y el error se
tragaba en silencio: ninguna explicación llegaba nunca a ms-xai.
"""

import logging
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4, uuid5

import httpx
import pytest

from src.application.use_cases import generar_recomendacion as modulo
from src.application.use_cases.generar_recomendacion import (
    GenerarRecomendacionCommand,
    GenerarRecomendacionUseCase,
)
from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion
from src.infrastructure.adapters.out_ import xai_rest_adapter
from src.infrastructure.adapters.out_.modelo_kt_mock_adapter import ModeloKtMockAdapter
from src.infrastructure.adapters.out_.xai_rest_adapter import (
    XaiRestAdapter,
    construir_pesos,
)


def _secuencia(ids=("i1", "i2", "i3")):
    return SecuenciaInteraccion(
        concepto_ids=["Listas", "Árboles", "Grafos"],
        respuestas_correctas=[True, False, True],
        interaccion_ids=list(ids),
    )


# ---------------------------------------------------------------------------
# Forma del cuerpo
# ---------------------------------------------------------------------------


def test_cada_peso_lleva_su_interaccion_y_concepto():
    pesos = construir_pesos([0.2, 0.5, 0.3], _secuencia())
    assert pesos == [
        {"interaccion_referencia_id": "i1", "peso": 0.2, "concepto": "Listas"},
        {"interaccion_referencia_id": "i2", "peso": 0.5, "concepto": "Árboles"},
        {"interaccion_referencia_id": "i3", "peso": 0.3, "concepto": "Grafos"},
    ]


def test_solo_los_campos_que_ms_xai_acepta():
    # PesoAtencionRequest tiene extra="forbid": cualquier campo de más es un 422.
    for peso in construir_pesos([0.5, 0.5], _secuencia()):
        assert set(peso) == {"interaccion_referencia_id", "peso", "concepto"}


def test_alineacion_igual_que_el_heatmap():
    # Hay un peso menos que interacciones: el ultimo paso es el que se predice.
    pesos = construir_pesos([0.6, 0.4], _secuencia())
    assert [p["concepto"] for p in pesos] == ["Listas", "Árboles"]


def test_sin_id_de_trazabilidad_usa_referencia_posicional():
    sec = _secuencia(ids=("i1", "", "i3"))
    pesos = construir_pesos([0.2, 0.5, 0.3], sec)
    ref = pesos[1]["interaccion_referencia_id"]
    # ms-xai valida el campo como UUID: la referencia de respaldo debe serlo.
    assert UUID(ref) == uuid5(sec.id, "1")
    assert construir_pesos([0.2, 0.5, 0.3], sec)[1]["interaccion_referencia_id"] == ref


# ---------------------------------------------------------------------------
# Respuesta de ms-xai
# ---------------------------------------------------------------------------


def _cliente_que_responde(estado, cuerpo, capturado):
    class _Respuesta:
        status_code = estado
        text = str(cuerpo)

        def json(self):
            return cuerpo

    class _Cliente:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            capturado["url"] = url
            capturado["json"] = json
            return _Respuesta()

    return lambda *a, **k: _Cliente()


@pytest.mark.asyncio
async def test_201_es_exito(monkeypatch):
    capturado = {}
    monkeypatch.setattr(xai_rest_adapter.settings, "environment", "local")
    monkeypatch.setattr(
        httpx, "AsyncClient", _cliente_que_responde(201, {"id": "x"}, capturado)
    )

    r = await XaiRestAdapter().generar_explicacion(
        uuid4(), [0.2, 0.5, 0.3], _secuencia()
    )

    assert r == {"id": "x"}
    assert capturado["url"].endswith("/xai/explain")
    assert len(capturado["json"]["pesos_atencion"]) == 3


@pytest.mark.asyncio
async def test_un_rechazo_se_registra_en_el_log(monkeypatch, caplog):
    monkeypatch.setattr(xai_rest_adapter.settings, "environment", "local")
    monkeypatch.setattr(
        httpx, "AsyncClient", _cliente_que_responde(422, {"detail": "extra"}, {})
    )

    with caplog.at_level(logging.WARNING):
        r = await XaiRestAdapter().generar_explicacion(
            uuid4(), [0.5, 0.5], _secuencia()
        )

    assert r == {}
    assert "422" in caplog.text


# ---------------------------------------------------------------------------
# Frescura de la cache
# ---------------------------------------------------------------------------


def _caso_de_uso(trazabilidad):
    cursos = AsyncMock()
    cursos.obtener_recursos_candidatos.return_value = [
        {"id": "r1", "titulo": "Práctica", "tipo": "ejercicio", "url": "u1"},
        {"id": "r2", "titulo": "Lectura", "tipo": "lectura", "url": "u2"},
        {"id": "r3", "titulo": "Quiz", "tipo": "quiz", "url": "u3"},
    ]
    repo = AsyncMock()
    repo.save.side_effect = lambda r: r
    return GenerarRecomendacionUseCase(
        trazabilidad, cursos, AsyncMock(), repo, MagicMock(), ModeloKtMockAdapter()
    )


@pytest.mark.asyncio
async def test_una_interaccion_nueva_invalida_la_cache(monkeypatch):
    monkeypatch.setattr(modulo, "_RECOMENDACION_CACHE", {})
    trazabilidad = AsyncMock()
    trazabilidad.obtener_preferencias.return_value = None
    trazabilidad.obtener_secuencia.return_value = _secuencia()
    uc = _caso_de_uso(trazabilidad)
    cmd = GenerarRecomendacionCommand(estudiante_id=uuid4(), curso_id=uuid4())

    primera = await uc.execute(cmd)
    assert (await uc.execute(cmd)) is primera  # mismo historial: cache

    trazabilidad.obtener_secuencia.return_value = _secuencia(
        ids=("i1", "i2", "i3", "i4")
    )
    trazabilidad.obtener_secuencia.return_value.concepto_ids.append("Pilas")
    trazabilidad.obtener_secuencia.return_value.respuestas_correctas.append(True)

    assert (await uc.execute(cmd)) is not primera  # historial nuevo: recalcula
