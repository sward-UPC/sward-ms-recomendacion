"""Abrir material no es responder: las vistas no entran a la secuencia del SAKT.

La sincronización guarda cada lectura completada —un resumen, un video, la
solución de un ejemplo— como interacción con ``es_vista=True``, y con
``is_correct=True`` porque no hay nada que corregir. El entrenamiento las
descarta (el export filtra ``es_vista=False``), pero la secuencia con la que se
sirve el modelo no las filtraba: el SAKT recibía aciertos que nadie respondió.
Con tres temas, cada uno de cinco recursos de lectura y tres cuestionarios, más
de la mitad de la entrada era eso.

Se comprueba lo que se le pide a ms-trazabilidad —el filtro viaja en la
consulta, para que el `limit` no se gaste en vistas— y que si de todos modos
llegan vistas, no se cuelan.
"""

import asyncio
from uuid import uuid4

from src.infrastructure.adapters.out_ import trazabilidad_rest_adapter as modulo
from src.infrastructure.adapters.out_.trazabilidad_rest_adapter import (
    TrazabilidadRestAdapter,
)


def _item(concepto: str, fecha: str, es_vista: bool) -> dict:
    return {
        "id": concepto,
        "concept_id": concepto,
        "is_correct": True,
        "es_vista": es_vista,
        "fecha": fecha,
    }


class _Respuesta:
    status_code = 200

    def __init__(self, cuerpo):
        self._cuerpo = cuerpo

    def json(self):
        return self._cuerpo


class _ClienteQueDevuelveTodo:
    """Un ms-trazabilidad que ignora el filtro y devuelve también las vistas."""

    ultimos_params: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, *args, **kwargs):
        type(self).ultimos_params = kwargs.get("params", {})
        return _Respuesta(
            [
                _item("Interés simple", "2026-10-01T10:00:00+00:00", True),
                _item("Interés simple", "2026-10-01T11:00:00+00:00", False),
                _item("Amortización", "2026-10-01T12:00:00+00:00", True),
            ]
        )


def test_se_le_pide_a_trazabilidad_que_excluya_las_vistas(monkeypatch):
    monkeypatch.setattr(modulo.settings, "environment", "local")
    monkeypatch.setattr(modulo.httpx, "AsyncClient", _ClienteQueDevuelveTodo)

    asyncio.run(TrazabilidadRestAdapter().obtener_secuencia(uuid4(), uuid4()))

    assert _ClienteQueDevuelveTodo.ultimos_params.get("soloCalificadas") == "true"


def test_si_llegan_vistas_igual_no_entran_a_la_secuencia(monkeypatch):
    monkeypatch.setattr(modulo.settings, "environment", "local")
    monkeypatch.setattr(modulo.httpx, "AsyncClient", _ClienteQueDevuelveTodo)

    secuencia = asyncio.run(
        TrazabilidadRestAdapter().obtener_secuencia(uuid4(), uuid4())
    )

    assert secuencia.concepto_ids == ["Interés simple"]
    assert secuencia.respuestas_correctas == [True]
