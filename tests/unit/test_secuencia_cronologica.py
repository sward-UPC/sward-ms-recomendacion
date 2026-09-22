"""La secuencia que recibe el modelo va de la interacción más antigua a la más
reciente, aunque ms-trazabilidad la devuelva al revés."""

import asyncio
from uuid import uuid4

from src.infrastructure.adapters.out_ import trazabilidad_rest_adapter as modulo
from src.infrastructure.adapters.out_.trazabilidad_rest_adapter import (
    TrazabilidadRestAdapter,
    _cronologico,
)


def _item(concepto: str, fecha: str, correcta: bool = True) -> dict:
    return {
        "id": concepto,
        "concept_id": concepto,
        "is_correct": correcta,
        "fecha": fecha,
    }


def test_ordena_de_la_mas_antigua_a_la_mas_reciente():
    items = [
        _item("c3", "2026-09-21T10:00:03.500000+00:00"),
        _item("c2", "2026-09-21T10:00:02+00:00"),
        _item("c1", "2026-09-21T10:00:01.000001+00:00"),
    ]
    assert [i["concept_id"] for i in _cronologico(items)] == ["c1", "c2", "c3"]


def test_fechas_sin_zona_o_ausentes_no_rompen_el_orden():
    items = [
        _item("c2", "2026-09-21T10:00:02"),
        {"id": "sin", "concept_id": "sin", "is_correct": True},
        _item("c1", "2026-09-21T10:00:01+00:00"),
    ]
    assert [i["concept_id"] for i in _cronologico(items)] == ["sin", "c1", "c2"]


class _Respuesta:
    status_code = 200

    def __init__(self, cuerpo):
        self._cuerpo = cuerpo

    def json(self):
        return self._cuerpo


class _Cliente:
    """Imita la respuesta de ms-trazabilidad: las más recientes primero."""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, *args, **kwargs):
        return _Respuesta(
            [
                _item("reciente", "2026-09-21T12:00:00+00:00", False),
                _item("medio", "2026-09-20T12:00:00+00:00", True),
                _item("antiguo", "2026-09-19T12:00:00+00:00", True),
            ]
        )


def test_obtener_secuencia_entrega_orden_cronologico(monkeypatch):
    monkeypatch.setattr(modulo.settings, "environment", "local")
    monkeypatch.setattr(modulo.httpx, "AsyncClient", _Cliente)

    secuencia = asyncio.run(
        TrazabilidadRestAdapter().obtener_secuencia(uuid4(), uuid4())
    )

    assert secuencia.concepto_ids == ["antiguo", "medio", "reciente"]
    assert secuencia.respuestas_correctas == [True, True, False]
    assert secuencia.interaccion_ids == ["antiguo", "medio", "reciente"]
