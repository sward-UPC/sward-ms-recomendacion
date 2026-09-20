"""Heatmap de atención con el veredicto de fidelidad.

Cubre que lo que llega al cliente sea coherente con lo verificado: solo se
marcan interacciones «suficientes» y solo se publica un contrafactual cuando las
pruebas lo respaldan.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from src.application.use_cases.consultar_atencion import ConsultarAtencionUseCase
from src.domain.entities.fidelidad_explicacion import (
    CRITERIO_EXHAUSTIVIDAD,
    CRITERIO_SUFICIENCIA,
    MOTIVO_FIEL,
    MOTIVO_NO_SUPERA_AZAR,
    Contrafactual,
    FidelidadExplicacion,
    PruebaFidelidad,
)
from src.domain.entities.prediccion_kt import PrediccionKT
from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion
from src.infrastructure.adapters.in_.recomendacion_router import _fidelidad_response


def _prueba(criterio, supera, k=1):
    return PruebaFidelidad(criterio=criterio, k=k, n_aleatorios=20, supera_azar_en=supera)


def _suficiente_no_necesaria():
    return FidelidadExplicacion(
        criterio=CRITERIO_SUFICIENCIA,
        n_pasado=3,
        suficiencia=_prueba(CRITERIO_SUFICIENCIA, 18),
        exhaustividad=_prueba(CRITERIO_EXHAUSTIVIDAD, 8, k=3),
        indices_suficientes=[1],
        conceptos_suficientes=["Árboles"],
        motivo=MOTIVO_FIEL,
    )


async def _consultar(fidelidad):
    secuencia = SecuenciaInteraccion(
        concepto_ids=["Listas", "Árboles", "Grafos", "Pilas"],
        respuestas_correctas=[True, False, True, True],
    )
    trazabilidad = MagicMock()
    trazabilidad.obtener_secuencia = AsyncMock(return_value=secuencia)
    modelo = MagicMock()
    modelo.predecir_dominio.return_value = PrediccionKT(
        probabilidad_dominio=0.64,
        pesos_atencion=[0.2, 0.5, 0.3],
        fidelidad=fidelidad,
    )
    return await ConsultarAtencionUseCase(trazabilidad, modelo).execute(uuid4(), uuid4())


async def test_marca_solo_las_interacciones_verificadas_como_suficientes():
    res = await _consultar(_suficiente_no_necesaria())
    assert [p.suficiente for p in res.puntos] == [False, True, False]
    assert res.fidelidad.es_fiel


async def test_sin_fidelidad_ninguna_interaccion_es_suficiente():
    res = await _consultar(None)
    assert not any(p.suficiente for p in res.puntos)
    assert res.fidelidad is None


def test_respuesta_sin_fidelidad_es_nula():
    assert _fidelidad_response(None) is None


def test_respuesta_suficiente_pero_no_necesaria():
    r = _fidelidad_response(_suficiente_no_necesaria())
    assert r.verificada is True
    assert r.criterio == CRITERIO_SUFICIENCIA
    assert r.confianza == 0.9
    assert r.n_comparaciones == 20
    assert r.conceptos_suficientes == ["Árboles"]
    assert r.es_necesaria is False
    assert r.contrafactual is None


def test_respuesta_no_verificada_no_nombra_nada():
    f = FidelidadExplicacion(
        suficiencia=_prueba(CRITERIO_SUFICIENCIA, 9),
        exhaustividad=_prueba(CRITERIO_EXHAUSTIVIDAD, 4, k=3),
        motivo=MOTIVO_NO_SUPERA_AZAR,
    )
    r = _fidelidad_response(f)
    assert r.verificada is False
    assert r.conceptos_suficientes == []
    assert r.contrafactual is None


def _necesaria(cf):
    return FidelidadExplicacion(
        suficiencia=_prueba(CRITERIO_SUFICIENCIA, 18),
        exhaustividad=_prueba(CRITERIO_EXHAUSTIVIDAD, 19, k=3),
        indices_suficientes=[1],
        conceptos_suficientes=["Árboles"],
        motivo=MOTIVO_FIEL,
        contrafactual=cf,
    )


def test_respuesta_publica_contrafactual_relevante():
    cf = Contrafactual(
        concepto="Árboles",
        acierto_original=False,
        probabilidad_original=0.42,
        probabilidad_contrafactual=0.71,
    )
    r = _fidelidad_response(_necesaria(cf))
    assert r.es_necesaria is True
    assert r.contrafactual.concepto == "Árboles"
    assert r.contrafactual.probabilidad_contrafactual == 0.71


def test_respuesta_omite_contrafactual_irrelevante():
    cf = Contrafactual(probabilidad_original=0.42, probabilidad_contrafactual=0.424)
    assert _fidelidad_response(_necesaria(cf)).contrafactual is None
