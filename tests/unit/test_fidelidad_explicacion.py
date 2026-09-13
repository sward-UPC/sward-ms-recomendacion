"""Invariantes de las entidades de verificación de fidelidad.

Puros: no importan torch ni pyKT.
"""

import pytest

from src.domain.entities.fidelidad_explicacion import (
    CRITERIO_EXHAUSTIVIDAD,
    CRITERIO_SUFICIENCIA,
    MOTIVO_FIEL,
    MOTIVO_NO_SUPERA_AZAR,
    MOTIVO_POCAS_INTERACCIONES,
    Contrafactual,
    FidelidadExplicacion,
    PruebaFidelidad,
)


def _prueba(criterio, supera, n=20, k=1, atencion=0.0, azar=0.0):
    return PruebaFidelidad(
        criterio=criterio,
        k=k,
        perdida_atencion=atencion,
        perdida_azar_media=azar,
        n_aleatorios=n,
        supera_azar_en=supera,
    )


# ---------------------------------------------------------------------------
# PruebaFidelidad
# ---------------------------------------------------------------------------


def test_confianza_es_la_fraccion_de_sorteos_ganados():
    assert _prueba(CRITERIO_SUFICIENCIA, supera=17).confianza == pytest.approx(0.85)


def test_sin_sorteos_la_confianza_es_la_del_azar():
    assert _prueba(CRITERIO_SUFICIENCIA, supera=0, n=0).confianza == 0.5


def test_no_puede_superar_mas_sorteos_de_los_hechos():
    with pytest.raises(ValueError):
        _prueba(CRITERIO_SUFICIENCIA, supera=11, n=10)


def test_criterio_invalido():
    with pytest.raises(ValueError):
        _prueba("causalidad", supera=1)


def test_ventaja_de_suficiencia_es_perder_menos():
    # Conservar solo lo atendido pierde 0.05; conservar al azar, 0.14.
    p = _prueba(CRITERIO_SUFICIENCIA, supera=15, atencion=0.05, azar=0.14)
    assert p.ventaja == pytest.approx(0.09)


def test_ventaja_de_exhaustividad_es_perder_mas():
    # Borrar lo atendido pierde 0.12; borrar al azar, 0.05.
    p = _prueba(CRITERIO_EXHAUSTIVIDAD, supera=18, atencion=0.12, azar=0.05)
    assert p.ventaja == pytest.approx(0.07)


# ---------------------------------------------------------------------------
# FidelidadExplicacion
# ---------------------------------------------------------------------------


def test_no_verificada_nunca_es_fiel():
    # Ante la duda el sistema calla: sin verificación no se afirma un motivo.
    f = FidelidadExplicacion.no_verificada(MOTIVO_POCAS_INTERACCIONES, n_pasado=2)
    assert f.es_fiel is False
    assert f.confianza == 0.5
    assert f.contrafactual is None
    assert f.conceptos_suficientes == []


def test_suficiente_pero_no_necesaria_es_fiel_con_criterio_suficiencia():
    # El resultado medido de la tesis: basta, pero no causa.
    f = FidelidadExplicacion(
        criterio=CRITERIO_SUFICIENCIA,
        suficiencia=_prueba(CRITERIO_SUFICIENCIA, supera=18),
        exhaustividad=_prueba(CRITERIO_EXHAUSTIVIDAD, supera=9, k=3),
        indices_suficientes=[4],
        conceptos_suficientes=["Árboles Binarios y AVL"],
        motivo=MOTIVO_FIEL,
    )
    assert f.es_suficiente and not f.es_necesaria
    assert f.es_fiel
    assert f.confianza == pytest.approx(0.9)


def test_el_mismo_resultado_no_es_fiel_con_criterio_exhaustividad():
    f = FidelidadExplicacion(
        criterio=CRITERIO_EXHAUSTIVIDAD,
        suficiencia=_prueba(CRITERIO_SUFICIENCIA, supera=18),
        exhaustividad=_prueba(CRITERIO_EXHAUSTIVIDAD, supera=9, k=3),
        motivo=MOTIVO_NO_SUPERA_AZAR,
    )
    assert not f.es_fiel
    assert f.confianza == pytest.approx(0.45)


def test_no_puede_nombrar_interacciones_suficientes_sin_verificarlo():
    with pytest.raises(ValueError):
        FidelidadExplicacion(
            suficiencia=_prueba(CRITERIO_SUFICIENCIA, supera=10),
            conceptos_suficientes=["Árboles Binarios y AVL"],
        )


def test_contrafactual_exige_necesidad_verificada():
    # Suficiente no alcanza: «si hubieras acertado…» afirma necesidad.
    with pytest.raises(ValueError):
        FidelidadExplicacion(
            suficiencia=_prueba(CRITERIO_SUFICIENCIA, supera=18),
            exhaustividad=_prueba(CRITERIO_EXHAUSTIVIDAD, supera=9, k=3),
            motivo=MOTIVO_FIEL,
            contrafactual=Contrafactual(),
        )


def test_contrafactual_permitido_con_necesidad_verificada():
    f = FidelidadExplicacion(
        suficiencia=_prueba(CRITERIO_SUFICIENCIA, supera=18),
        exhaustividad=_prueba(CRITERIO_EXHAUSTIVIDAD, supera=19, k=3),
        motivo=MOTIVO_FIEL,
        contrafactual=Contrafactual(probabilidad_contrafactual=0.7),
    )
    assert f.es_necesaria and f.contrafactual is not None


def test_motivo_no_puede_contradecir_las_pruebas():
    with pytest.raises(ValueError):
        FidelidadExplicacion(
            suficiencia=_prueba(CRITERIO_SUFICIENCIA, supera=5),
            motivo=MOTIVO_FIEL,
        )


def test_prueba_con_criterio_cruzado_es_invalida():
    with pytest.raises(ValueError):
        FidelidadExplicacion(suficiencia=_prueba(CRITERIO_EXHAUSTIVIDAD, supera=1))


@pytest.mark.parametrize("umbral", [-0.01, 1.01])
def test_umbral_fuera_de_rango(umbral):
    with pytest.raises(ValueError):
        FidelidadExplicacion(umbral=umbral)


# ---------------------------------------------------------------------------
# Contrafactual
# ---------------------------------------------------------------------------


def test_contrafactual_delta_y_relevancia():
    cf = Contrafactual(probabilidad_original=0.42, probabilidad_contrafactual=0.71)
    assert cf.delta == pytest.approx(0.29)
    assert cf.es_relevante


def test_contrafactual_irrelevante_bajo_un_punto():
    # Enunciar un cambio de 0.42 a 0.424 resta credibilidad: no se muestra.
    cf = Contrafactual(probabilidad_original=0.42, probabilidad_contrafactual=0.424)
    assert not cf.es_relevante


def test_contrafactual_probabilidad_invalida():
    with pytest.raises(ValueError):
        Contrafactual(probabilidad_original=1.2)
