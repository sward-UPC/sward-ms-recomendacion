"""Invariantes de las entidades de verificación de fidelidad.

Puros: no importan torch ni pyKT.
"""

import pytest

from src.domain.entities.fidelidad_explicacion import (
    MOTIVO_POCAS_INTERACCIONES,
    Contrafactual,
    FidelidadExplicacion,
)


def test_no_verificada_nunca_es_fiel():
    # Ante la duda el sistema calla: sin verificación no se afirma un motivo.
    f = FidelidadExplicacion.no_verificada(MOTIVO_POCAS_INTERACCIONES, n_pasado=2)
    assert f.es_fiel is False
    assert f.confianza == 0.5
    assert f.motivo == MOTIVO_POCAS_INTERACCIONES
    assert f.contrafactual is None


def test_fiel_sin_sorteos_es_invalido():
    with pytest.raises(ValueError):
        FidelidadExplicacion(es_fiel=True, n_aleatorios=0, confianza=1.0)


def test_no_puede_superar_mas_sorteos_de_los_hechos():
    with pytest.raises(ValueError):
        FidelidadExplicacion(n_aleatorios=10, supera_azar_en=11)


@pytest.mark.parametrize("confianza", [-0.01, 1.01])
def test_confianza_fuera_de_rango(confianza):
    with pytest.raises(ValueError):
        FidelidadExplicacion(confianza=confianza)


def test_delta_de_fidelidad():
    f = FidelidadExplicacion(
        comprehensiveness_atencion=0.12,
        comprehensiveness_azar_media=0.05,
        n_aleatorios=20,
        supera_azar_en=18,
        confianza=0.9,
        es_fiel=True,
    )
    assert f.delta == pytest.approx(0.07)


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
