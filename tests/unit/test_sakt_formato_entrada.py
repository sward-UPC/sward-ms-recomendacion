"""Formato de entrada del SAKT y verificación de fidelidad en línea.

El proyecto tiene dos pipelines de entrenamiento con formatos de entrada
distintos (relleno por la izquierda en training/train_sakt.py, por la derecha con
máscara en sward-model-training/train.py). Servir un checkpoint con el formato
equivocado cambia sus predicciones y hace que la atención caiga sobre el relleno.

Las pruebas de `detectar_formato_entrada` son puras. Las que ejercen el
adaptador construyen un SAKT diminuto en memoria y se saltan si torch o pyKT no
están instalados, igual que el resto de pruebas unitarias del servicio.
"""

import os
import sys
import tempfile
import types
import uuid

import pytest

from src.domain.entities.fidelidad_explicacion import (
    CRITERIO_EXHAUSTIVIDAD,
    CRITERIO_SUFICIENCIA,
    MOTIVO_DESACTIVADA,
    MOTIVO_POCAS_INTERACCIONES,
)
from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion
from src.infrastructure.adapters.out_ import sakt_pykt_adapter as ad_mod
from src.infrastructure.adapters.out_.sakt_pykt_adapter import (
    FORMATO_DERECHA,
    FORMATO_IZQUIERDA,
    detectar_formato_entrada,
)
from src.infrastructure.config.settings import settings

# ---------------------------------------------------------------------------
# detectar_formato_entrada (puro)
# ---------------------------------------------------------------------------


def test_formato_declarado_manda_sobre_la_huella():
    ck = {"formato_entrada": FORMATO_IZQUIERDA, "val_auc": 0.8, "epoch": 3}
    assert detectar_formato_entrada(ck) == FORMATO_IZQUIERDA


def test_huella_de_train_py_es_derecha():
    assert detectar_formato_entrada({"val_auc": 0.8, "epoch": 3}) == FORMATO_DERECHA


def test_huella_de_train_sakt_es_izquierda():
    ck = {"trained_at": "2026-06-20T00:00:00Z", "test_auc": 0.7}
    assert detectar_formato_entrada(ck) == FORMATO_IZQUIERDA


def test_sin_senal_conserva_el_formato_historico():
    assert detectar_formato_entrada({}) == FORMATO_IZQUIERDA


def test_huella_ambigua_conserva_el_formato_historico():
    ck = {"val_auc": 0.8, "trained_at": "2026-06-20T00:00:00Z"}
    assert detectar_formato_entrada(ck) == FORMATO_IZQUIERDA


def test_forzado_manda_sobre_todo():
    ck = {"formato_entrada": FORMATO_IZQUIERDA}
    assert detectar_formato_entrada(ck, forzado=FORMATO_DERECHA) == FORMATO_DERECHA


def test_forzado_invalido_falla_en_voz_alta():
    with pytest.raises(ValueError):
        detectar_formato_entrada({}, forzado="relleno_arriba")


# ---------------------------------------------------------------------------
# Adaptador con un SAKT diminuto
# ---------------------------------------------------------------------------

N_SKILLS = 6
SEQ_LEN = 20
CONCEPTOS = ["c1", "c3", "c0", "c2", "c5", "c4", "c1", "c2"]
RESPUESTAS = [1, 0, 1, 1, 0, 1, 0, 1]


def _crear_adaptador(monkeypatch, claves_extra: dict):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pykt")
    ad_mod._mock_turtle()
    from pykt.models.sakt import SAKT

    torch.manual_seed(0)
    modelo = SAKT(
        num_c=N_SKILLS,
        seq_len=SEQ_LEN,
        emb_size=16,
        num_attn_heads=2,
        dropout=0.0,
        num_en=1,
        emb_type="qid",
    )
    checkpoint = {
        "model_state_dict": modelo.state_dict(),
        "n_skills": N_SKILLS,
        "seq_len": SEQ_LEN,
        "emb_size": 16,
        "n_heads": 2,
        "dropout": 0.0,
        "n_layers": 1,
        "concept_index": {f"c{i}": i for i in range(N_SKILLS)},
        **claves_extra,
    }
    # _cargar_modelo no descarga si el archivo ya está en el temporal.
    version = f"test_{uuid.uuid4().hex}"
    ruta = os.path.join(tempfile.gettempdir(), f"sakt_{version}.pth")
    torch.save(checkpoint, ruta)
    boto3_falso = types.ModuleType("boto3")
    boto3_falso.client = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "boto3", boto3_falso)
    try:
        adaptador = ad_mod.SaktPyktAdapter(version=version)
    finally:
        os.remove(ruta)
    assert not adaptador._mock, "la carga real cayó a mock"
    return adaptador


def _secuencia(conceptos=CONCEPTOS, respuestas=RESPUESTAS):
    return SecuenciaInteraccion(
        estudiante_id=uuid.UUID(int=1),
        curso_id=uuid.UUID(int=2),
        concepto_ids=list(conceptos),
        respuestas_correctas=[bool(r) for r in respuestas],
    )


def _referencia_sin_relleno(adaptador, conceptos=CONCEPTOS, respuestas=RESPUESTAS):
    """Predicción del modelo sobre la secuencia tal cual, sin relleno alguno.

    Es la verdad de referencia: con relleno a la derecha y máscara causal, el
    último paso real no puede ver el relleno, así que debe coincidir con esto.
    """
    import torch

    idx = [int(c[1:]) for c in conceptos]
    with torch.no_grad():
        out = adaptador._model(
            torch.LongTensor([idx[:-1]]),
            torch.LongTensor([respuestas[:-1]]),
            torch.LongTensor([idx[1:]]),
        )
    return float(out[0, -1])


def test_checkpoint_de_train_py_se_sirve_por_la_derecha(monkeypatch):
    ad = _crear_adaptador(monkeypatch, {"val_auc": 0.7, "epoch": 1})
    assert ad._formato == FORMATO_DERECHA


def test_formato_derecha_equivale_a_la_secuencia_sin_relleno(monkeypatch):
    ad = _crear_adaptador(monkeypatch, {"val_auc": 0.7, "epoch": 1})
    pred = ad.predecir_dominio(_secuencia())
    assert pred.probabilidad_dominio == pytest.approx(
        _referencia_sin_relleno(ad), abs=1e-4
    )


def test_servirlo_por_la_izquierda_cambia_la_prediccion(monkeypatch):
    # Documenta el defecto: el mismo checkpoint servido con el otro formato no
    # predice lo mismo que el modelo sobre la secuencia real.
    monkeypatch.setattr(settings, "sakt_formato_entrada", FORMATO_IZQUIERDA)
    ad = _crear_adaptador(monkeypatch, {"val_auc": 0.7, "epoch": 1})
    pred = ad.predecir_dominio(_secuencia())
    assert abs(pred.probabilidad_dominio - _referencia_sin_relleno(ad)) > 1e-3


def test_formato_derecha_no_pone_atencion_sobre_el_relleno(monkeypatch):
    ad = _crear_adaptador(monkeypatch, {"val_auc": 0.7, "epoch": 1})
    idx = [int(c[1:]) for c in CONCEPTOS]
    q, r, qry = idx[:-1], RESPUESTAS[:-1], idx[1:]
    n = len(qry)
    ad._inferir([q], [r], qry)
    fila = ad._model.blocks[-1]._last_attn[0, n - 1]
    assert float(fila[n:].sum()) == pytest.approx(0.0, abs=1e-6)
    pesos = ad._extraer_atencion(n)
    assert len(pesos) == n
    assert sum(pesos) == pytest.approx(1.0, abs=1e-3)


def _config_verificacion(monkeypatch, criterio=CRITERIO_SUFICIENCIA, umbral=0.8):
    monkeypatch.setattr(settings, "xai_verificacion_activa", True)
    monkeypatch.setattr(settings, "xai_criterio_fidelidad", criterio)
    monkeypatch.setattr(settings, "xai_min_interacciones", 5)
    monkeypatch.setattr(settings, "xai_k_suficiencia", 1)
    monkeypatch.setattr(settings, "xai_k_exhaustividad", 2)
    monkeypatch.setattr(settings, "xai_n_aleatorios", 10)
    monkeypatch.setattr(settings, "xai_umbral_confianza", umbral)


def test_verificacion_calcula_ambas_pruebas(monkeypatch):
    _config_verificacion(monkeypatch)
    ad = _crear_adaptador(monkeypatch, {"val_auc": 0.7, "epoch": 1})
    f = ad.predecir_dominio(_secuencia()).fidelidad

    assert f.suficiencia.criterio == CRITERIO_SUFICIENCIA and f.suficiencia.k == 1
    assert f.exhaustividad.criterio == CRITERIO_EXHAUSTIVIDAD and f.exhaustividad.k == 2
    assert f.suficiencia.n_aleatorios == f.exhaustividad.n_aleatorios == 10
    assert f.latencia_ms > 0
    assert f.es_fiel == f.es_suficiente


def test_criterio_exhaustividad_decide_con_esa_prueba(monkeypatch):
    _config_verificacion(monkeypatch, criterio=CRITERIO_EXHAUSTIVIDAD)
    ad = _crear_adaptador(monkeypatch, {"val_auc": 0.7, "epoch": 1})
    f = ad.predecir_dominio(_secuencia()).fidelidad
    assert f.es_fiel == f.es_necesaria
    assert f.confianza == f.exhaustividad.confianza


def test_la_confianza_de_la_prediccion_es_la_medida(monkeypatch):
    # Reemplaza la constante 0.85 que no salía de ningún cálculo.
    _config_verificacion(monkeypatch)
    ad = _crear_adaptador(monkeypatch, {"val_auc": 0.7, "epoch": 1})
    pred = ad.predecir_dominio(_secuencia())
    assert pred.confianza == pred.fidelidad.confianza


def test_con_suficiencia_verificada_nombra_los_conceptos(monkeypatch):
    # Umbral 0 fuerza que la prueba pase para ejercer el camino positivo.
    _config_verificacion(monkeypatch, umbral=0.0)
    ad = _crear_adaptador(monkeypatch, {"val_auc": 0.7, "epoch": 1})
    f = ad.predecir_dominio(_secuencia()).fidelidad

    assert f.es_suficiente
    assert len(f.indices_suficientes) == 1
    # Nombre del concepto, no el índice interno del modelo.
    assert f.conceptos_suficientes[0] in CONCEPTOS
    assert f.conceptos_suficientes[0] == CONCEPTOS[f.indices_suficientes[0]]


def _modelo_indiferente(monkeypatch, ad):
    """Hace que toda perturbación dé la misma probabilidad.

    Así la atención nunca le gana estrictamente al azar y ambas pruebas fallan
    de forma determinista.
    """
    monkeypatch.setattr(ad, "_inferir", lambda qs, rs, qry, borrados=None: [0.7] * len(qs))


def test_sin_suficiencia_verificada_no_nombra_conceptos(monkeypatch):
    _config_verificacion(monkeypatch)
    ad = _crear_adaptador(monkeypatch, {"val_auc": 0.7, "epoch": 1})
    _modelo_indiferente(monkeypatch, ad)
    f = ad.predecir_dominio(_secuencia()).fidelidad
    assert not f.es_suficiente and not f.es_fiel
    assert f.suficiencia.supera_azar_en == 0
    assert f.indices_suficientes == [] and f.conceptos_suficientes == []


def test_contrafactual_solo_con_necesidad_verificada(monkeypatch):
    _config_verificacion(monkeypatch, umbral=0.0)
    ad = _crear_adaptador(monkeypatch, {"val_auc": 0.7, "epoch": 1})
    f = ad.predecir_dominio(_secuencia()).fidelidad
    assert f.es_necesaria  # umbral 0: pasa
    assert f.contrafactual is not None
    assert f.contrafactual.concepto in CONCEPTOS


def test_sin_necesidad_verificada_no_hay_contrafactual(monkeypatch):
    _config_verificacion(monkeypatch)
    ad = _crear_adaptador(monkeypatch, {"val_auc": 0.7, "epoch": 1})
    _modelo_indiferente(monkeypatch, ad)
    f = ad.predecir_dominio(_secuencia()).fidelidad
    assert not f.es_necesaria
    assert f.contrafactual is None


def test_verificacion_es_determinista_por_estudiante(monkeypatch):
    # Recargar la pantalla no puede hacer que el sistema cambie de opinión.
    _config_verificacion(monkeypatch)
    ad = _crear_adaptador(monkeypatch, {"val_auc": 0.7, "epoch": 1})
    a = ad.predecir_dominio(_secuencia()).fidelidad
    b = ad.predecir_dominio(_secuencia()).fidelidad
    assert (a.suficiencia.supera_azar_en, a.exhaustividad.supera_azar_en) == (
        b.suficiencia.supera_azar_en,
        b.exhaustividad.supera_azar_en,
    )


def test_criterio_invalido_hace_fallar_el_arranque():
    from pydantic import ValidationError

    from src.infrastructure.config.settings import Settings

    with pytest.raises(ValidationError):
        Settings(xai_criterio_fidelidad="causalidad")


def test_pocas_interacciones_se_abstiene(monkeypatch):
    _config_verificacion(monkeypatch)
    ad = _crear_adaptador(monkeypatch, {"val_auc": 0.7, "epoch": 1})
    pred = ad.predecir_dominio(_secuencia(CONCEPTOS[:4], RESPUESTAS[:4]))
    assert pred.fidelidad.motivo == MOTIVO_POCAS_INTERACCIONES
    assert pred.fidelidad.es_fiel is False


def test_verificacion_desactivada_conserva_el_comportamiento_anterior(monkeypatch):
    # Es la condición de control para comparar ambas versiones con usuarios.
    monkeypatch.setattr(settings, "xai_verificacion_activa", False)
    ad = _crear_adaptador(monkeypatch, {"val_auc": 0.7, "epoch": 1})
    pred = ad.predecir_dominio(_secuencia())
    assert pred.fidelidad.motivo == MOTIVO_DESACTIVADA
    assert pred.confianza == 0.85
