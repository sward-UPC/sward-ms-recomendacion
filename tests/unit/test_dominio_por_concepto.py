"""La lista de recomendaciones se ordena por el dominio que estima el SAKT, y el
texto nunca le atribuye al modelo una cifra que no calculó."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from src.application.use_cases.generar_recomendacion import (
    GenerarRecomendacionCommand,
    GenerarRecomendacionUseCase,
)
from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion
from src.infrastructure.adapters.out_.modelo_kt_mock_adapter import ModeloKtMockAdapter
from tests.unit.test_generar_recomendacion import CANDIDATOS
from tests.unit.test_sakt_formato_entrada import (
    CONCEPTOS,
    RESPUESTAS,
    _crear_adaptador,
    _secuencia,
)

# grafos: 2 de 2 aciertos (promedio 100 %); listas: 0 de 1 (0 %).
HISTORIA = SecuenciaInteraccion(
    estudiante_id=uuid4(),
    curso_id=uuid4(),
    concepto_ids=["grafos", "grafos", "listas"],
    respuestas_correctas=[True, True, False],
)


class _ModeloQueEstima(ModeloKtMockAdapter):
    """Mock que además estima el dominio por concepto, como el SAKT real."""

    def __init__(self, estimaciones: dict[str, float]):
        super().__init__()
        self._estimaciones = estimaciones

    def predecir_dominio_por_concepto(self, secuencia, conceptos):
        return {c: v for c, v in self._estimaciones.items() if c in conceptos}


def _caso(modelo):
    trazabilidad = AsyncMock()
    trazabilidad.obtener_preferencias.return_value = None
    trazabilidad.obtener_secuencia.return_value = HISTORIA
    cursos = AsyncMock()
    cursos.obtener_recursos_candidatos.return_value = CANDIDATOS
    repo = AsyncMock()
    repo.save.side_effect = lambda r: r
    uc = GenerarRecomendacionUseCase(
        trazabilidad, cursos, AsyncMock(), repo, MagicMock(), modelo
    )
    return uc, cursos


async def _ejecutar(uc):
    return await uc.execute(
        GenerarRecomendacionCommand(estudiante_id=uuid4(), curso_id=uuid4())
    )


@pytest.mark.asyncio
async def test_ordena_por_el_dominio_que_estima_el_modelo():
    # El modelo ve débil a grafos aunque su promedio sea 100 %.
    uc, cursos = _caso(_ModeloQueEstima({"grafos": 0.2, "listas": 0.9}))
    rec = await _ejecutar(uc)

    primera = cursos.obtener_recursos_candidatos.call_args_list[0]
    assert primera.kwargs.get("seccion") == "grafos"
    motivos = [i.motivo for i in rec.items if "grafos" in i.motivo]
    assert motivos and all(
        "el modelo SAKT estima tu dominio en 20%" in m for m in motivos
    )


@pytest.mark.asyncio
async def test_sin_estimacion_del_modelo_dice_que_es_un_promedio():
    uc, cursos = _caso(ModeloKtMockAdapter())
    rec = await _ejecutar(uc)

    primera = cursos.obtener_recursos_candidatos.call_args_list[0]
    assert primera.kwargs.get("seccion") == "listas"  # 0 % de aciertos
    assert rec.items
    assert all("SAKT" not in i.motivo for i in rec.items)
    assert any("llevas 0% de aciertos en este tema" in i.motivo for i in rec.items)


@pytest.mark.asyncio
async def test_estimacion_parcial_no_se_mezcla_con_promedios():
    # El modelo solo estima uno de los dos temas: se usa el promedio para ambos.
    uc, cursos = _caso(_ModeloQueEstima({"grafos": 0.2}))
    rec = await _ejecutar(uc)

    primera = cursos.obtener_recursos_candidatos.call_args_list[0]
    assert primera.kwargs.get("seccion") == "listas"
    assert all("SAKT" not in i.motivo for i in rec.items)


# ---------------------------------------------------------------------------
# Adaptador real (SAKT diminuto en memoria; se salta sin torch/pyKT)
# ---------------------------------------------------------------------------


def _referencia(adaptador, concepto: str) -> float:
    """El modelo sobre la historia completa, sin relleno, consultando el concepto."""
    import torch

    idx = [int(c[1:]) for c in CONCEPTOS]
    with torch.no_grad():
        out = adaptador._model(
            torch.LongTensor([idx]),
            torch.LongTensor([RESPUESTAS]),
            torch.LongTensor([idx[1:] + [int(concepto[1:])]]),
        )
    return float(out[0, -1])


def test_adaptador_estima_cada_concepto_con_toda_la_historia(monkeypatch):
    ad = _crear_adaptador(monkeypatch, {"val_auc": 0.7, "epoch": 1})
    estimado = ad.predecir_dominio_por_concepto(_secuencia(), ["c0", "c3", "zz"])

    assert set(estimado) == {"c0", "c3"}  # el desconocido se omite
    for c in ("c0", "c3"):
        assert estimado[c] == pytest.approx(_referencia(ad, c), abs=1e-4)


def test_adaptador_simulado_no_estima(monkeypatch):
    ad = _crear_adaptador(monkeypatch, {"val_auc": 0.7, "epoch": 1})
    ad._mock = True
    assert ad.predecir_dominio_por_concepto(_secuencia(), ["c0"]) == {}
