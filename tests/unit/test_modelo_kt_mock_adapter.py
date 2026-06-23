from uuid import uuid4

from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion
from src.infrastructure.adapters.out_.modelo_kt_mock_adapter import ModeloKtMockAdapter


def test_sin_historial_retorna_05():
    m = ModeloKtMockAdapter()
    s = SecuenciaInteraccion(estudiante_id=uuid4(), curso_id=uuid4())
    p = m.predecir_dominio(s)
    assert p.probabilidad_dominio == 0.5


def test_todo_correcto_retorna_10():
    m = ModeloKtMockAdapter()
    s = SecuenciaInteraccion(
        estudiante_id=uuid4(),
        curso_id=uuid4(),
        concepto_ids=["c1", "c2"],
        respuestas_correctas=[True, True],
    )
    p = m.predecir_dominio(s)
    assert p.probabilidad_dominio == 1.0


def test_todo_incorrecto_retorna_00():
    m = ModeloKtMockAdapter()
    s = SecuenciaInteraccion(
        estudiante_id=uuid4(),
        curso_id=uuid4(),
        concepto_ids=["c1"],
        respuestas_correctas=[False],
    )
    p = m.predecir_dominio(s)
    assert p.probabilidad_dominio == 0.0


def test_pesos_atencion_uniformes():
    m = ModeloKtMockAdapter()
    s = SecuenciaInteraccion(
        estudiante_id=uuid4(),
        curso_id=uuid4(),
        concepto_ids=["c1", "c2", "c3", "c4"],
        respuestas_correctas=[True, False, True, False],
    )
    p = m.predecir_dominio(s)
    assert len(p.pesos_atencion) == 4
    assert all(abs(w - 0.25) < 1e-9 for w in p.pesos_atencion)
