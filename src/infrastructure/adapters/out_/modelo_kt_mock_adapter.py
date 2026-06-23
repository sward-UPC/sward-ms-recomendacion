from src.domain.entities.prediccion_kt import PrediccionKT
from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion
from src.domain.ports.out_.modelo_kt_port import ModeloKTPort


class ModeloKtMockAdapter(ModeloKTPort):
    """Adaptador mock del modelo KT para development/tests.

    No carga torch/pyKT ni descarga de S3: estima el dominio como el promedio de
    aciertos y reparte la atención uniformemente. Es el camino que usaba
    ``ModeloSAKT(mock=True)``, sin cambios de comportamiento.
    """

    def __init__(self, version: str = "v1.0"):
        self.version = version

    def predecir_dominio(self, secuencia: SecuenciaInteraccion) -> PrediccionKT:
        if not secuencia.respuestas_correctas:
            prob = 0.5
        else:
            prob = sum(secuencia.respuestas_correctas) / len(
                secuencia.respuestas_correctas
            )
        n = len(secuencia.concepto_ids)
        pesos = [1.0 / n if n > 0 else 0.0] * n
        return PrediccionKT(
            estudiante_id=secuencia.estudiante_id,
            curso_id=secuencia.curso_id,
            probabilidad_dominio=round(prob, 4),
            confianza=0.7 if n >= 5 else 0.4,
            pesos_atencion=pesos,
        )

    def leer_info(self) -> dict:
        # La metadata describe el artefacto REAL entrenado en S3 (la consume el panel
        # admin), independientemente de que este entorno corra el modelo simulado.
        from src.infrastructure.adapters.out_.sakt_pykt_adapter import leer_info_modelo

        return leer_info_modelo()
