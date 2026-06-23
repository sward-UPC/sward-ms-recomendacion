from dataclasses import dataclass
from uuid import UUID

from src.application.ports.out_.modelo_kt_port import ModeloKTPort
from src.application.ports.out_.trazabilidad_client_port import TrazabilidadClientPort


@dataclass
class PuntoAtencion:
    """Una interacción pasada con el peso de atención que SAKT le asignó."""

    concepto: str
    acierto: bool
    peso: float


@dataclass
class AtencionResultado:
    probabilidad_dominio: float
    puntos: list[PuntoAtencion]


class ConsultarAtencionUseCase:
    """Heatmap de atención: a qué interacciones pasadas atendió SAKT al predecir."""

    def __init__(self, trazabilidad: TrazabilidadClientPort, modelo: ModeloKTPort):
        self._trazabilidad = trazabilidad
        self._modelo = modelo

    async def execute(self, estudiante_id: UUID, curso_id: UUID) -> AtencionResultado:
        secuencia = await self._trazabilidad.obtener_secuencia(estudiante_id, curso_id)
        pred = self._modelo.predecir_dominio(secuencia)
        # pesos_atencion alinea con las interacciones pasadas; zip recorta al menor.
        puntos = [
            PuntoAtencion(concepto=str(c), acierto=bool(r), peso=round(float(w), 4))
            for c, r, w in zip(
                secuencia.concepto_ids,
                secuencia.respuestas_correctas,
                pred.pesos_atencion,
            )
        ]
        return AtencionResultado(
            probabilidad_dominio=pred.probabilidad_dominio, puntos=puntos
        )
