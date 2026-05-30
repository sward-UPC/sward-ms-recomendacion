from dataclasses import dataclass
from uuid import UUID

from src.domain.entities.recomendacion import ItemRecomendado, Recomendacion
from src.domain.events.recomendacion_generada_event import RecomendacionGeneradaEvent
from src.domain.ports.out_.cursos_client_port import CursosClientPort
from src.domain.ports.out_.event_publisher_port import EventPublisherPort
from src.domain.ports.out_.recomendacion_repository_port import (
    RecomendacionRepositoryPort,
)
from src.domain.ports.out_.trazabilidad_client_port import TrazabilidadClientPort
from src.domain.ports.out_.xai_client_port import XaiClientPort
from src.domain.services.modelo_sakt import ModeloSAKT
from src.infrastructure.config.settings import settings


@dataclass
class GenerarRecomendacionCommand:
    estudiante_id: UUID
    curso_id: UUID


class GenerarRecomendacionUseCase:
    def __init__(
        self,
        trazabilidad: TrazabilidadClientPort,
        cursos: CursosClientPort,
        xai: XaiClientPort,
        repo: RecomendacionRepositoryPort,
        event_publisher: EventPublisherPort,
        modelo: ModeloSAKT,
    ):
        self._trazabilidad = trazabilidad
        self._cursos = cursos
        self._xai = xai
        self._repo = repo
        self._event_publisher = event_publisher
        self._modelo = modelo

    async def execute(self, cmd: GenerarRecomendacionCommand) -> Recomendacion:
        secuencia = await self._trazabilidad.obtener_secuencia(
            cmd.estudiante_id, cmd.curso_id
        )
        prediccion = self._modelo.predecir_dominio(secuencia)
        candidatos = await self._cursos.obtener_recursos_candidatos(
            cmd.curso_id, limit=20
        )
        items = self._rankear(candidatos, prediccion.probabilidad_dominio)

        # Garantizar mínimo RF-003-05
        while len(items) < settings.min_recomendaciones and items:
            items = items + items[: settings.min_recomendaciones - len(items)]

        rec = Recomendacion(
            estudiante_id=cmd.estudiante_id, curso_id=cmd.curso_id, items=items
        )
        guardada = await self._repo.save(rec)

        try:
            await self._xai.generar_explicacion(guardada.id, prediccion.pesos_atencion)
        except Exception:
            pass  # XAI falla de forma no bloqueante

        self._event_publisher.publish(
            RecomendacionGeneradaEvent(
                recomendacion_id=guardada.id,
                estudiante_id=cmd.estudiante_id,
                curso_id=cmd.curso_id,
            )
        )
        return guardada

    def _rankear(self, candidatos: list[dict], dominio: float) -> list[ItemRecomendado]:
        nivel_obj = (
            "basico" if dominio < 0.4 else "intermedio" if dominio < 0.7 else "avanzado"
        )
        orden = {"basico": 0, "intermedio": 1, "avanzado": 2}
        obj_idx = orden[nivel_obj]

        def score(r: dict) -> float:
            dist = abs(orden.get(r.get("nivel_dificultad", "intermedio"), 1) - obj_idx)
            return 1.0 / (1 + dist)

        ranked = sorted(candidatos, key=score, reverse=True)
        return [
            ItemRecomendado(
                recurso_id=str(r.get("id", "")),
                titulo=r.get("titulo", ""),
                tipo=r.get("tipo", ""),
                score=score(r),
                orden=i + 1,
                motivo=f"Recurso {r.get('nivel_dificultad', 'intermedio')} alineado con tu dominio ({dominio:.0%})",
                url=r.get("url", ""),
            )
            for i, r in enumerate(ranked)
        ]
