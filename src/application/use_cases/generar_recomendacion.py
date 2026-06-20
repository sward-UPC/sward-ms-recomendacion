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

        # Concepto más débil: menor promedio de aciertos (con al menos 1 interacción).
        concepto_debil = self._concepto_mas_debil(secuencia)

        candidatos = await self._cursos.obtener_recursos_candidatos(
            cmd.curso_id, limit=20, seccion=concepto_debil
        )
        # Fallback: si filtrar por sección no devuelve nada, usar el curso completo.
        if not candidatos and concepto_debil is not None:
            candidatos = await self._cursos.obtener_recursos_candidatos(
                cmd.curso_id, limit=20
            )

        items = self._rankear(
            candidatos, prediccion.probabilidad_dominio, concepto_debil
        )

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

    @staticmethod
    def _concepto_mas_debil(secuencia) -> str | None:
        """Concepto con menor promedio de aciertos (>= 1 interacción)."""
        conceptos = getattr(secuencia, "concepto_ids", None) or []
        aciertos = getattr(secuencia, "respuestas_correctas", None) or []
        if not conceptos:
            return None

        suma: dict[str, float] = {}
        cuenta: dict[str, int] = {}
        for concepto, correcta in zip(conceptos, aciertos):
            suma[concepto] = suma.get(concepto, 0.0) + (1.0 if correcta else 0.0)
            cuenta[concepto] = cuenta.get(concepto, 0) + 1

        if not cuenta:
            return None

        promedios = {c: suma[c] / cuenta[c] for c in cuenta}
        return min(promedios, key=lambda c: promedios[c])

    def _rankear(
        self, candidatos: list[dict], dominio: float, concepto_debil: str | None = None
    ) -> list[ItemRecomendado]:
        nivel_obj = (
            "basico" if dominio < 0.4 else "intermedio" if dominio < 0.7 else "avanzado"
        )
        orden = {"basico": 0, "intermedio": 1, "avanzado": 2}
        obj_idx = orden[nivel_obj]

        tipos_estudio = {"lectura", "video", "presentacion"}
        tipos_practica = {"ejercicio", "quiz"}
        tipos_foro = {"foro", "forum"}

        def es_foro(r: dict) -> bool:
            tipo = str(r.get("tipo", "")).lower()
            titulo = str(r.get("titulo", "")).lower()
            return tipo in tipos_foro or "foro" in titulo or "announcements" in titulo

        def es_estudio(r: dict) -> bool:
            return str(r.get("tipo", "")).lower() in tipos_estudio

        def alineamiento(r: dict) -> float:
            dist = abs(orden.get(r.get("nivel_dificultad", "intermedio"), 1) - obj_idx)
            return 1.0 / (1 + dist)

        def score(r: dict) -> float:
            # Estudio primero (boost), luego práctica; más el alineamiento de nivel.
            base = 0.6 if es_estudio(r) else 0.3 if not es_foro(r) else 0.0
            return round(min(1.0, base + 0.4 * alineamiento(r)), 4)

        # Excluir foros y rankear: estudio antes que práctica, luego por score.
        filtrados = [r for r in candidatos if not es_foro(r)]

        def clave_orden(r: dict) -> tuple:
            prioridad = (
                0
                if str(r.get("tipo", "")).lower() in tipos_estudio
                else 1
                if str(r.get("tipo", "")).lower() in tipos_practica
                else 2
            )
            return (prioridad, -score(r))

        ranked = sorted(filtrados, key=clave_orden)[: settings.max_recomendaciones]

        concepto_txt = concepto_debil or "los conceptos del curso"
        items: list[ItemRecomendado] = []
        for i, r in enumerate(ranked):
            estudio = es_estudio(r)
            motivo = (
                f"Refuerza {concepto_txt} (el modelo SAKT estima tu dominio en "
                f"{dominio:.0%}). "
                f"{'Material de estudio' if estudio else 'Práctica'} para afianzar."
            )
            items.append(
                ItemRecomendado(
                    recurso_id=str(r.get("id", "")),
                    titulo=r.get("titulo", ""),
                    tipo=r.get("tipo", ""),
                    score=score(r),
                    orden=i + 1,
                    motivo=motivo,
                    url=r.get("url", ""),
                )
            )
        return items
