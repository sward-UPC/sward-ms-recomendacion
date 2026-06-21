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

        # Top-K conceptos más débiles según el dominio estimado por sección. Antes
        # solo targeteábamos UNO → muy pocos items; ahora cubrimos varios conceptos
        # (estudiar + practicar por cada uno), tan rico como el motor del docente.
        debiles = self._conceptos_mas_debiles(secuencia, settings.max_conceptos_debiles)

        # Repartimos el cupo total entre los conceptos: con 1 concepto trae hasta
        # max_recomendaciones; con 3, ~2 cada uno. Así nunca queda escueto.
        por_concepto = (
            max(2, -(-settings.max_recomendaciones // len(debiles))) if debiles else 0
        )

        items: list[ItemRecomendado] = []
        usados: set[str] = set()
        for concepto, dominio_sec in debiles:
            candidatos = await self._cursos.obtener_recursos_candidatos(
                cmd.curso_id, limit=10, seccion=concepto
            )
            if not candidatos:
                continue
            items.extend(
                self._rankear_concepto(
                    candidatos, dominio_sec, concepto, usados, por_concepto
                )
            )

        # Fallback: sin interacciones (o sin candidatos por sección) → curso completo
        # con el dominio global, igual que antes para no romper el caso sin data.
        if not items:
            candidatos = await self._cursos.obtener_recursos_candidatos(
                cmd.curso_id, limit=20
            )
            items = self._rankear_concepto(
                candidatos,
                prediccion.probabilidad_dominio,
                None,
                usados,
                por_concepto=settings.max_recomendaciones,
            )

        # Límite global y renumeración del orden.
        items = items[: settings.max_recomendaciones]
        for i, it in enumerate(items):
            it.orden = i + 1

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
    def _conceptos_mas_debiles(secuencia, top_k: int) -> list[tuple[str, float]]:
        """Top-K conceptos con menor dominio (promedio de aciertos), de menor a mayor.

        Devuelve ``[(concepto, dominio_0a1), ...]``. Vacío si no hay interacciones.
        """
        conceptos = getattr(secuencia, "concepto_ids", None) or []
        aciertos = getattr(secuencia, "respuestas_correctas", None) or []
        if not conceptos:
            return []

        suma: dict[str, float] = {}
        cuenta: dict[str, int] = {}
        for concepto, correcta in zip(conceptos, aciertos):
            suma[concepto] = suma.get(concepto, 0.0) + (1.0 if correcta else 0.0)
            cuenta[concepto] = cuenta.get(concepto, 0) + 1

        promedios = [(c, suma[c] / cuenta[c]) for c in cuenta]
        promedios.sort(key=lambda x: x[1])  # más débil primero
        return promedios[:top_k]

    def _rankear_concepto(
        self,
        candidatos: list[dict],
        dominio: float,
        concepto: str | None,
        usados: set[str],
        por_concepto: int = 2,
    ) -> list[ItemRecomendado]:
        """Elige los mejores recursos de un concepto: prioriza 1 de estudio + 1 de
        práctica (estudiar y luego aplicar), sin repetir lo ya elegido en otros
        conceptos. ``por_concepto`` topa cuántos recursos aporta este concepto.
        """
        nivel_obj = (
            "basico" if dominio < 0.4 else "intermedio" if dominio < 0.7 else "avanzado"
        )
        orden = {"basico": 0, "intermedio": 1, "avanzado": 2}
        obj_idx = orden[nivel_obj]

        tipos_estudio = {"lectura", "video", "presentacion"}
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
            base = 0.6 if es_estudio(r) else 0.3 if not es_foro(r) else 0.0
            return round(min(1.0, base + 0.4 * alineamiento(r)), 4)

        def clave(r: dict) -> str:
            return str(r.get("url") or r.get("id") or r.get("titulo") or "")

        # Excluir foros y lo ya usado en otros conceptos.
        disponibles = [
            r for r in candidatos if not es_foro(r) and clave(r) not in usados
        ]
        estudio = sorted(
            [r for r in disponibles if es_estudio(r)], key=lambda r: -score(r)
        )
        practica = sorted(
            [r for r in disponibles if not es_estudio(r)], key=lambda r: -score(r)
        )

        # Intercala estudiar/practicar para que cada concepto traiga ambos formatos.
        elegidos: list[dict] = []
        i = j = 0
        while len(elegidos) < por_concepto and (i < len(estudio) or j < len(practica)):
            if i < len(estudio):
                elegidos.append(estudio[i])
                i += 1
            if len(elegidos) < por_concepto and j < len(practica):
                elegidos.append(practica[j])
                j += 1

        concepto_txt = concepto or "los conceptos del curso"
        items: list[ItemRecomendado] = []
        for r in elegidos:
            usados.add(clave(r))
            estudia = es_estudio(r)
            motivo = (
                f"Refuerza {concepto_txt} — el modelo SAKT estima tu dominio en "
                f"{dominio:.0%}. "
                f"{'Material de estudio' if estudia else 'Práctica'} para afianzar."
            )
            items.append(
                ItemRecomendado(
                    recurso_id=str(r.get("id", "")),
                    titulo=r.get("titulo", ""),
                    tipo=r.get("tipo", ""),
                    score=score(r),
                    orden=len(items) + 1,
                    motivo=motivo,
                    url=r.get("url", ""),
                )
            )
        return items
