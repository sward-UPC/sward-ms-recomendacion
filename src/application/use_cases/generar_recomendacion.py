import time
from dataclasses import dataclass
from uuid import UUID

from src.domain.entities.recomendacion import ItemRecomendado, Recomendacion
from src.domain.events.recomendacion_generada_event import RecomendacionGeneradaEvent
from src.domain.ports.out_.cursos_client_port import CursosClientPort
from src.domain.ports.out_.event_publisher_port import EventPublisherPort
from src.domain.ports.out_.modelo_kt_port import ModeloKTPort
from src.domain.ports.out_.recomendacion_repository_port import (
    RecomendacionRepositoryPort,
)
from src.domain.ports.out_.trazabilidad_client_port import TrazabilidadClientPort
from src.domain.ports.out_.xai_client_port import XaiClientPort


# Cache en memoria de la recomendación SAKT por estudiante+curso. Generar implica
# inferencia del modelo + varias llamadas a cursos; el resultado cambia despacio, así
# que lo cacheamos con TTL para que recargar la página sea instantáneo.
_RECOMENDACION_CACHE: dict[tuple[str, str], tuple[float, "Recomendacion"]] = {}


# Preferencia de formato: clasifica tipos (Moodle y SWARD) en práctica vs estudio,
# para que el motor priorice el formato en el que el estudiante rinde/consume mejor.
_PRACTICA_TIPOS = {
    "quiz",
    "assign",
    "workshop",
    "ejercicio",
    "practica",
    "práctica",
    "tarea",
    "taller",
}
# Moodle modname → tipo SWARD del catálogo de recursos (para match fino de formato).
_MOODLE_A_SWARD = {
    "url": "video",
    "page": "lectura",
    "book": "lectura",
    "lesson": "lectura",
    "resource": "lectura",
    "quiz": "quiz",
    "assign": "ejercicio",
    "workshop": "ejercicio",
}


def _categoria(tipo: str) -> str:
    """Categoría gruesa del recurso: 'practica' o 'estudio'."""
    return "practica" if (tipo or "").lower() in _PRACTICA_TIPOS else "estudio"


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
        modelo: ModeloKTPort,
        cache_ttl_s: int = 1800,
        max_conceptos_debiles: int = 3,
        max_recomendaciones: int = 6,
    ):
        self._trazabilidad = trazabilidad
        self._cursos = cursos
        self._xai = xai
        self._repo = repo
        self._event_publisher = event_publisher
        self._modelo = modelo
        self._cache_ttl_s = cache_ttl_s
        self._max_conceptos_debiles = max_conceptos_debiles
        self._max_recomendaciones = max_recomendaciones

    async def execute(self, cmd: GenerarRecomendacionCommand) -> Recomendacion:
        # Cache HIT: devuelve la recomendación sin re-inferir el SAKT ni llamar a
        # cursos (recargar la página es instantáneo).
        cache_key = (str(cmd.estudiante_id), str(cmd.curso_id))
        ahora = time.time()
        cacheada = _RECOMENDACION_CACHE.get(cache_key)
        if cacheada is not None and ahora - cacheada[0] < self._cache_ttl_s:
            print(f"[RECOMENDACION] cache HIT | {cache_key[0]}", flush=True)
            return cacheada[1]

        secuencia = await self._trazabilidad.obtener_secuencia(
            cmd.estudiante_id, cmd.curso_id
        )
        prediccion = self._modelo.predecir_dominio(secuencia)

        # Preferencia de formato (best-effort): en qué tipo de recurso rinde/consume
        # mejor el estudiante. El SAKT decide QUÉ conceptos reforzar; la preferencia
        # decide CON QUÉ formato (p.ej. liderar con práctica si ahí rinde mejor).
        prefs = await self._trazabilidad.obtener_preferencias(
            cmd.estudiante_id, cmd.curso_id
        )
        pref_tipo = ""
        if prefs:
            pref_tipo = (
                prefs.get("tipo_fuerte") or prefs.get("formato_mas_consumido") or ""
            )
        prefiere_practica = bool(pref_tipo) and _categoria(pref_tipo) == "practica"

        # Top-K conceptos más débiles según el dominio estimado por sección. Antes
        # solo targeteábamos UNO → muy pocos items; ahora cubrimos varios conceptos
        # (estudiar + practicar por cada uno), tan rico como el motor del docente.
        debiles = self._conceptos_mas_debiles(secuencia, self._max_conceptos_debiles)

        # Repartimos el cupo total entre los conceptos: con 1 concepto trae hasta
        # max_recomendaciones; con 3, ~2 cada uno. Así nunca queda escueto.
        por_concepto = (
            max(2, -(-self._max_recomendaciones // len(debiles))) if debiles else 0
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
                    candidatos,
                    dominio_sec,
                    concepto,
                    usados,
                    por_concepto,
                    pref_tipo=pref_tipo,
                    prefiere_practica=prefiere_practica,
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
                por_concepto=self._max_recomendaciones,
                pref_tipo=pref_tipo,
                prefiere_practica=prefiere_practica,
            )

        # Límite global y renumeración del orden.
        items = items[: self._max_recomendaciones]
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
        # Solo cacheamos si hubo items (no cacheamos resultados vacíos por falta de data).
        if guardada.items:
            _RECOMENDACION_CACHE[cache_key] = (ahora, guardada)
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
        pref_tipo: str = "",
        prefiere_practica: bool = False,
    ) -> list[ItemRecomendado]:
        """Elige los mejores recursos de un concepto: estudiar + practicar, sin
        repetir lo ya elegido en otros conceptos. ``por_concepto`` topa cuántos
        aporta este concepto.

        Si hay preferencia de formato (``pref_tipo``), prioriza ese formato:
        lidera con práctica cuando el estudiante rinde mejor practicando, y da un
        bonus de score al tipo preferido (así el modelo SÍ lo tiene en cuenta).
        """
        nivel_obj = (
            "basico" if dominio < 0.4 else "intermedio" if dominio < 0.7 else "avanzado"
        )
        orden = {"basico": 0, "intermedio": 1, "avanzado": 2}
        obj_idx = orden[nivel_obj]

        tipos_estudio = {"lectura", "video", "presentacion"}
        tipos_foro = {"foro", "forum"}

        # Tipo SWARD equivalente al formato preferido (Moodle), para el match fino.
        pref_sward = (
            _MOODLE_A_SWARD.get(pref_tipo.lower(), pref_tipo.lower())
            if pref_tipo
            else ""
        )

        def es_foro(r: dict) -> bool:
            tipo = str(r.get("tipo", "")).lower()
            titulo = str(r.get("titulo", "")).lower()
            return tipo in tipos_foro or "foro" in titulo or "announcements" in titulo

        def es_estudio(r: dict) -> bool:
            return str(r.get("tipo", "")).lower() in tipos_estudio

        def alineamiento(r: dict) -> float:
            dist = abs(orden.get(r.get("nivel_dificultad", "intermedio"), 1) - obj_idx)
            return 1.0 / (1 + dist)

        def bonus_pref(r: dict) -> float:
            """Bonus por alinear con la preferencia del estudiante."""
            if not pref_tipo:
                return 0.0
            t = str(r.get("tipo", "")).lower()
            if pref_sward and t == pref_sward:
                return 0.25  # match fino de formato (ej. video)
            if _categoria(t) == _categoria(pref_tipo):
                return 0.12  # match de categoría (práctica vs estudio)
            return 0.0

        def score(r: dict) -> float:
            # Base preferencia-aware: si rinde mejor practicando, la práctica pesa
            # más que el estudio (y viceversa, que es el comportamiento por defecto).
            if es_foro(r):
                base = 0.0
            elif es_estudio(r):
                base = 0.45 if prefiere_practica else 0.6
            else:
                base = 0.6 if prefiere_practica else 0.3
            return round(min(1.0, base + 0.4 * alineamiento(r) + bonus_pref(r)), 4)

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

        # Lidera con el formato preferido (práctica si ahí rinde mejor; si no,
        # estudiar→practicar), e intercala para traer variedad.
        buckets = [practica, estudio] if prefiere_practica else [estudio, practica]
        elegidos: list[dict] = []
        idxs = [0, 0]
        while len(elegidos) < por_concepto and any(
            idxs[k] < len(buckets[k]) for k in (0, 1)
        ):
            for k in (0, 1):
                if len(elegidos) < por_concepto and idxs[k] < len(buckets[k]):
                    elegidos.append(buckets[k][idxs[k]])
                    idxs[k] += 1

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
            # Explicación honesta cuando se priorizó por la preferencia de formato.
            if pref_tipo and _categoria(str(r.get("tipo", "")).lower()) == _categoria(
                pref_tipo
            ):
                fmt = "practicando" if prefiere_practica else "con material de estudio"
                motivo += f" Priorizado: rindes mejor {fmt}."
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
