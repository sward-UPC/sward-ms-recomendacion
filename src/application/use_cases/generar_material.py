import json
import re
import time
from dataclasses import dataclass, field
from uuid import UUID

from src.domain.ports.out_.cursos_client_port import CursosClientPort
from src.domain.ports.out_.llm_client_port import LlmClientPort
from src.domain.ports.out_.trazabilidad_client_port import TrazabilidadClientPort
from src.domain.ports.out_.youtube_client_port import YoutubeClientPort
from src.infrastructure.config.settings import settings

# Cache en memoria del material generado (clave: estudiante+curso). Generar cuesta
# una llamada a Bedrock + una a YouTube y tarda unos segundos; el concepto débil
# cambia despacio, así que cacheamos el resultado con TTL para abaratar y acelerar.
# Vive en el proceso (se pierde al redeploy, aceptable: se regenera una vez).
_MATERIAL_CACHE: dict[tuple[str, str], tuple[float, "MaterialGenerado"]] = {}


@dataclass
class GenerarMaterialCommand:
    estudiante_id: UUID
    curso_id: UUID


@dataclass
class MaterialGenerado:
    """Set de recursos educativos tipados (o fallback vacío) para un concepto débil.

    ``recursos`` es una lista de dicts tipados por ``tipo``:
    quiz, lectura, practica y (opcional) video.
    """

    disponible: bool
    concepto: str | None
    recursos: list[dict] = field(default_factory=list)
    # Dominio estimado por el SAKT en el concepto débil (0-100), para el XAI.
    dominio: int | None = None


class GenerarMaterialUseCase:
    """Genera un set de recursos educativos tipados con un LLM + un video de YouTube.

    El LLM (Bedrock) produce un quiz, una mini-lección y una práctica para
    reforzar el concepto débil; YouTube aporta un video real. Best-effort: si no
    hay clave o el LLM falla, devuelve un material no disponible en lugar de romper.
    """

    def __init__(
        self,
        trazabilidad: TrazabilidadClientPort,
        cursos: CursosClientPort,
        llm: LlmClientPort,
        youtube: YoutubeClientPort,
    ):
        self._trazabilidad = trazabilidad
        self._cursos = cursos
        self._llm = llm
        self._youtube = youtube

    async def execute(self, cmd: GenerarMaterialCommand) -> MaterialGenerado:
        # Cache HIT: devuelve sin tocar Bedrock/YouTube/trazabilidad (rápido y barato).
        cache_key = (str(cmd.estudiante_id), str(cmd.curso_id))
        ahora = time.time()
        cacheado = _MATERIAL_CACHE.get(cache_key)
        if cacheado is not None and ahora - cacheado[0] < settings.material_cache_ttl_s:
            print(f"[MATERIAL] cache HIT | {cache_key[0]}", flush=True)
            return cacheado[1]

        secuencia = await self._trazabilidad.obtener_secuencia(
            cmd.estudiante_id, cmd.curso_id
        )
        concepto_debil, dominio = self._concepto_mas_debil(secuencia)

        candidatos = await self._cursos.obtener_recursos_candidatos(
            cmd.curso_id, limit=8, seccion=concepto_debil
        )
        titulos = [str(r.get("titulo", "")) for r in candidatos if r.get("titulo")]

        concepto_txt = concepto_debil or "los conceptos del curso"
        prompt = self._construir_prompt(concepto_txt, dominio, titulos)

        texto = await self._llm.generar_texto(prompt)
        if texto is None:
            print(f"[MATERIAL] LLM None | concepto={concepto_debil!r}", flush=True)
            return MaterialGenerado(
                disponible=False, concepto=concepto_debil, dominio=dominio
            )

        parsed = self._extraer_json(texto)
        if parsed is None:
            print(
                f"[MATERIAL] JSON inválido | concepto={concepto_debil!r} "
                f"len_texto={len(texto)}",
                flush=True,
            )
            return MaterialGenerado(
                disponible=False, concepto=concepto_debil, dominio=dominio
            )

        recursos = self._construir_recursos(parsed)
        video = await self._buscar_video(parsed, concepto_txt)
        if video is not None:
            recursos.append(video)

        tipos = [r.get("tipo") for r in recursos]
        print(
            f"[MATERIAL] OK | concepto={concepto_debil!r} recursos={tipos}",
            flush=True,
        )
        resultado = MaterialGenerado(
            disponible=True,
            concepto=concepto_debil,
            recursos=recursos,
            dominio=dominio,
        )
        # Solo cacheamos generaciones exitosas (los fallbacks se reintentan luego).
        _MATERIAL_CACHE[cache_key] = (ahora, resultado)
        return resultado

    @staticmethod
    def _construir_recursos(parsed: dict) -> list[dict]:
        """Arma los recursos quiz/lectura/practica desde el JSON del LLM."""
        recursos: list[dict] = []

        quiz = parsed.get("quiz") or {}
        preguntas_quiz = []
        for p in quiz.get("preguntas", []) or []:
            if not isinstance(p, dict):
                continue
            preguntas_quiz.append(
                {
                    "enunciado": str(p.get("enunciado", "")),
                    "opciones": [str(o) for o in (p.get("opciones") or [])],
                    "correcta": int(p.get("correcta", 0) or 0),
                    "explicacion": str(p.get("explicacion", "")),
                }
            )
        if preguntas_quiz:
            recursos.append(
                {
                    "tipo": "quiz",
                    "titulo": str(quiz.get("titulo", "Quiz de refuerzo")),
                    "preguntas": preguntas_quiz,
                }
            )

        lectura = parsed.get("lectura") or {}
        contenido = str(lectura.get("contenido", ""))
        if contenido:
            flashcards = []
            for fc in lectura.get("flashcards", []) or []:
                if not isinstance(fc, dict):
                    continue
                frente = str(fc.get("frente", "")).strip()
                reverso = str(fc.get("reverso", "")).strip()
                if frente and reverso:
                    flashcards.append({"frente": frente, "reverso": reverso})
            recursos.append(
                {
                    "tipo": "lectura",
                    "titulo": str(lectura.get("titulo", "Mini-lección")),
                    "contenido": contenido,
                    "flashcards": flashcards,
                }
            )

        practica = parsed.get("practica") or {}
        ejercicios = []
        for e in practica.get("ejercicios", []) or []:
            if not isinstance(e, dict):
                continue
            ejercicios.append(
                {
                    "enunciado": str(e.get("enunciado", "")),
                    "pista": str(e.get("pista", "")),
                    "solucion": str(e.get("solucion", "")),
                }
            )
        if ejercicios:
            recursos.append(
                {
                    "tipo": "practica",
                    "titulo": str(practica.get("titulo", "Práctica")),
                    "ejercicios": ejercicios,
                }
            )

        return recursos

    async def _buscar_video(self, parsed: dict, concepto_txt: str) -> dict | None:
        """Busca un video real en YouTube con la query sugerida por el LLM."""
        query = str(parsed.get("video_query", "") or "").strip() or concepto_txt
        encontrado = await self._youtube.buscar_video(query)
        if not encontrado:
            return None
        return {
            "tipo": "video",
            "titulo": str(encontrado.get("titulo", "")),
            "video_id": str(encontrado.get("video_id", "")),
            "url": str(encontrado.get("url", "")),
            "query": query,
        }

    @staticmethod
    def _construir_prompt(concepto: str, dominio: int, titulos: list[str]) -> str:
        titulos_txt = ", ".join(titulos) if titulos else "ninguno disponible"
        return (
            "Eres un tutor universitario. Genera material de refuerzo para el "
            f'concepto "{concepto}" para un estudiante cuyo dominio estimado es '
            f"{dominio}%. Apóyate en estos recursos del curso: {titulos_txt}. "
            "Responde SOLO con un JSON válido (sin texto extra) con esta forma exacta: "
            "{"
            '"quiz": {"titulo": "...", "preguntas": [{"enunciado": "...", '
            '"opciones": ["a", "b", "c", "d"], "correcta": 0, "explicacion": "..."}]}, '
            '"lectura": {"titulo": "...", "contenido": "mini-lección de 2 párrafos", '
            '"flashcards": [{"frente": "término o pregunta corta", '
            '"reverso": "definición o respuesta breve"}]}, '
            '"practica": {"titulo": "...", "ejercicios": [{"enunciado": "...", '
            '"pista": "una ayuda que oriente sin dar la respuesta", "solucion": "..."}]}, '
            '"video_query": "mejor búsqueda de YouTube para el concepto"'
            "}. "
            "El quiz debe tener exactamente 3 preguntas de opción múltiple, cada una "
            'con exactamente 4 opciones, "correcta" como índice (0-3) de la opción '
            "correcta y una explicación de una frase. La lectura debe incluir 4 "
            "flashcards (frente/reverso) breves con los conceptos clave. La práctica "
            "debe tener 2 ejercicios, cada uno con una pista corta que oriente sin "
            "revelar la respuesta y con su solución. Sé conciso en todo."
        )

    @staticmethod
    def _extraer_json(texto: str) -> dict | None:
        """Extrae el primer objeto JSON del texto (tolera ```json ... ``` o ruido)."""
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except (ValueError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _concepto_mas_debil(secuencia) -> tuple[str | None, int]:
        """Concepto con menor promedio de aciertos y ese promedio en %.

        Reutiliza la lógica de generar_recomendacion: agrupa por concepto y toma
        el de menor promedio de aciertos (>= 1 interacción).
        """
        conceptos = getattr(secuencia, "concepto_ids", None) or []
        aciertos = getattr(secuencia, "respuestas_correctas", None) or []
        if not conceptos:
            return None, 50

        suma: dict[str, float] = {}
        cuenta: dict[str, int] = {}
        for concepto, correcta in zip(conceptos, aciertos):
            suma[concepto] = suma.get(concepto, 0.0) + (1.0 if correcta else 0.0)
            cuenta[concepto] = cuenta.get(concepto, 0) + 1

        if not cuenta:
            return None, 50

        promedios = {c: suma[c] / cuenta[c] for c in cuenta}
        concepto_debil = min(promedios, key=lambda c: promedios[c])
        dominio = round(promedios[concepto_debil] * 100)
        return concepto_debil, dominio
