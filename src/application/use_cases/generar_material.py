import json
import re
import time
from dataclasses import dataclass, field
from uuid import UUID

from src.application.ports.out_.cursos_client_port import CursosClientPort
from src.application.ports.out_.llm_client_port import LlmClientPort
from src.application.ports.out_.trazabilidad_client_port import TrazabilidadClientPort
from src.application.ports.out_.youtube_client_port import YoutubeClientPort

# Cache en memoria del material generado (clave: estudiante+curso). Generar cuesta
# una llamada a Bedrock + una a YouTube y tarda unos segundos; el concepto débil
# cambia despacio, así que cacheamos el resultado con TTL para abaratar y acelerar.
# Vive en el proceso (se pierde al redeploy, aceptable: se regenera una vez).
_MATERIAL_CACHE: dict[tuple[str, str], tuple[float, "MaterialGenerado"]] = {}


@dataclass
class GenerarMaterialCommand:
    estudiante_id: UUID
    curso_id: UUID
    # Si True, ignora el cache y genera material fresco ("Generar más").
    refrescar: bool = False
    # Formato en el que el alumno aprende/rinde mejor (lectura/video/quiz/práctica),
    # según la señal de preferencia. El LLM lo enfatiza si viene.
    formato_preferido: str | None = None
    # Concepto a EVITAR (el que ya se mostró): en "Generar más" rota al siguiente
    # concepto débil para traer material genuinamente nuevo, no el mismo.
    evitar_concepto: str | None = None


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
        cache_ttl_s: int = 21600,
    ):
        self._trazabilidad = trazabilidad
        self._cursos = cursos
        self._llm = llm
        self._youtube = youtube
        self._cache_ttl_s = cache_ttl_s

    async def execute(self, cmd: GenerarMaterialCommand) -> MaterialGenerado:
        # Cache HIT: devuelve sin tocar Bedrock/YouTube/trazabilidad (rápido y barato).
        cache_key = (str(cmd.estudiante_id), str(cmd.curso_id))
        ahora = time.time()
        cacheado = _MATERIAL_CACHE.get(cache_key)
        if (
            not cmd.refrescar
            and cacheado is not None
            and ahora - cacheado[0] < self._cache_ttl_s
        ):
            print(f"[MATERIAL] cache HIT | {cache_key[0]}", flush=True)
            return cacheado[1]

        secuencia = await self._trazabilidad.obtener_secuencia(
            cmd.estudiante_id, cmd.curso_id
        )
        # En "Generar más" rota al siguiente concepto débil (evita repetir el actual).
        concepto_debil, dominio = self._concepto_mas_debil(
            secuencia, evitar=cmd.evitar_concepto if cmd.refrescar else None
        )

        candidatos = await self._cursos.obtener_recursos_candidatos(
            cmd.curso_id, limit=8, seccion=concepto_debil
        )
        titulos = [str(r.get("titulo", "")) for r in candidatos if r.get("titulo")]

        concepto_txt = concepto_debil or "los conceptos del curso"
        prompt = self._construir_prompt(
            concepto_txt, dominio, titulos, cmd.formato_preferido
        )

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

        # Más peso al formato fuerte: el LLM ya lo hace más rico (los demás breves);
        # aquí además lo ponemos PRIMERO para que lidere el material.
        tipo_pref = (
            self._FORMATO_TIPO.get(str(cmd.formato_preferido).lower())
            if cmd.formato_preferido
            else None
        )
        if tipo_pref:
            recursos.sort(key=lambda r: 0 if r.get("tipo") == tipo_pref else 1)

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

    # Mapea el tipo de preferencia (modname Moodle / tipo SWARD) al formato del material.
    _FORMATO_PREF = {
        "lectura": "la lectura (mini-lección extensa y muchas flashcards)",
        "page": "la lectura (mini-lección extensa y muchas flashcards)",
        "book": "la lectura (mini-lección extensa y muchas flashcards)",
        "resource": "la lectura (mini-lección extensa y muchas flashcards)",
        "video": "el video y la lectura",
        "url": "el video y la lectura",
        "quiz": "el quiz (más preguntas y explicaciones detalladas)",
        "ejercicio": "la práctica (más ejercicios con pista y solución)",
        "practica": "la práctica (más ejercicios con pista y solución)",
        "assign": "la práctica (más ejercicios con pista y solución)",
        "presentacion": "la lectura (mini-lección extensa y muchas flashcards)",
    }

    # Mapea el formato preferido al TIPO de recurso del material, para liderar con él.
    _FORMATO_TIPO = {
        "lectura": "lectura",
        "page": "lectura",
        "book": "lectura",
        "resource": "lectura",
        "presentacion": "lectura",
        "video": "video",
        "url": "video",
        "quiz": "quiz",
        "ejercicio": "practica",
        "practica": "practica",
        "assign": "practica",
        "workshop": "practica",
    }

    @staticmethod
    def _construir_prompt(
        concepto: str,
        dominio: int,
        titulos: list[str],
        formato_preferido: str | None = None,
    ) -> str:
        titulos_txt = ", ".join(titulos) if titulos else "ninguno disponible"
        pref = ""
        if formato_preferido:
            enfasis = GenerarMaterialUseCase._FORMATO_PREF.get(
                str(formato_preferido).lower()
            )
            if enfasis:
                pref = (
                    f" Este estudiante aprende mejor con {enfasis}: genera ESE "
                    "formato mucho más rico y extenso, y haz los demás formatos más "
                    "breves (pero incluye igual los 4 campos del JSON)."
                )
        return (
            "Eres un tutor universitario. Genera material de refuerzo para el "
            f'concepto "{concepto}" para un estudiante cuyo dominio estimado es '
            f"{dominio}%.{pref} Apóyate en estos recursos del curso: {titulos_txt}. "
            "Responde SOLO con un JSON válido (sin texto extra) con esta forma exacta: "
            "{"
            '"quiz": {"titulo": "...", "preguntas": [{"enunciado": "...", '
            '"opciones": ["a", "b", "c", "d"], "correcta": 0, "explicacion": "..."}]}, '
            '"lectura": {"titulo": "...", "contenido": "mini-lección de ~3 párrafos", '
            '"flashcards": [{"frente": "término o pregunta corta", '
            '"reverso": "definición o respuesta breve"}]}, '
            '"practica": {"titulo": "...", "ejercicios": [{"enunciado": "...", '
            '"pista": "una ayuda que oriente sin dar la respuesta", "solucion": "..."}]}, '
            '"video_query": "mejor búsqueda de YouTube para el concepto"'
            "}. "
            "El quiz debe tener entre 3 y 5 preguntas de opción múltiple, cada una "
            'con exactamente 4 opciones, "correcta" como índice (0-3) de la opción '
            "correcta y una breve explicación. La lectura debe incluir entre 4 y 6 "
            "flashcards (frente/reverso) con los conceptos clave para memorizar. La "
            "práctica debe tener entre 2 y 3 ejercicios, cada uno con una pista que "
            "oriente sin revelar la respuesta y con su solución completa. Ajusta la "
            "profundidad al nivel del estudiante: si su dominio es bajo, incluye más "
            "preguntas, flashcards y ejercicios, con explicaciones más detalladas; si "
            "es alto, material más breve y retador. En 'contenido', 'solucion', "
            "'explicacion' y 'pista' usa formato markdown para legibilidad: **negrita** "
            "en los términos clave y listas con '- ' cuando enumeres varios puntos."
        )

    @staticmethod
    def _extraer_json(texto: str) -> dict | None:
        """Extrae el objeto JSON del texto (tolera ```json ... ```, ruido y truncación).

        Si la respuesta del LLM viene completa, la parsea directo. Si vino
        truncada por ``max_tokens`` (JSON incompleto), intenta repararla cerrando
        strings/llaves/corchetes abiertos para recuperar lo que sí se generó
        (p.ej. quiz + lectura aunque la práctica se haya cortado).
        """
        # Caso feliz: hay un objeto JSON completo y válido.
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except (ValueError, TypeError):
                pass
        # Salvataje de respuesta truncada.
        start = texto.find("{")
        if start == -1:
            return None
        try:
            data = json.loads(
                GenerarMaterialUseCase._reparar_json_truncado(texto[start:])
            )
            return data if isinstance(data, dict) else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _reparar_json_truncado(s: str) -> str:
        """Cierra strings/llaves/corchetes abiertos de un JSON truncado.

        Recorre el texto respetando strings y escapes, lleva la pila de
        contenedores abiertos, descarta un fragmento colgante al final (coma o
        ``"clave":`` sin valor) y cierra todo lo pendiente. Best-effort.
        """
        out: list[str] = []
        stack: list[str] = []
        en_string = False
        escape = False
        for ch in s:
            if escape:
                out.append(ch)
                escape = False
                continue
            if en_string and ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                en_string = not en_string
                out.append(ch)
                continue
            if not en_string:
                if ch == "{":
                    stack.append("}")
                elif ch == "[":
                    stack.append("]")
                elif ch in "}]" and stack and stack[-1] == ch:
                    stack.pop()
            out.append(ch)
        if en_string:
            out.append('"')
        texto = "".join(out).rstrip()
        # Quita coma colgante o un par ``"clave":`` sin valor al final.
        texto = re.sub(r",\s*$", "", texto)
        texto = re.sub(r',?\s*"[^"]*"\s*:\s*$', "", texto)
        texto = re.sub(r",\s*$", "", texto)
        return texto + "".join(reversed(stack))

    @staticmethod
    def _concepto_mas_debil(
        secuencia, evitar: str | None = None
    ) -> tuple[str | None, int]:
        """Concepto con menor promedio de aciertos y ese promedio en %.

        Agrupa por concepto y toma el de menor promedio de aciertos (>= 1
        interacción). Si ``evitar`` se indica y hay más de un concepto, devuelve
        el siguiente más débil distinto de ``evitar`` (para que "Generar más"
        rote y no repita el mismo concepto).
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
        # De más débil a más fuerte.
        ordenados = sorted(promedios, key=lambda c: promedios[c])
        # Rota: salta el concepto a evitar si hay alternativa.
        elegido = ordenados[0]
        if evitar is not None and len(ordenados) > 1:
            elegido = next((c for c in ordenados if c != evitar), ordenados[0])
        dominio = round(promedios[elegido] * 100)
        return elegido, dominio
