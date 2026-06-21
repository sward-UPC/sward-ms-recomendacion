import json
import re
from dataclasses import dataclass, field
from uuid import UUID

from src.domain.ports.out_.cursos_client_port import CursosClientPort
from src.domain.ports.out_.llm_client_port import LlmClientPort
from src.domain.ports.out_.trazabilidad_client_port import TrazabilidadClientPort


@dataclass
class GenerarMaterialCommand:
    estudiante_id: UUID
    curso_id: UUID


@dataclass
class MaterialGenerado:
    """Material de estudio generado (o fallback vacío) para un concepto débil."""

    disponible: bool
    concepto: str | None
    resumen: str = ""
    puntos_clave: list[str] = field(default_factory=list)
    preguntas: list[dict] = field(default_factory=list)


class GenerarMaterialUseCase:
    """Genera material de estudio nuevo con un LLM para reforzar el concepto débil.

    Best-effort: si no hay clave o el LLM falla, devuelve un material no disponible
    en lugar de romper.
    """

    def __init__(
        self,
        trazabilidad: TrazabilidadClientPort,
        cursos: CursosClientPort,
        llm: LlmClientPort,
    ):
        self._trazabilidad = trazabilidad
        self._cursos = cursos
        self._llm = llm

    async def execute(self, cmd: GenerarMaterialCommand) -> MaterialGenerado:
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
            return MaterialGenerado(disponible=False, concepto=concepto_debil)

        parsed = self._extraer_json(texto)
        if parsed is None:
            return MaterialGenerado(disponible=False, concepto=concepto_debil)

        return MaterialGenerado(
            disponible=True,
            concepto=concepto_debil,
            resumen=str(parsed.get("resumen", "")),
            puntos_clave=list(parsed.get("puntos_clave", []) or []),
            preguntas=list(parsed.get("preguntas", []) or []),
        )

    @staticmethod
    def _construir_prompt(concepto: str, dominio: int, titulos: list[str]) -> str:
        titulos_txt = ", ".join(titulos) if titulos else "ninguno disponible"
        return (
            "Eres un tutor universitario. Genera material breve para reforzar el "
            f'concepto "{concepto}" en un estudiante cuyo dominio estimado es '
            f"{dominio}%. Apóyate en estos recursos del curso: {titulos_txt}. "
            "Responde SOLO con un JSON válido (sin texto extra) con esta forma: "
            '{"resumen": "2-3 frases claras del concepto", "puntos_clave": '
            '["3 a 5 ideas clave"], "preguntas": [{"pregunta": "...", '
            '"respuesta": "..."}]} con 3 preguntas de práctica con su respuesta.'
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
