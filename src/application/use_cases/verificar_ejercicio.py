import json
import re
from dataclasses import dataclass

from src.application.ports.out_.llm_client_port import LlmClientPort


@dataclass
class VerificarEjercicioCommand:
    enunciado: str
    solucion: str
    respuesta: str


@dataclass
class VerificacionEjercicio:
    aprobado: bool
    feedback: str


class VerificarEjercicioUseCase:
    """Evalúa con un LLM si la respuesta del estudiante a un ejercicio es correcta.

    Best-effort: si el LLM no está disponible o devuelve algo inválido, no aprueba
    pero da un feedback amable para que el alumno reintente (nunca rompe).
    """

    def __init__(self, llm: LlmClientPort):
        self._llm = llm

    async def execute(self, cmd: VerificarEjercicioCommand) -> VerificacionEjercicio:
        respuesta = (cmd.respuesta or "").strip()
        if not respuesta:
            return VerificacionEjercicio(
                aprobado=False, feedback="Escribe tu respuesta para poder verificarla."
            )

        prompt = (
            "Eres un tutor universitario evaluando la respuesta de un estudiante a un "
            "ejercicio. Sé justo y alentador: aprueba si la respuesta es correcta en lo "
            "esencial, aunque esté redactada distinto o de forma incompleta.\n"
            f"Ejercicio: {cmd.enunciado}\n"
            f"Solución de referencia: {cmd.solucion}\n"
            f"Respuesta del estudiante: {respuesta}\n"
            'Responde SOLO con un JSON válido (sin texto extra): {"aprobado": true/false, '
            '"feedback": "comentario breve y alentador en español, máximo 2 frases; si '
            "no aprueba, una pista de qué le falta. Usa markdown (**negrita** en lo "
            'clave) si ayuda a la claridad"}.'
        )

        texto = await self._llm.generar_texto(prompt)
        if texto is None:
            return VerificacionEjercicio(
                aprobado=False,
                feedback="No pude verificar ahora mismo. Inténtalo de nuevo en un momento.",
            )

        parsed = self._extraer_json(texto)
        if parsed is None:
            return VerificacionEjercicio(
                aprobado=False,
                feedback="No pude interpretar la evaluación. Revisa tu respuesta e inténtalo de nuevo.",
            )

        return VerificacionEjercicio(
            aprobado=bool(parsed.get("aprobado", False)),
            feedback=str(parsed.get("feedback", "")).strip()
            or "Sigue intentando, vas bien.",
        )

    @staticmethod
    def _extraer_json(texto: str) -> dict | None:
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except (ValueError, TypeError):
            return None
        return data if isinstance(data, dict) else None
