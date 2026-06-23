"""Contratos HTTP de verificación de ejercicios resueltos por el estudiante."""

from pydantic import BaseModel, Field


class VerificarEjercicioRequest(BaseModel):
    enunciado: str = Field(..., description="Enunciado del ejercicio")
    solucion: str = Field(..., description="Solución de referencia del ejercicio")
    respuesta: str = Field(..., description="Respuesta escrita por el estudiante")


class VerificacionResponse(BaseModel):
    aprobado: bool = Field(
        ..., description="True si la IA da por correcta la respuesta"
    )
    feedback: str = Field(
        ..., description="Comentario breve y alentador para el alumno"
    )
