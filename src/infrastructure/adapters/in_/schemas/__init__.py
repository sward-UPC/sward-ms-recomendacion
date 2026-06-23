"""Contratos Pydantic (Request/Response) del adaptador de entrada HTTP.

Organizados por concern (recomendacion, material, ejercicio, atencion, modelo) en
vez de un único archivo, según buenas prácticas de organización. Se re-exportan
aquí para que las rutas importen desde `...adapters.in_.schemas` sin conocer el
submódulo.
"""

from .atencion import AtencionResponse, PuntoAtencionResponse
from .ejercicio import VerificacionResponse, VerificarEjercicioRequest
from .material import GenerarMaterialRequest, MaterialResponse
from .modelo import ModelInfoResponse
from .recomendacion import (
    CompletarRecomendacionResponse,
    GenerarRecomendacionRequest,
    GenerarRecomendacionResponse,
    ItemRecomendadoResponse,
    ListarRecomendacionesResponse,
)

__all__ = [
    "ItemRecomendadoResponse",
    "GenerarRecomendacionRequest",
    "GenerarRecomendacionResponse",
    "ListarRecomendacionesResponse",
    "CompletarRecomendacionResponse",
    "GenerarMaterialRequest",
    "MaterialResponse",
    "VerificarEjercicioRequest",
    "VerificacionResponse",
    "PuntoAtencionResponse",
    "AtencionResponse",
    "ModelInfoResponse",
]
