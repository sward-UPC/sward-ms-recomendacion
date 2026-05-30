from enum import StrEnum


class EstadoRecomendacion(StrEnum):
    PENDIENTE = "pendiente"
    COMPLETADO = "completado"
    DESCARTADO = "descartado"
