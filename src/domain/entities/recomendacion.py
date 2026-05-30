from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4

from src.domain.value_objects.estado_recomendacion import EstadoRecomendacion


@dataclass
class ItemRecomendado:
    recurso_id: str = ""
    titulo: str = ""
    tipo: str = ""
    score: float = 0.0
    orden: int = 0
    motivo: str = ""
    url: str = ""


@dataclass
class Recomendacion:
    id: UUID = field(default_factory=uuid4)
    estudiante_id: UUID = field(default_factory=uuid4)
    curso_id: UUID = field(default_factory=uuid4)
    estado: EstadoRecomendacion = EstadoRecomendacion.PENDIENTE
    items: list[ItemRecomendado] = field(default_factory=list)
    generada_en: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completada_en: datetime | None = None
