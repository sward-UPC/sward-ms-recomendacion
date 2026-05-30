from dataclasses import dataclass, field
from uuid import UUID, uuid4

from sward_shared.events.domain_event import DomainEvent


@dataclass
class RecomendacionGeneradaEvent(DomainEvent):
    recomendacion_id: UUID = field(default_factory=uuid4)
    estudiante_id: UUID = field(default_factory=uuid4)
    curso_id: UUID = field(default_factory=uuid4)
    source: str = "sward-ms-recomendacion"

    @property
    def event_type(self) -> str:
        return "sward.recomendacion.RecomendacionGenerada"
