from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4


@dataclass
class PrediccionKT:
    id: UUID = field(default_factory=uuid4)
    estudiante_id: UUID = field(default_factory=uuid4)
    curso_id: UUID = field(default_factory=uuid4)
    probabilidad_dominio: float = 0.5
    confianza: float = 0.5
    pesos_atencion: list[float] = field(default_factory=list)
    fecha_prediccion: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
