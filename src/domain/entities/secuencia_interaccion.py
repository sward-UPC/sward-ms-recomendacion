from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass
class SecuenciaInteraccion:
    id: UUID = field(default_factory=uuid4)
    estudiante_id: UUID = field(default_factory=uuid4)
    curso_id: UUID = field(default_factory=uuid4)
    concepto_ids: list[str] = field(default_factory=list)
    respuestas_correctas: list[bool] = field(default_factory=list)

    def to_vectors(self) -> tuple[list[int], list[int]]:
        concepto_map = {c: i for i, c in enumerate(sorted(set(self.concepto_ids)))}
        x = [concepto_map.get(c, 0) for c in self.concepto_ids]
        y = [1 if r else 0 for r in self.respuestas_correctas]
        return x, y
