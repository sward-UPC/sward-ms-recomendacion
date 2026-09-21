from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4

from src.domain.entities.fidelidad_explicacion import FidelidadExplicacion


@dataclass
class PrediccionKT:
    id: UUID = field(default_factory=uuid4)
    estudiante_id: UUID = field(default_factory=uuid4)
    curso_id: UUID = field(default_factory=uuid4)
    probabilidad_dominio: float = 0.5
    # Confianza de la predicción. Hasta ahora era una constante escrita a mano
    # (0.85) que no salía de ningún cálculo: el panel y la explicación mostraban
    # un número que no medía nada. Cuando la verificación de fidelidad está
    # activa, este valor pasa a ser la fracción de borrados aleatorios a los que
    # la atención le gana, es decir, evidencia medida sobre esta secuencia.
    confianza: float = 0.5
    pesos_atencion: list[float] = field(default_factory=list)
    # Resultado de contrastar la atención contra el borrado aleatorio para ESTA
    # secuencia. None cuando el adaptador no pudo o no debía verificar.
    fidelidad: FidelidadExplicacion | None = None
    fecha_prediccion: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
