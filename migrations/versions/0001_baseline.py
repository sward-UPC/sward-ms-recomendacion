"""baseline: esquema inicial de recomendación

Adopta el esquema existente. Crea todas las tablas del modelo con
``checkfirst=True``, de modo que:
  - en una BD nueva crea todo desde cero;
  - en la BD compartida de dev (que ya tiene las tablas creadas por el antiguo
    ``create_all`` del arranque) NO toca nada y solo registra la versión.

A partir de aquí, todo cambio de esquema (nuevas columnas/tablas/índices) debe
ir en su propia revisión Alembic, no como DDL inline en el arranque.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-06-23
"""

from collections.abc import Sequence

from alembic import op

from src.infrastructure.db.models.recomendacion_models import Base

revision: str = "0001_baseline"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # checkfirst=True -> idempotente sobre BD existente (no recrea tablas).
    Base.metadata.create_all(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
