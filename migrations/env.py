"""Entorno Alembic (async) para sward-ms-recomendacion.

- La URL se toma de `settings.database_url` (postgresql+asyncpg), igual que la app.
- `version_table = alembic_version_recomendacion`: cada microservicio versiona por
  separado, de modo que conviven en la BD compartida de dev sin pisarse, y en
  prod (BD separadas) funciona igual.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from src.infrastructure.config.settings import settings
from src.infrastructure.db.models.recomendacion_models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
VERSION_TABLE = "alembic_version_recomendacion"


def run_migrations_offline() -> None:
    """Modo offline: genera SQL sin conectarse (alembic upgrade --sql)."""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Modo online: abre un engine async y corre las migraciones."""
    connectable = create_async_engine(settings.database_url, poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
