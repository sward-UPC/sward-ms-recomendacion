from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.application.use_cases.consultar_recomendacion import (
    ConsultarRecomendacionUseCase,
)
from src.application.use_cases.generar_recomendacion import GenerarRecomendacionUseCase
from src.domain.entities.recomendacion import ItemRecomendado, Recomendacion
from src.infrastructure.adapters.in_.main import app
from src.infrastructure.db.database import get_session
from src.infrastructure.dependencies import (
    get_consultar_uc,
    get_generar_uc,
    require_jwt,
)


def _fake_recomendacion(n_items: int = 4) -> Recomendacion:
    return Recomendacion(
        estudiante_id=uuid4(),
        curso_id=uuid4(),
        items=[
            ItemRecomendado(
                recurso_id=f"r{i}",
                titulo=f"Recurso {i}",
                tipo="video",
                score=1.0 / (i + 1),
                orden=i + 1,
                motivo="alineado con tu dominio",
            )
            for i in range(n_items)
        ],
    )


@pytest.fixture
def fake_payload() -> dict:
    return {
        "sub": str(uuid4()),
        "rol": "estudiante",
        "permisos": ["leer"],
        "type": "access",
    }


@pytest.fixture
def auth_app(fake_payload):
    """App con los puertos mockeados y JWT sobreescrito por un payload fake."""
    generar_uc = AsyncMock(spec=GenerarRecomendacionUseCase)
    generar_uc.execute.return_value = _fake_recomendacion()

    consultar_uc = AsyncMock(spec=ConsultarRecomendacionUseCase)
    consultar_uc.execute.return_value = [_fake_recomendacion(3)]

    app.dependency_overrides[get_generar_uc] = lambda: generar_uc
    app.dependency_overrides[get_consultar_uc] = lambda: consultar_uc
    app.dependency_overrides[require_jwt] = lambda: fake_payload
    yield app
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def auth_client(auth_app) -> AsyncClient:
    transport = ASGITransport(app=auth_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def anon_client() -> AsyncClient:
    """Cliente sin overrides de auth: los endpoints protegidos exigen token real.

    Se sobreescribe ``get_session`` para no requerir una base de datos real; la
    autenticación se evalúa igualmente y debe bloquear el acceso sin token.
    """

    async def _fake_session():
        yield None

    app.dependency_overrides[get_session] = _fake_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()
