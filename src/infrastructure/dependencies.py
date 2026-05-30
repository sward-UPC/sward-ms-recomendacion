from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sward_shared.auth import build_require_jwt, build_require_service_key

from src.application.use_cases.completar_recomendacion import (
    CompletarRecomendacionUseCase,
)
from src.application.use_cases.consultar_recomendacion import (
    ConsultarRecomendacionUseCase,
)
from src.application.use_cases.generar_recomendacion import GenerarRecomendacionUseCase
from src.domain.services.modelo_sakt import ModeloSAKT
from src.infrastructure.adapters.out_.cursos_rest_adapter import CursosRestAdapter
from src.infrastructure.adapters.out_.eventbridge_adapter import EventBridgeAdapter
from src.infrastructure.adapters.out_.recomendacion_postgres_adapter import (
    RecomendacionPostgresAdapter,
)
from src.infrastructure.adapters.out_.trazabilidad_rest_adapter import (
    TrazabilidadRestAdapter,
)
from src.infrastructure.adapters.out_.xai_rest_adapter import XaiRestAdapter
from src.infrastructure.config.settings import settings
from src.infrastructure.db.database import get_session

# Dependencia de autenticación JWT reutilizable, compartida vía sward-shared.
require_jwt = build_require_jwt(settings.secret_key, algorithm=settings.jwt_algorithm)

# Validación entrante de service-key (modo dev permite sin claves configuradas).
require_service_key = build_require_service_key(settings.authorized_service_keys_set)


def get_modelo() -> ModeloSAKT:
    return ModeloSAKT(mock=(settings.environment == "development"))


def get_generar_uc(
    session: AsyncSession = Depends(get_session),
    modelo: ModeloSAKT = Depends(get_modelo),
) -> GenerarRecomendacionUseCase:
    return GenerarRecomendacionUseCase(
        trazabilidad=TrazabilidadRestAdapter(),
        cursos=CursosRestAdapter(),
        xai=XaiRestAdapter(),
        repo=RecomendacionPostgresAdapter(session),
        event_publisher=EventBridgeAdapter(),
        modelo=modelo,
    )


def get_consultar_uc(
    session: AsyncSession = Depends(get_session),
) -> ConsultarRecomendacionUseCase:
    return ConsultarRecomendacionUseCase(RecomendacionPostgresAdapter(session))


def get_completar_uc(
    session: AsyncSession = Depends(get_session),
) -> CompletarRecomendacionUseCase:
    return CompletarRecomendacionUseCase(RecomendacionPostgresAdapter(session))
