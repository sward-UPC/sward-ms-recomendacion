from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sward_shared.auth import build_require_jwt, build_require_service_key

from src.application.use_cases.completar_recomendacion import (
    CompletarRecomendacionUseCase,
)
from src.application.use_cases.consultar_atencion import ConsultarAtencionUseCase
from src.application.use_cases.consultar_modelo_info import ConsultarModeloInfoUseCase
from src.application.use_cases.consultar_recomendacion import (
    ConsultarRecomendacionUseCase,
)
from src.application.use_cases.generar_material import GenerarMaterialUseCase
from src.application.use_cases.generar_recomendacion import GenerarRecomendacionUseCase
from src.application.use_cases.verificar_ejercicio import VerificarEjercicioUseCase
from src.application.ports.out_.modelo_kt_port import ModeloKTPort
from src.infrastructure.adapters.out_.bedrock_llm_adapter import BedrockLlmAdapter
from src.infrastructure.adapters.out_.cursos_rest_adapter import CursosRestAdapter
from src.infrastructure.adapters.out_.eventbridge_adapter import EventBridgeAdapter
from src.infrastructure.adapters.out_.modelo_kt_mock_adapter import ModeloKtMockAdapter
from src.infrastructure.adapters.out_.recomendacion_postgres_adapter import (
    RecomendacionPostgresAdapter,
)
from src.infrastructure.adapters.out_.sakt_pykt_adapter import SaktPyktAdapter
from src.infrastructure.adapters.out_.trazabilidad_rest_adapter import (
    TrazabilidadRestAdapter,
)
from src.infrastructure.adapters.out_.xai_rest_adapter import XaiRestAdapter
from src.infrastructure.adapters.out_.youtube_rest_adapter import YoutubeRestAdapter
from src.infrastructure.config.settings import settings
from src.infrastructure.db.database import get_session

# Dependencia de autenticación JWT reutilizable, compartida vía sward-shared.
require_jwt = build_require_jwt(settings.secret_key, algorithm=settings.jwt_algorithm)

# Validación entrante de service-key (modo dev permite sin claves configuradas).
require_service_key = build_require_service_key(settings.authorized_service_keys_set)


def get_modelo() -> ModeloKTPort:
    # Composition root: el entorno decide el adaptador (mock en development,
    # SAKT real con torch/pyKT/S3 en el resto), igual que antes.
    if settings.environment == "development":
        return ModeloKtMockAdapter()
    return SaktPyktAdapter()


def get_generar_uc(
    session: AsyncSession = Depends(get_session),
    modelo: ModeloKTPort = Depends(get_modelo),
) -> GenerarRecomendacionUseCase:
    return GenerarRecomendacionUseCase(
        trazabilidad=TrazabilidadRestAdapter(),
        cursos=CursosRestAdapter(),
        xai=XaiRestAdapter(),
        repo=RecomendacionPostgresAdapter(session),
        event_publisher=EventBridgeAdapter(),
        modelo=modelo,
        cache_ttl_s=settings.recomendacion_cache_ttl_s,
        max_conceptos_debiles=settings.max_conceptos_debiles,
        max_recomendaciones=settings.max_recomendaciones,
    )


def get_material_uc() -> GenerarMaterialUseCase:
    return GenerarMaterialUseCase(
        trazabilidad=TrazabilidadRestAdapter(),
        cursos=CursosRestAdapter(),
        llm=BedrockLlmAdapter(),
        youtube=YoutubeRestAdapter(),
        cache_ttl_s=settings.material_cache_ttl_s,
    )


def get_verificar_ejercicio_uc() -> VerificarEjercicioUseCase:
    return VerificarEjercicioUseCase(llm=BedrockLlmAdapter())


def get_atencion_uc(
    modelo: ModeloKTPort = Depends(get_modelo),
) -> ConsultarAtencionUseCase:
    return ConsultarAtencionUseCase(TrazabilidadRestAdapter(), modelo)


def get_modelo_info_uc(
    modelo: ModeloKTPort = Depends(get_modelo),
) -> ConsultarModeloInfoUseCase:
    return ConsultarModeloInfoUseCase(
        modelo, es_mock=settings.environment == "development"
    )


def get_consultar_uc(
    session: AsyncSession = Depends(get_session),
) -> ConsultarRecomendacionUseCase:
    return ConsultarRecomendacionUseCase(RecomendacionPostgresAdapter(session))


def get_completar_uc(
    session: AsyncSession = Depends(get_session),
) -> CompletarRecomendacionUseCase:
    return CompletarRecomendacionUseCase(RecomendacionPostgresAdapter(session))
