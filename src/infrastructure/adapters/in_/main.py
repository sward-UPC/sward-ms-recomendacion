import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scalar_fastapi import get_scalar_api_reference

from src.application.use_cases.consultar_modelo_info import ModeloInfoNoDisponibleError
from src.infrastructure.adapters.in_.recomendacion_router import router
from src.infrastructure.config.settings import settings
from src.infrastructure.db.database import engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # El esquema lo gestiona Alembic (`alembic upgrade head` en el entrypoint del
    # contenedor); aquí solo liberamos el engine al apagar.
    yield
    await engine.dispose()


app = FastAPI(
    title="SWARD — Microservicio de Recomendación Adaptativa",
    version="0.1.0",
    openapi_url="/recommendations/openapi.json",
    description=(
        "Genera recomendaciones adaptativas de recursos y rutas de aprendizaje "
        "personalizadas para cada estudiante de la plataforma SWARD."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Recomendación",
            "description": "Generación y consulta de recomendaciones adaptativas de aprendizaje.",
        },
        {"name": "Health", "description": "Sonda de estado del servicio."},
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if not settings.is_development:
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
    return response


@app.exception_handler(KeyError)
async def not_found_handler(request: Request, exc: KeyError) -> JSONResponse:
    # Recurso inexistente (estudiante/curso/recomendación) señalado por la capa de
    # aplicación. str(KeyError) conserva las comillas, igual que la respuesta previa.
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)}
    )


@app.exception_handler(ValueError)
async def bad_request_handler(request: Request, exc: ValueError) -> JSONResponse:
    # Datos de entrada inválidos señalados por la capa de aplicación.
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)}
    )


@app.exception_handler(ModeloInfoNoDisponibleError)
async def modelo_info_no_disponible_handler(
    request: Request, exc: ModeloInfoNoDisponibleError
) -> JSONResponse:
    # No se pudo leer el artefacto del modelo desde S3 (endpoint interno model-info).
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Error no controlado en %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor."},
    )


app.include_router(router)


@app.get("/scalar", include_in_schema=False)
async def scalar_docs():
    """Renderiza la referencia de API interactiva (Scalar) del servicio."""
    return get_scalar_api_reference(openapi_url=app.openapi_url, title=app.title)


@app.get("/health", tags=["Health"], summary="Estado del servicio")
async def health():
    """Devuelve el estado de salud del microservicio y si usa el modelo mock."""
    return {
        "status": "ok",
        "service": settings.service_name,
        "mock_model": settings.environment == "development",
    }
