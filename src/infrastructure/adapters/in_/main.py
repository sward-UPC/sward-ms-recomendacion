import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scalar_fastapi import get_scalar_api_reference

from src.infrastructure.adapters.in_.recomendacion_router import router
from src.infrastructure.config.settings import settings
from src.infrastructure.db.database import engine
from src.infrastructure.db.models.recomendacion_models import Base


async def _init_db() -> None:
    for intento in range(10):
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Base de datos lista.")
            return
        except Exception as exc:
            logger.warning("BD no disponible (intento %d/10): %s", intento + 1, exc)
            await asyncio.sleep(5)
    logger.error("No se pudo conectar a la BD tras 10 intentos.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_init_db())
    yield
    await engine.dispose()


logger = logging.getLogger(__name__)

app = FastAPI(
    title="SWARD — Microservicio de Recomendación Adaptativa",
    version="0.1.0",
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
