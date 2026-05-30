from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.infrastructure.adapters.in_.recomendacion_router import router
from src.infrastructure.config.settings import settings
from src.infrastructure.db.database import engine
from src.infrastructure.db.models.recomendacion_models import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(
    title="SWARD — Recomendación Adaptativa", version="0.1.0", lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.include_router(router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": settings.service_name,
        "mock_model": settings.environment == "development",
    }
