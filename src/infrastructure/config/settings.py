from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    database_url: str = (
        "postgresql+asyncpg://sward:sward@localhost:5432/recomendacion_db"
    )
    aws_region: str = "us-east-1"
    aws_s3_model_bucket: str = "sward-models"
    sakt_model_s3_key: str = "sakt/v1.0/model.pth"
    trazabilidad_service_url: str = "http://localhost:8003"
    cursos_service_url: str = "http://localhost:8004"
    xai_service_url: str = "http://localhost:8006"
    eventbridge_bus_name: str = "sward-event-bus"
    environment: str = "development"
    service_name: str = "sward-ms-recomendacion"
    min_recomendaciones: int = 3


settings = Settings()
