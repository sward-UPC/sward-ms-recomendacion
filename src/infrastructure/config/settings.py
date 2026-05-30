from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SECRET_KEY = "dev-secret-change-in-production"


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

    # Autenticación JWT (token emitido por sward-ms-usuarios, HS256).
    secret_key: str = DEFAULT_SECRET_KEY
    jwt_algorithm: str = "HS256"
    # Clave propia que este servicio envía como X-Service-Key en llamadas salientes.
    service_key: str = ""
    # Claves de servicio entrantes autorizadas, separadas por coma.
    authorized_service_keys: str = ""

    @property
    def authorized_service_keys_set(self) -> set[str]:
        return {k.strip() for k in self.authorized_service_keys.split(",") if k.strip()}

    @model_validator(mode="after")
    def _validar_secreto_en_produccion(self) -> "Settings":
        if self.environment != "development" and self.secret_key == DEFAULT_SECRET_KEY:
            raise ValueError(
                "SECRET_KEY no puede ser el valor por defecto fuera de desarrollo."
            )
        return self


settings = Settings()
