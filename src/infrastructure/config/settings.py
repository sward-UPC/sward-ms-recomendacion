from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SECRET_KEY = "dev-secret-change-in-production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    database_url: str = (
        "postgresql+asyncpg://sward:sward@localhost:5432/recomendacion_db"
    )
    # Componentes inyectados por ECS task definition (CDK via Secrets Manager).
    db_username: str = ""
    db_password: str = ""
    database_host: str = ""
    database_port: str = "5432"
    database_name: str = ""

    aws_region: str = "us-east-1"
    aws_s3_model_bucket: str = "sward-models"
    # Modelo SAKT entrenado sobre conceptos de Moodle (secciones). El checkpoint
    # trae n_skills + concept_index, así que el servicio se adapta solo.
    sakt_model_s3_key: str = "sakt/moodle/model.pth"
    trazabilidad_service_url: str = "http://localhost:8003"
    cursos_service_url: str = "http://localhost:8004"
    xai_service_url: str = "http://localhost:8006"
    eventbridge_bus_name: str = "sward-event-bus"
    environment: str = "development"
    service_name: str = "sward-ms-recomendacion"
    min_recomendaciones: int = 3
    max_recomendaciones: int = 6
    # Cuántos conceptos débiles (secciones) cubre la recomendación SAKT. Antes solo
    # se targeteaba 1 → muy pocos items; con varios se cubre más, como el heurístico.
    max_conceptos_debiles: int = 3
    # Generación de material de estudio con LLM vía AWS Bedrock (usa el IAM del
    # task role, sin API key). El modelo debe estar habilitado en Bedrock.
    bedrock_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    # El material tipado (quiz + lectura + práctica en un JSON) es grande; con pocos
    # tokens se trunca y el JSON queda inválido → material no disponible. Haiku 4.5
    # admite respuestas largas; 8192 da margen para que el JSON cierre completo.
    # (Además el parser repara truncaciones, pero mejor no depender de eso.)
    bedrock_max_tokens: int = 8192
    # TTL del cache en memoria del material generado (segundos). Evita re-llamar a
    # Bedrock+YouTube en cada carga; el concepto débil cambia despacio. 6h por defecto.
    material_cache_ttl_s: int = 21600
    # TTL del cache de la recomendación SAKT (segundos). Evita re-inferir el modelo y
    # llamar a cursos en cada recarga. 30 min por defecto.
    recomendacion_cache_ttl_s: int = 1800
    # Clave de la YouTube Data API v3 para adjuntar un video real al material.
    # Best-effort: si está vacía, el material se genera sin el recurso de video.
    youtube_api_key: str = ""
    # Orígenes permitidos para CORS (configurables por entorno).
    cors_allowed_origins: list[str] = ["http://localhost:5173"]

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @model_validator(mode="after")
    def _compose_database_url(self) -> "Settings":
        if self.database_host and self.db_username:
            self.database_url = (
                f"postgresql+asyncpg://{self.db_username}:{self.db_password}"
                f"@{self.database_host}:{self.database_port}/{self.database_name}"
            )
        return self

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
