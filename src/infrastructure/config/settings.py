from typing import Literal

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
    # Formato de entrada del checkpoint. Vacío = se lee del propio checkpoint
    # (recomendado). Solo para forzarlo: "relleno_izquierda" | "relleno_derecha".
    sakt_formato_entrada: str = ""
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
    # ── Verificación de fidelidad de la explicación (ERASER en línea) ──────
    # Contrasta, por cada predicción, la atención contra perturbaciones al azar.
    # Si no la supera, el sistema se abstiene de dar un motivo en lugar de
    # inventarlo. Apagada, el comportamiento es el anterior (condición de control).
    xai_verificacion_activa: bool = True
    # Qué prueba decide si se muestra un motivo:
    #  - "suficiencia": conservar solo lo más atendido mantiene la predicción
    #    mejor que conservar algo al azar. Con el formato de entrada correcto es
    #    la propiedad que la atención SÍ cumple (k=1: p < 0.0001, r = +0.44).
    #  - "exhaustividad": quitar lo más atendido la cambia más que quitar algo al
    #    azar. La atención NO la cumple de forma sistemática (k=3: p = 0.65).
    # Ambas se calculan siempre; el contrafactual solo se ofrece si pasa la
    # exhaustividad, porque es una afirmación de necesidad.
    # Literal: un valor inválido hace fallar el arranque, en vez de degradar
    # en silencio a predicciones mock.
    xai_criterio_fidelidad: Literal["suficiencia", "exhaustividad"] = "suficiencia"
    # k de cada prueba: el valor con mayor tamaño de efecto en la evaluación
    # offline corregida para suficiencia, y el habitual de ERASER para exhaustividad.
    xai_k_suficiencia: int = 1
    xai_k_exhaustividad: int = 3
    # Sorteos aleatorios de contraste. Es el denominador de la confianza, así que
    # fija su resolución: con 20, la confianza avanza de 0.05 en 0.05. Todos se
    # resuelven en UNA sola pasada por lotes, así que subirlo casi no cuesta.
    xai_n_aleatorios: int = 20
    # Fracción de sorteos que la atención debe ganar para que una prueba pase.
    # 0.8 equivale a un contraste de permutación de una cola con p ≤ 0.2.
    xai_umbral_confianza: float = 0.8
    # Por debajo de esto la secuencia es demasiado corta para que la atención
    # signifique algo: es justo el régimen donde la medición offline cayó por
    # debajo del azar.
    xai_min_interacciones: int = 5

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
