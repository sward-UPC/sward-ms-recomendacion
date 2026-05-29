# sward-ms-recomendacion

Microservicio de recomendación adaptativa del sistema **SWARD**.  
Implementa el modelo **Self-Attentive Knowledge Tracing (SAKT)** mediante la librería **pyKT** para estimar el estado de conocimiento del estudiante y generar recomendaciones personalizadas de recursos educativos.

## Arquitectura

Arquitectura **Hexagonal (Ports & Adapters)**:

```
src/
  domain/           # SecuenciaInteraccion, ModeloSAKT, PrediccionKT, Recomendacion, ItemRecomendado
  application/      # ObtenerEstadoConocimientoUseCase, GenerarRecomendacionUseCase
  infrastructure/   # FastAPI routers, RecomendacionPostgresAdapter, TrazabilidadRestAdapter, XaiRestAdapter
```

## Stack

- Python 3.11 · FastAPI · pyKT · PyTorch · SQLAlchemy 2.0 · PostgreSQL
- boto3 (S3 para pesos del modelo + EventBridge) · httpx · Pydantic v2

## Carga del modelo SAKT

Los pesos del modelo se descargan automáticamente desde S3 al iniciar el servicio:

```bash
# Variable de entorno requerida:
SAKT_MODEL_S3_KEY=sakt/v1.0/model.pth
AWS_S3_MODEL_BUCKET=sward-models
```

## Desarrollo local

```bash
cp .env.example .env
docker compose up -d db
alembic upgrade head
uvicorn src.infrastructure.adapters.in_.main:app --reload --port 8005
```

## Tests

```bash
pytest tests/ -v --cov=src
```

## Endpoints principales

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/recommendations/generate` | Generar recomendaciones adaptativas |
| GET | `/recommendations` | Consultar recomendaciones del estudiante |
| PATCH | `/recommendations/{id}/complete` | Marcar recurso como completado |

## Métricas del modelo

- AUC objetivo: ≥ 0.75
- Latencia de inferencia: < 500 ms

## Proyecto

**TP202610051** — Universidad Peruana de Ciencias Aplicadas (UPC)  
Taller de Proyecto 1 / 2026
