# sward-ms-recomendacion

Microservicio de **recomendación adaptativa e IA** de la plataforma **SWARD**.

Estima el estado de conocimiento de cada estudiante con un modelo de *knowledge
tracing* **SAKT** (Self-Attentive Knowledge Tracing, sobre **pyKT**) y, a partir
de él, recomienda recursos del curso priorizando el **formato en el que el alumno
rinde mejor**. Además **genera material de estudio bajo demanda** con un LLM
(**AWS Bedrock**) y videos reales de **YouTube**, y expone los **pesos de atención**
del modelo como capa de explicabilidad (**XAI**).

---

## Qué hace

- **Recomendación adaptativa (SAKT).** Toma la secuencia de interacciones del
  estudiante (concepto, acierto) desde ms-trazabilidad, infiere el dominio por
  concepto, detecta los conceptos más débiles y rankea recursos del curso
  (estudiar + practicar) con un score que pondera dificultad y la **preferencia
  de formato** del alumno (práctica vs. estudio).
- **Material de estudio generado (LLM + YouTube).** Para el concepto débil, pide a
  Bedrock un **quiz**, una **mini-lección con flashcards** y una **práctica con
  ejercicios** (JSON tipado) y le adjunta un **video real** de YouTube. Enfatiza
  el formato preferido del estudiante. Todo *best-effort*: si el LLM o la clave no
  están, devuelve material no disponible en lugar de fallar.
- **Verificación de ejercicios con IA.** Evalúa con el LLM la respuesta escrita
  por el estudiante y devuelve aprobado/feedback.
- **Explicabilidad (XAI).** Expone los **pesos de atención reales** del SAKT
  (a qué interacciones pasadas atendió el modelo al predecir) como un *heatmap*.
- **Metadata del modelo (s2s).** Endpoint interno que lee del artefacto en S3 los
  hiperparámetros, AUC y fecha de entrenamiento para el panel admin.

---

## Stack

| Capa | Tecnología |
|------|-----------|
| Lenguaje / API | Python 3.11 · FastAPI · Pydantic v2 |
| Modelo KT | **pyKT** (SAKT) · **PyTorch** (inferencia en CPU) |
| Generación | **AWS Bedrock** (Converse API, Claude) · **YouTube Data API v3** |
| Persistencia | SQLAlchemy 2.0 (async) · PostgreSQL (asyncpg) · Alembic |
| Mensajería | AWS EventBridge (vía `sward-shared`) |
| Almacenamiento de modelo | AWS S3 (checkpoint `.pth`) |
| Auth | JWT HS256 (emitido por ms-usuarios) + X-Service-Key (s2s) |
| Compartido | `sward-shared` (eventos de dominio, adaptadores, auth) |

El SAKT (mock en desarrollo, real en producción) decide **qué** conceptos
reforzar; la preferencia de formato decide **con qué** formato; el LLM produce el
contenido cuando no hay un recurso del curso adecuado.

---

## Estructura (arquitectura hexagonal)

Ports & Adapters con nomenclatura `domain / application / infrastructure` y
puertos de entrada (`in_`) / salida (`out_`).

```
src/
├── domain/                                  # NÚCLEO (sin frameworks*)
│   ├── entities/
│   │   ├── secuencia_interaccion.py         # SecuenciaInteraccion (concepto, acierto)
│   │   ├── prediccion_kt.py                 # PrediccionKT (dominio + pesos_atencion)
│   │   └── recomendacion.py                 # Recomendacion, ItemRecomendado
│   ├── value_objects/
│   │   └── estado_recomendacion.py          # EstadoRecomendacion (PENDIENTE/COMPLETADO)
│   ├── events/
│   │   └── recomendacion_generada_event.py  # RecomendacionGeneradaEvent
│   ├── services/
│   │   └── modelo_sakt.py                   # ModeloSAKT (mock/real S3) + leer_info_modelo
│   └── ports/out_/                          # contratos (ABC) que el núcleo necesita
│       ├── trazabilidad_client_port.py
│       ├── cursos_client_port.py
│       ├── llm_client_port.py
│       ├── youtube_client_port.py
│       ├── xai_client_port.py
│       ├── recomendacion_repository_port.py
│       └── event_publisher_port.py
│
├── application/use_cases/                   # casos de uso (orquestación)
│   ├── generar_recomendacion.py             # SAKT + ranking por formato
│   ├── generar_material.py                  # LLM (Bedrock) + YouTube
│   ├── verificar_ejercicio.py               # evaluación con LLM
│   ├── consultar_atencion.py                # heatmap XAI
│   ├── consultar_recomendacion.py
│   └── completar_recomendacion.py
│
└── infrastructure/
    ├── adapters/
    │   ├── in_/                             # adaptadores de ENTRADA (driving)
    │   │   ├── main.py                      # app FastAPI + lifespan + CORS + handlers
    │   │   └── recomendacion_router.py      # endpoints + schemas Pydantic
    │   └── out_/                            # adaptadores de SALIDA (driven)
    │       ├── trazabilidad_rest_adapter.py # secuencia + preferencias (httpx)
    │       ├── cursos_rest_adapter.py       # recursos candidatos del curso
    │       ├── bedrock_llm_adapter.py       # AWS Bedrock (Converse)
    │       ├── youtube_rest_adapter.py      # YouTube Data API v3
    │       ├── xai_rest_adapter.py          # explicaciones XAI
    │       ├── recomendacion_postgres_adapter.py
    │       └── eventbridge_adapter.py       # publica eventos de dominio
    ├── config/settings.py                   # configuración (pydantic-settings)
    ├── db/                                   # engine, sesión, modelos ORM
    └── dependencies.py                      # composition root (cablea puertos↔adaptadores)

training/
├── train_sakt.py                            # entrenamiento del SAKT (CI → S3)
└── train_sakt_formato.py                    # variante experimental concepto×formato

docs/
└── sakt_formato_feature.md                  # diseño del feature de formato (experimental)
```

> \* **Nota de arquitectura:** `domain/services/modelo_sakt.py` importa
> `torch`/`pykt`/`boto3` y la config de infraestructura dentro de sus funciones,
> lo que rompe la regla de dependencia del núcleo. Está documentado y razonado en
> [`AUDIT_HEXAGONAL.md`](AUDIT_HEXAGONAL.md) con la recomendación de convertirlo
> en un puerto + adaptador de infraestructura.

---

## Endpoints

Prefijo: `/recommendations`. Todos requieren **JWT** salvo el interno (service-key).
Documentación interactiva en `/scalar` (OpenAPI en
`/recommendations/openapi.json`).

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/recommendations/generate` | Genera recomendaciones adaptativas (SAKT + ranking por formato). |
| `POST` | `/recommendations/material` | Genera material de estudio del concepto débil (quiz, lectura+flashcards, práctica, video). |
| `POST` | `/recommendations/verify-exercise` | Evalúa con el LLM la respuesta del alumno a un ejercicio. |
| `GET`  | `/recommendations` | Lista las recomendaciones del estudiante. |
| `PATCH`| `/recommendations/{id}/complete` | Marca una recomendación como completada. |
| `GET`  | `/recommendations/attention` | **XAI**: pesos de atención del SAKT sobre las interacciones del estudiante. |
| `GET`  | `/recommendations/internal/model-info` | **s2s** (service-key): metadata real del modelo (hiperparámetros, AUC, fecha) leída de S3. Oculto del schema. |
| `GET`  | `/health` | Sonda de estado (indica si corre el modelo mock). |
| `GET`  | `/scalar` | Referencia de API interactiva. |

**Autorización por rol:** un `estudiante` solo opera sobre sus propios datos (el
UUID se toma del JWT); `docente`/`admin` pueden indicar el `estudianteId` en el
body/query.

---

## El modelo SAKT: entrenamiento y recarga

El servicio **no entrena** en línea: carga un checkpoint preentrenado desde S3.

### Inferencia (runtime)

- En **desarrollo** (`ENVIRONMENT=development`) corre un **mock** determinista
  (promedio de aciertos), sin torch ni S3.
- En **producción** descarga el checkpoint de S3
  (`s3://<AWS_S3_MODEL_BUCKET>/<SAKT_MODEL_S3_KEY>`), reconstruye la red SAKT de
  pyKT y, mediante un *monkey-patch* del bloque de atención, **captura los pesos
  de atención** que pyKT normalmente descarta (eso alimenta el endpoint
  `/attention`). Si la carga falla, hace *fallback* al mock sin tumbar el servicio.

### Entrenamiento (offline, automatizado)

`training/train_sakt.py` entrena el SAKT en CPU y exporta un checkpoint
compatible con el loader (`n_skills`, `seq_len`, `emb_size`, `n_heads`, `dropout`,
`n_layers`, `concept_index`, `model_state_dict`, además de `test_auc`/`trained_at`
para la metadata). El formato de secuencia `q/r/qry` es **idéntico** al de
inferencia, para que entrenamiento e inferencia no diverjan.

Lo orquesta el workflow **`.github/workflows/train-sakt.yml`** (`Entrenar modelo
SAKT`), que corre **semanalmente (lunes 06:00 UTC)** y **a demanda**:

1. (En corridas programadas) enciende la infra mínima (RDS + ms-trazabilidad), que
   en dev se apaga por las noches.
2. Descarga las **interacciones reales** desde ms-trazabilidad
   (`GET /api/v1/dashboard/training-data`, con X-Service-Key).
3. Ejecuta `python training/train_sakt.py` → `model.pth`.
4. Sube el checkpoint a `s3://sward-models/sakt/moodle/model.pth`.
5. Fuerza un **redeploy de ms-recomendacion** (`ecs update-service
   --force-new-deployment`) para que el servicio recargue el modelo al arrancar.
6. (En corridas programadas) vuelve a apagar la infra.

> **Variante experimental (`train_sakt_formato.py`).** Redefine la unidad de
> conocimiento como el par compuesto **(concepto × formato)** para que el SAKT
> distinga, p.ej., "AVL practicado" de "AVL leído". Está aislada de producción y
> documentada en [`docs/sakt_formato_feature.md`](docs/sakt_formato_feature.md);
> queda bloqueada por la inclusión de `tipo_recurso` en el export de datos.

---

## Variables de entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | `development` → mock SAKT + adaptadores en dev. Otro valor → modo real. |
| `DATABASE_URL` | `postgresql+asyncpg://…/recomendacion_db` | Cadena async de PostgreSQL. |
| `DB_USERNAME` / `DB_PASSWORD` / `DATABASE_HOST` / `DATABASE_PORT` / `DATABASE_NAME` | — | Componentes inyectados por la task def de ECS (Secrets Manager); si están, componen `DATABASE_URL`. |
| `AWS_REGION` | `us-east-1` | Región AWS (S3, Bedrock, EventBridge). |
| `AWS_S3_MODEL_BUCKET` | `sward-models` | Bucket del checkpoint SAKT. |
| `SAKT_MODEL_S3_KEY` | `sakt/moodle/model.pth` | Key del checkpoint en S3. |
| `BEDROCK_MODEL_ID` | `us.anthropic.claude-haiku-4-5-…` | Modelo de Bedrock para generar material. |
| `BEDROCK_MAX_TOKENS` | `4096` | Tope de tokens (el JSON tipado es grande). |
| `YOUTUBE_API_KEY` | `""` | YouTube Data API v3 (best-effort: sin clave, sin video). |
| `TRAZABILIDAD_SERVICE_URL` | `http://localhost:8003` | URL de ms-trazabilidad. |
| `CURSOS_SERVICE_URL` | `http://localhost:8004` | URL de ms-cursos. |
| `XAI_SERVICE_URL` | `http://localhost:8006` | URL del servicio XAI. |
| `EVENTBRIDGE_BUS_NAME` | `sward-event-bus` | Event bus de EventBridge. |
| `MAX_RECOMENDACIONES` | `6` | Tope de ítems por recomendación. |
| `MAX_CONCEPTOS_DEBILES` | `3` | Conceptos débiles a cubrir. |
| `MATERIAL_CACHE_TTL_S` | `21600` | TTL del cache de material (6 h). |
| `RECOMENDACION_CACHE_TTL_S` | `1800` | TTL del cache de recomendación (30 min). |
| `SECRET_KEY` | `dev-secret-…` | Secreto JWT (HS256). **Obligatorio cambiarlo fuera de dev.** |
| `JWT_ALGORITHM` | `HS256` | Algoritmo del JWT. |
| `SERVICE_KEY` | `""` | X-Service-Key que este servicio envía en llamadas salientes. |
| `AUTHORIZED_SERVICE_KEYS` | `""` | Claves entrantes autorizadas (s2s), separadas por coma. |
| `CORS_ALLOWED_ORIGINS` | `["http://localhost:5173"]` | Orígenes CORS permitidos. |

---

## Correr en local

```bash
cp .env.example .env

# Opción A: solo la base y el servicio con uvicorn (recarga en caliente)
docker compose up -d db
uvicorn src.infrastructure.adapters.in_.main:app --reload --port 8005

# Opción B: todo con docker-compose (db + app en :8005)
docker compose up --build
```

En modo desarrollo el SAKT y los adaptadores externos usan **mocks**, por lo que no
hacen falta AWS ni claves para arrancar. Las tablas se crean al iniciar (lifespan).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q                 # 33 tests (unit + integración)
ruff check                # linting
```

Los tests unitarios cubren el SAKT (mock), la generación de recomendaciones y de
material, y la variante de formato; los de integración levantan la app FastAPI con
adaptadores fake. **No requieren torch/pyKT** (solo el modo real lo hace).

---

## Despliegue

Flujo CI/CD basado en los workflows de la org `sward-UPC`:

1. **CI** (`.github/workflows/ci.yml`) — en push/PR a `main`: reutiliza el workflow
   compartido `ci-microservice.yml` (tests + lint, con `sward-shared`).
2. **Build & Push** (`.github/workflows/build-push.yml`) — en push a `deploy`:
   construye la imagen y la publica en **GHCR**, luego dispara el redeploy del
   servicio `recomendacion` en el cluster ECS `sward-cluster`.
3. **Runtime** — el contenedor corre `uvicorn` (puerto 8000) en **ECS/Fargate**
   detrás del ALB; la configuración real (BD, secretos, URLs s2s) llega por la
   task definition (CDK + Secrets Manager).
4. **Reentrenamiento del modelo** (`train-sakt.yml`) — independiente del deploy de
   código: reentrena, sube el `.pth` a S3 y fuerza el redeploy para recargar.

---

## Proyecto

**TP202610051** — Universidad Peruana de Ciencias Aplicadas (UPC) · Taller de
Proyecto · 2026.
