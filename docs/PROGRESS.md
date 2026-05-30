# PROGRESS — sward-ms-recomendacion

## Sprint 4 — 2026-05-30

### Implementado
- [x] Entidades: SecuenciaInteraccion, PrediccionKT, Recomendacion, ItemRecomendado
- [x] Value object: EstadoRecomendacion
- [x] Evento: RecomendacionGeneradaEvent
- [x] Domain Service: ModeloSAKT (mock en dev, carga desde S3 en prod)
- [x] Use Cases: GenerarRecomendacion (mín. 3 items, XAI no bloqueante), ConsultarRecomendacion
- [x] Adaptadores: TrazabilidadRestAdapter, CursosRestAdapter, XaiRestAdapter (mock en dev)
- [x] RecomendacionPostgresAdapter
- [x] EventBridgeAdapter
- [x] Endpoints: POST /recommendations/generate, GET /recommendations, PATCH /recommendations/{id}/complete
- [x] Docker Compose + Dockerfile
- [x] Tests: 8 tests (4 ModeloSAKT + 4 GenerarRecomendacion)
- [x] GitHub Actions CI

### Pendiente
- [ ] Fine-tuning SAKT con datos piloto reales
- [ ] pyKT/torch para inferencia en prod (no incluido en requirements.txt base)
