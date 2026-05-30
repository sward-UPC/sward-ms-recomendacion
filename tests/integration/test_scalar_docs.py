"""Tests de integración de la documentación de API (Scalar)."""

import pytest


@pytest.mark.asyncio
async def test_scalar_responde_200(anon_client):
    resp = await anon_client.get("/scalar")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_openapi_json_responde_200(anon_client):
    resp = await anon_client.get("/openapi.json")
    assert resp.status_code == 200
    body = resp.json()
    assert body["info"]["title"] == "SWARD — Microservicio de Recomendación Adaptativa"
