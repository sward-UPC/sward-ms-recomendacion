from uuid import uuid4


async def test_generar_recomendacion_retorna_201_con_items(auth_client):
    resp = await auth_client.post(
        "/recommendations/generate",
        json={"estudianteId": str(uuid4()), "cursoId": str(uuid4())},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert len(body["items"]) >= 3


async def test_listar_recomendaciones_retorna_200(auth_client):
    resp = await auth_client.get(
        "/recommendations", params={"estudianteId": str(uuid4())}
    )
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_health_es_publico(anon_client):
    resp = await anon_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_endpoint_protegido_sin_token_retorna_401(anon_client):
    resp = await anon_client.get(
        "/recommendations", params={"estudianteId": str(uuid4())}
    )
    assert resp.status_code == 401
