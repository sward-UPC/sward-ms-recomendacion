"""Quién dicta un curso, preguntado a ms-cursos-recursos.

Lo importante es no confundir «no lo sé» con «no lo dicta» ni con «sí lo
dicta»: si la consulta falla, el adaptador lo dice con un error propio y quien
lo llama no entrega datos. Y no hay simulación en desarrollo, a diferencia de
los recursos candidatos.
"""

from uuid import UUID, uuid4

import httpx
import pytest

from src.application.ports.out_.cursos_client_port import AsignacionNoDisponibleError
from src.infrastructure.adapters.out_ import cursos_rest_adapter as modulo
from src.infrastructure.adapters.out_.cursos_rest_adapter import CursosRestAdapter

DOCENTE = UUID("11111111-1111-1111-1111-111111111111")


def _cliente(estado: int, cuerpo, capturado: dict, error: Exception | None = None):
    class _Respuesta:
        status_code = estado

        def json(self):
            if isinstance(cuerpo, Exception):
                raise cuerpo
            return cuerpo

    class _Cliente:
        def __init__(self, *args, **kwargs):
            capturado["timeout"] = kwargs.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, **kwargs):
            capturado["url"] = url
            capturado["headers"] = kwargs.get("headers", {})
            if error is not None:
                raise error
            return _Respuesta()

    return _Cliente


async def test_reenvia_el_token_del_usuario_a_la_ruta_del_curso(monkeypatch):
    capturado: dict = {}
    curso = uuid4()
    monkeypatch.setattr(modulo.settings, "cursos_service_url", "http://cursos:8000")
    monkeypatch.setattr(
        modulo.httpx,
        "AsyncClient",
        _cliente(200, {"docente_id": str(DOCENTE)}, capturado),
    )

    await CursosRestAdapter().docente_del_curso(curso, "abc.def.ghi")

    assert capturado["url"] == f"http://cursos:8000/courses/{curso}"
    assert capturado["headers"] == {"Authorization": "Bearer abc.def.ghi"}
    # Un tope corto: esta consulta va delante de cada petición del docente.
    assert capturado["timeout"] is not None


async def test_en_desarrollo_tambien_pregunta(monkeypatch):
    # Los recursos candidatos se simulan en desarrollo; la asignación no.
    capturado: dict = {}
    monkeypatch.setattr(modulo.settings, "environment", "development")
    monkeypatch.setattr(
        modulo.httpx,
        "AsyncClient",
        _cliente(200, {"docente_id": str(DOCENTE)}, capturado),
    )

    assert await CursosRestAdapter().docente_del_curso(uuid4(), "t") == DOCENTE
    assert "url" in capturado


async def test_curso_sin_docente_devuelve_none(monkeypatch):
    monkeypatch.setattr(
        modulo.httpx, "AsyncClient", _cliente(200, {"docente_id": None}, {})
    )

    assert await CursosRestAdapter().docente_del_curso(uuid4(), "t") is None


async def test_curso_que_no_existe_devuelve_none(monkeypatch):
    monkeypatch.setattr(
        modulo.httpx,
        "AsyncClient",
        _cliente(404, {"detail": "Curso no encontrado"}, {}),
    )

    assert await CursosRestAdapter().docente_del_curso(uuid4(), "t") is None


@pytest.mark.parametrize("estado", [401, 403, 500, 502, 503])
async def test_otra_respuesta_no_es_un_no(monkeypatch, estado):
    monkeypatch.setattr(modulo.httpx, "AsyncClient", _cliente(estado, {}, {}))

    with pytest.raises(AsignacionNoDisponibleError):
        await CursosRestAdapter().docente_del_curso(uuid4(), "t")


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("sin conexión"),
        httpx.ReadTimeout("tiempo agotado"),
    ],
)
async def test_servicio_caido_o_lento_es_no_disponible(monkeypatch, error):
    monkeypatch.setattr(modulo.httpx, "AsyncClient", _cliente(200, {}, {}, error=error))

    with pytest.raises(AsignacionNoDisponibleError):
        await CursosRestAdapter().docente_del_curso(uuid4(), "t")


@pytest.mark.parametrize(
    "cuerpo",
    [
        ValueError("no es JSON"),
        ["no", "es", "un", "curso"],
        {"docente_id": "no-es-un-uuid"},
    ],
)
async def test_respuesta_que_no_se_entiende_es_no_disponible(monkeypatch, cuerpo):
    monkeypatch.setattr(modulo.httpx, "AsyncClient", _cliente(200, cuerpo, {}))

    with pytest.raises(AsignacionNoDisponibleError):
        await CursosRestAdapter().docente_del_curso(uuid4(), "t")
