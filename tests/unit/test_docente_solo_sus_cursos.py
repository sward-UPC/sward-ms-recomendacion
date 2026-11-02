"""El docente solo ve a los estudiantes de los cursos que dicta.

`/recommendations/attention` entregaba el mapa de atención, la predicción y el
veredicto de fidelidad de cualquier estudiante en cualquier curso a todo rol
que no fuera «estudiante»: a cualquier docente, y también a un token sin rol o
con uno desconocido. Lo mismo pasaba al generar recomendaciones o material para
otro estudiante, que revela su concepto más débil y su dominio. La asignación
ya existía (``docente_id`` de cada curso en ms-cursos-recursos, la pone el
administrador) y el frontend ya filtraba el panel con ella; el servidor no la
miraba.

Ahora el docente tiene que ser el asignado al curso que pide, el administrador
sigue viendo todos, cualquier otro rol se queda con lo suyo y, si la asignación
no se puede consultar, no se entrega nada.
"""

from types import SimpleNamespace
from uuid import UUID

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.application.ports.out_.cursos_client_port import (
    AsignacionNoDisponibleError,
    CursosClientPort,
)
from src.domain.entities.recomendacion import ItemRecomendado, Recomendacion
from src.infrastructure.adapters.in_.main import app
from src.infrastructure.dependencies import (
    get_atencion_uc,
    get_consultar_uc,
    get_cursos_client,
    get_generar_uc,
    get_material_uc,
    require_jwt,
    token_del_usuario,
)

DOCENTE = "11111111-1111-1111-1111-111111111111"
OTRO_DOCENTE = "44444444-4444-4444-4444-444444444444"
ADMIN = "22222222-2222-2222-2222-222222222222"
ESTUDIANTE = "33333333-3333-3333-3333-333333333333"

CURSO_PROPIO = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
CURSO_AJENO = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
CURSO_SIN_DOCENTE = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")

TOKEN = "token-de-la-sesion"
CONCEPTO_DEBIL = "Amortización francesa"


def _usuario(rol: str | None, sub: str) -> dict:
    usuario = {"sub": sub, "type": "access"}
    if rol is not None:
        usuario["rol"] = rol
    return usuario


class _CursosFake(CursosClientPort):
    """ms-cursos-recursos con la asignación que puso el administrador."""

    def __init__(self):
        self.asignaciones = {
            CURSO_PROPIO: UUID(DOCENTE),
            CURSO_AJENO: UUID(OTRO_DOCENTE),
            CURSO_SIN_DOCENTE: None,
        }
        self.consultas: list[tuple[UUID, str]] = []

    async def obtener_recursos_candidatos(self, curso_id, limit=10, seccion=None):
        return []

    async def docente_del_curso(self, curso_id: UUID, token: str) -> UUID | None:
        self.consultas.append((curso_id, token))
        return self.asignaciones.get(curso_id)


class _CursosCaido(_CursosFake):
    """ms-cursos-recursos sin responder: tiempo agotado."""

    async def docente_del_curso(self, curso_id: UUID, token: str) -> UUID | None:
        self.consultas.append((curso_id, token))
        raise AsignacionNoDisponibleError("tiempo agotado")


class _Uc:
    """Caso de uso de mentira: devuelve datos fijos y anota para quién."""

    def __init__(self, resultado):
        self.resultado = resultado
        self.llamadas: list = []

    async def execute(self, *args, **kwargs):
        self.llamadas.append((args, kwargs))
        return self.resultado


def _siempre(objeto):
    # Sin parámetros a propósito: FastAPI tomaría un `lambda uc=uc: uc` como
    # parámetro de consulta y entregaría una copia, no el objeto que se vigila.
    return lambda: objeto


class _Entorno:
    def __init__(self):
        self.usuario = _usuario("docente", DOCENTE)
        self.cursos: _CursosFake = _CursosFake()
        self.ucs = {
            get_generar_uc: _Uc(
                Recomendacion(
                    items=[
                        ItemRecomendado(
                            recurso_id="r1",
                            titulo="Repaso",
                            tipo="lectura",
                            score=0.9,
                            orden=1,
                            motivo=f"Refuerza {CONCEPTO_DEBIL}",
                        )
                    ]
                )
            ),
            get_material_uc: _Uc(
                SimpleNamespace(
                    disponible=True,
                    concepto=CONCEPTO_DEBIL,
                    recursos=[],
                    dominio=35,
                )
            ),
            get_atencion_uc: _Uc(
                SimpleNamespace(
                    probabilidad_dominio=0.35,
                    puntos=[
                        SimpleNamespace(
                            concepto=CONCEPTO_DEBIL,
                            acierto=False,
                            peso=0.8,
                            suficiente=False,
                        )
                    ],
                    fidelidad=None,
                    fuente="modelo",
                )
            ),
            get_consultar_uc: _Uc([Recomendacion()]),
        }
        self.client: AsyncClient | None = None

    def como(self, rol: str | None, sub: str) -> None:
        self.usuario = _usuario(rol, sub)

    def se_entregaron_datos(self) -> bool:
        return any(uc.llamadas for uc in self.ucs.values())

    def para_quien_se_pidio(self) -> UUID:
        (args, kwargs) = next(uc.llamadas[0] for uc in self.ucs.values() if uc.llamadas)
        if args and isinstance(args[0], UUID):  # /attention: execute(est, curso)
            return args[0]
        return args[0].estudiante_id  # /generate y /material: un comando


@pytest_asyncio.fixture
async def entorno():
    e = _Entorno()
    app.dependency_overrides[require_jwt] = lambda: e.usuario
    app.dependency_overrides[token_del_usuario] = lambda: TOKEN
    app.dependency_overrides[get_cursos_client] = lambda: e.cursos
    for dependencia, uc in e.ucs.items():
        app.dependency_overrides[dependencia] = _siempre(uc)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        e.client = client
        yield e
    app.dependency_overrides.clear()


# Rutas en las que se indica un estudiante y un curso.
RUTAS_CON_CURSO = ["attention", "generate", "material"]


async def _pedir(e: _Entorno, ruta: str, curso: UUID, estudiante: str = ESTUDIANTE):
    if ruta == "attention":
        return await e.client.get(
            "/recommendations/attention",
            params={"estudianteId": estudiante, "cursoId": str(curso)},
        )
    return await e.client.post(
        f"/recommendations/{ruta}",
        json={"estudianteId": estudiante, "cursoId": str(curso)},
    )


@pytest.mark.parametrize("ruta", RUTAS_CON_CURSO)
async def test_docente_no_ve_a_un_estudiante_de_un_curso_que_no_dicta(entorno, ruta):
    resp = await _pedir(entorno, ruta, CURSO_AJENO)

    assert resp.status_code == 403, resp.text
    assert CONCEPTO_DEBIL not in resp.text
    assert not entorno.se_entregaron_datos()


@pytest.mark.parametrize("ruta", RUTAS_CON_CURSO)
async def test_docente_ve_a_un_estudiante_del_curso_que_dicta(entorno, ruta):
    resp = await _pedir(entorno, ruta, CURSO_PROPIO)

    assert resp.status_code in (200, 201), resp.text
    assert entorno.para_quien_se_pidio() == UUID(ESTUDIANTE)


@pytest.mark.parametrize("ruta", RUTAS_CON_CURSO)
async def test_administrador_ve_cualquier_curso(entorno, ruta):
    # Con ms-cursos-recursos caído a propósito: el administrador no depende de
    # la asignación, así que tampoco de que se pueda consultar.
    entorno.como("administrador", ADMIN)
    entorno.cursos = _CursosCaido()

    resp = await _pedir(entorno, ruta, CURSO_AJENO)

    assert resp.status_code in (200, 201), resp.text
    assert entorno.para_quien_se_pidio() == UUID(ESTUDIANTE)
    assert entorno.cursos.consultas == []


@pytest.mark.parametrize("ruta", RUTAS_CON_CURSO)
async def test_sin_poder_consultar_la_asignacion_no_se_entregan_datos(entorno, ruta):
    entorno.cursos = _CursosCaido()

    resp = await _pedir(entorno, ruta, CURSO_PROPIO)

    assert resp.status_code == 503, resp.text
    assert "curso" in resp.json()["detail"]
    assert not entorno.se_entregaron_datos()


async def test_un_curso_sin_docente_asignado_no_es_de_ningun_docente(entorno):
    resp = await _pedir(entorno, "attention", CURSO_SIN_DOCENTE)

    assert resp.status_code == 403
    assert not entorno.se_entregaron_datos()


async def test_se_consulta_la_asignacion_con_el_token_de_quien_pregunta(entorno):
    await _pedir(entorno, "attention", CURSO_PROPIO)

    assert entorno.cursos.consultas == [(CURSO_PROPIO, TOKEN)]


@pytest.mark.parametrize("rol", [None, "invitado"])
@pytest.mark.parametrize("ruta", RUTAS_CON_CURSO)
async def test_un_rol_desconocido_se_queda_con_lo_suyo(entorno, ruta, rol):
    # Antes bastaba con no ser «estudiante» para ver a cualquiera.
    yo = "55555555-5555-5555-5555-555555555555"
    entorno.como(rol, yo)

    resp = await _pedir(entorno, ruta, CURSO_AJENO)

    assert resp.status_code in (200, 201), resp.text
    assert entorno.para_quien_se_pidio() == UUID(yo)
    assert entorno.cursos.consultas == []


@pytest.mark.parametrize("ruta", RUTAS_CON_CURSO)
async def test_el_estudiante_sigue_viendo_solo_lo_suyo(entorno, ruta):
    entorno.como("estudiante", ESTUDIANTE)

    resp = await _pedir(entorno, ruta, CURSO_AJENO, estudiante=OTRO_DOCENTE)

    assert resp.status_code in (200, 201), resp.text
    assert entorno.para_quien_se_pidio() == UUID(ESTUDIANTE)
    assert entorno.cursos.consultas == []


@pytest.mark.parametrize("rol", ["estudiante", "docente"])
async def test_un_token_sin_un_sub_valido_no_ve_a_nadie(entorno, rol):
    # Sin identidad no hay «lo suyo» que entregar, ni un docente que comparar
    # con el asignado al curso.
    entorno.como(rol, "no-es-un-uuid")

    resp = await _pedir(entorno, "attention", CURSO_PROPIO)

    assert resp.status_code == 403
    assert not entorno.se_entregaron_datos()


async def test_docente_ve_lo_suyo_sin_consultar_la_asignacion(entorno):
    resp = await _pedir(entorno, "attention", CURSO_AJENO, estudiante=DOCENTE)

    assert resp.status_code == 200
    assert entorno.para_quien_se_pidio() == UUID(DOCENTE)
    assert entorno.cursos.consultas == []


async def test_docente_no_lista_las_recomendaciones_de_otro(entorno):
    # El listado no lleva curso: no hay una asignación que comprobar.
    resp = await entorno.client.get(
        "/recommendations", params={"estudianteId": ESTUDIANTE}
    )

    assert resp.status_code == 403
    assert not entorno.se_entregaron_datos()


async def test_administrador_lista_las_recomendaciones_de_cualquiera(entorno):
    entorno.como("administrador", ADMIN)

    resp = await entorno.client.get(
        "/recommendations", params={"estudianteId": ESTUDIANTE}
    )

    assert resp.status_code == 200
    (cmd,), _ = entorno.ucs[get_consultar_uc].llamadas[0]
    assert cmd.estudiante_id == UUID(ESTUDIANTE)


async def test_el_estudiante_lista_solo_las_suyas(entorno):
    entorno.como("estudiante", ESTUDIANTE)

    resp = await entorno.client.get(
        "/recommendations", params={"estudianteId": OTRO_DOCENTE}
    )

    assert resp.status_code == 200
    (cmd,), _ = entorno.ucs[get_consultar_uc].llamadas[0]
    assert cmd.estudiante_id == UUID(ESTUDIANTE)
