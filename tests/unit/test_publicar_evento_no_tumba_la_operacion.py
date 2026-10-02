"""Publicar un evento no puede tumbar la operación que lo provocó.

EventBridge alimenta a las lambdas de alertas y de notificaciones, y el evento
se publica DESPUÉS de haber hecho el trabajo. Aun así, la llamada iba sin
proteger en los seis microservicios: cualquier fallo al publicar subía como
excepción y se llevaba por delante la petición del usuario.

Pasó de verdad. El 1 de octubre de 2026 a ms-xai le faltaba el permiso
events:PutEvents sobre el bus; boto3 lanzaba AccessDeniedException, POST
/xai/explain respondía 500 y la explicación, que ya estaba generada y guardada,
se perdía al revertirse la sesión de base de datos. La tabla de explicaciones
estuvo en cero mientras los estudiantes del estudio usaban el sistema.
"""

import logging

import pytest

from src.infrastructure.adapters.out_ import eventbridge_adapter as modulo
from src.infrastructure.adapters.out_.eventbridge_adapter import EventBridgeAdapter


class _EventoFalso:
    event_type = "PruebaEvento"
    event_id = "11111111-1111-1111-1111-111111111111"


class _PublicadorQueFalla:
    def __init__(self, **kwargs):
        pass

    def publish(self, event):
        raise RuntimeError("EventBridge no está disponible")


@pytest.fixture
def publicador_roto(monkeypatch):
    """Sustituye el publicador compartido por uno que siempre falla."""
    import sward_shared.adapters.eventbridge as compartido

    monkeypatch.setattr(compartido, "EventBridgeAdapter", _PublicadorQueFalla)
    monkeypatch.setattr(modulo.settings, "environment", "production")


def test_un_fallo_al_publicar_no_sube(publicador_roto):
    # Si esto lanza, la operación del usuario se cae por un evento de telemetría.
    EventBridgeAdapter().publish(_EventoFalso())


def test_el_fallo_queda_registrado_con_su_traza(publicador_roto, caplog):
    # Blindar no es callar: el fallo tiene que quedar en el registro, con traza,
    # o nadie se entera de que los eventos dejaron de salir.
    with caplog.at_level(logging.ERROR):
        EventBridgeAdapter().publish(_EventoFalso())

    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert any("PruebaEvento" in r.getMessage() for r in caplog.records)
    assert any(r.exc_info for r in caplog.records), "falta la traza"
