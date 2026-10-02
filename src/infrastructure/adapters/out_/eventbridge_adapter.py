import logging

from sward_shared.events.domain_event import DomainEvent

from src.application.ports.out_.event_publisher_port import EventPublisherPort
from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


class EventBridgeAdapter(EventPublisherPort):
    def publish(self, event: DomainEvent) -> None:
        if settings.environment == "development":
            logger.info("DEV — evento: %s", event.event_type)
            return
        from sward_shared.adapters.eventbridge import EventBridgeAdapter as Shared

        try:
            Shared(
                event_bus_name=settings.eventbridge_bus_name,
                source="sward-ms-recomendacion",
                region=settings.aws_region,
            ).publish(event)
        except Exception:
            # Publicar es telemetría: alimenta a las lambdas de alertas y de
            # notificaciones. Que falle no puede tumbar la operación que lo
            # provocó, porque el evento se publica DESPUÉS de haber hecho el
            # trabajo. El 1 de octubre de 2026 a ms-xai le faltaba el permiso
            # events:PutEvents: la excepción subió sin capturar, POST
            # /xai/explain respondió 500, y la explicación —ya generada y
            # guardada— se perdió al revertirse la sesión.
            #
            # Se registra con traza completa: esto se sigue adelante, pero no
            # se calla nunca.
            logger.exception(
                "no se pudo publicar el evento %s (id=%s); la operación continúa",
                event.event_type,
                event.event_id,
            )
