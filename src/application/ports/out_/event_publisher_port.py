from abc import ABC, abstractmethod

from sward_shared.events.domain_event import DomainEvent


class EventPublisherPort(ABC):
    @abstractmethod
    def publish(self, event: DomainEvent) -> None: ...
