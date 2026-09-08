"""Recording interfaces independent of pipeline execution."""

from dataclasses import dataclass, field
from typing import Protocol

from .events import Event


class Recorder(Protocol):
    def record(self, event: Event) -> None:
        """Consume one event synchronously."""
        ...


@dataclass
class InMemoryRecorder:
    """Keep events in order for inspection; intended for small runs."""

    events: list[Event] = field(default_factory=list)

    def record(self, event: Event) -> None:
        self.events.append(event)
