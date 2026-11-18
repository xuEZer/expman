"""Recording interfaces independent of pipeline execution."""

import logging
from dataclasses import dataclass, field
from typing import Protocol

from expman.core.events import Event

_logger = logging.getLogger(__name__)


class Recorder(Protocol):
    def record(self, event: Event) -> None:
        """Consume one event synchronously."""
        ...


def record_event(recorder: Recorder, event: Event) -> None:
    """Log sink failures while letting the experiment continue."""
    try:
        recorder.record(event)
    except Exception:
        _logger.warning("Recorder failed for run %s", event.run_id, exc_info=True)


@dataclass
class InMemoryRecorder:
    """Keep events in order for inspection; intended for small runs."""

    events: list[Event] = field(default_factory=list)

    def record(self, event: Event) -> None:
        self.events.append(event)
