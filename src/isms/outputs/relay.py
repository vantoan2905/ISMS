"""Interface for a relay output."""
from typing import Protocol


class RelayController(Protocol):
    @property
    def is_on(self) -> bool:
        """Return the current relay state."""

    def set_state(self, on: bool) -> None:
        """Apply the requested relay state."""
