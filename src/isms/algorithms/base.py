"""Base contract for camera algorithms."""
from abc import ABC, abstractmethod
from ..core.frame_context import FrameContext


class Algorithm(ABC):
    @abstractmethod
    def process(self, context: FrameContext) -> None:
        """Update one frame context in place."""
