"""Per-frame context shared by camera algorithms."""
from dataclasses import dataclass, field


@dataclass
class FrameContext:
    camera_id: str
    frame: object
    timestamp: float
    persons: list = field(default_factory=list)
    events: list = field(default_factory=list)
    frame_id: int = 0
    pts_ns: int = 0
    received_at_utc: object = None
    observations: list = field(default_factory=list)

