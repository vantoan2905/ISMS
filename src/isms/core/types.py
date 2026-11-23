"""Frame packet exchanged between media and application layers."""
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class Frame:
    frame_id: int
    pts_ns: int
    received_at_utc: datetime
    received_at_monotonic_ns: int
    image: np.ndarray

