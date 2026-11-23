"""Observations exchanged between algorithms and the event engine."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    type: str
    track_id: int
    rule_id: str
    pts_ns: int
    confidence: float = 1.0

