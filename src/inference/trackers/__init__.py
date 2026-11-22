"""Object trackers and their registry."""

from .registry import TRACKER_REGISTRY
from .base import (
    TrackAssignment,
    Tracker,
    MatchingTracker,
    default_tracker_options_by_camera,
    detections_for_tracking,
    tracker_provider,
)
from . import centroid, iou, passthrough, botsort_hungarian, botsort_hungarian_v2, mask_ocsort  # noqa: F401  (register on import)

__all__ = [
    "Tracker",
    "TrackAssignment",
    "MatchingTracker",
    "TRACKER_REGISTRY",
    "default_tracker_options_by_camera",
    "detections_for_tracking",
    "tracker_provider",
]
