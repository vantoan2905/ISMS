"""Object trackers and their registry."""

from . import tracker_base, tracker_config
from .registry import TRACKER_REGISTRY
from .tracker_base import (
    TrackAssignment,
    Tracker,
    MatchingTracker,
    default_tracker_options_by_camera,
    detections_for_tracking,
    tracker_provider,
)
from . import centroid_tracker, iou_tracker, passthrough_tracker, association_tracker, native_tracker, mask_tracker  # noqa: F401

__all__ = [
    "Tracker",
    "TrackAssignment",
    "MatchingTracker",
    "TRACKER_REGISTRY",
    "default_tracker_options_by_camera",
    "detections_for_tracking",
    "tracker_provider",
]

# Preserve existing import paths for callers using earlier module names.
import sys as _sys

for _old, _module in {
    "base": tracker_base,
    "config": tracker_config,
    "centroid": centroid_tracker,
    "iou": iou_tracker,
    "passthrough": passthrough_tracker,
    "botsort_hungarian": association_tracker,
    "botsort_native": native_tracker,
    "mask_ocsort": mask_tracker,
}.items():
    _sys.modules[f"{__name__}.{_old}"] = _module
