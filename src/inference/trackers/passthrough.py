"""Forward valid upstream object IDs."""

from __future__ import annotations

from .config import is_valid_object_id
from .registry import TRACKER_REGISTRY
from .base import TrackAssignment, Tracker


@TRACKER_REGISTRY.register("passthrough")
class PassthroughTracker(Tracker):
    """Assign the upstream object ID when available."""

    def __init__(self, **_ignored):
        pass

    def update(self, source_id, frame_number, detections):
        assignments = []
        for detection_index, detection in enumerate(detections):
            detection.track_id = (
                int(detection.object_id) if is_valid_object_id(detection.object_id) else None
            )
            if detection.track_id is not None:
                assignments.append(
                    TrackAssignment(
                        detection_index=detection_index,
                        track_id=detection.track_id,
                        association_score=1.0,
                        state="confirmed",
                    )
                )
        return assignments
