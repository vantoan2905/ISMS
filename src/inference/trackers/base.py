"""Tracker interface + a base for id-preferring, fallback-matching trackers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import (
    DEFAULT_FPS,
    DEFAULT_TRACKING_DISTANCE,
    DEFAULT_TRACKING_MAX_AGE,
    is_valid_object_id,
)


@dataclass(frozen=True)
class TrackAssignment:
    """Association for one detection in the sequence passed to ``update``."""

    detection_index: int
    track_id: int
    association_score: float
    state: Literal["tentative", "confirmed", "lost"]
    lineage_id: int | None = None
    predecessor_track_id: int | None = None
    association_reason: str | None = None


class Tracker:
    """Update detections in place and return their indexed track assignments."""

    wants_low_confidence_detections = False

    def update(self, source_id, frame_number, detections):
        raise NotImplementedError

    def reset(self, source_id=None):
        pass


def tracker_provider(name):
    """Return the one authoritative ID provider for a tracker stage name."""
    return "upstream" if str(name) == "passthrough" else "app"


def default_tracker_options_by_camera(name=None):
    """Return camera overrides supplied by the selected tracker."""
    from .registry import TRACKER_REGISTRY

    if name and TRACKER_REGISTRY.has(str(name)):
        hook = getattr(TRACKER_REGISTRY.get(str(name)), "default_options_by_camera", None)
        if hook is not None:
            return hook()
    return {}


def detections_for_tracking(tracker, detections, confidence_threshold):
    """Include weak detections only for trackers that can associate them separately."""
    if tracker is not None and getattr(
        tracker, "wants_low_confidence_detections", False
    ):
        return list(detections)
    return [
        detection
        for detection in detections
        if detection.confidence >= confidence_threshold
    ]


@dataclass
class _Track:
    track_id: int
    class_id: int
    centroid: tuple
    bbox: tuple
    #: Timestamp in seconds of the frame this track was last matched on.
    last_seen: float


class MatchingTracker(Tracker):
    """Base tracker: honor a valid upstream ``object_id``, else match to a recent
    track via a subclass-defined similarity, else start a new (negative) id.

    Subclasses implement ``_match(state, detection, used_ids, timestamp_s)``.
    """

    def __init__(
        self,
        max_tracking_distance=DEFAULT_TRACKING_DISTANCE,
        max_track_age=DEFAULT_TRACKING_MAX_AGE,
        fps=DEFAULT_FPS,
        **_ignored,
    ):
        self.max_tracking_distance = float(max_tracking_distance)
        self.max_track_age = int(max_track_age)
        self.max_track_age_seconds = float(max_track_age) / max(float(fps), 1e-6)
        self._states = {}  # source_id -> {"tracks": {id: _Track}, "next": int}

    # -- overridable ------------------------------------------------------

    def _match(self, state, detection, used_ids, frame_number):
        """Return the id of a matching existing track, or None."""
        raise NotImplementedError

    # -- machinery --------------------------------------------------------

    def _state(self, source_id):
        if source_id not in self._states:
            self._states[source_id] = {"tracks": {}, "next": 1}
        return self._states[source_id]

    def _new_id(self, state):
        track_id = -state["next"]
        state["next"] += 1
        return track_id

    def reset(self, source_id=None):
        if source_id is None:
            self._states.clear()
        else:
            self._states.pop(source_id, None)

    def update(self, source_id, timestamp_s, detections):
        state = self._state(source_id)
        tracks = state["tracks"]
        used = set()
        assignments = []
        for detection_index, detection in enumerate(detections):
            if is_valid_object_id(detection.object_id) and detection.object_id not in used:
                track_id = int(detection.object_id)
            else:
                track_id = self._match(state, detection, used, timestamp_s)
                if track_id is None:
                    track_id = self._new_id(state)
            existing = tracks.get(track_id)
            if existing is None:
                tracks[track_id] = _Track(
                    track_id=track_id,
                    class_id=detection.class_id,
                    centroid=detection.centroid,
                    bbox=detection.bbox,
                    last_seen=timestamp_s,
                )
            else:
                existing.class_id = detection.class_id
                existing.centroid = detection.centroid
                existing.bbox = detection.bbox
                existing.last_seen = timestamp_s
            detection.track_id = track_id
            used.add(track_id)
            assignments.append(
                TrackAssignment(
                    detection_index=detection_index,
                    track_id=track_id,
                    association_score=1.0,
                    state="confirmed",
                )
            )

        expired = [
            tid
            for tid, track in tracks.items()
            if timestamp_s - track.last_seen > self.max_track_age_seconds
        ]
        for tid in expired:
            del tracks[tid]
        return assignments
