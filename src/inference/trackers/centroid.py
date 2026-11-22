"""Centroid-distance tracker."""

from __future__ import annotations

import math

from .registry import TRACKER_REGISTRY
from .base import MatchingTracker


@TRACKER_REGISTRY.register("centroid")
class CentroidTracker(MatchingTracker):
    """Match an untracked detection to the nearest same-class track within
    ``max_tracking_distance`` (centroid distance)."""

    def _match(self, state, detection, used_ids, timestamp_s):
        best_id = None
        best_distance = None
        cx, cy = detection.centroid
        for track in state["tracks"].values():
            if track.track_id in used_ids or track.class_id != detection.class_id:
                continue
            if timestamp_s - track.last_seen > self.max_track_age_seconds:
                continue
            tx, ty = track.centroid
            distance = math.hypot(cx - tx, cy - ty)
            if distance > self.max_tracking_distance:
                continue
            if best_distance is None or distance < best_distance:
                best_id = track.track_id
                best_distance = distance
        return best_id
