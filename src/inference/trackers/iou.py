"""IoU-overlap fallback tracker."""

from __future__ import annotations

from .config import (
    DEFAULT_FPS,
    DEFAULT_TRACKING_DISTANCE,
    DEFAULT_TRACKING_MAX_AGE,
)
from .registry import TRACKER_REGISTRY
from .base import MatchingTracker


def _bbox_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@TRACKER_REGISTRY.register("iou")
class IouTracker(MatchingTracker):
    """Match an untracked detection to the highest-IoU same-class track above a
    threshold."""

    def __init__(
        self,
        iou_match_threshold=0.3,
        max_tracking_distance=DEFAULT_TRACKING_DISTANCE,
        max_track_age=DEFAULT_TRACKING_MAX_AGE,
        fps=DEFAULT_FPS,
        **_ignored,
    ):
        super().__init__(
            max_tracking_distance=max_tracking_distance,
            max_track_age=max_track_age,
            fps=fps,
        )
        self.iou_match_threshold = float(iou_match_threshold)

    def _match(self, state, detection, used_ids, timestamp_s):
        best_id = None
        best_iou = self.iou_match_threshold
        for track in state["tracks"].values():
            if track.track_id in used_ids or track.class_id != detection.class_id:
                continue
            if timestamp_s - track.last_seen > self.max_track_age_seconds:
                continue
            score = _bbox_iou(detection.bbox, track.bbox)
            if score >= best_iou:
                best_id = track.track_id
                best_iou = score
        return best_id
