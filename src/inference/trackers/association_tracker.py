"""Multi-source motion tracking using the optional botsort_native extension."""

from __future__ import annotations

from pathlib import Path
import math
import sys

import cv2
import numpy as np

from .tracker_config import (
    DEFAULT_BOTSORT_MATCH_IOU,
    DEFAULT_TRACKING_MAX_AGE,
)
from .registry import TRACKER_REGISTRY
from .tracker_base import TrackAssignment, Tracker

_NATIVE_DIR = Path(__file__).resolve().parents[3] / "build" / "botsort_native"


def _import_native():
    if str(_NATIVE_DIR) not in sys.path:
        sys.path.insert(0, str(_NATIVE_DIR))
    try:
        import botsort_native as native  # noqa: E402
    except ImportError as exc:  # pragma: no cover - depends on the build step
        raise ImportError(
            "Install the optional botsort_native extension "
            "before using botsort_hungarian (looked in {}).".format(_NATIVE_DIR)
        ) from exc
    return native


def _iou(a, b):
    """IoU of two (left, top, width, height) boxes."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    denom = max(0.0, a[2]) * max(0.0, a[3]) + max(0.0, b[2]) * max(0.0, b[3]) - inter
    return inter / denom if denom > 0.0 else 0.0


def _placed_mask(detection):
    """Resample a detection mask into its bounding box."""
    mask = getattr(detection, "mask", None)
    if mask is None:
        return None
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2 or binary.size == 0 or not binary.any():
        return None
    left, top, width, height = (int(round(v)) for v in detection.bbox)
    if width <= 0 or height <= 0:
        return None
    if binary.shape != (height, width):
        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    return left, top, binary.astype(bool)


def _placed_mask_iou(a, b):
    """IoU of two ``_placed_mask`` results in the shared bbox coordinate space."""
    ax, ay, am = a
    bx, by, bm = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1 = min(ax + am.shape[1], bx + bm.shape[1])
    y1 = min(ay + am.shape[0], by + bm.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = int(
        np.logical_and(
            am[y0 - ay : y1 - ay, x0 - ax : x1 - ax],
            bm[y0 - by : y1 - by, x0 - bx : x1 - bx],
        ).sum()
    )
    if inter == 0:
        return 0.0
    union = int(am.sum()) + int(bm.sum()) - inter
    return inter / union if union > 0 else 0.0


def suppress_duplicate_masks(detections, iou_threshold):
    """Suppress overlapping same-class masks in descending confidence order."""
    if not iou_threshold or len(detections) < 2:
        return list(range(len(detections))), {}
    order = sorted(
        range(len(detections)), key=lambda i: -float(detections[i].confidence)
    )
    placed = {}
    kept = []
    suppressed = {}
    for index in order:
        detection = detections[index]
        if index not in placed:
            placed[index] = _placed_mask(detection)
        mine = placed[index]
        duplicate_of = None
        if mine is not None:
            for kept_index in kept:
                other = placed[kept_index]
                if other is None or detections[kept_index].class_id != detection.class_id:
                    continue
                if _iou(detection.bbox, detections[kept_index].bbox) <= 0.0:
                    continue
                iou = _placed_mask_iou(mine, other)
                if iou >= iou_threshold:
                    duplicate_of = (kept_index, iou)
                    break
        if duplicate_of is None:
            kept.append(index)
        else:
            suppressed[index] = duplicate_of
    return sorted(kept), suppressed


@TRACKER_REGISTRY.register("botsort_hungarian")
class BotSortHungarianTracker(Tracker):
    """Run a motion tracker independently for each source."""

    # The native tracker has a second association stage for detections between
    # ``track_low_thresh`` and ``track_high_thresh``, so the sub-threshold
    # detections are useful to it rather than dangerous (see the module docstring).
    wants_low_confidence_detections = True

    def __init__(
        self,
        match_iou=DEFAULT_BOTSORT_MATCH_IOU,
        max_track_age=DEFAULT_TRACKING_MAX_AGE,
        botsort_config=None,
        tracker_options=None,
        tracker_options_by_source=None,
        fps=10.0,
        max_tracking_distance=120.0,
        **_ignored,
    ):
        self._native = _import_native()
        self._fps = max(float(fps), 1e-6)
        self._max_track_age = int(max_track_age)
        self._base_options = dict(botsort_config or {})
        self._base_options.update(tracker_options or {})
        self._options_by_source = {
            int(source_id): dict(options)
            for source_id, options in (tracker_options_by_source or {}).items()
        }
        self._settings_cache = {}
        self._match_iou = float(match_iou)
        self._max_tracking_distance = float(max_tracking_distance)
        self._trackers = {}  # source_id -> native.BotSortTracker
        self._lineages = {}  # source_id -> visible track id -> lineage id
        self._history = {}  # source_id -> visible track id -> observation dict
        # Compatibility for diagnostics/tests that inspect the effective
        # single-source native configuration. Multi-source operation must use
        # ``_settings_for(source_id)`` because each camera can differ.
        self._config = self._settings_for(0)["native"]

    @staticmethod
    def _validate_fraction(name, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("{} must be between 0 and 1".format(name))
        return value

    def _settings_for(self, source_id):
        source_id = int(source_id)
        cached = self._settings_cache.get(source_id)
        if cached is not None:
            return cached
        options = dict(self._base_options)
        options.update(self._options_by_source.get(source_id, {}))
        wrapper = {
            "enforce_match_iou": bool(options.pop("enforce_match_iou", False)),
            "lineage_enabled": bool(options.pop("lineage_enabled", False)),
            "lineage_max_seconds": max(
                0.0, float(options.pop("lineage_max_seconds", 1.5))
            ),
            "lineage_distance_scale": max(
                0.0, float(options.pop("lineage_distance_scale", 1.0))
            ),
            "mask_resolution": max(4, int(options.pop("mask_resolution", 24))),
            "duplicate_mask_iou": self._duplicate_mask_iou(
                options.pop("duplicate_mask_iou", None)
            ),
        }
        track_buffer_seconds = options.pop("track_buffer_seconds", None)
        if track_buffer_seconds is not None:
            options["track_buffer"] = max(
                1, round(float(track_buffer_seconds) * self._fps)
            )
        options.setdefault("track_buffer", self._max_track_age)
        options.setdefault("track_high_thresh", 0.70)
        options.setdefault("new_track_thresh", 0.75)
        for name in (
            "mask_min_track_confidence",
            "mask_min_track_coverage",
            "mask_min_detection_precision",
            "mask_similarity_weight",
        ):
            if name in options:
                options[name] = self._validate_fraction(name, options[name])
        settings = {"native": options, "wrapper": wrapper}
        self._settings_cache[source_id] = settings
        return settings

    @staticmethod
    def _duplicate_mask_iou(value):
        """Validate ``duplicate_mask_iou``: None/0 disables it, else a fraction in (0, 1]."""
        if value is None or value is False:
            return None
        value = float(value)
        if value == 0.0:
            return None
        if not 0.0 < value <= 1.0:
            raise ValueError(
                "duplicate_mask_iou must be in (0, 1] (or 0/null to disable), got {}".format(
                    value
                )
            )
        return value

    @property
    def uses_masks(self):
        source_ids = set(self._options_by_source) or {0}
        return any(
            bool(
                self._settings_for(source_id)["native"].get(
                    "mask_conditioned_association", False
                )
            )
            or self._settings_for(source_id)["wrapper"]["duplicate_mask_iou"] is not None
            for source_id in source_ids
        )

    def _tracker_for(self, source_id):
        tracker = self._trackers.get(source_id)
        if tracker is None:
            tracker = self._native.BotSortTracker(
                **self._settings_for(source_id)["native"]
            )
            self._trackers[source_id] = tracker
        return tracker

    @staticmethod
    def _compact_mask(mask, resolution):
        """Return a small bbox-local binary mask suitable for native matching."""
        if mask is None:
            return None
        binary = np.asarray(mask, dtype=np.uint8)
        if binary.ndim != 2 or binary.size == 0 or not binary.any():
            return None
        return np.ascontiguousarray(
            cv2.resize(
                binary,
                (int(resolution), int(resolution)),
                interpolation=cv2.INTER_NEAREST,
            ),
            dtype=np.uint8,
        )

    def reset(self, source_id=None):
        if source_id is None:
            self._trackers.clear()
            self._lineages.clear()
            self._history.clear()
        else:
            self._trackers.pop(source_id, None)
            self._lineages.pop(source_id, None)
            self._history.pop(source_id, None)

    def _lineage_for(self, source_id, timestamp_s, track_id, detection, current_ids):
        """Return the lineage and predecessor of a new native track."""
        lineages = self._lineages.setdefault(source_id, {})
        if track_id in lineages:
            return lineages[track_id], None
        wrapper = self._settings_for(source_id)["wrapper"]
        predecessor = None
        best_distance = None
        if wrapper["lineage_enabled"]:
            for candidate_id, observation in self._history.setdefault(source_id, {}).items():
                if candidate_id == track_id or candidate_id in current_ids:
                    continue
                gap = float(timestamp_s) - observation["timestamp_s"]
                if not 0.0 <= gap <= wrapper["lineage_max_seconds"]:
                    continue
                if observation["class_id"] != detection.class_id:
                    continue
                dx = detection.centroid[0] - observation["centroid"][0]
                dy = detection.centroid[1] - observation["centroid"][1]
                distance = math.hypot(dx, dy)
                if distance > (
                    self._max_tracking_distance * wrapper["lineage_distance_scale"]
                ):
                    continue
                velocity = observation.get("velocity")
                if velocity is not None and dx * velocity[0] + dy * velocity[1] < 0.0:
                    continue
                if best_distance is None or distance < best_distance:
                    predecessor, best_distance = candidate_id, distance
        lineage = lineages.get(predecessor, track_id)
        lineages[track_id] = lineage
        return lineage, predecessor

    def update(self, source_id, frame_number, detections):
        tracker = self._tracker_for(source_id)
        wrapper = self._settings_for(source_id)["wrapper"]
        mask_enabled = bool(
            self._settings_for(source_id)["native"].get(
                "mask_conditioned_association", False
            )
        )
        all_detections = detections
        suppressed = {}
        kept_indices = None
        if wrapper["duplicate_mask_iou"] is not None:
            kept_indices, suppressed = suppress_duplicate_masks(
                all_detections, wrapper["duplicate_mask_iou"]
            )
            if suppressed:
                # The native tracker never sees a suppressed duplicate, so it can
                # neither start a second track nor extend a neighbour's track.
                detections = [all_detections[i] for i in kept_indices]
            else:
                kept_indices = None
        boxes = [det.bbox for det in detections]
        if mask_enabled:
            resolution = wrapper["mask_resolution"]
            dets = [
                (
                    box[0],
                    box[1],
                    box[2],
                    box[3],
                    det.confidence,
                    det.class_id,
                    self._compact_mask(det.mask, resolution),
                )
                for det, box in zip(detections, boxes)
            ]
            tracked = tracker.update_with_masks(dets)
        else:
            dets = [
                (box[0], box[1], box[2], box[3], det.confidence, det.class_id)
                for det, box in zip(detections, boxes)
            ]
            tracked = tracker.update(dets)

        for det in all_detections:
            det.track_id = None
        for index, (kept_index, mask_iou) in suppressed.items():
            duplicate = all_detections[index]
            duplicate.association_state = "suppressed_duplicate"
            duplicate.association_reason = "duplicate_mask_iou"
            duplicate.association_score = float(mask_iou)

        assignments = []
        used_indices = set()
        current_ids = {int(obj.track_id) for obj in tracked}
        for tracked_obj in tracked:
            detection_index = int(tracked_obj.source_detection_index)
            if not 0 <= detection_index < len(detections):
                continue
            if detection_index in used_indices:
                continue
            detection = detections[detection_index]
            if detection.class_id != tracked_obj.class_id:
                continue
            association_iou = _iou(tracked_obj.bbox, detection.bbox)
            if wrapper["enforce_match_iou"] and association_iou < self._match_iou:
                continue
            track_id = int(tracked_obj.track_id)
            lineage_id, predecessor_track_id = self._lineage_for(
                source_id, frame_number, track_id, detection, current_ids
            )
            detection.track_id = track_id
            detection.lineage_id = (
                lineage_id if wrapper["lineage_enabled"] else None
            )
            detection.predecessor_track_id = predecessor_track_id
            detection.association_score = association_iou
            association_reason = str(
                getattr(
                    tracked_obj,
                    "association_reason",
                    "native_source_detection_index",
                )
            )
            detection.association_reason = association_reason
            detection.association_state = "confirmed"
            used_indices.add(detection_index)
            assignments.append(
                TrackAssignment(
                    # Index into the caller's list, not the deduplicated one.
                    detection_index=(
                        kept_indices[detection_index]
                        if kept_indices is not None
                        else detection_index
                    ),
                    track_id=track_id,
                    association_score=association_iou,
                    state="confirmed",
                    lineage_id=detection.lineage_id,
                    predecessor_track_id=predecessor_track_id,
                    association_reason=association_reason,
                )
            )
            history = self._history.setdefault(source_id, {})
            previous = history.get(track_id)
            velocity = None
            if previous is not None:
                elapsed = max(float(frame_number) - previous["timestamp_s"], 1e-6)
                velocity = (
                    (detection.centroid[0] - previous["centroid"][0]) / elapsed,
                    (detection.centroid[1] - previous["centroid"][1]) / elapsed,
                )
            history[track_id] = {
                "timestamp_s": float(frame_number),
                "centroid": detection.centroid,
                "class_id": detection.class_id,
                "velocity": velocity,
            }
        cutoff = float(frame_number) - max(
            wrapper["lineage_max_seconds"] * 2.0, 1.0
        )
        history = self._history.setdefault(source_id, {})
        for track_id in [
            track_id
            for track_id, observation in history.items()
            if observation["timestamp_s"] < cutoff
        ]:
            del history[track_id]
        return assignments
