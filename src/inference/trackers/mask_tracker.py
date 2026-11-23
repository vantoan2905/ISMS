"""Track objects using mask overlap, centroid motion and box overlap."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .registry import TRACKER_REGISTRY
from .tracker_base import TrackAssignment, Tracker


def _bbox_iou(a, b):
    ax1, ay1, aw, ah = (float(value) for value in a)
    bx1, by1, bw, bh = (float(value) for value in b)
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - intersection
    return intersection / union if union > 0.0 else 0.0


def _rasterized_mask(mask, bbox):
    if mask is None:
        return None
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2 or binary.size == 0 or not binary.any():
        return None
    width = max(1, int(round(float(bbox[2]))))
    height = max(1, int(round(float(bbox[3]))))
    return cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def _global_mask_iou(mask_a, bbox_a, mask_b, bbox_b):
    """IoU of bbox-local masks without allocating a full-frame canvas."""
    a = _rasterized_mask(mask_a, bbox_a)
    b = _rasterized_mask(mask_b, bbox_b)
    if a is None or b is None:
        return _bbox_iou(bbox_a, bbox_b)

    ax, ay = int(round(bbox_a[0])), int(round(bbox_a[1]))
    bx, by = int(round(bbox_b[0])), int(round(bbox_b[1]))
    left, top = max(ax, bx), max(ay, by)
    right = min(ax + a.shape[1], bx + b.shape[1])
    bottom = min(ay + a.shape[0], by + b.shape[0])
    intersection = 0
    if right > left and bottom > top:
        a_view = a[top - ay : bottom - ay, left - ax : right - ax]
        b_view = b[top - by : bottom - by, left - bx : right - bx]
        intersection = int(np.count_nonzero(a_view & b_view))
    union = int(np.count_nonzero(a)) + int(np.count_nonzero(b)) - intersection
    return intersection / float(union) if union > 0 else 0.0


@dataclass
class _MaskTrack:
    track_id: int
    class_id: int
    bbox: tuple
    mask: object
    centroid: tuple
    timestamp_s: float
    previous_centroid: tuple | None = None
    previous_timestamp_s: float | None = None
    hits: int = 1
    lost_since_s: float | None = None

    def predicted_centroid(self, timestamp_s):
        if self.previous_centroid is None or self.previous_timestamp_s is None:
            return self.centroid
        elapsed = self.timestamp_s - self.previous_timestamp_s
        if elapsed <= 0.0:
            return self.centroid
        vx = (self.centroid[0] - self.previous_centroid[0]) / elapsed
        vy = (self.centroid[1] - self.previous_centroid[1]) / elapsed
        horizon = max(0.0, float(timestamp_s) - self.timestamp_s)
        return (self.centroid[0] + vx * horizon, self.centroid[1] + vy * horizon)


@TRACKER_REGISTRY.register("mask_ocsort")
class MaskOCSortTracker(Tracker):
    """Short-horizon OC-SORT/ByteTrack association specialized for masks."""

    wants_low_confidence_detections = True

    _DEFAULTS = {
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.5,
        "association_threshold": 0.25,
        "second_association_threshold": 0.15,
        "min_confirmed_hits": 2,
        "max_lost_seconds": 1.5,
        "mask_iou_weight": 0.45,
        "bbox_iou_weight": 0.20,
        "motion_weight": 0.20,
        "spatial_weight": 0.15,
        "max_normalized_distance": 4.0,
    }

    def __init__(
        self,
        tracker_options=None,
        tracker_options_by_source=None,
        max_track_age=None,
        fps=10.0,
        **_ignored,
    ):
        self._global_options = dict(self._DEFAULTS)
        self._global_options.update(tracker_options or {})
        if max_track_age is not None and "max_lost_seconds" not in (tracker_options or {}):
            self._global_options["max_lost_seconds"] = float(max_track_age) / max(
                float(fps), 1e-6
            )
        self._options_by_source = {
            int(source_id): dict(options)
            for source_id, options in (tracker_options_by_source or {}).items()
        }
        self._states = {}

    def _options(self, source_id):
        options = dict(self._global_options)
        options.update(self._options_by_source.get(int(source_id), {}))
        return options

    def _state(self, source_id):
        return self._states.setdefault(int(source_id), {"tracks": {}, "next_id": 1})

    def reset(self, source_id=None):
        if source_id is None:
            self._states.clear()
        else:
            self._states.pop(int(source_id), None)

    @staticmethod
    def _centroid(detection):
        if detection.mask is not None:
            mask = np.asarray(detection.mask)
            if mask.ndim == 2 and mask.size:
                ys, xs = np.nonzero(mask)
                if xs.size:
                    x, y, width, height = detection.bbox
                    return (x + (xs.mean() + 0.5) * width / mask.shape[1],
                            y + (ys.mean() + 0.5) * height / mask.shape[0])
        return detection.centroid

    def _association_score(self, track, detection, timestamp_s, options):
        if track.class_id != detection.class_id:
            return -1.0
        centroid = self._centroid(detection)
        predicted = track.predicted_centroid(timestamp_s)
        distance = math.hypot(centroid[0] - predicted[0], centroid[1] - predicted[1])
        scale = math.sqrt(
            max(1.0, 0.5 * (track.bbox[2] * track.bbox[3] + detection.bbox[2] * detection.bbox[3]))
        )
        normalized_distance = distance / scale
        limit = max(1e-6, float(options["max_normalized_distance"]))
        if normalized_distance > limit:
            return -1.0
        spatial_score = max(0.0, 1.0 - normalized_distance / limit)

        # Direction consistency is observation-centric: compare the next
        # detection to the velocity extrapolated from the last two observations.
        raw_distance = math.hypot(
            centroid[0] - track.centroid[0], centroid[1] - track.centroid[1]
        )
        motion_score = 1.0 if raw_distance <= 1e-6 else max(
            0.0, 1.0 - distance / max(raw_distance, scale)
        )
        return (
            float(options["mask_iou_weight"])
            * _global_mask_iou(track.mask, track.bbox, detection.mask, detection.bbox)
            + float(options["bbox_iou_weight"]) * _bbox_iou(track.bbox, detection.bbox)
            + float(options["motion_weight"]) * motion_score
            + float(options["spatial_weight"]) * spatial_score
        )

    def _associate(self, tracks, detection_indices, detections, timestamp_s, threshold, options):
        if not tracks or not detection_indices:
            return [], list(range(len(tracks))), list(detection_indices)
        scores = np.full((len(tracks), len(detection_indices)), -1.0, dtype=np.float64)
        for track_index, track in enumerate(tracks):
            for column, detection_index in enumerate(detection_indices):
                scores[track_index, column] = self._association_score(
                    track, detections[detection_index], timestamp_s, options
                )
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(-scores)
        matches = []
        used_tracks = set()
        used_detections = set()
        for row, column in zip(rows, columns):
            score = float(scores[row, column])
            if score < float(threshold):
                continue
            detection_index = detection_indices[column]
            matches.append((row, detection_index, score))
            used_tracks.add(row)
            used_detections.add(detection_index)
        return (
            matches,
            [index for index in range(len(tracks)) if index not in used_tracks],
            [index for index in detection_indices if index not in used_detections],
        )

    @staticmethod
    def _observe(track, detection, timestamp_s):
        track.previous_centroid = track.centroid
        track.previous_timestamp_s = track.timestamp_s
        track.centroid = MaskOCSortTracker._centroid(detection)
        track.timestamp_s = timestamp_s
        track.bbox = tuple(detection.bbox)
        track.mask = None if detection.mask is None else np.asarray(detection.mask, dtype=bool).copy()
        track.class_id = detection.class_id
        track.hits += 1
        track.lost_since_s = None

    def update(self, source_id, timestamp_s, detections):
        timestamp_s = float(timestamp_s)
        detections = list(detections)
        for detection in detections:
            detection.track_id = None

        options = self._options(source_id)
        state = self._state(source_id)
        track_map = state["tracks"]
        max_lost = max(0.0, float(options["max_lost_seconds"]))
        expired = [
            track_id
            for track_id, track in track_map.items()
            if timestamp_s - track.timestamp_s > max_lost
        ]
        for track_id in expired:
            del track_map[track_id]

        tracks = list(track_map.values())
        high = [
            index for index, detection in enumerate(detections)
            if detection.confidence >= float(options["track_high_thresh"])
        ]
        low = [
            index for index, detection in enumerate(detections)
            if float(options["track_low_thresh"]) <= detection.confidence
            < float(options["track_high_thresh"])
        ]
        matches, unmatched_tracks, unmatched_high = self._associate(
            tracks,
            high,
            detections,
            timestamp_s,
            options["association_threshold"],
            options,
        )
        low_tracks = [tracks[index] for index in unmatched_tracks]
        low_matches, still_unmatched, _unmatched_low = self._associate(
            low_tracks,
            low,
            detections,
            timestamp_s,
            options["second_association_threshold"],
            options,
        )

        assignments = []
        min_hits = max(1, int(options["min_confirmed_hits"]))
        matched_track_ids = set()
        for track_index, detection_index, score in matches:
            track = tracks[track_index]
            self._observe(track, detections[detection_index], timestamp_s)
            state_name = "confirmed" if track.hits >= min_hits else "tentative"
            detections[detection_index].track_id = (
                track.track_id if state_name == "confirmed" else None
            )
            matched_track_ids.add(track.track_id)
            assignments.append(TrackAssignment(detection_index, track.track_id, score, state_name))

        for low_track_index, detection_index, score in low_matches:
            track = low_tracks[low_track_index]
            self._observe(track, detections[detection_index], timestamp_s)
            state_name = "confirmed" if track.hits >= min_hits else "tentative"
            detections[detection_index].track_id = (
                track.track_id if state_name == "confirmed" else None
            )
            matched_track_ids.add(track.track_id)
            assignments.append(TrackAssignment(detection_index, track.track_id, score, state_name))

        unmatched_track_objects = [
            low_tracks[index] for index in still_unmatched
            if low_tracks[index].track_id not in matched_track_ids
        ]
        for track in unmatched_track_objects:
            if track.lost_since_s is None:
                track.lost_since_s = timestamp_s

        for detection_index in unmatched_high:
            detection = detections[detection_index]
            if detection.confidence < float(options["new_track_thresh"]):
                continue
            track_id = int(state["next_id"])
            state["next_id"] += 1
            mask = None if detection.mask is None else np.asarray(detection.mask, dtype=bool).copy()
            track = _MaskTrack(
                track_id=track_id,
                class_id=detection.class_id,
                bbox=tuple(detection.bbox),
                mask=mask,
                centroid=self._centroid(detection),
                timestamp_s=timestamp_s,
            )
            track_map[track_id] = track
            state_name = "confirmed" if min_hits <= 1 else "tentative"
            detection.track_id = track_id if state_name == "confirmed" else None
            assignments.append(
                TrackAssignment(detection_index, track_id, 1.0, state_name)
            )

        assignments.sort(key=lambda assignment: assignment.detection_index)
        return assignments


__all__ = ["MaskOCSortTracker", "_global_mask_iou"]
