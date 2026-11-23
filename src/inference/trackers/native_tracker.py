"""Python adapter for the C++ multi-source BoT-SORT tracker."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import yaml

from .registry import TRACKER_REGISTRY
from .tracker_base import TrackAssignment, Tracker

_REPO = Path(__file__).resolve().parents[3]
_NATIVE_DIR = _REPO / "build" / "botsort_native"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "native_tracker.yaml"


def _import_native():
    if str(_NATIVE_DIR) not in sys.path:
        sys.path.insert(0, str(_NATIVE_DIR))
    try:
        import botsort_native as native  # noqa: E402
    except ImportError as exc:  # pragma: no cover - depends on the build step
        raise ImportError(
            "botsort_native is not built. Run src/inference/trackers/native/build.sh "
            "before using botsort_native (looked in {}).".format(_NATIVE_DIR)
        ) from exc
    return native


@lru_cache(maxsize=None)
def _cameras_section(path):
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    cameras = data.get("cameras") or {}
    if not isinstance(cameras, dict) or any(
        not isinstance(options, dict) for options in cameras.values()
    ):
        raise ValueError("{}: 'cameras' must map camera names to mappings".format(path))
    return cameras


@TRACKER_REGISTRY.register("botsort_native")
class NativeBotSortTracker(Tracker):
    """Apply YAML defaults followed by global and source-specific overrides."""

    wants_low_confidence_detections = True

    def __init__(
        self,
        config_path=None,
        tracker_options=None,
        tracker_options_by_source=None,
        fps=None,
        botsort_config=None,
        **_ignored,
    ):
        native = _import_native()
        options = {}
        if fps is not None:
            options["fps"] = float(fps)
        options.update(botsort_config or {})
        options.update(tracker_options or {})
        self.config_path = str(config_path or DEFAULT_CONFIG_PATH)
        self._native = native.MultiSourceTracker(
            self.config_path,
            options,
            {
                int(source_id): dict(source_options)
                for source_id, source_options in (tracker_options_by_source or {}).items()
            },
        )

    @classmethod
    def default_options_by_camera(cls, config_path=None):
        """The per-camera map a run uses when none is given: the YAML ``cameras``."""
        return copy.deepcopy(_cameras_section(str(config_path or DEFAULT_CONFIG_PATH)))

    def settings(self, source_id=0):
        """Effective options of one source, as a dict (for logs and tests)."""
        return self._native.settings(int(source_id))

    @property
    def uses_masks(self):
        return bool(self._native.uses_masks)

    def uses_reid(self, source_id=0):
        """Whether ``source_id`` fuses ``Detection.embedding`` into association."""
        return bool(self._native.uses_reid_for(int(source_id)))

    def uses_frames(self, source_id=0):
        """Whether ``source_id`` runs camera motion compensation (wants ``frame``)."""
        return bool(self._native.uses_frames_for(int(source_id)))

    def reset(self, source_id=None):
        self._native.reset(source_id)

    def update(self, source_id, frame_number, detections, frame=None):
        """Track one frame of one source.

        ``frame`` is optional and only used when the source has a ``cmc_method``:
        a uint8 gray (HxW), BGR (HxWx3) or BGRA/RGBA (HxWx4) image in the same
        pixel coordinates as the detection boxes. Without it the camera is
        treated as fixed for that frame. ``Detection.embedding`` feeds ReID when
        the source has ``with_reid``; a detection without one is matched by IoU.
        """
        source_id = int(source_id)
        with_masks = self._native.uses_masks_for(source_id)
        embeddings = None
        if self._native.uses_reid_for(source_id):
            embeddings = [
                None
                if getattr(det, "embedding", None) is None
                else np.asarray(det.embedding, dtype=np.float32).reshape(-1)
                for det in detections
            ]
        if frame is not None and not self._native.uses_frames_for(source_id):
            frame = None  # do not pay for the numpy conversion when CMC is off
        rows = []
        for det in detections:
            left, top, width, height = det.bbox
            row = (left, top, width, height, det.confidence, det.class_id)
            if with_masks:
                mask = det.mask
                row += (None if mask is None else np.asarray(mask),)
            rows.append(row)

        native_assignments = self._native.update(
            source_id, float(frame_number), rows, frame=frame, embeddings=embeddings
        )

        for det in detections:
            det.track_id = None

        assignments = []
        for native_assignment in native_assignments:
            detection = detections[native_assignment.detection_index]
            detection.track_id = native_assignment.track_id
            detection.lineage_id = native_assignment.lineage_id
            detection.predecessor_track_id = native_assignment.predecessor_track_id
            detection.association_score = native_assignment.association_score
            detection.association_reason = native_assignment.association_reason
            detection.association_state = "confirmed"
            assignments.append(
                TrackAssignment(
                    detection_index=native_assignment.detection_index,
                    track_id=native_assignment.track_id,
                    association_score=native_assignment.association_score,
                    state="confirmed",
                    lineage_id=native_assignment.lineage_id,
                    predecessor_track_id=native_assignment.predecessor_track_id,
                    association_reason=native_assignment.association_reason,
                )
            )
        return assignments
