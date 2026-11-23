"""Generate padded head crops from COCO facial keypoints."""
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeadCrop:
    person_index: int
    box: tuple[int, int, int, int]
    image: np.ndarray


def head_box(keypoints, scores, image_shape, padding=0.5, min_score=0.3):
    """Return an expanded head ROI from COCO nose, eye and ear keypoints.

    Requires at least two visible facial keypoints so a single uncertain point
    cannot generate an arbitrary crop. The scale is estimated from their spread.
    """
    points = np.asarray(keypoints, dtype=float)
    confidence = np.asarray(scores, dtype=float).reshape(-1)
    if points.ndim != 2 or points.shape[0] < 5 or points.shape[1] != 2 or len(confidence) < 5:
        raise ValueError('Expected COCO keypoints and confidence scores')
    if not math.isfinite(padding) or padding < 0 or not 0 <= min_score <= 1:
        raise ValueError('Invalid padding or confidence threshold')
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError('Invalid image dimensions')
    face = points[:5]
    valid = (confidence[:5] >= min_score) & np.isfinite(confidence[:5]) & np.isfinite(face).all(axis=1)
    face = face[valid]
    if len(face) < 2:
        return None
    x1, y1 = face.min(axis=0)
    x2, y2 = face.max(axis=0)
    span = max(x2 - x1, (y2 - y1) * 1.5)
    if span < 2:
        return None
    # Facial points lie below the crown; extend upwards for the helmet.
    center_x = (x1 + x2) / 2
    side = span * (2 + 2 * padding)
    center_y = y1 - span * 0.35
    left = max(0, math.floor(center_x - side / 2))
    top = max(0, math.floor(center_y - side / 2))
    right = min(width, math.ceil(center_x + side / 2))
    bottom = min(height, math.ceil(center_y + side / 2))
    return (left, top, right, bottom) if right > left and bottom > top else None

