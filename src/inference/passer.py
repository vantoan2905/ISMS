"""Parse decoded YOLOX predictions into image-space detections."""

import math

import torch
from torchvision.ops import batched_nms

from .detector import Detection


def parse_yolox(output, image_size, input_size=(416, 416), labels=("head",),
                confidence=0.4, nms_iou=0.65):
    """Parse one decoded YOLOX batch in BGR letterbox coordinates.

    output has shape (1, N, 5 + classes) with center-x, center-y, width,
    height, objectness and class probabilities. image_size and input_size
    are (height, width). Preprocessing must resize uniformly and pad on the
    right and bottom, as in YOLOX's preproc function.
    """
    if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[0] != 1:
        raise ValueError("Expected a YOLOX tensor with shape (1, N, 5 + classes)")
    if not labels or output.shape[2] != 5 + len(labels):
        raise ValueError("The number of labels must match the model output")
    if len(image_size) != 2 or len(input_size) != 2:
        raise ValueError("Image and input sizes must be (height, width)")
    height, width = image_size
    input_height, input_width = input_size
    if min(height, width, input_height, input_width) <= 0:
        raise ValueError("Image and input dimensions must be positive")
    if not (math.isfinite(confidence) and 0 <= confidence <= 1 and
            math.isfinite(nms_iou) and 0 <= nms_iou <= 1):
        raise ValueError("Confidence and NMS thresholds must be between 0 and 1")

    rows = output[0].detach().float()
    if rows.shape[0] == 0:
        return []
    class_scores, class_ids = rows[:, 5:].max(dim=1)
    scores = rows[:, 4] * class_scores
    valid = torch.isfinite(rows).all(dim=1) & (rows[:, 2] > 0) & (rows[:, 3] > 0)
    valid &= (rows[:, 4] >= 0) & (rows[:, 4] <= 1)
    valid &= (class_scores >= 0) & (class_scores <= 1) & (scores >= confidence)
    rows, scores, class_ids = rows[valid], scores[valid], class_ids[valid]
    if not rows.shape[0]:
        return []
    boxes = torch.cat((rows[:, :2] - rows[:, 2:4] / 2,
                       rows[:, :2] + rows[:, 2:4] / 2), dim=1)
    # Match YOLOX: NMS happens in model coordinates before clipping.
    keep = batched_nms(boxes, scores, class_ids, nms_iou)
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
    ratio = min(input_height / height, input_width / width)
    boxes = boxes / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
    results = []
    for box, score, class_id in zip(boxes.cpu().tolist(), scores.cpu().tolist(),
                                    class_ids.cpu().tolist()):
        x1, y1, x2, y2 = (int(value) for value in box)
        if x2 > x1 and y2 > y1:
            results.append(Detection(labels[class_id], score, (x1, y1, x2, y2)))
    return results
