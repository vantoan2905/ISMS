"""RTMPose-Tiny inference and padded head crops for a helmet classifier."""

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


class PoseHeadExtractor:
    """Run RTMPose on person boxes and return head crops.

    RTMPose is top-down: person_boxes must be supplied as Nx4 xyxy boxes by a
    person detector. Set pose_weights to a compatible RTMPose-Tiny ONNX model.
    """

    def __init__(self, pose_weights, device='cpu', padding=0.5, min_score=0.3):
        from rtmlib import RTMPose

        self.pose = RTMPose(str(pose_weights), model_input_size=(192, 256),
                            backend='onnxruntime', device=device, to_openpose=False)
        self.padding = padding
        self.min_score = min_score

    def extract(self, frame, person_boxes):
        crops, _, _ = self.extract_with_pose(frame, person_boxes)
        return crops

    def extract_with_pose(self, frame, person_boxes):
        """Return head crops and COCO keypoints and scores for every person."""
        boxes = np.asarray(person_boxes, dtype=np.float32).reshape(-1, 4)
        if not len(boxes):
            return [], np.empty((0, 17, 2)), np.empty((0, 17))
        keypoints, scores = self.pose(frame, bboxes=boxes)
        crops = []
        for index, (points, confidence) in enumerate(zip(keypoints, scores)):
            box = head_box(points, confidence, frame.shape, self.padding, self.min_score)
            if box is not None:
                x1, y1, x2, y2 = box
                crops.append(HeadCrop(index, box, frame[y1:y2, x1:x2].copy()))
        return crops, keypoints, scores


class RTMOHeadExtractor:
    """Detect people and COCO body keypoints in one RTMO inference."""

    def __init__(self, pose_weights, device='cpu', padding=0.5,
                 min_score=0.3, person_threshold=0.5, nms_iou=0.45,
                 cpu_threads=2):
        from rtmlib import RTMO

        self.pose = RTMO(str(pose_weights), model_input_size=(640, 640),
                         backend='onnxruntime', device=device, to_openpose=False)
        if device == 'cpu':
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = cpu_threads
            self.pose.session = ort.InferenceSession(str(pose_weights), sess_options=options,
                                                     providers=['CPUExecutionProvider'])
        self.padding = padding
        self.min_score = min_score
        self.person_threshold = person_threshold
        self.nms_iou = nms_iou

    def extract(self, frame):
        crops, _, _ = self.extract_with_pose(frame)
        return crops

    def extract_with_pose(self, frame):
        from rtmlib.tools.object_detection.post_processings import multiclass_nms

        image, ratio = self.pose.preprocess(frame)
        detections, poses = self.pose.inference(image)
        if (detections.ndim != 3 or poses.ndim != 4 or detections.shape[0] != 1 or
                poses.shape[0] != 1 or detections.shape[1] != poses.shape[1] or
                detections.shape[2] != 5 or poses.shape[2:] != (17, 3)):
            raise ValueError('Expected RTMO outputs (1, N, 5) and (1, N, 17, 3)')
        if detections.shape[1] == 0:
            return [], np.empty((0, 17, 2)), np.empty((0, 17))
        boxes = detections[0, :, :4] / ratio
        person_scores = detections[0, :, 4]
        _, keep = multiclass_nms(boxes, person_scores[:, None],
                                 nms_thr=self.nms_iou, score_thr=self.person_threshold)
        if keep is None or len(keep) == 0:
            return [], np.empty((0, 17, 2)), np.empty((0, 17))
        keypoints = poses[0, keep, :, :2] / ratio
        scores = poses[0, keep, :, 2]
        crops = []
        for index, (points, confidence) in enumerate(zip(keypoints, scores)):
            box = head_box(points, confidence, frame.shape, self.padding, self.min_score)
            if box is not None:
                x1, y1, x2, y2 = box
                crops.append(HeadCrop(index, box, frame[y1:y2, x1:x2].copy()))
        return crops, keypoints, scores
