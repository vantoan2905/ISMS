import numpy as np

from src.inference.pose_head import head_box


def test_head_box_includes_headroom_and_clips_edges():
    points = np.array([[50, 40], [45, 38], [55, 38], [40, 40], [60, 40]])
    box = head_box(points, np.ones(5), (100, 100, 3), padding=0.5)
    assert box is not None
    x1, y1, x2, y2 = box
    assert x1 <= 40 and x2 >= 60
    assert y1 < 38 and y2 > 40
    assert 0 <= x1 < x2 <= 100 and 0 <= y1 < y2 <= 100


def test_head_box_requires_multiple_valid_points():
    points = np.array([[50, 40], [45, 38], [55, 38], [40, 40], [60, 40]])
    assert head_box(points, [1, 0, 0, 0, 0], (100, 100)) is None
    assert head_box(points, [0, 0, 0, 0, 0], (100, 100)) is None


def test_rtmo_empty_predictions_and_head_roi():
    from src.inference.pose_head import RTMOHeadExtractor

    class FakePose:
        def __init__(self, empty):
            self.empty = empty

        def preprocess(self, frame):
            return frame, 1.0

        def inference(self, image):
            if self.empty:
                return np.empty((1, 0, 5), np.float32), np.empty((1, 0, 17, 3), np.float32)
            points = np.zeros((1, 1, 17, 3), np.float32)
            points[0, 0, :5] = [[50, 40, 0.9], [45, 38, 0.9], [55, 38, 0.9],
                                 [40, 40, 0.9], [60, 40, 0.9]]
            return np.array([[[20, 10, 80, 90, 0.9]]], np.float32), points

    extractor = RTMOHeadExtractor.__new__(RTMOHeadExtractor)
    extractor.padding = 0.5
    extractor.min_score = 0.3
    extractor.person_threshold = 0.5
    extractor.nms_iou = 0.45
    frame = np.zeros((100, 100, 3), np.uint8)
    extractor.pose = FakePose(True)
    crops, points, scores = extractor.extract_with_pose(frame)
    assert crops == [] and points.shape == (0, 17, 2)
    extractor.pose = FakePose(False)
    crops, points, scores = extractor.extract_with_pose(frame)
    assert len(crops) == 1 and points.shape == (1, 17, 2)
    assert crops[0].image.size > 0
