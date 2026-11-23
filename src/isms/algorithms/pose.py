from .base import Algorithm
from src.inference.detector import Detection
from ..inference.pose_estimator import RTMOHeadExtractor
from ..tracking.person_tracker import CentroidTracker


class PoseAlgorithm(Algorithm):
    def __init__(self, weights=None, person_threshold=0.5, min_keypoint_score=0.3, extractor=None):
        self.extractor = extractor or RTMOHeadExtractor(weights, person_threshold=person_threshold,
                                                        min_score=min_keypoint_score)
        self.tracker = CentroidTracker()
        self.min_score = min_keypoint_score

    def process(self, context):
        crops, keypoints, scores, boxes = self.extractor.infer(context.frame)
        detections = [Detection('person', 1.0, tuple(map(int, box))) for box in boxes]
        tracks = self.tracker.update(detections)
        by_person = {crop.person_index: crop for crop in crops}
        for index, (points, confidence, track) in enumerate(zip(keypoints, scores, tracks)):
            context.persons.append({'track_id': track.id, 'box': track.detection.box,
                                    'keypoints': points, 'scores': confidence,
                                    'head': by_person.get(index)})
