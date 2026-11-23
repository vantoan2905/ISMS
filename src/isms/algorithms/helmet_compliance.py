"""Per-person helmet classification and violation confirmation."""
import cv2

from .base import Algorithm
from ..events.models import Observation


class HelmetCompliance(Algorithm):
    def __init__(self, classifier, min_confidence=0.8, confirm_frames=3, min_crop_size=16, lost_timeout=2,
                 min_sharpness=10.0):
        self.classifier = classifier
        self.min_confidence = min_confidence
        self.confirm_frames = confirm_frames
        self.min_crop_size = min_crop_size
        self.lost_timeout_ns = int(lost_timeout * 1e9)
        self.min_sharpness = min_sharpness
        self.states = {}

    def process(self, context):
        for person in context.persons:
            track = person['track_id']
            state = self.states.setdefault(track, {'count': 0, 'active': False, 'last': context.pts_ns})
            state['last'] = context.pts_ns
            crop = person.get('head')
            person['helmet'] = None
            if crop is None or min(crop.image.shape[:2]) < self.min_crop_size:
                state['count'] = 0
                continue
            gray = cv2.cvtColor(crop.image, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < self.min_sharpness:
                state['count'] = 0
                continue
            probabilities = self.classifier.classify(crop.image)
            best = max(probabilities, key=probabilities.get)
            score = probabilities[best]
            person['helmet_probability'] = probabilities['helmet']
            if score < self.min_confidence:
                state['count'] = 0
                continue
            person['helmet'] = best == 'helmet'
            if best == 'no_helmet':
                state['count'] += 1
                if state['count'] >= self.confirm_frames and not state['active']:
                    state['active'] = True
                    context.observations.append(Observation('NO_HELMET', track, 'helmet', context.pts_ns, score))
            else:
                state['count'] = 0
                if state['active']:
                    state['active'] = False
                    context.observations.append(Observation('HELMET_RESOLVED', track, 'helmet', context.pts_ns, score))
        for track, state in list(self.states.items()):
            if context.pts_ns - state['last'] > self.lost_timeout_ns:
                if state['active']:
                    context.observations.append(Observation('HELMET_RESOLVED', track, 'helmet', context.pts_ns))
                del self.states[track]

    def reset(self):
        self.states.clear()

