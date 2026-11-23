from .base import Algorithm
from ..inference.helmet_classifier import HelmetClassifier


class HelmetClassificationAlgorithm(Algorithm):
    def __init__(self, weights, threshold=0.5):
        self.model = HelmetClassifier(weights)
        self.threshold = threshold

    def process(self, context):
        for person in context.persons:
            if person['head'] is None:
                person['helmet'] = None
                continue
            probabilities = self.model.classify(person['head'].image)
            person['helmet'] = probabilities['helmet'] >= self.threshold
            person['helmet_probability'] = probabilities['helmet']
