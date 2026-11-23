"""Interfaces used by the shared inference executor."""
from typing import Protocol


class PoseEstimator(Protocol):
    def infer(self, frame):
        """Return head crops, keypoints, confidence scores and person boxes."""


class HelmetClassifier(Protocol):
    def classify(self, bgr_crop):
        """Return helmet class probabilities for a BGR head crop."""
