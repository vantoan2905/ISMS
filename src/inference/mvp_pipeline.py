"""Compatibility imports for the original MVP pipeline."""
from .executor import InferenceExecutor
from ..algorithms.pipeline import CameraAlgorithmPipeline

__all__ = ["CameraAlgorithmPipeline", "InferenceExecutor"]
