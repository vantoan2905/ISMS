"""Compose camera-specific pose and rule algorithms."""
import time

from .factory import create_algorithms
from .pose import PoseAlgorithm


class CameraAlgorithmPipeline:
    def __init__(self, camera, executor):
        self.pose = PoseAlgorithm(extractor=executor.pose)
        self.algorithms = create_algorithms(camera, executor)
        self.last_pts = None

    def process(self, context):
        if self.last_pts is not None and context.pts_ns < self.last_pts:
            self.pose.tracker = type(self.pose.tracker)()
            for algorithm in self.algorithms:
                algorithm.reset()
        self.last_pts = context.pts_ns
        start = time.perf_counter()
        self.pose.process(context)
        context.pose_ms = (time.perf_counter() - start) * 1000
        for algorithm in self.algorithms:
            algorithm.process(context)
