"""Serialize access to models shared by camera pipelines."""
import asyncio

from .helmet_classifier import HelmetClassifier
from .pose_estimator import RTMOHeadExtractor


class InferenceExecutor:
    def __init__(self, config):
        models = config['models']
        self.pose = RTMOHeadExtractor(models['pose'])
        cameras = config['cameras']
        needs_helmet = any(rule['type'] == 'helmet_compliance'
                           for camera in cameras if camera.get('enabled', True)
                           for rule in camera.get('algorithms', []))
        self.classifier = HelmetClassifier(models['helmet_classifier']) if needs_helmet else None
        self.lock = asyncio.Lock()

    async def process(self, pipeline, context):
        async with self.lock:
            await asyncio.to_thread(pipeline.process, context)

