import asyncio
import logging
import os
from ..inference.pipeline import run_camera
from ..event_check.event_engine import EventEngine

logger = logging.getLogger(__name__)


class CameraManager:
    def __init__(self, config, event_service):
        self.config = config
        self.event_service = event_service
        self.stop = asyncio.Event()
        self.tasks = []

    async def start(self):
        cameras = [camera for camera in self.config.get("cameras", []) if camera.get("enabled")]
        if not cameras:
            return
        if self.config.get('pipeline', 'legacy') == 'algorithms':
            from ..inference.algorithm_pipeline import run_algorithm_camera
            for camera in cameras:
                camera = dict(camera)
                if camera.get('relay_url') is None and os.getenv('SAFETY_RELAY_URL'):
                    camera['relay_url'] = os.environ['SAFETY_RELAY_URL']
                task = asyncio.create_task(run_algorithm_camera(camera, self.config, self.event_service, self.stop))
                task.add_done_callback(self._report_failure)
                self.tasks.append(task)
        else:
            from ..inference.detector import Detector
            model = self.config['model']
            detector = Detector(model['weights'], model.get('confidence', 0.4))
            for camera in cameras:
                camera = dict(camera)
                if camera.get('relay_url') is None and os.getenv('SAFETY_RELAY_URL'):
                    camera['relay_url'] = os.environ['SAFETY_RELAY_URL']
                engine = EventEngine(self.config.get('rules', {}).get('cooldown_seconds', 10),
                                     model.get('helmet_class', 'helmet'))
                task = asyncio.create_task(run_camera(camera, detector, engine, self.event_service, self.stop))
                task.add_done_callback(self._report_failure)
                self.tasks.append(task)

    @staticmethod
    def _report_failure(task):
        if not task.cancelled() and task.exception():
            error = task.exception()
            logger.error('Camera task failed', exc_info=(type(error), error, error.__traceback__))

    async def close(self):
        self.stop.set()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
