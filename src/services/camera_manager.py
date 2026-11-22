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
        from ..inference.detector import Detector
        model = self.config["model"]
        detector = Detector(model["weights"], model.get("confidence", 0.4))
        for camera in cameras:
            camera = dict(camera)
            if camera.get("relay_url") is None and os.getenv("SAFETY_RELAY_URL"):
                camera["relay_url"] = os.environ["SAFETY_RELAY_URL"]
            engine = EventEngine(self.config.get("rules", {}).get("cooldown_seconds", 10),
                                 model.get("helmet_class", "helmet"))
            task = asyncio.create_task(run_camera(camera, detector, engine, self.event_service, self.stop))
            task.add_done_callback(lambda t: logger.error("Camera task failed", exc_info=t.exception())
                                   if not t.cancelled() and t.exception() else None)
            self.tasks.append(task)

    async def close(self):
        self.stop.set()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
