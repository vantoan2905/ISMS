"""Manage independent camera tasks and a shared inference executor."""
import asyncio
import logging

from ..inference.executor import InferenceExecutor
from .camera_runtime import run_camera

logger = logging.getLogger(__name__)


class CameraManager:
    def __init__(self, config, engine, *, executor_factory=None, camera_runner=None):
        self.config = config
        self.engine = engine
        self.executor_factory = executor_factory or InferenceExecutor
        self.camera_runner = camera_runner or run_camera
        self.stop = asyncio.Event()
        self.tasks = []

    async def start(self):
        cameras = [camera for camera in self.config['cameras'] if camera.get('enabled', True)]
        if not cameras:
            raise ValueError('No cameras enabled')
        executor = await asyncio.to_thread(self.executor_factory, self.config)

        async def guarded(camera):
            try:
                await self.camera_runner(camera, self.config, executor, self.engine, self.stop)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception('Camera %s failed', camera['id'])
                await self.engine.offline(camera['id'], exc)

        self.tasks = [asyncio.create_task(guarded(camera), name=camera['id']) for camera in cameras]

    async def close(self):
        self.stop.set()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
