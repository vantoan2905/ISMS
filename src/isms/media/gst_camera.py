"""Subprocess bridge to a system Python GStreamer appsink."""
import asyncio
import os
import struct
import sys
from datetime import datetime, timezone

from ..core.types import Frame
from pathlib import Path

import numpy as np


class CameraReader:
    def __init__(self, camera, capacity=3):
        self.camera = camera
        self.width = int(camera.get('width', 640))
        self.height = int(camera.get('height', 480))
        if self.width <= 0 or self.height <= 0 or self.width % 4:
            raise ValueError('Width must be positive and divisible by 4, height must be positive')
        self.capacity = capacity
        self.process = None

    async def start(self):
        script = Path(__file__).with_name('gst_runtime.py')
        executable = os.getenv('ISMS_GSTREAMER_PYTHON', '/usr/bin/python3')
        self.process = await asyncio.create_subprocess_exec(
            executable, str(script), '--source', str(self.camera['source']),
            '--width', str(self.width), '--height', str(self.height),
            '--capacity', str(self.capacity), stdout=asyncio.subprocess.PIPE)

    async def read(self):
        try:
            header = await self.process.stdout.readexactly(struct.calcsize('!QqQQI'))
        except asyncio.IncompleteReadError as exc:
            code = await self.process.wait()
            if exc.partial or code:
                raise RuntimeError(f'GStreamer source failed (status {code})') from exc
            return None
        frame_id, pts_ns, received_ns, monotonic_ns, size = struct.unpack('!QqQQI', header)
        expected = self.width * self.height * 3
        if size != expected:
            raise RuntimeError(f'Unexpected GStreamer frame size {size}; expected {expected}')
        payload = await self.process.stdout.readexactly(size)
        image = np.frombuffer(payload, np.uint8).reshape(self.height, self.width, 3).copy()
        return Frame(frame_id, pts_ns, datetime.fromtimestamp(received_ns / 1e9, timezone.utc), monotonic_ns, image)

    async def close(self):
        if self.process and self.process.returncode is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.process.communicate(), 3)
            except asyncio.TimeoutError:
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
                await self.process.communicate()
