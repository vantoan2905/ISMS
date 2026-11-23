"""Bounded queue that keeps recent frames for a camera."""
import asyncio


class FrameQueue:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError('Frame queue capacity must be positive')
        self.queue = asyncio.Queue(maxsize=capacity)
        self.dropped = 0

    def put_latest(self, frame):
        if self.queue.full():
            self.queue.get_nowait()
            self.dropped += 1
        self.queue.put_nowait(frame)

    async def get(self):
        return await self.queue.get()
