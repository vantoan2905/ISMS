import asyncio
from .relay_controller import send_relay


class EventService:
    def __init__(self, database):
        self.database = database
        self.subscribers: set[asyncio.Queue] = set()

    async def publish(self, event: dict, relay_url=None):
        saved = await asyncio.to_thread(self.database.insert, event)
        for queue in tuple(self.subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(saved)
        if relay_url:
            await asyncio.to_thread(send_relay, relay_url, saved)
        return saved

    def subscribe(self):
        queue = asyncio.Queue(maxsize=100)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue):
        self.subscribers.discard(queue)
