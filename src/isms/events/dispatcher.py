"""Fan out incident updates to bounded subscriber queues."""
import asyncio


class EventDispatcher:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.subscribers = set()

    def publish(self, event):
        for queue in tuple(self.subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(dict(event))

    def subscribe(self):
        queue = asyncio.Queue(maxsize=self.capacity)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue):
        self.subscribers.discard(queue)
