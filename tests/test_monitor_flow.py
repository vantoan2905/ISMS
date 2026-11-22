import asyncio

import numpy as np

from src.event_check.event_engine import EventEngine
from src.inference.detector import Detection
from src.inference.pipeline import run_camera
from src.services.event_service import EventService
from src.services import relay_simulator
from src.storage.database import Database


def test_camera_event_reaches_database_subscriber_and_relay(tmp_path, monkeypatch):
    class Capture:
        def __init__(self, source):
            self.remaining = True

        def isOpened(self):
            return True

        def read(self):
            if self.remaining:
                self.remaining = False
                return True, np.zeros((100, 100, 3), dtype=np.uint8)
            return False, None

        def release(self):
            pass

    class Detector:
        labels = {"person"}

        def detect(self, frame):
            return [Detection("person", 0.9, (10, 10, 30, 90))]

    received = []
    monkeypatch.setattr("src.inference.pipeline.cv2.VideoCapture", Capture)
    monkeypatch.setattr("src.services.event_service.send_relay",
                        lambda url, event: received.append((url, event)))
    db = Database(str(tmp_path / "events.db"))
    service = EventService(db)
    camera = {
        "id": "camera1", "source": "demo", "relay_url": "http://relay/events",
        "zones": [{"id": "zone", "restricted": True,
                   "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]}],
    }

    async def run():
        queue = service.subscribe()
        await run_camera(camera, Detector(), EventEngine(), service, asyncio.Event())
        event = await asyncio.wait_for(queue.get(), 1)
        for _ in range(100):
            if received:
                break
            await asyncio.sleep(0.01)
        return event

    event = asyncio.run(run())
    assert event["type"] == "zone_intrusion"
    assert db.list_events() == [event]
    assert received == [("http://relay/events", event)]
    relay_simulator.receive_event(event)
    assert relay_simulator.list_events()[0] == event
