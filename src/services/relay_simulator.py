"""HTTP relay simulator for observing safety alerts."""

import logging
from collections import deque
from threading import Lock

from fastapi import FastAPI

app = FastAPI(title="Safety relay simulator")
logger = logging.getLogger("uvicorn.error")
events = deque(maxlen=100)
lock = Lock()


@app.post("/events", status_code=202)
def receive_event(event: dict):
    with lock:
        events.append(event)
    logger.warning("Relay activated: %s", event)
    return {"status": "activated"}


@app.get("/events")
def list_events():
    with lock:
        return list(reversed(events))
