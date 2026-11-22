import asyncio
import logging

import cv2

from .tracker import CentroidTracker
from ..event_check.event_engine import EventEngine
from .video_pipeline import Pipeline

logger = logging.getLogger(__name__)


async def run_camera(camera, detector, engine: EventEngine, event_service, stop: asyncio.Event):
    source = camera["source"]
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        logger.error("Cannot open camera %s (%s)", camera["id"], source)
        return
    tracker = CentroidTracker()
    helmet_supported = engine.helmet_class in detector.labels
    if camera.get("helmet_required") and not helmet_supported:
        logger.warning("Camera %s: model has no '%s' class; helmet rule disabled",
                       camera["id"], engine.helmet_class)
    try:
        while not stop.is_set():
            ok, frame = await asyncio.to_thread(capture.read)
            if not ok:
                logger.info("Camera %s stream ended", camera["id"])
                break
            detections = await asyncio.to_thread(detector.detect, frame)
            tracks = tracker.update(detections)
            for event in engine.evaluate(camera, tracks, detections, helmet_supported):
                await event_service.publish(event, camera.get("relay_url"))
            await asyncio.sleep(0)
    finally:
        capture.release()

