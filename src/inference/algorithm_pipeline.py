import asyncio
import time
from datetime import datetime, timezone

import cv2
import numpy as np

from ..algorithms.base import FrameContext
from ..algorithms.classification import HelmetClassificationAlgorithm
from ..algorithms.helmet_event import HelmetEventAlgorithm
from ..algorithms.pose import PoseAlgorithm
from ..algorithms.roi import ROIEntryAlgorithm, ROIStationaryAlgorithm
from ..services.gstreamer import GStreamerCamera

SKELETON = ((0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))


class AlgorithmPipeline:
    def __init__(self, camera, config):
        self.camera = camera
        settings = {**config.get('algorithms', {}), **camera.get('algorithms', {})}
        self.algorithms = [
            PoseAlgorithm(settings.get('pose_weights', 'models/rtmo_m/rtmo_m_640x640_coco.onnx')),
            HelmetClassificationAlgorithm(settings.get('classifier_weights', 'models/helmet_classifier_efficientnet/best.pt'),
                                          settings.get('helmet_threshold', 0.5)),
            HelmetEventAlgorithm(settings.get('helmet_cooldown_seconds', 10)),
            ROIEntryAlgorithm(camera.get('zones', [])),
            ROIStationaryAlgorithm(settings.get('stationary_seconds', 10), settings.get('stationary_movement_pixels', 30)),
        ]

    def process(self, frame, timestamp):
        context = FrameContext(self.camera['id'], frame, timestamp)
        for algorithm in self.algorithms:
            algorithm.process(context)
        for zone in self.camera.get('zones', []):
            points = np.asarray(zone['polygon'], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], True, (255, 160, 0), 2)
        for person in context.persons:
            points, scores = person['keypoints'], person['scores']
            for a, b in SKELETON:
                if (scores[a] >= 0.3 and scores[b] >= 0.3 and
                        np.isfinite(points[[a, b]]).all()):
                    p1 = tuple(map(int, points[a]))
                    p2 = tuple(map(int, points[b]))
                    cv2.line(frame, p1, p2, (255, 180, 0), 2)
            x1, y1, x2, y2 = person['box']
            status = 'unknown' if person['helmet'] is None else ('helmet' if person['helmet'] else 'no helmet')
            color = (0, 180, 0) if person['helmet'] else ((0, 0, 255) if person['helmet'] is False else (0, 200, 255))
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{person['track_id']} {status}", (max(0,x1), max(15,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return context


async def run_algorithm_camera(camera, config, service, stop):
    pipeline = await asyncio.to_thread(AlgorithmPipeline, camera, config)
    stream = GStreamerCamera(camera)
    count = 0
    try:
        await asyncio.to_thread(stream.open)
        while not stop.is_set():
            frame = await asyncio.to_thread(stream.read, stop)
            if frame is None:
                break
            source = camera['source']
            is_live = isinstance(source, int) or str(source).startswith(('rtsp://', 'rtsps://'))
            timestamp = time.monotonic() if is_live else count / stream.fps
            context = await asyncio.to_thread(pipeline.process, frame, timestamp)
            for event in context.events:
                event.update(camera_id=camera['id'], timestamp=datetime.now(timezone.utc).isoformat())
                await service.publish(event, camera.get('relay_url'))
            await asyncio.to_thread(stream.write, frame)
            count += 1
    finally:
        await asyncio.to_thread(stream.close)
