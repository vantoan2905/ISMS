"""Run one bounded camera ingest queue and independent per-camera rules."""
import asyncio
import logging
from pathlib import Path

import cv2
import time

from ..algorithms.base import FrameContext
from ..algorithms.pipeline import CameraAlgorithmPipeline
from ..media.gst_camera import CameraReader
from ..media.video_output import render_overlay
from ..core.frame_queue import FrameQueue

logger = logging.getLogger(__name__)


async def run_camera(camera, config, executor, engine, stop):
    capacity = config.get('runtime', {}).get('max_frame_queue', 3)
    queue = FrameQueue(capacity)
    reader = CameraReader(camera, capacity)
    pipeline = CameraAlgorithmPipeline(camera, executor)
    failed = None
    processed = 0
    last_pts = None
    writer = None

    async def ingest():
        nonlocal failed
        try:
            await reader.start()
            while not stop.is_set():
                frame = await reader.read()
                if frame is None:
                    break
                queue.put_latest(frame)
        except Exception as exc:
            failed = exc
        finally:
            if asyncio.current_task().cancelling():
                queue.put_latest(None)
            else:
                await queue.queue.put(None)

    producer = asyncio.create_task(ingest())
    try:
        while True:
            packet = await queue.get()
            if packet is None:
                break
            # PTS drives video-time rules; monotonic fallback is used if PTS is absent.
            pts = packet.pts_ns if packet.pts_ns >= 0 else packet.received_at_monotonic_ns
            if last_pts is not None and pts < last_pts:
                await engine.reset_camera(camera['id'], packet.received_at_utc)
            last_pts = pts
            context = FrameContext(camera['id'], packet.image, pts / 1e9,
                                   frame_id=packet.frame_id, pts_ns=pts,
                                   received_at_utc=packet.received_at_utc)
            await executor.process(pipeline, context)
            await engine.handle(camera['id'], context)
            video_dir = config.get('runtime', {}).get('video_dir')
            if video_dir:
                if writer is None:
                    Path(video_dir).mkdir(parents=True, exist_ok=True)
                    output = str(Path(video_dir) / f"{camera['id']}.mp4")
                    height, width = context.frame.shape[:2]
                    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'),
                                             float(camera.get('record_fps', 8)), (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f'Cannot open demo output {output}')
                image = render_overlay(context, camera, engine.relay_on)
                await asyncio.to_thread(writer.write, image)
            processed += 1
            if processed % 100 == 0:
                logger.info('camera=%s processed=%d queue_dropped=%d pose_ms=%.1f alert_latency_ms=%.1f',
                            camera['id'], processed, queue.dropped, context.pose_ms,
                            (time.monotonic_ns() - packet.received_at_monotonic_ns) / 1e6)
    finally:
        if writer is not None:
            writer.release()
        if not producer.done():
            producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        await reader.close()
        logger.info('camera=%s processed=%d queue_dropped=%d', camera['id'], processed, queue.dropped)
    if failed:
        await engine.offline(camera['id'], failed)
    elif not stop.is_set():
        await engine.offline(camera['id'], 'End of stream')
