"""Run the configured camera analytics application."""
import argparse
import asyncio
import logging
import signal
import uuid
from pathlib import Path

from src.isms.app.camera_manager import CameraManager
from src.isms.config.loader import load_config
from src.isms.events.engine import EventEngine
from src.isms.storage.database import Database

logger = logging.getLogger(__name__)


async def serve(config):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    if config.get('runtime', {}).get('record_demo', True):
        config.setdefault('runtime', {})['video_dir'] = str(Path('videos/demo') / uuid.uuid4().hex[:12])
        logger.info('Demo video output: %s', config['runtime']['video_dir'])
    database = Database(config.get('storage', {}).get('path', 'safety_events.db'))
    engine = EventEngine(database, config.get('storage', {}).get('snapshot_dir', 'snapshots'),
                         config.get('runtime', {}).get('enable_mock_relay', True))
    manager = CameraManager(config, engine)
    await manager.start()
    stopper = asyncio.create_task(stop.wait())
    done = asyncio.gather(*manager.tasks)
    try:
        await asyncio.wait([done, stopper], return_when=asyncio.FIRST_COMPLETED)
    finally:
        await manager.close()
        stopper.cancel()
        await asyncio.gather(done, stopper, return_exceptions=True)
    logger.info('Application stopped; relay=%s', 'ON' if engine.relay_on else 'OFF')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/cameras.yaml')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    asyncio.run(serve(load_config(args.config)))


if __name__ == '__main__':
    main()
