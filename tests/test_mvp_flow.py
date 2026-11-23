import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from src.algorithms.base import FrameContext
from src.algorithms.helmet_compliance import HelmetCompliance
from src.algorithms.zone_rules import ZoneRules
from src.events.engine import EventEngine
from src.events.models import Observation
from src.media.frame_queue import FrameQueue
from src.storage.database import Database


def context(timestamp, persons):
    return FrameContext('cam', np.zeros((100, 100, 3), np.uint8), timestamp,
                        persons=persons, pts_ns=int(timestamp*1e9),
                        received_at_utc=datetime.now(timezone.utc))


def person(track, x, y, crop=None):
    return {'track_id': track, 'box': (x-2, y-10, x+2, y),
            'keypoints': np.zeros((17, 2)), 'scores': np.zeros(17), 'head': crop}


def test_roi_entry_dwell_and_exit():
    rules = ZoneRules({'zone': [[0.2,0.2],[0.8,0.2],[0.8,0.8],[0.2,0.8]]},
                      [{'type':'roi_entry','roi_id':'zone'},
                       {'type':'roi_dwell','roi_id':'zone','threshold_ms':2000}],
                      margin_pixels=0, confirm_frames=2)
    def step(t, x):
        ctx = context(t, [person(1, x, 50)])
        rules.process(ctx)
        return [o.type for o in ctx.observations]
    assert step(0, 5) == []
    assert step(1, 50) == []
    assert step(2, 50) == ['ROI_ENTER']
    assert step(3, 50) == []
    assert step(4, 50) == ['ROI_DWELL']
    assert step(5, 50) == []
    assert step(6, 5) == []
    assert step(7, 5) == ['ROI_EXIT']


def test_helmet_unknown_does_not_emit_violation():
    class Classifier:
        def classify(self, image):
            return {'helmet': 0.45, 'no_helmet': 0.55}
    algorithm = HelmetCompliance(Classifier(), min_confidence=0.8, confirm_frames=1)
    for head in (None, SimpleNamespace(image=np.zeros((20, 20, 3), np.uint8))):
        ctx = context(1, [person(1, 40, 50, head)])
        algorithm.process(ctx)
        assert not ctx.observations
        assert ctx.persons[0]['helmet'] is None


def test_queue_drop_and_relay_multiple_tracks(tmp_path):
    async def run():
        queue = FrameQueue(1)
        queue.put_latest(1)
        queue.put_latest(2)
        assert await queue.get() == 2 and queue.dropped == 1
        db = Database(str(tmp_path / 'events.db'))
        engine = EventEngine(db, tmp_path / 'snapshots')
        ctx = context(1, [])
        ctx.observations = [Observation('ROI_ENTER', 1, 'zone', 1), Observation('ROI_ENTER', 2, 'zone', 1)]
        await engine.handle('cam', ctx)
        assert engine.relay_on and len(engine.active) == 2
        await engine.handle('cam', ctx)
        assert len(db.list_incidents()) == 2
        ctx.observations = [Observation('ROI_EXIT', 1, 'zone', 2)]
        await engine.handle('cam', ctx)
        assert engine.relay_on and len(engine.active) == 1
        ctx.observations = [Observation('ROI_EXIT', 2, 'zone', 3)]
        await engine.handle('cam', ctx)
        assert not engine.relay_on and not engine.active
        assert all(event['status'] == 'RESOLVED' for event in db.list_incidents())
        assert len(Database(str(tmp_path / 'events.db')).list_incidents()) == 2
    asyncio.run(run())


def test_api_uses_mvp_incidents_and_stops_cameras(tmp_path, monkeypatch):
    import yaml
    from fastapi.testclient import TestClient
    from src.api import main as api

    config = {
        'models': {'pose': 'unused.onnx'},
        'storage': {'path': str(tmp_path / 'events.db'), 'snapshot_dir': str(tmp_path / 'snapshots')},
        'cameras': [{'id': 'cam', 'source': 'unused.mp4', 'algorithms': []}],
    }
    config_path = tmp_path / 'cameras.yaml'
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(api, 'InferenceExecutor', lambda config: object())

    async def wait_for_stop(camera, config, executor, engine, stop):
        await stop.wait()

    monkeypatch.setattr(api, 'run_camera', wait_for_stop)
    with TestClient(api.create_app(config_path)) as client:
        assert client.get('/events').json() == []
        health = client.get('/health').json()
        assert health['active_cameras'] == 1
        assert health['relay_on'] is False


def test_loads_models_relative_to_camera_config(tmp_path):
    import yaml
    from src.isms.config.loader import load_config

    (tmp_path / 'models.yaml').write_text(yaml.safe_dump({'pose': 'pose.onnx'}))
    camera_config = {
        'model_config': 'models.yaml',
        'cameras': [{'id': 'cam', 'source': 'video.mp4', 'algorithms': []}],
    }
    path = tmp_path / 'cameras.yaml'
    path.write_text(yaml.safe_dump(camera_config))
    assert load_config(path)['models']['pose'] == 'pose.onnx'
