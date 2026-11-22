from src.inference.detector import Detection
from src.inference.tracker import CentroidTracker
from src.event_check.event_engine import EventEngine
from src.event_check.zone_intrusion import inside_polygon


def test_polygon_and_tracker():
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert inside_polygon((50, 50), polygon)
    assert not inside_polygon((150, 50), polygon)
    tracker = CentroidTracker()
    first = tracker.update([Detection("person", .9, (10, 10, 30, 90))])[0]
    second = tracker.update([Detection("person", .9, (12, 10, 32, 90))])[0]
    assert first.id == second.id


def test_events_and_cooldown():
    person = Detection("person", .9, (20, 10, 40, 90))
    tracks = CentroidTracker().update([person])
    camera = {"id": "test", "zones": [{"id": "z", "restricted": True,
              "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]}], "helmet_required": True}
    engine = EventEngine()
    assert {e["type"] for e in engine.evaluate(camera, tracks, [person])} == {"zone_intrusion", "missing_helmet"}
    assert engine.evaluate(camera, tracks, [person]) == []
    assert len(EventEngine().evaluate(camera, tracks, [person], helmet_supported=False)) == 1
