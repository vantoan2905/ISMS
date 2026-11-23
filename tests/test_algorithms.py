import numpy as np

from src.algorithms.base import FrameContext
from src.algorithms.roi import ROIEntryAlgorithm, ROIStationaryAlgorithm


def person(x, y, track=1):
    points = np.zeros((17, 2), dtype=float)
    scores = np.zeros(17)
    return {'track_id': track, 'box': (x-5, y-10, x+5, y), 'keypoints': points, 'scores': scores}


def test_entry_and_stationary_are_per_zone_and_reset_on_movement():
    entry = ROIEntryAlgorithm([{'id': 'zone', 'polygon': [[0,0], [100,0], [100,100], [0,100]]}])
    stationary = ROIStationaryAlgorithm(duration_seconds=2, movement_pixels=10)
    def step(x, y, timestamp):
        context = FrameContext('camera', None, timestamp, [person(x, y)])
        entry.process(context)
        stationary.process(context)
        return [event['type'] for event in context.events]
    assert step(150, 50, 0) == []
    assert step(50, 50, 1) == ['zone_intrusion']
    assert step(52, 50, 2) == []
    assert step(52, 50, 3) == ['stationary_in_zone']
    assert step(52, 50, 4) == []
    assert step(70, 50, 5) == []
    assert step(70, 50, 7) == ['stationary_in_zone']
    assert step(150, 50, 8) == []
    assert step(50, 50, 9) == ['zone_intrusion']
