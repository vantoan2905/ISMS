import importlib
from types import SimpleNamespace

from src.inference.trackers import TRACKER_REGISTRY
from src.inference.trackers.native_tracker import DEFAULT_CONFIG_PATH


def test_tracker_module_names_preserve_registration_and_behavior():
    prefix = 'src.inference.trackers.'
    for old, new in (
        ('base', 'tracker_base'),
        ('config', 'tracker_config'),
        ('centroid', 'centroid_tracker'),
        ('iou', 'iou_tracker'),
        ('passthrough', 'passthrough_tracker'),
        ('botsort_hungarian', 'association_tracker'),
        ('botsort_native', 'native_tracker'),
        ('mask_ocsort', 'mask_tracker'),
    ):
        assert importlib.import_module(prefix + old) is importlib.import_module(prefix + new)
    assert DEFAULT_CONFIG_PATH.is_file()
    tracker = TRACKER_REGISTRY.get('centroid')()
    def detection(x):
        return SimpleNamespace(object_id=None, class_id=0, centroid=(x, 0), bbox=(x, 0, x + 5, 5))
    first = tracker.update(1, 0.0, [detection(0)])[0]
    following = tracker.update(1, 0.1, [detection(1)])[0]
    other_camera = tracker.update(2, 0.1, [detection(300)])[0]
    assert first.track_id == following.track_id == other_camera.track_id
    assert tracker._states[1]['tracks'][first.track_id].centroid == (1, 0)
    assert tracker._states[2]['tracks'][other_camera.track_id].centroid == (300, 0)
