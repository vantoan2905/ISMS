"""Construct stateful algorithms for one configured camera."""
from .helmet_compliance import HelmetCompliance
from .zone_rules import ZoneRules


def create_algorithms(camera, executor):
    algorithms = []
    rules = camera.get('algorithms', [])
    for rule in rules:
        if rule['type'] == 'helmet_compliance':
            algorithms.append(HelmetCompliance(
                executor.classifier, min_confidence=rule.get('confidence_threshold', 0.8),
                confirm_frames=rule.get('confirm_frames', 3),
                min_crop_size=rule.get('min_crop_size', 16),
                min_sharpness=rule.get('min_sharpness', 10.0),
                lost_timeout=rule.get('track_timeout_seconds', 2)))
    zone_rules = [rule for rule in rules if rule['type'] in ('roi_entry', 'roi_dwell')]
    if zone_rules:
        algorithms.append(ZoneRules(camera.get('rois', {}), zone_rules,
                                         lost_timeout=camera.get('track_timeout_seconds', 2),
                                         margin_pixels=camera.get('roi_margin_pixels', 3),
                                         confirm_frames=camera.get('roi_confirm_frames', 2)))
    return algorithms
