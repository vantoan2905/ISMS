"""Validate the camera configuration used by the runtime."""


def validate_config(config):
    if not isinstance(config, dict) or not isinstance(config.get('cameras'), list):
        raise ValueError('Config must contain a camera list')
    models = config.get('models', {})
    if not models.get('pose'):
        raise ValueError('models.pose is required')
    ids = set()
    for camera in config['cameras']:
        camera_id = camera.get('id')
        if not camera_id or camera_id in ids:
            raise ValueError(f'Camera ID is missing or duplicated: {camera_id}')
        ids.add(camera_id)
        if not camera.get('source'):
            raise ValueError(f'Camera {camera_id} has no source')
        rules = camera.get('algorithms', [])
        if not isinstance(rules, list):
            raise ValueError(f'Camera {camera_id} algorithms must be a list')
        for rule in rules:
            kind = rule.get('type')
            if kind not in ('helmet_compliance', 'roi_entry', 'roi_dwell'):
                raise ValueError(f'Unknown algorithm {kind} on camera {camera_id}')
            if kind == 'helmet_compliance' and not models.get('helmet_classifier'):
                raise ValueError('models.helmet_classifier is required')
            if kind in ('roi_entry', 'roi_dwell'):
                roi = rule.get('roi_id')
                polygon = camera.get('rois', {}).get(roi)
                if not isinstance(polygon, list) or len(polygon) < 3 or any(
                    not isinstance(p, (list, tuple)) or len(p) != 2 or
                    any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in p)
                    for p in polygon
                ):
                    raise ValueError(f'Invalid normalized ROI {roi} on camera {camera_id}')
                if kind == 'roi_dwell' and rule.get('threshold_ms', 0) <= 0:
                    raise ValueError('ROI dwell threshold_ms must be positive')
    if config.get('runtime', {}).get('max_frame_queue', 3) < 1:
        raise ValueError('max_frame_queue must be positive')
    return config

