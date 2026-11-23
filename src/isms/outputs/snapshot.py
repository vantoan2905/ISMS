"""Store event snapshots locally."""
from pathlib import Path

import cv2


def save_snapshot(directory, event_id, frame):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{event_id}.jpg'
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f'Failed to write snapshot {path}')
    return str(path)
