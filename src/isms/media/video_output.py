"""Draw the diagnostic overlay on a copy of a processed frame."""
import cv2
import numpy as np


def render_overlay(context, camera, relay_on):
    image = context.frame.copy()
    height, width = image.shape[:2]
    for polygon in camera.get('rois', {}).values():
        vertices = np.asarray([(int(x*width), int(y*height)) for x, y in polygon], np.int32)
        cv2.polylines(image, [vertices], True, (255, 180, 0), 2)
    for person in context.persons:
        x1, y1, x2, y2 = person['box']
        label = 'UNKNOWN' if person.get('helmet') is None else ('HELMET' if person['helmet'] else 'NO_HELMET')
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(image, f"{person['track_id']} {label}", (max(0, x1), max(15, y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        for point, score in zip(person['keypoints'], person['scores']):
            if score >= 0.3 and np.isfinite(point).all():
                cv2.circle(image, tuple(map(int, point)), 2, (0, 255, 255), -1)
    cv2.putText(image, f"RELAY {'ON' if relay_on else 'OFF'}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if relay_on else (0, 220, 0), 2)
    return image
