"""Run RTMO pose detection and helmet classification on video."""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.inference.helmet_classifier import HelmetClassifier
from src.inference.pose_head import RTMOHeadExtractor


# COCO 17-keypoint skeleton, zero-based indices.
SKELETON = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
            (13, 15), (12, 14), (14, 16))


def draw_pose(frame, points, scores, threshold=0.3):
    height, width = frame.shape[:2]
    valid = [i < len(points) and i < len(scores) and np.isfinite(points[i]).all()
             and np.isfinite(scores[i]) and scores[i] >= threshold
             and 0 <= points[i][0] < width and 0 <= points[i][1] < height
             for i in range(17)]
    for a, b in SKELETON:
        if valid[a] and valid[b]:
            cv2.line(frame, tuple(np.rint(points[a]).astype(int)),
                     tuple(np.rint(points[b]).astype(int)), (255, 180, 0), 2, cv2.LINE_AA)
    for index in range(17):
        if valid[index]:
            cv2.circle(frame, tuple(np.rint(points[index]).astype(int)), 3,
                       (0, 220, 255), -1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('videos', type=Path, nargs='+')
    parser.add_argument('--classifier-weights', type=Path, default=Path('models/helmet_classifier_efficientnet/best.pt'))
    parser.add_argument('--output', type=Path, default=Path('videos/rtmo_helmet'))
    parser.add_argument('--max-side', type=int, default=960)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pose = RTMOHeadExtractor('models/rtmo_m/rtmo_m_640x640_coco.onnx')
    classifier = HelmetClassifier(args.classifier_weights,
                                  device='cuda' if torch.cuda.is_available() else 'cpu')
    summary = {}
    for path in args.videos:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open video {path}')
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if width <= 0 or height <= 0 or fps <= 0:
            raise RuntimeError(f'Invalid video properties: {path}')
        scale = min(1.0, args.max_side / max(width, height))
        size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
        outpath = args.output / f'{path.stem}_pose_helmet.mp4'
        writer = cv2.VideoWriter(str(outpath), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f'Cannot write video {outpath}')
        stats = {'frames': 0, 'person_detections': 0, 'head_crops': 0,
                 'helmet_predictions': 0, 'no_helmet_predictions': 0, 'persons_without_head_crop': 0}
        started = time.monotonic()
        try:
            with (args.output / f'{path.stem}_predictions.csv').open('w', newline='') as handle:
                rows = csv.writer(handle)
                rows.writerow(['frame', 'time_seconds', 'person_index', 'x1', 'y1', 'x2', 'y2',
                               'class', 'helmet_probability'])
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    heads, keypoints, scores = pose.extract_with_pose(frame)
                    for points, confidence in zip(keypoints, scores):
                        draw_pose(frame, points, confidence)
                    for head in heads:
                        probs = classifier.classify(head.image)
                        has_helmet = probs['helmet'] >= 0.5
                        label = 'helmet' if has_helmet else 'no_helmet'
                        color = (0, 210, 0) if has_helmet else (0, 0, 255)
                        x1, y1, x2, y2 = head.box
                        cv2.rectangle(frame, (x1, y1), (x2-1, y2-1), color, 2)
                        cv2.putText(frame, f'{label} {max(probs.values()):.2f}', (x1, max(y1-4, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                        rows.writerow([stats['frames'], f"{stats['frames']/fps:.3f}", head.person_index,
                                       x1, y1, x2, y2, label, f"{probs['helmet']:.6f}"])
                        stats['helmet_predictions' if has_helmet else 'no_helmet_predictions'] += 1
                    stats['person_detections'] += len(keypoints)
                    stats['head_crops'] += len(heads)
                    stats['persons_without_head_crop'] += len(keypoints) - len(heads)
                    stats['frames'] += 1
                    writer.write(frame)
                    if stats['frames'] % 250 == 0:
                        print(f'{path.name}: {stats["frames"]} frames', flush=True)
        finally:
            capture.release()
            writer.release()
        stats['fps'] = fps
        stats['output_size'] = size
        stats['elapsed_seconds'] = time.monotonic() - started
        summary[path.name] = stats
        print(f'{path.name}: {stats}', flush=True)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
