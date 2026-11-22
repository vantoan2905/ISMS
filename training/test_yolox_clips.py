"""Run YOLOX head detection on video frames and save annotated copies."""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training/vendor/yolox_upstream'))
from yolox.data.data_augment import preproc
from src.inference.passer import parse_yolox
from yolox_hard_hat import Exp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('videos', nargs='+', type=Path)
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/yolox_hard_hat/best_weights.pth')
    parser.add_argument('--output', type=Path, default=ROOT / 'videos/yolox_hard_hat_test')
    parser.add_argument('--threshold', type=float, default=0.4)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Exp().get_model().to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True)['model'])
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.videos:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {path}')
        width, height = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if width <= 0 or height <= 0 or fps <= 0:
            raise RuntimeError(f'Invalid video properties: {path}')
        scale = min(1.0, 960 / max(width, height))
        size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
        target = args.output / f'{path.stem}_detected.mp4'
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f'Cannot create {target}')
        frames = with_detections = detections = 0
        try:
            with (args.output / f'{path.stem}_detections.csv').open('w', newline='') as f:
                rows = csv.writer(f)
                rows.writerow(['frame', 'time_seconds', 'score', 'x1', 'y1', 'x2', 'y2'])
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    img, _ = preproc(frame, (416, 416))
                    with torch.inference_mode():
                        output = model(torch.from_numpy(img).unsqueeze(0).to(device))
                        predictions = parse_yolox(output, frame.shape[:2], confidence=args.threshold)
                    count = 0
                    for detection in predictions:
                        x1, y1, x2, y2 = detection.box
                        score = detection.confidence
                        rows.writerow([frames, f'{frames/fps:.3f}', f'{score:.4f}', x1, y1, x2, y2])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{detection.label} {score:.2f}', (x1, max(16, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        count += 1
                    with_detections += count > 0
                    detections += count
                    writer.write(frame)
                    frames += 1
        finally:
            cap.release()
            writer.release()
        print(f'{path.name}: frames={frames}, frames_with_head={with_detections}, detections={detections}', flush=True)


if __name__ == '__main__':
    main()
