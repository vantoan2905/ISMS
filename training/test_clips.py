"""Run the trained head detector on videos and save annotated copies."""
import argparse
import csv
from pathlib import Path

import cv2
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms.functional import to_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("models/hard_hat/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("videos/hard_hat_test"))
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--max-side", type=int, default=960)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=2)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)

    for path in args.videos:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video properties: {path}")
        scale = min(1.0, args.max_side / max(width, height))
        size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
        output = args.output / f"{path.stem}_detected.mp4"
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot create video: {output}")
        frame_count = 0
        detected_frames = 0
        detections_count = 0
        csv_path = args.output / f"{path.stem}_detections.csv"
        try:
            with csv_path.open("w", newline="") as file:
                rows = csv.writer(file)
                rows.writerow(["frame", "time_seconds", "score", "x1", "y1", "x2", "y2"])
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = to_tensor(rgb).to(device)
                    with torch.inference_mode():
                        prediction = model([image])[0]
                    boxes = prediction["boxes"].cpu()
                    scores = prediction["scores"].cpu()
                    count = 0
                    for box, score in zip(boxes, scores):
                        if score.item() < args.threshold:
                            continue
                        x1, y1, x2, y2 = map(int, box.tolist())
                        rows.writerow([frame_count, f"{frame_count / fps:.3f}", f"{score.item():.4f}",
                                       x1, y1, x2, y2])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"head {score.item():.2f}", (x1, max(16, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        count += 1
                    detected_frames += count > 0
                    detections_count += count
                    writer.write(frame)
                    frame_count += 1
                    if frame_count % 250 == 0:
                        print(f"{path.name}: {frame_count} frames", flush=True)
        finally:
            capture.release()
            writer.release()
        print(f"{path.name}: frames={frame_count}, frames_with_head={detected_frames}, "
              f"detections={detections_count}, output={output}", flush=True)


if __name__ == "__main__":
    main()
