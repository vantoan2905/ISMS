"""Evaluate person detection, RTMPose head crops and helmet classification on VOC."""
import argparse
import csv
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from rtmlib import YOLOX

from src.inference.helmet_classifier import HelmetClassifier
from src.inference.pose_head import PoseHeadExtractor, RTMOHeadExtractor


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('data/VOC2028'))
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--classifier-weights', type=Path, default=Path('models/helmet_classifier_efficientnet/best.pt'))
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--backend', choices=['rtmo', 'rtmpose'], default='rtmo')
    parser.add_argument('--iou', type=float, default=0.1)
    parser.add_argument('--max-images', type=int, default=None)
    args = parser.parse_args()
    ids = (args.source / 'ImageSets/Main' / f'{args.split}.txt').read_text().split()
    if args.max_images is not None:
        ids = ids[:args.max_images]
    if args.output is None:
        args.output = Path('models/helmet_classifier') / ('flow_test_rtmo' if args.backend == 'rtmo' else 'flow_test')
    args.output.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.backend == 'rtmo':
        pose = RTMOHeadExtractor('models/rtmo_m/rtmo_m_640x640_coco.onnx')
    else:
        detector = YOLOX('models/person_detector/yolox_tiny_humanart.onnx', model_input_size=(416, 416),
                         backend='onnxruntime', device='cpu', score_thr=0.5)
        pose = PoseHeadExtractor('models/rtmpose_tiny/rtmpose-t-body17.onnx', device='cpu')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        detector.session = ort.InferenceSession(detector.onnx_model, sess_options=options, providers=['CPUExecutionProvider'])
        pose.pose.session = ort.InferenceSession(pose.pose.onnx_model, sess_options=options, providers=['CPUExecutionProvider'])
    classifier = HelmetClassifier(args.classifier_weights, device=device)
    counts = {'images': 0, 'person_boxes': 0, 'head_crops': 0, 'gt_heads': 0,
              'matched': 0, 'unmatched_ground_truth': 0, 'unmatched_predictions': 0,
              'invalid_images': 0, 'confusion': [[0, 0], [0, 0]]}
    started = time.monotonic()
    with (args.output / 'predictions.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image_id', 'prediction_box', 'predicted_class', 'helmet_probability',
                         'matched_label', 'matched_iou'])
        for idx, image_id in enumerate(ids):
            root = ET.parse(args.source / 'Annotations' / f'{image_id}.xml').getroot()
            path = args.source / 'JPEGImages' / root.findtext('filename')
            if not path.exists():
                path = args.source / 'JPEGImages' / f'{image_id}.jpg'
            frame = cv2.imread(str(path))
            if frame is None:
                counts['invalid_images'] += 1
                continue
            gt = []
            for obj in root.findall('object'):
                name = obj.findtext('name')
                if name not in ('person', 'hat') or obj.findtext('difficult', '0') == '1':
                    continue
                box = obj.find('bndbox')
                coords = [float(box.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
                coords[0] -= 1
                coords[1] -= 1
                if coords[2] > coords[0] and coords[3] > coords[1]:
                    gt.append((coords, 1 if name == 'hat' else 0))
            if args.backend == 'rtmo':
                crops, keypoints, _ = pose.extract_with_pose(frame)
                person_count = len(keypoints)
            else:
                persons = detector(frame)
                crops = pose.extract(frame, persons)
                person_count = len(persons)
            predictions = [(crop, classifier.classify(crop.image)) for crop in crops]
            candidates = sorted(((iou(crop.box, truth[0]), pi, gi)
                                 for pi, (crop, _) in enumerate(predictions)
                                 for gi, truth in enumerate(gt)), reverse=True)
            matched_predictions, matched_gt, matches = set(), set(), {}
            for score, pi, gi in candidates:
                if score < args.iou:
                    break
                if pi not in matched_predictions and gi not in matched_gt:
                    matched_predictions.add(pi)
                    matched_gt.add(gi)
                    matches[pi] = (gi, score)
            for pi, (crop, probs) in enumerate(predictions):
                predicted = int(probs['helmet'] >= 0.5)
                match = matches.get(pi)
                if match:
                    gi, score = match
                    label = gt[gi][1]
                    counts['confusion'][label][predicted] += 1
                else:
                    label, score = '', ''
                writer.writerow([image_id, crop.box, predicted, probs['helmet'], label, score])
            counts['images'] += 1
            counts['person_boxes'] += person_count
            counts['head_crops'] += len(crops)
            counts['gt_heads'] += len(gt)
            counts['matched'] += len(matches)
            counts['unmatched_ground_truth'] += len(gt) - len(matched_gt)
            counts['unmatched_predictions'] += len(predictions) - len(matched_predictions)
            if (idx + 1) % 20 == 0:
                handle.flush()
                print(f'{idx+1}/{len(ids)} images; matched={counts["matched"]}, '
                      f'elapsed={time.monotonic()-started:.1f}s', flush=True)
    counts['backend'] = args.backend
    counts['iou_threshold'] = args.iou
    counts['classifier_threshold'] = 0.5
    counts['elapsed_seconds'] = time.monotonic() - started
    (args.output / 'metrics.json').write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()
