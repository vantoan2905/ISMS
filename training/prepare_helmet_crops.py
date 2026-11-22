"""Export positive helmet crops from COCO boxes without altering the source."""
import argparse
import csv
import json
import math
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/helmet'))
    parser.add_argument('--output', type=Path, default=Path('data/helmet_classification'))
    parser.add_argument('--padding', type=float, default=0.5)
    args = parser.parse_args()
    if not math.isfinite(args.padding) or args.padding < 0:
        parser.error('padding must be non-negative')
    for split in ('train', 'valid', 'test'):
        folder = args.data / split
        data = json.loads((folder / '_annotations.coco.json').read_text())
        names = {image['id']: image['file_name'] for image in data['images']}
        categories = {category['id'] for category in data['categories'] if category['name'] == 'helmet'}
        target = args.output / split / 'helmet'
        target.mkdir(parents=True, exist_ok=True)
        count = 0
        with (args.output / split / 'manifest.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['crop', 'source_image', 'annotation_id', 'category_id', 'box_xywh'])
            for ann in data['annotations']:
                if ann['category_id'] not in categories or ann.get('iscrowd', 0):
                    continue
                image = cv2.imread(str(folder / names[ann['image_id']]))
                if image is None:
                    raise FileNotFoundError(names[ann['image_id']])
                x, y, w, h = ann['bbox']
                height, width = image.shape[:2]
                left = max(0, math.floor(x - w * args.padding))
                top = max(0, math.floor(y - h * args.padding))
                right = min(width, math.ceil(x + w * (1 + args.padding)))
                bottom = min(height, math.ceil(y + h * (1 + args.padding)))
                if right <= left or bottom <= top:
                    continue
                name = f"{ann['image_id']}_{ann['id']}.jpg"
                if not cv2.imwrite(str(target / name), image[top:bottom, left:right]):
                    raise OSError(f'Failed to write {target / name}')
                writer.writerow([f'helmet/{name}', names[ann['image_id']], ann['id'], ann['category_id'], ann['bbox']])
                count += 1
        print(f'{split}: {count} positive helmet crops')


if __name__ == '__main__':
    main()
