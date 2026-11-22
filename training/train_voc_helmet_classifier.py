"""Train a binary crop classifier on VOC head boxes."""
import argparse
import csv
import json
import math
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_b0, mobilenet_v3_small
from src.inference.vendor.pp_lcnet import PPLCNet_x0_5

LABELS = {'hat': 1, 'person': 0}


def crop_dataset(source, output, padding):
    split_sets = {}
    for split in ('train', 'val', 'test'):
        ids = (source / 'ImageSets' / 'Main' / f'{split}.txt').read_text().split()
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate image IDs in {split}')
        split_sets[split] = set(ids)
    if any(split_sets[a] & split_sets[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('Source image IDs overlap between splits')
    for split, ids in split_sets.items():
        directory = output / split
        directory.mkdir(parents=True, exist_ok=True)
        counts = Counter()
        with (directory / 'manifest.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['crop', 'source_id', 'label', 'source_label', 'bbox'])
            for image_id in sorted(ids):
                root = ET.parse(source / 'Annotations' / f'{image_id}.xml').getroot()
                image_path = source / 'JPEGImages' / root.findtext('filename')
                if not image_path.exists():
                    image_path = source / 'JPEGImages' / f'{image_id}.jpg'
                with Image.open(image_path) as im:
                    image = im.convert('RGB')
                width, height = image.size
                for index, obj in enumerate(root.findall('object')):
                    label = obj.findtext('name')
                    if label not in LABELS or obj.findtext('difficult', '0') == '1':
                        continue
                    node = obj.find('bndbox')
                    x1, y1, x2, y2 = [float(node.findtext(key)) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
                    w, h = x2 - x1, y2 - y1
                    if w <= 0 or h <= 0:
                        continue
                    # Pascal VOC starts at 1; convert to a zero-based, exclusive crop.
                    left = max(0, math.floor(x1 - 1 - w * padding))
                    top = max(0, math.floor(y1 - 1 - h * padding))
                    right = min(width, math.ceil(x2 + w * padding))
                    bottom = min(height, math.ceil(y2 + h * padding))
                    if right <= left or bottom <= top:
                        continue
                    class_name = 'helmet' if label == 'hat' else 'no_helmet'
                    dest = directory / class_name
                    dest.mkdir(exist_ok=True)
                    filename = f'{image_id}_{index}.jpg'
                    image.crop((left, top, right, bottom)).save(dest / filename, quality=95)
                    writer.writerow([f'{class_name}/{filename}', image_id, LABELS[label], label,
                                     f'{x1},{y1},{x2},{y2}'])
                    counts[class_name] += 1
        print(f'{split} crops: {dict(counts)}', flush=True)


class Crops(Dataset):
    def __init__(self, path, train=False):
        self.path = path
        with (path / 'manifest.csv').open(newline='') as f:
            self.rows = list(csv.DictReader(f))
        augmentation = [transforms.RandomHorizontalFlip()] if train else []
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)), *augmentation, transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.path / row['crop']) as image:
            return self.transform(image.convert('RGB')), int(row['label'])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    matrix = torch.zeros((2, 2), dtype=torch.int64)
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(1).cpu()
        for truth, prediction in zip(labels, predictions):
            matrix[truth, prediction] += 1
    return matrix.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', choices=['mobilenet_v3_small', 'pplcnet_x0_5', 'efficientnet_b0'], default='mobilenet_v3_small')
    parser.add_argument('--reuse-crops', action='store_true')
    parser.add_argument('--source', type=Path, default=Path('data/VOC2028'))
    parser.add_argument('--crops', type=Path, default=Path('data/voc_helmet_crops'))
    parser.add_argument('--output', type=Path, default=Path('models/helmet_classifier'))
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--padding', type=float, default=0.35)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.reuse_crops:
        crop_dataset(args.source, args.crops, args.padding)
    train = Crops(args.crops / 'train', True)
    valid = Crops(args.crops / 'val')
    test = Crops(args.crops / 'test')
    labels = [int(row['label']) for row in train.rows]
    counts = Counter(labels)
    if set(counts) != {0, 1}:
        raise ValueError('Both classes are required')
    weights = [1 / counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights, num_samples=2 * min(counts.values()), replacement=True)
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, num_workers=args.workers)
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=args.workers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device, flush=True)
    if args.architecture == 'pplcnet_x0_5':
        model = PPLCNet_x0_5(class_num=2).to(device)
    elif args.architecture == 'efficientnet_b0':
        model = efficientnet_b0(weights=None, num_classes=2).to(device)
    else:
        model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    args.output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        matrix = evaluate(model, valid_loader, device)
        recalls = [matrix[i][i] / max(1, sum(matrix[i])) for i in range(2)]
        score = sum(recalls) / 2
        print(f'epoch {epoch + 1}/{args.epochs} loss={total_loss/len(train_loader):.4f} '
              f'val_balanced_accuracy={score:.4f} confusion={matrix}', flush=True)
        if score > best:
            best = score
            torch.save({'model': model.state_dict(), 'epoch': epoch+1, 'val_balanced_accuracy': score,
                        'classes': ['no_helmet', 'helmet'], 'architecture': args.architecture,
                        'input_size': 128, 'normalization': {'mean': [0.5]*3, 'std': [0.5]*3}},
                       args.output / 'best.pt')
    checkpoint = torch.load(args.output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    matrix = evaluate(model, test_loader, device)
    (args.output / 'metrics.json').write_text(json.dumps({
        'best_epoch': checkpoint['epoch'], 'validation_balanced_accuracy': checkpoint['val_balanced_accuracy'],
        'test_confusion_rows_actual_columns_predicted': matrix,
        'classes': checkpoint['classes'], 'train_counts': dict(counts), 'padding': args.padding,
    }, indent=2))
    print('test confusion rows actual, columns predicted:', matrix, flush=True)


if __name__ == '__main__':
    main()
