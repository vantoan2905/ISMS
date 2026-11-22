"""Train a head detector from COCO annotations."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as F


class HardHat(Dataset):
    def __init__(self, folder, augment=False):
        self.folder = folder
        self.coco = COCO(str(folder / "_annotations.coco.json"))
        self.ids = sorted(self.coco.imgs)
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        metadata = self.coco.imgs[image_id]
        with Image.open(self.folder / metadata["file_name"]) as im:
            image = F.to_tensor(im.convert("RGB"))
        _, height, width = image.shape
        boxes = []
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id], catIds=[1])):
            x, y, w, h = ann["bbox"]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.augment and random.random() < 0.5:
            image = F.hflip(image)
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        return image, {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64),
                       "image_id": image_id}


def collate(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions = []
    for images, targets in loader:
        outputs = model([im.to(device) for im in images])
        for output, target in zip(outputs, targets):
            boxes = output["boxes"].cpu()
            for box, score in zip(boxes, output["scores"].cpu()):
                x1, y1, x2, y2 = box.tolist()
                predictions.append({"image_id": target["image_id"], "category_id": 1,
                                    "bbox": [x1, y1, x2-x1, y2-y1], "score": score.item()})
    if not predictions:
        return 0.0
    coco = loader.dataset.coco
    evaluation = COCOeval(coco, coco.loadRes(predictions), "bbox")
    evaluation.params.imgIds = loader.dataset.ids
    evaluation.params.catIds = [1]
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    return float(evaluation.stats[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/Hard_Hat"))
    parser.add_argument("--output", type=Path, default=Path("models/hard_hat"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    train = DataLoader(HardHat(args.data / "train", True), batch_size=args.batch_size,
                       shuffle=True, num_workers=args.workers, collate_fn=collate)
    valid = DataLoader(HardHat(args.data / "valid"), batch_size=args.batch_size,
                       num_workers=args.workers, collate_fn=collate)
    test = DataLoader(HardHat(args.data / "test"), batch_size=args.batch_size,
                      num_workers=args.workers, collate_fn=collate)
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for step, (images, targets) in enumerate(train):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items() if k != "image_id"} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, batch {step}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch+1} batch {step}/{len(train)} loss {loss.item():.4f}", flush=True)
        scheduler.step()
        score = evaluate(model, valid, device)
        print(f"epoch {epoch+1}: loss={total/len(train):.4f} val_mAP={score:.4f}", flush=True)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epoch": epoch+1, "val_map": score, "classes": ["background", "head"],
                      "architecture": "ssdlite320_mobilenet_v3_large", "seed": args.seed}
        torch.save(checkpoint, args.output / "last.pt")
        if score > best:
            best = score
            torch.save(checkpoint, args.output / "best.pt")
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    score = evaluate(model, test, device)
    (args.output / "metrics.json").write_text(json.dumps(
        {"best_epoch": checkpoint["epoch"], "validation_map": checkpoint["val_map"],
         "test_map": score, "classes": checkpoint["classes"]}, indent=2))


if __name__ == "__main__":
    main()
