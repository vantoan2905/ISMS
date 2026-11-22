from pathlib import Path
from yolox.exp import Exp as BaseExp

ROOT = Path(__file__).resolve().parents[1]


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.375
        self.num_classes = 1
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.multiscale_range = 0
        self.enable_mixup = False
        self.data_dir = str(ROOT / "data/yolox_hard_hat")
        self.train_ann = "train.json"
        self.val_ann = "valid.json"
        self.test_ann = "test.json"
        self.max_epoch = 30
        self.warmup_epochs = 2
        self.no_aug_epochs = 5
        self.eval_interval = 1
        self.data_num_workers = 4
        self.output_dir = str(ROOT / "models")
        self.exp_name = "yolox_hard_hat"
        self.seed = 42
