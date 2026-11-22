"""Use the held-out test split with YOLOX's normal COCO evaluator."""
from yolox_hard_hat import Exp as TrainExp
from yolox.data import COCODataset, ValTransform


class Exp(TrainExp):
    def get_eval_dataset(self, **kwargs):
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.test_ann,
            name="test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        )
