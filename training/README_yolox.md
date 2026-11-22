# YOLOX head detection

YOLOX-Tiny is trained from randomly initialized weights for 30 epochs at 416×416 on the Hard Hat dataset head class. The original train, validation, and test splits are preserved. The upstream source is in training/vendor/yolox_upstream, with its Apache-2.0 license and commit. Its evaluator uses pycocotools in place of the compiled evaluator.

The annotations in data/yolox_hard_hat omit the unused Workers category. Its image directories are symlinks to the original images. Checkpoints and training logs are in models/yolox_hard_hat.

SSDLite uses 320×320 input and a different training configuration, so input resolution and training procedure must be considered when comparing test mAP.
