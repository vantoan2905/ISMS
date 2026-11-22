# Inference

- `pipeline.py` processes live camera frames and passes detections to the rule engine and event service.
- `detector.py` and `tracker.py` implement the live YOLO detector and centroid tracker.
- `video_pipeline.py` processes video files using the optional native extension. The `Pipeline` class is also available through `src.inference.pipeline`.
- `trackers/` contains the optional tracker implementations and native BoT-SORT code.
- `core/` contains the native C++ decoder, encoder, ONNX inference, output parser and Python bindings.

The live and file pipelines have independent detection implementations. The camera manager starts the live pipeline.

## RTMO-m helmet flow

`RTMOHeadExtractor` in `pose_head.py` runs the COCO 17-keypoint RTMO-m ONNX model and derives padded head crops from the first five face keypoints. `HelmetClassifier` classifies these crops. `training/test_pose_clips.py` uses this flow to produce annotated videos, while `training/evaluate_pose_helmet.py` defaults to RTMO (`--backend rtmpose` selects the earlier top-down flow).

The RTMO-m ONNX model at `models/rtmo_m/rtmo_m_640x640_coco.onnx` is a third-party export distributed at https://github.com/namas191297/rtmo-ort/releases/latest/download/rtmo_m_640x640_coco.onnx. SHA-256: `7fa00bed51f00133e96745b62604c36bb2d49529dfdef4e9f8ece6c7f3f7694a`. The model is labeled COCO by its distributor; its export provenance has not been independently verified against the original MMPose checkpoint.

The RTMO video and VOC evaluation scripts default to the EfficientNet-B0 helmet classifier at `models/helmet_classifier_efficientnet/best.pt`. Pass `--classifier-weights models/helmet_classifier/best.pt` to use the earlier MobileNetV3-Small checkpoint. EfficientNet-B0 was trained from random initialization on the existing VOC crops with the same split and 128×128 preprocessing as MobileNetV3-Small.
