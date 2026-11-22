"""Offline video processing using the native inference modules."""

from pathlib import Path

import cv2


class Pipeline:
    """Process files or streams using the C++ decoder, ONNX model and H.264 encoder."""

    def __init__(self, srcs: list):
        from .core import load_native

        native = load_native()
        self.decode = native.Decoder
        self.encode = native.Encoder
        self.infer = native.Inference
        self.passer = native.parse_detections
        self.srcs = list(srcs)

    def pipeline(self, outputs, model_path=None, threshold=0.4):
        """Write one MOV per source, drawing detections if an ONNX model is supplied."""
        outputs = [Path(path) for path in outputs]
        if len(outputs) != len(self.srcs):
            raise ValueError("Provide one output path per source")
        inputs = [Path(source) for source in self.srcs]
        if model_path is not None:
            inputs.append(Path(model_path))
        for i, output in enumerate(outputs):
            for other in inputs + outputs[:i]:
                if output.resolve() == other.resolve() or (
                    output.exists() and other.exists() and output.samefile(other)
                ):
                    raise ValueError(f"Output path is already in use: {output}")

        model = self.infer(str(model_path)) if model_path is not None else None
        for source, output in zip(self.srcs, outputs):
            decoder = self.decode(str(source))
            frame = decoder.read()
            if frame is None:
                raise ValueError(f"No frames in {source}")
            height, width = frame.shape[:2]
            encoder = self.encode(str(output), width, height, decoder.fps)
            while frame is not None:
                if model is not None:
                    detections = self.passer(model.run(frame), width, height, threshold)
                    self.draw(frame, detections)
                encoder.write(frame)
                frame = decoder.read()
            encoder.finish()
        return outputs

    @staticmethod
    def draw(frame, detections):
        for class_id, _, (x1, y1, x2, y2) in detections:
            cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (0, 220, 0), 2)
            cv2.putText(frame, str(class_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 220, 0), 2)

