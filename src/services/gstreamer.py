"""GStreamer frame input and annotated output for an individual camera."""
import os
import select
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np


class GStreamerCamera:
    def __init__(self, camera):
        self.camera = camera
        self.width = int(camera.get('width', 640))
        self.height = int(camera.get('height', 480))
        self.fps = float(camera.get('fps', 15))
        if self.width <= 0 or self.height <= 0 or self.width % 4 or self.fps <= 0:
            raise ValueError('GStreamer requires positive size/fps and width divisible by 4')
        self.rate = Fraction(str(self.fps)).limit_denominator(1001)
        self.reader = None
        self.writer = None

    def open(self):
        source = self.camera['source']
        if isinstance(source, int):
            input_elements = ['v4l2src', f'device=/dev/video{source}']
        else:
            uri = str(source)
            if '://' not in uri:
                uri = Path(uri).resolve().as_uri()
            input_elements = ['uridecodebin', f'uri={uri}']
        caps = f'video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.rate.numerator}/{self.rate.denominator}'
        command = ['gst-launch-1.0', '-q', *input_elements, '!', 'videoconvert', '!', 'videoscale', '!',
                   'videorate', '!', caps, '!', 'fdsink', 'fd=1', 'sync=false']
        self.reader = subprocess.Popen(command, stdout=subprocess.PIPE)
        output = self.camera.get('output')
        if output:
            if isinstance(source, str) and '://' not in source and Path(source).resolve() == Path(output).resolve():
                raise ValueError('Camera input and output paths must be different')
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not self.camera.get('overwrite_output', False):
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                descriptor = os.open(path, flags, 0o644)
                os.close(descriptor)
            command = ['gst-launch-1.0', '-q', 'fdsrc', 'fd=0', '!', 'rawvideoparse',
                       'format=bgr', f'width={self.width}', f'height={self.height}',
                       f'framerate={self.rate.numerator}/{self.rate.denominator}', '!', 'videoconvert',
                       '!', 'x264enc', 'tune=zerolatency', '!', 'h264parse', '!', 'mp4mux',
                       '!', 'filesink', f'location={path}']
            self.writer = subprocess.Popen(command, stdin=subprocess.PIPE)

    def read(self, stop=None):
        size = self.width * self.height * 3
        chunks = bytearray()
        while len(chunks) < size:
            if stop is not None and stop.is_set():
                return None
            ready, _, _ = select.select([self.reader.stdout], [], [], 0.25)
            if not ready:
                continue
            part = os.read(self.reader.stdout.fileno(), size-len(chunks))
            if not part:
                if chunks:
                    raise RuntimeError('Incomplete GStreamer frame')
                code = self.reader.wait()
                if code:
                    raise RuntimeError(f'GStreamer input exited with status {code}')
                return None
            chunks.extend(part)
        return np.frombuffer(chunks, np.uint8).reshape(self.height, self.width, 3).copy()

    def write(self, frame):
        if self.writer:
            if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
                raise ValueError('Unexpected output frame dimensions or format')
            self.writer.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        error = None
        if self.writer:
            try:
                self.writer.stdin.close()
                if self.writer.wait(timeout=15):
                    error = RuntimeError('GStreamer output failed')
            except (BrokenPipeError, subprocess.TimeoutExpired) as exc:
                self.writer.kill()
                self.writer.wait()
                error = RuntimeError(f'GStreamer output failed: {exc}')
        if self.reader:
            if self.reader.poll() is None:
                self.reader.terminate()
                try:
                    self.reader.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.reader.kill()
                    self.reader.wait()
            self.reader.stdout.close()
        if error:
            raise error
