"""Read GStreamer appsink frames and PTS over a framed stdout stream."""
import argparse
import struct
import sys
import time
if __package__:
    from .source_factory import source_uri
else:
    from source_factory import source_uri

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--capacity', type=int, default=3)
    args = parser.parse_args()
    Gst.init(None)
    pipeline = Gst.Pipeline.new('camera')
    source = Gst.ElementFactory.make('uridecodebin', 'source')
    converter = Gst.ElementFactory.make('videoconvert', 'converter')
    scaler = Gst.ElementFactory.make('videoscale', 'scaler')
    capsfilter = Gst.ElementFactory.make('capsfilter', 'format')
    sink = Gst.ElementFactory.make('appsink', 'sink')
    if not all((pipeline, source, converter, scaler, capsfilter, sink)):
        raise RuntimeError('Missing GStreamer elements required for camera ingestion')
    uri = source_uri(args.source)
    source.set_property('uri', uri)
    capsfilter.set_property('caps', Gst.Caps.from_string(
        f'video/x-raw,format=BGR,width={args.width},height={args.height}'))
    sink.set_property('emit-signals', False)
    sink.set_property('max-buffers', args.capacity)
    sink.set_property('drop', True)
    sink.set_property('sync', True)
    for element in (source, converter, scaler, capsfilter, sink):
        pipeline.add(element)
    if not converter.link(scaler) or not scaler.link(capsfilter) or not capsfilter.link(sink):
        raise RuntimeError('Unable to link GStreamer video elements')

    def pad_added(_source, pad):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps and caps.get_size() and caps.get_structure(0).get_name().startswith('video/'):
            target = converter.get_static_pad('sink')
            if not target.is_linked():
                pad.link(target)

    source.connect('pad-added', pad_added)
    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        raise RuntimeError('Unable to start GStreamer pipeline')
    bus = pipeline.get_bus()
    index = 0
    try:
        while True:
            sample = sink.emit('try-pull-sample', 100 * Gst.MSECOND)
            if sample:
                buffer = sample.get_buffer()
                pts = -1 if buffer.pts == Gst.CLOCK_TIME_NONE else int(buffer.pts)
                received_ns = time.time_ns()
                monotonic_ns = time.monotonic_ns()
                ok, mapping = buffer.map(Gst.MapFlags.READ)
                if not ok:
                    raise RuntimeError('Unable to map GStreamer frame')
                try:
                    payload = bytes(mapping.data)
                finally:
                    buffer.unmap(mapping)
                sys.stdout.buffer.write(struct.pack('!QqQQI', index, pts, received_ns, monotonic_ns, len(payload)))
                sys.stdout.buffer.write(payload)
                sys.stdout.buffer.flush()
                index += 1
                continue
            msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg:
                if msg.type == Gst.MessageType.ERROR:
                    error, debug = msg.parse_error()
                    raise RuntimeError(f'{error.message}: {debug}')
                break
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    main()
