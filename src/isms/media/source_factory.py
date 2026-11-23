"""Convert local files and stream addresses into GStreamer source URIs."""
from pathlib import Path


def source_uri(source):
    return source if "://" in source else Path(source).resolve().as_uri()
