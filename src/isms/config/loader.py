"""Load and validate camera and model configuration."""
from pathlib import Path

import yaml

from .schema import validate_config


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(config, dict) and "model_config" in config:
        model_path = Path(path).parent / config["model_config"]
        config["models"] = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    return validate_config(config)
