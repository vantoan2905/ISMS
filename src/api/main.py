"""Compatibility alias for the ISMS API module."""
import sys
from src.isms.api import main as _module
sys.modules[__name__] = _module
