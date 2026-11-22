"""Defaults used when a tracker is constructed without explicit options."""

from numbers import Integral

DEFAULT_FPS = 10.0
DEFAULT_TRACKING_DISTANCE = 120.0
DEFAULT_TRACKING_MAX_AGE = 15
DEFAULT_BOTSORT_MATCH_IOU = 0.3


def is_valid_object_id(value):
    """Accept unsigned upstream IDs except the all-ones unassigned marker."""
    return isinstance(value, Integral) and not isinstance(value, bool) and 0 <= int(value) < 2**64 - 1
