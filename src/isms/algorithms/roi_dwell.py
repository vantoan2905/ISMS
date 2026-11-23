"""Video timestamp based ROI dwell checks."""


def dwell_elapsed(pts_ns, entered_ns, threshold_ns):
    return pts_ns - entered_ns >= threshold_ns
