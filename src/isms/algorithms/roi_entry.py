"""Person position and boundary hysteresis for ROI entry and exit."""
import math


def position(person):
    points, scores = person['keypoints'], person['scores']
    ankles = [points[i] for i in (15, 16) if scores[i] >= 0.3 and all(math.isfinite(v) for v in points[i])]
    if ankles:
        return tuple(sum(point[i] for point in ankles) / len(ankles) for i in range(2))
    x1, _, x2, y2 = person['box']
    return ((x1+x2)/2, y2)



def crossed_boundary(distance, inside, margin):
    """Return whether a point is clearly on the opposite side of the ROI."""
    return distance < -margin if inside else distance > margin
