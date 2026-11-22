def inside_polygon(point: tuple[int, int], polygon: list[list[int]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        previous = current
    return inside


def intrusions(tracks, zones):
    for track in tracks:
        x1, _, x2, y2 = track.detection.box
        foot = ((x1 + x2) // 2, y2)
        for zone in zones:
            if zone.get("restricted") and inside_polygon(foot, zone["polygon"]):
                yield track.id, zone["id"]
