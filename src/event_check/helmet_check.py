def missing_helmets(tracks, detections, helmet_class="helmet"):
    helmets = [d for d in detections if d.label == helmet_class]
    for track in tracks:
        x1, y1, x2, y2 = track.detection.box
        head_bottom = y1 + (y2 - y1) * 0.35
        has_helmet = any(x1 <= helmet.center[0] <= x2 and y1 <= helmet.center[1] <= head_bottom
                         for helmet in helmets)
        if not has_helmet:
            yield track.id
