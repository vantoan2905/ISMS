from datetime import datetime, timezone
from .helmet_check import missing_helmets
from .zone_intrusion import intrusions


class EventEngine:
    def __init__(self, cooldown_seconds=10, helmet_class="helmet"):
        self.cooldown_seconds = cooldown_seconds
        self.helmet_class = helmet_class
        self.last_seen = {}

    def evaluate(self, camera, tracks, detections, helmet_supported=True):
        now = datetime.now(timezone.utc)
        findings = [("zone_intrusion", track_id, zone_id)
                    for track_id, zone_id in intrusions(tracks, camera.get("zones", []))]
        if camera.get("helmet_required") and helmet_supported:
            findings += [("missing_helmet", track_id, None)
                         for track_id in missing_helmets(tracks, detections, self.helmet_class)]
        events = []
        for kind, track_id, zone_id in findings:
            key = (camera["id"], kind, track_id, zone_id)
            previous = self.last_seen.get(key)
            if previous and (now - previous).total_seconds() < self.cooldown_seconds:
                continue
            self.last_seen[key] = now
            events.append({"camera_id": camera["id"], "type": kind, "track_id": track_id,
                           "zone_id": zone_id, "timestamp": now.isoformat()})
        return events
