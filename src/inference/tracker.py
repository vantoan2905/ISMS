from dataclasses import dataclass
from .detector import Detection


@dataclass(frozen=True)
class Track:
    id: int
    detection: Detection


class CentroidTracker:
    def __init__(self, max_distance: int = 80, max_missed: int = 12):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self._next_id = 1
        self._active: dict[int, tuple[tuple[int, int], int]] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        candidates = []
        for index, detection in enumerate(detections):
            if detection.label != "person":
                continue
            for track_id, (center, _) in self._active.items():
                distance = sum((a - b) ** 2 for a, b in zip(center, detection.center))
                if distance <= self.max_distance ** 2:
                    candidates.append((distance, track_id, index))
        assigned: dict[int, int] = {}
        used = set()
        for _, track_id, index in sorted(candidates):
            if track_id not in used and index not in assigned:
                assigned[index] = track_id
                used.add(track_id)
        output = []
        next_active = {track_id: (center, missed + 1) for track_id, (center, missed) in self._active.items()
                       if track_id not in used and missed + 1 <= self.max_missed}
        for index, detection in enumerate(detections):
            if detection.label != "person":
                continue
            track_id = assigned.get(index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
            next_active[track_id] = (detection.center, 0)
            output.append(Track(track_id, detection))
        self._active = next_active
        return output
