from .base import Algorithm


class HelmetEventAlgorithm(Algorithm):
    def __init__(self, cooldown_seconds=10):
        self.cooldown = cooldown_seconds
        self.last_event = {}

    def process(self, context):
        active = set()
        for person in context.persons:
            track_id = person['track_id']
            active.add(track_id)
            if person.get('helmet') is not False:
                continue
            last = self.last_event.get(track_id)
            if last is None or context.timestamp - last >= self.cooldown:
                context.events.append({'type': 'missing_helmet', 'track_id': track_id, 'zone_id': None})
                self.last_event[track_id] = context.timestamp
        self.last_event = {key: value for key, value in self.last_event.items() if key in active}
