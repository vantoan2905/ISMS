import math
from src.isms.algorithms.roi_entry import position

from .base import Algorithm
from ..event_check.zone_intrusion import inside_polygon


class ROIEntryAlgorithm(Algorithm):
    def __init__(self, zones):
        self.zones = zones
        self.inside = set()

    def process(self, context):
        current = set()
        for person in context.persons:
            person['position'] = position(person)
            person['zones'] = []
            for zone in self.zones:
                key = (person['track_id'], zone['id'])
                if inside_polygon(person['position'], zone['polygon']):
                    person['zones'].append(zone['id'])
                    current.add(key)
                    if key not in self.inside:
                        context.events.append({'type': 'zone_intrusion', 'track_id': key[0], 'zone_id': key[1]})
        self.inside = current


class ROIStationaryAlgorithm(Algorithm):
    def __init__(self, duration_seconds=10, movement_pixels=30):
        self.duration = duration_seconds
        self.movement = movement_pixels
        self.states = {}

    def process(self, context):
        current = set()
        for person in context.persons:
            for zone in person.get('zones', []):
                key = (person['track_id'], zone)
                current.add(key)
                point = person['position']
                state = self.states.get(key)
                if state is None or math.dist(point, state['anchor']) > self.movement:
                    self.states[key] = {'anchor': point, 'since': context.timestamp, 'emitted': False}
                elif not state['emitted'] and context.timestamp - state['since'] >= self.duration:
                    context.events.append({'type': 'stationary_in_zone', 'track_id': key[0], 'zone_id': zone})
                    state['emitted'] = True
        self.states = {key: value for key, value in self.states.items() if key in current}
