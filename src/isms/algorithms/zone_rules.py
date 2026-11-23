"""Per-camera ROI entry, exit, and dwell rules."""
import cv2
import numpy as np

from .base import Algorithm
from .roi_entry import crossed_boundary, position
from .roi_dwell import dwell_elapsed
from ..events.models import Observation


class ZoneRules(Algorithm):
    def __init__(self, rois, rules, lost_timeout=2, margin_pixels=3, confirm_frames=2):
        self.rois = rois
        self.rules = rules
        self.timeout_ns = int(lost_timeout * 1e9)
        self.margin = margin_pixels
        self.confirm_frames = confirm_frames
        self.states = {}
        self.dwell = {r['roi_id']: int(r['threshold_ms'] * 1e6)
                      for r in rules if r['type'] == 'roi_dwell'}
        self.entries = {r['roi_id'] for r in rules if r['type'] == 'roi_entry'}

    def process(self, context):
        height, width = context.frame.shape[:2]
        for person in context.persons:
            point = position(person)
            for zone, polygon in self.rois.items():
                if zone not in self.entries and zone not in self.dwell:
                    continue
                vertices = np.asarray([(x*width, y*height) for x,y in polygon], np.float32)
                distance = cv2.pointPolygonTest(vertices, point, True)
                key = (person['track_id'], zone)
                state = self.states.get(key)
                if state is None:
                    initial_inside = distance >= 0
                    state = {'inside': initial_inside, 'last': context.pts_ns, 'entered': context.pts_ns,
                             'count': 0, 'dwell': False}
                    self.states[key] = state
                    if initial_inside and zone in self.entries:
                        context.observations.append(Observation('ROI_OCCUPIED', key[0], zone, context.pts_ns))
                    continue
                state['last'] = context.pts_ns
                opposite = crossed_boundary(distance, state['inside'], self.margin)
                state['count'] = state['count'] + 1 if opposite else 0
                if state['count'] >= self.confirm_frames:
                    state['inside'] = not state['inside']
                    state['count'] = 0
                    state['entered'] = context.pts_ns
                    state['dwell'] = False
                    if zone in self.entries:
                        kind = 'ROI_ENTER' if state['inside'] else 'ROI_EXIT'
                        context.observations.append(Observation(kind, key[0], zone, context.pts_ns))
                if (state['inside'] and zone in self.dwell and not state['dwell'] and
                        dwell_elapsed(context.pts_ns, state['entered'], self.dwell[zone])):
                    state['dwell'] = True
                    context.observations.append(Observation('ROI_DWELL', key[0], zone, context.pts_ns))
        for key, state in list(self.states.items()):
            if context.pts_ns - state['last'] > self.timeout_ns:
                if state['inside'] and key[1] in self.entries:
                    context.observations.append(Observation('ROI_EXIT', key[0], key[1], context.pts_ns))
                del self.states[key]

    def reset(self):
        self.states.clear()
