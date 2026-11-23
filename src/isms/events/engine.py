"""Persist violation lifecycles and control a shared mock relay."""
import asyncio
import logging
from datetime import datetime, timezone
import uuid
from pathlib import Path

from .dispatcher import EventDispatcher
from ..outputs.mock_relay import MockRelay
from ..outputs.snapshot import save_snapshot

logger = logging.getLogger(__name__)


class EventEngine:
    def __init__(self, database, snapshot_dir='snapshots', relay_enabled=True):
        self.database = database
        self.snapshot_dir = Path(snapshot_dir)
        self.relay_enabled = relay_enabled
        self.active = {}
        for event in database.list_active_incidents():
            if event['type'] in ('ROI_ENTER', 'ROI_OCCUPIED', 'NO_HELMET'):
                kind = 'ROI_ENTER' if event['type'].startswith('ROI_') else 'NO_HELMET'
                self.active[(event['camera_id'], event['rule_id'], event['track_id'], kind)] = event
        self.relay = MockRelay(relay_enabled and any(key[3] == 'ROI_ENTER' for key in self.active))
        self.lock = asyncio.Lock()
        self.dispatcher = EventDispatcher()

    @property
    def relay_on(self):
        return self.relay.is_on

    @relay_on.setter
    def relay_on(self, value):
        self.relay.set_state(value)

    async def handle(self, camera_id, context):
        async with self.lock:
            for obs in context.observations:
                key_type = 'ROI_ENTER' if obs.type in ('ROI_ENTER', 'ROI_OCCUPIED', 'ROI_EXIT') else 'NO_HELMET'
                key = (camera_id, obs.rule_id, obs.track_id, key_type)
                if obs.type in ('ROI_EXIT', 'HELMET_RESOLVED'):
                    event = self.active.pop(key, None)
                    if event:
                        event['status'] = 'RESOLVED'
                        event['ended_at_utc'] = context.received_at_utc.isoformat()
                        await asyncio.to_thread(self.database.update_incident, event)
                        self._notify(event)
                elif obs.type in ('ROI_ENTER', 'ROI_OCCUPIED', 'NO_HELMET', 'ROI_DWELL'):
                    if obs.type != 'ROI_DWELL' and key in self.active:
                        continue
                    event_id = str(uuid.uuid4())
                    snapshot = await asyncio.to_thread(save_snapshot, self.snapshot_dir, event_id, context.frame)
                    event = {'event_id': event_id, 'camera_id': camera_id, 'track_id': obs.track_id,
                             'rule_id': obs.rule_id, 'type': obs.type,
                             'severity': 'CRITICAL' if obs.type.startswith('ROI_') else 'WARNING',
                             'status': 'RESOLVED' if obs.type == 'ROI_DWELL' else 'ACTIVE',
                             'timestamp': context.received_at_utc.isoformat(),
                             'started_at_utc': context.received_at_utc.isoformat(),
                             'ended_at_utc': context.received_at_utc.isoformat() if obs.type == 'ROI_DWELL' else None,
                             'snapshot_path': str(snapshot), 'zone_id': obs.rule_id,
                             'metadata': {'frame_id': context.frame_id, 'pts_ns': obs.pts_ns,
                                          'confidence': obs.confidence}}
                    await asyncio.to_thread(self.database.insert_incident, event)
                    if obs.type != 'ROI_DWELL':
                        self.active[key] = event
                    logger.info('Event %s %s camera=%s track=%s rule=%s', event_id, obs.type,
                                camera_id, obs.track_id, obs.rule_id)
                    self._notify(event)
            relay = self.relay_enabled and any(k[3] == 'ROI_ENTER' for k in self.active)
            if relay != self.relay_on:
                self.relay_on = relay
                logger.warning('Mock relay %s', 'ON' if relay else 'OFF')

    async def reset_camera(self, camera_id, ended_at):
        async with self.lock:
            for key, event in list(self.active.items()):
                if key[0] != camera_id:
                    continue
                event['status'] = 'RESOLVED'
                event['ended_at_utc'] = ended_at.isoformat()
                event['metadata']['resolution_reason'] = 'stream_timestamp_reset'
                await asyncio.to_thread(self.database.update_incident, event)
                self._notify(event)
                del self.active[key]
            self.relay_on = self.relay_enabled and any(k[3] == 'ROI_ENTER' for k in self.active)

    async def offline(self, camera_id, message):
        logger.error('Camera %s offline: %s; active relay state retained', camera_id, message)
        # Never resolve intrusion solely because the video source failed.
        async with self.lock:
            event = {'event_id': str(uuid.uuid4()), 'camera_id': camera_id, 'track_id': None,
                     'rule_id': 'camera', 'type': 'CAMERA_OFFLINE', 'severity': 'WARNING',
                     'status': 'ACTIVE', 'timestamp': datetime.now(timezone.utc).isoformat(),
                     'snapshot_path': None, 'metadata': {'reason': str(message)}}
            event['started_at_utc'] = event['timestamp']
            event['ended_at_utc'] = None
            await asyncio.to_thread(self.database.insert_incident, event)
            self._notify(event)

    def _notify(self, event):
        self.dispatcher.publish(event)

    def subscribe(self):
        return self.dispatcher.subscribe()

    def unsubscribe(self, queue):
        self.dispatcher.unsubscribe(queue)
