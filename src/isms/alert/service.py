"""Consume new safety incidents and forward announcements to device adapters."""
import asyncio
import logging
from urllib.parse import urlsplit

from .devices import Display, Speaker
from .messages_vi import MESSAGES as VIETNAMESE_MESSAGES

logger = logging.getLogger(__name__)

MESSAGES = {
    "ROI_ENTER": "A person has entered a hazardous area.",
    "ROI_OCCUPIED": "A person is in a hazardous area.",
    "ROI_DWELL": "A person has remained in a hazardous area beyond the allowed time.",
    "NO_HELMET": "A person may not be wearing a safety helmet.",
}


class AlertService:
    def __init__(self, devices, language="en"):
        if language not in ("en", "vi"):
            raise ValueError("alert_language must be en or vi")
        self.devices = devices
        self.messages = MESSAGES if language == "en" else VIETNAMESE_MESSAGES
        self.worker = None

    @classmethod
    def from_config(cls, config):
        adapters = {"display": Display, "speaker": Speaker}
        devices = []
        for item in config.get("alert_devices", []):
            if not isinstance(item, dict) or item.get("type") not in adapters:
                raise ValueError("alert_devices entries require type display or speaker")
            endpoint = item.get("endpoint")
            if endpoint is not None:
                url = urlsplit(endpoint)
                if url.scheme not in ("http", "https") or not url.hostname:
                    raise ValueError("Alert device endpoint must be an HTTP(S) URL")
            devices.append(adapters[item["type"]](item.get("name", item["type"]), endpoint))
        return cls(devices, config.get("alert_language", "en"))

    def start(self, engine):
        if self.devices:
            self.queue = engine.subscribe()
            self.worker = asyncio.create_task(self._run(), name="alert-devices")

    async def close(self, engine):
        if self.worker is not None:
            engine.unsubscribe(self.queue)
            await self.queue.put(None)
            await self.worker
            self.worker = None

    async def _run(self):
        while (event := await self.queue.get()) is not None:
            kind = event["type"]
            if kind not in self.messages or (event["status"] != "ACTIVE" and kind != "ROI_DWELL"):
                continue
            payload = {key: event[key] for key in ("event_id", "camera_id", "track_id", "type", "severity")}
            payload["text"] = f'{event["camera_id"]}: {self.messages[kind]}'
            await asyncio.gather(*(self._send(device, payload) for device in self.devices))

    async def _send(self, device, payload):
        try:
            await device.send(payload)
        except Exception:
            logger.exception("Alert %s failed for device %s", payload["event_id"], device.name)
