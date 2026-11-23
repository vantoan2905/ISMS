"""Consume new safety incidents and forward announcements to device adapters."""
import asyncio
import logging
from urllib.parse import urlsplit

from .devices import Display, Speaker

logger = logging.getLogger(__name__)

MESSAGES = {
    "ROI_ENTER": "Có người đi vào khu vực nguy hiểm.",
    "ROI_OCCUPIED": "Có người trong khu vực nguy hiểm.",
    "ROI_DWELL": "Có người ở trong khu vực nguy hiểm quá thời gian quy định.",
    "NO_HELMET": "Phát hiện người có khả năng không đội mũ bảo hộ.",
}


class AlertService:
    def __init__(self, devices):
        self.devices = devices
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
        return cls(devices)

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
            if kind not in MESSAGES or (event["status"] != "ACTIVE" and kind != "ROI_DWELL"):
                continue
            payload = {key: event[key] for key in ("event_id", "camera_id", "track_id", "type", "severity")}
            payload["text"] = f'{event["camera_id"]}: {MESSAGES[kind]}'
            await asyncio.gather(*(self._send(device, payload) for device in self.devices))

    async def _send(self, device, payload):
        try:
            await device.send(payload)
        except Exception:
            logger.exception("Alert %s failed for device %s", payload["event_id"], device.name)
