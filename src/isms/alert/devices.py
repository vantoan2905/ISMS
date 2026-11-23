"""Adapters for signage displays and announcement speakers."""
import asyncio
import json
import logging
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class AnnouncementDevice:
    def __init__(self, name, endpoint=None):
        self.name = name
        self.endpoint = endpoint

    async def send(self, payload):
        command = dict(payload, device=self.name, action=self.action)
        if self.endpoint:
            await asyncio.to_thread(self._post, command)
        else:
            logger.warning("Simulated %s: %s", self.name, command["text"])

    def _post(self, command):
        request = Request(self.endpoint, data=json.dumps(command).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=3):
            pass


class Display(AnnouncementDevice):
    action = "display_text"


class Speaker(AnnouncementDevice):
    action = "speak"
