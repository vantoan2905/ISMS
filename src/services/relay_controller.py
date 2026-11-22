import json
import logging
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def send_relay(url: str | None, event: dict):
    if not url:
        return
    try:
        request = Request(url, data=json.dumps(event).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=3) as response:
            response.read()
    except Exception:
        logger.exception("Relay delivery failed: %s", url)
