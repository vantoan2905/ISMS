import asyncio
import json
from contextlib import nullcontext

from src.isms.alert.service import AlertService
from src.isms.events.dispatcher import EventDispatcher


def test_device_commands_and_failure_isolation(monkeypatch, caplog):
    sent = []

    def post(request, timeout):
        command = json.loads(request.data)
        if command["action"] == "speak":
            raise OSError("Speaker disconnected")
        sent.append(command)
        return nullcontext()

    monkeypatch.setattr("src.isms.alert.devices.urlopen", post)

    async def run():
        engine = EventDispatcher()
        alerts = AlertService.from_config({"alert_devices": [
            {"type": kind, "endpoint": "http://gateway.test/commands"}
            for kind in ("speaker", "display")
        ]})
        alerts.start(engine)
        event = dict(event_id="a", camera_id="cam1", track_id=2, type="ROI_ENTER",
                     severity="CRITICAL", status="ACTIVE")
        engine.publish(event)
        engine.publish(dict(event, status="RESOLVED"))
        engine.publish(dict(event, event_id="b", type="ROI_DWELL", status="RESOLVED"))
        engine.publish(dict(event, event_id="c", type="CAMERA_OFFLINE"))
        await alerts.close(engine)
        assert not engine.subscribers

    asyncio.run(run())
    assert [cmd["event_id"] for cmd in sent] == ["a", "b"]
    assert all(cmd["action"] == "display_text" and "cam1" in cmd["text"] for cmd in sent)
    assert sent[0]["text"] == "cam1: A person has entered a hazardous area."
    assert "failed for device speaker" in caplog.text


def test_vietnamese_alerts_can_be_selected(monkeypatch):
    sent = []

    def post(request, timeout):
        sent.append(json.loads(request.data))
        return nullcontext()

    monkeypatch.setattr('src.isms.alert.devices.urlopen', post)

    async def run():
        engine = EventDispatcher()
        alerts = AlertService.from_config({'alert_language': 'vi', 'alert_devices': [
            {'type': 'speaker', 'endpoint': 'http://gateway.test/commands'},
        ]})
        alerts.start(engine)
        engine.publish(dict(event_id='a', camera_id='cam1', track_id=1,
                            type='ROI_ENTER', severity='CRITICAL', status='ACTIVE'))
        await alerts.close(engine)

    asyncio.run(run())
    assert sent[0]['text'] == 'cam1: Có người đi vào khu vực nguy hiểm.'
