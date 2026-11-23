import os
from contextlib import asynccontextmanager
from pathlib import Path
import yaml
from fastapi import FastAPI, Query, WebSocket
from fastapi.responses import HTMLResponse
from src.services.camera_manager import CameraManager
from src.services.event_service import EventService
from ..storage.database import Database
from .websocket import events_socket
from ..events.engine import EventEngine
from ..inference.executor import InferenceExecutor
from ..app.camera_runtime import run_camera
from ..config.loader import load_config
from ..app.camera_manager import CameraManager as MVPCameraManager

ROOT = Path(__file__).resolve().parents[3]


def create_app(config_path=None):
    path = Path(config_path or os.getenv("SAFETY_CONFIG", ROOT / "configs/cameras.yaml"))
    config = yaml.safe_load(path.read_text()) or {}
    db_path = Path(os.getenv("SAFETY_DATABASE") or config.get("storage", {}).get("path", "safety_events.db"))
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    database = Database(str(db_path))
    mvp = 'models' in config or 'model_config' in config
    if mvp:
        config = load_config(path)
        service = EventEngine(database, config.get('storage', {}).get('snapshot_dir', 'snapshots'),
                              config.get('runtime', {}).get('enable_mock_relay', True))
        manager = MVPCameraManager(config, service, executor_factory=InferenceExecutor, camera_runner=run_camera)
    else:
        service = EventService(database)
        manager = CameraManager(config, service)

    @asynccontextmanager
    async def lifespan(app):
        await manager.start()
        try:
            yield
        finally:
            await manager.close()

    app = FastAPI(title="Industrial Safety Monitor", lifespan=lifespan)
    app.state.event_service = service

    @app.get("/")
    async def dashboard():
        return HTMLResponse((ROOT / "dashboard/index.html").read_text())

    @app.get("/health")
    async def health():
        return {"status": "ok", "active_cameras": sum(not task.done() for task in manager.tasks),
                "relay_on": getattr(service, 'relay_on', None)}

    @app.get("/events")
    async def events(limit: int = Query(100, ge=1, le=1000)):
        return database.list_incidents(limit) if mvp else database.list_events(limit)

    @app.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket):
        await events_socket(websocket, service)

    return app


app = create_app()
