import json
import sqlite3
from pathlib import Path

from .event_repository import IncidentRepository


class Database(IncidentRepository):
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL,
                type TEXT NOT NULL, track_id INTEGER NOT NULL, zone_id TEXT,
                timestamp TEXT NOT NULL, payload TEXT NOT NULL)""")

        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS incidents (
                event_id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, started_at_utc TEXT NOT NULL,
                status TEXT NOT NULL, payload TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_incidents_camera_time ON incidents(camera_id, started_at_utc)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def insert(self, event: dict) -> dict:
        with self.connect() as db:
            cursor = db.execute("INSERT INTO events (camera_id,type,track_id,zone_id,timestamp,payload) VALUES (?,?,?,?,?,?)",
                                (event["camera_id"], event["type"], event["track_id"],
                                 event.get("zone_id"), event["timestamp"], json.dumps(event)))
            return {"id": cursor.lastrowid, **event}

    def list_events(self, limit=100):
        with self.connect() as db:
            rows = db.execute("SELECT id,payload FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": row_id, **json.loads(payload)} for row_id, payload in rows]
