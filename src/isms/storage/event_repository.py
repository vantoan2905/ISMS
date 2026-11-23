"""SQLite persistence operations for incident lifecycles."""
import json


class IncidentRepository:
    def insert_incident(self, event):
        with self.connect() as db:
            db.execute("INSERT INTO incidents VALUES (?, ?, ?, ?, ?)",
                       (event['event_id'], event['camera_id'], event['started_at_utc'],
                        event['status'], json.dumps(event)))

    def update_incident(self, event):
        with self.connect() as db:
            db.execute("UPDATE incidents SET status=?, payload=? WHERE event_id=?",
                       (event['status'], json.dumps(event), event['event_id']))

    def list_active_incidents(self):
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM incidents WHERE status = 'ACTIVE'").fetchall()
        return [json.loads(row[0]) for row in rows]

    def list_incidents(self, limit=100):
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM incidents ORDER BY started_at_utc DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]

