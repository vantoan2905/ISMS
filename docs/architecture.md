# Architecture

Each enabled camera starts an asynchronous capture task. OpenCV reads frames; a shared YOLO model detects objects; a per-camera centroid tracker gives people temporary IDs. Zone and helmet rules produce findings, which the event engine deduplicates by camera, type, track, and zone for a configurable cooldown. EventService saves each finding in SQLite, broadcasts it to WebSocket subscribers, and optionally posts it to a relay URL. `/events` lists stored events and `/ws/events` streams new ones.

Tracking is intentionally lightweight: IDs may change after occlusion or large motion. A production deployment should use a stronger tracker and camera reconnect logic. One process should own each camera; multiple API workers would duplicate capture and events.
