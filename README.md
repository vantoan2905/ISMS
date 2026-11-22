# Industrial Safety Monitoring System

Camera based safety event starter project. It detects people, tracks them between frames, reports restricted zone entry, and can report missing helmets when a compatible custom model is configured. Events are stored in SQLite, shown in a live dashboard, and optionally sent to a relay HTTP endpoint.

## Run with uv

```bash
uv sync --extra dev
uv run uvicorn src.api.main:app --reload
```

Open http://localhost:8000 for the dashboard, `/docs` for API documentation, and `/health` for status. The sample camera is disabled by default. To process a camera, edit `configs/cameras.yaml`, set its source and `enabled: true`, then install the model extra:

```bash
uv sync --extra dev --extra yolo
uv run uvicorn src.api.main:app
```

The sample `yolo11n.pt` model detects people but has no helmet class, so it only supports zone intrusion. For helmet alerts, configure weights trained with both `person` and `helmet` classes. Polygon points are pixel coordinates for the source video. Person foot position determines zone entry. Camera source can be a webcam integer, video file, or RTSP URL. An optional `relay_url` receives each event as JSON via HTTP POST; secure and authenticate this endpoint before production use.

Run tests with `uv run pytest`. Docker deployment: `docker compose up --build`. For Docker, set `storage.path: /app/data/safety_events.db` to persist events in the named volume. Webcam devices require an explicit device mapping in Compose; RTSP sources are usually simpler.

- For install GStreamer
```bash
apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio

```
## Central server and relay simulator

`docker compose up --build` starts the monitor on port 8000 and an HTTP relay simulator on port 8001. The simulator logs received alerts and exposes the latest 100 at `http://localhost:8001/events`. Events are stored in the `monitor-data` volume. A camera may override the simulator by setting its own `relay_url`.

Add a separate entry to `configs/cameras.yaml` for each camera and set `enabled: true`. Use an RTSP URL or a path under `/app/videos` in Docker. Zone coordinates are in source frame pixels. Model files can be placed under `models/` and referenced as `/app/models/...`. Helmet detection requires weights with both person and helmet classes.

The central server uses the Python YOLO detector and centroid tracker for live event generation. The C++ ONNX pipeline in `src/inference/core` processes video separately and does not generate safety events.
