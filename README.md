# Industrial Safety Monitoring System

Tài liệu chính: [Cách tiếp cận bài toán và thiết kế hệ thống](docs/00_problem_approach.md).

This MVP runs three camera sources through GStreamer, shared RTMO-m pose inference, camera-local tracking and configurable helmet and ROI rules. EfficientNet-B0 classifies pose-derived head crops. Events are saved in SQLite with snapshots; ROI occupancy drives a shared mock relay.

## Setup and run

The configured demo uses the three MP4 files named in `configs/cameras.yaml` and the RTMO-m and EfficientNet-B0 weights configured in `configs/models.yaml`. These files are not included in Git. Update the YAML if your paths differ. Run commands from the project root.

On Debian/Ubuntu, install the GStreamer Python bindings and plugins for the system Python:

```bash
sudo apt install python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-libav
uv sync --group pose --group training --extra dev
./scripts/run_demo.sh
```

The command starts all enabled cameras and writes annotated videos under the logged `videos/demo/<run-id>/` directory. Each camera has its own output. Press Ctrl-C to stop. `uv run --group pose --group training python -m src.isms.main --config configs/cameras.yaml` is equivalent. To skip recording, set `runtime.record_demo: false`.

The GStreamer reader is a Python process using `gi.repository.Gst` directly and an appsink; it sends copied BGR frames and their actual PTS to the main process. It defaults to `/usr/bin/python3` because PyGObject is often installed for the system Python, while uv may use a different Python ABI. Set `ISMS_GSTREAMER_PYTHON` to a Python executable with the GStreamer GI bindings installed if needed. GStreamer and codec plugins must be available to that interpreter. The processing width must be divisible by four. Local MP4 paths and RTSP URLs are supported.

## Configuration and behavior

`configs/cameras.yaml` assigns rules to each camera. ROIs use normalized coordinates `(x, y)` between 0 and 1 after resizing. Camera 1 runs helmet, entry and dwell rules; camera 2 runs entry; camera 3 runs helmet. A track first observed inside emits `ROI_OCCUPIED`, while crossing from outside emits `ROI_ENTER`; both activate the relay. It stays on until all active intrusion tracks have exited or timed out. A camera source failure retains active relay state and creates a `CAMERA_OFFLINE` event. PTS drives dwell time; wall-clock time records events. When PTS is unavailable, the reader uses monotonic receive time as a fallback.

Head crops that are missing, too small, heavily blurred or below the configured confidence threshold are `UNKNOWN`, never an automatic no-helmet violation. Helmet alerts require consecutive confident predictions. Thresholds and ROI polygons need calibration on the actual cameras. Tracking is a centroid baseline, so occlusion and identity changes can affect events. Videos are diagnostic recordings at `record_fps` (default 8), not timestamp-preserving evidence. Queue drop counts cover the application queue, not frames dropped internally by appsink.

SQLite defaults to `safety_events.db`; snapshots default to `snapshots/`. Both paths are configurable. The relay is a software simulator, not a safety-rated hardware controller. Active incidents are restored after restart; check retained incidents before reusing camera/track IDs.

## Source layout

The current flow is organized by responsibility:

```text
src/isms/
  main.py                       CLI entry point
  config/                       YAML loading and validation
  core/                         Frame context and bounded queue
  app/                          Camera manager and per-camera runtime
  media/                        GStreamer appsink reader and video output
  inference/                    Shared executor, pose, head crop and helmet model
  tracking/                     Per-camera person tracker
  algorithms/                   Pose, helmet, ROI, factory and rule pipeline
  events/                       Observations and event lifecycle
  outputs/                      Mock relay and snapshots
  storage/                      SQLite persistence and incident repository
  api/                          FastAPI and WebSocket endpoints
```

The earlier inference and event modules remain available for existing integrations. The original MVP import paths also remain as compatibility shims.

## API and tests

```bash
uv run --group pose --group training uvicorn src.isms.api.main:app --host 0.0.0.0 --port 8000
uv run --group pose --group training --extra dev pytest -q
```

`GET /health` reports camera tasks and mock relay state; `GET /events` lists incident history; `/ws/events` streams incident updates. Existing dashboard support is basic. API and CLI each start their own camera processing, so normally run only one. The Docker setup is optional and requires the source video paths to be mounted into the container; it has not been validated as a demo deployment.

## Announcement devices

Configure `alert_devices` in `configs/cameras.yaml` for the CLI or API. Each
`display` or `speaker` logs a simulated announcement by default. Add an
`endpoint` to send an HTTP JSON POST to a compatible device gateway. For example:

```yaml
alert_devices:
  - {type: display, name: entrance_screen, endpoint: "http://signage.local/commands"}
  - {type: speaker, name: workshop_speaker}
```

Commands include `action` (`display_text` or `speak`), `device`, Vietnamese
`text`, `event_id`, `camera_id`, `track_id`, `type` and `severity`. The service
announces a new intrusion, occupancy, dwell or missing-helmet incident once
at creation. Set `alert_devices: []` to disable announcements. Delivery uses
the bounded event subscription. Under overload it can drop older updates.
Device failures are logged without retry. The incident history remains in SQLite.

