# Setup and deployment

Run the commands below from the repository root. The current application is a three-camera MVP that uses GStreamer, RTMO-m pose estimation and an EfficientNet-B0 helmet classifier.

## 1. Prerequisites and artifacts

- Ubuntu/Debian with Python support, `uv`, and GStreamer with PyGObject bindings.
- For Docker, Docker Engine or Docker Desktop with Compose and, under WSL, Docker Desktop WSL integration enabled.
- The RTMO-m ONNX model and EfficientNet-B0 checkpoint at the locations in `configs/models.yaml`.
- Three MP4 files at the camera `source` paths in `configs/cameras.yaml`, or accessible RTSP URLs.

The `models/`, source videos, datasets, generated videos and SQLite databases are excluded from Git. Place the weight files and video files locally before starting. The project does not download them automatically.

```text
models/rtmo_m/rtmo_m_640x640_coco.onnx
models/helmet_classifier_efficientnet/best.pt
4281404-uhd_3840_2160_25fps.mp4
4284182-uhd_3840_2160_25fps.mp4
14656573_2160_3840_30fps.mp4
```

These are the default paths; update the YAML if your files are elsewhere. Model provenance is described in [`src/inference/README.md`](../src/inference/README.md). The RTMO export and dataset/model/video permissions must be checked before redistribution.

## 2. Set up with uv

Install `uv` using the installation method suitable for your environment. Install the system GStreamer bindings and plugins, then install the Python dependencies:

```bash
sudo apt update
sudo apt install -y python3 python3-gi gir1.2-gstreamer-1.0 \
  gir1.2-gst-plugins-base-1.0 gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-libav
uv sync --group pose --group training --extra dev
```

The pose and training groups include the inference libraries required by the configured checkpoints. Install the GStreamer packages in the system Python environment: the reader defaults to `/usr/bin/python3`, independently of the Python selected by `uv`.

Check the binding and plugins:

```bash
/usr/bin/python3 -c 'import gi; gi.require_version("Gst", "1.0"); from gi.repository import Gst; Gst.init(None); print(Gst.version_string())'
gst-inspect-1.0 uridecodebin
gst-inspect-1.0 appsink
```

If `gst-inspect-1.0` is missing, install `gstreamer1.0-tools`. Set `ISMS_GSTREAMER_PYTHON` to another executable with working PyGObject and GStreamer bindings if `/usr/bin/python3` is unsuitable.

## 3. Configure cameras and models

- `configs/models.yaml` specifies the pose ONNX file and helmet checkpoint. Relative paths are resolved from the working directory, normally the project root.
- `configs/cameras.yaml` specifies each camera `id`, `source`, processing `width`/`height`, ROI and enabled algorithms. Sources can be local paths or RTSP URLs.
- ROI points are normalized from 0 to 1 relative to the resized frame. Width must be positive and divisible by four. Calibrate ROI polygons for each camera.
- The configured `alert_devices` log simulated display/speaker announcements until an HTTP `endpoint` is supplied. English is the default; `alert_language: vi` selects Vietnamese. The mock relay is enabled by `runtime.enable_mock_relay`.
- SQLite and snapshot paths are set under `storage`. Keep credentials out of committed configuration.

The current camera 1 helmet and ROI rules run independently. Camera 2 has ROI only, and camera 3 has helmet checking only. The current RTMO adapter runs on CPU; the classifier selects CUDA if PyTorch detects it, otherwise CPU. The default container has no GPU device configured.

## 4. Run the CLI demo

```bash
./scripts/run_demo.sh
```

Or choose another config:

```bash
uv run --group pose --group training python -m src.isms.main --config configs/cameras.yaml
```

The CLI starts all enabled cameras. It logs the output directory `videos/demo/<run-id>/`, records one annotated MP4 per camera, stores SQLite history at `storage.path` and saves snapshots under `storage.snapshot_dir`. Press `Ctrl+C` to stop. Set `runtime.record_demo: false` to skip video recording.

An MP4 end-of-stream currently creates a `CAMERA_OFFLINE` event. Frame queues drop frames under load, and demo video is written at a fixed default 8 FPS rather than the source timestamps. A camera error retains active intrusion relay state; check retained incidents before reusing the database for a new demo.

## 5. Run the API

```bash
uv run --group pose --group training uvicorn src.isms.api.main:app --host 0.0.0.0 --port 8000
```

- Dashboard: `http://localhost:8000/`
- API docs: `http://localhost:8000/docs`
- Health and relay state: `http://localhost:8000/health`
- Incident history: `http://localhost:8000/events`
- Live incident updates: `ws://localhost:8000/ws/events`

Set `SAFETY_CONFIG` to select another YAML file or `SAFETY_DATABASE` to override the API database path. The CLI and API each create camera tasks; run one at a time for a single monitoring instance. The API uses the configured video output directory only if `runtime.video_dir` is explicitly supplied; the CLI generates its own run directory.

## 6. Run using Docker Compose

The supplied `Dockerfile` builds the API image. Compose mounts `./configs` at `/app/configs`, `./videos` at `/app/videos`, `./models` at `/app/models`, and a named volume at `/app/data`. The default camera YAML references source videos in the host project root, which are **not** mounted there in the container. Create a separate Docker config and place the files in the mounted video directory:

```bash
mkdir -p videos
cp 4281404-uhd_3840_2160_25fps.mp4 videos/
cp 4284182-uhd_3840_2160_25fps.mp4 videos/
cp 14656573_2160_3840_30fps.mp4 videos/
cp configs/cameras.yaml configs/cameras.docker.yaml
```

Edit `configs/cameras.docker.yaml` and set its three `source` values to `/app/videos/<filename>`. For persistent snapshots, also set `storage.snapshot_dir: /app/data/snapshots`. Keep `model_config: models.yaml` so the file in `/app/configs` is loaded; the default relative model paths resolve under `/app/models`.

Build and run the monitor with this config:

```bash
docker compose build monitor
docker compose run --rm --service-ports \
  -e SAFETY_CONFIG=/app/configs/cameras.docker.yaml monitor
```

Open `http://localhost:8000/health` and `/docs`. Stop with `Ctrl+C`. The Compose environment sets `SAFETY_DATABASE=/app/data/safety_events.db` for the API, so incidents persist in the `monitor-data` named volume. Snapshots persist there only if `storage.snapshot_dir` is set as described. `docker compose run` starts the declared `relay` dependency; the MVP uses the in-process mock relay and does not route its events to that HTTP relay service through `SAFETY_RELAY_URL`.

The image uses the Dockerfile's default CPU container configuration. GPU access needs separate container/runtime setup. The Compose workflow and GPU configuration have not been validated.

## 7. Tests and troubleshooting

```bash
uv run --group pose --group training --extra dev pytest -q
```

- **Cannot open a source:** check the configured path from the process working directory or the container mount; verify the RTSP URL and network access.
- **Missing `gi` or GStreamer element:** install the bindings/plugins for the reader's system Python and check with `gst-inspect-1.0`.
- **Missing model/checkpoint:** place files at the paths in `configs/models.yaml`; they are not included in the repository.
- **Frames dropped:** inspect `queue_dropped` in logs, reduce the source workload or optimize inference. This counter excludes drops inside appsink.
- **No video output:** use the CLI with recording enabled and inspect its logged output directory. API recording requires `runtime.video_dir`.
- **Docker unavailable in WSL:** enable WSL integration in Docker Desktop, then verify `docker compose version` before building.
