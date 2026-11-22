# Deployment

Configure cameras in `configs/cameras.yaml`, then start one API worker with `uv run uvicorn src.api.main:app --host 0.0.0.0`. Docker Compose builds the same service. Use a persistent absolute SQLite path; mount model weights and video files as needed. Limit access to the dashboard and API with a reverse proxy and TLS. Keep RTSP credentials outside version control. For production, add camera reconnects, retention policy, monitoring, and authenticated relay delivery. The current relay expects an HTTP endpoint that accepts JSON POST requests.
