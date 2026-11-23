FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 libglib2.0-0 libgl1 gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY dashboard ./dashboard
COPY configs ./configs
RUN uv sync --locked --no-dev --extra yolo --group pose --group training
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
CMD ["uvicorn", "src.isms.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
