# Architecture

```mermaid
flowchart TB
    subgraph sources["Nguồn video / camera"]
        C1["CAM-01: video 1"]
        C2["CAM-02: video 2"]
        C3["CAM-03: video 3"]
    end

    CFG["Cấu hình camera, ROI và model"]
    subgraph central["Ứng dụng ISMS"]
        CM["Camera manager"]
        GST["GStreamer appsink<br/>decode, chuyển BGR và lấy PTS"]
        Q["Hàng đợi giới hạn riêng cho từng camera<br/>bỏ frame cũ khi quá tải"]
        EX["Inference executor dùng chung"]
        POSE["RTMO-m<br/>bbox người + keypoints"]
        TRACK["Tracker riêng từng camera"]
        CROP["Crop vùng đầu từ pose"]
        HELMET["EfficientNet-B0<br/>HELMET / NO_HELMET / UNKNOWN"]
        ROI["ROI entry / exit / dwell"]
        RULES["Observations"]
        EVENT["Event engine<br/>lifecycle + chống trùng"]
        OVERLAY["Vẽ bbox, keypoints, ROI và trạng thái"]
    end

    DB[("SQLite incidents")]
    SNAP["Snapshot"]
    RELAY["Mock relay"]
    API["FastAPI / WebSocket"]
    DASH["Dashboard"]
    VIDEO["Video demo MP4"]

    CFG --> CM
    CM --> GST
    C1 --> GST
    C2 --> GST
    C3 --> GST
    GST --> Q --> EX --> POSE --> TRACK
    TRACK --> CROP --> HELMET --> RULES
    TRACK --> ROI --> RULES
    RULES --> EVENT
    EVENT --> DB
    EVENT --> SNAP
    EVENT --> RELAY
    EVENT --> API --> DASH
    TRACK --> OVERLAY
    HELMET --> OVERLAY
    ROI --> OVERLAY
    RELAY --> OVERLAY --> VIDEO
```

The reader process owns GStreamer objects and copies BGR frames to the application. A system Python process is used to accommodate PyGObject builds that do not match the uv environment's Python ABI. Inference is serialized through one shared executor. Tracking and rule state are independent per camera. Queues drop the oldest frame when full and report application queue drops. The pipeline propagates GStreamer PTS; missing PTS falls back to monotonic receive time, and a backwards PTS resets per-camera rule state.

`src/isms/config` validates configuration, `src/isms/app` manages camera tasks, `src/isms/media` handles ingestion, and `src/isms/inference` owns shared model execution. `src/isms/algorithms` composes camera-specific rules, while `src/isms/events` manages observations and incident lifecycles. The CLI and API share the same camera manager.

RTMO-m provides person boxes and COCO keypoints. The head crop is classified by EfficientNet-B0, subject to crop size, sharpness, confidence and consecutive-frame checks. ROI uses ankle keypoints when available, otherwise the bottom of the person box. A first observation inside is distinguished from an outside-to-inside crossing. Dwell uses PTS instead of frame count. The event engine deduplicates active violations, stores lifecycle changes and keeps the shared relay on while any intrusion remains active. It retains active events if a camera fails. The tracker is a baseline and may lose identity under occlusion.

Annotated MP4 files are diagnostic visualizations written at a configured fixed frame rate. They do not preserve source timing and should not be treated as evidence. No custom C code is required: GStreamer itself is native code accessed through its Python bindings.
