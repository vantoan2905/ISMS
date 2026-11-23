# Sequence thuật toán cho từng camera

Mỗi sequence là một nhánh logic, các camera có thể chạy đồng thời và chia sẻ executor. Tracker và rule có state riêng theo camera.

## Camera 1 — mũ bảo hộ và ROI

```mermaid
sequenceDiagram
    participant C as Camera 1 / appsink
    participant X as Executor + RTMO-m
    participant T as Tracker camera 1
    participant H as Head crop + EfficientNet
    participant R as ROI entry + dwell
    participant E as Event engine
    participant O as SQLite / relay / alert
    C->>X: Frame BGR + PTS
    X->>T: Bbox người + keypoints
    T->>H: Track và keypoints
    H-->>E: NO_HELMET sau xác nhận, hoặc HELMET_RESOLVED
    Note over H,E: UNKNOWN không tạo NO_HELMET
    T->>R: Track, vị trí chân, PTS
    R-->>E: ROI_OCCUPIED / ENTER / EXIT / DWELL
    E->>O: Ghi sự kiện, snapshot và cập nhật relay
    E-->>O: Phát thông báo cho sự kiện được hỗ trợ
```

## Camera 2 — ROI

```mermaid
sequenceDiagram
    participant C as Camera 2 / appsink
    participant X as Executor + RTMO-m
    participant T as Tracker camera 2
    participant R as ROI entry
    participant E as Event engine
    participant O as SQLite / relay / alert
    C->>X: Frame BGR + PTS
    X->>T: Bbox người + keypoints
    T->>R: Track và vị trí chân
    R-->>E: ROI_OCCUPIED / ENTER / EXIT
    E->>O: Lưu sự kiện, cập nhật relay và cảnh báo
```

## Camera 3 — mũ bảo hộ

```mermaid
sequenceDiagram
    participant C as Camera 3 / appsink
    participant X as Executor + RTMO-m
    participant T as Tracker camera 3
    participant H as Head crop + EfficientNet
    participant E as Event engine
    participant O as SQLite / alert
    C->>X: Frame BGR + PTS
    X->>T: Bbox người + keypoints
    T->>H: Track và keypoints
    H-->>E: NO_HELMET hoặc HELMET_RESOLVED
    Note over H,E: Crop thiếu hoặc confidence thấp → UNKNOWN
    E->>O: Lưu sự kiện, snapshot và thông báo
```

ROI ở camera 2 không bật dwell. Camera 3 không cấu hình ROI. Xem `configs/cameras.yaml` để thay đổi rule.

## Sequence đầy đủ từ RTSP đến event và thiết bị

Ví dụ dưới đây dùng camera 1. Camera 2 bỏ nhánh phân loại mũ; camera 3 bỏ nhánh ROI. Các `box` thể hiện vị trí triển khai. RTSP và thiết bị HTTP cần được cấu hình và kiểm chứng trong môi trường thực.

```mermaid
sequenceDiagram
    box rgb(226, 239, 250) Camera và mạng
        participant CAM as Camera IP / MP4
        participant NET as Router / network
    end
    box rgb(232, 244, 232) AI server - media
        participant GST as GStreamer worker
        participant Q as Queue camera
    end
    box rgb(255, 244, 220) AI server - inference và rules
        participant EX as Shared executor
        participant POSE as RTMO-m CPU
        participant TR as Tracker camera 1
        participant HC as Head crop
        participant CL as EfficientNet CPU hoặc CUDA
        participant ROI as ROI rule
    end
    box rgb(245, 230, 250) AI server - event và output
        participant EV as Event engine
        participant FS as Snapshot JPEG
        participant DB as SQLite
        participant REL as Mock relay
        participant AL as Alert service
        participant API as API / WebSocket
        participant REC as Overlay + VideoWriter
    end
    box rgb(235, 235, 235) Bên ngoài server
        participant DEV as Display / speaker gateway
        participant UI as Dashboard / API client
    end

    CAM->>NET: Stream RTSP đã encode
    NET->>GST: URI RTSP và stream
    Note over CAM,GST: Demo MP4 đọc trực tiếp từ file, không đi qua router
    GST->>GST: uridecodebin decode → BGR / scale → appsink
    GST->>Q: Copy frame, frame ID, PTS, thời gian nhận
    Note over Q: Hàng đợi đầy thì bỏ frame cũ
    Q->>EX: Frame mới
    EX->>POSE: Inference frame
    POSE-->>TR: Bbox người và keypoints
    TR->>HC: Track ID và điểm mặt
    HC->>HC: Ước lượng vùng đầu, mở rộng và crop
    alt Crop thiếu, quá nhỏ hoặc quá mờ
        HC-->>EX: UNKNOWN
    else Crop hợp lệ
        HC->>CL: Ảnh crop đầu
        CL-->>EX: Xác suất helmet / no helmet
        alt Confidence thấp
            EX-->>EX: UNKNOWN
        else No helmet được xác nhận
            EX->>EV: Observation NO_HELMET
        end
    end
    TR->>ROI: Vị trí chân + track ID + PTS
    ROI-->>EV: ROI_OCCUPIED / ENTER / EXIT / DWELL khi có thay đổi
    alt Incident mới và không trùng
        EV->>FS: Lưu frame gốc thành JPEG
        FS-->>EV: Đường dẫn snapshot
        EV->>DB: Insert incident + metadata
        EV->>AL: Dispatcher phát event
        AL->>DEV: Log mô phỏng hoặc HTTP POST
        EV->>API: Dispatcher phát event
        API-->>UI: WebSocket update
    else Incident kết thúc
        EV->>DB: Update trạng thái RESOLVED
        EV->>API: Dispatcher phát update
    end
    EV->>REL: Tính trạng thái từ intrusion active
    TR->>REC: Frame + pose + bbox + track
    REL->>REC: Trạng thái relay
    REC->>REC: Vẽ overlay → encode mp4v → lưu MP4
    UI->>API: GET /events hoặc GET /health
    API->>DB: Đọc lịch sử cho GET /events
    DB-->>API: Incident
    API-->>UI: JSON response
```
