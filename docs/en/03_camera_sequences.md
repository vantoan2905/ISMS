# Camera Algorithm Sequences

Camera tasks may run concurrently through one shared inference executor. Each camera has its own tracker and rule state.

## Camera 1 — helmet and ROI

```mermaid
sequenceDiagram
    participant C as Camera 1 / appsink
    participant X as Executor + RTMO-m
    participant T as Camera 1 tracker
    participant H as Head crop + EfficientNet
    participant R as ROI entry + dwell
    participant E as Event engine
    participant O as SQLite / relay / alert
    C->>X: BGR frame + PTS
    X->>T: Person boxes + keypoints
    T->>H: Track + keypoints
    H-->>E: Confirmed NO_HELMET or HELMET_RESOLVED
    Note over H,E: UNKNOWN does not create NO_HELMET
    T->>R: Track, foot position and PTS
    R-->>E: ROI_OCCUPIED / ENTER / EXIT / DWELL
    E->>O: Store incident and snapshot, update relay
    E-->>O: Announce supported events
```

## Camera 2 — ROI

```mermaid
sequenceDiagram
    participant C as Camera 2 / appsink
    participant X as Executor + RTMO-m
    participant T as Camera 2 tracker
    participant R as ROI entry
    participant E as Event engine
    participant O as SQLite / relay / alert
    C->>X: BGR frame + PTS
    X->>T: Person boxes + keypoints
    T->>R: Track and foot position
    R-->>E: ROI_OCCUPIED / ENTER / EXIT
    E->>O: Store incident, update relay and alert
```

## Camera 3 — helmet

```mermaid
sequenceDiagram
    participant C as Camera 3 / appsink
    participant X as Executor + RTMO-m
    participant T as Camera 3 tracker
    participant H as Head crop + EfficientNet
    participant E as Event engine
    participant O as SQLite / alert
    C->>X: BGR frame + PTS
    X->>T: Person boxes + keypoints
    T->>H: Track and keypoints
    H-->>E: NO_HELMET or HELMET_RESOLVED
    Note over H,E: Missing crop or low confidence becomes UNKNOWN
    E->>O: Store incident and snapshot, announce event
```

Camera 2 has no dwell rule enabled. Camera 3 has no ROI rule. Rules are set in `configs/cameras.yaml`.

## Complete sequence from RTSP to devices

This example uses camera 1. Camera 2 omits helmet classification and camera 3 omits ROI. Boxes indicate deployment areas. RTSP and HTTP gateways require configuration and verification in the target environment.

```mermaid
sequenceDiagram
    box rgb(226, 239, 250) Camera and network
        participant CAM as IP camera / MP4
        participant NET as Router / network
    end
    box rgb(232, 244, 232) AI server - media
        participant GST as GStreamer worker
        participant Q as Camera queue
    end
    box rgb(255, 244, 220) AI server - inference and rules
        participant EX as Shared executor
        participant POSE as RTMO-m CPU
        participant TR as Camera 1 tracker
        participant HC as Head crop
        participant CL as EfficientNet CPU or CUDA
        participant ROI as ROI rule
    end
    box rgb(245, 230, 250) AI server - events and output
        participant EV as Event engine
        participant FS as JPEG snapshot
        participant DB as SQLite
        participant REL as Mock relay
        participant AL as Alert service
        participant API as API / WebSocket
        participant REC as Overlay + VideoWriter
    end
    box rgb(235, 235, 235) Outside the server
        participant DEV as Display / speaker gateway
        participant UI as Dashboard / API client
    end
    CAM->>NET: Encoded RTSP stream
    NET->>GST: RTSP URI and stream
    Note over CAM,GST: Demo MP4 is read locally without the router
    GST->>GST: uridecodebin decode → BGR / scale → appsink
    GST->>Q: Copy frame, frame ID, PTS and receive times
    Note over Q: Drop older frames when queue is full
    Q->>EX: Frame
    EX->>POSE: Run inference
    POSE-->>TR: Person boxes and keypoints
    TR->>HC: Track ID and facial keypoints
    HC->>HC: Estimate, pad and crop head region
    alt Crop missing, too small or blurred
        HC-->>EX: UNKNOWN
    else Valid crop
        HC->>CL: Head crop
        CL-->>EX: Helmet / no-helmet probabilities
        alt Low confidence
            EX-->>EX: UNKNOWN
        else Confirmed no helmet
            EX->>EV: NO_HELMET observation
        end
    end
    TR->>ROI: Foot position + track ID + PTS
    ROI-->>EV: ROI_OCCUPIED / ENTER / EXIT / DWELL on transition
    alt New nonduplicate incident
        EV->>FS: Save original frame to JPEG
        FS-->>EV: Snapshot path
        EV->>DB: Insert incident and metadata
        EV->>AL: Publish incident
        AL->>DEV: Simulated log or HTTP POST
        EV->>API: Publish incident
        API-->>UI: WebSocket update
    else Incident resolved
        EV->>DB: Update status to RESOLVED
        EV->>API: Publish update
    end
    EV->>REL: Recompute state from active intrusions
    TR->>REC: Frame + pose + boxes + track
    REL->>REC: Relay status
    REC->>REC: Draw overlay → mp4v encoding → MP4
    UI->>API: GET /events or GET /health
    API->>DB: Read incident history for GET /events
    DB-->>API: Incidents
    API-->>UI: JSON response
```
