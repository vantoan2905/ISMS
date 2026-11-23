# Hướng dẫn cài đặt và triển khai

Chạy các lệnh dưới đây từ thư mục gốc của repository. Ứng dụng hiện là MVP giám sát ba camera, dùng GStreamer, RTMO-m để phát hiện pose và EfficientNet-B0 để phân loại mũ.

## 1. Chuẩn bị

- Ubuntu/Debian, `uv`, GStreamer và binding PyGObject cho Python hệ thống.
- Nếu dùng Docker, cần Docker Engine hoặc Docker Desktop có Compose. Trên WSL cần bật WSL integration của Docker Desktop.
- File RTMO-m ONNX và checkpoint EfficientNet-B0 theo `configs/models.yaml`.
- Ba file MP4 theo `source` trong `configs/cameras.yaml`, hoặc URL RTSP mà máy chạy có thể truy cập.

Model, video nguồn, dataset, video output và database nằm ngoài Git và không được tải tự động. Đường dẫn mặc định là:

```text
models/rtmo_m/rtmo_m_640x640_coco.onnx
models/helmet_classifier_efficientnet/best.pt
4281404-uhd_3840_2160_25fps.mp4
4284182-uhd_3840_2160_25fps.mp4
14656573_2160_3840_30fps.mp4
```

Có thể đổi đường dẫn trong YAML. Thông tin nguồn model nằm trong [`src/inference/README.md`](../src/inference/README.md); cần kiểm tra quyền sử dụng model, dataset và video trước khi phân phối.

## 2. Cài đặt bằng uv

Cài `uv` theo phương pháp phù hợp với môi trường, sau đó cài GStreamer và dependency Python:

```bash
sudo apt update
sudo apt install -y python3 python3-gi gir1.2-gstreamer-1.0 \
  gir1.2-gst-plugins-base-1.0 gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-libav
uv sync --group pose --group training --extra dev
```

Hai nhóm `pose` và `training` chứa các thư viện inference cần cho checkpoint hiện tại. Worker GStreamer mặc định dùng `/usr/bin/python3`, độc lập với phiên bản Python mà `uv` chọn.

Kiểm tra binding và plugin:

```bash
/usr/bin/python3 -c 'import gi; gi.require_version("Gst", "1.0"); from gi.repository import Gst; Gst.init(None); print(Gst.version_string())'
gst-inspect-1.0 uridecodebin
gst-inspect-1.0 appsink
```

Nếu thiếu `gst-inspect-1.0`, cài `gstreamer1.0-tools`. Có thể đặt biến `ISMS_GSTREAMER_PYTHON` trỏ tới Python khác đã cài PyGObject và GStreamer.

## 3. Cấu hình camera và model

- `configs/models.yaml` chứa đường dẫn pose ONNX và helmet checkpoint. Đường dẫn tương đối được tính từ thư mục chạy, thông thường là thư mục gốc project.
- `configs/cameras.yaml` chứa `id`, `source`, `width`/`height`, ROI và algorithm của từng camera. `source` có thể là file hoặc URL RTSP.
- Tọa độ ROI chuẩn hóa từ 0 đến 1 theo frame sau resize. Chiều rộng phải dương và chia hết cho 4. Cần hiệu chỉnh polygon theo từng camera.
- `alert_devices` mặc định ghi log mô phỏng. Có thể thêm `endpoint` HTTP cho gateway. Thông báo mặc định tiếng Anh; đặt `alert_language: vi` để dùng tiếng Việt. `runtime.enable_mock_relay` bật/tắt relay mô phỏng.
- `storage` đặt đường dẫn SQLite và snapshot. Không đưa thông tin đăng nhập RTSP vào Git.

Camera 1 hiện chạy rule mũ và ROI độc lập; camera 2 chỉ chạy ROI; camera 3 chỉ kiểm tra mũ. Adapter RTMO hiện dùng CPU. Classifier tự chọn CUDA nếu PyTorch nhận diện được, nếu không dùng CPU. Container mặc định chưa cấu hình GPU.

## 4. Chạy demo CLI

```bash
./scripts/run_demo.sh
```

Hoặc chỉ định cấu hình khác:

```bash
uv run --group pose --group training python -m src.isms.main --config configs/cameras.yaml
```

CLI chạy các camera được bật và in thư mục `videos/demo/<run-id>/`. Mỗi camera có một MP4 overlay; SQLite lưu tại `storage.path` và snapshot lưu tại `storage.snapshot_dir`. Dừng bằng `Ctrl+C`. Đặt `runtime.record_demo: false` để tắt ghi video.

Khi đọc hết MP4, hệ thống hiện tạo `CAMERA_OFFLINE`. Hàng đợi có thể bỏ frame khi quá tải; video demo ghi mặc định 8 FPS cố định và không giữ PTS nguồn. Khi camera lỗi, relay giữ trạng thái intrusion active; cần kiểm tra các incident còn active khi dùng lại database cho lượt demo mới.

## 5. Chạy API

```bash
uv run --group pose --group training uvicorn src.isms.api.main:app --host 0.0.0.0 --port 8000
```

- Dashboard: `http://localhost:8000/`
- Tài liệu API: `http://localhost:8000/docs`
- Trạng thái camera và relay: `http://localhost:8000/health`
- Lịch sử incident: `http://localhost:8000/events`
- Cập nhật trực tiếp: `ws://localhost:8000/ws/events`

Dùng `SAFETY_CONFIG` để chọn YAML hoặc `SAFETY_DATABASE` để ghi đè đường dẫn database của API. CLI và API đều tạo camera task riêng, nên chỉ chạy một trong hai cho một instance giám sát. API chỉ ghi video khi cấu hình `runtime.video_dir`; CLI tự tạo thư mục output theo lượt chạy.

## 6. Chạy bằng Docker Compose

`Dockerfile` build API image. Compose mount `./configs` vào `/app/configs`, `./videos` vào `/app/videos`, `./models` vào `/app/models` và named volume vào `/app/data`. YAML mặc định trỏ tới video tại thư mục gốc host, nhưng các file đó không được mount vào vị trí tương ứng trong container. Tạo cấu hình Docker riêng và đặt video vào thư mục được mount:

```bash
mkdir -p videos
cp 4281404-uhd_3840_2160_25fps.mp4 videos/
cp 4284182-uhd_3840_2160_25fps.mp4 videos/
cp 14656573_2160_3840_30fps.mp4 videos/
cp configs/cameras.yaml configs/cameras.docker.yaml
```

Sửa ba giá trị `source` trong `configs/cameras.docker.yaml` thành `/app/videos/<tên-file>`. Đặt thêm `storage.snapshot_dir: /app/data/snapshots` để snapshot nằm trong volume. Giữ `model_config: models.yaml`; đường dẫn model tương đối mặc định sẽ được tính dưới `/app/models`.

```bash
docker compose build monitor
docker compose run --rm --service-ports \
  -e SAFETY_CONFIG=/app/configs/cameras.docker.yaml monitor
```

Mở `http://localhost:8000/health` và `/docs`, dừng bằng `Ctrl+C`. Compose đặt `SAFETY_DATABASE=/app/data/safety_events.db` cho API nên lịch sử nằm trong named volume `monitor-data`. Snapshot chỉ nằm trong volume nếu cấu hình `storage.snapshot_dir` như trên. `docker compose run` khởi động dependency `relay`; flow MVP dùng mock relay trong process và không gửi event tới relay HTTP qua `SAFETY_RELAY_URL`.

Cấu hình container mặc định chạy CPU; GPU cần cấu hình runtime/device riêng và chưa được kiểm thử. Quy trình Compose chưa thể chạy thử trong WSL hiện tại vì Docker Desktop chưa bật WSL integration.

## 7. Kiểm thử và xử lý lỗi

```bash
uv run --group pose --group training --extra dev pytest -q
```

- **Không mở được nguồn:** kiểm tra đường dẫn từ thư mục chạy hoặc container mount, và kiểm tra URL RTSP cùng mạng.
- **Thiếu `gi` hoặc phần tử GStreamer:** cài binding/plugin cho Python của worker và kiểm tra bằng `gst-inspect-1.0`.
- **Thiếu model/checkpoint:** đặt file đúng đường dẫn `configs/models.yaml`; Git không chứa các file này.
- **Bị bỏ frame:** xem `queue_dropped`, giảm tải nguồn hoặc tối ưu inference. Bộ đếm không bao gồm frame bị bỏ bên trong appsink.
- **Không có video output:** chạy CLI và bật recording, kiểm tra đường dẫn được ghi trong log. API cần `runtime.video_dir`.
- **Docker không chạy trong WSL:** bật WSL integration trong Docker Desktop và kiểm tra `docker compose version`.
