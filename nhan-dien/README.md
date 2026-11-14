# Nhận diện giọt mưa

Phần này gồm bộ suy luận **YOLOv8n-OBB** hiện có và mã huấn luyện **YOLO26n semantic segmentation** được bổ sung từ file TXT. Hai phần là hai thí nghiệm khác nhau: checkpoint OBB đi kèm không phải kết quả của script huấn luyện phân đoạn.

## Thành phần

| Đường dẫn | Nội dung |
| --- | --- |
| `test_raindrop.py` | Nhận diện trên ảnh, vẽ khung xoay, xuất JSON và CSV |
| `model/best.pt` | Checkpoint nhận diện có sẵn, khoảng 25 MB |
| `model/metadata.json` | Nguồn checkpoint, dung lượng và SHA256 |
| `01_SETUP_CPU.cmd` | Tạo môi trường và cài PyTorch CPU cùng Ultralytics |
| `02_TEST_FULL.cmd` | Chạy suy luận toàn ảnh |
| `03_TEST_GRID16X3.cmd` | Chạy chế độ chia lưới để so sánh |
| `training/` | Mã huấn luyện phân đoạn 1 kênh trên Kaggle và hướng dẫn |

## Chạy trên Windows

Dùng Python 3.11 hoặc 3.12 cho bộ cài CMD. Mở thư mục `nhan-dien` sau khi tải repo:

1. Chạy `01_SETUP_CPU.cmd` và chờ thông báo `SETUP OK`. Bước này cần Internet để tải thư viện.
2. Đặt ảnh JPG, PNG, BMP, TIFF hoặc WebP vào thư mục `input`.
3. Chạy `02_TEST_FULL.cmd`.
4. Xem `output/full_001/images`, `detections` và `summary.csv`. Các lần sau tạo thư mục mới để giữ kết quả cũ.
5. Có thể chạy `03_TEST_GRID16X3.cmd` để so sánh với chế độ lưới.

Ultralytics được khóa ở phiên bản **8.4.175**, theo bộ nhận diện cục bộ. Môi trường inference này độc lập với môi trường training phân đoạn.

### Chạy bằng dòng lệnh

Tại thư mục `nhan-dien`:

```powershell
.venv\Scripts\python.exe test_raindrop.py --input "D:\Anh_mua" --mode full --conf 0.25 --device cpu
```

```powershell
.venv\Scripts\python.exe test_raindrop.py --input "D:\Anh_mua" --mode grid --cols 16 --rows 3 --imgsz 256 --conf 0.25 --device cpu
```

Trên hệ điều hành khác, dùng môi trường Python có PyTorch phù hợp, cài `pip install -r requirements.txt`, rồi gọi `python test_raindrop.py` với cùng tham số. Có thể đổi `--weights`, `--output`, `--imgsz`, `--conf`, `--iou` và `--device`. GPU cần PyTorch hỗ trợ CUDA; bộ cài CMD mặc định dùng CPU.

## FULL và GRID16X3

- **FULL:** đưa toàn ảnh qua letterbox ở kích thước mặc định 640×640.
- **GRID16X3:** chia ảnh thành 16 cột × 3 hàng không chồng lấn; suy luận từng crop ở 256×256 rồi đổi tọa độ về ảnh gốc.

Lưới 16×3 là chế độ so sánh được suy ra từ tên run cũ `rain16x3_fast_group_batch64`. Chưa có notebook tạo crop của run OBB để xác nhận đây là toàn bộ tiền xử lý gốc. Giọt nằm trên đường chia có thể bị cắt, bỏ sót hoặc tạo nhiều khung.

## Đọc kết quả

- `images/`: ảnh được vẽ khung dự đoán; ảnh nguồn được giữ nguyên.
- `detections/`: JSON chứa lớp, confidence và bốn góc khung xoay theo tọa độ pixel sau khi xử lý EXIF.
- `summary.csv`: số khung, thời gian xử lý và lỗi của từng ảnh.
- `settings.json`: thông số của lần chạy.

Confidence không phải độ chính xác của toàn bộ model. Số khung không đồng nghĩa số giọt thật. Thời gian xuất kết quả bao gồm đọc, suy luận và ghi ảnh, nên không dùng trực tiếp làm FPS. Cần nhãn test độc lập để đánh giá precision, recall hoặc mAP.

Đầu ra OBB là khung xoay, chưa phải mask theo biên giọt và chưa tự nối với pipeline phục hồi Hybrid.

## Phân đoạn giọt mưa

Xem [hướng dẫn huấn luyện](training/README.md). Script này tạo mask hai lớp background/raindrop từ ảnh grayscale một kênh, với lựa chọn 640×360 hoặc 1280×720. Repository chưa kèm checkpoint được huấn luyện bằng script đó.

## Phạm vi upload

Bản repo giữ mã chạy, checkpoint đã chọn, script cài đặt và mã huấn luyện. Môi trường `.venv`, cache, dataset, ZIP kết quả và output hàng loạt không được đưa lên. Các file nguồn trên máy được giữ nguyên; bản upload dùng thư mục `input` cạnh script thay cho đường dẫn cá nhân cố định.

Checkpoint SHA256: `f406e9b69a66f4e4630816fb9605780a1ef1c9830aa2dabf407858d36d9c8659`.
