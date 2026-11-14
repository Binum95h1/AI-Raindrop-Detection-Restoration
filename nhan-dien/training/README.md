# Huấn luyện YOLO26n semantic segmentation một kênh

Mã được lấy từ file TXT người dùng cung cấp. Chỉ tách script bên trong để chạy độc lập; không thực hiện huấn luyện khi chuẩn bị repository.

## Hai cách chạy

**Trong notebook Kaggle:** chép nội dung `kaggle_train_sem_gray_1ch.py` vào một cell, đặt `PERCENT = 50` hoặc `100`, bật GPU rồi chạy. Wrapper cài `ultralytics>=8.4.147`, ghi script vào `/kaggle/working`, gọi subprocess và tạo liên kết tải ZIP.

**Chạy script độc lập trên Kaggle:** cài các gói trong `requirements.txt` với PyTorch phù hợp GPU, sau đó chạy:

```bash
python train_sem_gray_1ch.py --percent 100
```

Hoặc dùng `--percent 50` để tạo dữ liệu 640×360. Giá trị 100 giữ 1280×720. Đây là tỷ lệ kích thước chiều rộng/chiều cao, không phải tỷ lệ số ảnh dùng huấn luyện.

## Dữ liệu đầu vào

Thêm dataset Muaxla vào Kaggle, kiểm tra hai đường dẫn trong script:

```text
/kaggle/input/datasets/nguyentuananh1/muaxla/Train/drop
/kaggle/input/datasets/nguyentuananh1/muaxla/Train/mask_drop/01drop/01drop
```

Ảnh và mask phải khớp đường dẫn tương đối khi bỏ phần mở rộng. Ảnh nguồn có kích thước 1280×720. Mask phải chứa 0/255 hoặc 0/1; mask nhiều kênh phải có ba kênh màu giống nhau.

Script chia train/validation theo thư mục scene đầu tiên, seed 42; gom scene để đạt khoảng 25% ảnh validation, nên tỷ lệ thực tế có thể khác 25%. Cần ít nhất hai scene. Ảnh được đọc grayscale; mask resize bằng nearest-neighbor và ánh xạ thành 0 background / 1 raindrop.

## Cấu hình trong mã

| Tham số | Giá trị |
| --- | --- |
| Model khởi tạo | `yolo26n-sem.pt` |
| Kênh ảnh | 1 |
| Lớp | background, raindrop |
| Epoch tối đa / patience | 100 / 15 |
| Batch tổng | 4 |
| Optimizer | AdamW |
| Learning rate ban đầu | 0.001 |
| `cls_pw` | 0.5 |
| Thiết bị | GPU 0 hoặc [0, 1] nếu có ít nhất hai GPU |

Mã kiểm tra GPU, sự hiện diện của `cls_pw`, task `semantic` và số kênh conv đầu tiên. Khả năng chạy phụ thuộc phiên bản Ultralytics thực tế; chưa xác minh đầy đủ API hoặc hoàn tất một lần train trong lần upload này. Yêu cầu phiên bản tối thiểu được giữ theo mã người dùng, chưa phải môi trường đã khóa để tái lập.

## Kết quả và chạy lại

Dataset xử lý được ghi vào `/kaggle/working/rain_sem_1ch_{50|100}pct`. Khi chạy lại, script xóa các thư mục `images` và `masks` bên trong đúng dataset xử lý đó trước khi tạo lại.

Run nằm trong `/kaggle/working/runs_rain_sem_1ch/yolo26n_sem_{50|100}pct_1ch_pw0p5_b4`. Script từ chối chạy khi `weights/last.pt` của tên run đã tồn tại; đổi `NAME` trong mã để mở thí nghiệm mới.

ZIP đầu ra chứa `weights/best.pt`, `weights/last.pt` và, nếu có, `results.csv`, `args.yaml`. Script kiểm tra conv đầu tiên của checkpoint tốt nhất còn một kênh. Checkpoint OBB ở `../model/best.pt` thuộc thí nghiệm khác và không được dùng làm checkpoint phân đoạn này.
