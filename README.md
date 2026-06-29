# Roundabout Vehicle Detection — YOLOv11

> Capstone Project · Machine and Deep Learning 03  
> Phân tích lưu lượng giao thông tại vòng xuyến qua ảnh hàng không

---

## Giới thiệu

Dự án xây dựng hệ thống phát hiện phương tiện giao thông (xe ô tô, xe tải, xe buýt, xe máy) trong ảnh/video chụp từ UAV tại các nút giao vòng xuyến, sử dụng kiến trúc **YOLOv11**.

Gồm 2 phần:

| Phần | Mô tả |
|------|-------|
| `training/` | Notebook huấn luyện YOLOv11 trên tập Roundabout Aerial Images |
| Web App | Demo inference qua giao diện web FastAPI |

---

## Cấu trúc dự án

```
├── training/
│   └── roundabout_yolov11.ipynb   # Notebook training (Colab/Kaggle)
├── templates/
│   └── index.html                 # Giao diện web
├── static/
│   └── results/                   # Ảnh/video kết quả (tự sinh, không commit)
├── main.py                        # FastAPI server
├── best.pt                        # Model đã train (YOLOv11)
├── run.bat                        # Script chạy nhanh (Windows)
└── requirements.txt
```

---

## Quy trình thực hiện

### 1. Data Exploration & EDA
- Tải bộ dữ liệu **Roundabout Aerial Images** (YOLO format)
- Thống kê phân bố nhãn theo từng class (Car, Truck, Bus, Motorcycle...)
- Vẽ **Heatmap** vị trí tâm bounding box → phát hiện spatial bias (phương tiện tập trung ở vùng biên trái `x < 0.3`)
- Hiển thị ≥ 5 ảnh mẫu kèm bounding box gốc

### 2. Data Preparation
- Chia tập: **Train / Validation / Test** theo tỷ lệ chuẩn (Stratified Split)
- Cấu hình file `roundabout.yaml` với đường dẫn và danh sách class
- Kiểm tra tính nhất quán: mỗi ảnh có đúng 1 file nhãn tương ứng

### 3. Augmentation Strategy
Dựa trên kết quả Heatmap, áp dụng các phép biến đổi hình học để xử lý tính đa hướng của phương tiện tại vòng xuyến:

| Augmentation | Lý do |
|---|---|
| `degrees=180` | Phương tiện di chuyển mọi hướng trong vòng xuyến |
| `fliplr=0.5` | Giảm spatial bias trái/phải |
| `flipud=0.5` | Xử lý góc nhìn top-down đối xứng |
| `mosaic=1.0` | Tăng đa dạng mật độ phương tiện |
| `scale=0.5` | Mô phỏng phương tiện ở nhiều khoảng cách |

### 4. Model Training — YOLOv11
- Kiến trúc: **YOLOv11s / YOLOv11m**
- Input resolution: `imgsz=640` (giữ đặc trưng phương tiện nhỏ trong ảnh hàng không)
- Tối ưu siêu tham số với **Optuna**: lr0, optimizer (AdamW/SGD), momentum, scale, mosaic
- Output: `best.pt`

### 5. Evaluation & Diagnostic
- Chỉ số tổng thể: `mAP@50`, `mAP@50-95`
- So sánh Per-class mAP: Không Augmentation vs Có Augmentation
- Phân tích **Confusion Matrix**: phát hiện nhầm lẫn Truck↔Bus (góc nhìn top-down)
- Phân tích IoU tại vùng biên trái (`x < 0.3`) — vùng có tỷ lệ lỗi cao nhất

---

## Cài đặt & Chạy Demo

### Yêu cầu
- Python 3.10+
- Windows (có `run.bat`) hoặc Linux/Mac

### Cài thư viện

```bash
pip install -r requirements.txt
```

### Khởi động web app

**Windows** — double-click `run.bat`  
hoặc chạy lệnh:

```bash
uvicorn main:app --reload
```

Mở trình duyệt tại **http://127.0.0.1:8000**

### Sử dụng

1. Upload ảnh (JPG, PNG, WEBP) hoặc video (MP4, AVI, MOV, MKV)
2. Nhấn **Detect**
3. Xem kết quả bounding box + confidence score cho từng phương tiện

---

## Kết quả

| Metric | Giá trị |
|--------|---------|
| mAP@50 | — |
| mAP@50-95 | — |
| Best epoch | — |

> Điền kết quả thực tế sau khi train xong.

---

## Công nghệ sử dụng

- [YOLOv11](https://github.com/ultralytics/ultralytics) — Object Detection
- [FastAPI](https://fastapi.tiangolo.com/) — Web Backend  
- [OpenCV](https://opencv.org/) — Xử lý ảnh
- [Optuna](https://optuna.org/) — Hyperparameter Tuning
- [Google Colab / Kaggle](https://colab.research.google.com/) — Training GPU

---

## Notebook Training

Notebook đầy đủ (EDA → Train → Evaluate) được lưu trên Google Drive:

**[Mở trên Google Drive](https://drive.google.com/drive/folders/1g-m2ngEMykuLrMfbZTnFc4BsxzUUkSWk?usp=sharing)**

> File `capstone_roundabout.ipynb` — chạy trên Google Colab (cần GPU)

---

## Dataset

**Roundabout Aerial Images** (YOLO format)  
Classes: `car`, `truck`, `bus`, `motorcycle` (và các class khác theo thứ tự ID gốc)

---

## Tác giả

**Nguyen Duong Cong Thanh**  
Machine and Deep Learning 03 — Tháng 3, 2026
