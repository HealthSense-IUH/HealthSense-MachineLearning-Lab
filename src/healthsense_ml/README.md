# healthsense_ml — thư viện dùng chung của mọi thí nghiệm

Đây là **định nghĩa gốc** của phép đo: lọc tín hiệu, dò nhịp, chuỗi NN và 16 đặc trưng HRV (13 đặc trưng dùng huấn luyện sau khi bỏ nhóm LF). `HealthSense-AI-Service/app/services/hrv_v4.py` là bản sao được kiểm tra tương đương bằng `tests/parity_check.py` bên service; sửa một bên thì phải chạy lại parity.

## Thứ tự đọc

| # | Module | Việc |
|---|---|---|
| 1 | `config.py` | Hằng số: 125 Hz, dải lọc 0,5–8 Hz, cửa sổ 30 s bước 10 s, danh sách đặc trưng, đường dẫn `data/`, `models/`, `results/` |
| 2 | `data_loading.py` | Nạp PPG thô MIMIC PERform AF từ `data/raw/mimic_perform/` |
| 3 | `signal_processing.py` | Lọc bandpass → dò nhịp theo z-score → chuỗi NN trong giới hạn sinh lý |
| 4 | `beat_validation.py` | Chấm bộ dò nhịp PPG bằng R-peak của ECG ghi song song |
| 5 | `hrv_features.py` | Chuỗi NN → 16 đặc trưng HRV theo Task Force 1996 (thời gian, tần số, Poincaré, SampEn) |
| 6 | `feature_extraction.py` | Cắt cửa sổ trượt và gom bảng đặc trưng `data/features/v4/mimic_features_v4.csv` |
| 7 | `afdb.py` | Cùng pipeline cho MIT-BIH AFDB (ECG) → `data/features/v5/afdb_features_v5_afib_only.csv` |
| 8 | `training.py` | Registry model và lưới tham số, mặt nạ IQR chỉ fit trên train |
| 9 | `evaluation.py` | LOSO, metric hai mức cửa sổ và bệnh nhân |
| 10 | `manifest.py` | Sinh `data/MANIFEST.csv` (SHA-256 của dữ liệu và model), chạy `python -m healthsense_ml.manifest` |

Ba module đầu tiên (1, 3, 5) là phần AI-Service import trực tiếp; không đổi chữ ký hàm của chúng.

Cách chạy pipeline V4 đầu cuối: `notebooks/v4/00_prepare_features.ipynb` → `notebooks/v4/01_evaluate_loso_cross_dataset.ipynb` → `notebooks/v4/02_train_pooled_model.ipynb`.

Phần V5 và V6 (PPG_AC, đa domain, stacking) không nằm ở đây mà trong `notebooks/v5/` và `notebooks/v6/` (kèm `scripts/`), vì chúng có tham số đóng băng riêng; chúng vẫn import thư viện này để tính đặc trưng.
