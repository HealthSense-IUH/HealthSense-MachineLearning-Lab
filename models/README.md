# models/ — model nạp được vào chương trình, mỗi phiên bản một thư mục

| Thư mục | Model | Commit? | Sinh bởi |
|---|---|---|---|
| `v4/` | `healthsense_afib_pipeline.pkl` (248 KB): sklearn Pipeline `StandardScaler → XGBClassifier`, 13 đặc trưng HRV theo thứ tự trong `model_card.json`, train trên 60 bệnh nhân MIMIC + AFDB có cân bằng nguồn. **Đây là bản AI-Service đang chạy** (`HealthSense-AI-Service/app/models/`, sao chép từng byte) | pkl + JSON | `notebooks/v4/02_train_pooled_model` |
| `v5/` | `healthsense_af_v5_rf_ac_frozen.pkl` (1 MB): Random Forest 14 đặc trưng (thêm PPG_AC), đóng băng trên MIMIC; kèm `.json` và `.sha256` | chỉ JSON, SHA | `notebooks/v5/01_train_frozen_rf` |
| `v6/` | `healthsense_af_v6c_stacking.pkl` (1,9 GB): Extra Trees + Random Forest + XGBoost → Logistic Regression meta, 137 476 cửa sổ / 248 bệnh nhân / 4 domain; kèm `.json`. Thẻ JSON của V6A, V6B giữ lại để đối chiếu lịch sử | chỉ JSON | `notebooks/v6/02_train_stacking`; bản chuẩn xin từ máy đã train, đối chiếu SHA-256 trong `data/MANIFEST.csv` |

## Ba quy tắc

1. **Scaler nằm trong Pipeline.** AI-Service từng nhận một `MLPClassifier` trần (`best_model_8165.pkl`) không kèm scaler và cho kết quả rác mà không báo lỗi; `load_model()` bên đó nay từ chối model không đóng gói tiền xử lý.
2. **Thẻ JSON cạnh mỗi pkl** ghi phiên bản, danh sách đặc trưng đúng thứ tự và số liệu thật (các đời v1–v3 từng bị thổi phồng 6–7 điểm vì data leakage).
3. **Notebook train không đè bản đóng băng.** Mặc định ghi `*_retrained.pkl` / `*_retrained.json` cạnh bản chuẩn để so sánh; chỉ ghi đè khi bật `OVERWRITE_FROZEN`. Muốn thay bản triển khai V4 thì bật cờ, chạy lại, rồi chép sang AI-Service và chạy `tests/parity_check.py` bên đó.

Số liệu, bảng, hình không nằm ở đây mà ở [`results/`](../results/README.md).
