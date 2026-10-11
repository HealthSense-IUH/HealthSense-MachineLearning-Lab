# 05 — Hiệu chỉnh và chính sách cảnh báo của V5

| Script | Việc |
|---|---|
| `calibration_loso.py` | Hiệu chỉnh xác suất đánh giá bằng LOSO |
| `alert_policy_sweep.py` | Quét ngưỡng và số cửa sổ liên tiếp (ví dụ 3-of-3 tại 0,90) |
| `nested_policy_validation.py` | Kiểm định lồng để chọn chính sách không nhìn test |
| `audit_selected_policy.py` | Soi lại chính sách đã chọn trên từng bệnh nhân |

Đầu ra (`*.csv`, `*.log`) nằm trong `outputs/`. Kết quả được ghi vào metadata của V5 (`models/mimic/healthsense_af_v5_rf_ac_frozen.json`). Ở V6 chính sách này được thay bằng Global rule V2 trong `08`.
