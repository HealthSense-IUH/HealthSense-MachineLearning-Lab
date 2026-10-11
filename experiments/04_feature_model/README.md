# 04 — Đặc trưng và chọn model (nền của V5)

Thí nghiệm sinh bảng 14 đặc trưng (13 HRV + PPG_AC) trên MIMIC và chọn model trước khi đóng băng V5.

| Script | Việc |
|---|---|
| `build_af_features.py` | Bảng đặc trưng HRV cơ bản cho MIMIC |
| `build_cosen_ac_features.py` | Thêm CoSEn và PPG_AC → `af_features_cosen_ac.csv`, đầu vào của `06/train_frozen_model.py` |
| `ac_quality_audit.py` | Kiểm tra PPG_AC có bị chất lượng tín hiệu chi phối không |
| `feature_ablation_v1.py`, `af_feature_ablation.py`, `cosen_ac_ablation.py` | Bỏ từng nhóm đặc trưng xem mất bao nhiêu điểm |
| `iqr_audit.py` | Kiểm tra ảnh hưởng của lọc ngoại lai IQR |
| `model_selection_bootstrap.py` | So các model bằng bootstrap theo bệnh nhân |

Mọi đầu ra (`*.csv`, `*.log`) nằm trong `outputs/`; riêng `outputs/af_features_cosen_ac.csv` được commit vì là đầu vào của V5.
