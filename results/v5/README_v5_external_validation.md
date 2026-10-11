# 06 — Đóng băng V5 và kiểm định ngoài

| Script | Việc |
|---|---|
| `train_frozen_model.py` | Huấn luyện và đóng băng Random Forest 14 đặc trưng trên MIMIC → `models/mimic/`. Bản notebook: `notebooks/v5/01_train_frozen_rf.ipynb` |
| `deepbeat_frozen_v5_external.py` | Chạy V5 trên DeepBeat |
| `pulsewatch_frozen_v5_external.py` | Chạy V5 trên PulseWatch, kể cả cửa sổ PAC/PVC |
| `liu_pseudo30_frozen_v5.py` | Chạy V5 trên Liu 2022 với cửa sổ giả 30 giây |
| `peakdet_pac_pvc_frozen_v5.py` | Bộ thử PAC/PVC hard negative |

Thư mục `audits/` chứa 23 script kiểm tra dữ liệu PulseWatch, Liu, DeepBeat và bản đóng băng SQI `deepbeat_sqi_v1_frozen.joblib`; notebook `defense/01` dẫn tới chúng. Các file `*.csv` dẫn xuất sinh ra ở đây bị gitignore vì dẫn xuất từ dữ liệu Synapse. Kết luận của bước này (AF so với nhịp xoang dễ, PAC/PVC là hard negative) là lý do chuyển sang huấn luyện đa domain ở `07`.
