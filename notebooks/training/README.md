# Notebook huấn luyện các mô hình

Mỗi notebook là bản đọc được, có giải thích từng bước, của một script huấn luyện trong repo. Chúng dùng chung thư viện `src/healthsense_ml` nên số ra phải trùng với script.

| Notebook | Model sinh ra | Script gốc | Đầu vào cần có | Thời gian |
|---|---|---|---|---|
| `01_train_v4_pooled_model.ipynb` | `models/final/healthsense_afib_pipeline.pkl` (V4, đang chạy trên AI-Service) | `scripts/run_final_model.py` | `data/features/v4/`, `data/features/v5/afdb_features_v5_afib_only.csv` | vài phút |
| `02_train_v5_frozen_rf.ipynb` | `models/mimic/healthsense_af_v5_rf_ac_frozen.pkl` | `experiments/06_external_validation/train_frozen_model.py` | `experiments/04_feature_model/outputs/af_features_cosen_ac.csv` | vài phút |
| `03_train_v6c_stacking.ipynb` | `models/multidomain/healthsense_af_v6c_stacking.pkl` (V6C, báo cáo trong bài báo) | `experiments/07_v6_multidomain/train_v6c_stacking.py` | `data/features/v7/healthsense_af_v6c_pacaware_train.csv` | 20 đến 40 phút |

Cách chạy từ gốc repo, dùng venv của repo:

```bash
venv\Scripts\jupyter lab notebooks/training
```

hoặc chạy không cần mở giao diện:

```bash
venv\Scripts\jupyter nbconvert --to notebook --execute --inplace notebooks/training/03_train_v6c_stacking.ipynb
```

Mặc định notebook 02 và 03 ghi ra tên `*_retrained.pkl` để không đè lên model đã đóng băng; bật `OVERWRITE_FROZEN = True` khi cố ý tạo lại bản chuẩn.

Các bước sau huấn luyện (Platt scaling, Global rule V2, LODO, bootstrap) không nằm ở đây mà trong `experiments/08_v6_locked_protocol`, vì chúng dò tham số trên DEV rồi khoá, không phải huấn luyện. Tệp `.pkl` sinh ra bị gitignore; xem `data/README.md` về cách chia sẻ dữ liệu và model.
