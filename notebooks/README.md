# notebooks/ — mỗi phiên bản model một thư mục, chạy theo số thứ tự

| Thư mục | Chuỗi notebook | Sinh ra |
|---|---|---|
| `v4/` | `00_prepare_features` → `01_evaluate_loso_cross_dataset` → `02_train_pooled_model` | `data/features/v4/`, kết quả LOSO và cross-dataset ở `results/v4/`, model ở `models/final/` (bản AI-Service đang dùng là `models/healthsense_afib_pipeline.pkl`) |
| `v5/` | `01_train_frozen_rf` → `02_external_validation` → `review_visual` | model V5 ở `models/mimic/`, kết quả kiểm định ngoài ở `results/v5/` |
| `v6/` | `00_extract_features_and_splits` → `01_prepare_training_table` → `02_train_stacking` → `03_locked_protocol` → `review_visual` | `data/features/v6,v7/`, `data/splits/`, model V6C ở `models/multidomain/`, số liệu công bố ở `results/v6/` |
| `defense/` | 3 notebook bằng chứng, đọc từ `results/` | hình và bảng để bảo vệ |

Mỗi thư mục phiên bản có `scripts/` (các bước dài, notebook gọi bằng `run(...)`); riêng `v6/protocol/` là 18 script của protocol khoá. Mọi notebook tự tìm gốc repo nên mở từ đâu cũng chạy; môi trường là `venv/` của repo.

Chạy không cần giao diện:

```bash
venv\Scripts\jupyter nbconvert --to notebook --execute --inplace notebooks/v6/02_train_stacking.ipynb
```

Notebook huấn luyện mặc định ghi `*_retrained` hoặc `*_rebuilt`, chỉ ghi đè hiện vật đóng băng khi bật `OVERWRITE_FROZEN` / `OVERWRITE`. Những bước cần dữ liệu Synapse (V5 kiểm định ngoài, V6 trích đặc trưng) chỉ chạy được trên máy đã tải dữ liệu đó.
