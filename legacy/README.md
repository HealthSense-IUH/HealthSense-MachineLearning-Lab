# Legacy — Thí Nghiệm v1–v3 (chỉ để tham khảo)

Toàn bộ nội dung thư mục này thuộc các phiên bản pipeline cũ, **không dùng
cho kết quả báo cáo** vì có data leakage (xem README gốc, mục "Vì sao có v4?").

| Thư mục | Nội dung |
|---|---|
| `data/features/` | Bảng đặc trưng cũ của các thí nghiệm ban đầu (Custom PPG, MIT-BIH, MIMIC train cũ). |
| `data/features/v3_scales/` | Bảng đặc trưng v3 theo 4 quy mô (1360/4083/8165/16358) — KHÔNG có `record_id`. |
| `notebooks/` | Notebook thí nghiệm v1–v2 (Custom_PPG_Walking, MIMIC_Training 01–11, MIT_BIH_AF, visualize_new_data). |
| `notebooks/v3_pipeline/` | Notebook pipeline v3 (trích xuất, chuẩn hóa, huấn luyện đa quy mô). |
| `models/benchmark_v3/` | Kết quả benchmark v3 (bị leakage — con số 94–99% không phản ánh bệnh nhân mới). |
| `models/summary_top_models.csv` | Bảng tổng hợp mô hình cũ (Stacking/MLP 98%+ — bị leakage). |

Pipeline hiện hành: `src/healthsense_ml/` + `scripts/run_v4_*.py` + `scripts/run_cross_dataset.py`.

## Muốn hiểu v1–v3 mà không phải đọc 20 notebook cũ?

Toàn bộ các phiên bản này đã được **tái dựng thành code chạy được** ở
`src/v1/` … `src/v4/`, kèm 5 notebook báo cáo tiếng Việt ở `src/report/`
(đã chạy sẵn, nhúng đủ kết quả và biểu đồ).

Bắt đầu ở [`src/report/00_tong_quan.ipynb`](../src/report/00_tong_quan.ipynb).

Thư mục `legacy/` này giữ nguyên các notebook **gốc** để đối chiếu khi cần.
