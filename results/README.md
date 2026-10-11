# results/ — thứ đem đi báo cáo

Số liệu và biểu đồ. Không có model nào ở đây — model nằm ở
[`models/`](../models/README.md).

## Kiểm định pipeline v4 (sản phẩm)

| Thư mục / file | Nội dung |
|---|---|
| `benchmark_v4/` | LOSO trên 35 bệnh nhân MIMIC: metrics 2 mức, dự đoán từng cửa sổ, confusion matrix, ROC, biểu đồ xác suất theo bệnh nhân |
| `cross_dataset/` | Kiểm định chéo MIMIC (PPG) ↔ MIT-BIH AFDB (ECG) — bằng chứng tổng quát hóa mạnh nhất |
| `pooled_loso_results.csv` | LOSO gộp 60 bệnh nhân — con số của model đang triển khai |
| `pooled_loso_predictions.csv` | Dự đoán từng cửa sổ của lần chạy trên |
| `beat_validation/` | Chấm điểm bộ dò nhịp PPG bằng R-peak trên ECG ghi song song; cũng là nơi lộ ra 2 ca `non_af_012` (nhãn sai) và `non_af_014` (PPG hỏng) |

Sinh lại bằng bốn script trong `scripts/`: `run_v4_extraction.py` → `run_v4_benchmark.py` → `run_cross_dataset.py` → `run_final_model.py`, hoặc notebook `notebooks/training/01_train_v4_pooled_model.ipynb`.

## Kết quả V5 và V6

Không nằm ở đây. V5: `experiments/06_external_validation/`. V6 (số liệu công bố): `experiments/08_v6_locked_protocol/artifacts/`, gồm bảng cửa sổ, LODO, Global rule V2 mức bệnh nhân, Platt, PAC/PVC; xem README của thư mục đó.
