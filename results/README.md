# results/ — thứ đem đi báo cáo

Số liệu và biểu đồ. Không có model nào ở đây — model nằm ở
[`models/`](../models/README.md).

## Bảo tàng phiên bản

| File | Nội dung |
|---|---|
| `v1.json` … `v4.json` | Kết quả từng đời pipeline: cấu hình, điểm chấm kiểu cũ, điểm chấm bằng LOSO, phần chênh lệch, danh sách lỗi rò rỉ, và `original_claim` (con số bản gốc từng công bố) |

Sinh lại: `python src/vN/pipeline.py`.
Đọc dễ hiểu: [`src/report/00_final_report.ipynb`](../src/report/00_final_report.ipynb).

## Kiểm định pipeline v4 (sản phẩm)

| Thư mục / file | Nội dung |
|---|---|
| `benchmark_v4/` | LOSO trên 35 bệnh nhân MIMIC: metrics 2 mức, dự đoán từng cửa sổ, confusion matrix, ROC, biểu đồ xác suất theo bệnh nhân |
| `cross_dataset/` | Kiểm định chéo MIMIC (PPG) ↔ MIT-BIH AFDB (ECG) — bằng chứng tổng quát hóa mạnh nhất |
| `pooled_loso_results.csv` | LOSO gộp 60 bệnh nhân — con số của model đang triển khai |
| `pooled_loso_predictions.csv` | Dự đoán từng cửa sổ của lần chạy trên |
| `beat_validation/` | Chấm điểm bộ dò nhịp PPG bằng R-peak trên ECG ghi song song; cũng là nơi lộ ra 2 ca `non_af_012` (nhãn sai) và `non_af_014` (PPG hỏng) |

⚠️ Năm mục này **không sinh lại được**: các script tạo ra chúng
(`scripts/run_*.py`) đã bị gỡ khỏi repo. Khôi phục bằng:

```bash
git checkout d3123cf -- scripts
```
