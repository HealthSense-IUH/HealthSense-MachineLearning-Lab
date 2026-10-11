# results/ — số liệu đã sinh, chia theo phiên bản

| Thư mục | Nội dung | Sinh bởi |
|---|---|---|
| `v4/` | `benchmark_v4/` LOSO 35 bệnh nhân MIMIC, `cross_dataset/` MIMIC ↔ AFDB, `beat_validation/` dò nhịp so với R-peak ECG, `pooled_loso_*` 60 bệnh nhân; `baseline/`, `label_audit/`, `beat_detector/`, `sqi/` là bảng nhỏ của các kiểm tra sớm mà notebook defense đọc | `notebooks/v4/` |
| `v5/` | Bảng tóm tắt, đặc trưng và dự đoán khi kiểm định ngoài V5 (toàn bộ ignore, dẫn xuất từ Synapse); `README_v5_external_validation.md` ghi lại số liệu | `notebooks/v5/02_external_validation` |
| `v6/` | Số liệu công bố: cửa sổ, LODO, Global rule V2, Platt, PAC/PVC, SHA-256 của freeze (commit); dự đoán từng cửa sổ và verified streams (ignore) | `notebooks/v6/03_locked_protocol` |

Model không nằm ở đây, xem `models/`.
