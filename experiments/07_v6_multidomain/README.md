# 07 — Phát triển V6: đa domain, hard negative, stacking

Thư mục làm việc của quá trình nghiên cứu V6, nhiều script là các nhánh thử nghiệm. **Đường chính** để tái tạo model V6C chỉ gồm các script in đậm; phần đánh giá cuối cùng đã được chuyển sang protocol khoá ở `08`.

Bố cục: 9 script đường chính ở mặt trước; `dev/` chứa các nhánh huấn luyện và đánh giá trong quá trình phát triển; `temporal_archive/` chứa lớp thời gian V6D đến V6H đã gác lại; `artifacts/` là split và dự đoán (gitignore).

## Chuẩn bị dữ liệu

| Script | Việc |
|---|---|
| `prepare_deepbeat_signal_memmap.py` | Đọc tín hiệu DeepBeat vào memmap |
| **`extract_deepbeat_v6_features.py`** | Trích 14 đặc trưng cho DeepBeat và MIMIC → `data/features/v6/` |
| `build_deepbeat_v6_manifest.py`, `audit_deepbeat_subjects.py` | Danh sách bệnh nhân DeepBeat và kiểm tra trùng |
| **`build_hard_domain_subject_splits.py`** | Chia PulseWatch và Liu thành train, DEV, TEST theo bệnh nhân → `artifacts/hard_domain_splits/` |
| `build_v6b_manifest.py`, **`build_v6b_clean_manifest.py`** | Gộp bốn domain thành bảng train → `data/features/v7/` |

## Huấn luyện

| Script | Việc |
|---|---|
| `dev/train_v6a_multidomain.py` | V6A: Random Forest đa domain |
| `dev/train_v6b_multidomain.py`, `dev/train_v6b_clean.py` | V6B: thêm làm sạch nhãn |
| `dev/train_v6c_pacaware.py` | V6C biến thể Random Forest có trọng số PAC/PVC |
| **`train_v6c_stacking.py`** | **V6C chính thức**: Extra Trees + Random Forest + XGBoost, meta-learner hồi quy logistic → `models/multidomain/`. Bản notebook: `notebooks/training/03_train_v6c_stacking.ipynb` |

## Đánh giá trong quá trình phát triển (`dev/`)

`evaluate_v6a_external_reused.py`, `mimic_loso_v6a_retention.py`, `bootstrap_v6a_validation.py`, `evaluate_v6b_*.py`, `evaluate_v6c_all.py`, `evaluate_v6c_stack_external.py`, `benchmark_v6c_models.py`, `benchmark_v6c_sota_models.py`, `evaluate_v6c_sota_external.py`, `sqi_temporal_ablation.py`. Số liệu ở đây là số phát triển; số công bố lấy từ `08`.

## Lớp thời gian thử nghiệm V6D đến V6H (`temporal_archive/`, đã thay thế)

`generate_*_temporal_predictions.py`, `generate_v6e_*.py`, `evaluate_v6e_uncertainty.py`, `evaluate_v6f_metrics.py`, `temporal_alert_evaluation*.py`, `v6d_*.py`, `v6f_*.py`, `v6g_*.py`, `v6h_*.py`. Các luật này được dò trên tập đã xem nhiều lần nên bị gác lại; Global rule V2 trong `08` là bản được giữ.

Thư mục `artifacts/` (gitignore) chứa split và dự đoán dẫn xuất từ dữ liệu Synapse; xem `data/README.md` về cách chia sẻ nội bộ.
