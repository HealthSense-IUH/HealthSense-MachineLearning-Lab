# 08 — Protocol đánh giá khoá của V6 (nguồn số liệu công bố)

Chạy **đúng thứ tự số**, từ gốc repo, sau khi có `models/v6/healthsense_af_v6c_stacking.pkl`, `data/features/v6/` và `07/artifacts/hard_domain_splits/`. Tham số dò trên DEV được đóng băng kèm SHA-256 trước khi chạm TEST.

| # | Script | Việc | Đầu ra chính |
|---|---|---|---|
| 01 | `generate_locked_predictions.py` | Dự đoán của V6C trên DEV, TEST của ba domain | `*_locked_predictions.csv` (gitignore) |
| 02 | `audit_temporal_and_splits.py` | Kiểm tra thứ tự thời gian và không trùng bệnh nhân | `*_temporal_order_audit.csv` |
| 03 | `build_verified_streams.py` | Chuỗi cửa sổ liên tục đã xác minh cho từng bệnh nhân | `*_verified_streams.csv` |
| 04 | `tune_global_rule_dev.py` | Dò luật cảnh báo V1 trên DEV | `FROZEN_GLOBAL_RULE.json` |
| 05 | `evaluate_frozen_rule_test.py` | Chấm luật V1 trên TEST | |
| 06 | `tune_global_rule_v2_dev.py` | Dò Global rule V2 (p ≥ 0,8, bất định ≤ 0,05, 5 cửa sổ; rescue 0,9 và 2) | `FROZEN_GLOBAL_RULE_V2.json` |
| 07 | `evaluate_global_rule_v2_test.py` | Chấm V2 trên TEST ở mức bệnh nhân | `global_rule_v2_reused_test_summary.csv` |
| 08 | `calibration_diagnostics.py` | Platt scaling fit trên DEV, Brier và LogLoss trước, sau | `FROZEN_PLATT_CALIBRATOR.json`, `calibration_summary.csv` |
| 09 | `subject_cluster_bootstrap.py` | Khoảng tin cậy mức cửa sổ bằng bootstrap theo cụm bệnh nhân | |
| 10 | `exact_subject_ci.py` | Khoảng tin cậy Wilson và exact ở mức bệnh nhân | `global_rule_v2_exact_subject_ci.csv` |
| 11 | `lodo_domain_generalization.py` | Leave-One-Domain-Out, huấn luyện lại 4 lần | `lodo_*_summary.json` |
| 12 | `collect_lodo_results.py` | Gom LODO thành một bảng | `lodo_domain_generalization_summary.csv` |
| 13 | `ppg_ac_ablation_dev.py` | 13 so với 14 đặc trưng trên DEV | |
| 14 | `pac_pvc_challenge.py` | Tỉ lệ báo nhầm trên bộ PAC/PVC | `pac_pvc_challenge_summary.csv` |
| 15 | `deepbeat_lodo_forensics.py` | Vì sao LODO DeepBeat kém: FN so với TP | `deepbeat_lodo_*.csv` |
| 16 | `deepbeat_domain_exposure.py` | Thêm bệnh nhân DeepBeat vào train thì được gì | `deepbeat_domain_exposure_summary.csv` |
| 17 | `deepbeat_lodo_ppg_ac_ablation.py` | PPG_AC trên LODO DeepBeat | |
| 18 | `finalize_v6_research_freeze.py` | Chốt freeze, ghi SHA-256 mọi hiện vật | `V6_RESEARCH_FREEZE.md`, `V6_RESEARCH_FREEZE_SHA256.csv` |

Các bảng tổng hợp trong `artifacts/` được commit và là nguồn của bài báo; dự đoán từng cửa sổ thì không. Bài báo và notebook `review/`, `defense/` đọc từ đây.
