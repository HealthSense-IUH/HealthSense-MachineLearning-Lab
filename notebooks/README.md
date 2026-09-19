# HealthSense AI notebooks — V5 & V6

Hai notebook này được thiết kế để đặt vào repository `HealthSense_rungtamnhi/notebooks/`.

## Files

- `HealthSense_V5_Visual_Review.ipynb`
  - frozen V5 RF;
  - exact 14-feature contract;
  - imports code feature extraction từ `src/healthsense_ml`;
  - external validation scripts trong `experiments/06_external_validation`;
  - ROC / PR / probability distributions / confusion matrix;
  - PAC/PVC hard-negative visualization.

- `HealthSense_V6_Research_Freeze_Visual_Review.ipynb`
  - V6C stacking bundle;
  - locked predictions;
  - verified temporal streams;
  - Global Rule V2;
  - calibration;
  - exact confidence intervals;
  - LODO;
  - PPG_AC matched ablation;
  - PAC/PVC;
  - DeepBeat forensic/domain exposure;
  - research freeze/provenance/checksums.

## Usage

```bash
cd /home/phuc/Documents/HealthSense_rungtamnhi
conda activate healthsense-af-v5
jupyter lab
```

Mở notebook từ folder `notebooks/`.

Notebook tự tìm repo root nếu đang chạy từ root hoặc từ đường dẫn mặc định:

`/home/phuc/Documents/HealthSense_rungtamnhi`

## Reuse existing `.py`

Notebook không re-implement training pipeline.

- V5 import trực tiếp `src.healthsense_ml.*` và có cell gọi `experiments/06_external_validation/*.py`.
- V6 đọc artifacts sinh bởi `experiments/08_v6_locked_protocol/*.py`; có cell tùy chọn gọi lại scripts.
- Các bước retrain/LODO nặng mặc định không tự chạy.

## GitHub recommendation

Trước khi push:

1. Chạy notebook từ đầu trong env `healthsense-af-v5`.
2. Kiểm tra mọi path.
3. Chọn giữ outputs/figures nếu muốn GitHub render report trực tiếp.
4. Nếu notebook quá lớn, Clear Outputs rồi commit figures riêng vào `notebooks/figures/`.
5. Không commit raw datasets hoặc model 1.88 GB vào git thường; dùng release/LFS/storage phù hợp nếu cần.
