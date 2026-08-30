"""Bước 1 — Trích xuất đặc trưng v4 (CÓ record_id).

Đọc data/raw/mimic_perform (tự tải từ Kaggle nếu chưa có)
-> lọc bandpass, phát hiện nhịp, chuỗi NN
-> cửa sổ trượt 30s/10s -> 16 đặc trưng HRV + record_id
-> lưu data/features/mimic_features_v4.csv

Chạy:  python scripts/run_v4_extraction.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from healthsense_ml import config
from healthsense_ml.feature_extraction import extract_all


def main():
    print('=' * 60)
    print('🔬 HEALTHSENSE ML v4 — TRÍCH XUẤT ĐẶC TRƯNG (record-aware)')
    print(f'   Cửa sổ: {config.WINDOW_S}s | Bước: {config.STEP_S}s | '
          f'Tối thiểu {config.MIN_BEATS_PER_WINDOW} nhịp/cửa sổ')
    print('=' * 60)

    if not os.path.isdir(config.RAW_DIR):
        print('\n📥 Chưa có dữ liệu raw — đang tải từ Kaggle (public, ~100MB)...')
        from healthsense_ml.data_loading import download_dataset
        download_dataset()

    extract_all(verbose=True)
    print('\n✅ Hoàn tất! Bước tiếp theo: python scripts/run_v4_benchmark.py')


if __name__ == '__main__':
    main()
