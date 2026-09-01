"""Nạp dữ liệu MIMIC PERform AF theo từng bệnh nhân (record).

Nguồn: data/raw/mimic_perform/
  af/mimic_perform_af_XXX_data.csv       (19 bệnh nhân Rung Nhĩ)
  non-af/mimic_perform_non_af_XXX_data.csv (16 bệnh nhân bình thường)

Mỗi file: cột Time (s), PPG (chuẩn hóa 0-1), ECG, (resp) @ 125 Hz, ~20 phút.
record_id là danh tính bệnh nhân — bắt buộc giữ xuyên suốt pipeline để
chia train/test theo bệnh nhân (chống subject leakage).
"""

import os
import re
import glob

import numpy as np
import pandas as pd

from . import config


def list_records():
    """Liệt kê toàn bộ record: [(record_id, label, csv_path)].

    label: 1 = AFib, 0 = Normal.
    """
    records = []
    patterns = [
        (os.path.join(config.RAW_DIR, 'af', 'mimic_perform_af_*_data.csv'), 1),
        (os.path.join(config.RAW_DIR, 'non-af', 'mimic_perform_non_af_*_data.csv'), 0),
    ]
    for pattern, label in patterns:
        for path in sorted(glob.glob(pattern)):
            name = os.path.basename(path)
            m = re.match(r'(mimic_perform_(?:non_)?af_\d+)_data\.csv', name)
            record_id = m.group(1) if m else os.path.splitext(name)[0]
            records.append((record_id, label, path))

    if not records:
        raise FileNotFoundError(
            f'Không tìm thấy dữ liệu raw tại {config.RAW_DIR}.\n'
            f'Tải về bằng: python -c "import sys; sys.path.insert(0, \'src\'); '
            f'from healthsense_ml.data_loading import download_dataset; download_dataset()"'
        )
    return records


def load_ppg(csv_path):
    """Đọc 1 file record, trả về (t, ppg) dạng numpy array."""
    df = pd.read_csv(csv_path, usecols=['Time', 'PPG'])
    t = df['Time'].to_numpy(dtype=np.float64)
    ppg = df['PPG'].to_numpy(dtype=np.float64)
    # Vá NaN hiếm gặp bằng nội suy tuyến tính
    if np.isnan(ppg).any():
        idx = np.arange(len(ppg))
        good = ~np.isnan(ppg)
        ppg = np.interp(idx, idx[good], ppg[good])
    return t, ppg


def download_dataset():
    """Tải dataset MIMIC PERform AF từ Kaggle về data/raw/mimic_perform.

    Dùng kagglehub — dataset public, không cần API token.
    """
    import shutil
    import kagglehub

    cache_path = kagglehub.dataset_download(
        'raditya0/mimic-perform-iii-af-and-non-af-dataset')

    os.makedirs(config.RAW_DIR, exist_ok=True)
    for item in os.listdir(cache_path):
        src = os.path.join(cache_path, item)
        dst = os.path.join(config.RAW_DIR, item)
        if not os.path.exists(dst):
            shutil.move(src, dst)
    print(f'Dataset sẵn sàng tại: {config.RAW_DIR}')
    return config.RAW_DIR
