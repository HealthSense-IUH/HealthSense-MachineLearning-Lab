"""Đọc tín hiệu thô MIMIC PERform theo kênh.

Khác với healthsense_ml.data_loading (chỉ cần kênh PPG cho pipeline v4),
module này cho phép chọn kênh vì **v3 lịch sử đã trích đặc trưng từ kênh ECG**
chứ không phải PPG — một chi tiết quan trọng khi so sánh các phiên bản
(xem src/v3/README.md).

Mỗi file record: Time (s), PPG (0-1), ECG, resp — 125 Hz, ~20 phút/bệnh nhân.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

FS = 125  # Hz — tần số lấy mẫu của MIMIC PERform

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'mimic_perform')
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'data', 'features')

# Hai thư mục tách bạch, đừng trộn:
#   models/  = thứ NẠP ĐƯỢC vào chương trình (.pkl + thẻ model đi kèm)
#   results/ = thứ ĐEM ĐI BÁO CÁO (số liệu, biểu đồ)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


def list_records():
    """Liệt kê toàn bộ bệnh nhân: [(record_id, label, csv_path)].

    label: 1 = AFib, 0 = Normal. Thứ tự ổn định (sorted) để mọi phiên bản
    nhìn thấy cùng một tập dữ liệu.
    """
    records = []
    patterns = [
        (os.path.join(RAW_DIR, 'af', 'mimic_perform_af_*_data.csv'), 1),
        (os.path.join(RAW_DIR, 'non-af', 'mimic_perform_non_af_*_data.csv'), 0),
    ]
    for pattern, label in patterns:
        for path in sorted(glob.glob(pattern)):
            name = os.path.basename(path)
            m = re.match(r'(mimic_perform_(?:non_)?af_\d+)_data\.csv', name)
            record_id = m.group(1) if m else os.path.splitext(name)[0]
            records.append((record_id, label, path))

    if not records:
        raise FileNotFoundError(
            f'Không tìm thấy dữ liệu thô tại {RAW_DIR}.\n'
            f'Tải về bằng: python -c "import sys; sys.path.insert(0, \'src\'); '
            f'from vlab.raw import download_dataset; download_dataset()"'
        )
    return records


def download_dataset():
    """Tải dataset MIMIC PERform AF từ Kaggle về data/raw/mimic_perform.

    Dùng kagglehub — dataset public, không cần API token.
    """
    import shutil

    import kagglehub

    cache_path = kagglehub.dataset_download(
        'raditya0/mimic-perform-iii-af-and-non-af-dataset')

    os.makedirs(RAW_DIR, exist_ok=True)
    for item in os.listdir(cache_path):
        src = os.path.join(cache_path, item)
        dst = os.path.join(RAW_DIR, item)
        if not os.path.exists(dst):
            shutil.move(src, dst)
    print(f'Dataset sẵn sàng tại: {RAW_DIR}')
    return RAW_DIR


def load_channel(csv_path, channel='PPG'):
    """Đọc 1 record, trả về (t, signal) của kênh yêu cầu.

    channel: 'PPG' (v1, v2, v4) hoặc 'ECG' (v3).
    NaN hiếm gặp được vá bằng nội suy tuyến tính.
    """
    if channel not in ('PPG', 'ECG'):
        raise ValueError(f"channel phải là 'PPG' hoặc 'ECG', nhận: {channel!r}")

    df = pd.read_csv(csv_path, usecols=['Time', channel])
    t = df['Time'].to_numpy(dtype=np.float64)
    sig = df[channel].to_numpy(dtype=np.float64)

    if np.isnan(sig).any():
        idx = np.arange(len(sig))
        good = ~np.isnan(sig)
        if not good.any():
            return t, np.zeros_like(sig)
        sig = np.interp(idx, idx[good], sig[good])
    return t, sig


def features_path(filename):
    """Đường dẫn file đặc trưng trong data/features/ (tạo thư mục nếu thiếu)."""
    os.makedirs(FEATURES_DIR, exist_ok=True)
    return os.path.join(FEATURES_DIR, filename)


def models_path(*parts):
    """Đường dẫn trong models/ (tạo thư mục cha nếu thiếu)."""
    path = os.path.join(MODELS_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
