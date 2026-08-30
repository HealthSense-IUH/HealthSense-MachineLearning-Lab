"""Trích xuất bảng đặc trưng bằng cửa sổ trượt — CÓ record_id.

Khác biệt then chốt so với pipeline cũ: mỗi hàng đặc trưng mang theo
record_id (danh tính bệnh nhân). Không có cột này thì không thể chia
train/test theo bệnh nhân, và mọi kết quả đều có nguy cơ subject leakage.
"""

import numpy as np
import pandas as pd

from . import config
from .data_loading import list_records, load_ppg
from .signal_processing import extract_nn_series


def extract_record_features(record_id, label, csv_path):
    """Trích đặc trưng cửa sổ trượt cho 1 bệnh nhân."""
    from .hrv_features import compute_hrv_features

    t, ppg = load_ppg(csv_path)
    nn_ms, nn_times = extract_nn_series(ppg)
    if len(nn_ms) == 0:
        return []

    rows = []
    t_end = t[-1]
    start = 0.0
    while start + config.WINDOW_S <= t_end:
        stop = start + config.WINDOW_S
        mask = (nn_times >= start) & (nn_times < stop)
        nn_win = nn_ms[mask]
        nn_t_win = nn_times[mask]

        if len(nn_win) >= config.MIN_BEATS_PER_WINDOW:
            feats = compute_hrv_features(nn_win, nn_t_win)
            feats['record_id'] = record_id
            feats['t_start'] = start
            feats['status'] = label
            rows.append(feats)

        start += config.STEP_S
    return rows


def extract_all(verbose=True):
    """Trích đặc trưng toàn bộ dataset -> DataFrame và lưu FEATURES_V4_FILE."""
    import os

    all_rows = []
    records = list_records()
    for i, (record_id, label, path) in enumerate(records, 1):
        rows = extract_record_features(record_id, label, path)
        all_rows.extend(rows)
        if verbose:
            tag = 'AFib' if label == 1 else 'Normal'
            print(f'  [{i:>2}/{len(records)}] {record_id} ({tag}): {len(rows)} cửa sổ')

    df = pd.DataFrame(all_rows)
    cols = ['record_id', 't_start', 'status'] + config.ALL_FEATURES
    df = df[cols]

    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    df.to_csv(config.FEATURES_V4_FILE, index=False)
    if verbose:
        n_af = (df['status'] == 1).sum()
        print(f'\nTổng: {len(df)} cửa sổ ({n_af} AFib / {len(df) - n_af} Normal) '
              f'từ {df["record_id"].nunique()} bệnh nhân')
        print(f'Đã lưu: {config.FEATURES_V4_FILE}')
    return df


def load_features():
    """Nạp bảng đặc trưng v4 (trích xuất nếu chưa có)."""
    import os
    if not os.path.exists(config.FEATURES_V4_FILE):
        return extract_all()
    return pd.read_csv(config.FEATURES_V4_FILE)
