"""Kiểm thử khói: pipeline HRV của thư viện chạy trọn trên PPG tổng hợp 60 nhịp/phút."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from healthsense_ml import config  # noqa: E402
from healthsense_ml.hrv_features import compute_hrv_features  # noqa: E402
from healthsense_ml.signal_processing import extract_nn_series  # noqa: E402


def synthetic_ppg(seconds=30, fs=config.FS, bpm=60.0):
    t = np.arange(0, seconds, 1.0 / fs)
    beat = np.sin(2 * np.pi * (bpm / 60.0) * t)
    return 1000.0 + 200.0 * beat + 20.0 * np.sin(2 * np.pi * 0.25 * t)


def test_pipeline_runs_on_synthetic_signal():
    ppg = synthetic_ppg()
    nn, nn_times = extract_nn_series(ppg, config.FS)
    assert 25 <= len(nn) <= 35, f"60 bpm trong 30 s phải cho khoảng 30 nhịp, nhận {len(nn)}"
    feats = compute_hrv_features(nn, nn_times)
    assert set(config.ALL_FEATURES) <= set(feats), "thiếu đặc trưng so với config.ALL_FEATURES"
    assert all(np.isfinite(feats[f]) for f in config.ALL_FEATURES)
    assert 55 <= feats["HR_mean"] <= 65
