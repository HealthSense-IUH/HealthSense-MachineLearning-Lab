"""Phân tích bổ sung: kiểm chứng ngưỡng spectral concentration của cổng SQI.

AI Service (HealthSense-AI-AFib-Prediction-Service, app/services/hrv_v4.py) từ
chối bản ghi nếu tỉ lệ năng lượng dải 0.5-3.5 Hz trên toàn dải đã lọc 0.5-8 Hz
nhỏ hơn 0.70. Ghi chú trong code nêu "35 bệnh nhân MIMIC: min 0.86; nhiễu trắng
~0.41" nhưng chưa có script tái lập. Script này tái lập phần có thể làm từ dữ
liệu trong repo:

- 35 bản ghi MIMIC PERform AF (19 AF, 16 non-AF; PPG đầu ngón tay, 125 Hz, 20 phút)
  trong data/raw/mimic_perform/: tỉ lệ trên toàn bản ghi và trên các đoạn không
  chồng lấn dài 30 s và 60 s (60 s = một lần đo trên vòng đeo);
- nhiễu trắng Gauss mô phỏng (1000 đoạn 60 s, 125 Hz, seed cố định).

Dữ liệu đo bằng thiết bị thật không có trong repo nên KHÔNG được kiểm ở đây.
Hàm lọc dùng healthsense_core.signal_processing.bandpass_filter (Butterworth bậc 4,
0.5-8 Hz, SOS, zero-phase); hàm spectral concentration chép nguyên từ AI Service.

Chạy từ gốc repo:  PYTHONPATH=src python notebooks/v6/scripts/supp_sqi_spectral_threshold_check.py
Đầu ra: results/v6/supplementary/sqi_spectral_*.csv và sqi_spectral_summary.json
"""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import signal as sp_signal

from healthsense_core import config
from healthsense_core.signal_processing import bandpass_filter

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw/mimic_perform"
OUT = ROOT / "results/v6/supplementary"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.70
HR_BAND = (0.5, 3.5)  # Hz, như SQI_HR_BAND trong AI Service
FS = 125.0
SEGMENTS_S = (30, 60)
N_NOISE = 1000
SEED = 20261111


def spectral_concentration(ppg_filtered, fs):
    """Chép từ AI Service app/services/hrv_v4.py::_spectral_concentration."""
    freqs, psd = sp_signal.welch(ppg_filtered, fs=fs, nperseg=min(len(ppg_filtered), 1024))
    band = psd[(freqs >= HR_BAND[0]) & (freqs <= HR_BAND[1])].sum()
    total = psd[(freqs >= config.BANDPASS_LOW) & (freqs <= config.BANDPASS_HIGH)].sum()
    return float(band / total) if total > 0 else 0.0


def load_ppg(path):
    ppg = pd.read_csv(path, usecols=["PPG"])["PPG"].to_numpy(dtype=float)
    if np.isnan(ppg).any():  # nội suy NaN như AI Service
        idx = np.arange(len(ppg))
        good = ~np.isnan(ppg)
        ppg = np.interp(idx, idx[good], ppg[good])
    return ppg


record_rows, segment_rows = [], []
for group in ["af", "non-af"]:
    for path in sorted((RAW / group).glob("*_data.csv")):
        rec = path.name.replace("_data.csv", "")
        ppg = load_ppg(path)
        row = {"record": rec, "group": group, "duration_s": len(ppg) / FS,
               "ratio_whole_record": spectral_concentration(bandpass_filter(ppg, FS), FS)}
        for seg_s in SEGMENTS_S:
            n = int(seg_s * FS)
            vals = []
            for start in range(0, len(ppg) - n + 1, n):
                seg = ppg[start:start + n]
                if np.ptp(seg) == 0:
                    continue
                r = spectral_concentration(bandpass_filter(seg, FS), FS)
                vals.append(r)
                segment_rows.append({"record": rec, "group": group, "segment_s": seg_s,
                                     "start_s": start / FS, "ratio": r})
            v = np.array(vals)
            row[f"n_seg_{seg_s}s"] = len(v)
            row[f"min_{seg_s}s"] = float(v.min())
            row[f"p05_{seg_s}s"] = float(np.percentile(v, 5))
            row[f"median_{seg_s}s"] = float(np.median(v))
            row[f"n_below_{THRESHOLD:.2f}_{seg_s}s"] = int((v < THRESHOLD).sum())
        record_rows.append(row)
        print(rec, group, f"whole={row['ratio_whole_record']:.3f}", f"min60={row['min_60s']:.3f}", flush=True)

records = pd.DataFrame(record_rows)
segments = pd.DataFrame(segment_rows)

rng = np.random.default_rng(SEED)
n = int(60 * FS)
noise = np.array([spectral_concentration(bandpass_filter(rng.standard_normal(n), FS), FS)
                  for _ in range(N_NOISE)])


def describe(v):
    v = np.asarray(v, dtype=float)
    return {"n": int(v.size), "min": float(v.min()), "p01": float(np.percentile(v, 1)),
            "p05": float(np.percentile(v, 5)), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max()),
            f"fraction_below_{THRESHOLD:.2f}": float((v < THRESHOLD).mean())}


summary = {
    "threshold": THRESHOLD,
    "mimic_perform_records": int(len(records)),
    "whole_record": describe(records["ratio_whole_record"]),
    "segments": {f"{s}s": describe(segments.loc[segments["segment_s"] == s, "ratio"]) for s in SEGMENTS_S},
    "segments_by_group_60s": {g: describe(segments.loc[(segments["segment_s"] == 60) & (segments["group"] == g), "ratio"])
                              for g in ["af", "non-af"]},
    "white_noise_60s": describe(noise),
    "not_covered": "Recordings from the MAX30101/MAX30102 prototypes and hand-motion noise are not in the repository.",
    "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                 "scipy": scipy.__version__},
    "noise_seed": SEED,
}

records.to_csv(OUT / "sqi_spectral_records.csv", index=False)
segments.to_csv(OUT / "sqi_spectral_segments.csv", index=False)
(OUT / "sqi_spectral_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
