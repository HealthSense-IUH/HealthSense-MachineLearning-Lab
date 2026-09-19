import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import neurokit2 as nk

from healthsense_ml import config
from healthsense_ml.data_loading import list_records
from healthsense_ml.signal_processing import bandpass_filter, detect_beats
from healthsense_ml.beat_validation import (
    detect_r_peaks,
    match_beats,
    window_hr_mae,
)

FS = config.FS
SEGMENT_S = 300.0


def fill_nan(x):
    x = np.asarray(x, dtype=float)
    if not np.isnan(x).any():
        return x

    idx = np.arange(len(x))
    good = ~np.isnan(x)

    if good.sum() < 10:
        return np.zeros_like(x)

    return np.interp(idx, idx[good], x[good])


def current_detector(ppg):
    filt = bandpass_filter(ppg, FS)
    return detect_beats(filt, FS)


def elgendi_detector(ppg):
    cleaned = nk.ppg_clean(
        ppg,
        sampling_rate=FS,
        method="elgendi",
    )

    _, info = nk.ppg_peaks(
        cleaned,
        sampling_rate=FS,
        method="elgendi",
        correct_artifacts=False,
    )

    return np.asarray(info["PPG_Peaks"]) / FS


DETECTORS = {
    "current_scipy": current_detector,
    "neurokit_elgendi": elgendi_detector,
}


def local_match(ecg_t, ppg_t, t_end):
    n_ecg_total = 0
    n_ppg_total = 0
    n_match_total = 0
    lags = []

    start = 0.0

    while start < t_end:
        stop = min(start + SEGMENT_S, t_end)

        e = ecg_t[(ecg_t >= start) & (ecg_t < stop)]
        p = ppg_t[(ppg_t >= start) & (ppg_t < stop)]

        if len(e) >= 10 and len(p) >= 10:
            result = match_beats(e, p)

            if result is not None:
                n_ecg_total += result["n_ecg_beats"]
                n_ppg_total += result["n_ppg_beats"]
                n_match_total += result["n_matched"]
                lags.append(result["ptt_ms"])

        start = stop

    if n_ecg_total == 0 or n_ppg_total == 0:
        return None

    sensitivity = n_match_total / n_ecg_total
    ppv = n_match_total / n_ppg_total

    f1 = (
        2 * sensitivity * ppv / (sensitivity + ppv)
        if sensitivity + ppv > 0
        else 0.0
    )

    return {
        "sensitivity": sensitivity,
        "ppv": ppv,
        "f1": f1,
        "n_ecg": n_ecg_total,
        "n_ppg": n_ppg_total,
        "n_match": n_match_total,
        "median_alignment_lag_ms": (
            float(np.median(lags)) if lags else np.nan
        ),
        "lag_iqr_ms": (
            float(np.percentile(lags, 75) - np.percentile(lags, 25))
            if len(lags) >= 2
            else np.nan
        ),
    }


rows = []

for i, (rid, label, path) in enumerate(list_records(), 1):
    print(f"[{i:02d}/35] {rid}")

    df = pd.read_csv(path)

    t = df["Time"].to_numpy(float)
    ppg = fill_nan(df["PPG"].to_numpy(float))
    ecg = df["ECG"].to_numpy(float)

    ecg_t = detect_r_peaks(ecg)

    if len(ecg_t) < 200:
        continue

    for method, detector in DETECTORS.items():
        ppg_t = detector(ppg)

        global_result = match_beats(ecg_t, ppg_t)
        local_result = local_match(ecg_t, ppg_t, t[-1])

        if global_result is None or local_result is None:
            continue

        rows.append({
            "record_id": rid,
            "label": "AF" if label else "Non-AF",
            "method": method,

            "global_f1": global_result["f1"],
            "global_sensitivity": global_result["sensitivity"],
            "global_ppv": global_result["ppv"],
            "global_lag_ms": global_result["ptt_ms"],

            "local_f1": local_result["f1"],
            "local_sensitivity": local_result["sensitivity"],
            "local_ppv": local_result["ppv"],
            "local_lag_ms": local_result["median_alignment_lag_ms"],
            "lag_iqr_ms": local_result["lag_iqr_ms"],

            "hr_mae_bpm": window_hr_mae(
                ecg_t,
                ppg_t,
                t[-1],
            ),
        })


out = pd.DataFrame(rows)

out.to_csv(
    "experiments/02_beat_detector/local_alignment_results.csv",
    index=False,
)

summary = (
    out.groupby(["method", "label"])
    .agg(
        n=("record_id", "nunique"),
        global_f1=("global_f1", "median"),
        local_f1=("local_f1", "median"),
        local_sensitivity=("local_sensitivity", "median"),
        local_ppv=("local_ppv", "median"),
        lag_iqr_ms=("lag_iqr_ms", "median"),
    )
    .reset_index()
)

summary["f1_delta"] = summary["local_f1"] - summary["global_f1"]

summary.to_csv(
    "experiments/02_beat_detector/local_alignment_summary.csv",
    index=False,
)

print()
print(summary.to_string(index=False))

print()
print("Largest improvements:")

print(
    out.assign(
        delta=out.local_f1 - out.global_f1
    )
    .sort_values("delta", ascending=False)
    [[
        "record_id",
        "label",
        "method",
        "global_f1",
        "local_f1",
        "delta",
        "global_lag_ms",
        "local_lag_ms",
        "lag_iqr_ms",
    ]]
    .head(20)
    .to_string(index=False)
)
