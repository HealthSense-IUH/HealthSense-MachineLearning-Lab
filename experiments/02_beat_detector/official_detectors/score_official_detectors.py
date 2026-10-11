import sys
sys.path.insert(0, "src")

from pathlib import Path

import numpy as np
import pandas as pd

from healthsense_ml import config
from healthsense_ml.data_loading import list_records
from healthsense_ml.signal_processing import (
    bandpass_filter,
    detect_beats,
)
from healthsense_ml.beat_validation import (
    detect_r_peaks,
    match_beats,
    window_hr_mae,
)

FS = config.FS

OUT = Path(
    "experiments/02_beat_detector/octave_output"
)


def fill_nan(x):
    x = np.asarray(x, dtype=float)

    if not np.isnan(x).any():
        return x

    idx = np.arange(len(x))
    good = ~np.isnan(x)

    return np.interp(
        idx,
        idx[good],
        x[good],
    )


def load_octave_peaks(path):
    peaks = np.loadtxt(
        path,
        delimiter=",",
        ndmin=1,
    )

    # MATLAB/Octave indices are 1-based.
    return (peaks - 1.0) / FS


def current_detector(ppg):
    filt = bandpass_filter(ppg, FS)
    return detect_beats(filt, FS)


def estimate_lag(ecg_t, ppg_t):
    idx = np.searchsorted(ecg_t, ppg_t) - 1
    valid = idx >= 0

    d = (
        ppg_t[valid]
        - ecg_t[idx[valid]]
    )

    d = d[(d > 0) & (d < 1.0)]

    if len(d) < 10:
        return None

    return float(np.median(d))


def matched_intervals(ecg_t, ppg_t):
    """
    Match using the same global-lag idea as the existing scorer,
    then compare intervals between successive matched beats.
    """

    lag = estimate_lag(ecg_t, ppg_t)

    if lag is None:
        return np.nan

    p = np.asarray(ppg_t) - lag
    e = np.asarray(ecg_t)

    tol = 0.150

    i = 0
    j = 0

    matched_e = []
    matched_p = []

    while i < len(e) and j < len(p):

        diff = p[j] - e[i]

        if abs(diff) <= tol:
            matched_e.append(e[i])
            matched_p.append(p[j])

            i += 1
            j += 1

        elif diff < -tol:
            j += 1

        else:
            i += 1

    if len(matched_e) < 10:
        return np.nan

    eppi = np.diff(matched_e) * 1000.0
    pppi = np.diff(matched_p) * 1000.0

    return float(
        np.mean(
            np.abs(eppi - pppi)
        )
    )


rows = []

for i, (rid, label, path) in enumerate(
    list_records(),
    1,
):

    print(f"[{i:02d}/35] {rid}")

    df = pd.read_csv(path)

    t = df["Time"].to_numpy(float)
    ppg = fill_nan(
        df["PPG"].to_numpy(float)
    )

    ecg = fill_nan(
        df["ECG"].to_numpy(float)
    )

    ecg_t = detect_r_peaks(ecg)

    detectors = {
        "current_scipy":
            current_detector(ppg),

        "msptdfastv2":
            load_octave_peaks(
                OUT / f"{rid}_msptdfastv2.csv"
            ),

        "qppgfast":
            load_octave_peaks(
                OUT / f"{rid}_qppgfast.csv"
            ),
    }

    for method, ppg_t in detectors.items():

        m = match_beats(ecg_t, ppg_t)

        if m is None:
            continue

        rows.append({
            "record_id": rid,
            "label": (
                "AF"
                if label
                else "Non-AF"
            ),
            "method": method,

            "f1": m["f1"],
            "sensitivity": m["sensitivity"],
            "ppv": m["ppv"],
            "alignment_lag_ms": m["ptt_ms"],

            "hr_mae_bpm":
                window_hr_mae(
                    ecg_t,
                    ppg_t,
                    t[-1],
                ),

            "ppi_mae_ms":
                matched_intervals(
                    ecg_t,
                    ppg_t,
                ),

            "n_ecg": len(ecg_t),
            "n_ppg": len(ppg_t),
        })


result = pd.DataFrame(rows)

result.to_csv(
    "experiments/02_beat_detector/"
    "official_detector_results.csv",
    index=False,
)


summary = (
    result
    .groupby(["method", "label"])
    .agg(
        n_subjects=("record_id", "nunique"),

        median_f1=("f1", "median"),
        min_f1=("f1", "min"),

        median_sensitivity=(
            "sensitivity",
            "median",
        ),

        median_ppv=("ppv", "median"),

        median_hr_mae=(
            "hr_mae_bpm",
            "median",
        ),

        median_ppi_mae=(
            "ppi_mae_ms",
            "median",
        ),
    )
    .reset_index()
)


summary.to_csv(
    "experiments/02_beat_detector/"
    "official_detector_summary.csv",
    index=False,
)

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(summary.to_string(index=False))


print()
print("=" * 100)
print("AF / NON-AF GAP")
print("=" * 100)

pivot = summary.pivot(
    index="method",
    columns="label",
    values="median_f1",
)

pivot["gap"] = (
    pivot["Non-AF"]
    - pivot["AF"]
).abs()

print(pivot.to_string())


print()
print("=" * 100)
print("WORST CASES")
print("=" * 100)

print(
    result
    .sort_values("f1")
    [[
        "record_id",
        "label",
        "method",
        "f1",
        "sensitivity",
        "ppv",
        "ppi_mae_ms",
        "hr_mae_bpm",
        "n_ecg",
        "n_ppg",
    ]]
    .head(30)
    .to_string(index=False)
)
