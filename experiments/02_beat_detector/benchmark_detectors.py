import os
import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import neurokit2 as nk

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
OUTDIR = "experiments/02_beat_detector"


def fill_nan(x):
    x = np.asarray(x, dtype=np.float64)

    if not np.isnan(x).any():
        return x

    idx = np.arange(len(x))
    good = ~np.isnan(x)

    if good.sum() < 10:
        return np.zeros_like(x)

    return np.interp(idx, idx[good], x[good])


def detector_current(ppg):
    """Current HealthSense v4 detector."""
    filtered = bandpass_filter(ppg, FS)
    return detect_beats(filtered, FS)


def detector_elgendi(ppg):
    """NeuroKit2 implementation of the Elgendi PPG detector."""
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

    peaks = np.asarray(info["PPG_Peaks"], dtype=np.float64)
    return peaks / FS


DETECTORS = {
    "current_scipy": detector_current,
    "neurokit_elgendi": detector_elgendi,
}


def evaluate_record(record_id, label, path):
    df = pd.read_csv(path)

    t = df["Time"].to_numpy(dtype=np.float64)
    ppg = fill_nan(df["PPG"].to_numpy(dtype=np.float64))
    ecg = df["ECG"].to_numpy(dtype=np.float64)

    ecg_t = detect_r_peaks(ecg)

    duration_min = (t[-1] - t[0]) / 60.0
    hr_ecg = len(ecg_t) / duration_min if duration_min > 0 else 0

    ecg_ok = (
        len(ecg_t) >= 200
        and 30 <= hr_ecg <= 220
    )

    rows = []

    if not ecg_ok:
        return rows

    for method, detector in DETECTORS.items():

        try:
            ppg_t = detector(ppg)
            matched = match_beats(ecg_t, ppg_t)

            if matched is None:
                continue

            row = {
                "record_id": record_id,
                "label": "AF" if label else "Non-AF",
                "method": method,
                "n_ecg": len(ecg_t),
                "n_ppg": len(ppg_t),
                "hr_ecg": hr_ecg,
                **matched,
            }

            row["hr_mae_bpm"] = window_hr_mae(
                ecg_t,
                ppg_t,
                t[-1],
            )

            rows.append(row)

        except Exception as exc:
            print(
                f"ERROR {record_id} {method}: "
                f"{type(exc).__name__}: {exc}"
            )

    return rows


def main():

    os.makedirs(OUTDIR, exist_ok=True)

    rows = []

    records = list_records()

    for i, (record_id, label, path) in enumerate(records, 1):

        print(
            f"[{i:02d}/{len(records)}] "
            f"{record_id}"
        )

        rows.extend(
            evaluate_record(
                record_id,
                label,
                path,
            )
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        f"{OUTDIR}/beat_detector_results.csv",
        index=False,
    )

    summary = (
        result
        .groupby(["method", "label"])
        .agg(
            n_subjects=("record_id", "nunique"),
            median_f1=("f1", "median"),
            median_sensitivity=("sensitivity", "median"),
            median_ppv=("ppv", "median"),
            median_hr_mae=("hr_mae_bpm", "median"),
            median_ptt_ms=("ptt_ms", "median"),
        )
        .reset_index()
    )

    summary.to_csv(
        f"{OUTDIR}/beat_detector_summary.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))

    print()
    print("=" * 80)
    print("AF vs NON-AF F1 GAP")
    print("=" * 80)

    pivot = summary.pivot(
        index="method",
        columns="label",
        values="median_f1",
    )

    if "AF" in pivot.columns and "Non-AF" in pivot.columns:
        pivot["gap"] = (
            pivot["Non-AF"] -
            pivot["AF"]
        ).abs()

    print(pivot.to_string())

    print()
    print("=" * 80)
    print("10 WORST RECORD/METHOD COMBINATIONS")
    print("=" * 80)

    worst = (
        result
        .sort_values("f1")
        [[
            "record_id",
            "label",
            "method",
            "f1",
            "sensitivity",
            "ppv",
            "hr_mae_bpm",
            "ptt_ms",
        ]]
        .head(10)
    )

    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
