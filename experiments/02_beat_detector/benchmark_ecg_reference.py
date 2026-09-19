import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import neurokit2 as nk
import wfdb
from wfdb import processing

from healthsense_ml import config
from healthsense_ml.data_loading import list_records
from healthsense_ml.beat_validation import detect_r_peaks

FS = config.FS
TOL_S = 0.080

OUT = "experiments/02_beat_detector"


def fill_nan(x):
    x = np.asarray(x, dtype=float)

    if not np.isnan(x).any():
        return x

    idx = np.arange(len(x))
    good = ~np.isnan(x)

    if good.sum() < 10:
        return np.zeros_like(x)

    return np.interp(
        idx,
        idx[good],
        x[good],
    )


def detector_current(ecg):
    return detect_r_peaks(ecg)


def detector_neurokit(ecg):
    cleaned = nk.ecg_clean(
        ecg,
        sampling_rate=FS,
        method="neurokit",
    )

    _, info = nk.ecg_peaks(
        cleaned,
        sampling_rate=FS,
        method="neurokit",
        correct_artifacts=False,
    )

    peaks = np.asarray(
        info["ECG_R_Peaks"],
        dtype=float,
    )

    return peaks / FS


def detector_xqrs(ecg):
    samples = processing.xqrs_detect(
        sig=ecg,
        fs=FS,
        verbose=False,
    )

    return np.asarray(
        samples,
        dtype=float,
    ) / FS


DETECTORS = {
    "current": detector_current,
    "neurokit": detector_neurokit,
    "xqrs": detector_xqrs,
}


def match_same_signal(ref, test):
    """
    Match two ECG R-peak sequences.
    No PTT/alignment is allowed because both detect
    the same electrical R-wave.
    """

    ref = np.asarray(ref)
    test = np.asarray(test)

    i = 0
    j = 0
    matched = 0

    while i < len(ref) and j < len(test):

        diff = test[j] - ref[i]

        if abs(diff) <= TOL_S:
            matched += 1
            i += 1
            j += 1

        elif diff < -TOL_S:
            j += 1

        else:
            i += 1

    sensitivity = (
        matched / len(ref)
        if len(ref)
        else 0.0
    )

    ppv = (
        matched / len(test)
        if len(test)
        else 0.0
    )

    f1 = (
        2 * sensitivity * ppv /
        (sensitivity + ppv)
        if sensitivity + ppv > 0
        else 0.0
    )

    return {
        "matched": matched,
        "ref_beats": len(ref),
        "test_beats": len(test),
        "sensitivity": sensitivity,
        "ppv": ppv,
        "f1": f1,
    }


rows = []

for idx, (rid, label, path) in enumerate(
    list_records(),
    1,
):

    print(f"[{idx:02d}/35] {rid}")

    df = pd.read_csv(path)

    ecg = fill_nan(
        df["ECG"].to_numpy(float)
    )

    detected = {}

    for name, detector in DETECTORS.items():

        try:
            detected[name] = detector(ecg)

        except Exception as exc:
            print(
                f"ERROR {rid} {name}: "
                f"{type(exc).__name__}: {exc}"
            )

    pairs = [
        ("current", "neurokit"),
        ("current", "xqrs"),
        ("neurokit", "xqrs"),
    ]

    for a, b in pairs:

        if (
            a not in detected
            or b not in detected
        ):
            continue

        metrics = match_same_signal(
            detected[a],
            detected[b],
        )

        rows.append({
            "record_id": rid,
            "label": (
                "AF"
                if label
                else "Non-AF"
            ),
            "reference": a,
            "comparison": b,
            **metrics,
        })


result = pd.DataFrame(rows)

result.to_csv(
    f"{OUT}/ecg_reference_results.csv",
    index=False,
)


summary = (
    result
    .groupby([
        "reference",
        "comparison",
        "label",
    ])
    .agg(
        n_subjects=("record_id", "nunique"),
        median_f1=("f1", "median"),
        min_f1=("f1", "min"),
        median_sensitivity=("sensitivity", "median"),
        median_ppv=("ppv", "median"),
    )
    .reset_index()
)

summary.to_csv(
    f"{OUT}/ecg_reference_summary.csv",
    index=False,
)


print()
print("=" * 85)
print("ECG REFERENCE AGREEMENT")
print("=" * 85)
print(summary.to_string(index=False))


print()
print("=" * 85)
print("WORST ECG DISAGREEMENTS")
print("=" * 85)

print(
    result
    .sort_values("f1")
    [[
        "record_id",
        "label",
        "reference",
        "comparison",
        "f1",
        "sensitivity",
        "ppv",
        "ref_beats",
        "test_beats",
    ]]
    .head(20)
    .to_string(index=False)
)
