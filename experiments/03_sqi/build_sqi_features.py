import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal

from healthsense_ml import config
from healthsense_ml.data_loading import list_records
from healthsense_ml.signal_processing import (
    bandpass_filter,
    detect_beats,
)

FS = config.FS
WINDOW = 30.0
STEP = 10.0

quality = pd.read_csv(
    "experiments/03_sqi/window_quality.csv"
)

# NaN beat-F1 = zero successful matches -> quality failure
quality["beat_f1"] = quality["beat_f1"].fillna(0.0)


def fill_nan(x):
    x = np.asarray(x, dtype=float)

    if not np.isnan(x).any():
        return x

    idx = np.arange(len(x))
    good = ~np.isnan(x)

    return np.interp(idx, idx[good], x[good])


def bandpower(x, lo, hi):
    if len(x) < FS * 5:
        return np.nan

    f, p = signal.welch(
        x,
        fs=FS,
        nperseg=min(len(x), 512),
    )

    mask = (f >= lo) & (f < hi)

    if not mask.any():
        return 0.0

    return float(np.trapezoid(p[mask], f[mask]))


def template_consistency(filtered, peaks):
    """
    Fixed short region around pulse peak.
    Avoids using inter-beat interval variability.
    """

    before = int(0.12 * FS)
    after = int(0.28 * FS)

    beats = []

    for peak in peaks:
        if peak - before < 0:
            continue

        if peak + after >= len(filtered):
            continue

        beat = filtered[
            peak - before:
            peak + after + 1
        ].copy()

        sd = np.std(beat)

        if sd == 0:
            continue

        beat = (
            beat - np.mean(beat)
        ) / sd

        beats.append(beat)

    if len(beats) < 5:
        return np.nan

    beats = np.asarray(beats)

    template = np.median(
        beats,
        axis=0,
    )

    corrs = []

    for beat in beats:
        c = np.corrcoef(
            beat,
            template,
        )[0, 1]

        if np.isfinite(c):
            corrs.append(c)

    return (
        float(np.median(corrs))
        if corrs
        else np.nan
    )


def detector_agreement(x):
    """
    Count agreement only.
    Do not use interval irregularity.
    """

    filtered = bandpass_filter(x, FS)

    current_t = detect_beats(
        filtered,
        FS,
    )

    try:
        cleaned = nk.ppg_clean(
            x,
            sampling_rate=FS,
            method="elgendi",
        )

        _, info = nk.ppg_peaks(
            cleaned,
            sampling_rate=FS,
            method="elgendi",
            correct_artifacts=False,
        )

        elgendi_n = len(
            info["PPG_Peaks"]
        )

    except Exception:
        elgendi_n = np.nan

    current_n = len(current_t)

    if (
        current_n == 0 or
        not np.isfinite(elgendi_n)
    ):
        agreement = np.nan
    else:
        agreement = (
            1.0 -
            abs(current_n - elgendi_n) /
            max(current_n, elgendi_n)
        )

    return current_t, agreement


rows = []

for num, (rid, label, path) in enumerate(
    list_records(),
    1,
):
    print(f"[{num:02d}/35] {rid}")

    df = pd.read_csv(path)

    t = df["Time"].to_numpy(float)
    raw = fill_nan(
        df["PPG"].to_numpy(float)
    )

    start = 0.0

    while start + WINDOW <= t[-1]:

        stop = start + WINDOW

        mask = (
            (t >= start) &
            (t < stop)
        )

        x = raw[mask]

        if len(x) < FS * WINDOW * 0.95:
            start += STEP
            continue

        filtered = bandpass_filter(x, FS)

        current_t, agreement = detector_agreement(x)

        peak_samples = np.asarray(
            np.round(current_t * FS),
            dtype=int,
        )

        z = (
            filtered - np.mean(filtered)
        ) / (
            np.std(filtered) + 1e-12
        )

        _, props = signal.find_peaks(
            z,
            distance=int(
                config.MIN_BEAT_DISTANCE_S * FS
            ),
            prominence=0,
        )

        prominences = props.get(
            "prominences",
            np.array([]),
        )

        signal_power = bandpower(
            x, 0.5, 8.0
        )

        hf_noise = bandpower(
            x, 8.0, 20.0
        )

        baseline = bandpower(
            x, 0.05, 0.5
        )

        total = (
            signal_power +
            hf_noise +
            baseline +
            1e-12
        )

        dx = np.diff(x)

        x_min = np.min(x)
        x_max = np.max(x)

        eps = max(
            (x_max - x_min) * 0.005,
            1e-12,
        )

        clipping = np.mean(
            (x <= x_min + eps) |
            (x >= x_max - eps)
        )

        rows.append({
            "record_id": rid,
            "label": (
                "AF"
                if label
                else "Non-AF"
            ),
            "t_start": start,

            "signal_std": np.std(x),
            "signal_iqr":
                np.percentile(x, 75) -
                np.percentile(x, 25),

            "derivative_mad":
                np.median(
                    np.abs(
                        dx - np.median(dx)
                    )
                ),

            "hf_noise_ratio":
                hf_noise / total,

            "baseline_ratio":
                baseline / total,

            "clipping_fraction":
                clipping,

            "n_peaks":
                len(current_t),

            "median_prominence":
                (
                    np.median(prominences)
                    if len(prominences)
                    else np.nan
                ),

            "prominence_iqr":
                (
                    np.percentile(
                        prominences, 75
                    ) -
                    np.percentile(
                        prominences, 25
                    )
                    if len(prominences)
                    else np.nan
                ),

            "template_corr":
                template_consistency(
                    filtered,
                    peak_samples,
                ),

            "detector_count_agreement":
                agreement,
        })

        start += STEP


features = pd.DataFrame(rows)

out = features.merge(
    quality[
        [
            "record_id",
            "t_start",
            "beat_f1",
        ]
    ],
    on=[
        "record_id",
        "t_start",
    ],
    how="inner",
)

out.to_csv(
    "experiments/03_sqi/"
    "sqi_features.csv",
    index=False,
)

print()
print("Windows:", len(out))
print("Subjects:", out.record_id.nunique())
print(out.isna().sum())
