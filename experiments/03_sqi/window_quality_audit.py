import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from healthsense_ml import config
from healthsense_ml.data_loading import list_records
from healthsense_ml.signal_processing import bandpass_filter, detect_beats
from healthsense_ml.beat_validation import detect_r_peaks, match_beats

WINDOW = 30.0
STEP = 10.0


def fill_nan(x):
    x = np.asarray(x, dtype=float)

    if not np.isnan(x).any():
        return x

    idx = np.arange(len(x))
    good = ~np.isnan(x)
    return np.interp(idx, idx[good], x[good])


def estimate_global_lag(ecg_t, ppg_t):
    idx = np.searchsorted(ecg_t, ppg_t) - 1
    valid = idx >= 0

    d = ppg_t[valid] - ecg_t[idx[valid]]
    d = d[(d > 0) & (d < 1.0)]

    if len(d) < 10:
        return None

    return float(np.median(d))


def match_aligned(ecg_t, ppg_t, lag, tol=0.150):
    e = np.asarray(ecg_t)
    p = np.asarray(ppg_t) - lag

    i = j = matched = 0

    while i < len(e) and j < len(p):
        diff = p[j] - e[i]

        if abs(diff) <= tol:
            matched += 1
            i += 1
            j += 1
        elif diff < -tol:
            j += 1
        else:
            i += 1

    sens = matched / len(e) if len(e) else np.nan
    ppv = matched / len(p) if len(p) else np.nan

    if np.isnan(sens) or np.isnan(ppv) or sens + ppv == 0:
        f1 = np.nan
    else:
        f1 = 2 * sens * ppv / (sens + ppv)

    return matched, sens, ppv, f1


rows = []

for num, (rid, label, path) in enumerate(list_records(), 1):

    print(f"[{num:02d}/35] {rid}")

    df = pd.read_csv(path)

    t = df["Time"].to_numpy(float)
    ppg = fill_nan(df["PPG"].to_numpy(float))
    ecg = fill_nan(df["ECG"].to_numpy(float))

    filtered = bandpass_filter(ppg)
    ppg_t = detect_beats(filtered)
    ecg_t = detect_r_peaks(ecg)

    lag = estimate_global_lag(ecg_t, ppg_t)

    if lag is None:
        continue

    start = 0.0

    while start + WINDOW <= t[-1]:

        stop = start + WINDOW

        e = ecg_t[
            (ecg_t >= start) &
            (ecg_t < stop)
        ]

        p = ppg_t[
            (ppg_t >= start) &
            (ppg_t < stop)
        ]

        if len(e) >= 10 and len(p) >= 10:

            matched, sens, ppv, f1 = match_aligned(
                e, p, lag
            )

            raw_ppi = np.diff(p) * 1000.0

            invalid_ppi = np.mean(
                (raw_ppi < config.NN_MIN_MS) |
                (raw_ppi > config.NN_MAX_MS)
            ) if len(raw_ppi) else np.nan

            rows.append({
                "record_id": rid,
                "label": "AF" if label else "Non-AF",
                "t_start": start,

                "beat_f1": f1,
                "sensitivity": sens,
                "ppv": ppv,

                "n_ecg": len(e),
                "n_ppg": len(p),

                "beat_count_error": abs(len(e) - len(p)),
                "invalid_ppi_fraction": invalid_ppi,
            })

        start += STEP


out = pd.DataFrame(rows)

out.to_csv(
    "experiments/03_sqi/window_quality.csv",
    index=False,
)


print()
print("=" * 90)
print("WINDOW QUALITY")
print("=" * 90)

for label in ["AF", "Non-AF"]:

    x = out[out.label == label]

    print()
    print(label)
    print("windows       :", len(x))
    print("median F1     :", round(x.beat_f1.median(), 4))

    for threshold in [0.70, 0.80, 0.90, 0.95]:
        coverage = (x.beat_f1 >= threshold).mean()

        print(
            f"F1 >= {threshold:.2f}: "
            f"{coverage:.3%} coverage"
        )


print()
print("=" * 90)
print("ALL")
print("=" * 90)

for threshold in [0.70, 0.80, 0.90, 0.95]:

    keep = out.beat_f1 >= threshold

    print(
        f"threshold={threshold:.2f} | "
        f"coverage={keep.mean():.3%} | "
        f"kept={keep.sum()}/{len(out)}"
    )


print()
print("=" * 90)
print("WORST WINDOWS")
print("=" * 90)

print(
    out.sort_values("beat_f1")[
        [
            "record_id",
            "label",
            "t_start",
            "beat_f1",
            "sensitivity",
            "ppv",
            "n_ecg",
            "n_ppg",
            "invalid_ppi_fraction",
        ]
    ].head(30).to_string(index=False)
)
