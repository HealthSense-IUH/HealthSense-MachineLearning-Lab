import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from healthsense_ml import config
from healthsense_ml.data_loading import list_records, load_ppg
from healthsense_ml.signal_processing import (
    extract_nn_series,
    bandpass_filter,
)

FS = config.FS
WINDOW = config.WINDOW_S
STEP = config.STEP_S

BASE_FILE = (
    "experiments/04_feature_model/"
    "af_specific_features.csv"
)


def sampen_reference(y, M, r):
    """
    Python translation of the reference sampen.m logic
    used by comp_cosEn.m.
    """

    y = np.asarray(y, dtype=float)
    n = len(y)

    lastrun = np.zeros(n, dtype=int)
    run = np.zeros(n, dtype=int)

    A = np.zeros(M, dtype=int)
    B = np.zeros(M, dtype=int)

    for i in range(n - 1):

        nj = n - i - 1
        y1 = y[i]

        for jj in range(nj):

            j = i + jj + 1

            if abs(y[j] - y1) < r:

                run[jj] = (
                    lastrun[jj] + 1
                )

                M1 = min(
                    M,
                    run[jj],
                )

                for m in range(M1):

                    A[m] += 1

                    if j < n - 1:
                        B[m] += 1

            else:
                run[jj] = 0

        lastrun[:nj] = run[:nj]

    e = np.full(
        M,
        np.nan,
        dtype=float,
    )

    N = n * (n - 1) / 2.0

    if N > 0 and A[0] > 0:
        e[0] = -np.log(
            A[0] / N
        )

    for m in range(1, M):

        if (
            A[m] > 0
            and B[m - 1] > 0
        ):
            e[m] = -np.log(
                A[m] /
                B[m - 1]
            )

    return e, A, B


def cosen(rr_ms):
    """
    Reference-style COSEn.

    RR is converted to seconds.
    r starts at 0.030 s and rises by 0.001 s
    until A(M) >= 5.
    """

    rr = (
        np.asarray(
            rr_ms,
            dtype=float,
        )
        / 1000.0
    )

    if len(rr) < 5:
        return np.nan, np.nan

    M = 2
    r = 0.030
    dr = 0.001
    min_count = 5

    while r <= 0.500:

        e, A, B = sampen_reference(
            rr,
            M,
            r,
        )

        if (
            A[M - 1] >= min_count
            and np.isfinite(e[M - 1])
        ):

            value = (
                e[M - 1]
                + np.log(2.0 * r)
                - np.log(
                    np.mean(rr)
                )
            )

            return (
                float(value),
                float(r),
            )

        r += dr

    return np.nan, np.nan


def autocorrelation_feature(x):
    """
    Mean absolute normalized autocorrelation
    over non-zero lags.

    FFT implementation keeps this practical for
    30 s * 125 Hz windows.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    x = bandpass_filter(
        x,
        FS,
    )

    x = (
        x - np.mean(x)
    )

    n = len(x)

    if n < 10:
        return np.nan

    nfft = 1 << (
        (2 * n - 1).bit_length()
    )

    spectrum = np.fft.rfft(
        x,
        n=nfft,
    )

    ac = np.fft.irfft(
        spectrum *
        np.conj(spectrum),
        n=nfft,
    )[:n]

    # same 1/N factor for every lag
    ac = ac / n

    if (
        not np.isfinite(ac[0])
        or abs(ac[0]) < 1e-12
    ):
        return np.nan

    ac = ac / ac[0]

    return float(
        np.mean(
            np.abs(ac[1:])
        )
    )


rows = []

records = list_records()

for i, (rid, label, path) in enumerate(
    records,
    1,
):

    print(
        f"[{i:02d}/{len(records)}] "
        f"{rid}"
    )

    t, ppg = load_ppg(path)

    nn_ms, nn_times = (
        extract_nn_series(ppg)
    )

    start = 0.0

    while (
        start + WINDOW
        <= t[-1]
    ):

        stop = start + WINDOW

        nn_mask = (
            (nn_times >= start)
            &
            (nn_times < stop)
        )

        nn_win = nn_ms[nn_mask]

        sample_mask = (
            (t >= start)
            &
            (t < stop)
        )

        ppg_win = ppg[sample_mask]

        if (
            len(nn_win)
            >= config.MIN_BEATS_PER_WINDOW
        ):

            c, r_used = cosen(
                nn_win
            )

            ac = (
                autocorrelation_feature(
                    ppg_win
                )
            )

            rows.append({
                "record_id": rid,
                "t_start": start,
                "COSEn": c,
                "COSEn_r": r_used,
                "PPG_AC": ac,
            })

        start += STEP


extra = pd.DataFrame(rows)

base = pd.read_csv(
    BASE_FILE
)

base["t_start"] = (
    base["t_start"]
    .round(6)
)

extra["t_start"] = (
    extra["t_start"]
    .round(6)
)

out = base.merge(
    extra,
    on=[
        "record_id",
        "t_start",
    ],
    how="left",
    validate="one_to_one",
)

out.to_csv(
    "experiments/04_feature_model/"
    "af_features_cosen_ac.csv",
    index=False,
)


print()
print("=" * 80)
print("COSEn + AC EXTRACTION")
print("=" * 80)

print("Windows :", len(out))
print(
    "Subjects:",
    out.record_id.nunique(),
)

print()
print(
    out[
        [
            "COSEn",
            "COSEn_r",
            "PPG_AC",
        ]
    ].isna().sum()
)

print()
print(
    out.groupby("status")[
        [
            "COSEn",
            "COSEn_r",
            "PPG_AC",
        ]
    ].median().round(5)
)

print()
print(
    out[
        [
            "COSEn",
            "COSEn_r",
            "PPG_AC",
        ]
    ].describe().round(5)
)
