import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from healthsense_ml import config
from healthsense_ml.data_loading import list_records, load_ppg
from healthsense_ml.signal_processing import extract_nn_series
from healthsense_ml.hrv_features import compute_hrv_features


def extra_features(nn):

    nn = np.asarray(nn, dtype=float)
    diff = np.diff(nn)

    mean_nn = np.mean(nn)

    pnn40 = (
        np.mean(np.abs(diff) > 40) * 100
        if len(diff)
        else 0.0
    )

    pnn70 = (
        np.mean(np.abs(diff) > 70) * 100
        if len(diff)
        else 0.0
    )

    # Turning Point Ratio
    if len(nn) >= 3:
        turning = (
            ((nn[1:-1] > nn[:-2]) &
             (nn[1:-1] > nn[2:]))
            |
            ((nn[1:-1] < nn[:-2]) &
             (nn[1:-1] < nn[2:]))
        )

        tpr = np.mean(turning)
    else:
        tpr = 0.0

    median_nn = np.median(nn)

    mad_nn = np.median(
        np.abs(nn - median_nn)
    )

    rmssd = (
        np.sqrt(np.mean(diff ** 2))
        if len(diff)
        else 0.0
    )

    nrmssd = (
        rmssd / mean_nn
        if mean_nn > 0
        else 0.0
    )

    if len(diff) > 1:

        sd_diff = np.std(
            diff,
            ddof=1,
        )

        sdnn = np.std(
            nn,
            ddof=1,
        )

        sd1 = np.sqrt(
            0.5
        ) * sd_diff

        sd2_sq = (
            2 * sdnn ** 2
            - 0.5 * sd_diff ** 2
        )

        sd2 = (
            np.sqrt(sd2_sq)
            if sd2_sq > 0
            else 0.0
        )

        sd1_sd2 = (
            sd1 / sd2
            if sd2 > 0
            else 0.0
        )

    else:
        sd1_sd2 = 0.0

    return {
        "pNN40": pnn40,
        "pNN70": pnn70,
        "TPR": tpr,
        "MAD_NN": mad_nn,
        "nRMSSD": nrmssd,
        "SD1_SD2_ratio": sd1_sd2,
    }


rows = []

records = list_records()

for i, (rid, label, path) in enumerate(
    records,
    1,
):

    print(
        f"[{i:02d}/{len(records)}] {rid}"
    )

    t, ppg = load_ppg(path)

    nn_ms, nn_times = extract_nn_series(
        ppg
    )

    start = 0.0

    while (
        start + config.WINDOW_S
        <= t[-1]
    ):

        stop = (
            start
            + config.WINDOW_S
        )

        mask = (
            (nn_times >= start)
            &
            (nn_times < stop)
        )

        nn_win = nn_ms[mask]
        time_win = nn_times[mask]

        if (
            len(nn_win)
            >= config.MIN_BEATS_PER_WINDOW
        ):

            feats = compute_hrv_features(
                nn_win,
                time_win,
            )

            feats.update(
                extra_features(
                    nn_win
                )
            )

            feats["record_id"] = rid
            feats["t_start"] = start
            feats["status"] = label

            rows.append(feats)

        start += config.STEP_S


df = pd.DataFrame(rows)

df.to_csv(
    "experiments/04_feature_model/"
    "af_specific_features.csv",
    index=False,
)

print()
print("Windows :", len(df))
print("Subjects:", df.record_id.nunique())

print()
print(
    df[
        [
            "pNN40",
            "pNN70",
            "TPR",
            "MAD_NN",
            "nRMSSD",
            "SD1_SD2_ratio",
        ]
    ].describe().round(4)
)
