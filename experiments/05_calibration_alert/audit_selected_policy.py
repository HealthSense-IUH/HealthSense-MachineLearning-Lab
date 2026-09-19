import numpy as np
import pandas as pd


SOURCE = (
    "experiments/04_feature_model/"
    "af_features_cosen_ac.csv"
)

PRED = (
    "experiments/05_calibration_alert/"
    "calibration_loso_predictions.csv"
)

THRESHOLD = 0.90
K = 3
N = 3


source = pd.read_csv(SOURCE)

pred = pd.read_csv(PRED)

pred = pred[
    pred["method"] == "raw"
].copy()


# Restore t_start exactly as previous sweep
parts = []

for rid in source.record_id.unique():

    s = (
        source[
            source.record_id == rid
        ]
        .sort_values("t_start")
        .reset_index(drop=True)
    )

    p = (
        pred[
            pred.record_id == rid
        ]
        .reset_index(drop=True)
    )

    p["t_start"] = (
        s["t_start"].to_numpy()
    )

    parts.append(p)


df = pd.concat(
    parts,
    ignore_index=True,
)


def apply_policy(prob):

    positive = (
        np.asarray(prob)
        >= THRESHOLD
    ).astype(int)

    alert = np.zeros(
        len(positive),
        dtype=int,
    )

    for i in range(len(positive)):

        if i < N - 1:
            continue

        recent = positive[
            i - N + 1:i + 1
        ]

        if recent.sum() >= K:
            alert[i] = 1

    return positive, alert


def count_episodes(alert):

    alert = np.asarray(
        alert,
        dtype=int,
    )

    starts = (
        (alert == 1)
        &
        np.r_[
            True,
            alert[:-1] == 0
        ]
    )

    return int(starts.sum())


rows = []

for rid, x in df.groupby(
    "record_id"
):

    x = x.sort_values(
        "t_start"
    )

    status = int(
        x.status.iloc[0]
    )

    positive, alert = (
        apply_policy(
            x.prob.to_numpy()
        )
    )

    duration_s = (
        x.t_start.max()
        - x.t_start.min()
        + 30
    )

    episodes = (
        count_episodes(alert)
    )

    first = np.flatnonzero(
        alert
    )

    rows.append({
        "record_id":
            rid,

        "status":
            status,

        "windows":
            len(x),

        "prob_positive_windows":
            int(positive.sum()),

        "prob_positive_fraction":
            positive.mean(),

        "alert_windows":
            int(alert.sum()),

        "alert_episodes":
            episodes,

        "episodes_per_hour":
            episodes
            / (duration_s / 3600),

        "max_prob":
            x.prob.max(),

        "median_prob":
            x.prob.median(),

        "first_alert_s":
            (
                x.iloc[
                    first[0]
                ].t_start
                + 30
                if len(first)
                else np.nan
            ),
    })


result = pd.DataFrame(rows)

result.to_csv(
    "experiments/05_calibration_alert/"
    "selected_policy_subject_audit.csv",
    index=False,
)


print("=" * 110)
print("NON-AF FALSE ALERT SUBJECTS")
print("=" * 110)

print(
    result[
        (result.status == 0)
        &
        (result.alert_episodes > 0)
    ]
    .sort_values(
        "episodes_per_hour",
        ascending=False,
    )
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 110)
print("AF SUBJECTS")
print("=" * 110)

print(
    result[
        result.status == 1
    ]
    .sort_values(
        "first_alert_s",
        ascending=False,
    )
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)
