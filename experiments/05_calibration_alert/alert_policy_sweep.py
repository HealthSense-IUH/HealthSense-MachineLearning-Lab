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


source = pd.read_csv(SOURCE)

pred = pd.read_csv(PRED)

pred = pred[
    pred["method"] == "raw"
].copy()


# ---------------------------------------------------------
# Restore t_start
# ---------------------------------------------------------

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

    if len(s) != len(p):
        raise RuntimeError(
            f"{rid}: "
            f"source={len(s)} "
            f"pred={len(p)}"
        )

    p["t_start"] = (
        s["t_start"]
        .to_numpy()
    )

    parts.append(p)


df = pd.concat(
    parts,
    ignore_index=True,
)


# ---------------------------------------------------------
# K-of-N voting
# ---------------------------------------------------------

def apply_policy(prob, threshold, k, n):

    positive = (
        np.asarray(prob)
        >= threshold
    ).astype(int)

    alert = np.zeros(
        len(positive),
        dtype=int,
    )

    for i in range(len(positive)):

        start = max(
            0,
            i - n + 1,
        )

        recent = positive[
            start:i + 1
        ]

        # Require a complete N-window history.
        if len(recent) < n:
            continue

        if recent.sum() >= k:
            alert[i] = 1

    return alert


def count_alert_episodes(alert):

    alert = np.asarray(
        alert,
        dtype=int,
    )

    if len(alert) == 0:
        return 0

    starts = (
        (alert == 1)
        &
        np.r_[
            True,
            alert[:-1] == 0
        ]
    )

    return int(
        starts.sum()
    )


POLICIES = [
    (1, 1),
    (2, 3),
    (3, 3),
    (3, 5),
]


THRESHOLDS = np.arange(
    0.30,
    0.91,
    0.05,
)


rows = []


for threshold in THRESHOLDS:

    for k, n in POLICIES:

        subject_rows = []

        all_truth = []
        all_alert = []

        for rid, x in (
            df.groupby("record_id")
        ):

            x = x.sort_values(
                "t_start"
            )

            truth = int(
                x.status.iloc[0]
            )

            alert = apply_policy(
                x.prob.to_numpy(),
                threshold,
                k,
                n,
            )

            all_truth.extend(
                [truth] * len(alert)
            )

            all_alert.extend(
                alert.tolist()
            )

            duration_s = (
                x.t_start.max()
                - x.t_start.min()
                + 30.0
            )

            duration_h = (
                duration_s / 3600.0
            )

            episodes = (
                count_alert_episodes(
                    alert
                )
            )

            alert_idx = np.flatnonzero(
                alert
            )

            first_alert_s = (
                float(
                    x.iloc[
                        alert_idx[0]
                    ].t_start
                    + 30.0
                )
                if len(alert_idx)
                else np.nan
            )

            subject_rows.append({
                "record_id":
                    rid,

                "status":
                    truth,

                "any_alert":
                    int(
                        alert.any()
                    ),

                "alert_episodes":
                    episodes,

                "episodes_per_hour":
                    (
                        episodes
                        / duration_h
                        if duration_h > 0
                        else np.nan
                    ),

                "first_alert_s":
                    first_alert_s,
            })


        subject = pd.DataFrame(
            subject_rows
        )

        af = subject[
            subject.status == 1
        ]

        non = subject[
            subject.status == 0
        ]

        all_truth = np.asarray(
            all_truth
        )

        all_alert = np.asarray(
            all_alert
        )

        tp = np.sum(
            (all_truth == 1)
            &
            (all_alert == 1)
        )

        fn = np.sum(
            (all_truth == 1)
            &
            (all_alert == 0)
        )

        tn = np.sum(
            (all_truth == 0)
            &
            (all_alert == 0)
        )

        fp = np.sum(
            (all_truth == 0)
            &
            (all_alert == 1)
        )

        window_sens = (
            tp / (tp + fn)
            if tp + fn
            else np.nan
        )

        window_spec = (
            tn / (tn + fp)
            if tn + fp
            else np.nan
        )

        subject_sens = (
            af.any_alert.mean()
        )

        subject_spec = (
            1.0
            - non.any_alert.mean()
        )

        detected_af = af[
            af.any_alert == 1
        ]

        rows.append({
            "threshold":
                round(
                    float(threshold),
                    2,
                ),

            "policy":
                f"{k}-of-{n}",

            "window_sensitivity":
                window_sens,

            "window_specificity":
                window_spec,

            "subject_sensitivity":
                subject_sens,

            "subject_specificity":
                subject_spec,

            "af_detected":
                int(
                    af.any_alert.sum()
                ),

            "af_total":
                len(af),

            "nonaf_false_subjects":
                int(
                    non.any_alert.sum()
                ),

            "nonaf_total":
                len(non),

            "median_false_alerts_per_hour":
                non[
                    "episodes_per_hour"
                ].median(),

            "mean_false_alerts_per_hour":
                non[
                    "episodes_per_hour"
                ].mean(),

            "median_first_alert_s":
                detected_af[
                    "first_alert_s"
                ].median()
                if len(detected_af)
                else np.nan,
        })


result = pd.DataFrame(rows)


result.to_csv(
    "experiments/05_calibration_alert/"
    "alert_policy_sweep.csv",
    index=False,
)


print("=" * 120)
print("ALERT POLICY SWEEP")
print("=" * 120)

show = result[
    [
        "threshold",
        "policy",
        "subject_sensitivity",
        "subject_specificity",
        "nonaf_false_subjects",
        "median_false_alerts_per_hour",
        "window_sensitivity",
        "window_specificity",
        "median_first_alert_s",
    ]
]

print(
    show.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 120)
print("HIGH-SENSITIVITY CANDIDATES")
print("=" * 120)

candidate = result[
    result[
        "subject_sensitivity"
    ] >= 0.95
].copy()

candidate = candidate.sort_values(
    [
        "subject_specificity",
        "mean_false_alerts_per_hour",
        "window_specificity",
    ],
    ascending=[
        False,
        True,
        False,
    ],
)

print(
    candidate[
        [
            "threshold",
            "policy",
            "subject_sensitivity",
            "subject_specificity",
            "af_detected",
            "nonaf_false_subjects",
            "median_false_alerts_per_hour",
            "mean_false_alerts_per_hour",
            "window_sensitivity",
            "window_specificity",
            "median_first_alert_s",
        ]
    ]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)
