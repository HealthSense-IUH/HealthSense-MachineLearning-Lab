from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


ROOT = Path(
    "/home/phuc/Documents/HealthSense_rungtamnhi"
)

ART = (
    ROOT /
    "experiments/08_v6_locked_protocol/artifacts"
)


DATASETS = [
    "deepbeat",
    "pulsewatch",
    "liu",
]


N_BOOT = 5000
SEED = 20260918

rng = np.random.default_rng(
    SEED
)


def ece(
    y,
    p,
    bins=10
):

    edges = np.linspace(
        0,
        1,
        bins + 1
    )

    ids = np.digitize(
        p,
        edges[1:-1]
    )


    value = 0.0


    for b in range(bins):

        mask = (
            ids == b
        )

        if not mask.any():
            continue


        value += (
            mask.mean()
            *
            abs(
                y[mask].mean()
                -
                p[mask].mean()
            )
        )


    return float(value)


def window_metrics(
    df,
    probability_col
):

    y = (
        df["label"]
        .astype(int)
        .to_numpy()
    )

    p = (
        df[
            probability_col
        ]
        .astype(float)
        .to_numpy()
    )


    if len(
        np.unique(y)
    ) < 2:

        return None


    return {
        "AUROC":
            roc_auc_score(
                y,
                p
            ),

        "PR_AUC":
            average_precision_score(
                y,
                p
            ),

        "Brier":
            brier_score_loss(
                y,
                p
            ),

        "ECE":
            ece(
                y,
                p
            ),
    }


def subject_metrics(
    gt,
    pred
):

    gt = np.asarray(
        gt,
        dtype=int
    )

    pred = np.asarray(
        pred,
        dtype=int
    )


    TP = int(
        (
            (gt == 1)
            &
            (pred == 1)
        ).sum()
    )

    TN = int(
        (
            (gt == 0)
            &
            (pred == 0)
        ).sum()
    )

    FP = int(
        (
            (gt == 0)
            &
            (pred == 1)
        ).sum()
    )

    FN = int(
        (
            (gt == 1)
            &
            (pred == 0)
        ).sum()
    )


    sens = (
        TP / (TP + FN)
        if TP + FN
        else np.nan
    )

    spec = (
        TN / (TN + FP)
        if TN + FP
        else np.nan
    )

    ppv = (
        TP / (TP + FP)
        if TP + FP
        else np.nan
    )

    f1 = (
        2 * TP
        /
        (
            2 * TP
            + FP
            + FN
        )
        if (
            2 * TP
            + FP
            + FN
        )
        else np.nan
    )


    return {
        "Sensitivity":
            sens,

        "Specificity":
            spec,

        "Balanced_Accuracy":
            (
                (sens + spec) / 2
                if (
                    not np.isnan(sens)
                    and
                    not np.isnan(spec)
                )
                else np.nan
            ),

        "PPV":
            ppv,

        "F1":
            f1,
    }


def percentile_ci(
    values
):

    x = np.asarray(
        values,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]


    if len(x) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
            0
        )


    return (
        float(
            np.mean(x)
        ),

        float(
            np.quantile(
                x,
                0.025
            )
        ),

        float(
            np.quantile(
                x,
                0.975
            )
        ),

        len(x)
    )


# ============================================================
# Window-level subject-cluster bootstrap
# ============================================================

window_rows = []


for dataset in DATASETS:

    f = (
        ART /
        f"{dataset}_test_calibrated_predictions.csv"
    )


    df = pd.read_csv(
        f,
        low_memory=False
    )


    df["subject_id"] = (
        df["subject_id"]
        .astype(str)
    )


    subjects = (
        df["subject_id"]
        .unique()
    )


    subject_frames = {
        s:
            df[
                df["subject_id"]
                ==
                s
            ]
        for s in subjects
    }


    for probability_col, method in [
        (
            "meta_probability",
            "RAW"
        ),
        (
            "calibrated_probability",
            "PLATT_DEV_FROZEN"
        ),
    ]:

        point = window_metrics(
            df,
            probability_col
        )


        boot = {
            "AUROC": [],
            "PR_AUC": [],
            "Brier": [],
            "ECE": [],
        }


        for _ in range(
            N_BOOT
        ):

            sampled = rng.choice(
                subjects,
                size=len(subjects),
                replace=True
            )


            parts = []


            for replicate_id, s in enumerate(
                sampled
            ):

                part = (
                    subject_frames[s]
                    .copy()
                )

                # Treat repeated draws as separate
                # bootstrap clusters.
                part[
                    "_bootstrap_cluster"
                ] = replicate_id

                parts.append(
                    part
                )


            sample = pd.concat(
                parts,
                ignore_index=True
            )


            m = window_metrics(
                sample,
                probability_col
            )


            if m is None:
                continue


            for metric, value in m.items():

                boot[
                    metric
                ].append(
                    value
                )


        for metric in [
            "AUROC",
            "PR_AUC",
            "Brier",
            "ECE",
        ]:

            mean, low, high, valid = (
                percentile_ci(
                    boot[
                        metric
                    ]
                )
            )


            window_rows.append({
                "dataset":
                    dataset,

                "split":
                    "REUSED_TEST",

                "method":
                    method,

                "metric":
                    metric,

                "point_estimate":
                    point[
                        metric
                    ],

                "bootstrap_mean":
                    mean,

                "CI95_low":
                    low,

                "CI95_high":
                    high,

                "valid_bootstrap_samples":
                    valid,

                "requested_bootstrap_samples":
                    N_BOOT,

                "bootstrap_unit":
                    "subject",
            })


window_ci = pd.DataFrame(
    window_rows
)


window_ci.to_csv(
    ART /
    "window_metrics_subject_cluster_bootstrap.csv",
    index=False
)


# ============================================================
# V2 subject-level bootstrap
# ============================================================

subject_rows = []


for dataset in DATASETS:

    f = (
        ART /
        f"{dataset}_global_rule_v2_subject_results.csv"
    )


    df = pd.read_csv(
        f
    )


    point = subject_metrics(
        df["gt"],
        df["alert"]
    )


    boot = {
        "Sensitivity": [],
        "Specificity": [],
        "Balanced_Accuracy": [],
        "PPV": [],
        "F1": [],
    }


    n = len(df)


    for _ in range(
        N_BOOT
    ):

        idx = rng.integers(
            0,
            n,
            size=n
        )


        sample = df.iloc[
            idx
        ]


        m = subject_metrics(
            sample["gt"],
            sample["alert"]
        )


        for metric, value in m.items():

            boot[
                metric
            ].append(
                value
            )


    for metric in [
        "Sensitivity",
        "Specificity",
        "Balanced_Accuracy",
        "PPV",
        "F1",
    ]:

        mean, low, high, valid = (
            percentile_ci(
                boot[
                    metric
                ]
            )
        )


        subject_rows.append({
            "dataset":
                dataset,

            "evaluation":
                "GLOBAL_RULE_V2_REUSED_TEST_EXPLORATORY",

            "metric":
                metric,

            "point_estimate":
                point[
                    metric
                ],

            "bootstrap_mean":
                mean,

            "CI95_low":
                low,

            "CI95_high":
                high,

            "valid_bootstrap_samples":
                valid,

            "requested_bootstrap_samples":
                N_BOOT,

            "bootstrap_unit":
                "subject",
        })


subject_ci = pd.DataFrame(
    subject_rows
)


subject_ci.to_csv(
    ART /
    "global_rule_v2_subject_bootstrap.csv",
    index=False
)


# ============================================================
# Print
# ============================================================

print(
    "\n" +
    "=" * 100
)

print(
    "WINDOW METRICS — SUBJECT CLUSTER BOOTSTRAP"
)

print(
    window_ci.to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "GLOBAL RULE V2 — SUBJECT BOOTSTRAP"
)

print(
    subject_ci.to_string(
        index=False
    )
)
