from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

ROOT = Path(
    "experiments/06_external_validation"
)

PRED = ROOT / (
    "pulsewatch_frozen_v5_predictions.csv"
)

FEAT = ROOT / (
    "pulsewatch_frozen_v5_features.csv"
)

OUT_SUBJECT = ROOT / (
    "pulsewatch_v5_subject_class_performance.csv"
)

OUT_PAIRWISE = ROOT / (
    "pulsewatch_v5_pairwise_auc.csv"
)

OUT_COVERAGE = ROOT / (
    "pulsewatch_v5_coverage_by_class.csv"
)

pred = pd.read_csv(
    PRED,
    dtype={"uid": str},
    low_memory=False,
)

feat = pd.read_csv(
    FEAT,
    dtype={"uid": str},
    low_memory=False,
)

pred["uid"] = (
    pred["uid"]
    .astype(str)
    .str.zfill(3)
)

feat["uid"] = (
    feat["uid"]
    .astype(str)
    .str.zfill(3)
)

pred["positive_05"] = (
    pred["af_probability"]
    >= 0.50
)

pred["positive_09"] = (
    pred["af_probability"]
    >= 0.90
)


# ============================================================
# COVERAGE
# ============================================================

coverage = (
    feat.groupby(
        "class_name"
    )
    .agg(
        total=(
            "segment",
            "size",
        ),

        feature_ok=(
            "feature_ok",
            "sum",
        ),
    )
    .reset_index()
)

coverage["coverage"] = (
    coverage["feature_ok"]
    /
    coverage["total"]
)

coverage["no_decision"] = (
    coverage["total"]
    -
    coverage["feature_ok"]
)

coverage.to_csv(
    OUT_COVERAGE,
    index=False,
)


# ============================================================
# PAIRWISE AUC
# ============================================================

pair_rows = []

for negative in [
    "NSR",
    "PAC_PVC",
]:

    g = pred[
        pred["class_name"].isin(
            [
                "AF",
                negative,
            ]
        )
    ].copy()

    y = (
        g["class_name"]
        == "AF"
    ).astype(int)

    auc = roc_auc_score(
        y,
        g["af_probability"],
    )

    pr = average_precision_score(
        y,
        g["af_probability"],
    )

    pair_rows.append(
        {
            "comparison":
                f"AF_vs_{negative}",

            "n":
                len(g),

            "AF":
                int(y.sum()),

            "negative":
                int((1-y).sum()),

            "roc_auc":
                auc,

            "pr_auc":
                pr,
        }
    )

pairwise = pd.DataFrame(
    pair_rows
)

pairwise.to_csv(
    OUT_PAIRWISE,
    index=False,
)


# ============================================================
# SUBJECT × CLASS
#
# This is important because segment count per subject
# varies enormously.
# ============================================================

subject = (
    pred.groupby(
        [
            "uid",
            "class_name",
            "format",
            "trial",
        ],
        as_index=False,
    )
    .agg(
        segments=(
            "segment",
            "size",
        ),

        mean_probability=(
            "af_probability",
            "mean",
        ),

        median_probability=(
            "af_probability",
            "median",
        ),

        positive_rate_05=(
            "positive_05",
            "mean",
        ),

        positive_rate_09=(
            "positive_09",
            "mean",
        ),
    )
)

subject.to_csv(
    OUT_SUBJECT,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print("=" * 90)
print("06A.4.4 PULSEWATCH V5 ROBUSTNESS AUDIT")
print("=" * 90)


print()
print("===== COVERAGE =====")

print(
    coverage.to_string(
        index=False
    )
)


print()
print("===== PAIRWISE DISCRIMINATION =====")

print(
    pairwise.to_string(
        index=False
    )
)


print()
print("===== SUBJECT-MACRO @ 0.90 =====")

for cls in [
    "AF",
    "NSR",
    "PAC_PVC",
]:

    g = subject[
        subject[
            "class_name"
        ]
        == cls
    ]

    print()
    print(
        cls,
        "subjects =",
        len(g),
    )

    print(
        "median positive rate =",
        g[
            "positive_rate_09"
        ].median()
    )

    print(
        "mean positive rate   =",
        g[
            "positive_rate_09"
        ].mean()
    )

    print(
        "p10 / p90            =",
        g[
            "positive_rate_09"
        ].quantile(
            [0.1, 0.9]
        ).to_dict()
    )


print()
print("===== PAC/PVC vs NSR FALSE-POSITIVE RISK =====")

nsr = pred[
    pred["class_name"]
    == "NSR"
]

pac = pred[
    pred["class_name"]
    == "PAC_PVC"
]

nsr_fp = (
    nsr["positive_09"]
    .mean()
)

pac_fp = (
    pac["positive_09"]
    .mean()
)

print(
    "NSR FPR @0.90     :",
    nsr_fp,
)

print(
    "PAC/PVC FPR @0.90 :",
    pac_fp,
)

print(
    "Risk ratio        :",
    (
        pac_fp / nsr_fp
        if nsr_fp > 0
        else np.nan
    )
)


print()
print("===== WORST AF SUBJECTS @ 0.90 =====")

print(
    subject[
        subject["class_name"]
        == "AF"
    ]
    .sort_values(
        "positive_rate_09"
    )
    .head(15)
    .to_string(
        index=False
    )
)


print()
print("===== WORST PAC/PVC FALSE-POSITIVE SUBJECTS =====")

print(
    subject[
        subject["class_name"]
        == "PAC_PVC"
    ]
    .sort_values(
        "positive_rate_09",
        ascending=False,
    )
    .head(15)
    .to_string(
        index=False
    )
)


print()
print("Saved:")
print(" ", OUT_COVERAGE)
print(" ", OUT_PAIRWISE)
print(" ", OUT_SUBJECT)
