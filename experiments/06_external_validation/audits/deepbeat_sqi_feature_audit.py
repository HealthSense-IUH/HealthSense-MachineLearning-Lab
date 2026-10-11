from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

ROOT = Path("experiments/06_external_validation")

FEATURE_FILE = (
    ROOT /
    "deepbeat_frozen_v5_features.csv"
)

PRED_FILE = (
    ROOT /
    "deepbeat_frozen_v5_predictions.csv"
)

OUT_FEATURE = (
    ROOT /
    "deepbeat_sqi_feature_audit.csv"
)

OUT_MODEL = (
    ROOT /
    "deepbeat_sqi_model_by_quality.csv"
)

QUALITY_NAMES = {
    0: "poor",
    1: "acceptable",
    2: "excellent",
}


# ============================================================
# LOAD
# ============================================================

feat = pd.read_csv(
    FEATURE_FILE
)

pred = pd.read_csv(
    PRED_FILE
)

print("=" * 105)
print("DEEPBEAT SQI EXPLORATORY AUDIT")
print("=" * 105)

print(
    "\nFeature rows:",
    len(feat)
)

print(
    "Prediction rows:",
    len(pred)
)

print(
    "Subjects:",
    feat.subject_id.nunique()
)


# ============================================================
# QUALITY / RHYTHM DISTRIBUTION
# ============================================================

print()
print("=" * 105)
print("[1] QUALITY × RHYTHM")
print("=" * 105)

quality_rhythm = pd.crosstab(
    feat["quality_name"],
    feat["rhythm"],
    margins=True,
)

print(
    quality_rhythm.to_string()
)


# ============================================================
# COVERAGE
# ============================================================

print()
print("=" * 105)
print("[2] FEATURE COVERAGE")
print("=" * 105)

coverage = (
    feat.groupby(
        [
            "rhythm",
            "quality_name",
        ]
    )
    ["feature_ok"]
    .agg(
        total="count",
        usable="sum",
        coverage="mean",
    )
)

print(
    coverage.to_string(
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ============================================================
# MODEL BEHAVIOUR BY QUALITY
# ============================================================

print()
print("=" * 105)
print("[3] BASELINE MODEL BY QUALITY")
print("=" * 105)

model_rows = []

for q in [
    0,
    1,
    2,
]:

    qname = QUALITY_NAMES[q]

    for rhythm in [
        "AF",
        "NON_AF",
    ]:

        d = pred[
            (pred.quality == q)
            &
            (pred.rhythm == rhythm)
        ].copy()

        if len(d) == 0:
            continue

        positive_rate = float(
            d[
                "fixed_090_positive"
            ].mean()
        )

        p_median = float(
            d[
                "af_probability"
            ].median()
        )

        p_mean = float(
            d[
                "af_probability"
            ].mean()
        )

        model_rows.append({
            "quality":
                q,

            "quality_name":
                qname,

            "rhythm":
                rhythm,

            "n":
                len(d),

            "positive_rate":
                positive_rate,

            "p_af_mean":
                p_mean,

            "p_af_median":
                p_median,

            "metric_name":
                (
                    "sensitivity"
                    if rhythm == "AF"
                    else "false_positive_rate"
                ),
        })

        print(
            f"{qname:12s} "
            f"{rhythm:7s} "
            f"n={len(d):5d} "
            f"positive="
            f"{positive_rate*100:7.3f}% "
            f"p_med="
            f"{p_median:.4f}"
        )


pd.DataFrame(
    model_rows
).to_csv(
    OUT_MODEL,
    index=False,
)


# ============================================================
# FEATURE CANDIDATES
#
# Important:
# HRV features can encode AF itself.
# Therefore every SQI proxy is checked:
#   - globally
#   - inside AF only
#   - inside non-AF only
#
# Target:
#   quality > 0 = acceptable/excellent
#   quality = 0 = poor
# ============================================================

candidate_features = [
    "n_nn",
    "PPG_AC",
    "HR_mean",
    "Mean_NN",
    "SDNN",
    "RMSSD",
    "CV",
    "HF",
    "Total_Power",
    "HF_norm",
    "SD1",
    "SD2",
    "SampEn",
]

candidate_features = [
    x
    for x in candidate_features
    if x in feat.columns
]


usable = feat[
    feat.feature_ok == True
].copy()

usable[
    "quality_good"
] = (
    usable.quality > 0
).astype(int)


rows = []


def evaluate_feature(
    data,
    subset_name,
    feature,
):

    d = data[
        [
            feature,
            "quality",
            "quality_good",
        ]
    ].copy()

    d = d.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if (
        len(d) == 0
        or
        d.quality_good.nunique()
        < 2
    ):
        return

    auc_raw = roc_auc_score(
        d.quality_good,
        d[feature],
    )

    if auc_raw >= 0.5:

        auc_directionless = (
            auc_raw
        )

        direction = (
            "higher = better quality"
        )

    else:

        auc_directionless = (
            1.0 - auc_raw
        )

        direction = (
            "lower = better quality"
        )

    def qmedian(q):

        x = d[
            d.quality == q
        ][feature]

        return (
            float(x.median())
            if len(x)
            else np.nan
        )

    rows.append({
        "subset":
            subset_name,

        "feature":
            feature,

        "n":
            len(d),

        "n_poor":
            int(
                (
                    d.quality == 0
                ).sum()
            ),

        "n_good":
            int(
                (
                    d.quality > 0
                ).sum()
            ),

        "auc_raw_good_quality":
            float(
                auc_raw
            ),

        "auc_directionless":
            float(
                auc_directionless
            ),

        "direction":
            direction,

        "median_poor":
            qmedian(0),

        "median_acceptable":
            qmedian(1),

        "median_excellent":
            qmedian(2),
    })


subsets = {
    "ALL":
        usable,

    "AF_ONLY":
        usable[
            usable.rhythm
            == "AF"
        ],

    "NON_AF_ONLY":
        usable[
            usable.rhythm
            == "NON_AF"
        ],
}


for subset_name, data in (
    subsets.items()
):

    for feature in (
        candidate_features
    ):

        evaluate_feature(
            data,
            subset_name,
            feature,
        )


result = pd.DataFrame(
    rows
)

result.to_csv(
    OUT_FEATURE,
    index=False,
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 105)
print("[4] BEST QUALITY-SEPARATING FEATURES")
print("=" * 105)

for subset in [
    "ALL",
    "AF_ONLY",
    "NON_AF_ONLY",
]:

    print(
        f"\n{subset}"
    )

    d = (
        result[
            result.subset == subset
        ]
        .sort_values(
            "auc_directionless",
            ascending=False,
        )
    )

    print(
        d[
            [
                "feature",
                "auc_directionless",
                "direction",
                "median_poor",
                "median_acceptable",
                "median_excellent",
            ]
        ]
        .head(12)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )


# ============================================================
# CONSISTENCY CHECK
#
# A deployable SQI candidate should ideally:
# - separate good/poor in AF
# - separate good/poor in non-AF
# - have the SAME direction in both groups
# ============================================================

print()
print("=" * 105)
print("[5] AF / NON-AF CONSISTENCY")
print("=" * 105)

af = (
    result[
        result.subset
        == "AF_ONLY"
    ]
    [
        [
            "feature",
            "auc_directionless",
            "direction",
        ]
    ]
    .rename(
        columns={
            "auc_directionless":
                "auc_AF",

            "direction":
                "direction_AF",
        }
    )
)

non = (
    result[
        result.subset
        == "NON_AF_ONLY"
    ]
    [
        [
            "feature",
            "auc_directionless",
            "direction",
        ]
    ]
    .rename(
        columns={
            "auc_directionless":
                "auc_NON_AF",

            "direction":
                "direction_NON_AF",
        }
    )
)

consistency = af.merge(
    non,
    on="feature",
    how="inner",
)

consistency[
    "same_direction"
] = (
    consistency.direction_AF
    ==
    consistency.direction_NON_AF
)

consistency[
    "worst_stratum_auc"
] = consistency[
    [
        "auc_AF",
        "auc_NON_AF",
    ]
].min(axis=1)

consistency = (
    consistency.sort_values(
        [
            "same_direction",
            "worst_stratum_auc",
        ],
        ascending=[
            False,
            False,
        ],
    )
)


print(
    consistency.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.5f}",
    )
)


print()
print("=" * 105)
print("IMPORTANT")
print("=" * 105)

print(
    "This is an exploratory SQI audit."
)

print(
    "Do NOT choose a quality threshold "
    "from DeepBeat test data yet."
)

print(
    "Features that separate quality only "
    "globally but fail within AF/non-AF "
    "are likely rhythm-confounded and "
    "should NOT be used as SQI gates."
)

print()
print("SAVED:")
print(OUT_FEATURE)
print(OUT_MODEL)
