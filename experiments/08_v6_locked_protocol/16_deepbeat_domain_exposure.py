from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

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

MODEL = (
    ROOT /
    "models/multidomain/"
    "healthsense_af_v6c_stacking.pkl"
)


bundle = joblib.load(
    MODEL
)

FEATURES = list(
    bundle["features"]
)

BASE = bundle[
    "base_models"
]

META = bundle[
    "meta"
]


# ============================================================
# Load forensic DeepBeat table
#
# IMPORTANT:
# Use DEV + TEST only.
#
# The frozen full model was trained with DeepBeat TRAIN,
# so evaluating it on TRAIN would be contaminated.
# ============================================================

df = pd.read_csv(
    ART /
    "deepbeat_lodo_forensic_predictions.csv",
    low_memory=False
)


df["subject_eval"] = (
    df["subject_eval"]
    .astype(str)
)


df = df[
    df["source_split"]
    .isin([
        "dev",
        "test"
    ])
].copy()


print(
    "rows:",
    len(df)
)

print(
    "subjects:",
    df[
        "subject_eval"
    ].nunique()
)

print(
    df[
        "source_split"
    ].value_counts()
)


# ============================================================
# Frozen FULL multidomain model inference
# ============================================================

X = df[
    FEATURES
]


names = list(
    BASE.keys()
)


base_prob = np.column_stack([
    BASE[name]
    .predict_proba(
        X
    )[:, 1]

    for name
    in names
])


full_probability = (
    META.predict_proba(
        base_prob
    )[:, 1]
)


df[
    "full_multidomain_probability"
] = full_probability


# Existing LODO probability
df[
    "lodo_probability"
] = (
    df[
        "stack_probability"
    ]
)


# ============================================================
# Metrics
# ============================================================

def metrics(
    y,
    p,
    threshold=0.5
):

    y = np.asarray(
        y,
        dtype=int
    )

    p = np.asarray(
        p,
        dtype=float
    )


    pred = (
        p >= threshold
    ).astype(int)


    TP = int(
        (
            (y == 1)
            &
            (pred == 1)
        ).sum()
    )

    TN = int(
        (
            (y == 0)
            &
            (pred == 0)
        ).sum()
    )

    FP = int(
        (
            (y == 0)
            &
            (pred == 1)
        ).sum()
    )

    FN = int(
        (
            (y == 1)
            &
            (pred == 0)
        ).sum()
    )


    return {
        "AUROC":
            float(
                roc_auc_score(
                    y,
                    p
                )
            ),

        "PR_AUC":
            float(
                average_precision_score(
                    y,
                    p
                )
            ),

        "Brier":
            float(
                brier_score_loss(
                    y,
                    p
                )
            ),

        "TP":
            TP,

        "TN":
            TN,

        "FP":
            FP,

        "FN":
            FN,

        "sensitivity":
            TP / (TP + FN),

        "specificity":
            TN / (TN + FP),
    }


summary = []


for split in [
    "dev",
    "test",
    "DEV_TEST_COMBINED"
]:

    if split == "DEV_TEST_COMBINED":

        g = df

    else:

        g = df[
            df[
                "source_split"
            ]
            ==
            split
        ]


    for method, col in [
        (
            "LODO_NO_DEEPBEAT",
            "lodo_probability"
        ),
        (
            "FULL_WITH_DEEPBEAT_TRAIN",
            "full_multidomain_probability"
        ),
    ]:

        m = metrics(
            g[
                "label_eval"
            ],
            g[
                col
            ]
        )


        summary.append({
            "split":
                split,

            "method":
                method,

            "rows":
                len(g),

            "subjects":
                g[
                    "subject_eval"
                ].nunique(),

            **m
        })


summary = pd.DataFrame(
    summary
)


summary.to_csv(
    ART /
    "deepbeat_domain_exposure_summary.csv",
    index=False
)


# ============================================================
# Direct probability changes caused by DeepBeat exposure
# ============================================================

df[
    "probability_gain_from_domain_exposure"
] = (
    df[
        "full_multidomain_probability"
    ]
    -
    df[
        "lodo_probability"
    ]
)


probability_summary = (
    df.groupby(
        [
            "source_split",
            "label_eval"
        ]
    )
    .agg(
        windows=(
            "label_eval",
            "size"
        ),

        lodo_mean=(
            "lodo_probability",
            "mean"
        ),

        full_mean=(
            "full_multidomain_probability",
            "mean"
        ),

        mean_gain=(
            "probability_gain_from_domain_exposure",
            "mean"
        ),

        median_gain=(
            "probability_gain_from_domain_exposure",
            "median"
        ),
    )
    .reset_index()
)


probability_summary.to_csv(
    ART /
    "deepbeat_domain_exposure_probability_gain.csv",
    index=False
)


# ============================================================
# AF phenotype quartiles
#
# Descriptive ONLY.
# No decision threshold is selected from these bins.
# ============================================================

af = df[
    df[
        "label_eval"
    ]
    ==
    1
].copy()


phenotype_rows = []


for feature in [
    "PPG_AC",
    "CV",
    "RMSSD",
    "pNN50",
    "NN50",
    "SampEn",
]:

    try:

        af[
            "_quartile"
        ] = pd.qcut(
            af[
                feature
            ],
            q=4,
            labels=[
                "Q1",
                "Q2",
                "Q3",
                "Q4"
            ],
            duplicates="drop"
        )

    except Exception:

        continue


    for quartile, g in af.groupby(
        "_quartile",
        observed=True
    ):

        phenotype_rows.append({
            "feature":
                feature,

            "quartile":
                str(
                    quartile
                ),

            "windows":
                len(g),

            "feature_min":
                float(
                    g[
                        feature
                    ].min()
                ),

            "feature_median":
                float(
                    g[
                        feature
                    ].median()
                ),

            "feature_max":
                float(
                    g[
                        feature
                    ].max()
                ),

            "LODO_mean_probability":
                float(
                    g[
                        "lodo_probability"
                    ].mean()
                ),

            "FULL_mean_probability":
                float(
                    g[
                        "full_multidomain_probability"
                    ].mean()
                ),

            "LODO_sensitivity_05":
                float(
                    (
                        g[
                            "lodo_probability"
                        ]
                        >= 0.5
                    ).mean()
                ),

            "FULL_sensitivity_05":
                float(
                    (
                        g[
                            "full_multidomain_probability"
                        ]
                        >= 0.5
                    ).mean()
                ),
        })


phenotype = pd.DataFrame(
    phenotype_rows
)


phenotype.to_csv(
    ART /
    "deepbeat_af_phenotype_quartiles.csv",
    index=False
)


# ============================================================
# Spearman relationship between AF probability and features
# ============================================================

correlation_rows = []


for feature in FEATURES:

    for method, col in [
        (
            "LODO",
            "lodo_probability"
        ),
        (
            "FULL",
            "full_multidomain_probability"
        ),
    ]:

        valid = (
            af[
                [
                    feature,
                    col
                ]
            ]
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna()
        )


        rho, pvalue = spearmanr(
            valid[
                feature
            ],
            valid[
                col
            ]
        )


        correlation_rows.append({
            "feature":
                feature,

            "method":
                method,

            "n":
                len(valid),

            "spearman_rho":
                float(
                    rho
                ),

            "pvalue":
                float(
                    pvalue
                ),
        })


correlation = pd.DataFrame(
    correlation_rows
)


correlation.to_csv(
    ART /
    "deepbeat_af_feature_probability_correlation.csv",
    index=False
)


# ============================================================
# Rhythm/source diagnostic if available
# ============================================================

for column in [
    "rhythm",
    "source"
]:

    if column not in df.columns:
        continue


    rows = []


    for value, g in df.groupby(
        column,
        dropna=False
    ):

        for method, pcol in [
            (
                "LODO",
                "lodo_probability"
            ),
            (
                "FULL",
                "full_multidomain_probability"
            ),
        ]:

            if (
                g[
                    "label_eval"
                ].nunique()
                <
                2
            ):

                auc = np.nan

            else:

                auc = roc_auc_score(
                    g[
                        "label_eval"
                    ],
                    g[
                        pcol
                    ]
                )


            rows.append({
                column:
                    value,

                "method":
                    method,

                "rows":
                    len(g),

                "subjects":
                    g[
                        "subject_eval"
                    ].nunique(),

                "AF_windows":
                    int(
                        (
                            g[
                                "label_eval"
                            ]
                            ==
                            1
                        ).sum()
                    ),

                "mean_probability":
                    float(
                        g[
                            pcol
                        ].mean()
                    ),

                "AUROC":
                    auc,
            })


    pd.DataFrame(
        rows
    ).to_csv(
        ART /
        f"deepbeat_domain_exposure_by_{column}.csv",
        index=False
    )


# ============================================================
# Save predictions
# ============================================================

df.to_csv(
    ART /
    "deepbeat_domain_exposure_predictions.csv",
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
    "DOMAIN EXPOSURE"
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "PROBABILITY GAIN"
)

print(
    probability_summary.to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "PPG_AC QUARTILES"
)

print(
    phenotype[
        phenotype[
            "feature"
        ]
        ==
        "PPG_AC"
    ].to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "AF FEATURE ↔ PROBABILITY CORRELATION"
)

print(
    correlation[
        correlation[
            "method"
        ]
        ==
        "LODO"
    ]
    .sort_values(
        "spearman_rho"
    )
    .to_string(
        index=False
    )
)
