from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

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

MANIFEST = (
    ROOT /
    "data/features/v7/"
    "healthsense_af_v6c_pacaware_train.csv"
)


FEATURES = [
    "HR_mean",
    "Mean_NN",
    "SDNN",
    "RMSSD",
    "NN50",
    "pNN50",
    "CV",
    "HF",
    "Total_Power",
    "HF_norm",
    "SD1",
    "SD2",
    "SampEn",
    "PPG_AC",
]


# ============================================================
# Helpers
# ============================================================

def valid_mask(
    df
):

    mask = (
        df[FEATURES]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .notna()
        .all(axis=1)
    )


    if "feature_ok" in df.columns:

        ok = (
            df["feature_ok"]
            .astype(str)
            .str.lower()
            .isin([
                "1",
                "true",
                "yes"
            ])
        )

        mask &= ok


    return mask


def safe_metrics(
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
        "rows":
            len(y),

        "AF_windows":
            int(
                (y == 1).sum()
            ),

        "nonAF_windows":
            int(
                (y == 0).sum()
            ),

        "AUROC":
            (
                float(
                    roc_auc_score(
                        y,
                        p
                    )
                )
                if len(
                    np.unique(y)
                ) == 2
                else np.nan
            ),

        "PR_AUC":
            (
                float(
                    average_precision_score(
                        y,
                        p
                    )
                )
                if (
                    y == 1
                ).any()
                else np.nan
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
            (
                TP /
                (
                    TP + FN
                )
                if (
                    TP + FN
                )
                else np.nan
            ),

        "specificity":
            (
                TN /
                (
                    TN + FP
                )
                if (
                    TN + FP
                )
                else np.nan
            ),
    }


def robust_stats(
    x
):

    x = np.asarray(
        x,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]


    if len(x) == 0:

        return {
            "median":
                np.nan,

            "q1":
                np.nan,

            "q3":
                np.nan,

            "iqr":
                np.nan,
        }


    q1 = np.quantile(
        x,
        0.25
    )

    median = np.quantile(
        x,
        0.50
    )

    q3 = np.quantile(
        x,
        0.75
    )


    return {
        "median":
            float(median),

        "q1":
            float(q1),

        "q3":
            float(q3),

        "iqr":
            float(
                q3 - q1
            ),
    }


# ============================================================
# Reconstruct exact DeepBeat LODO evaluation order
# ============================================================

parts = []


for split, file in [
    (
        "train",
        ROOT /
        "data/features/v6/"
        "deepbeat_train_v6_14f.csv"
    ),
    (
        "dev",
        ROOT /
        "data/features/v6/"
        "deepbeat_validate_v6_14f.csv"
    ),
    (
        "test",
        ROOT /
        "data/features/v6/"
        "deepbeat_test_independent_v6_14f.csv"
    ),
]:

    x = pd.read_csv(
        file,
        low_memory=False
    )


    mask = valid_mask(
        x
    )


    print(
        split,
        "raw=",
        len(x),
        "valid=",
        int(
            mask.sum()
        ),
        "dropped=",
        int(
            (~mask).sum()
        )
    )


    x = (
        x.loc[
            mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    x["source_split"] = split

    x["subject_eval"] = (
        x["subject_id"]
        .astype(str)
    )

    x["label_eval"] = (
        pd.to_numeric(
            x["label"],
            errors="raise"
        )
        .astype(int)
    )


    parts.append(
        x
    )


deepbeat = pd.concat(
    parts,
    ignore_index=True
)


# ============================================================
# Attach exact LODO predictions
# ============================================================

pred = pd.read_csv(
    ART /
    "lodo_deepbeat_predictions.csv",
    low_memory=False
)


pred["subject_id"] = (
    pred["subject_id"]
    .astype(str)
)


if len(pred) != len(deepbeat):

    raise RuntimeError(
        f"Row count mismatch: "
        f"pred={len(pred)} "
        f"features={len(deepbeat)}"
    )


if not np.array_equal(
    pred[
        "label"
    ].astype(int).to_numpy(),
    deepbeat[
        "label_eval"
    ].astype(int).to_numpy()
):

    raise RuntimeError(
        "Label sequence mismatch"
    )


if not np.array_equal(
    pred[
        "subject_id"
    ].to_numpy(),
    deepbeat[
        "subject_eval"
    ].to_numpy()
):

    raise RuntimeError(
        "Subject sequence mismatch"
    )


if not np.array_equal(
    pred[
        "source_split"
    ].astype(str).to_numpy(),
    deepbeat[
        "source_split"
    ].astype(str).to_numpy()
):

    raise RuntimeError(
        "Split sequence mismatch"
    )


deepbeat[
    "stack_probability"
] = pred[
    "stack_probability"
].to_numpy()


for col in [
    "ExtraTrees_prob",
    "RandomForest_prob",
    "XGBoost_prob",
]:

    if col in pred.columns:

        deepbeat[
            col
        ] = pred[
            col
        ].to_numpy()


deepbeat[
    "prediction_05"
] = (
    deepbeat[
        "stack_probability"
    ]
    >= 0.5
).astype(int)


deepbeat[
    "error_type"
] = np.select(
    [
        (
            deepbeat["label_eval"]
            == 1
        )
        &
        (
            deepbeat["prediction_05"]
            == 1
        ),

        (
            deepbeat["label_eval"]
            == 0
        )
        &
        (
            deepbeat["prediction_05"]
            == 0
        ),

        (
            deepbeat["label_eval"]
            == 0
        )
        &
        (
            deepbeat["prediction_05"]
            == 1
        ),

        (
            deepbeat["label_eval"]
            == 1
        )
        &
        (
            deepbeat["prediction_05"]
            == 0
        ),
    ],

    [
        "TP",
        "TN",
        "FP",
        "FN",
    ],

    default="UNKNOWN"
)


deepbeat.to_csv(
    ART /
    "deepbeat_lodo_forensic_predictions.csv",
    index=False
)


print(
    "\nDeepBeat rows:",
    len(deepbeat)
)

print(
    "subjects:",
    deepbeat[
        "subject_eval"
    ].nunique()
)

print(
    "errors:"
)

print(
    deepbeat[
        "error_type"
    ].value_counts()
)


# ============================================================
# 1. Performance by DeepBeat source split
# ============================================================

split_rows = []


for split, g in deepbeat.groupby(
    "source_split"
):

    m = safe_metrics(
        g["label_eval"],
        g["stack_probability"]
    )


    split_rows.append({
        "split":
            split,

        "subjects":
            g[
                "subject_eval"
            ].nunique(),

        **m,
    })


split_summary = pd.DataFrame(
    split_rows
)


split_summary.to_csv(
    ART /
    "deepbeat_lodo_split_summary.csv",
    index=False
)


# ============================================================
# 2. Base learner vs stacking
# ============================================================

model_rows = []


probability_columns = [
    (
        "STACK",
        "stack_probability"
    )
]


for c in [
    "ExtraTrees_prob",
    "RandomForest_prob",
    "XGBoost_prob",
]:

    if c in deepbeat.columns:

        probability_columns.append(
            (
                c.replace(
                    "_prob",
                    ""
                ),
                c
            )
        )


for name, col in probability_columns:

    m = safe_metrics(
        deepbeat[
            "label_eval"
        ],
        deepbeat[
            col
        ]
    )


    model_rows.append({
        "model":
            name,

        **m,
    })


model_summary = pd.DataFrame(
    model_rows
)


model_summary.to_csv(
    ART /
    "deepbeat_lodo_base_vs_stack.csv",
    index=False
)


# ============================================================
# 3. Subject-level forensic summary
# ============================================================

subject_rows = []


for subject_id, g in deepbeat.groupby(
    "subject_eval"
):

    af = g[
        g["label_eval"]
        ==
        1
    ]

    nonaf = g[
        g["label_eval"]
        ==
        0
    ]


    TP = int(
        (
            g["error_type"]
            ==
            "TP"
        ).sum()
    )

    TN = int(
        (
            g["error_type"]
            ==
            "TN"
        ).sum()
    )

    FP = int(
        (
            g["error_type"]
            ==
            "FP"
        ).sum()
    )

    FN = int(
        (
            g["error_type"]
            ==
            "FN"
        ).sum()
    )


    subject_rows.append({
        "subject_id":
            subject_id,

        "split":
            g[
                "source_split"
            ].iloc[0],

        "windows":
            len(g),

        "AF_windows":
            len(af),

        "nonAF_windows":
            len(nonaf),

        "TP":
            TP,

        "TN":
            TN,

        "FP":
            FP,

        "FN":
            FN,

        "AF_sensitivity":
            (
                TP /
                (
                    TP + FN
                )
                if (
                    TP + FN
                )
                else np.nan
            ),

        "nonAF_specificity":
            (
                TN /
                (
                    TN + FP
                )
                if (
                    TN + FP
                )
                else np.nan
            ),

        "AF_mean_probability":
            (
                float(
                    af[
                        "stack_probability"
                    ].mean()
                )
                if len(af)
                else np.nan
            ),

        "AF_median_probability":
            (
                float(
                    af[
                        "stack_probability"
                    ].median()
                )
                if len(af)
                else np.nan
            ),

        "nonAF_mean_probability":
            (
                float(
                    nonaf[
                        "stack_probability"
                    ].mean()
                )
                if len(nonaf)
                else np.nan
            ),

        "max_probability":
            float(
                g[
                    "stack_probability"
                ].max()
            ),
    })


subject_summary = pd.DataFrame(
    subject_rows
)


subject_summary.to_csv(
    ART /
    "deepbeat_lodo_subject_summary.csv",
    index=False
)


# ============================================================
# 4. Build non-DeepBeat reference population
# ============================================================

manifest = pd.read_csv(
    MANIFEST,
    low_memory=False
)


reference = manifest[
    manifest["domain"]
    .astype(str)
    !=
    "DEEPBEAT"
].copy()


reference = reference.loc[
    valid_mask(
        reference
    )
].copy()


reference[
    "label_eval"
] = (
    pd.to_numeric(
        reference[
            "label_v7"
        ],
        errors="raise"
    )
    .astype(int)
)


print(
    "\nReference rows:",
    len(reference)
)

print(
    reference[
        "domain"
    ].value_counts()
)


# ============================================================
# 5. Feature domain shift:
#    DeepBeat vs non-DeepBeat, stratified by class
# ============================================================

shift_rows = []


for label in [
    0,
    1
]:

    db = deepbeat[
        deepbeat[
            "label_eval"
        ]
        ==
        label
    ]


    ref = reference[
        reference[
            "label_eval"
        ]
        ==
        label
    ]


    for feature in FEATURES:

        a = (
            pd.to_numeric(
                db[
                    feature
                ],
                errors="coerce"
            )
            .dropna()
            .to_numpy()
        )

        b = (
            pd.to_numeric(
                ref[
                    feature
                ],
                errors="coerce"
            )
            .dropna()
            .to_numpy()
        )


        sa = robust_stats(
            a
        )

        sb = robust_stats(
            b
        )


        pooled_iqr = (
            (
                sa["iqr"]
                +
                sb["iqr"]
            )
            /
            2
        )


        robust_shift = (
            (
                sa["median"]
                -
                sb["median"]
            )
            /
            (
                pooled_iqr
                +
                1e-12
            )
        )


        ks = ks_2samp(
            a,
            b,
            alternative="two-sided",
            method="auto"
        )


        shift_rows.append({
            "label":
                label,

            "feature":
                feature,

            "deepbeat_n":
                len(a),

            "reference_n":
                len(b),

            "deepbeat_median":
                sa["median"],

            "deepbeat_iqr":
                sa["iqr"],

            "reference_median":
                sb["median"],

            "reference_iqr":
                sb["iqr"],

            "robust_median_shift":
                robust_shift,

            "abs_robust_median_shift":
                abs(
                    robust_shift
                ),

            "KS_statistic":
                float(
                    ks.statistic
                ),

            "KS_pvalue":
                float(
                    ks.pvalue
                ),
        })


feature_shift = pd.DataFrame(
    shift_rows
)


feature_shift = (
    feature_shift
    .sort_values(
        [
            "label",
            "abs_robust_median_shift"
        ],
        ascending=[
            True,
            False
        ]
    )
)


feature_shift.to_csv(
    ART /
    "deepbeat_lodo_feature_domain_shift.csv",
    index=False
)


# ============================================================
# 6. DeepBeat AF: FN phenotype vs TP phenotype
# ============================================================

tp = deepbeat[
    deepbeat[
        "error_type"
    ]
    ==
    "TP"
]


fn = deepbeat[
    deepbeat[
        "error_type"
    ]
    ==
    "FN"
]


fn_rows = []


if (
    len(tp)
    and
    len(fn)
):

    for feature in FEATURES:

        a = (
            pd.to_numeric(
                tp[
                    feature
                ],
                errors="coerce"
            )
            .dropna()
            .to_numpy()
        )

        b = (
            pd.to_numeric(
                fn[
                    feature
                ],
                errors="coerce"
            )
            .dropna()
            .to_numpy()
        )


        stp = robust_stats(
            a
        )

        sfn = robust_stats(
            b
        )


        pooled_iqr = (
            (
                stp["iqr"]
                +
                sfn["iqr"]
            )
            /
            2
        )


        shift = (
            (
                sfn["median"]
                -
                stp["median"]
            )
            /
            (
                pooled_iqr
                +
                1e-12
            )
        )


        ks = ks_2samp(
            a,
            b
        )


        fn_rows.append({
            "feature":
                feature,

            "TP_n":
                len(a),

            "FN_n":
                len(b),

            "TP_median":
                stp["median"],

            "FN_median":
                sfn["median"],

            "FN_minus_TP_robust_shift":
                shift,

            "abs_shift":
                abs(
                    shift
                ),

            "KS_statistic":
                float(
                    ks.statistic
                ),

            "KS_pvalue":
                float(
                    ks.pvalue
                ),
        })


fn_vs_tp = pd.DataFrame(
    fn_rows
)


fn_vs_tp = (
    fn_vs_tp
    .sort_values(
        "abs_shift",
        ascending=False
    )
)


fn_vs_tp.to_csv(
    ART /
    "deepbeat_lodo_fn_vs_tp_features.csv",
    index=False
)


# ============================================================
# 7. Diagnostic threshold trade-off
#
# IMPORTANT:
# This is NOT model selection.
# Do not tune the final model from this table.
# ============================================================

threshold_rows = []


for threshold in [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]:

    m = safe_metrics(
        deepbeat[
            "label_eval"
        ],
        deepbeat[
            "stack_probability"
        ],
        threshold=threshold
    )


    threshold_rows.append({
        "threshold":
            threshold,

        **m,
    })


threshold_summary = pd.DataFrame(
    threshold_rows
)


threshold_summary.to_csv(
    ART /
    "deepbeat_lodo_threshold_diagnostic_ONLY.csv",
    index=False
)


# ============================================================
# 8. Optional source/rhythm audits
# ============================================================

for column in [
    "rhythm",
    "source"
]:

    if column not in deepbeat.columns:
        continue


    rows = []


    for value, g in deepbeat.groupby(
        column,
        dropna=False
    ):

        m = safe_metrics(
            g[
                "label_eval"
            ],
            g[
                "stack_probability"
            ]
        )


        rows.append({
            column:
                value,

            "subjects":
                g[
                    "subject_eval"
                ].nunique(),

            **m,
        })


    pd.DataFrame(
        rows
    ).to_csv(
        ART /
        f"deepbeat_lodo_by_{column}.csv",
        index=False
    )


# ============================================================
# Print concise forensic report
# ============================================================

print(
    "\n" +
    "=" * 100
)

print(
    "DEEPBEAT LODO — SPLIT PERFORMANCE"
)

print(
    split_summary.to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "BASE MODELS VS STACK"
)

print(
    model_summary.to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "WORST AF SUBJECTS"
)

af_subjects = (
    subject_summary[
        subject_summary[
            "AF_windows"
        ]
        >
        0
    ]
    .sort_values(
        [
            "AF_sensitivity",
            "AF_mean_probability"
        ]
    )
)


print(
    af_subjects[
        [
            "subject_id",
            "split",
            "AF_windows",
            "TP",
            "FN",
            "AF_sensitivity",
            "AF_mean_probability",
            "AF_median_probability",
        ]
    ]
    .head(30)
    .to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "TOP DOMAIN-SHIFT FEATURES — NON-AF"
)

print(
    feature_shift[
        feature_shift[
            "label"
        ]
        ==
        0
    ][
        [
            "feature",
            "deepbeat_median",
            "reference_median",
            "robust_median_shift",
            "KS_statistic",
        ]
    ]
    .head(14)
    .to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "TOP DOMAIN-SHIFT FEATURES — AF"
)

print(
    feature_shift[
        feature_shift[
            "label"
        ]
        ==
        1
    ][
        [
            "feature",
            "deepbeat_median",
            "reference_median",
            "robust_median_shift",
            "KS_statistic",
        ]
    ]
    .head(14)
    .to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "AF FALSE-NEGATIVE PHENOTYPE"
)

print(
    fn_vs_tp[
        [
            "feature",
            "TP_median",
            "FN_median",
            "FN_minus_TP_robust_shift",
            "KS_statistic",
        ]
    ]
    .head(14)
    .to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "THRESHOLD DIAGNOSTIC ONLY — DO NOT TUNE"
)

print(
    threshold_summary[
        [
            "threshold",
            "sensitivity",
            "specificity",
            "TP",
            "TN",
            "FP",
            "FN",
        ]
    ]
    .to_string(
        index=False
    )
)
