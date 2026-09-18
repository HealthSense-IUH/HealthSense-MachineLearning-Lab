from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


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

MODEL = (
    ROOT /
    "models/multidomain/"
    "healthsense_af_v6c_stacking.pkl"
)


# ============================================================
# Frozen architecture
# ============================================================

bundle = joblib.load(
    MODEL
)

BASE_TEMPLATE = (
    bundle["base_models"]
)

META_TEMPLATE = (
    bundle["meta"]
)

FEATURES_14 = list(
    bundle["features"]
)

FEATURES_13 = [
    f
    for f in FEATURES_14
    if f != "PPG_AC"
]


print("14f:", FEATURES_14)
print("13f:", FEATURES_13)


# ============================================================
# Helpers
# ============================================================

def common_valid_mask(
    df
):
    """
    Build ONE common cohort using all 14 features.

    Therefore 13F and 14F are trained/evaluated
    on exactly the same rows.
    """

    mask = (
        df[FEATURES_14]
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
                "yes",
            ])
        )

        mask &= ok


    return mask


def metrics(
    y,
    p
):

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
    }


def load_dev_sets():

    out = {}


    # ========================================================
    # DeepBeat DEV
    # ========================================================

    x = pd.read_csv(
        ROOT /
        "data/features/v6/"
        "deepbeat_validate_v6_14f.csv",
        low_memory=False
    )


    x["label_eval"] = (
        pd.to_numeric(
            x["label"],
            errors="raise"
        )
        .astype(int)
    )


    x["subject_eval"] = (
        x["subject_id"]
        .astype(str)
    )


    out["deepbeat"] = x


    # ========================================================
    # PulseWatch DEV
    # ========================================================

    x = pd.read_csv(
        ROOT /
        "experiments/07_v6_multidomain/"
        "artifacts/hard_domain_splits/"
        "pulsewatch_dev.csv",
        low_memory=False
    )


    if "class_name" in x.columns:

        x["label_eval"] = (
            x["class_name"]
            .astype(str)
            .str.upper()
            .eq("AF")
            .astype(int)
        )

    else:

        x["label_eval"] = (
            pd.to_numeric(
                x["label"],
                errors="raise"
            )
            .eq(1)
            .astype(int)
        )


    x["subject_eval"] = (
        x["uid"]
        .astype(str)
    )


    out["pulsewatch"] = x


    # ========================================================
    # Liu DEV
    # ========================================================

    x = pd.read_csv(
        ROOT /
        "experiments/07_v6_multidomain/"
        "artifacts/hard_domain_splits/"
        "liu_dev.csv",
        low_memory=False
    )


    if "binary_label_clean" in x.columns:

        x["label_eval"] = (
            pd.to_numeric(
                x[
                    "binary_label_clean"
                ],
                errors="raise"
            )
            .astype(int)
        )

    elif "binary_label" in x.columns:

        x["label_eval"] = (
            pd.to_numeric(
                x["binary_label"],
                errors="raise"
            )
            .astype(int)
        )

    else:

        x["label_eval"] = (
            x["rhythm"]
            .astype(str)
            .str.upper()
            .eq("AF")
            .astype(int)
        )


    x["subject_eval"] = (
        x["subject_id"]
        .astype(str)
    )


    out["liu"] = x


    return out


# ============================================================
# Load training manifest
# ============================================================

train_raw = pd.read_csv(
    MANIFEST,
    low_memory=False
)


print(
    "\nRaw training rows:",
    len(train_raw)
)


# ============================================================
# Diagnose invalid rows
# ============================================================

valid_mask = common_valid_mask(
    train_raw
)


invalid = (
    train_raw
    .loc[
        ~valid_mask
    ]
    .copy()
)


print(
    "Common-valid rows:",
    int(
        valid_mask.sum()
    )
)

print(
    "Dropped rows:",
    len(invalid)
)


if len(invalid):

    diagnostic_cols = [
        c
        for c in [
            "domain",
            "subject_id",
            "class_name",
            "label_v7",
        ]
        if c in invalid.columns
    ]


    print(
        "\nDropped-row metadata:"
    )

    print(
        invalid[
            diagnostic_cols
        ].to_string(
            index=False
        )
    )


    print(
        "\nInvalid feature(s):"
    )


    for idx, row in invalid.iterrows():

        bad = []


        for feature in FEATURES_14:

            value = pd.to_numeric(
                pd.Series(
                    [row[feature]]
                ),
                errors="coerce"
            ).iloc[0]


            if not np.isfinite(value):

                bad.append(
                    feature
                )


        print(
            "row",
            idx,
            bad
        )


invalid.to_csv(
    ART /
    "ppg_ac_ablation_dropped_training_rows.csv",
    index=False
)


# ============================================================
# IMPORTANT:
# filter ONCE before y/groups/folds.
# ============================================================

train = (
    train_raw
    .loc[
        valid_mask
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


train["domain"] = (
    train["domain"]
    .astype(str)
)

train["subject_id"] = (
    train["subject_id"]
    .astype(str)
)


train["group_id"] = (
    train["domain"]
    +
    ":"
    +
    train["subject_id"]
)


y = (
    pd.to_numeric(
        train["label_v7"],
        errors="raise"
    )
    .astype(int)
    .to_numpy()
)


groups = (
    train["group_id"]
    .to_numpy()
)


print(
    "\nMatched training rows:",
    len(train)
)

print(
    "Matched groups:",
    train[
        "group_id"
    ].nunique()
)

print(
    "Labels:",
    pd.Series(
        y
    )
    .value_counts()
    .to_dict()
)


print(
    "\nDomain counts:"
)

print(
    train["domain"]
    .value_counts()
)


# ============================================================
# SAME folds for 13F and 14F
# ============================================================

gkf = GroupKFold(
    n_splits=5
)


splits = list(
    gkf.split(
        train,
        y,
        groups
    )
)


print(
    "\nGroupKFold prepared:"
)


for i, (
    tr,
    va
) in enumerate(
    splits,
    start=1
):

    print(
        f"fold {i}: "
        f"train={len(tr)} "
        f"val={len(va)} "
        f"train_groups="
        f"{len(set(groups[tr]))} "
        f"val_groups="
        f"{len(set(groups[va]))}"
    )


# ============================================================
# Build ONE matched DEV cohort per dataset
# ============================================================

dev_raw = load_dev_sets()

dev = {}


for dataset, df in dev_raw.items():

    mask = common_valid_mask(
        df
    )


    dropped = int(
        (~mask).sum()
    )


    matched = (
        df.loc[
            mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    dev[
        dataset
    ] = matched


    print(
        "\nDEV",
        dataset,
        "raw=",
        len(df),
        "matched=",
        len(matched),
        "dropped=",
        dropped,
        "subjects=",
        matched[
            "subject_eval"
        ].nunique(),
        "labels=",
        matched[
            "label_eval"
        ]
        .value_counts()
        .to_dict()
    )


# ============================================================
# Matched ablation
# ============================================================

summary_rows = []
prediction_parts = []


for variant, features in [
    (
        "13F_NO_PPG_AC",
        FEATURES_13
    ),
    (
        "14F_WITH_PPG_AC",
        FEATURES_14
    ),
]:

    print(
        "\n" +
        "=" * 100
    )

    print(
        "VARIANT:",
        variant
    )


    X = train[
        features
    ]


    base_names = list(
        BASE_TEMPLATE.keys()
    )


    # --------------------------------------------------------
    # OOF base probabilities
    # --------------------------------------------------------

    oof = np.zeros(
        (
            len(train),
            len(base_names)
        ),
        dtype=float
    )


    start = time.time()


    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        splits,
        start=1
    ):

        print(
            f"\n{variant} "
            f"fold {fold}/5"
        )


        for j, name in enumerate(
            base_names
        ):

            print(
                "  fitting",
                name
            )


            model = clone(
                BASE_TEMPLATE[
                    name
                ]
            )


            model.fit(
                X.iloc[
                    train_idx
                ],
                y[
                    train_idx
                ]
            )


            oof[
                val_idx,
                j
            ] = (
                model.predict_proba(
                    X.iloc[
                        val_idx
                    ]
                )[:, 1]
            )


    if not np.isfinite(
        oof
    ).all():

        raise RuntimeError(
            f"{variant}: invalid OOF prediction"
        )


    # --------------------------------------------------------
    # Fit meta learner on same OOF structure
    # --------------------------------------------------------

    meta = clone(
        META_TEMPLATE
    )


    meta.fit(
        oof,
        y
    )


    oof_stack = (
        meta.predict_proba(
            oof
        )[:, 1]
    )


    print(
        "\nOOF:",
        metrics(
            y,
            oof_stack
        )
    )


    # --------------------------------------------------------
    # Final base learners on ALL matched training data
    # --------------------------------------------------------

    final_models = {}


    for name in base_names:

        print(
            "final fit",
            variant,
            name
        )


        model = clone(
            BASE_TEMPLATE[
                name
            ]
        )


        model.fit(
            X,
            y
        )


        final_models[
            name
        ] = model


    # --------------------------------------------------------
    # Evaluate same matched DEV rows
    # --------------------------------------------------------

    for dataset, d in dev.items():

        Xd = d[
            features
        ]


        base_probability = (
            np.column_stack([
                final_models[
                    name
                ]
                .predict_proba(
                    Xd
                )[:, 1]

                for name
                in base_names
            ])
        )


        probability = (
            meta.predict_proba(
                base_probability
            )[:, 1]
        )


        yd = (
            d["label_eval"]
            .astype(int)
            .to_numpy()
        )


        m = metrics(
            yd,
            probability
        )


        summary_rows.append({
            "variant":
                variant,

            "dataset":
                dataset,

            "n_features":
                len(features),

            "rows":
                len(d),

            "subjects":
                d[
                    "subject_eval"
                ].nunique(),

            **m,
        })


        pred = pd.DataFrame({
            "variant":
                variant,

            "dataset":
                dataset,

            "subject_id":
                d[
                    "subject_eval"
                ].astype(str),

            "label":
                yd,

            "probability":
                probability,
        })


        prediction_parts.append(
            pred
        )


    print(
        "\nElapsed:",
        time.time()
        -
        start,
        "seconds"
    )


# ============================================================
# Save
# ============================================================

summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    ART /
    "ppg_ac_ablation_dev_summary.csv",
    index=False
)


predictions = pd.concat(
    prediction_parts,
    ignore_index=True
)


predictions.to_csv(
    ART /
    "ppg_ac_ablation_dev_predictions.csv",
    index=False
)


# ============================================================
# Delta:
#
# AUROC:
#   positive = 14F better
#
# PR-AUC:
#   positive = 14F better
#
# Brier:
#   positive = 14F lower/better
# ============================================================

s13 = (
    summary[
        summary["variant"]
        ==
        "13F_NO_PPG_AC"
    ]
    .set_index(
        "dataset"
    )
)


s14 = (
    summary[
        summary["variant"]
        ==
        "14F_WITH_PPG_AC"
    ]
    .set_index(
        "dataset"
    )
)


delta_rows = []


for dataset in sorted(
    set(
        s13.index
    )
    &
    set(
        s14.index
    )
):

    delta_rows.append({
        "dataset":
            dataset,

        "delta_AUROC_14_minus_13":
            (
                s14.loc[
                    dataset,
                    "AUROC"
                ]
                -
                s13.loc[
                    dataset,
                    "AUROC"
                ]
            ),

        "delta_PR_AUC_14_minus_13":
            (
                s14.loc[
                    dataset,
                    "PR_AUC"
                ]
                -
                s13.loc[
                    dataset,
                    "PR_AUC"
                ]
            ),

        "brier_improvement_14_vs_13":
            (
                s13.loc[
                    dataset,
                    "Brier"
                ]
                -
                s14.loc[
                    dataset,
                    "Brier"
                ]
            ),
    })


delta = pd.DataFrame(
    delta_rows
)


delta.to_csv(
    ART /
    "ppg_ac_ablation_dev_delta.csv",
    index=False
)


# ============================================================
# Macro comparison
# ============================================================

macro = (
    summary
    .groupby(
        "variant"
    )
    .agg(
        macro_AUROC=(
            "AUROC",
            "mean"
        ),
        macro_PR_AUC=(
            "PR_AUC",
            "mean"
        ),
        macro_Brier=(
            "Brier",
            "mean"
        ),
    )
    .reset_index()
)


macro.to_csv(
    ART /
    "ppg_ac_ablation_dev_macro.csv",
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
    "PPG_AC MATCHED ABLATION"
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
    "14F - 13F DELTA"
)

print(
    delta.to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "MACRO"
)

print(
    macro.to_string(
        index=False
    )
)
