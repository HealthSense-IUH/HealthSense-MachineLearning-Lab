from pathlib import Path
import argparse
import json
import hashlib
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
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

MODEL_PATH = (
    ROOT /
    "models/multidomain/"
    "healthsense_af_v6c_stacking.pkl"
)


VALID_DOMAINS = [
    "DEEPBEAT",
    "PULSEWATCH",
    "LIU",
    "MIMIC_PERFORM",
]


parser = argparse.ArgumentParser()

parser.add_argument(
    "--holdout",
    required=True,
    choices=VALID_DOMAINS
)

args = parser.parse_args()

HOLDOUT = args.holdout


# ============================================================
# Load frozen architecture
# ============================================================

bundle = joblib.load(
    MODEL_PATH
)


FEATURES = list(
    bundle["features"]
)

BASE_TEMPLATE = (
    bundle["base_models"]
)

META_TEMPLATE = (
    bundle["meta"]
)


print(
    "HOLDOUT:",
    HOLDOUT
)

print(
    "FEATURES:",
    FEATURES
)

print(
    "BASE MODELS:",
    list(
        BASE_TEMPLATE.keys()
    )
)

print(
    "META:",
    META_TEMPLATE
)


# ============================================================
# Helpers
# ============================================================

def clean_features(df):

    valid = (
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
            .isin(
                [
                    "1",
                    "true",
                    "yes"
                ]
            )
        )

        valid &= ok


    return (
        df.loc[
            valid
        ]
        .copy()
    )


def load_deepbeat_eval():

    files = [
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
    ]


    parts = []


    for split, f in files:

        x = pd.read_csv(
            f,
            low_memory=False
        )

        x = clean_features(
            x
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

        x["source_split"] = split

        parts.append(x)


    return pd.concat(
        parts,
        ignore_index=True
    )


def load_pulsewatch_eval():

    base = (
        ROOT /
        "experiments/07_v6_multidomain/"
        "artifacts/hard_domain_splits"
    )


    parts = []


    for split in [
        "train",
        "dev",
        "test"
    ]:

        f = (
            base /
            f"pulsewatch_{split}.csv"
        )

        x = pd.read_csv(
            f,
            low_memory=False
        )

        x = clean_features(
            x
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

        x["source_split"] = split

        parts.append(x)


    return pd.concat(
        parts,
        ignore_index=True
    )


def load_liu_eval():

    base = (
        ROOT /
        "experiments/07_v6_multidomain/"
        "artifacts/hard_domain_splits"
    )


    parts = []


    for split in [
        "train",
        "dev",
        "test"
    ]:

        f = (
            base /
            f"liu_{split}.csv"
        )

        x = pd.read_csv(
            f,
            low_memory=False
        )

        x = clean_features(
            x
        )


        if (
            "binary_label_clean"
            in x.columns
        ):

            x["label_eval"] = (
                pd.to_numeric(
                    x[
                        "binary_label_clean"
                    ],
                    errors="raise"
                )
                .astype(int)
            )

        elif (
            "binary_label"
            in x.columns
        ):

            x["label_eval"] = (
                pd.to_numeric(
                    x[
                        "binary_label"
                    ],
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

        x["source_split"] = split

        parts.append(x)


    return pd.concat(
        parts,
        ignore_index=True
    )


def load_mimic_eval():

    m = pd.read_csv(
        MANIFEST,
        low_memory=False
    )


    x = m[
        m["domain"]
        ==
        "MIMIC_PERFORM"
    ].copy()


    x = clean_features(
        x
    )


    x["label_eval"] = (
        pd.to_numeric(
            x["label_v7"],
            errors="raise"
        )
        .astype(int)
    )


    x["subject_eval"] = (
        x["subject_id"]
        .astype(str)
    )


    x["source_split"] = (
        "manifest"
    )


    return x


LOADERS = {
    "DEEPBEAT":
        load_deepbeat_eval,

    "PULSEWATCH":
        load_pulsewatch_eval,

    "LIU":
        load_liu_eval,

    "MIMIC_PERFORM":
        load_mimic_eval,
}


def evaluate(
    y,
    p
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
        p >= 0.5
    ).astype(int)


    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            pred,
            labels=[
                0,
                1
            ]
        ).ravel()
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
            int(tp),

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "sensitivity_at_0.5":
            (
                float(
                    tp /
                    (
                        tp + fn
                    )
                )
                if (
                    tp + fn
                )
                else np.nan
            ),

        "specificity_at_0.5":
            (
                float(
                    tn /
                    (
                        tn + fp
                    )
                )
                if (
                    tn + fp
                )
                else np.nan
            ),
    }


# ============================================================
# Training data
# ============================================================

manifest = pd.read_csv(
    MANIFEST,
    low_memory=False
)


manifest["domain"] = (
    manifest["domain"]
    .astype(str)
)


manifest["subject_id"] = (
    manifest["subject_id"]
    .astype(str)
)


train = manifest[
    manifest["domain"]
    !=
    HOLDOUT
].copy()


train = clean_features(
    train
)


train["group_id"] = (
    train["domain"]
    +
    ":"
    +
    train["subject_id"]
)


y_train = (
    pd.to_numeric(
        train["label_v7"],
        errors="raise"
    )
    .astype(int)
    .to_numpy()
)


X_train = (
    train[
        FEATURES
    ]
)


groups = (
    train["group_id"]
    .to_numpy()
)


print(
    "\nTRAIN DOMAIN COUNTS"
)

print(
    train["domain"]
    .value_counts()
)


print(
    "\nTRAIN SUBJECTS BY DOMAIN"
)

print(
    train.groupby(
        "domain"
    )[
        "subject_id"
    ].nunique()
)


print(
    "\nTRAIN TOTAL:",
    len(train)
)

print(
    "GROUPS:",
    train[
        "group_id"
    ].nunique()
)

print(
    "LABELS:",
    pd.Series(
        y_train
    )
    .value_counts()
    .to_dict()
)


# ============================================================
# Evaluation domain
# ============================================================

eval_df = (
    LOADERS[
        HOLDOUT
    ]()
)


X_eval = (
    eval_df[
        FEATURES
    ]
)


y_eval = (
    eval_df[
        "label_eval"
    ]
    .astype(int)
    .to_numpy()
)


print(
    "\nEVAL ROWS:",
    len(eval_df)
)

print(
    "EVAL SUBJECTS:",
    eval_df[
        "subject_eval"
    ].nunique()
)

print(
    "EVAL LABELS:",
    pd.Series(
        y_eval
    )
    .value_counts()
    .to_dict()
)


# ============================================================
# OOF predictions for meta learner
# ============================================================

model_names = list(
    BASE_TEMPLATE.keys()
)


oof = np.zeros(
    (
        len(train),
        len(model_names)
    ),
    dtype=float
)


n_splits = min(
    5,
    train[
        "group_id"
    ].nunique()
)


gkf = GroupKFold(
    n_splits=n_splits
)


start = time.time()


for fold, (
    tr_idx,
    va_idx
) in enumerate(
    gkf.split(
        X_train,
        y_train,
        groups
    ),
    start=1
):

    print(
        "\n" +
        "=" * 80
    )

    print(
        f"OOF FOLD {fold}/{n_splits}"
    )

    print(
        "train rows:",
        len(tr_idx),
        "val rows:",
        len(va_idx)
    )


    for j, name in enumerate(
        model_names
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
            X_train.iloc[
                tr_idx
            ],
            y_train[
                tr_idx
            ]
        )


        oof[
            va_idx,
            j
        ] = (
            model.predict_proba(
                X_train.iloc[
                    va_idx
                ]
            )[:, 1]
        )


if not np.isfinite(
    oof
).all():

    raise RuntimeError(
        "OOF contains invalid values"
    )


# ============================================================
# Meta model
# ============================================================

meta = clone(
    META_TEMPLATE
)


meta.fit(
    oof,
    y_train
)


oof_stack = (
    meta.predict_proba(
        oof
    )[:, 1]
)


print(
    "\nOOF STACK METRICS"
)

print(
    evaluate(
        y_train,
        oof_stack
    )
)


# ============================================================
# Final base models, trained on ALL remaining domains
# ============================================================

eval_base = np.zeros(
    (
        len(eval_df),
        len(model_names)
    ),
    dtype=float
)


for j, name in enumerate(
    model_names
):

    print(
        "\nFINAL FIT:",
        name
    )


    model = clone(
        BASE_TEMPLATE[
            name
        ]
    )


    model.fit(
        X_train,
        y_train
    )


    eval_base[
        :,
        j
    ] = (
        model.predict_proba(
            X_eval
        )[:, 1]
    )


eval_prob = (
    meta.predict_proba(
        eval_base
    )[:, 1]
)


metrics = evaluate(
    y_eval,
    eval_prob
)


elapsed = (
    time.time()
    -
    start
)


# ============================================================
# Save predictions
# ============================================================

pred = pd.DataFrame({
    "domain":
        HOLDOUT,

    "subject_id":
        eval_df[
            "subject_eval"
        ].astype(str),

    "source_split":
        eval_df[
            "source_split"
        ].astype(str),

    "label":
        y_eval,

    "stack_probability":
        eval_prob,
})


for j, name in enumerate(
    model_names
):

    pred[
        f"{name}_prob"
    ] = eval_base[
        :,
        j
    ]


pred.to_csv(
    ART /
    f"lodo_{HOLDOUT.lower()}_predictions.csv",
    index=False
)


summary = {
    "heldout_domain":
        HOLDOUT,

    "evaluation_type":
        (
            "RETROSPECTIVE_LEAVE_ONE_DOMAIN_OUT"
        ),

    "heldout_domain_used_for_training":
        False,

    "heldout_domain_used_for_meta_training":
        False,

    "group_definition":
        "domain:subject_id",

    "n_splits":
        n_splits,

    "train_rows":
        len(train),

    "train_groups":
        int(
            train[
                "group_id"
            ].nunique()
        ),

    "eval_rows":
        len(eval_df),

    "eval_subjects":
        int(
            eval_df[
                "subject_eval"
            ].nunique()
        ),

    "elapsed_seconds":
        elapsed,

    **metrics,
}


summary_path = (
    ART /
    f"lodo_{HOLDOUT.lower()}_summary.json"
)


summary_path.write_text(
    json.dumps(
        summary,
        indent=2
    )
)


sha = hashlib.sha256(
    summary_path.read_bytes()
).hexdigest()


(
    ART /
    f"lodo_{HOLDOUT.lower()}_summary_SHA256.txt"
).write_text(
    sha
    + "  "
    + summary_path.name
    + "\n"
)


print(
    "\n" +
    "=" * 100
)

print(
    "LODO RESULT"
)

print(
    json.dumps(
        summary,
        indent=2
    )
)

print(
    "\nSHA256:",
    sha
)
