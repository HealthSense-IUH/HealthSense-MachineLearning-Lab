import sys
sys.path.insert(0, "src")

from pathlib import Path
import json
import hashlib

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)


# ============================================================
# CONFIG
# ============================================================

MIMIC = Path(
    "data/features/v6/"
    "mimic_perform_v6_14f.csv"
)

DEEP_TRAIN = Path(
    "data/features/v6/"
    "deepbeat_train_v6_14f.csv"
)

DEEP_VAL = Path(
    "data/features/v6/"
    "deepbeat_validate_v6_14f.csv"
)

DEEP_TEST = Path(
    "data/features/v6/"
    "deepbeat_test_independent_v6_14f.csv"
)

V5_MODEL = Path(
    "models/mimic/"
    "healthsense_af_v5_rf_ac_frozen.pkl"
)

OUT_MODEL = Path(
    "models/multidomain/"
    "healthsense_af_v6a_rf_ac.pkl"
)

OUT_META = Path(
    "models/multidomain/"
    "healthsense_af_v6a_rf_ac.json"
)

OUT_DIR = Path(
    "experiments/07_v6_multidomain/artifacts"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_MODEL.parent.mkdir(
    parents=True,
    exist_ok=True,
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
# LOAD TRAINING DOMAINS
# ============================================================

mimic = pd.read_csv(MIMIC)

mimic = mimic.rename(
    columns={
        "status": "label",
    }
)

mimic["domain"] = "MIMIC_PERFORM"

mimic["subject_key"] = (
    "MIMIC:"
    + mimic["record_id"].astype(str)
)


deep = pd.read_csv(DEEP_TRAIN)

deep = deep[
    deep["feature_ok"] == True
].copy()

deep["domain"] = "DEEPBEAT"

deep["subject_key"] = (
    "DEEPBEAT:"
    + deep["subject_id"]
      .astype(int)
      .astype(str)
)


train = pd.concat(
    [
        mimic[
            [
                "domain",
                "subject_key",
                "label",
                *FEATURES,
            ]
        ],
        deep[
            [
                "domain",
                "subject_key",
                "label",
                *FEATURES,
            ]
        ],
    ],
    ignore_index=True,
)


# ============================================================
# AUDIT
# ============================================================

print("=" * 100)
print("HEALTHSENSE AF V6-A — MULTIDOMAIN TRAINING")
print("=" * 100)

print("\nFeature schema:")
for i, f in enumerate(
    FEATURES,
    start=1,
):
    print(f"{i:02d}. {f}")

x = train[
    FEATURES
].to_numpy(dtype=float)

if not np.isfinite(x).all():
    raise RuntimeError(
        "Training data contains non-finite features"
    )

print("\nTRAINING DATA")

summary = (
    train.groupby(
        ["domain", "label"]
    )
    .agg(
        windows=("label", "size"),
        subjects=("subject_key", "nunique"),
    )
)

print(summary.to_string())

print(
    "\nTotal windows :",
    len(train)
)

print(
    "Total subjects:",
    train.subject_key.nunique()
)


# ============================================================
# SAMPLE WEIGHTS
#
# Equal total contribution:
#
#   MIMIC        = 50%
#   DeepBeat     = 50%
#
# Inside each domain:
#
#   NON_AF       = 50%
#   AF           = 50%
#
# Inside each domain/class:
#
#   every subject contributes equally.
#
# Inside each subject:
#
#   every window divides that subject's weight.
#
# ============================================================

train["sample_weight"] = 0.0

domains = sorted(
    train.domain.unique()
)

for domain in domains:

    dmask = (
        train.domain == domain
    )

    labels = sorted(
        train.loc[
            dmask,
            "label"
        ].unique()
    )

    domain_weight = (
        1.0 / len(domains)
    )

    class_weight = (
        domain_weight
        / len(labels)
    )

    for label in labels:

        cell_mask = (
            dmask
            & (
                train.label == label
            )
        )

        subjects = sorted(
            train.loc[
                cell_mask,
                "subject_key"
            ].unique()
        )

        subject_weight = (
            class_weight
            / len(subjects)
        )

        for subject in subjects:

            smask = (
                cell_mask
                & (
                    train.subject_key
                    == subject
                )
            )

            n_windows = int(
                smask.sum()
            )

            train.loc[
                smask,
                "sample_weight"
            ] = (
                subject_weight
                / n_windows
            )


# Normalize mean weight to 1
train["sample_weight"] *= (
    len(train)
    / train.sample_weight.sum()
)


print("\nWEIGHT AUDIT")

weight_domain = (
    train.groupby("domain")
    .sample_weight
    .sum()
)

print("\nBy domain:")
print(
    (
        weight_domain
        / weight_domain.sum()
    ).to_string()
)

weight_class = (
    train.groupby(
        ["domain", "label"]
    )
    .sample_weight
    .sum()
)

print("\nBy domain/class:")
print(
    (
        weight_class
        / train.sample_weight.sum()
    ).to_string()
)


# ============================================================
# MODEL
#
# SAME MODEL ARCHITECTURE AS FROZEN V5
# ============================================================

model = Pipeline([
    (
        "scaler",
        StandardScaler(),
    ),
    (
        "clf",
        RandomForestClassifier(
            max_depth=10,
            n_jobs=-1,
            random_state=42,
        ),
    ),
])


X_train = train[
    FEATURES
]

y_train = (
    train["label"]
    .astype(int)
)

weights = (
    train["sample_weight"]
    .to_numpy(dtype=float)
)


print("\nTraining V6-A...")

model.fit(
    X_train,
    y_train,
    clf__sample_weight=weights,
)

print("Training complete.")


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    OUT_MODEL,
)


# ============================================================
# EVALUATION HELPERS
# ============================================================

def metrics_at_05(
    y,
    p,
):
    pred = (
        p >= 0.5
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            pred,
            labels=[0, 1],
        )
        .ravel()
    )

    sensitivity = (
        tp / (tp + fn)
        if tp + fn
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else np.nan
    )

    return {
        "sensitivity_05":
            sensitivity,

        "specificity_05":
            specificity,
    }


def evaluate(
    model,
    df,
    split_name,
    model_name,
):

    d = df.copy()

    if "feature_ok" in d.columns:
        d = d[
            d.feature_ok == True
        ].copy()

    y = (
        d.label
        .astype(int)
        .to_numpy()
    )

    p = model.predict_proba(
        d[FEATURES]
    )[:, 1]

    auc = roc_auc_score(
        y,
        p,
    )

    ap = average_precision_score(
        y,
        p,
    )

    brier = brier_score_loss(
        y,
        p,
    )

    threshold_metrics = (
        metrics_at_05(
            y,
            p,
        )
    )

    print()
    print("-" * 100)
    print(
        model_name,
        "|",
        split_name
    )
    print("-" * 100)

    print(
        f"windows       : {len(d):,}"
    )

    print(
        f"AUROC         : {auc:.6f}"
    )

    print(
        f"PR-AUC        : {ap:.6f}"
    )

    print(
        f"Brier         : {brier:.6f}"
    )

    print(
        "Sensitivity@.5:",
        f"{threshold_metrics['sensitivity_05']:.6f}"
    )

    print(
        "Specificity@.5:",
        f"{threshold_metrics['specificity_05']:.6f}"
    )


    # ----------------------------------------
    # SUBJECT AGGREGATION
    # ----------------------------------------

    d["probability"] = p

    subject = (
        d.groupby("subject_id")
        .agg(
            label=("label", "first"),
            probability_mean=(
                "probability",
                "mean",
            ),
            probability_median=(
                "probability",
                "median",
            ),
            windows=(
                "probability",
                "size",
            ),
        )
        .reset_index()
    )

    subject_auc_mean = (
        roc_auc_score(
            subject.label,
            subject.probability_mean,
        )
        if subject.label.nunique() == 2
        else np.nan
    )

    subject_auc_median = (
        roc_auc_score(
            subject.label,
            subject.probability_median,
        )
        if subject.label.nunique() == 2
        else np.nan
    )

    print(
        "subjects      :",
        len(subject)
    )

    print(
        "subject AUC μ :",
        f"{subject_auc_mean:.6f}"
    )

    print(
        "subject AUC med:",
        f"{subject_auc_median:.6f}"
    )

    pred_out = d[
        [
            "global_index",
            "subject_id",
            "label",
            "rhythm",
            "quality",
            "quality_name",
        ]
    ].copy()

    pred_out["model"] = (
        model_name
    )

    pred_out["probability"] = p

    pred_out.to_csv(
        OUT_DIR /
        (
            f"{split_name}_"
            f"{model_name}_predictions.csv"
        ),
        index=False,
    )

    subject["model"] = (
        model_name
    )

    subject.to_csv(
        OUT_DIR /
        (
            f"{split_name}_"
            f"{model_name}_subject.csv"
        ),
        index=False,
    )

    return {
        "model":
            model_name,

        "split":
            split_name,

        "windows":
            len(d),

        "subjects":
            len(subject),

        "auroc":
            auc,

        "pr_auc":
            ap,

        "brier":
            brier,

        "sensitivity_05":
            threshold_metrics[
                "sensitivity_05"
            ],

        "specificity_05":
            threshold_metrics[
                "specificity_05"
            ],

        "subject_auc_mean":
            subject_auc_mean,

        "subject_auc_median":
            subject_auc_median,
    }


# ============================================================
# V5 VS V6-A
#
# Validation is used for development.
# Independent test is reported but NOT used to tune.
# ============================================================

val = pd.read_csv(
    DEEP_VAL
)

test = pd.read_csv(
    DEEP_TEST
)

v5 = joblib.load(
    V5_MODEL
)

results = []

for split_name, data in [
    (
        "deepbeat_validate",
        val,
    ),
    (
        "deepbeat_test_independent",
        test,
    ),
]:

    results.append(
        evaluate(
            v5,
            data,
            split_name,
            "v5",
        )
    )

    results.append(
        evaluate(
            model,
            data,
            split_name,
            "v6a",
        )
    )


result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    OUT_DIR /
    "v6a_vs_v5_classifier_summary.csv",
    index=False,
)


print()
print("=" * 100)
print("V5 VS V6-A SUMMARY")
print("=" * 100)

print(
    result_df.to_string(
        index=False
    )
)


# ============================================================
# METADATA
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


metadata = {
    "model_name":
        "HealthSense AF V6-A",

    "purpose":
        (
            "Controlled multidomain retraining "
            "using the frozen V5 feature schema "
            "and RF architecture."
        ),

    "features":
        FEATURES,

    "training_domains": {
        "MIMIC_PERFORM":
            {
                "windows":
                    int(
                        (
                            train.domain
                            == "MIMIC_PERFORM"
                        ).sum()
                    ),

                "subjects":
                    int(
                        train.loc[
                            train.domain
                            == "MIMIC_PERFORM",
                            "subject_key",
                        ].nunique()
                    ),
            },

        "DEEPBEAT":
            {
                "windows":
                    int(
                        (
                            train.domain
                            == "DEEPBEAT"
                        ).sum()
                    ),

                "subjects":
                    int(
                        train.loc[
                            train.domain
                            == "DEEPBEAT",
                            "subject_key",
                        ].nunique()
                    ),
            },
    },

    "weight_policy":
        (
            "Equal domain -> equal class "
            "within domain -> equal subject "
            "within domain/class -> equal "
            "window within subject."
        ),

    "estimator": {
        "type":
            "RandomForestClassifier",

        "max_depth":
            10,

        "random_state":
            42,

        "n_estimators":
            100,
    },

    "model_sha256":
        sha256(
            OUT_MODEL
        ),
}

with open(
    OUT_META,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )


print()
print("Saved model :", OUT_MODEL)
print("Saved meta  :", OUT_META)
print(
    "Model SHA256:",
    metadata[
        "model_sha256"
    ]
)
