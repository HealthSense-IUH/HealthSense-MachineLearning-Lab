import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    LeaveOneGroupOut,
    GroupKFold,
    GridSearchCV,
)

from healthsense_ml import config
from healthsense_ml.feature_extraction import load_features
from healthsense_ml.training import (
    build_model_registry,
    iqr_train_mask,
)
from healthsense_ml import evaluation


df = load_features()

FEATURES = config.CORE_FEATURES

X = df[FEATURES]
y = df["status"].to_numpy()
groups = df["record_id"].to_numpy()

logo = LeaveOneGroupOut()


# ---------------------------------------------------------
# 1. Audit how many training rows IQR removes
# ---------------------------------------------------------

retention_rows = []

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups),
    1,
):

    X_train = X.iloc[train_idx]
    y_train = y[train_idx]

    keep = iqr_train_mask(X_train)

    for label in [0, 1]:

        label_mask = y_train == label

        total = int(label_mask.sum())

        kept = int(
            (
                keep.to_numpy()
                & label_mask
            ).sum()
        )

        retention_rows.append({
            "fold": fold,
            "test_subject":
                groups[test_idx][0],

            "label":
                "AF" if label else "Non-AF",

            "total": total,
            "kept": kept,
            "removed": total - kept,

            "retention":
                kept / total
                if total else np.nan,
        })


retention = pd.DataFrame(
    retention_rows
)

retention.to_csv(
    "experiments/04_feature_model/"
    "iqr_retention.csv",
    index=False,
)


print("=" * 90)
print("IQR RETENTION")
print("=" * 90)

print(
    retention.groupby("label")
    .agg(
        median_retention=(
            "retention",
            "median",
        ),
        min_retention=(
            "retention",
            "min",
        ),
        median_removed=(
            "removed",
            "median",
        ),
    )
    .round(4)
)


# ---------------------------------------------------------
# 2. LOSO benchmark:
#    current IQR vs no IQR
# ---------------------------------------------------------

def run_variant(use_iqr):

    registry = build_model_registry()

    predictions = []

    for model_name, spec in registry.items():

        print()
        print(
            model_name,
            "|",
            "IQR" if use_iqr else "NO-IQR"
        )

        for fold, (
            train_idx,
            test_idx,
        ) in enumerate(
            logo.split(X, y, groups),
            1,
        ):

            X_tr = X.iloc[train_idx]
            y_tr = y[train_idx]
            g_tr = groups[train_idx]

            X_te = X.iloc[test_idx]
            y_te = y[test_idx]

            if use_iqr:

                keep = iqr_train_mask(
                    X_tr
                )

                X_fit = X_tr[keep]

                y_fit = y_tr[
                    keep.to_numpy()
                ]

                g_fit = g_tr[
                    keep.to_numpy()
                ]

            else:

                X_fit = X_tr
                y_fit = y_tr
                g_fit = g_tr

            search = GridSearchCV(
                spec["pipeline"],
                spec["param_grid"],

                cv=GroupKFold(
                    config.INNER_CV_FOLDS
                ),

                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
            )

            search.fit(
                X_fit,
                y_fit,
                groups=g_fit,
            )

            probs = search.predict_proba(
                X_te
            )[:, 1]

            test_subject = (
                groups[test_idx][0]
            )

            for prob, truth in zip(
                probs,
                y_te,
            ):

                predictions.append({
                    "record_id":
                        test_subject,

                    "status":
                        int(truth),

                    "model":
                        model_name,

                    "prob":
                        float(prob),

                    "pred":
                        int(prob >= 0.5),
                })

    pred = pd.DataFrame(
        predictions
    )

    summary = evaluation.summarize(
        pred
    )

    summary["IQR"] = (
        "IQR"
        if use_iqr
        else "NO-IQR"
    )

    return pred, summary


all_summary = []

for use_iqr in [True, False]:

    pred, summary = run_variant(
        use_iqr
    )

    tag = (
        "iqr"
        if use_iqr
        else "no_iqr"
    )

    pred.to_csv(
        f"experiments/04_feature_model/"
        f"predictions_{tag}.csv",
        index=False,
    )

    all_summary.append(
        summary
    )


result = pd.concat(
    all_summary,
    ignore_index=True,
)

result.to_csv(
    "experiments/04_feature_model/"
    "iqr_ablation_results.csv",
    index=False,
)


print()
print("=" * 100)
print("IQR vs NO-IQR")
print("=" * 100)

cols = [
    "IQR",
    "Model",
    "Level",
    "Accuracy",
    "Recall (Sensitivity)",
    "Specificity",
    "F1-Score",
    "ROC-AUC",
    "FN",
    "FP",
]

print(
    result[cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)
