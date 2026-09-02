import sys
sys.path.insert(0, "src")

import pandas as pd

from sklearn.model_selection import (
    LeaveOneGroupOut,
    GroupKFold,
    GridSearchCV,
)

from healthsense_ml import config
from healthsense_ml.feature_extraction import load_features
from healthsense_ml.training import build_model_registry
from healthsense_ml import evaluation


df = load_features()

CURRENT13 = config.CORE_FEATURES

FREQ = [
    "HF",
    "Total_Power",
    "HF_norm",
]

ABLATIONS = {
    "current13":
        CURRENT13,

    "no_frequency":
        [
            f for f in CURRENT13
            if f not in FREQ
        ],

    "no_sampen":
        [
            f for f in CURRENT13
            if f != "SampEn"
        ],

    "no_frequency_no_sampen":
        [
            f for f in CURRENT13
            if f not in FREQ
            and f != "SampEn"
        ],
}


def run_loso(feature_cols):

    X = df[feature_cols]
    y = df["status"].to_numpy()
    groups = df["record_id"].to_numpy()

    logo = LeaveOneGroupOut()
    registry = build_model_registry()

    predictions = []

    for model_name, spec in registry.items():

        print()
        print(
            model_name,
            "|",
            len(feature_cols),
            "features"
        )

        for train_idx, test_idx in logo.split(
            X, y, groups
        ):

            X_tr = X.iloc[train_idx]
            y_tr = y[train_idx]
            g_tr = groups[train_idx]

            X_te = X.iloc[test_idx]
            y_te = y[test_idx]

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

            # NO IQR
            search.fit(
                X_tr,
                y_tr,
                groups=g_tr,
            )

            probs = search.predict_proba(
                X_te
            )[:, 1]

            record_id = groups[
                test_idx
            ][0]

            for p, truth in zip(
                probs,
                y_te,
            ):
                predictions.append({
                    "record_id":
                        record_id,
                    "status":
                        int(truth),
                    "model":
                        model_name,
                    "prob":
                        float(p),
                    "pred":
                        int(p >= 0.5),
                })

    return pd.DataFrame(predictions)


summaries = []

for name, features in ABLATIONS.items():

    print()
    print("=" * 90)
    print(name)
    print(features)
    print("=" * 90)

    pred = run_loso(features)

    pred.to_csv(
        f"experiments/04_feature_model/"
        f"pred_{name}.csv",
        index=False,
    )

    summary = evaluation.summarize(
        pred
    )

    summary["Feature_Set"] = name
    summary["N_Features"] = len(features)

    summaries.append(summary)


result = pd.concat(
    summaries,
    ignore_index=True,
)

result.to_csv(
    "experiments/04_feature_model/"
    "feature_ablation_v1.csv",
    index=False,
)


print()
print("=" * 110)
print("FEATURE ABLATION")
print("=" * 110)

cols = [
    "Feature_Set",
    "N_Features",
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
