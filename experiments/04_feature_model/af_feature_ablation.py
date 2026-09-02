import sys
sys.path.insert(0, "src")

import pandas as pd

from sklearn.model_selection import (
    LeaveOneGroupOut,
    GroupKFold,
    GridSearchCV,
)

from healthsense_ml import config
from healthsense_ml.training import build_model_registry
from healthsense_ml import evaluation


df = pd.read_csv(
    "experiments/04_feature_model/"
    "af_specific_features.csv"
)

BASE = config.CORE_FEATURES

SETS = {

    "baseline13":
        BASE,

    "plus_pnn":
        BASE + [
            "pNN40",
            "pNN70",
        ],

    "plus_tpr":
        BASE + [
            "TPR",
        ],

    "plus_robust":
        BASE + [
            "MAD_NN",
            "nRMSSD",
            "SD1_SD2_ratio",
        ],

    "plus_all_simple":
        BASE + [
            "pNN40",
            "pNN70",
            "TPR",
            "MAD_NN",
            "nRMSSD",
            "SD1_SD2_ratio",
        ],
}


def run_loso(features):

    X = df[features]
    y = df["status"].to_numpy()
    groups = df["record_id"].to_numpy()

    logo = LeaveOneGroupOut()

    registry = build_model_registry()

    predictions = []

    for model_name, spec in registry.items():

        print(
            model_name,
            "|",
            len(features),
            "features",
        )

        for train_idx, test_idx in logo.split(
            X,
            y,
            groups,
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

            # v5 decision: NO IQR
            search.fit(
                X_tr,
                y_tr,
                groups=g_tr,
            )

            probs = search.predict_proba(
                X_te
            )[:, 1]

            rid = groups[
                test_idx
            ][0]

            for prob, truth in zip(
                probs,
                y_te,
            ):

                predictions.append({
                    "record_id": rid,
                    "status": int(truth),
                    "model": model_name,
                    "prob": float(prob),
                    "pred": int(
                        prob >= 0.5
                    ),
                })

    return pd.DataFrame(
        predictions
    )


summaries = []

for name, features in SETS.items():

    print()
    print("=" * 90)
    print(name)
    print(features)
    print("=" * 90)

    pred = run_loso(
        features
    )

    pred.to_csv(
        "experiments/04_feature_model/"
        f"pred_{name}_af_features.csv",
        index=False,
    )

    summary = evaluation.summarize(
        pred
    )

    summary["Feature_Set"] = name
    summary["N_Features"] = len(
        features
    )

    summaries.append(
        summary
    )


result = pd.concat(
    summaries,
    ignore_index=True,
)

result.to_csv(
    "experiments/04_feature_model/"
    "af_feature_ablation.csv",
    index=False,
)


print()
print("=" * 110)
print("AF FEATURE ABLATION")
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
