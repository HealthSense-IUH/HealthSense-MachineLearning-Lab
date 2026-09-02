import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
)

df = pd.read_csv(
    "experiments/03_sqi/sqi_features.csv"
)

FEATURES = [
    "signal_std",
    "signal_iqr",
    "derivative_mad",
    "hf_noise_ratio",
    "baseline_ratio",
    "clipping_fraction",
    "n_peaks",
    "median_prominence",
    "prominence_iqr",
    "template_corr",
    "detector_count_agreement",
]

df["quality_good"] = (
    df["beat_f1"] >= 0.70
).astype(int)

predictions = []

subjects = df.record_id.unique()

for i, subject in enumerate(subjects, 1):

    print(
        f"[{i:02d}/{len(subjects)}] {subject}"
    )

    train = df[
        df.record_id != subject
    ].copy()

    test = df[
        df.record_id == subject
    ].copy()

    # Equalise contribution of AF/Non-AF and good/bad
    group_count = (
        train.groupby(
            ["label", "quality_good"]
        )["record_id"]
        .transform("size")
    )

    weights = 1.0 / group_count

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=7,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    model.fit(
        train[FEATURES],
        train["quality_good"],
        rf__sample_weight=weights,
    )

    proba = model.predict_proba(
        test[FEATURES]
    )[:, 1]

    temp = test[
        [
            "record_id",
            "label",
            "t_start",
            "beat_f1",
            "quality_good",
        ]
    ].copy()

    temp["quality_score"] = proba

    predictions.append(temp)


pred = pd.concat(
    predictions,
    ignore_index=True,
)

pred.to_csv(
    "experiments/03_sqi/"
    "sqi_loso_predictions.csv",
    index=False,
)

auc = roc_auc_score(
    pred.quality_good,
    pred.quality_score,
)

print()
print("=" * 90)
print("SQI LOSO")
print("=" * 90)

print("ROC-AUC:", round(auc, 4))


for threshold in [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]:

    accepted = (
        pred.quality_score >= threshold
    )

    print()
    print(
        f"SQI threshold = {threshold:.2f}"
    )

    print(
        "Overall coverage:",
        round(accepted.mean(), 4),
    )

    for label in ["AF", "Non-AF"]:

        mask = pred.label == label

        coverage = accepted[mask].mean()

        accepted_f1 = pred.loc[
            mask & accepted,
            "beat_f1",
        ]

        bad_leak = (
            pred.loc[
                mask & accepted,
                "beat_f1",
            ] < 0.70
        ).mean()

        print(
            f"{label:6s} "
            f"coverage={coverage:.3f} "
            f"accepted median F1="
            f"{accepted_f1.median():.3f} "
            f"bad leakage={bad_leak:.3f}"
        )
