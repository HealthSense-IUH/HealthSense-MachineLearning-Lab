import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    GroupKFold,
    GridSearchCV,
)

from healthsense_ml.training import build_model_registry


df = pd.read_csv(
    "experiments/04_feature_model/"
    "af_features_cosen_ac.csv"
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

X = df[FEATURES]
y = df["status"].to_numpy()
groups = df["record_id"].to_numpy()

rf_spec = build_model_registry()["Random Forest"]

outer = LeaveOneGroupOut()

rows = []


def ece(y_true, prob, bins=10):

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    total = len(y_true)
    score = 0.0

    for lo, hi in zip(
        edges[:-1],
        edges[1:],
    ):

        if hi == 1.0:
            mask = (
                (prob >= lo)
                &
                (prob <= hi)
            )
        else:
            mask = (
                (prob >= lo)
                &
                (prob < hi)
            )

        if not mask.any():
            continue

        accuracy = np.mean(
            y_true[mask]
        )

        confidence = np.mean(
            prob[mask]
        )

        score += (
            mask.sum() / total
        ) * abs(
            accuracy - confidence
        )

    return score


for fold, (
    train_idx,
    test_idx,
) in enumerate(
    outer.split(X, y, groups),
    1,
):

    test_subject = (
        groups[test_idx][0]
    )

    print(
        f"[{fold:02d}/35] "
        f"{test_subject}"
    )

    X_tr = X.iloc[train_idx]
    y_tr = y[train_idx]
    g_tr = groups[train_idx]

    X_te = X.iloc[test_idx]
    y_te = y[test_idx]

    # --------------------------------------------------
    # 1. Tune RF only inside outer training subjects
    # --------------------------------------------------

    inner = GroupKFold(
        n_splits=3
    )

    search = GridSearchCV(
        clone(
            rf_spec["pipeline"]
        ),
        rf_spec["param_grid"],
        cv=inner,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )

    search.fit(
        X_tr,
        y_tr,
        groups=g_tr,
    )

    # Raw model
    raw_model = (
        search.best_estimator_
    )

    raw_prob = (
        raw_model.predict_proba(
            X_te
        )[:, 1]
    )

    # --------------------------------------------------
    # 2. Group-aware calibration
    # --------------------------------------------------

    # Hyperparameters already selected using
    # outer-training subjects only.
    base_model = clone(
        rf_spec["pipeline"]
    )

    base_model.set_params(
        **search.best_params_
    )

    calibration_splits = list(
        GroupKFold(
            n_splits=3
        ).split(
            X_tr,
            y_tr,
            g_tr,
        )
    )

    for method in [
        "sigmoid",
        "isotonic",
    ]:

        calibrated = (
            CalibratedClassifierCV(
                estimator=clone(
                    base_model
                ),
                method=method,
                cv=calibration_splits,
                ensemble=False,
            )
        )

        calibrated.fit(
            X_tr,
            y_tr,
        )

        prob = (
            calibrated.predict_proba(
                X_te
            )[:, 1]
        )

        for truth, p in zip(
            y_te,
            prob,
        ):

            rows.append({
                "record_id":
                    test_subject,
                "status":
                    int(truth),
                "method":
                    method,
                "prob":
                    float(p),
            })

    for truth, p in zip(
        y_te,
        raw_prob,
    ):

        rows.append({
            "record_id":
                test_subject,
            "status":
                int(truth),
            "method":
                "raw",
            "prob":
                float(p),
        })


pred = pd.DataFrame(rows)

pred.to_csv(
    "experiments/05_calibration_alert/"
    "calibration_loso_predictions.csv",
    index=False,
)


print()
print("=" * 90)
print("CALIBRATION RESULTS")
print("=" * 90)

summary = []

for method in [
    "raw",
    "sigmoid",
    "isotonic",
]:

    x = pred[
        pred.method == method
    ]

    yt = x.status.to_numpy()
    p = x.prob.to_numpy()

    summary.append({
        "method":
            method,

        "auc":
            roc_auc_score(
                yt,
                p,
            ),

        "brier":
            brier_score_loss(
                yt,
                p,
            ),

        "log_loss":
            log_loss(
                yt,
                p,
            ),

        "ece10":
            ece(
                yt,
                p,
                bins=10,
            ),
    })


summary = pd.DataFrame(
    summary
)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.5f}",
    )
)


print()
print("=" * 90)
print("SUBJECT-LEVEL")
print("=" * 90)

subject = (
    pred.groupby(
        [
            "method",
            "record_id",
            "status",
        ]
    )["prob"]
    .mean()
    .reset_index()
)

subject_rows = []

for method in [
    "raw",
    "sigmoid",
    "isotonic",
]:

    x = subject[
        subject.method == method
    ]

    subject_rows.append({
        "method":
            method,

        "auc":
            roc_auc_score(
                x.status,
                x.prob,
            ),

        "brier":
            brier_score_loss(
                x.status,
                x.prob,
            ),
    })


subject_summary = pd.DataFrame(
    subject_rows
)

print(
    subject_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.5f}",
    )
)
