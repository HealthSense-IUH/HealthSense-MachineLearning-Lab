from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from scipy.special import expit, logit

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
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


DATASETS = [
    "deepbeat",
    "pulsewatch",
    "liu",
]


EPS = 1e-6


# ============================================================
# Utility
# ============================================================

def safe_logit(p):

    p = np.clip(
        np.asarray(
            p,
            dtype=float
        ),
        EPS,
        1 - EPS
    )

    return logit(p)


def platt_fit(
    probability,
    y,
    sample_weight=None
):

    X = safe_logit(
        probability
    ).reshape(-1, 1)


    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000
    )


    model.fit(
        X,
        y,
        sample_weight=sample_weight
    )


    return model


def platt_predict(
    model,
    probability
):

    X = safe_logit(
        probability
    ).reshape(-1, 1)

    return (
        model.predict_proba(X)
        [:, 1]
    )


def apply_coefficients(
    probability,
    intercept,
    slope
):

    x = safe_logit(
        probability
    )

    return expit(
        intercept +
        slope * x
    )


def ece_fixed(
    y,
    p,
    bins=10
):

    y = np.asarray(y)
    p = np.asarray(p)


    edges = np.linspace(
        0,
        1,
        bins + 1
    )


    idx = np.digitize(
        p,
        edges[1:-1],
        right=False
    )


    ece = 0.0


    for b in range(bins):

        mask = (
            idx == b
        )


        if not mask.any():
            continue


        frac = mask.mean()

        observed = y[
            mask
        ].mean()

        predicted = p[
            mask
        ].mean()


        ece += (
            frac
            *
            abs(
                observed -
                predicted
            )
        )


    return float(ece)


def ece_quantile(
    y,
    p,
    bins=10
):

    df = pd.DataFrame({
        "y": y,
        "p": p
    })


    try:

        df["bin"] = pd.qcut(
            df["p"],
            q=bins,
            duplicates="drop"
        )

    except Exception:

        return np.nan


    total = len(df)

    ece = 0.0


    for _, g in df.groupby(
        "bin",
        observed=True
    ):

        if len(g) == 0:
            continue


        ece += (
            len(g) / total
            *
            abs(
                g["y"].mean()
                -
                g["p"].mean()
            )
        )


    return float(ece)


def calibration_slope_intercept(
    y,
    p
):

    X = safe_logit(
        p
    ).reshape(-1, 1)


    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000
    )


    model.fit(
        X,
        y
    )


    return (
        float(
            model.intercept_[0]
        ),
        float(
            model.coef_[0, 0]
        )
    )


def metrics(
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


    auc = roc_auc_score(
        y,
        p
    )

    ap = average_precision_score(
        y,
        p
    )

    brier = brier_score_loss(
        y,
        p
    )

    ll = log_loss(
        y,
        np.clip(
            p,
            EPS,
            1-EPS
        )
    )

    intercept, slope = (
        calibration_slope_intercept(
            y,
            p
        )
    )


    return {
        "AUROC":
            auc,

        "PR_AUC":
            ap,

        "Brier":
            brier,

        "LogLoss":
            ll,

        "ECE_fixed_10":
            ece_fixed(
                y,
                p,
                bins=10
            ),

        "ECE_quantile_10":
            ece_quantile(
                y,
                p,
                bins=10
            ),

        "calibration_intercept":
            intercept,

        "calibration_slope":
            slope,
    }


def balanced_weights(
    df
):

    # Give equal total weight to every dataset,
    # then equal total weight to every subject
    # within the dataset.

    weight = np.zeros(
        len(df),
        dtype=float
    )


    for dataset, dg in df.groupby(
        "dataset"
    ):

        subjects = (
            dg["subject_id"]
            .astype(str)
            .unique()
        )

        dataset_weight = (
            1.0 /
            df["dataset"].nunique()
        )

        subject_weight = (
            dataset_weight
            /
            len(subjects)
        )


        for subject_id, sg in dg.groupby(
            "subject_id"
        ):

            row_weight = (
                subject_weight
                /
                len(sg)
            )

            weight[
                sg.index.to_numpy()
            ] = row_weight


    # Normalize so mean sample weight = 1.
    weight = (
        weight
        /
        weight.mean()
    )

    return weight


# ============================================================
# Load DEV
# ============================================================

dev_parts = []


for dataset in DATASETS:

    f = (
        ART /
        f"{dataset}_dev_verified_streams.csv"
    )

    x = pd.read_csv(
        f,
        low_memory=False
    )


    x = x[
        [
            "subject_id",
            "label",
            "meta_probability",
            "ensemble_std",
            "stream_id",
            "order_in_stream"
        ]
    ].copy()


    x["dataset"] = dataset

    x["subject_id"] = (
        x["subject_id"]
        .astype(str)
    )

    x["group_id"] = (
        dataset
        + ":"
        + x["subject_id"]
    )


    dev_parts.append(x)


dev = pd.concat(
    dev_parts,
    ignore_index=True
)


dev["sample_weight"] = (
    balanced_weights(
        dev
    )
)


print("=" * 100)
print("DEV CALIBRATION DATA")

print(
    dev.groupby(
        "dataset"
    ).agg(
        rows=("label", "size"),
        subjects=("subject_id", "nunique"),
        AF=("label", "sum")
    )
)


# ============================================================
# Group OOF Platt calibration
# ============================================================

groups = (
    dev["group_id"]
    .to_numpy()
)

y = (
    dev["label"]
    .astype(int)
    .to_numpy()
)

raw_p = (
    dev["meta_probability"]
    .to_numpy()
)

weights = (
    dev["sample_weight"]
    .to_numpy()
)


n_groups = (
    dev["group_id"]
    .nunique()
)

n_splits = min(
    5,
    n_groups
)


gkf = GroupKFold(
    n_splits=n_splits
)


oof = np.full(
    len(dev),
    np.nan
)


for fold, (
    train_idx,
    val_idx
) in enumerate(
    gkf.split(
        dev,
        y,
        groups
    ),
    start=1
):

    train_y = y[
        train_idx
    ]


    if len(
        np.unique(
            train_y
        )
    ) < 2:

        raise RuntimeError(
            f"Fold {fold}: training fold has one class"
        )


    model = platt_fit(
        raw_p[
            train_idx
        ],
        train_y,
        sample_weight=weights[
            train_idx
        ]
    )


    oof[
        val_idx
    ] = platt_predict(
        model,
        raw_p[
            val_idx
        ]
    )


if np.isnan(oof).any():

    raise RuntimeError(
        "OOF calibration contains NaN"
    )


dev["calibrated_probability_oof"] = (
    oof
)


# ============================================================
# Final Platt model on ALL DEV
# ============================================================

final_calibrator = platt_fit(
    raw_p,
    y,
    sample_weight=weights
)


intercept = float(
    final_calibrator.intercept_[0]
)

slope = float(
    final_calibrator.coef_[0, 0]
)


print("\nFINAL CALIBRATOR")
print("intercept:", intercept)
print("slope:", slope)


calibrator_json = {
    "version":
        "v6_global_platt_calibrator_1",

    "input":
        "meta_probability",

    "formula":
        (
            "calibrated_probability = "
            "sigmoid(intercept + "
            "slope * logit(meta_probability))"
        ),

    "intercept":
        intercept,

    "slope":
        slope,

    "fit_data":
        "DEV_ONLY",

    "fit_weighting":
        (
            "equal dataset weight, "
            "equal subject weight within dataset, "
            "equal row weight within subject"
        ),

    "development_oof":
        (
            f"{n_splits}-fold GroupKFold "
            "using dataset:subject_id"
        ),

    "test_used_for_fit":
        False,

    "protocol_note":
        (
            "Calibration is frozen for future "
            "external validation. "
            "Reused test metrics remain exploratory."
        ),
}


cal_path = (
    ART /
    "FROZEN_PLATT_CALIBRATOR.json"
)


cal_path.write_text(
    json.dumps(
        calibrator_json,
        indent=2
    )
)


sha = hashlib.sha256(
    cal_path.read_bytes()
).hexdigest()


(
    ART /
    "FROZEN_PLATT_CALIBRATOR_SHA256.txt"
).write_text(
    sha
    + "  "
    + cal_path.name
    + "\n"
)


# ============================================================
# DEV diagnostic metrics
# ============================================================

summary = []


for dataset in DATASETS:

    x = dev[
        dev["dataset"]
        ==
        dataset
    ]


    for probability_col, method in [
        (
            "meta_probability",
            "RAW"
        ),
        (
            "calibrated_probability_oof",
            "PLATT_OOF"
        ),
    ]:

        m = metrics(
            x["label"],
            x[
                probability_col
            ]
        )

        summary.append({
            "dataset":
                dataset,

            "split":
                "DEV",

            "method":
                method,

            "rows":
                len(x),

            "subjects":
                x[
                    "subject_id"
                ].nunique(),

            **m
        })


# pooled DEV
for probability_col, method in [
    (
        "meta_probability",
        "RAW"
    ),
    (
        "calibrated_probability_oof",
        "PLATT_OOF"
    ),
]:

    m = metrics(
        dev["label"],
        dev[
            probability_col
        ]
    )

    summary.append({
        "dataset":
            "POOLED",

        "split":
            "DEV",

        "method":
            method,

        "rows":
            len(dev),

        "subjects":
            dev[
                "group_id"
            ].nunique(),

        **m
    })


dev[
    [
        "dataset",
        "subject_id",
        "label",
        "meta_probability",
        "calibrated_probability_oof",
        "ensemble_std",
        "stream_id",
        "order_in_stream"
    ]
].to_csv(
    ART /
    "dev_platt_oof_predictions.csv",
    index=False
)


# ============================================================
# Apply FINAL frozen calibrator to TEST
# ============================================================

for dataset in DATASETS:

    f = (
        ART /
        f"{dataset}_test_verified_streams.csv"
    )


    x = pd.read_csv(
        f,
        low_memory=False
    )


    x["subject_id"] = (
        x["subject_id"]
        .astype(str)
    )


    x["calibrated_probability"] = (
        apply_coefficients(
            x[
                "meta_probability"
            ].to_numpy(),
            intercept,
            slope
        )
    )


    x.to_csv(
        ART /
        f"{dataset}_test_calibrated_predictions.csv",
        index=False
    )


    for probability_col, method in [
        (
            "meta_probability",
            "RAW"
        ),
        (
            "calibrated_probability",
            "PLATT_DEV_FROZEN"
        ),
    ]:

        m = metrics(
            x["label"],
            x[
                probability_col
            ]
        )


        summary.append({
            "dataset":
                dataset,

            "split":
                "REUSED_TEST",

            "method":
                method,

            "rows":
                len(x),

            "subjects":
                x[
                    "subject_id"
                ].nunique(),

            **m
        })


summary = pd.DataFrame(
    summary
)


summary.to_csv(
    ART /
    "calibration_summary.csv",
    index=False
)


print("\n" + "=" * 100)
print("CALIBRATION SUMMARY")

print(
    summary.to_string(
        index=False
    )
)


print("\nFrozen calibrator:")
print(cal_path)
print("SHA256:", sha)
