from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

V5_PATH = Path(
    "models/mimic/"
    "healthsense_af_v5_rf_ac_frozen.pkl"
)

V6_PATH = Path(
    "models/multidomain/"
    "healthsense_af_v6a_rf_ac_frozen.pkl"
)

OUT = Path(
    "experiments/07_v6_multidomain/artifacts"
)

OUT.mkdir(
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


THRESHOLDS = [
    0.50,
    0.85,
    0.90,
]


v5 = joblib.load(
    V5_PATH
)

v6 = joblib.load(
    V6_PATH
)


# ============================================================
# HELPERS
# ============================================================

def finite_usable(df):

    d = df.copy()

    if "feature_ok" in d.columns:

        d = d[
            d.feature_ok == True
        ].copy()

    finite = np.isfinite(
        d[FEATURES]
        .to_numpy(dtype=float)
    ).all(axis=1)

    return (
        d.loc[finite]
        .copy()
        .reset_index(drop=True)
    )


def add_probabilities(df):

    d = finite_usable(
        df
    )

    d["p_v5"] = (
        v5.predict_proba(
            d[FEATURES]
        )[:, 1]
    )

    d["p_v6a"] = (
        v6.predict_proba(
            d[FEATURES]
        )[:, 1]
    )

    return d


def binary_metrics(
    y,
    p,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    out = {
        "n":
            len(y),

        "positive":
            int(
                np.sum(y == 1)
            ),

        "negative":
            int(
                np.sum(y == 0)
            ),
    }

    if np.unique(y).size == 2:

        out[
            "auroc"
        ] = roc_auc_score(
            y,
            p,
        )

        out[
            "pr_auc"
        ] = average_precision_score(
            y,
            p,
        )

        out[
            "brier"
        ] = brier_score_loss(
            y,
            p,
        )

    else:

        out["auroc"] = np.nan
        out["pr_auc"] = np.nan
        out["brier"] = np.nan

    for t in THRESHOLDS:

        pred = (
            p >= t
        ).astype(int)

        tn, fp, fn, tp = (
            confusion_matrix(
                y,
                pred,
                labels=[0, 1],
            )
            .ravel()
        )

        sens = (
            tp / (tp + fn)
            if tp + fn
            else np.nan
        )

        spec = (
            tn / (tn + fp)
            if tn + fp
            else np.nan
        )

        out[
            f"sensitivity_{t:.2f}"
        ] = sens

        out[
            f"specificity_{t:.2f}"
        ] = spec

        out[
            f"fpr_{t:.2f}"
        ] = (
            fp / (fp + tn)
            if fp + tn
            else np.nan
        )

    return out


results = []


def compare(
    dataset,
    comparison,
    d,
    y,
):

    for model_name, col in [
        ("V5", "p_v5"),
        ("V6A", "p_v6a"),
    ]:

        m = binary_metrics(
            y,
            d[col],
        )

        results.append({
            "dataset":
                dataset,

            "comparison":
                comparison,

            "model":
                model_name,

            **m,
        })


# ============================================================
# PULSEWATCH
# ============================================================

pulse = pd.read_csv(
    "experiments/06_external_validation/"
    "pulsewatch_frozen_v5_features.csv"
)

pulse = add_probabilities(
    pulse
)

pulse["class_norm"] = (
    pulse["class_name"]
    .astype(str)
    .str.upper()
    .str.strip()
)

pulse["y_af"] = (
    pulse.class_norm == "AF"
).astype(int)


print("=" * 100)
print("PULSEWATCH")
print("=" * 100)

print(
    pulse.class_norm
    .value_counts()
    .to_string()
)


compare(
    "Pulsewatch",
    "AF_vs_ALL",
    pulse,
    pulse.y_af,
)


# AF vs NSR
mask = (
    (pulse.class_norm == "AF")
    |
    pulse.class_norm.str.contains(
        "NSR",
        regex=False,
    )
)

if (
    mask.sum() > 0
    and pulse.loc[
        mask,
        "y_af"
    ].nunique() == 2
):

    compare(
        "Pulsewatch",
        "AF_vs_NSR",
        pulse.loc[mask],
        pulse.loc[
            mask,
            "y_af"
        ],
    )


# AF vs PAC/PVC
ectopy = (
    pulse.class_norm.str.contains(
        "PAC",
        regex=False,
    )
    |
    pulse.class_norm.str.contains(
        "PVC",
        regex=False,
    )
)

mask = (
    (pulse.class_norm == "AF")
    |
    ectopy
)

if (
    mask.sum() > 0
    and pulse.loc[
        mask,
        "y_af"
    ].nunique() == 2
):

    compare(
        "Pulsewatch",
        "AF_vs_PAC_PVC",
        pulse.loc[mask],
        pulse.loc[
            mask,
            "y_af"
        ],
    )


pulse.to_csv(
    OUT /
    "pulsewatch_v5_vs_v6a_predictions.csv",
    index=False,
)


# ============================================================
# LIU
# ============================================================

liu = pd.read_csv(
    "experiments/06_external_validation/"
    "liu_pseudo30_frozen_v5_predictions.csv"
)

liu = add_probabilities(
    liu
)

liu["rhythm_norm"] = (
    liu["rhythm"]
    .astype(str)
    .str.upper()
    .str.strip()
)

liu["y_af"] = (
    liu.rhythm_norm == "AF"
).astype(int)


# Sanity check against previous binary label
if "binary_label" in liu.columns:

    agreement = (
        liu.y_af.to_numpy()
        ==
        liu.binary_label
           .astype(int)
           .to_numpy()
    )

    if not agreement.all():
        raise RuntimeError(
            "Liu AF mapping disagrees "
            "with binary_label"
        )


print()
print("=" * 100)
print("LIU")
print("=" * 100)

print(
    liu.rhythm_norm
    .value_counts()
    .to_string()
)


compare(
    "Liu",
    "AF_vs_ALL",
    liu,
    liu.y_af,
)


# AF vs SR
mask = (
    liu.rhythm_norm.isin(
        [
            "AF",
            "SR",
        ]
    )
)

compare(
    "Liu",
    "AF_vs_SR",
    liu.loc[mask],
    liu.loc[
        mask,
        "y_af"
    ],
)


# AF vs PAC/PVC
mask = (
    liu.rhythm_norm.isin(
        [
            "AF",
            "PAC",
            "PVC",
        ]
    )
)

compare(
    "Liu",
    "AF_vs_PAC_PVC",
    liu.loc[mask],
    liu.loc[
        mask,
        "y_af"
    ],
)


liu.to_csv(
    OUT /
    "liu_v5_vs_v6a_predictions.csv",
    index=False,
)


# ============================================================
# PEAKDET PAC/PVC CHALLENGE
#
# All rows are non-AF.
# Main outcome = false positive probability / FPR.
# ============================================================

peak = pd.read_csv(
    "experiments/06_external_validation/"
    "peakdet_pac_pvc_frozen_v5_predictions.csv"
)

peak = add_probabilities(
    peak
)

print()
print("=" * 100)
print("PEAKDET PAC/PVC")
print("=" * 100)

if "class_name" in peak.columns:

    print(
        peak.class_name
        .value_counts()
        .to_string()
    )


for model_name, col in [
    ("V5", "p_v5"),
    ("V6A", "p_v6a"),
]:

    p = peak[col].to_numpy()

    row = {
        "dataset":
            "PeakDet",

        "comparison":
            "PAC_PVC_NON_AF_CHALLENGE",

        "model":
            model_name,

        "n":
            len(peak),

        "positive":
            0,

        "negative":
            len(peak),

        "auroc":
            np.nan,

        "pr_auc":
            np.nan,

        "brier":
            float(
                np.mean(
                    p ** 2
                )
            ),

        "prob_mean":
            float(
                np.mean(p)
            ),

        "prob_median":
            float(
                np.median(p)
            ),

        "prob_p90":
            float(
                np.quantile(
                    p,
                    0.90,
                )
            ),

        "prob_p95":
            float(
                np.quantile(
                    p,
                    0.95,
                )
            ),
    }

    for t in THRESHOLDS:

        row[
            f"sensitivity_{t:.2f}"
        ] = np.nan

        row[
            f"specificity_{t:.2f}"
        ] = float(
            np.mean(
                p < t
            )
        )

        row[
            f"fpr_{t:.2f}"
        ] = float(
            np.mean(
                p >= t
            )
        )

    results.append(
        row
    )


peak.to_csv(
    OUT /
    "peakdet_v5_vs_v6a_predictions.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    results
)

summary.to_csv(
    OUT /
    "v6a_reused_external_benchmark.csv",
    index=False,
)


print()
print("=" * 100)
print("REUSED EXTERNAL BENCHMARK")
print("=" * 100)

show = [
    c for c in [
        "dataset",
        "comparison",
        "model",
        "n",
        "positive",
        "negative",
        "auroc",
        "pr_auc",
        "brier",
        "sensitivity_0.50",
        "specificity_0.50",
        "fpr_0.50",
        "sensitivity_0.90",
        "specificity_0.90",
        "fpr_0.90",
        "prob_mean",
        "prob_median",
        "prob_p95",
    ]
    if c in summary.columns
]

print(
    summary[
        show
    ].to_string(
        index=False
    )
)

print()
print(
    "Saved:",
    OUT /
    "v6a_reused_external_benchmark.csv"
)
