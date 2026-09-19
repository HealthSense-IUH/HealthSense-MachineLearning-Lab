import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


features = pd.read_csv(
    "experiments/04_feature_model/"
    "af_features_cosen_ac.csv"
)

quality = pd.read_csv(
    "experiments/03_sqi/"
    "window_quality.csv"
)

features["t_start"] = (
    features["t_start"].round(6)
)

quality["t_start"] = (
    quality["t_start"].round(6)
)

quality["beat_f1"] = (
    quality["beat_f1"]
    .fillna(0.0)
)

df = features.merge(
    quality[
        [
            "record_id",
            "t_start",
            "beat_f1",
        ]
    ],
    on=[
        "record_id",
        "t_start",
    ],
    how="inner",
    validate="one_to_one",
)


print("=" * 90)
print("AC vs BEAT QUALITY")
print("=" * 90)

for status, label in [
    (0, "Non-AF"),
    (1, "AF"),
]:

    x = df[
        df.status == status
    ]

    rho, p = spearmanr(
        x["PPG_AC"],
        x["beat_f1"],
    )

    print(
        f"{label:6s}: "
        f"rho={rho:.4f} "
        f"p={p:.3e}"
    )


print()
print("=" * 90)
print("AC BY QUALITY STRATUM")
print("=" * 90)

df["quality_group"] = pd.cut(
    df["beat_f1"],
    bins=[
        -np.inf,
        0.70,
        0.90,
        np.inf,
    ],
    labels=[
        "poor",
        "medium",
        "good",
    ],
    right=False,
)

table = (
    df.groupby(
        [
            "quality_group",
            "status",
        ],
        observed=True,
    )
    .agg(
        windows=(
            "PPG_AC",
            "size",
        ),
        median_ac=(
            "PPG_AC",
            "median",
        ),
        median_beat_f1=(
            "beat_f1",
            "median",
        ),
    )
)

print(table.round(5))


print()
print("=" * 90)
print("AF AUC USING PPG_AC ONLY")
print("=" * 90)

# Lower AC is expected for AF, hence use -PPG_AC.
for group in [
    "ALL",
    "poor",
    "medium",
    "good",
]:

    if group == "ALL":
        x = df.copy()
    else:
        x = df[
            df.quality_group == group
        ].copy()

    if (
        len(x) == 0
        or x.status.nunique() < 2
    ):
        continue

    auc = roc_auc_score(
        x["status"],
        -x["PPG_AC"],
    )

    print(
        f"{group:6s} | "
        f"windows={len(x):4d} | "
        f"AUC={auc:.4f}"
    )


print()
print("=" * 90)
print("SUBJECT-LEVEL PPG_AC ONLY")
print("=" * 90)

subject = (
    df.groupby(
        [
            "record_id",
            "status",
        ]
    )
    .agg(
        median_ac=(
            "PPG_AC",
            "median",
        ),
        median_f1=(
            "beat_f1",
            "median",
        ),
    )
    .reset_index()
)

subject_auc = roc_auc_score(
    subject["status"],
    -subject["median_ac"],
)

print(
    "Subject AUC:",
    round(subject_auc, 4),
)

print()
print(
    subject
    .sort_values("median_ac")
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)
