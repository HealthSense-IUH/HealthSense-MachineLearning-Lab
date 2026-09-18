from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

ROOT = Path(
    "experiments/07_v6_multidomain/artifacts"
)

V5 = ROOT / "deepbeat_validate_v5_predictions.csv"
V6 = ROOT / "deepbeat_validate_v6a_predictions.csv"

OUT = ROOT / "v6a_validation_subject_bootstrap.csv"

N_BOOT = 5000
SEED = 42

v5 = pd.read_csv(V5)
v6 = pd.read_csv(V6)

keys = [
    "global_index",
    "subject_id",
    "label",
]

m = v5[
    keys + ["probability"]
].merge(
    v6[
        keys + ["probability"]
    ],
    on=keys,
    suffixes=(
        "_v5",
        "_v6a",
    ),
    validate="one_to_one",
)

print("=" * 100)
print("V6-A VALIDATION — SUBJECT CLUSTER BOOTSTRAP")
print("=" * 100)

print("Rows    :", len(m))
print(
    "Subjects:",
    m.subject_id.nunique()
)

print("\nSubjects / class:")
print(
    m.groupby("subject_id")
    .label.first()
    .value_counts()
    .sort_index()
)

subjects = np.array(
    sorted(
        m.subject_id.unique()
    )
)

rng = np.random.default_rng(
    SEED
)

rows = []

for b in range(N_BOOT):

    sampled = rng.choice(
        subjects,
        size=len(subjects),
        replace=True,
    )

    chunks = []

    # Replicate entire subject clusters.
    for replicate_id, uid in enumerate(
        sampled
    ):
        d = m[
            m.subject_id == uid
        ].copy()

        # artificial bootstrap subject id,
        # so duplicate sampled subjects remain duplicates
        d["bootstrap_subject"] = (
            replicate_id
        )

        chunks.append(d)

    x = pd.concat(
        chunks,
        ignore_index=True,
    )

    y = x.label.to_numpy()

    if np.unique(y).size < 2:
        continue

    p5 = (
        x.probability_v5
        .to_numpy()
    )

    p6 = (
        x.probability_v6a
        .to_numpy()
    )

    auc5 = roc_auc_score(
        y,
        p5,
    )

    auc6 = roc_auc_score(
        y,
        p6,
    )

    ap5 = average_precision_score(
        y,
        p5,
    )

    ap6 = average_precision_score(
        y,
        p6,
    )

    brier5 = brier_score_loss(
        y,
        p5,
    )

    brier6 = brier_score_loss(
        y,
        p6,
    )

    rows.append({
        "iteration": b,
        "v5_auc": auc5,
        "v6a_auc": auc6,
        "delta_auc": auc6 - auc5,
        "v5_pr_auc": ap5,
        "v6a_pr_auc": ap6,
        "delta_pr_auc": ap6 - ap5,
        "v5_brier": brier5,
        "v6a_brier": brier6,
        # positive = improvement
        "delta_brier_improvement":
            brier5 - brier6,
    })


boot = pd.DataFrame(
    rows
)

boot.to_csv(
    OUT,
    index=False,
)


def report(col):

    x = (
        boot[col]
        .dropna()
        .to_numpy()
    )

    lo, med, hi = np.quantile(
        x,
        [
            0.025,
            0.5,
            0.975,
        ]
    )

    print(
        f"{col:28s} "
        f"median={med:+.6f} "
        f"95%CI=[{lo:+.6f}, {hi:+.6f}] "
        f"P(>0)={np.mean(x > 0):.4f}"
    )


print()
print(
    "Valid bootstrap iterations:",
    len(boot)
)

print()
report("delta_auc")
report("delta_pr_auc")
report("delta_brier_improvement")

print()
print("Saved:", OUT)
