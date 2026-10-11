from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


MIMIC = Path(
    "data/features/v6/"
    "mimic_perform_v6_14f.csv"
)

DEEP = Path(
    "data/features/v6/"
    "deepbeat_train_v6_14f.csv"
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


def make_model():

    return Pipeline([
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


def subject_class_weights(
    df,
):
    """
    One-domain weighting:
      class equal
      -> subject equal within class
      -> window equal within subject
    """

    w = np.zeros(
        len(df),
        dtype=float,
    )

    labels = sorted(
        df.label.unique()
    )

    for label in labels:

        class_idx = np.where(
            df.label.to_numpy()
            == label
        )[0]

        subjects = sorted(
            df.iloc[class_idx]
            .subject_key.unique()
        )

        class_total = (
            1.0 / len(labels)
        )

        subject_total = (
            class_total
            / len(subjects)
        )

        for subject in subjects:

            idx = np.where(
                (
                    df.label.to_numpy()
                    == label
                )
                &
                (
                    df.subject_key
                    .to_numpy()
                    == subject
                )
            )[0]

            w[idx] = (
                subject_total
                / len(idx)
            )

    w *= (
        len(w)
        / w.sum()
    )

    return w


def multidomain_weights(
    df,
):
    """
    Same V6-A policy:
      domain equal
      -> class equal
      -> subject equal
      -> window equal
    """

    w = np.zeros(
        len(df),
        dtype=float,
    )

    domains = sorted(
        df.domain.unique()
    )

    domain_total = (
        1.0
        / len(domains)
    )

    for domain in domains:

        dm = (
            df.domain.to_numpy()
            == domain
        )

        labels = sorted(
            df.loc[
                dm,
                "label"
            ].unique()
        )

        class_total = (
            domain_total
            / len(labels)
        )

        for label in labels:

            cell = (
                dm
                &
                (
                    df.label.to_numpy()
                    == label
                )
            )

            subjects = sorted(
                df.loc[
                    cell,
                    "subject_key"
                ].unique()
            )

            subject_total = (
                class_total
                / len(subjects)
            )

            for subject in subjects:

                idx = np.where(
                    cell
                    &
                    (
                        df.subject_key
                        .to_numpy()
                        == subject
                    )
                )[0]

                w[idx] = (
                    subject_total
                    / len(idx)
                )

    w *= (
        len(w)
        / w.sum()
    )

    return w


# ============================================================
# DATA
# ============================================================

mimic = pd.read_csv(
    MIMIC
)

mimic = mimic.rename(
    columns={
        "status":
            "label",
    }
)

mimic["domain"] = (
    "MIMIC_PERFORM"
)

mimic["subject_key"] = (
    "MIMIC:"
    + mimic.record_id.astype(str)
)


deep = pd.read_csv(
    DEEP
)

deep = deep[
    deep.feature_ok == True
].copy()

deep["domain"] = (
    "DEEPBEAT"
)

deep["subject_key"] = (
    "DEEPBEAT:"
    + deep.subject_id
      .astype(int)
      .astype(str)
)


subjects = sorted(
    mimic.record_id.unique()
)


print("=" * 100)
print("MIMIC LOSO — SOURCE-DOMAIN RETENTION")
print("=" * 100)

print(
    "MIMIC subjects:",
    len(subjects)
)

print(
    "DeepBeat train windows:",
    len(deep)
)


pred_rows = []


for fold, held in enumerate(
    subjects,
    start=1,
):

    print(
        f"[{fold:02d}/{len(subjects):02d}] "
        f"held={held}"
    )

    test = mimic[
        mimic.record_id == held
    ].copy()

    mimic_train = mimic[
        mimic.record_id != held
    ].copy()


    # ========================================================
    # CONTROL: MIMIC only
    # ========================================================

    baseline = make_model()

    wb = subject_class_weights(
        mimic_train
    )

    baseline.fit(
        mimic_train[
            FEATURES
        ],
        mimic_train.label,
        clf__sample_weight=wb,
    )

    p_baseline = (
        baseline.predict_proba(
            test[
                FEATURES
            ]
        )[:, 1]
    )


    # ========================================================
    # V6-A STYLE MULTIDOMAIN
    # ========================================================

    multi = pd.concat(
        [
            mimic_train[
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

    wm = multidomain_weights(
        multi
    )

    v6a = make_model()

    v6a.fit(
        multi[
            FEATURES
        ],
        multi.label,
        clf__sample_weight=wm,
    )

    p_v6a = (
        v6a.predict_proba(
            test[
                FEATURES
            ]
        )[:, 1]
    )


    for j, (_, r) in enumerate(
        test.iterrows()
    ):

        pred_rows.append({
            "record_id":
                held,

            "label":
                int(
                    r.label
                ),

            "t_start":
                r.t_start,

            "p_mimic_only":
                float(
                    p_baseline[j]
                ),

            "p_v6a":
                float(
                    p_v6a[j]
                ),
        })


pred = pd.DataFrame(
    pred_rows
)

pred.to_csv(
    OUT /
    "mimic_loso_v6a_retention_predictions.csv",
    index=False,
)


# ============================================================
# WINDOW LEVEL
# ============================================================

y = pred.label.to_numpy()

p0 = pred.p_mimic_only.to_numpy()
p1 = pred.p_v6a.to_numpy()


def metrics(
    p,
):
    return {
        "AUROC":
            roc_auc_score(
                y,
                p,
            ),

        "PR_AUC":
            average_precision_score(
                y,
                p,
            ),

        "Brier":
            brier_score_loss(
                y,
                p,
            ),
    }


m0 = metrics(
    p0
)

m1 = metrics(
    p1
)


# ============================================================
# SUBJECT LEVEL
# ============================================================

subject = (
    pred.groupby(
        "record_id"
    )
    .agg(
        label=(
            "label",
            "first",
        ),

        p_mimic_only=(
            "p_mimic_only",
            "mean",
        ),

        p_v6a=(
            "p_v6a",
            "mean",
        ),

        windows=(
            "label",
            "size",
        ),
    )
    .reset_index()
)

subject.to_csv(
    OUT /
    "mimic_loso_v6a_retention_subject.csv",
    index=False,
)


subject_auc_0 = roc_auc_score(
    subject.label,
    subject.p_mimic_only,
)

subject_auc_1 = roc_auc_score(
    subject.label,
    subject.p_v6a,
)


summary = pd.DataFrame([
    {
        "model":
            "MIMIC_ONLY",

        **m0,

        "subject_AUROC":
            subject_auc_0,
    },
    {
        "model":
            "V6A_MULTIDOMAIN",

        **m1,

        "subject_AUROC":
            subject_auc_1,
    },
])

summary.to_csv(
    OUT /
    "mimic_loso_v6a_retention_summary.csv",
    index=False,
)


print()
print("=" * 100)
print("RESULT")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)

print()
print("DELTA V6A - MIMIC_ONLY")

print(
    "Window AUROC :",
    f"{m1['AUROC'] - m0['AUROC']:+.6f}"
)

print(
    "PR-AUC       :",
    f"{m1['PR_AUC'] - m0['PR_AUC']:+.6f}"
)

print(
    "Brier improve:",
    f"{m0['Brier'] - m1['Brier']:+.6f}"
)

print(
    "Subject AUROC:",
    f"{subject_auc_1 - subject_auc_0:+.6f}"
)
