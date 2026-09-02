import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


N_BOOT = 5000
SEED = 42

rng = np.random.default_rng(SEED)


CANDIDATES = {
    "RF_AC": (
        "experiments/04_feature_model/"
        "pred_plus_ac_04d.csv",
        "Random Forest",
    ),

    "XGB_COSEN_AC": (
        "experiments/04_feature_model/"
        "pred_plus_cosen_ac_04d.csv",
        "XGBoost",
    ),
}


data = {}

for name, (path, model_name) in CANDIDATES.items():

    df = pd.read_csv(path)

    df = df[
        df["model"] == model_name
    ].copy()

    # Stable window index inside each subject
    df["window_id"] = (
        df.groupby("record_id")
        .cumcount()
    )

    data[name] = df


subjects = sorted(
    set.intersection(
        *[
            set(x.record_id.unique())
            for x in data.values()
        ]
    )
)

print("Subjects:", len(subjects))


def window_auc_for_sample(df, sampled_subjects):

    parts = []

    for boot_id, subject in enumerate(
        sampled_subjects
    ):

        x = df[
            df.record_id == subject
        ].copy()

        # sampled subject must behave as a new
        # independent bootstrap cluster
        x["boot_subject"] = boot_id

        parts.append(x)

    x = pd.concat(
        parts,
        ignore_index=True,
    )

    if x.status.nunique() < 2:
        return np.nan

    return roc_auc_score(
        x.status,
        x.prob,
    )


def subject_auc_for_sample(
    df,
    sampled_subjects,
):

    rows = []

    for boot_id, subject in enumerate(
        sampled_subjects
    ):

        x = df[
            df.record_id == subject
        ]

        rows.append({
            "boot_subject": boot_id,
            "status":
                int(x.status.iloc[0]),

            # same aggregation currently used
            # for subject probability comparison
            "prob":
                float(x.prob.mean()),
        })

    x = pd.DataFrame(rows)

    if x.status.nunique() < 2:
        return np.nan

    return roc_auc_score(
        x.status,
        x.prob,
    )


boot_rows = []

for b in range(N_BOOT):

    sampled = rng.choice(
        subjects,
        size=len(subjects),
        replace=True,
    )

    row = {"bootstrap": b}

    for name, df in data.items():

        row[
            f"{name}_window_auc"
        ] = window_auc_for_sample(
            df,
            sampled,
        )

        row[
            f"{name}_subject_auc"
        ] = subject_auc_for_sample(
            df,
            sampled,
        )

    boot_rows.append(row)


boot = pd.DataFrame(boot_rows)

boot["delta_window"] = (
    boot["XGB_COSEN_AC_window_auc"]
    -
    boot["RF_AC_window_auc"]
)

boot["delta_subject"] = (
    boot["XGB_COSEN_AC_subject_auc"]
    -
    boot["RF_AC_subject_auc"]
)


boot.to_csv(
    "experiments/04_feature_model/"
    "model_selection_bootstrap.csv",
    index=False,
)


def ci(x):

    x = x.dropna()

    return (
        np.percentile(x, 2.5),
        np.median(x),
        np.percentile(x, 97.5),
    )


print()
print("=" * 90)
print("SUBJECT-CLUSTER BOOTSTRAP")
print("=" * 90)


for name in CANDIDATES:

    lo, med, hi = ci(
        boot[f"{name}_window_auc"]
    )

    print(
        f"{name:15s} "
        f"window AUC "
        f"{med:.4f} "
        f"[{lo:.4f}, {hi:.4f}]"
    )

    lo, med, hi = ci(
        boot[f"{name}_subject_auc"]
    )

    print(
        f"{name:15s} "
        f"subject AUC "
        f"{med:.4f} "
        f"[{lo:.4f}, {hi:.4f}]"
    )


print()
print("=" * 90)
print("PAIRED DIFFERENCE: XGB - RF")
print("=" * 90)

for metric in [
    "delta_window",
    "delta_subject",
]:

    x = boot[metric].dropna()

    lo, med, hi = ci(x)

    print(
        metric,
        f"median={med:+.4f}",
        f"95%CI=[{lo:+.4f}, {hi:+.4f}]",
        f"P(XGB>RF)={(x > 0).mean():.4f}",
    )
