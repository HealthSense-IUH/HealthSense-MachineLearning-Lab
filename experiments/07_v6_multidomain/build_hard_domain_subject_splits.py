from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

OUT = Path(
    "experiments/07_v6_multidomain/artifacts/"
    "hard_domain_splits"
)
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

pulse = pd.read_csv(
    "experiments/06_external_validation/"
    "pulsewatch_frozen_v5_features.csv"
)
pulse = pulse[pulse.feature_ok == True].copy()
pulse["binary_label"] = (
    pulse.class_name == "AF"
).astype(int)

liu = pd.read_csv(
    "experiments/06_external_validation/"
    "liu_pseudo30_frozen_v5_predictions.csv"
)
liu = liu[liu.feature_ok == True].copy()
liu["binary_label_clean"] = (
    liu.rhythm == "AF"
).astype(int)

# ============================================================
# SPLIT HELPER
# 70% train / 15% dev / 15% final test approximately
# ============================================================

def make_split(df, group_col, class_col, required_classes, name):

    groups = df[group_col].astype(str)

    for seed in range(42, 10000):

        first = GroupShuffleSplit(
            n_splits=1,
            train_size=0.70,
            random_state=seed,
        )

        tr_idx, rest_idx = next(
            first.split(df, groups=groups)
        )

        train = df.iloc[tr_idx].copy()
        rest = df.iloc[rest_idx].copy()

        second = GroupShuffleSplit(
            n_splits=1,
            train_size=0.50,
            random_state=seed + 100000,
        )

        dv_idx, te_idx = next(
            second.split(
                rest,
                groups=rest[group_col].astype(str)
            )
        )

        dev = rest.iloc[dv_idx].copy()
        test = rest.iloc[te_idx].copy()

        parts = {
            "train": train,
            "dev": dev,
            "test": test,
        }

        good = True

        for split_name, d in parts.items():

            classes = set(
                d[class_col]
                .astype(str)
                .unique()
                .tolist()
            )

            # Require all requested classes in every split
            if not set(required_classes).issubset(classes):
                good = False
                break

            # Binary AF/non-AF must both exist
            binary_col = (
                "binary_label"
                if name == "pulsewatch"
                else "binary_label_clean"
            )

            if d[binary_col].nunique() != 2:
                good = False
                break

        if not good:
            continue

        # leakage check
        gs = {
            k: set(v[group_col].astype(str))
            for k, v in parts.items()
        }

        assert not (gs["train"] & gs["dev"])
        assert not (gs["train"] & gs["test"])
        assert not (gs["dev"] & gs["test"])

        print()
        print("="*100)
        print(name.upper(), "seed =", seed)
        print("="*100)

        for split_name, d in parts.items():
            print()
            print(split_name.upper())
            print(
                "subjects:",
                d[group_col].nunique(),
                "windows:",
                len(d),
            )
            print(
                d[class_col]
                .value_counts()
                .to_string()
            )

            d.to_csv(
                OUT / f"{name}_{split_name}.csv",
                index=False,
            )

        pd.DataFrame({
            "dataset": [name],
            "seed": [seed],
        }).to_csv(
            OUT / f"{name}_split_seed.csv",
            index=False,
        )

        return

    raise RuntimeError(
        f"Could not find valid split for {name}"
    )


make_split(
    pulse,
    group_col="uid",
    class_col="class_name",
    required_classes=[
        "AF",
        "NSR",
        "PAC_PVC",
    ],
    name="pulsewatch",
)

make_split(
    liu,
    group_col="subject_id",
    class_col="rhythm",
    required_classes=[
        "AF",
        "SR",
        "PAC",
        "PVC",
        "SVT",
        "VT",
    ],
    name="liu",
)

print()
print("DONE")
