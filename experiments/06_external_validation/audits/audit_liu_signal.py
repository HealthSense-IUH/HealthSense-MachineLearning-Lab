from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(
    "data/raw/PPGArrhythmiaDetection-main"
)

DATA_DIR = ROOT / "valid_testDataset"

LABEL_NAMES = {
    0: "SR",
    1: "PVC",
    2: "PAC",
    3: "VT",
    4: "SVT",
    5: "AF",
}


def read_subject_list(path):
    return {
        x.strip().strip("'").strip('"')
        for x in path.read_text().splitlines()
        if x.strip()
    }


valid = read_subject_list(
    ROOT / "validsubjects.txt"
)

test = read_subject_list(
    ROOT / "testsubjects.txt"
)


rows = []

for path in sorted(
    DATA_DIR.glob("*.mat")
):

    mat = loadmat(
        path,
        squeeze_me=True,
    )

    ppg = np.asarray(
        mat["ppgseg"],
        dtype=float,
    )

    labels = np.asarray(
        mat["labels"]
    ).reshape(-1).astype(int)

    if ppg.shape[1] != 1000:
        raise RuntimeError(
            f"{path.name}: {ppg.shape}"
        )

    split = (
        "validation"
        if path.name in valid
        else "test"
    )

    for i in range(len(labels)):

        x = ppg[i]

        rows.append({
            "subject":
                path.stem,

            "split":
                split,

            "segment":
                i,

            "label":
                labels[i],

            "class":
                LABEL_NAMES[
                    labels[i]
                ],

            "min":
                np.nanmin(x),

            "max":
                np.nanmax(x),

            "mean":
                np.nanmean(x),

            "std":
                np.nanstd(x),

            "range":
                np.nanmax(x)
                - np.nanmin(x),

            "nan_fraction":
                np.mean(
                    ~np.isfinite(x)
                ),

            "zero_fraction":
                np.mean(
                    x == 0
                ),

            "constant":
                int(
                    np.nanstd(x)
                    < 1e-10
                ),
        })


df = pd.DataFrame(rows)

df.to_csv(
    "experiments/06_external_validation/"
    "liu_signal_audit.csv",
    index=False,
)


print("=" * 90)
print("LIU SIGNAL AUDIT")
print("=" * 90)

print(
    "Segments:",
    len(df),
)

print(
    "Subjects:",
    df.subject.nunique(),
)

print(
    "NaN-containing:",
    (
        df.nan_fraction > 0
    ).sum(),
)

print(
    "Constant:",
    df.constant.sum(),
)


print()
print("=" * 90)
print("AMPLITUDE DISTRIBUTION")
print("=" * 90)

print(
    df[
        [
            "min",
            "max",
            "mean",
            "std",
            "range",
            "zero_fraction",
        ]
    ]
    .describe(
        percentiles=[
            .01,
            .05,
            .25,
            .5,
            .75,
            .95,
            .99,
        ]
    )
    .round(5)
)


print()
print("=" * 90)
print("BY CLASS")
print("=" * 90)

print(
    df.groupby(
        "class"
    )[
        [
            "std",
            "range",
        ]
    ]
    .median()
    .round(5)
)


print()
print("=" * 90)
print("BY SPLIT")
print("=" * 90)

print(
    df.groupby(
        "split"
    )[
        [
            "std",
            "range",
        ]
    ]
    .median()
    .round(5)
)
