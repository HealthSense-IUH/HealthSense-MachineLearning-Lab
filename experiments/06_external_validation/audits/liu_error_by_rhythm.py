from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score


ROOT = Path(
    "data/raw/PPGArrhythmiaDetection-main"
)

DATA_DIR = ROOT / "valid_testDataset"

PRED_FILE = (
    "experiments/06_external_validation/"
    "liu_AF_vs_ALL_predictions.csv"
)

LABEL_NAMES = {
    0: "SR",
    1: "PVC",
    2: "PAC",
    3: "VT",
    4: "SVT",
    5: "AF",
}


def read_list(path):
    return [
        x.strip()
        .strip("'")
        .strip('"')
        for x in path.read_text().splitlines()
        if x.strip()
    ]


test_subjects = read_list(
    ROOT / "testsubjects.txt"
)


truth_rows = []

for filename in test_subjects:

    path = DATA_DIR / filename

    mat = loadmat(
        path,
        squeeze_me=True,
    )

    labels = np.asarray(
        mat["labels"]
    ).reshape(-1).astype(int)

    for i, label in enumerate(labels):

        truth_rows.append({
            "subject":
                path.stem,

            "segment":
                i,

            "original_label":
                label,

            "rhythm":
                LABEL_NAMES[label],

            "binary_status":
                int(label == 5),
        })


truth = pd.DataFrame(
    truth_rows
)

pred = pd.read_csv(
    PRED_FILE
)


if len(truth) != len(pred):

    raise RuntimeError(
        f"Length mismatch: "
        f"truth={len(truth)} "
        f"pred={len(pred)}"
    )


# Prediction file was generated in the same
# subject/segment traversal order.
df = truth.copy()

df["prob"] = (
    pred["prob"].to_numpy()
)

df["pred"] = (
    df["prob"] >= 0.5
).astype(int)


print("=" * 100)
print("LIU AF-vs-ALL ERROR AUDIT BY RHYTHM")
print("=" * 100)


rows = []

af_prob = df[
    df.rhythm == "AF"
]["prob"].to_numpy()


for label, rhythm in LABEL_NAMES.items():

    x = df[
        df.original_label == label
    ]

    if rhythm == "AF":

        rows.append({
            "rhythm":
                rhythm,

            "segments":
                len(x),

            "median_prob":
                x.prob.median(),

            "mean_prob":
                x.prob.mean(),

            "p90_prob":
                x.prob.quantile(0.90),

            "positive_rate_at_05":
                x.pred.mean(),

            "auc_af_vs_class":
                np.nan,
        })

        continue

    pair = df[
        df.original_label.isin(
            [label, 5]
        )
    ].copy()

    y = (
        pair.original_label
        == 5
    ).astype(int)

    auc = roc_auc_score(
        y,
        pair.prob,
    )

    rows.append({
        "rhythm":
            rhythm,

        "segments":
            len(x),

        "median_prob":
            x.prob.median(),

        "mean_prob":
            x.prob.mean(),

        "p90_prob":
            x.prob.quantile(0.90),

        "positive_rate_at_05":
            x.pred.mean(),

        "auc_af_vs_class":
            auc,
    })


result = pd.DataFrame(
    rows
)


print(
    result.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 100)
print("FALSE-POSITIVE CONTRIBUTION")
print("=" * 100)

non_af = df[
    df.original_label != 5
].copy()

fp = non_af[
    non_af.pred == 1
]

fp_table = (
    fp.groupby("rhythm")
    .size()
    .rename("FP")
    .reset_index()
)

all_table = (
    non_af.groupby("rhythm")
    .size()
    .rename("segments")
    .reset_index()
)

out = all_table.merge(
    fp_table,
    on="rhythm",
    how="left",
)

out["FP"] = (
    out["FP"]
    .fillna(0)
    .astype(int)
)

out["FPR"] = (
    out["FP"]
    / out["segments"]
)

out["share_of_all_FP"] = (
    out["FP"]
    / max(out["FP"].sum(), 1)
)


print(
    out.sort_values(
        "FPR",
        ascending=False,
    )
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


result.to_csv(
    "experiments/06_external_validation/"
    "liu_error_by_rhythm.csv",
    index=False,
)
