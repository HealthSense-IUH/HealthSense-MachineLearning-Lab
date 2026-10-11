from pathlib import Path
from collections import Counter
import ast
import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path("data/raw/PPGArrhythmiaDetection-main")
DATA = ROOT / "valid_testDataset"
OUT = Path("experiments/06_external_validation")

LABELS = {
    0: "SR",
    1: "PVC",
    2: "PAC",
    3: "VT",
    4: "SVT",
    5: "AF",
}

# ------------------------------------------------------------
# Official test subject filenames
# ------------------------------------------------------------

test_file = ROOT / "testsubjects.txt"

subjects = []

for line in test_file.read_text(
    encoding="utf-8",
    errors="replace"
).splitlines():

    line = line.strip()

    if not line:
        continue

    try:
        name = ast.literal_eval(line)
    except Exception:
        name = line.strip("'\"")

    subjects.append(name)

print("=" * 105)
print("LIU OFFICIAL TEST-SET STRUCTURE AUDIT")
print("=" * 105)

print("Official test subjects:", len(subjects))

rows = []
run_rows = []
schemas = []

for si, fname in enumerate(subjects, 1):

    p = DATA / fname

    if not p.exists():
        print("MISSING:", p)
        continue

    m = sio.loadmat(
        p,
        squeeze_me=True,
        struct_as_record=False
    )

    keys = [
        k for k in m.keys()
        if not k.startswith("__")
    ]

    if "ppgseg" not in m or "labels" not in m:
        print(
            "BAD SCHEMA:",
            fname,
            keys
        )
        continue

    ppg = np.asarray(m["ppgseg"])
    labels = np.asarray(
        m["labels"]
    ).reshape(-1)

    # normalize orientation:
    # expected N segments x 1000 samples
    if ppg.ndim == 1:
        ppg = ppg.reshape(1, -1)

    if (
        ppg.shape[0] != len(labels)
        and
        ppg.shape[1] == len(labels)
    ):
        ppg = ppg.T

    schemas.append({
        "subject": fname,
        "ppg_shape": str(ppg.shape),
        "labels_shape": str(labels.shape),
        "keys": ",".join(keys),
    })

    print(
        f"[{si:02d}/{len(subjects)}] "
        f"{fname}: "
        f"ppg={ppg.shape}, "
        f"labels={labels.shape}"
    )

    # --------------------------------------------------------
    # segment rows
    # --------------------------------------------------------

    for i, lab in enumerate(labels):

        lab = int(lab)

        x = np.asarray(
            ppg[i],
            dtype=float
        ).reshape(-1)

        rows.append({
            "subject":
                fname.replace(".mat", ""),

            "file":
                fname,

            "segment_index":
                i,

            "label":
                lab,

            "rhythm":
                LABELS.get(
                    lab,
                    f"UNKNOWN_{lab}"
                ),

            "samples":
                len(x),

            "mean":
                float(np.nanmean(x)),

            "std":
                float(np.nanstd(x)),

            "min":
                float(np.nanmin(x)),

            "max":
                float(np.nanmax(x)),

            "first":
                float(x[0]),

            "last":
                float(x[-1]),
        })

    # --------------------------------------------------------
    # same-rhythm contiguous runs based on stored row order
    # --------------------------------------------------------

    start = 0

    for i in range(1, len(labels) + 1):

        boundary = (
            i == len(labels)
            or labels[i] != labels[start]
        )

        if not boundary:
            continue

        lab = int(labels[start])

        run_rows.append({
            "subject":
                fname.replace(".mat", ""),

            "label":
                lab,

            "rhythm":
                LABELS.get(
                    lab,
                    f"UNKNOWN_{lab}"
                ),

            "start_index":
                start,

            "end_index":
                i - 1,

            "segments":
                i - start,

            "duration_if_contiguous_s":
                (i - start) * 10,
        })

        start = i


df = pd.DataFrame(rows)
runs = pd.DataFrame(run_rows)
schemas = pd.DataFrame(schemas)

print("\n" + "=" * 105)
print("[1] DATASET SIZE")
print("=" * 105)

print("subjects:", df.subject.nunique())
print("segments:", len(df))

print("\nLABEL COUNTS")
print(
    df["rhythm"]
    .value_counts()
    .to_string()
)

print("\nSUBJECTS PER RHYTHM")
print(
    df.groupby("rhythm")
      ["subject"]
      .nunique()
      .sort_values(
          ascending=False
      )
      .to_string()
)

print("\n" + "=" * 105)
print("[2] SAMPLES PER SEGMENT")
print("=" * 105)

print(
    df["samples"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n" + "=" * 105)
print("[3] SUBJECT x RHYTHM")
print("=" * 105)

table = (
    df.groupby(
        ["subject", "rhythm"]
    )
    .size()
    .unstack(fill_value=0)
)

print(table.to_string())

print("\n" + "=" * 105)
print("[4] SAME-LABEL STORED-ORDER RUNS")
print("=" * 105)

print(
    runs.groupby("rhythm")
    .agg(
        runs=("segments", "size"),
        median_segments=(
            "segments",
            "median"
        ),
        max_segments=(
            "segments",
            "max"
        ),
        runs_ge_3=(
            "segments",
            lambda x:
                int((x >= 3).sum())
        ),
    )
    .to_string()
)

print("\nLONGEST AF RUNS")
print(
    runs[
        runs.rhythm == "AF"
    ]
    .sort_values(
        "segments",
        ascending=False
    )
    .head(30)
    .to_string(
        index=False
    )
)

# ------------------------------------------------------------
# Boundary-value diagnostic
#
# This does NOT prove temporal continuity.
# It only tells us whether segments appear independently scaled.
# ------------------------------------------------------------

boundaries = []

for subject, g in df.groupby(
    "subject",
    sort=False
):

    g = g.sort_values(
        "segment_index"
    )

    prev = None

    for r in g.itertuples():

        if prev is not None:

            boundaries.append({
                "subject":
                    subject,

                "prev_index":
                    prev.segment_index,

                "cur_index":
                    r.segment_index,

                "same_label":
                    prev.label == r.label,

                "boundary_jump":
                    abs(
                        r.first
                        -
                        prev.last
                    ),

                "prev_std":
                    prev.std,

                "cur_std":
                    r.std,
            })

        prev = r

b = pd.DataFrame(boundaries)

print("\n" + "=" * 105)
print("[5] ADJACENT-STORED-SEGMENT BOUNDARY AUDIT")
print("=" * 105)

if len(b):

    print(
        b["boundary_jump"]
        .describe(
            percentiles=[
                .1, .5, .9, .99
            ]
        )
    )

    print("\nSAME LABEL")
    print(
        b[
            b.same_label
        ]["boundary_jump"]
        .describe(
            percentiles=[
                .1, .5, .9, .99
            ]
        )
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df.to_csv(
    OUT /
    "liu_official_test_manifest.csv",
    index=False
)

runs.to_csv(
    OUT /
    "liu_official_test_label_runs.csv",
    index=False
)

schemas.to_csv(
    OUT /
    "liu_official_test_schema.csv",
    index=False
)

b.to_csv(
    OUT /
    "liu_official_test_boundaries.csv",
    index=False
)

print("\n" + "=" * 105)
print("SAVED")
print("=" * 105)

for name in [
    "liu_official_test_manifest.csv",
    "liu_official_test_label_runs.csv",
    "liu_official_test_schema.csv",
    "liu_official_test_boundaries.csv",
]:
    print(OUT / name)
