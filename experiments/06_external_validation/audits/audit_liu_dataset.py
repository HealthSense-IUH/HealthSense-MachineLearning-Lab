from pathlib import Path
from collections import Counter

import numpy as np
from scipy.io import loadmat, whosmat


ROOT = Path(
    "data/raw/PPGArrhythmiaDetection-main"
)

DATA_DIR = ROOT / "valid_testDataset"

LABELS = {
    0: "SR",
    1: "PVC",
    2: "PAC",
    3: "VT",
    4: "SVT",
    5: "AF",
}


def read_subject_list(path):
    return [
        line.strip().strip("'").strip('"')
        for line in path.read_text().splitlines()
        if line.strip()
    ]


valid = read_subject_list(
    ROOT / "validsubjects.txt"
)

test = read_subject_list(
    ROOT / "testsubjects.txt"
)

files = sorted(
    DATA_DIR.glob("*.mat")
)


print("=" * 90)
print("LIU DATASET STRUCTURE AUDIT")
print("=" * 90)

print("Dataset files      :", len(files))
print("Validation subjects:", len(valid))
print("Test subjects      :", len(test))
print(
    "Overlap valid/test :",
    len(set(valid) & set(test)),
)

print(
    "Split union        :",
    len(set(valid) | set(test)),
)

print(
    "Files == split     :",
    {x.name for x in files}
    == (set(valid) | set(test)),
)


# ------------------------------------------------------
# Inspect first subject
# ------------------------------------------------------

first = files[0]

print()
print("=" * 90)
print("FIRST FILE")
print("=" * 90)

print("File:", first.name)

print(
    "MAT variables:",
    whosmat(first),
)

sample = loadmat(
    first,
    squeeze_me=True,
)

print(
    "Keys:",
    [
        k
        for k in sample.keys()
        if not k.startswith("__")
    ],
)


# ------------------------------------------------------
# Whole-dataset audit
# ------------------------------------------------------

total_counts = Counter()

valid_counts = Counter()
test_counts = Counter()

subject_rows = []

shapes = Counter()

dtypes = Counter()

bad_labels = []

missing_keys = []

for i, path in enumerate(
    files,
    1,
):

    mat = loadmat(
        path,
        squeeze_me=True,
    )

    if (
        "ppgseg" not in mat
        or "labels" not in mat
    ):
        missing_keys.append(
            path.name
        )
        continue

    ppg = np.asarray(
        mat["ppgseg"]
    )

    labels = np.asarray(
        mat["labels"]
    ).reshape(-1)

    labels = labels.astype(int)

    # Record raw orientation before changing anything.
    shapes[str(ppg.shape)] += 1
    dtypes[str(ppg.dtype)] += 1

    # Detect which axis corresponds to 1000 samples.
    # Each segment should be 10 s × 100 Hz = 1000 samples.
    if ppg.ndim != 2:
        raise RuntimeError(
            f"{path.name}: "
            f"unexpected PPG shape {ppg.shape}"
        )

    if ppg.shape[1] == 1000:
        n_segments = ppg.shape[0]

    elif ppg.shape[0] == 1000:
        n_segments = ppg.shape[1]

    else:
        raise RuntimeError(
            f"{path.name}: "
            f"neither axis is 1000 samples: "
            f"{ppg.shape}"
        )

    if n_segments != len(labels):
        raise RuntimeError(
            f"{path.name}: "
            f"segments={n_segments}, "
            f"labels={len(labels)}"
        )

    unique = sorted(
        set(labels.tolist())
    )

    invalid = [
        x for x in unique
        if x not in LABELS
    ]

    if invalid:
        bad_labels.append(
            (path.name, invalid)
        )

    counts = Counter(
        labels.tolist()
    )

    total_counts.update(
        counts
    )

    split = (
        "validation"
        if path.name in valid
        else "test"
    )

    if split == "validation":
        valid_counts.update(
            counts
        )
    else:
        test_counts.update(
            counts
        )

    subject_rows.append({
        "subject":
            path.stem,

        "split":
            split,

        "segments":
            n_segments,

        **{
            LABELS[k]:
                counts.get(k, 0)
            for k in LABELS
        },
    })


print()
print("=" * 90)
print("PPG FORMAT")
print("=" * 90)

print("Observed shapes:")

for shape, n in (
    shapes.most_common()
):
    print(
        f"  {shape}: "
        f"{n} subjects"
    )

print(
    "Dtypes:",
    dict(dtypes),
)


def print_counts(
    title,
    counts,
):
    print()
    print(title)

    total = sum(
        counts.values()
    )

    print(
        "Total segments:",
        total,
    )

    for k, name in LABELS.items():

        n = counts.get(
            k,
            0,
        )

        pct = (
            100 * n / total
            if total
            else 0
        )

        print(
            f"{k} {name:4s}: "
            f"{n:6d} "
            f"({pct:6.2f}%)"
        )


print()
print("=" * 90)
print("CLASS DISTRIBUTION")
print("=" * 90)

print_counts(
    "ALL",
    total_counts,
)

print_counts(
    "VALIDATION",
    valid_counts,
)

print_counts(
    "TEST",
    test_counts,
)


print()
print("=" * 90)
print("SUBJECT DISTRIBUTION")
print("=" * 90)

segments_per_subject = np.array(
    [
        r["segments"]
        for r in subject_rows
    ]
)

print(
    "min    :",
    int(
        segments_per_subject.min()
    ),
)

print(
    "median :",
    float(
        np.median(
            segments_per_subject
        )
    ),
)

print(
    "mean   :",
    float(
        segments_per_subject.mean()
    ),
)

print(
    "max    :",
    int(
        segments_per_subject.max()
    ),
)


for k, name in LABELS.items():

    subjects_with_class = sum(
        r[name] > 0
        for r in subject_rows
    )

    print(
        f"Subjects with {name:4s}: "
        f"{subjects_with_class}"
    )


print()
print("=" * 90)
print("VALIDATION CHECKS")
print("=" * 90)

print(
    "Missing required keys:",
    missing_keys,
)

print(
    "Invalid labels:",
    bad_labels,
)

print(
    "Expected samples/segment:",
    1000,
)

print(
    "Expected sampling rate:",
    "100 Hz",
)

print(
    "Expected duration:",
    "10 s",
)
