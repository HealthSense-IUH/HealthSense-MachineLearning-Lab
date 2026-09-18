from pathlib import Path
import gc
import numpy as np
import pandas as pd

ROOT = Path("data/raw")
OUT = Path("experiments/07_v6_multidomain/artifacts")
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": ROOT / "train.npz",
    "validate": ROOT / "validate.npz",
    "test": ROOT / "test.npz",
}

subject_sets = {}
summary_all = []

for split, path in FILES.items():

    print()
    print("=" * 100)
    print(split.upper(), path)
    print("=" * 100)

    with np.load(path, allow_pickle=True) as z:

        # --------------------------------------------------
        # SUBJECT
        # --------------------------------------------------
        params = z["parameters"]

        subjects = np.asarray(
            params[:, 2],
            dtype=np.int32
        )

        del params
        gc.collect()

        # --------------------------------------------------
        # RHYTHM
        # --------------------------------------------------
        rhythm = z["rhythm"]

        rhythm_idx = np.argmax(
            rhythm,
            axis=1
        ).astype(np.int8)

        del rhythm
        gc.collect()

        # --------------------------------------------------
        # QUALITY
        # --------------------------------------------------
        qa = z["qa_label"]

        qa_idx = np.argmax(
            qa,
            axis=1
        ).astype(np.int8)

        del qa
        gc.collect()

    if not (
        len(subjects)
        == len(rhythm_idx)
        == len(qa_idx)
    ):
        raise RuntimeError(
            f"Length mismatch in {split}"
        )

    unique_subjects = np.unique(subjects)

    subject_sets[split] = set(
        unique_subjects.tolist()
    )

    print("Rows:", len(subjects))
    print(
        "Subjects:",
        len(unique_subjects)
    )
    print(
        "Subject IDs:",
        unique_subjects.tolist()
    )

    print("\nRhythm raw argmax counts:")
    vals, cnts = np.unique(
        rhythm_idx,
        return_counts=True
    )

    for v, c in zip(vals, cnts):
        print(
            f"  rhythm_col_{v}: {c:,}"
        )

    print("\nQA raw argmax counts:")
    vals, cnts = np.unique(
        qa_idx,
        return_counts=True
    )

    for v, c in zip(vals, cnts):
        print(
            f"  qa_col_{v}: {c:,}"
        )

    df = pd.DataFrame({
        "subject_id": subjects,
        "rhythm_idx": rhythm_idx,
        "qa_idx": qa_idx,
    })

    # Per-subject total
    total = (
        df.groupby("subject_id")
        .size()
        .rename("n_windows")
        .reset_index()
    )

    # Rhythm counts
    rhythm_tab = pd.crosstab(
        df["subject_id"],
        df["rhythm_idx"]
    )

    rhythm_tab.columns = [
        f"rhythm_{int(c)}"
        for c in rhythm_tab.columns
    ]

    rhythm_tab = (
        rhythm_tab
        .reset_index()
    )

    # QA counts
    qa_tab = pd.crosstab(
        df["subject_id"],
        df["qa_idx"]
    )

    qa_tab.columns = [
        f"qa_{int(c)}"
        for c in qa_tab.columns
    ]

    qa_tab = (
        qa_tab
        .reset_index()
    )

    sub = (
        total
        .merge(
            rhythm_tab,
            on="subject_id",
            how="left"
        )
        .merge(
            qa_tab,
            on="subject_id",
            how="left"
        )
    )

    sub.insert(
        0,
        "split",
        split
    )

    sub.to_csv(
        OUT /
        f"deepbeat_{split}_subject_summary.csv",
        index=False
    )

    summary_all.append(sub)

    print("\nPer-subject:")
    print(
        sub.to_string(
            index=False
        )
    )

    del df
    del subjects
    del rhythm_idx
    del qa_idx
    gc.collect()


# ============================================================
# SUBJECT OVERLAP
# ============================================================

print()
print("=" * 100)
print("SUBJECT OVERLAP")
print("=" * 100)

pairs = [
    ("train", "validate"),
    ("train", "test"),
    ("validate", "test"),
]

overlap_rows = []

for a, b in pairs:

    overlap = sorted(
        subject_sets[a]
        & subject_sets[b]
    )

    print(
        f"{a} ∩ {b}: "
        f"{len(overlap)} subjects"
    )

    print(overlap)

    for uid in overlap:
        overlap_rows.append({
            "split_a": a,
            "split_b": b,
            "subject_id": uid,
        })

pd.DataFrame(
    overlap_rows
).to_csv(
    OUT /
    "deepbeat_subject_overlap_all_splits.csv",
    index=False
)

all_summary = pd.concat(
    summary_all,
    ignore_index=True
)

all_summary.to_csv(
    OUT /
    "deepbeat_all_subject_summary.csv",
    index=False
)

print()
print("Saved:")
print(
    OUT /
    "deepbeat_all_subject_summary.csv"
)
print(
    OUT /
    "deepbeat_subject_overlap_all_splits.csv"
)
