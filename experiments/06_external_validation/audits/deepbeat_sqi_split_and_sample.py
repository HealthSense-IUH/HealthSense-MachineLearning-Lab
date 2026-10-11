from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

VALIDATE = Path(
    "data/raw/validate.npz"
)

TEST = Path(
    "data/raw/test.npz"
)

OUT = Path(
    "experiments/06_external_validation"
)

SEED = 20260916

# Maximum selected validation segments per:
# rhythm x quality cell
TARGET_PER_CELL = 2000

QUALITY_NAMES = {
    0: "poor",
    1: "acceptable",
    2: "excellent",
}

RHYTHM_NAMES = {
    0: "NON_AF",
    1: "AF",
}


# ============================================================
# LOAD METADATA ONLY
# ============================================================

def load_metadata(path, split):

    print(
        f"Loading metadata: {path}"
    )

    with np.load(
        path,
        allow_pickle=True,
    ) as z:

        qa = z["qa_label"]
        rhythm = z["rhythm"]
        params = z["parameters"]

    q = np.argmax(
        qa,
        axis=1,
    ).astype(int)

    y = np.argmax(
        rhythm,
        axis=1,
    ).astype(int)

    subjects = (
        params[:, 2]
        .astype(int)
    )

    sources = np.array([
        str(x).strip()
        for x in params[:, 1]
    ])

    timestamps = pd.to_datetime(
        params[:, 0]
    )

    df = pd.DataFrame({
        "split":
            split,

        "global_index":
            np.arange(
                len(y),
                dtype=int,
            ),

        "subject_id":
            subjects,

        "source":
            sources,

        "timestamp":
            timestamps,

        "label":
            y,

        "rhythm":
            [
                RHYTHM_NAMES[int(x)]
                for x in y
            ],

        "quality":
            q,

        "quality_name":
            [
                QUALITY_NAMES[int(x)]
                for x in q
            ],
    })

    return df


val = load_metadata(
    VALIDATE,
    "validate",
)

test = load_metadata(
    TEST,
    "test",
)


# ============================================================
# SUBJECT OVERLAP
# ============================================================

val_subjects = set(
    val.subject_id.unique()
)

test_subjects = set(
    test.subject_id.unique()
)

overlap = sorted(
    val_subjects
    &
    test_subjects
)

only_val = sorted(
    val_subjects
    -
    test_subjects
)

only_test = sorted(
    test_subjects
    -
    val_subjects
)


print()
print("=" * 105)
print("DEEPBEAT VALIDATE vs TEST SUBJECT OVERLAP")
print("=" * 105)

print(
    "Validate subjects:",
    len(val_subjects)
)

print(
    sorted(
        val_subjects
    )
)

print(
    "\nTest subjects:",
    len(test_subjects)
)

print(
    sorted(
        test_subjects
    )
)

print(
    "\nOVERLAP:",
    len(overlap)
)

print(overlap)

print(
    "\nValidate-only:",
    len(only_val),
    only_val,
)

print(
    "\nTest-only:",
    len(only_test),
    only_test,
)


# ============================================================
# LABEL OF EACH SUBJECT IN EACH SPLIT
# ============================================================

def subject_label_table(df):

    return (
        df.groupby("subject_id")
        .agg(
            segments=(
                "label",
                "size"
            ),

            af_segments=(
                "label",
                lambda x:
                    int(
                        (x == 1).sum()
                    )
            ),

            non_af_segments=(
                "label",
                lambda x:
                    int(
                        (x == 0).sum()
                    )
            ),

            poor=(
                "quality",
                lambda x:
                    int(
                        (x == 0).sum()
                    )
            ),

            acceptable=(
                "quality",
                lambda x:
                    int(
                        (x == 1).sum()
                    )
            ),

            excellent=(
                "quality",
                lambda x:
                    int(
                        (x == 2).sum()
                    )
            ),
        )
        .reset_index()
    )


val_sub = subject_label_table(
    val
)

test_sub = subject_label_table(
    test
)


print()
print("=" * 105)
print("OVERLAPPING SUBJECT DETAILS")
print("=" * 105)

if overlap:

    print("\nVALIDATE")

    print(
        val_sub[
            val_sub.subject_id.isin(
                overlap
            )
        ].to_string(
            index=False
        )
    )

    print("\nTEST")

    print(
        test_sub[
            test_sub.subject_id.isin(
                overlap
            )
        ].to_string(
            index=False
        )
    )


# ============================================================
# EXACT METADATA KEY OVERLAP
#
# Duplicate rows can share timestamps, so compare unique keys
# only to avoid a many-to-many merge explosion.
# ============================================================

if overlap:

    vkeys = (
        val[
            val.subject_id.isin(
                overlap
            )
        ]
        [
            [
                "subject_id",
                "source",
                "timestamp",
            ]
        ]
        .drop_duplicates()
    )

    tkeys = (
        test[
            test.subject_id.isin(
                overlap
            )
        ]
        [
            [
                "subject_id",
                "source",
                "timestamp",
            ]
        ]
        .drop_duplicates()
    )

    exact_keys = vkeys.merge(
        tkeys,
        on=[
            "subject_id",
            "source",
            "timestamp",
        ],
        how="inner",
    )

else:

    exact_keys = pd.DataFrame(
        columns=[
            "subject_id",
            "source",
            "timestamp",
        ]
    )


print()
print("=" * 105)
print("EXACT SUBJECT + SOURCE + TIMESTAMP OVERLAP")
print("=" * 105)

print(
    "Unique shared keys:",
    len(exact_keys)
)

if len(exact_keys):

    print(
        exact_keys
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# CLEAN TEST SET
#
# Remove every subject that occurs in validation.
# This will be the PRIMARY SQI evaluation cohort.
# ============================================================

test_independent = test[
    ~test.subject_id.isin(
        val_subjects
    )
].copy()


print()
print("=" * 105)
print("SUBJECT-INDEPENDENT TEST COHORT")
print("=" * 105)

print(
    "Segments:",
    len(test_independent)
)

print(
    "Subjects:",
    test_independent
    .subject_id
    .nunique()
)

print(
    "Subject IDs:",
    sorted(
        test_independent
        .subject_id
        .unique()
    )
)

print()
print(
    pd.crosstab(
        test_independent.rhythm,
        test_independent.quality_name,
        margins=True,
    ).to_string()
)


# ============================================================
# BALANCED VALIDATION SAMPLING
#
# Round-robin over subjects:
#   - prevents a large subject from dominating
#   - exhausts small subjects first if necessary
#   - then redistributes remaining quota
#
# This is for SQI DEVELOPMENT ONLY.
# ============================================================

rng = np.random.default_rng(
    SEED
)


def subject_balanced_sample(
    data,
    target,
):

    pools = {}

    for uid, g in data.groupby(
        "subject_id"
    ):

        arr = (
            g["global_index"]
            .to_numpy(
                dtype=int
            )
            .copy()
        )

        rng.shuffle(arr)

        pools[int(uid)] = {
            "values":
                arr,

            "pos":
                0,
        }

    subjects = sorted(
        pools.keys()
    )

    selected = []

    while (
        len(selected) < target
    ):

        active = [
            uid
            for uid in subjects
            if (
                pools[uid]["pos"]
                <
                len(
                    pools[uid][
                        "values"
                    ]
                )
            )
        ]

        if not active:
            break

        active = np.array(
            active,
            dtype=int,
        )

        rng.shuffle(
            active
        )

        for uid in active:

            if (
                len(selected)
                >= target
            ):
                break

            p = pools[
                int(uid)
            ]

            idx = p[
                "values"
            ][
                p["pos"]
            ]

            p["pos"] += 1

            selected.append(
                int(idx)
            )

    return selected


sample_parts = []

for rhythm in [
    "AF",
    "NON_AF",
]:

    for quality in [
        0,
        1,
        2,
    ]:

        cell = val[
            (val.rhythm == rhythm)
            &
            (val.quality == quality)
        ].copy()

        selected = (
            subject_balanced_sample(
                cell,
                TARGET_PER_CELL,
            )
        )

        part = val[
            val.global_index.isin(
                selected
            )
        ].copy()

        part[
            "sampling_cell"
        ] = (
            rhythm
            + "_"
            + QUALITY_NAMES[
                quality
            ].upper()
        )

        sample_parts.append(
            part
        )


sample = pd.concat(
    sample_parts,
    ignore_index=True,
)

sample = sample.sort_values(
    "global_index"
).reset_index(
    drop=True
)


# ============================================================
# SAMPLE AUDIT
# ============================================================

print()
print("=" * 105)
print("SQI VALIDATION SAMPLE")
print("=" * 105)

print(
    "Selected segments:",
    len(sample)
)

print(
    "Subjects:",
    sample.subject_id.nunique()
)

print()
print("CELL COUNTS")

print(
    sample.groupby(
        [
            "rhythm",
            "quality_name",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
    .to_string()
)


print()
print("SUBJECT COUNTS WITHIN EACH CELL")

cell_subject = (
    sample.groupby(
        [
            "rhythm",
            "quality_name",
            "subject_id",
        ]
    )
    .size()
    .rename("n")
    .reset_index()
)

print(
    cell_subject
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

overlap_df = pd.DataFrame({
    "subject_id":
        overlap
})

overlap_df.to_csv(
    OUT /
    "deepbeat_validate_test_subject_overlap.csv",
    index=False,
)

exact_keys.to_csv(
    OUT /
    "deepbeat_validate_test_exact_key_overlap.csv",
    index=False,
)

test_independent.to_csv(
    OUT /
    "deepbeat_test_subject_independent_manifest.csv",
    index=False,
)

sample.to_csv(
    OUT /
    "deepbeat_validate_sqi_sample_manifest.csv",
    index=False,
)


print()
print("=" * 105)
print("SAVED")
print("=" * 105)

for f in [
    "deepbeat_validate_test_subject_overlap.csv",
    "deepbeat_validate_test_exact_key_overlap.csv",
    "deepbeat_test_subject_independent_manifest.csv",
    "deepbeat_validate_sqi_sample_manifest.csv",
]:

    print(
        OUT / f
    )
