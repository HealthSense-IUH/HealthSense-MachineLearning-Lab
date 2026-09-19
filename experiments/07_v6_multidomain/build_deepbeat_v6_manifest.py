from pathlib import Path
import gc
import numpy as np
import pandas as pd

ROOT = Path("data/raw")
OUT = Path("experiments/07_v6_multidomain/artifacts")
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_SECONDS = 25
WINDOW_NS = WINDOW_SECONDS * 1_000_000_000

RHYTHM_NAMES = {
    0: "NON_AF",
    1: "AF",
}

QUALITY_NAMES = {
    0: "poor",
    1: "acceptable",
    2: "excellent",
}


def load_metadata(path, split):

    print()
    print("=" * 100)
    print(f"LOAD {split.upper()}: {path}")
    print("=" * 100)

    with np.load(
        path,
        allow_pickle=True,
    ) as z:

        rhythm = z["rhythm"]
        qa = z["qa_label"]
        params = z["parameters"]

        y = np.argmax(
            rhythm,
            axis=1
        ).astype(np.int8)

        q = np.argmax(
            qa,
            axis=1
        ).astype(np.int8)

        subjects = (
            params[:, 2]
            .astype(np.int32)
        )

        sources = np.asarray([
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
                dtype=np.int64
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

    del rhythm
    del qa
    del params
    del subjects
    del sources
    del timestamps
    del y
    del q
    gc.collect()

    return df


def temporal_thin(df):

    print(
        f"Temporal thinning: minimum "
        f"{WINDOW_SECONDS}s start-time separation"
    )

    # Sort by physical recording stream.
    d = (
        df.sort_values(
            [
                "subject_id",
                "source",
                "timestamp",
                "global_index",
            ]
        )
        .reset_index(drop=True)
    )

    subject = d["subject_id"].to_numpy()
    source = d["source"].to_numpy()
    ts = d["timestamp"].astype("int64").to_numpy()

    keep = np.zeros(
        len(d),
        dtype=bool
    )

    last_subject = None
    last_source = None
    last_timestamp = None

    for i in range(len(d)):

        uid = subject[i]
        src = source[i]
        t = ts[i]

        new_stream = (
            uid != last_subject
            or src != last_source
        )

        if new_stream:

            keep[i] = True

            last_subject = uid
            last_source = src
            last_timestamp = t

            continue

        if (
            t - last_timestamp
            >= WINDOW_NS
        ):

            keep[i] = True
            last_timestamp = t

    out = (
        d.loc[keep]
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"Before : {len(df):,}"
    )

    print(
        f"After  : {len(out):,}"
    )

    print(
        "Reduction:",
        f"{100 * (1 - len(out)/len(df)):.2f}%"
    )

    return out


def print_summary(name, df):

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    print(
        "Windows:",
        f"{len(df):,}"
    )

    print(
        "Subjects:",
        df.subject_id.nunique()
    )

    print(
        "Subject IDs:",
        sorted(
            df.subject_id.unique()
            .tolist()
        )
    )

    print("\nRhythm:")
    print(
        df.rhythm
        .value_counts()
        .to_string()
    )

    print("\nQuality:")
    print(
        df.quality_name
        .value_counts()
        .to_string()
    )

    print("\nRhythm x quality:")
    print(
        pd.crosstab(
            df.rhythm,
            df.quality_name
        ).to_string()
    )

    print("\nPer-subject window count:")
    print(
        df.groupby(
            ["rhythm", "subject_id"]
        )
        .size()
        .groupby(level=0)
        .describe()
        .to_string()
    )


def process(
    split,
    filename,
    subject_filter=None,
):

    df = load_metadata(
        ROOT / filename,
        split
    )

    if subject_filter is not None:

        before = len(df)

        df = (
            df[
                df.subject_id.isin(
                    subject_filter
                )
            ]
            .copy()
        )

        print()
        print(
            f"Subject filter: "
            f"{before:,} -> {len(df):,}"
        )

    print_summary(
        f"{split.upper()} RAW",
        df
    )

    thin = temporal_thin(df)

    print_summary(
        f"{split.upper()} THIN25",
        thin
    )

    all_path = (
        OUT /
        f"deepbeat_{split}_thin25_all.csv"
    )

    thin.to_csv(
        all_path,
        index=False
    )

    # AF classifier operates downstream of SQI:
    # remove DeepBeat ground-truth "poor" windows.
    eligible = (
        thin[
            thin.quality >= 1
        ]
        .copy()
        .reset_index(drop=True)
    )

    eligible_path = (
        OUT /
        f"deepbeat_{split}_v6_classifier.csv"
    )

    eligible.to_csv(
        eligible_path,
        index=False
    )

    print_summary(
        f"{split.upper()} CLASSIFIER ELIGIBLE "
        "(acceptable + excellent)",
        eligible
    )

    print()
    print("Saved:")
    print(all_path)
    print(eligible_path)

    return thin, eligible


# ============================================================
# TRAIN
# subjects 1-137
# ============================================================

train_all, train_cls = process(
    "train",
    "train.npz",
)


# ============================================================
# VALIDATION
# subjects 138-153
# subject-independent from TRAIN
# ============================================================

val_all, val_cls = process(
    "validate",
    "validate.npz",
)


# ============================================================
# FINAL TEST
# ONLY subjects 154-167
#
# 146-153 overlap validate and MUST NOT be used
# for the v6 independent final test.
# ============================================================

test_subjects = set(
    range(154, 168)
)

test_all, test_cls = process(
    "test_independent",
    "test.npz",
    subject_filter=test_subjects,
)


# ============================================================
# LEAKAGE ASSERTIONS
# ============================================================

train_subjects = set(
    train_cls.subject_id.unique()
)

val_subjects = set(
    val_cls.subject_id.unique()
)

test_subjects_actual = set(
    test_cls.subject_id.unique()
)

assert not (
    train_subjects
    & val_subjects
), "TRAIN / VALIDATE subject leakage"

assert not (
    train_subjects
    & test_subjects_actual
), "TRAIN / TEST subject leakage"

assert not (
    val_subjects
    & test_subjects_actual
), "VALIDATE / TEST subject leakage"


print()
print("=" * 100)
print("FINAL SUBJECT-INDEPENDENCE CHECK")
print("=" * 100)

print(
    "TRAIN ∩ VALIDATE:",
    sorted(
        train_subjects
        & val_subjects
    )
)

print(
    "TRAIN ∩ TEST:",
    sorted(
        train_subjects
        & test_subjects_actual
    )
)

print(
    "VALIDATE ∩ TEST:",
    sorted(
        val_subjects
        & test_subjects_actual
    )
)

print()
print("TRAIN subjects     :", len(train_subjects))
print("VALIDATE subjects  :", len(val_subjects))
print("FINAL TEST subjects:", len(test_subjects_actual))

print()
print("DONE")
