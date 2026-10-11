from pathlib import Path
import io
import tarfile
import time

import numpy as np
import pandas as pd
from scipy.io import whosmat, loadmat


ROOT = Path("experiments/06_external_validation")

ECG_DIR = Path(
    "data/raw/Pulsewatch/"
    "09_Aligned_ECG_LinearInterp"
)

PAC_FILE = ROOT / (
    "pulsewatch_v5_pac_false_alert_shortlist.csv"
)

AF_FILE = ROOT / (
    "pulsewatch_v5_af_miss_shortlist.csv"
)

OUT_MAP = ROOT / (
    "pulsewatch_v5_failure_ecg_mapping.csv"
)

OUT_SCHEMA = ROOT / (
    "pulsewatch_v5_failure_ecg_schema.csv"
)

PROGRESS_EVERY = 500


# ============================================================
# LOAD RUN SHORTLISTS
# ============================================================

pac = pd.read_csv(
    PAC_FILE,
    dtype={"uid": str},
)

af = pd.read_csv(
    AF_FILE,
    dtype={"uid": str},
)

pac["uid"] = (
    pac["uid"]
    .astype(str)
    .str.zfill(3)
)

af["uid"] = (
    af["uid"]
    .astype(str)
    .str.zfill(3)
)

pac["failure_type"] = (
    "PAC_PVC_FALSE_ALERT"
)

af["failure_type"] = (
    "AF_MISSED_ALERT"
)

runs = pd.concat(
    [
        pac,
        af,
    ],
    ignore_index=True,
)


# ============================================================
# EXPAND RUNS -> INDIVIDUAL SEGMENTS
# ============================================================

rows = []

for _, r in runs.iterrows():

    for idx in range(
        int(r["start_index"]),
        int(r["end_index"]) + 1,
    ):

        segment = (
            f"{r['session']}"
            f"_ppg_{idx:04d}"
        )

        member = (
            f"{r['uid']}/"
            f"{segment}"
            f"_final_align_output_LinearInterp.mat"
        )

        rows.append(
            {
                "failure_type":
                    r["failure_type"],

                "run_id":
                    r["run_id"],

                "uid":
                    r["uid"],

                "class_name":
                    r["class_name"],

                "session":
                    r["session"],

                "segment_index":
                    idx,

                "segment":
                    segment,

                "ecg_tar":
                    str(
                        ECG_DIR /
                        f"{r['uid']}.tar"
                    ),

                "ecg_member":
                    member,
            }
        )


segments = pd.DataFrame(
    rows
)

# Same segment may appear only once logically,
# but preserve provenance then deduplicate.
segments = (
    segments
    .drop_duplicates(
        subset=[
            "failure_type",
            "run_id",
            "segment",
        ]
    )
    .reset_index(drop=True)
)

TOTAL = len(segments)

print("=" * 90)
print("06A.4.6b PULSEWATCH FAILURE ECG MAPPING")
print("=" * 90)

print(
    "Failure runs:",
    f"{len(runs):,}",
)

print(
    "Expected ECG segments:",
    f"{TOTAL:,}",
)

print(
    "Subjects:",
    segments["uid"].nunique(),
)

print()

print(
    segments[
        "failure_type"
    ]
    .value_counts()
    .to_string()
)

print()


# ============================================================
# INDEX TAR MEMBERS
# ============================================================

tar_cache = {}

mapped = []

processed = 0
t0 = time.time()

for uid, g in segments.groupby(
    "uid",
    sort=True,
):

    tar_path = (
        ECG_DIR /
        f"{uid}.tar"
    )

    if not tar_path.exists():

        members = set()

    else:

        with tarfile.open(
            tar_path,
            "r",
        ) as tf:

            members = set(
                tf.getnames()
            )

    tar_cache[uid] = members

    for _, r in g.iterrows():

        exists = (
            r["ecg_member"]
            in members
        )

        rec = r.to_dict()

        rec[
            "tar_exists"
        ] = tar_path.exists()

        rec[
            "ecg_exists"
        ] = exists

        mapped.append(rec)

        processed += 1

        if (
            processed % PROGRESS_EVERY == 0
            or processed == TOTAL
        ):

            elapsed = (
                time.time()
                - t0
            )

            rate = (
                processed / elapsed
                if elapsed > 0
                else 0
            )

            pct = (
                processed
                / TOTAL
                * 100
            )

            eta = (
                (TOTAL - processed)
                / rate
                if rate > 0
                else np.nan
            )

            print(
                f"Mapped "
                f"{processed:,}/{TOTAL:,} "
                f"({pct:6.2f}%)"
                f" | "
                f"{rate:7.1f} seg/s"
                f" | ETA "
                f"{eta/60:5.1f} min",
                flush=True,
            )


mapping = pd.DataFrame(
    mapped
)

mapping.to_csv(
    OUT_MAP,
    index=False,
)


# ============================================================
# AVAILABILITY SUMMARY
# ============================================================

print()
print("=" * 90)
print("ECG AVAILABILITY")
print("=" * 90)

print(
    "Available:",
    f"{int(mapping.ecg_exists.sum()):,}",
    "/",
    f"{len(mapping):,}",
)

print(
    "Coverage :",
    f"{mapping.ecg_exists.mean()*100:.3f}%"
)

print()

print(
    pd.crosstab(
        mapping["failure_type"],
        mapping["ecg_exists"],
    ).to_string()
)


# ============================================================
# CHOOSE REPRESENTATIVE FILES FOR SCHEMA INSPECTION
#
# PAC:
#   UID 075 — dominant false-alert patient
#
# AF:
#   UID 413, 410, 416 — dominant misses
# ============================================================

priority = [
    ("PAC_PVC_FALSE_ALERT", "075"),
    ("AF_MISSED_ALERT", "413"),
    ("AF_MISSED_ALERT", "410"),
    ("AF_MISSED_ALERT", "416"),
]

samples = []

for failure_type, uid in priority:

    g = mapping[
        (mapping["failure_type"] == failure_type)
        &
        (mapping["uid"] == uid)
        &
        (mapping["ecg_exists"])
    ]

    if len(g):

        samples.append(
            g.iloc[0]
        )


# ============================================================
# MAT SCHEMA
# ============================================================

schema_rows = []

print()
print("=" * 90)
print("MAT FILE SCHEMA")
print("=" * 90)


for sample in samples:

    uid = sample["uid"]

    tar_path = Path(
        sample["ecg_tar"]
    )

    member = sample[
        "ecg_member"
    ]

    print()
    print(
        f"### {sample['failure_type']} "
        f"UID={uid}"
    )

    print(
        "Segment:",
        sample["segment"],
    )

    print(
        "Member :",
        member,
    )

    with tarfile.open(
        tar_path,
        "r",
    ) as tf:

        fobj = tf.extractfile(
            member
        )

        raw = fobj.read()

    # ----------------------------------------
    # First try standard MATLAB v5/v7 schema.
    # ----------------------------------------

    try:

        info = whosmat(
            io.BytesIO(raw)
        )

        print(
            "Variables:",
            len(info),
        )

        for name, shape, matlab_class in info:

            print(
                f"  {name:35s} "
                f"shape={str(shape):20s} "
                f"class={matlab_class}"
            )

            schema_rows.append(
                {
                    "failure_type":
                        sample[
                            "failure_type"
                        ],

                    "uid":
                        uid,

                    "segment":
                        sample[
                            "segment"
                        ],

                    "variable":
                        name,

                    "shape":
                        str(shape),

                    "matlab_class":
                        matlab_class,

                    "load_ok":
                        True,

                    "error":
                        "",
                }
            )

        # ------------------------------------
        # Inspect numeric arrays safely.
        # ------------------------------------

        mat = loadmat(
            io.BytesIO(raw),
            squeeze_me=True,
            struct_as_record=False,
        )

        print()
        print(
            "Numeric array preview:"
        )

        for name, value in mat.items():

            if name.startswith("__"):
                continue

            arr = np.asarray(
                value
            )

            if (
                np.issubdtype(
                    arr.dtype,
                    np.number,
                )
                and arr.size > 0
            ):

                finite = (
                    np.isfinite(arr)
                    if np.issubdtype(
                        arr.dtype,
                        np.floating,
                    )
                    else np.ones(
                        arr.shape,
                        dtype=bool,
                    )
                )

                finite_values = (
                    arr[finite]
                    if finite.any()
                    else np.array([])
                )

                print(
                    f"  {name:35s}"
                    f" dtype={arr.dtype}"
                    f" shape={arr.shape}"
                    f" size={arr.size}",
                    end="",
                )

                if (
                    finite_values.size
                    and
                    np.issubdtype(
                        finite_values.dtype,
                        np.number,
                    )
                ):

                    try:

                        print(
                            f" min="
                            f"{np.min(finite_values):.6g}"
                            f" max="
                            f"{np.max(finite_values):.6g}"
                        )

                    except Exception:

                        print()

                else:

                    print()

    except Exception as e:

        print(
            "MAT read ERROR:",
            repr(e),
        )

        schema_rows.append(
            {
                "failure_type":
                    sample[
                        "failure_type"
                    ],

                "uid":
                    uid,

                "segment":
                    sample[
                        "segment"
                    ],

                "variable":
                    "",

                "shape":
                    "",

                "matlab_class":
                    "",

                "load_ok":
                    False,

                "error":
                    repr(e),
            }
        )


schema = pd.DataFrame(
    schema_rows
)

schema.to_csv(
    OUT_SCHEMA,
    index=False,
)


print()
print("=" * 90)
print("SAVED")
print("=" * 90)

print(
    OUT_MAP
)

print(
    OUT_SCHEMA
)
