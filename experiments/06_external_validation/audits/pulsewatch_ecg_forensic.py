from pathlib import Path
from datetime import datetime
import io
import os
import re
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat


ROOT = Path(
    "experiments/06_external_validation"
)

ECG_DIR = Path(
    "data/raw/Pulsewatch/"
    "09_Aligned_ECG_LinearInterp"
)

MAPPING = ROOT / (
    "pulsewatch_v5_failure_ecg_mapping.csv"
)

OUT_NEAREST = ROOT / (
    "pulsewatch_v5_failure_ecg_nearest.csv"
)

OUT_ARCHIVE = ROOT / (
    "pulsewatch_ecg_archive_inventory.csv"
)

OUT_SCHEMA = ROOT / (
    "pulsewatch_ecg_mat_schema_v2.csv"
)


TARGET_UIDS = [
    "075",  # dominant PAC/PVC false alerts
    "017",  # AF miss with exact ECG
    "410",
    "413",
    "416",
]


# ============================================================
# SEGMENT PARSER
# ============================================================

SEG_RE = re.compile(
    r"(?P<uid>\d{3})_"
    r"(?P<dt>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})"
    r"_ppg_(?P<idx>\d{4})"
)


def parse_segment_name(text):

    m = SEG_RE.search(
        str(text)
    )

    if not m:
        return None

    try:
        dt = datetime.strptime(
            m.group("dt"),
            "%Y_%m_%d_%H_%M_%S",
        )

    except Exception:
        return None

    return {
        "uid":
            m.group("uid"),

        "dt":
            dt,

        "segment_index":
            int(
                m.group("idx")
            ),
    }


# ============================================================
# LOAD FAILURE MAPPING
# ============================================================

mapping = pd.read_csv(
    MAPPING,
    dtype={"uid": str},
    low_memory=False,
)

mapping["uid"] = (
    mapping["uid"]
    .astype(str)
    .str.zfill(3)
)


print("=" * 90)
print("06A.4.6c PULSEWATCH ECG FORENSIC AUDIT")
print("=" * 90)

print(
    "Failure mapping rows:",
    f"{len(mapping):,}"
)

print(
    "Exact ECG:",
    f"{int(mapping.ecg_exists.sum()):,}",
)

print()


# ============================================================
# BUILD COMPLETE ECG ARCHIVE INVENTORY
# ============================================================

inventory_rows = []

for uid_i, uid in enumerate(
    TARGET_UIDS,
    1,
):

    tar_path = (
        ECG_DIR /
        f"{uid}.tar"
    )

    print(
        f"[ARCHIVE {uid_i}/{len(TARGET_UIDS)}] "
        f"UID {uid}"
    )

    if not tar_path.exists():

        print(
            "  TAR MISSING:",
            tar_path,
        )

        continue

    with tarfile.open(
        tar_path,
        "r",
    ) as tf:

        members = [
            x
            for x in tf.getnames()
            if x.endswith(".mat")
        ]

    print(
        "  MAT members:",
        f"{len(members):,}",
    )

    print(
        "  First members:"
    )

    for x in members[:5]:
        print(
            "   ",
            x,
        )

    for member in members:

        parsed = parse_segment_name(
            member
        )

        if parsed is None:
            continue

        inventory_rows.append(
            {
                "uid":
                    uid,

                "member":
                    member,

                "ecg_datetime":
                    parsed["dt"],

                "segment_index":
                    parsed[
                        "segment_index"
                    ],
            }
        )


inventory = pd.DataFrame(
    inventory_rows
)

inventory.to_csv(
    OUT_ARCHIVE,
    index=False,
)


# ============================================================
# NEAREST-MATCH AUDIT
#
# Important:
# diagnostic only.
# We are NOT substituting nearest ECG as reference yet.
# ============================================================

target = mapping[
    mapping["uid"].isin(
        TARGET_UIDS
    )
].copy()

TOTAL = len(target)

nearest_rows = []

t0 = time.time()

for n, (_, r) in enumerate(
    target.iterrows(),
    1,
):

    parsed = parse_segment_name(
        r["segment"]
    )

    rec = r.to_dict()

    rec[
        "expected_datetime"
    ] = (
        parsed["dt"]
        if parsed
        else pd.NaT
    )

    rec[
        "expected_segment_index"
    ] = (
        parsed["segment_index"]
        if parsed
        else np.nan
    )

    rec[
        "nearest_same_index_member"
    ] = ""

    rec[
        "nearest_same_index_delta_s"
    ] = np.nan

    rec[
        "nearest_any_member"
    ] = ""

    rec[
        "nearest_any_delta_s"
    ] = np.nan


    if parsed is not None:

        inv = inventory[
            inventory["uid"]
            == r["uid"]
        ]

        if len(inv):

            # --------------------------------
            # SAME PPG INDEX
            # --------------------------------

            same_idx = inv[
                inv[
                    "segment_index"
                ]
                ==
                parsed[
                    "segment_index"
                ]
            ].copy()

            if len(same_idx):

                dt_sec = (
                    same_idx[
                        "ecg_datetime"
                    ]
                    - parsed["dt"]
                ).dt.total_seconds().abs()

                j = dt_sec.idxmin()

                rec[
                    "nearest_same_index_member"
                ] = same_idx.loc[
                    j,
                    "member",
                ]

                rec[
                    "nearest_same_index_delta_s"
                ] = float(
                    dt_sec.loc[j]
                )

            # --------------------------------
            # ANY SEGMENT INDEX
            # --------------------------------

            dt_sec = (
                inv[
                    "ecg_datetime"
                ]
                - parsed["dt"]
            ).dt.total_seconds().abs()

            j = dt_sec.idxmin()

            rec[
                "nearest_any_member"
            ] = inv.loc[
                j,
                "member",
            ]

            rec[
                "nearest_any_delta_s"
            ] = float(
                dt_sec.loc[j]
            )


    nearest_rows.append(
        rec
    )


    if (
        n % 500 == 0
        or n == TOTAL
    ):

        elapsed = (
            time.time()
            - t0
        )

        rate = (
            n / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"Nearest audit "
            f"{n:,}/{TOTAL:,} "
            f"({n/TOTAL*100:6.2f}%)"
            f" | {rate:7.1f} seg/s",
            flush=True,
        )


nearest = pd.DataFrame(
    nearest_rows
)

nearest.to_csv(
    OUT_NEAREST,
    index=False,
)


# ============================================================
# NEAREST-MATCH SUMMARY
# ============================================================

print()
print("=" * 90)
print("NEAREST ECG MATCH BY UID")
print("=" * 90)

for uid in TARGET_UIDS:

    g = nearest[
        nearest["uid"]
        == uid
    ]

    if len(g) == 0:
        continue

    print()
    print(
        "UID",
        uid,
        "rows =",
        len(g),
    )

    print(
        "  exact =",
        int(
            g["ecg_exists"]
            .sum()
        ),
    )

    same = g[
        "nearest_same_index_delta_s"
    ].dropna()

    anyd = g[
        "nearest_any_delta_s"
    ].dropna()

    if len(same):

        print(
            "  nearest SAME index delta(s):"
        )

        print(
            same.describe(
                percentiles=[
                    .50,
                    .90,
                    .99,
                ]
            ).to_string()
        )

        for limit in [
            1,
            5,
            60,
            600,
            3600,
        ]:

            print(
                f"    <= {limit:4d}s :",
                int(
                    (same <= limit)
                    .sum()
                ),
                "/",
                len(same),
            )

    if len(anyd):

        print(
            "  nearest ANY delta(s):"
        )

        print(
            anyd.describe(
                percentiles=[
                    .50,
                    .90,
                    .99,
                ]
            ).to_string()
        )


# ============================================================
# MAT FILE FORMAT / SCHEMA
# ============================================================

def inspect_mat(
    uid,
    member,
    label,
):

    tar_path = (
        ECG_DIR /
        f"{uid}.tar"
    )

    print()
    print("=" * 90)
    print(
        "MAT INSPECTION:",
        label,
    )
    print("=" * 90)

    print(
        "UID   :",
        uid,
    )

    print(
        "Member:",
        member,
    )

    with tarfile.open(
        tar_path,
        "r",
    ) as tf:

        fobj = tf.extractfile(
            member
        )

        if fobj is None:
            raise RuntimeError(
                "Cannot extract MAT"
            )

        raw = fobj.read()


    print(
        "Bytes :",
        f"{len(raw):,}",
    )

    print(
        "Header:",
        repr(
            raw[:128]
        )
    )


    if raw.startswith(
        b"\x89HDF\r\n\x1a\n"
    ):

        file_format = (
            "HDF5 / MATLAB v7.3"
        )

    elif raw.startswith(
        b"MATLAB"
    ):

        file_format = (
            "MATLAB v5/v6/v7"
        )

    else:

        file_format = (
            "unknown"
        )


    print(
        "Detected format:",
        file_format,
    )


    # Use a REAL temporary file instead of
    # BytesIO. This avoids file-like quirks.
    with tempfile.NamedTemporaryFile(
        suffix=".mat",
        delete=False,
    ) as tmp:

        tmp.write(raw)

        temp_name = (
            tmp.name
        )


    schema_rows = []

    try:

        if file_format.startswith(
            "HDF5"
        ):

            try:

                import h5py

                with h5py.File(
                    temp_name,
                    "r",
                ) as h5:

                    print(
                        "HDF5 objects:"
                    )

                    def visitor(
                        name,
                        obj,
                    ):

                        shape = getattr(
                            obj,
                            "shape",
                            None,
                        )

                        dtype = getattr(
                            obj,
                            "dtype",
                            None,
                        )

                        print(
                            f"  {name:40s}"
                            f" shape={shape}"
                            f" dtype={dtype}"
                        )

                        schema_rows.append(
                            {
                                "label":
                                    label,

                                "uid":
                                    uid,

                                "member":
                                    member,

                                "format":
                                    file_format,

                                "variable":
                                    name,

                                "shape":
                                    str(shape),

                                "dtype":
                                    str(dtype),

                                "error":
                                    "",
                            }
                        )

                    h5.visititems(
                        visitor
                    )

            except Exception as e:

                print(
                    "HDF5 ERROR:",
                    repr(e),
                )

                schema_rows.append(
                    {
                        "label":
                            label,

                        "uid":
                            uid,

                        "member":
                            member,

                        "format":
                            file_format,

                        "variable":
                            "",

                        "shape":
                            "",

                        "dtype":
                            "",

                        "error":
                            repr(e),
                    }
                )

        else:

            # --------------------------------
            # whosmat
            # --------------------------------

            try:

                info = whosmat(
                    temp_name
                )

                print(
                    "whosmat result:",
                    repr(info),
                )

                if info is not None:

                    for (
                        name,
                        shape,
                        cls,
                    ) in info:

                        schema_rows.append(
                            {
                                "label":
                                    label,

                                "uid":
                                    uid,

                                "member":
                                    member,

                                "format":
                                    file_format,

                                "variable":
                                    name,

                                "shape":
                                    str(shape),

                                "dtype":
                                    cls,

                                "error":
                                    "",
                            }
                        )

            except Exception as e:

                print(
                    "whosmat ERROR:",
                    repr(e),
                )


            # --------------------------------
            # loadmat
            # --------------------------------

            try:

                mat = loadmat(
                    temp_name,
                    squeeze_me=False,
                    struct_as_record=False,
                )

                keys = [
                    k
                    for k in mat.keys()
                    if not k.startswith(
                        "__"
                    )
                ]

                print(
                    "loadmat keys:",
                    keys,
                )

                for key in keys:

                    value = mat[key]

                    arr = np.asarray(
                        value
                    )

                    print(
                        f"  {key:40s}"
                        f" shape={arr.shape}"
                        f" dtype={arr.dtype}"
                    )

                    schema_rows.append(
                        {
                            "label":
                                label,

                            "uid":
                                uid,

                            "member":
                                member,

                            "format":
                                file_format,

                            "variable":
                                key,

                            "shape":
                                str(
                                    arr.shape
                                ),

                            "dtype":
                                str(
                                    arr.dtype
                                ),

                            "error":
                                "",
                        }
                    )

            except Exception as e:

                print(
                    "loadmat ERROR:",
                    repr(e),
                )

                schema_rows.append(
                    {
                        "label":
                            label,

                        "uid":
                            uid,

                        "member":
                            member,

                        "format":
                            file_format,

                        "variable":
                            "",

                        "shape":
                            "",

                        "dtype":
                            "",

                        "error":
                            repr(e),
                    }
                )


    finally:

        try:
            os.unlink(
                temp_name
            )

        except Exception:
            pass


    return schema_rows


schema_rows = []


# ============================================================
# REPRESENTATIVE PAC/PVC FILE
# ============================================================

pac = nearest[
    (nearest["failure_type"]
     == "PAC_PVC_FALSE_ALERT")
    &
    (nearest["ecg_exists"])
]

if len(pac):

    r = pac.iloc[0]

    schema_rows.extend(
        inspect_mat(
            r["uid"],
            r["ecg_member"],
            "PAC_PVC_FALSE_ALERT",
        )
    )


# ============================================================
# REPRESENTATIVE AF FILE WITH EXACT ECG
# ============================================================

af = nearest[
    (nearest["failure_type"]
     == "AF_MISSED_ALERT")
    &
    (nearest["ecg_exists"])
]

if len(af):

    r = af.iloc[0]

    schema_rows.extend(
        inspect_mat(
            r["uid"],
            r["ecg_member"],
            "AF_MISSED_ALERT_EXACT",
        )
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
    OUT_ARCHIVE
)

print(
    OUT_NEAREST
)

print(
    OUT_SCHEMA
)
