from pathlib import Path
import io
import tarfile
import time

import numpy as np
import pandas as pd


ROOT = Path(
    "experiments/06_external_validation"
)

SRC = ROOT / "pulsewatch_waveform_audit_v2.csv"

OUT = ROOT / "pulsewatch_timestamp_column_probe.csv"
OUT_SUMMARY = ROOT / "pulsewatch_timestamp_column_summary.csv"

PROGRESS_EVERY = 1000

df = pd.read_csv(
    SRC,
    dtype={"uid": str},
    low_memory=False,
)

df["uid"] = (
    df["uid"]
    .astype(str)
    .str.zfill(3)
)

# Chỉ các format có >1 cột.
df = df[
    df["format"].isin(
        [
            "3col",
            "4col",
        ]
    )
].copy()

TOTAL = len(df)


# ============================================================
# TIMESTAMP UNIT
# ============================================================

def infer_scale_to_ms(median_dt):
    """
    Diagnostic inference only.

    Return:
        unit
        scale -> milliseconds
    """

    if not np.isfinite(median_dt):
        return "unknown", np.nan

    # seconds:
    # ~0.020 for 50 Hz
    if 0.005 <= median_dt <= 0.100:
        return "seconds", 1000.0

    # milliseconds:
    # ~20 for 50 Hz
    if 5 <= median_dt <= 100:
        return "milliseconds", 1.0

    # microseconds:
    # ~20000 for 50 Hz
    if 5000 <= median_dt <= 100000:
        return "microseconds", 0.001

    # nanoseconds:
    # ~20,000,000 for 50 Hz
    if 5e6 <= median_dt <= 1e8:
        return "nanoseconds", 1e-6

    return "unknown", np.nan


def audit_column(x):
    rec = {
        "finite_all": False,
        "strictly_increasing": False,

        "monotonic_fraction": np.nan,
        "nonpositive_diffs": np.nan,

        "timestamp_unit": "unknown",

        "median_dt_raw": np.nan,
        "median_dt_ms": np.nan,

        "mean_dt_ms": np.nan,
        "p01_dt_ms": np.nan,
        "p99_dt_ms": np.nan,

        "duration_ms": np.nan,
        "effective_fs_hz": np.nan,

        "max_gap_ms": np.nan,
        "jitter_mad_ms": np.nan,

        "plausible_50hz_30s": False,
    }

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    finite = np.isfinite(x)

    rec["finite_all"] = bool(
        finite.all()
    )

    if not finite.all():
        return rec

    if len(x) < 2:
        return rec

    d = np.diff(x)

    rec["strictly_increasing"] = bool(
        np.all(d > 0)
    )

    rec["monotonic_fraction"] = float(
        np.mean(d > 0)
    )

    rec["nonpositive_diffs"] = int(
        np.sum(d <= 0)
    )

    positive = d[d > 0]

    if len(positive) == 0:
        return rec

    median_raw = float(
        np.median(positive)
    )

    rec["median_dt_raw"] = median_raw

    unit, scale = infer_scale_to_ms(
        median_raw
    )

    rec["timestamp_unit"] = unit

    if not np.isfinite(scale):
        return rec

    d_ms = positive * scale

    med_ms = float(
        np.median(d_ms)
    )

    rec["median_dt_ms"] = med_ms

    rec["mean_dt_ms"] = float(
        np.mean(d_ms)
    )

    rec["p01_dt_ms"] = float(
        np.quantile(
            d_ms,
            0.01,
        )
    )

    rec["p99_dt_ms"] = float(
        np.quantile(
            d_ms,
            0.99,
        )
    )

    rec["max_gap_ms"] = float(
        np.max(d_ms)
    )

    rec["jitter_mad_ms"] = float(
        np.median(
            np.abs(
                d_ms - med_ms
            )
        )
    )

    duration_ms = float(
        (
            x[-1]
            - x[0]
        )
        * scale
    )

    rec["duration_ms"] = duration_ms

    if med_ms > 0:
        rec["effective_fs_hz"] = (
            1000.0 / med_ms
        )

    # Diagnostic only.
    #
    # Known Pulsewatch expectation:
    # approximately 50 Hz × 30 sec.
    #
    # Do NOT use this to delete segments yet.
    rec["plausible_50hz_30s"] = bool(
        rec["monotonic_fraction"] >= 0.999
        and 18 <= med_ms <= 22
        and 28000 <= duration_ms <= 32000
    )

    return rec


# ============================================================
# AUDIT
# ============================================================

print("=" * 90)
print("06A.4.2b PULSEWATCH TIMESTAMP COLUMN DISCOVERY")
print("=" * 90)

print("Segments :", TOTAL)
print(
    "TAR files:",
    df["tar_path"].nunique(),
)

print()
print(
    df["format"]
    .value_counts()
    .to_string()
)

print()

rows = []

processed = 0
t0 = time.time()

for tar_path, g in df.groupby(
    "tar_path",
    sort=True,
):

    with tarfile.open(
        tar_path,
        "r",
    ) as tf:

        for _, r in g.iterrows():

            try:

                fobj = tf.extractfile(
                    r["member"]
                )

                if fobj is None:
                    raise RuntimeError(
                        "extractfile returned None"
                    )

                text = (
                    fobj.read()
                    .decode(
                        "utf-8",
                        errors="strict",
                    )
                )

                arr = np.loadtxt(
                    io.StringIO(text),
                    dtype=np.float64,
                )

                if arr.ndim == 1:
                    arr = arr.reshape(
                        -1,
                        1,
                    )

                n_rows, n_cols = arr.shape

                # -----------------------------------------
                # PPG itself is column 1 (zero-based).
                #
                # Candidate timing columns:
                #
                # 3-col:
                #   col0
                #   col2
                #
                # 4-col:
                #   col0
                #   col2
                #   col3
                #
                # We deliberately audit ALL candidates.
                # -----------------------------------------

                if n_cols == 3:
                    candidates = [
                        0,
                        2,
                    ]

                elif n_cols == 4:
                    candidates = [
                        0,
                        2,
                        3,
                    ]

                else:
                    candidates = []

                for col in candidates:

                    stats = audit_column(
                        arr[:, col]
                    )

                    rec = {
                        "segment":
                            r["segment"],

                        "uid":
                            r["uid"],

                        "class_name":
                            r["class_name"],

                        "label":
                            r["label"],

                        "trial":
                            r["trial"],

                        "format":
                            r["format"],

                        "column":
                            f"col{col}",

                        "column_index":
                            col,

                        "n_rows":
                            n_rows,

                        "error":
                            "",
                    }

                    rec.update(stats)

                    rows.append(rec)

            except Exception as e:

                rows.append(
                    {
                        "segment":
                            r["segment"],

                        "uid":
                            r["uid"],

                        "class_name":
                            r["class_name"],

                        "label":
                            r["label"],

                        "trial":
                            r["trial"],

                        "format":
                            r["format"],

                        "column":
                            "ERROR",

                        "column_index":
                            -1,

                        "error":
                            repr(e),
                    }
                )

            processed += 1

            # ====================================================
            # PROGRESS PRINT
            # ====================================================

            if (
                processed % PROGRESS_EVERY == 0
                or processed == TOTAL
            ):

                elapsed = (
                    time.time() - t0
                )

                rate = (
                    processed / elapsed
                    if elapsed > 0
                    else 0
                )

                remaining = (
                    TOTAL - processed
                )

                eta_sec = (
                    remaining / rate
                    if rate > 0
                    else np.nan
                )

                pct = (
                    processed
                    / TOTAL
                    * 100.0
                )

                print(
                    f"Processed "
                    f"{processed:,}/{TOTAL:,} "
                    f"({pct:6.2f}%)"
                    f" | "
                    f"{rate:7.1f} seg/s"
                    f" | "
                    f"ETA "
                    f"{eta_sec / 60:6.1f} min",
                    flush=True,
                )


# ============================================================
# SAVE
# ============================================================

out = pd.DataFrame(rows)

out.to_csv(
    OUT,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

valid = out[
    out["column"] != "ERROR"
].copy()


summary = (
    valid.groupby(
        [
            "format",
            "column",
        ],
        as_index=False,
    )
    .agg(
        segments=(
            "segment",
            "size",
        ),

        finite_all=(
            "finite_all",
            "sum",
        ),

        strictly_increasing=(
            "strictly_increasing",
            "sum",
        ),

        median_monotonic_fraction=(
            "monotonic_fraction",
            "median",
        ),

        known_unit=(
            "timestamp_unit",
            lambda x:
                int(
                    (
                        x != "unknown"
                    ).sum()
                ),
        ),

        plausible_50hz_30s=(
            "plausible_50hz_30s",
            "sum",
        ),

        median_dt_ms=(
            "median_dt_ms",
            "median",
        ),

        median_fs_hz=(
            "effective_fs_hz",
            "median",
        ),

        median_duration_ms=(
            "duration_ms",
            "median",
        ),

        median_max_gap_ms=(
            "max_gap_ms",
            "median",
        ),

        median_jitter_mad_ms=(
            "jitter_mad_ms",
            "median",
        ),
    )
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)


print()
print("=" * 90)
print("TIMESTAMP COLUMN DISCOVERY RESULT")
print("=" * 90)

print()
print(summary.to_string(index=False))


print()
print("===== UNIT BY FORMAT / COLUMN =====")

print(
    pd.crosstab(
        [
            valid["format"],
            valid["column"],
        ],
        valid["timestamp_unit"],
    )
)


print()
print("===== PLAUSIBLE 50 Hz / 30 s =====")

tmp = (
    valid.groupby(
        [
            "format",
            "column",
        ]
    )["plausible_50hz_30s"]
    .agg(
        [
            "sum",
            "count",
        ]
    )
)

tmp["percent"] = (
    tmp["sum"]
    / tmp["count"]
    * 100
)

print(
    tmp.to_string()
)


print()
print("===== BEST STRICT-MONOTONIC COUNTS =====")

print(
    valid.groupby(
        [
            "format",
            "column",
        ]
    )["strictly_increasing"]
    .agg(
        [
            "sum",
            "count",
        ]
    )
    .to_string()
)


print()
print("Saved:")
print(" ", OUT)
print(" ", OUT_SUMMARY)
