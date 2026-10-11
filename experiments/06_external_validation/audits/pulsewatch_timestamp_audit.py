from pathlib import Path
import io
import tarfile

import numpy as np
import pandas as pd


ROOT = Path("experiments/06_external_validation")

SRC = ROOT / "pulsewatch_waveform_audit_v2.csv"
OUT = ROOT / "pulsewatch_timestamp_audit.csv"

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

# Only formats that actually contain timestamps.
df = df[
    df["format"].isin(
        ["3col", "4col"]
    )
].copy()


def infer_scale_to_ms(median_dt):
    """
    Return:
        unit_name,
        multiply factor -> milliseconds

    Diagnostic only. Nothing is excluded.
    """

    if not np.isfinite(median_dt):
        return "unknown", np.nan

    # seconds, ~0.020
    if 0.005 <= median_dt <= 0.100:
        return "seconds", 1000.0

    # milliseconds, ~20
    if 5 <= median_dt <= 100:
        return "milliseconds", 1.0

    # microseconds, ~20000
    if 5000 <= median_dt <= 100000:
        return "microseconds", 0.001

    # nanoseconds, ~20,000,000
    if 5e6 <= median_dt <= 1e8:
        return "nanoseconds", 1e-6

    return "unknown", np.nan


rows = []

print("=" * 90)
print("06A.4.2 PULSEWATCH TIMESTAMP / SAMPLING AUDIT")
print("=" * 90)

print("Segments with timestamp:", len(df))
print("TAR files:", df["tar_path"].nunique())

done = 0

for tar_path, g in df.groupby(
    "tar_path",
    sort=True,
):

    print(
        f"[{done:6d}/{len(df)}] "
        f"{Path(tar_path).name} "
        f"n={len(g)}"
    )

    with tarfile.open(
        tar_path,
        "r",
    ) as tf:

        for _, r in g.iterrows():

            rec = {
                "segment": r["segment"],
                "uid": r["uid"],
                "class_name": r["class_name"],
                "label": r["label"],
                "trial": r["trial"],
                "format": r["format"],

                "read_ok": False,
                "n_samples": np.nan,

                "timestamp_unit":
                    "unknown",

                "timestamp_finite":
                    False,

                "strictly_increasing":
                    False,

                "nonpositive_diffs":
                    np.nan,

                "median_dt_raw":
                    np.nan,

                "median_dt_ms":
                    np.nan,

                "mean_dt_ms":
                    np.nan,

                "p01_dt_ms":
                    np.nan,

                "p99_dt_ms":
                    np.nan,

                "max_gap_ms":
                    np.nan,

                "jitter_mad_ms":
                    np.nan,

                "duration_ms":
                    np.nan,

                "effective_fs_hz":
                    np.nan,

                "error": "",
            }

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
                    arr = arr.reshape(-1, 1)

                n_rows, n_cols = arr.shape

                rec["n_samples"] = n_rows

                # Official Pulsewatch convention used
                # in our parser audit.
                if (
                    r["format"] == "3col"
                    and n_cols == 3
                ):
                    timestamp = arr[:, 2]

                elif (
                    r["format"] == "4col"
                    and n_cols == 4
                ):
                    timestamp = arr[:, 3]

                else:
                    raise RuntimeError(
                        f"Format mismatch: "
                        f"{r['format']} / "
                        f"{n_cols} cols"
                    )

                rec["read_ok"] = True

                finite = np.isfinite(
                    timestamp
                )

                rec["timestamp_finite"] = (
                    bool(finite.all())
                )

                if not finite.all():
                    rows.append(rec)
                    continue

                d = np.diff(timestamp)

                rec["nonpositive_diffs"] = int(
                    np.sum(d <= 0)
                )

                rec["strictly_increasing"] = bool(
                    np.all(d > 0)
                )

                positive = d[d > 0]

                if len(positive) == 0:
                    rows.append(rec)
                    continue

                median_dt = float(
                    np.median(positive)
                )

                rec["median_dt_raw"] = (
                    median_dt
                )

                unit, scale = (
                    infer_scale_to_ms(
                        median_dt
                    )
                )

                rec["timestamp_unit"] = unit

                if not np.isfinite(scale):
                    rows.append(rec)
                    continue

                d_ms = positive * scale

                rec["median_dt_ms"] = float(
                    np.median(d_ms)
                )

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

                med = np.median(d_ms)

                rec["jitter_mad_ms"] = float(
                    np.median(
                        np.abs(
                            d_ms - med
                        )
                    )
                )

                rec["duration_ms"] = float(
                    (
                        timestamp[-1]
                        - timestamp[0]
                    )
                    * scale
                )

                if med > 0:
                    rec["effective_fs_hz"] = (
                        1000.0 / med
                    )

            except Exception as e:
                rec["error"] = repr(e)

            rows.append(rec)

    done += len(g)


out = pd.DataFrame(rows)

out.to_csv(
    OUT,
    index=False,
)


print()
print("=" * 90)
print("TIMESTAMP AUDIT RESULT")
print("=" * 90)

print("Audited :", len(out))

print(
    "Read OK :",
    int(out["read_ok"].sum())
)

print(
    "Finite  :",
    int(out["timestamp_finite"].sum())
)

print(
    "Strictly increasing:",
    int(out["strictly_increasing"].sum())
)

print(
    "Non-monotonic:",
    int(
        (
            ~out["strictly_increasing"]
        ).sum()
    )
)


print()
print("===== FORMAT =====")

print(
    out["format"]
    .value_counts()
    .to_string()
)


print()
print("===== INFERRED TIMESTAMP UNIT =====")

print(
    pd.crosstab(
        out["format"],
        out["timestamp_unit"],
    )
)


print()
print("===== SAMPLING STATISTICS BY FORMAT =====")

cols = [
    "median_dt_ms",
    "effective_fs_hz",
    "duration_ms",
    "jitter_mad_ms",
    "max_gap_ms",
]

for fmt, g in out.groupby("format"):

    print()
    print("###", fmt)

    print(
        g[cols]
        .describe(
            percentiles=[
                .001,
                .01,
                .05,
                .50,
                .95,
                .99,
                .999,
            ]
        )
        .to_string()
    )


print()
print("===== NON-POSITIVE TIMESTAMP DIFFS =====")

bad_order = out[
    out["nonpositive_diffs"]
    .fillna(0)
    > 0
]

print("Segments:", len(bad_order))

if len(bad_order):

    print(
        bad_order[
            [
                "segment",
                "uid",
                "class_name",
                "format",
                "nonpositive_diffs",
                "median_dt_ms",
                "max_gap_ms",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


print()
print("===== LARGEST TIMESTAMP GAPS =====")

print(
    out.sort_values(
        "max_gap_ms",
        ascending=False,
    )[
        [
            "segment",
            "uid",
            "class_name",
            "format",
            "median_dt_ms",
            "duration_ms",
            "max_gap_ms",
            "jitter_mad_ms",
        ]
    ]
    .head(30)
    .to_string(
        index=False
    )
)


print()
print("Saved:", OUT)
