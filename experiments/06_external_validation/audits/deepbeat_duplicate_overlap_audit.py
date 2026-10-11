from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

NPZ = Path("data/raw/test.npz")
OUT = Path("experiments/06_external_validation")

FS = 32.0
WINDOW_S = 25.0
N = int(FS * WINDOW_S)

print("=" * 100)
print("DEEPBEAT DUPLICATE / OVERLAP AUDIT")
print("=" * 100)

with np.load(NPZ, allow_pickle=True) as z:
    signal = z["signal"][:, :, 0]
    rhythm = z["rhythm"]
    qa = z["qa_label"]
    params = z["parameters"]

labels = np.argmax(rhythm, axis=1)
quality = np.argmax(qa, axis=1)

timestamps = pd.to_datetime(params[:, 0])
sources = np.array([
    str(x).strip()
    for x in params[:, 1]
])
subjects = params[:, 2].astype(int)

# ------------------------------------------------------------
# Signal hashes
# ------------------------------------------------------------

print("Hashing signals...")

hashes = []

for x in signal:
    h = hashlib.sha1(
        np.ascontiguousarray(x).tobytes()
    ).hexdigest()
    hashes.append(h)

manifest = pd.DataFrame({
    "original_index": np.arange(len(signal)),
    "timestamp": timestamps,
    "source": sources,
    "subject": subjects,
    "label": labels,
    "quality": quality,
    "signal_hash": hashes,
})

print("\nSegments:", len(manifest))
print("Unique signal hashes:",
      manifest.signal_hash.nunique())

# ============================================================
# 1. Duplicate timestamp keys
# ============================================================

key_cols = [
    "subject",
    "source",
    "timestamp",
]

groups = (
    manifest
    .groupby(key_cols)
    .agg(
        rows=("original_index", "size"),
        hashes=("signal_hash", "nunique"),
        labels=("label", "nunique"),
        qualities=("quality", "nunique"),
    )
    .reset_index()
)

dup = groups[
    groups["rows"] > 1
].copy()

print("\n" + "=" * 100)
print("[1] DUPLICATE TIMESTAMP KEYS")
print("=" * 100)

print("Duplicate groups :", len(dup))
print("Rows in groups   :", int(dup.rows.sum()))

print("\nGROUP SIZE")
print(
    dup["rows"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nIDENTICAL SIGNAL WITHIN DUP KEY:")
print(
    (dup["hashes"] == 1)
    .value_counts()
    .to_string()
)

print("\nDifferent labels at same key:",
      int((dup["labels"] > 1).sum()))

print("Different quality at same key:",
      int((dup["qualities"] > 1).sum()))

# ============================================================
# 2. Exact signal repetition globally
# ============================================================

hc = (
    manifest["signal_hash"]
    .value_counts()
)

print("\n" + "=" * 100)
print("[2] EXACT SIGNAL REPETITION")
print("=" * 100)

print("Unique waveforms :", len(hc))
print(
    "Hashes repeated :",
    int((hc > 1).sum())
)
print(
    "Rows belonging to repeated waveform:",
    int(hc[hc > 1].sum())
)

print("\nMost repeated:")
print(
    hc.head(20).to_string()
)

# ============================================================
# 3. Drop only exact duplicate key+waveform
# ============================================================

canonical = (
    manifest
    .sort_values("original_index")
    .drop_duplicates(
        subset=[
            "subject",
            "source",
            "timestamp",
            "signal_hash",
        ],
        keep="first"
    )
    .copy()
)

print("\n" + "=" * 100)
print("[3] AFTER SAFE EXACT DEDUP")
print("=" * 100)

print("Before :", len(manifest))
print("After  :", len(canonical))
print("Removed:", len(manifest) - len(canonical))

# Temporal ordering
canonical = canonical.sort_values(
    ["subject", "source", "timestamp"]
).reset_index(drop=True)

canonical["dt_s"] = (
    canonical
    .groupby(["subject", "source"])
    ["timestamp"]
    .diff()
    .dt.total_seconds()
)

print("\nDT AFTER DEDUP")
print(
    canonical["dt_s"]
    .describe(
        percentiles=[
            .01, .10, .25, .50,
            .75, .90, .99
        ]
    )
)

print("\nCOMMON DT")
print(
    canonical["dt_s"]
    .value_counts()
    .head(30)
    .to_string()
)

# ============================================================
# 4. Check overlapping windows consistency
# ============================================================

print("\n" + "=" * 100)
print("[4] OVERLAPPING WINDOW CONSISTENCY")
print("=" * 100)

overlap_rows = []

for (uid, source), g in canonical.groupby(
    ["subject", "source"]
):
    g = g.sort_values("timestamp")

    rows = list(g.itertuples())

    for prev, cur in zip(rows[:-1], rows[1:]):
        dt = (
            cur.timestamp
            - prev.timestamp
        ).total_seconds()

        if not (0 < dt < WINDOW_S):
            continue

        shift_float = dt * FS
        shift = int(round(shift_float))

        # only compare when timestamp aligns with sample grid
        if abs(shift - shift_float) > 0.05:
            continue

        if shift <= 0 or shift >= N:
            continue

        a = signal[
            int(prev.original_index)
        ][shift:]

        b = signal[
            int(cur.original_index)
        ][:N-shift]

        if len(a) < 10:
            continue

        sa = np.std(a)
        sb = np.std(b)

        if sa == 0 or sb == 0:
            corr = np.nan
        else:
            corr = np.corrcoef(a, b)[0, 1]

        # z-normalized RMSE
        if sa > 0 and sb > 0:
            az = (a - np.mean(a)) / sa
            bz = (b - np.mean(b)) / sb
            zrmse = float(
                np.sqrt(
                    np.mean(
                        (az - bz) ** 2
                    )
                )
            )
        else:
            zrmse = np.nan

        overlap_rows.append({
            "subject": uid,
            "source": source,
            "prev_index":
                int(prev.original_index),
            "cur_index":
                int(cur.original_index),
            "dt_s": dt,
            "overlap_s":
                WINDOW_S - dt,
            "overlap_samples":
                len(a),
            "corr": corr,
            "zrmse": zrmse,
        })

overlap = pd.DataFrame(overlap_rows)

print("Overlapping pairs:", len(overlap))

if len(overlap):

    print("\nCORRELATION")
    print(
        overlap["corr"]
        .describe(
            percentiles=[
                .01,.10,.50,.90,.99
            ]
        )
    )

    print(
        "\ncorr >= 0.99:",
        int(
            (overlap["corr"] >= .99)
            .sum()
        ),
        "/",
        len(overlap)
    )

    print("\nBY DT")
    print(
        overlap.groupby("dt_s")
        .agg(
            pairs=("corr", "size"),
            median_corr=("corr", "median"),
            min_corr=("corr", "min"),
            median_zrmse=("zrmse", "median"),
        )
        .head(30)
        .to_string()
    )

# ============================================================
# 5. Subject/class after safe dedup
# ============================================================

print("\n" + "=" * 100)
print("[5] SUBJECT / CLASS AFTER DEDUP")
print("=" * 100)

print(
    canonical
    .groupby(
        ["subject", "label"]
    )
    .size()
    .unstack(fill_value=0)
    .to_string()
)

# ============================================================
# Save
# ============================================================

manifest.to_csv(
    OUT /
    "deepbeat_test_waveform_manifest.csv",
    index=False
)

canonical.to_csv(
    OUT /
    "deepbeat_test_exact_dedup_manifest.csv",
    index=False
)

if len(overlap):
    overlap.to_csv(
        OUT /
        "deepbeat_test_overlap_consistency.csv",
        index=False
    )

np.save(
    OUT /
    "deepbeat_test_exact_dedup_indices.npy",
    canonical["original_index"]
    .to_numpy(dtype=int)
)

print("\n" + "=" * 100)
print("SAVED")
print("=" * 100)

print(
    OUT /
    "deepbeat_test_waveform_manifest.csv"
)
print(
    OUT /
    "deepbeat_test_exact_dedup_manifest.csv"
)
print(
    OUT /
    "deepbeat_test_overlap_consistency.csv"
)
print(
    OUT /
    "deepbeat_test_exact_dedup_indices.npy"
)
