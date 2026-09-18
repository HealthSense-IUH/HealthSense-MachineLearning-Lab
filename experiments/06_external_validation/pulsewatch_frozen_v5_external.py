import sys
sys.path.insert(0, "src")

from pathlib import Path
import argparse
import io
import tarfile
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)

from healthsense_ml.signal_processing import (
    extract_nn_series,
    bandpass_filter,
)

from healthsense_ml.hrv_features import (
    compute_hrv_features,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("experiments/06_external_validation")

MANIFEST = ROOT / "pulsewatch_manifest_3class_strict.csv"
AUDIT = ROOT / "pulsewatch_waveform_audit_v2.csv"

MODEL_PATH = Path(
    "models/mimic/"
    "healthsense_af_v5_rf_ac_frozen.pkl"
)

FS = 50.0
N_SAMPLES = 1500
TARGET_DT_MS = 20.0

MIN_BEATS = 10
PROGRESS_EVERY = 500

FEATURES = [
    "HR_mean",
    "Mean_NN",
    "SDNN",
    "RMSSD",
    "NN50",
    "pNN50",
    "CV",
    "HF",
    "Total_Power",
    "HF_norm",
    "SD1",
    "SD2",
    "SampEn",
    "PPG_AC",
]


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Smoke-test only first N segments",
)

args = parser.parse_args()

suffix = (
    f"_smoke{args.limit}"
    if args.limit
    else ""
)

OUT_FEATURES = (
    ROOT /
    f"pulsewatch_frozen_v5_features{suffix}.csv"
)

OUT_PRED = (
    ROOT /
    f"pulsewatch_frozen_v5_predictions{suffix}.csv"
)

OUT_SUMMARY = (
    ROOT /
    f"pulsewatch_frozen_v5_summary{suffix}.csv"
)

OUT_COVERAGE = (
    ROOT /
    f"pulsewatch_frozen_v5_coverage{suffix}.csv"
)


# ============================================================
# PPG AC — SAME DEFINITION AS FROZEN V5
# ============================================================

def autocorrelation_feature(x, fs=FS):

    x = np.asarray(
        x,
        dtype=float,
    )

    x = bandpass_filter(
        x,
        fs=fs,
    )

    x = x - np.mean(x)

    n = len(x)

    if n < 10:
        return np.nan

    nfft = 1 << (
        (2 * n - 1).bit_length()
    )

    spectrum = np.fft.rfft(
        x,
        n=nfft,
    )

    ac = np.fft.irfft(
        spectrum
        * np.conj(spectrum),
        n=nfft,
    )[:n]

    ac = ac / n

    if (
        not np.isfinite(ac[0])
        or abs(ac[0]) < 1e-12
    ):
        return np.nan

    ac = ac / ac[0]

    return float(
        np.mean(
            np.abs(
                ac[1:]
            )
        )
    )


# ============================================================
# RAW PARSER
# ============================================================

def parse_ppg(arr):

    if arr.ndim == 1:
        arr = arr.reshape(
            -1,
            1,
        )

    n_rows, n_cols = arr.shape

    if n_rows != N_SAMPLES:
        raise ValueError(
            f"Expected 1500 rows, got {n_rows}"
        )

    if n_cols == 1:

        return (
            arr[:, 0].astype(float),
            None,
            "1col_legacy",
        )

    if n_cols == 3:

        # Pulsewatch:
        # col1 = raw PPG
        # col0 = usable timestamp ver2
        return (
            arr[:, 1].astype(float),
            arr[:, 0].astype(float),
            "3col",
        )

    if n_cols == 4:

        # Pulsewatch:
        # col1 = raw PPG
        # col0 = timestamp ver2
        return (
            arr[:, 1].astype(float),
            arr[:, 0].astype(float),
            "4col",
        )

    raise ValueError(
        f"Unexpected column count: {n_cols}"
    )


# ============================================================
# TIMEBASE NORMALIZATION
# ============================================================

def nominal_timebase(ppg):

    return (
        np.asarray(
            ppg,
            dtype=float,
        ),
        {
            "timing_method":
                "nominal_50hz",

            "median_dt_ms":
                20.0,

            "duration_ms":
                29980.0,

            "nonpositive_diffs":
                0,

            "timestamp_valid":
                False,

            "timestamp_fallback":
                False,
        },
    )


def normalize_timebase(ppg, timestamp):

    ppg = np.asarray(
        ppg,
        dtype=float,
    )

    timestamp = np.asarray(
        timestamp,
        dtype=float,
    )

    if (
        len(ppg) != N_SAMPLES
        or len(timestamp) != N_SAMPLES
        or not np.isfinite(ppg).all()
        or not np.isfinite(timestamp).all()
    ):
        raise ValueError(
            "Invalid PPG/timestamp"
        )

    d = np.diff(timestamp)

    positive = d[d > 0]

    if len(positive) == 0:

        y, info = nominal_timebase(
            ppg
        )

        info[
            "timing_method"
        ] = "nominal_50hz_fallback"

        info[
            "timestamp_fallback"
        ] = True

        return y, info

    median_dt = float(
        np.median(
            positive
        )
    )

    duration = float(
        timestamp[-1]
        - timestamp[0]
    )

    nonpositive = int(
        np.sum(
            d <= 0
        )
    )

    # --------------------------------------------
    # Prespecified structural timing criterion.
    # NO LABELS are used.
    # --------------------------------------------

    timestamp_valid = bool(
        18.0 <= median_dt <= 22.0
        and
        28000.0 <= duration <= 32000.0
    )

    if not timestamp_valid:

        y, info = nominal_timebase(
            ppg
        )

        info.update(
            {
                "timing_method":
                    "nominal_50hz_fallback",

                "median_dt_ms":
                    median_dt,

                "duration_ms":
                    duration,

                "nonpositive_diffs":
                    nonpositive,

                "timestamp_valid":
                    False,

                "timestamp_fallback":
                    True,
            }
        )

        return y, info

    # --------------------------------------------
    # Repair non-monotonic timestamps.
    #
    # Keep samples only when time is strictly
    # greater than the last accepted timestamp.
    # --------------------------------------------

    t = (
        timestamp
        - timestamp[0]
    )

    keep = np.zeros(
        len(t),
        dtype=bool,
    )

    keep[0] = True

    last_t = t[0]

    for i in range(
        1,
        len(t),
    ):

        if t[i] > last_t:

            keep[i] = True
            last_t = t[i]

    t_clean = t[keep]
    ppg_clean = ppg[keep]

    if len(t_clean) < 2:

        y, info = nominal_timebase(
            ppg
        )

        info.update(
            {
                "timing_method":
                    "nominal_50hz_fallback",

                "median_dt_ms":
                    median_dt,

                "duration_ms":
                    duration,

                "nonpositive_diffs":
                    nonpositive,

                "timestamp_valid":
                    False,

                "timestamp_fallback":
                    True,
            }
        )

        return y, info

    target = (
        np.arange(
            N_SAMPLES,
            dtype=float,
        )
        * TARGET_DT_MS
    )

    # np.interp:
    # - linear interpolation internally
    # - if target exceeds observed final time,
    #   hold last value (safe padding).
    uniform_ppg = np.interp(
        target,
        t_clean,
        ppg_clean,
    )

    if (
        len(uniform_ppg)
        != N_SAMPLES
        or
        not np.isfinite(
            uniform_ppg
        ).all()
    ):
        raise ValueError(
            "Resampling failed"
        )

    return (
        uniform_ppg,
        {
            "timing_method":
                "timestamp_col0_interp",

            "median_dt_ms":
                median_dt,

            "duration_ms":
                duration,

            "nonpositive_diffs":
                nonpositive,

            "timestamp_valid":
                True,

            "timestamp_fallback":
                False,

            "timestamp_samples_kept":
                int(
                    keep.sum()
                ),
        },
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(ppg):

    nn_ms, nn_times = (
        extract_nn_series(
            ppg,
            fs=FS,
        )
    )

    n_beats = (
        len(nn_ms) + 1
        if len(nn_ms)
        else 0
    )

    if len(nn_ms) < MIN_BEATS:

        return (
            None,
            n_beats,
            len(nn_ms),
        )

    feats = compute_hrv_features(
        nn_ms,
        nn_times,
    )

    feats["PPG_AC"] = (
        autocorrelation_feature(
            ppg,
            fs=FS,
        )
    )

    for f in FEATURES:

        if (
            f not in feats
            or
            not np.isfinite(
                feats[f]
            )
        ):

            return (
                None,
                n_beats,
                len(nn_ms),
            )

    return (
        {
            f: float(
                feats[f]
            )
            for f in FEATURES
        },
        n_beats,
        len(nn_ms),
    )


# ============================================================
# LOAD METADATA
# ============================================================

manifest = pd.read_csv(
    MANIFEST,
    dtype={"uid": str},
)

audit = pd.read_csv(
    AUDIT,
    dtype={"uid": str},
    low_memory=False,
)

manifest["uid"] = (
    manifest["uid"]
    .astype(str)
    .str.zfill(3)
)

audit["uid"] = (
    audit["uid"]
    .astype(str)
    .str.zfill(3)
)

meta = manifest.merge(
    audit[
        [
            "segment",
            "format",
        ]
    ],
    on="segment",
    how="left",
    validate="one_to_one",
)

if args.limit:

    meta = meta.iloc[
        :args.limit
    ].copy()

TOTAL = len(meta)


print("=" * 90)
print("06A.4.3 PULSEWATCH FROZEN V5 EXTERNAL VALIDATION")
print("=" * 90)

print(
    "Segments:",
    f"{TOTAL:,}",
)

print(
    "Subjects:",
    meta["uid"].nunique(),
)

print()
print(
    meta["class_name"]
    .value_counts()
    .to_string()
)

print()
print(
    meta["format"]
    .value_counts()
    .to_string()
)

print()


# ============================================================
# PROCESS RAW DATA
# ============================================================

rows = []

processed = 0
feature_ok = 0
feature_fail = 0
fallback_count = 0

t0 = time.time()

for tar_path, g in meta.groupby(
    "tar_path",
    sort=True,
):

    with tarfile.open(
        tar_path,
        "r",
    ) as tf:

        for _, r in g.iterrows():

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

                "feature_ok":
                    False,

                "error":
                    "",
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

                ppg, ts, fmt = (
                    parse_ppg(
                        arr
                    )
                )

                if ts is None:

                    ppg_uniform, timing = (
                        nominal_timebase(
                            ppg
                        )
                    )

                else:

                    ppg_uniform, timing = (
                        normalize_timebase(
                            ppg,
                            ts,
                        )
                    )

                rec.update(
                    timing
                )

                if rec.get(
                    "timestamp_fallback",
                    False,
                ):
                    fallback_count += 1

                (
                    feats,
                    n_beats,
                    n_nn,
                ) = extract_features(
                    ppg_uniform
                )

                rec["n_beats"] = (
                    n_beats
                )

                rec["n_nn"] = (
                    n_nn
                )

                if feats is None:

                    feature_fail += 1

                    rec[
                        "error"
                    ] = (
                        "feature_extraction_failed"
                    )

                else:

                    rec.update(
                        feats
                    )

                    rec[
                        "feature_ok"
                    ] = True

                    feature_ok += 1

            except Exception as e:

                feature_fail += 1

                rec["error"] = repr(
                    e
                )

            rows.append(rec)

            processed += 1

            # ====================================================
            # LIVE PROGRESS
            # ====================================================

            if (
                processed
                % PROGRESS_EVERY
                == 0
                or
                processed == TOTAL
            ):

                elapsed = (
                    time.time()
                    - t0
                )

                rate = (
                    processed
                    / elapsed
                    if elapsed > 0
                    else 0
                )

                remaining = (
                    TOTAL
                    - processed
                )

                eta = (
                    remaining
                    / rate
                    if rate > 0
                    else np.nan
                )

                pct = (
                    processed
                    / TOTAL
                    * 100
                )

                coverage = (
                    feature_ok
                    / processed
                    * 100
                )

                print(
                    f"Processed "
                    f"{processed:,}/{TOTAL:,} "
                    f"({pct:6.2f}%)"
                    f" | "
                    f"{rate:6.1f} seg/s"
                    f" | "
                    f"ETA {eta/60:6.1f} min"
                    f" | "
                    f"feature OK "
                    f"{feature_ok:,}"
                    f" ({coverage:5.1f}%)"
                    f" | "
                    f"FAIL {feature_fail:,}"
                    f" | "
                    f"timing fallback "
                    f"{fallback_count:,}",
                    flush=True,
                )


# ============================================================
# SAVE FEATURES
# ============================================================

df = pd.DataFrame(
    rows
)

df.to_csv(
    OUT_FEATURES,
    index=False,
)

valid = df[
    df["feature_ok"]
].copy()


# ============================================================
# FROZEN MODEL
# ============================================================

print()
print("=" * 90)
print("FROZEN MODEL")
print("=" * 90)

print(
    "Loading:",
    MODEL_PATH,
)

model = joblib.load(
    MODEL_PATH
)

X = valid[
    FEATURES
]

valid[
    "af_probability"
] = model.predict_proba(
    X
)[:, 1]

valid[
    "y_true"
] = (
    valid["class_name"]
    == "AF"
).astype(int)

valid[
    "pred_0_5"
] = (
    valid["af_probability"]
    >= 0.50
).astype(int)

valid[
    "pred_0_9"
] = (
    valid["af_probability"]
    >= 0.90
).astype(int)

valid.to_csv(
    OUT_PRED,
    index=False,
)


# ============================================================
# METRIC HELPERS
# ============================================================

def binary_metrics(
    x,
    threshold,
):

    y = x[
        "y_true"
    ].to_numpy()

    p = x[
        "af_probability"
    ].to_numpy()

    pred = (
        p >= threshold
    ).astype(int)

    out = {
        "n":
            len(x),

        "af_n":
            int(
                np.sum(y == 1)
            ),

        "nonaf_n":
            int(
                np.sum(y == 0)
            ),

        "threshold":
            threshold,

        "roc_auc":
            np.nan,

        "pr_auc":
            np.nan,

        "sensitivity":
            np.nan,

        "specificity":
            np.nan,

        "balanced_accuracy":
            np.nan,

        "f1":
            np.nan,
    }

    if len(
        np.unique(y)
    ) == 2:

        out[
            "roc_auc"
        ] = roc_auc_score(
            y,
            p,
        )

        out[
            "pr_auc"
        ] = average_precision_score(
            y,
            p,
        )

        tn, fp, fn, tp = (
            confusion_matrix(
                y,
                pred,
                labels=[
                    0,
                    1,
                ],
            )
            .ravel()
        )

        out[
            "sensitivity"
        ] = (
            tp
            / (tp + fn)
            if tp + fn
            else np.nan
        )

        out[
            "specificity"
        ] = (
            tn
            / (tn + fp)
            if tn + fp
            else np.nan
        )

        out[
            "balanced_accuracy"
        ] = balanced_accuracy_score(
            y,
            pred,
        )

        out[
            "f1"
        ] = f1_score(
            y,
            pred,
            zero_division=0,
        )

    return out


summary_rows = []

for threshold in [
    0.50,
    0.90,
]:

    # Overall
    m = binary_metrics(
        valid,
        threshold,
    )

    m.update(
        {
            "group_type":
                "overall",

            "group":
                "ALL",
        }
    )

    summary_rows.append(m)

    # Per format
    for name, g in valid.groupby(
        "format"
    ):

        m = binary_metrics(
            g,
            threshold,
        )

        m.update(
            {
                "group_type":
                    "format",

                "group":
                    name,
            }
        )

        summary_rows.append(m)

    # Per trial
    for name, g in valid.groupby(
        "trial"
    ):

        m = binary_metrics(
            g,
            threshold,
        )

        m.update(
            {
                "group_type":
                    "trial",

                "group":
                    name,
            }
        )

        summary_rows.append(m)


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)


# ============================================================
# COVERAGE / HARD NEGATIVE REPORT
# ============================================================

coverage = (
    df.groupby(
        [
            "class_name",
            "format",
            "trial",
        ],
        dropna=False,
    )
    .agg(
        segments=(
            "segment",
            "size",
        ),

        feature_ok=(
            "feature_ok",
            "sum",
        ),

        median_beats=(
            "n_beats",
            "median",
        ),

        timing_fallback=(
            "timestamp_fallback",
            "sum",
        ),
    )
    .reset_index()
)

coverage[
    "coverage"
] = (
    coverage["feature_ok"]
    / coverage["segments"]
)

coverage.to_csv(
    OUT_COVERAGE,
    index=False,
)


# ============================================================
# PRINT FINAL RESULT
# ============================================================

print()
print("=" * 90)
print("FEATURE COVERAGE")
print("=" * 90)

print(
    f"Input       : {len(df):,}"
)

print(
    f"Feature OK  : "
    f"{int(df.feature_ok.sum()):,}"
    f" "
    f"({df.feature_ok.mean()*100:.2f}%)"
)

print(
    f"Feature FAIL: "
    f"{int((~df.feature_ok).sum()):,}"
)

fallback_total = int(
    df.get(
        "timestamp_fallback",
        pd.Series(
            False,
            index=df.index,
        ),
    )
    .fillna(False)
    .sum()
)

print(
    f"Timing fallback: "
    f"{fallback_total:,}"
)


print()
print("=" * 90)
print("OVERALL FROZEN V5")
print("=" * 90)

print(
    summary[
        summary[
            "group_type"
        ]
        == "overall"
    ].to_string(
        index=False
    )
)


print()
print("=" * 90)
print("PER FORMAT")
print("=" * 90)

print(
    summary[
        summary[
            "group_type"
        ]
        == "format"
    ].to_string(
        index=False
    )
)


print()
print("=" * 90)
print("PER TRIAL")
print("=" * 90)

print(
    summary[
        summary[
            "group_type"
        ]
        == "trial"
    ].to_string(
        index=False
    )
)


print()
print("=" * 90)
print("NON-AF FALSE POSITIVE RATE @ 0.90")
print("=" * 90)

for cls in [
    "NSR",
    "PAC_PVC",
]:

    g = valid[
        valid[
            "class_name"
        ]
        == cls
    ]

    fp_rate = (
        (
            g[
                "af_probability"
            ]
            >= 0.90
        )
        .mean()
        if len(g)
        else np.nan
    )

    print(
        f"{cls:8s}: "
        f"{fp_rate:.4f} "
        f"({int((g.af_probability >= 0.90).sum()):,}"
        f"/{len(g):,})"
    )


print()
print("Saved:")
print(" ", OUT_FEATURES)
print(" ", OUT_PRED)
print(" ", OUT_SUMMARY)
print(" ", OUT_COVERAGE)
