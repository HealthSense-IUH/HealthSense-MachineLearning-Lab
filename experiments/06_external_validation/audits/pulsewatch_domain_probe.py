from pathlib import Path
import io
import tarfile

import numpy as np
import pandas as pd

from scipy import signal, stats

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, balanced_accuracy_score, f1_score


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("experiments/06_external_validation")

MANIFEST = ROOT / "pulsewatch_manifest_3class_strict.csv"
AUDIT = ROOT / "pulsewatch_waveform_audit_v2.csv"

OUT_SEG = ROOT / "pulsewatch_domain_probe_segment_features.csv"
OUT_SUBJ = ROOT / "pulsewatch_domain_probe_subject_features.csv"
OUT_SUM = ROOT / "pulsewatch_domain_probe_summary.csv"

FS = 50.0
LOW = 0.5
HIGH = 8.0
ORDER = 4

RANDOM_STATE = 42


# ============================================================
# PREPROCESSING
# ============================================================

def bandpass(x, fs=FS):
    sos = signal.butter(
        ORDER,
        [LOW, HIGH],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return signal.sosfiltfilt(sos, x)


def zscore_segment(x):
    std = np.std(x)

    if std <= 0 or not np.isfinite(std):
        return None

    return (x - np.mean(x)) / std


# ============================================================
# FEATURE EXTRACTION
#
# Deliberately scale invariant / post-normalization.
# ============================================================

def spectral_features(x, fs=FS):
    f, pxx = signal.welch(
        x,
        fs=fs,
        nperseg=min(512, len(x)),
    )

    mask = (f >= LOW) & (f <= HIGH)

    f = f[mask]
    pxx = pxx[mask]

    power_sum = np.sum(pxx)

    if power_sum <= 0:
        return np.nan, np.nan, np.nan

    p = pxx / power_sum

    dominant_freq = float(
        f[np.argmax(pxx)]
    )

    centroid = float(
        np.sum(f * p)
    )

    entropy = float(
        -np.sum(
            p * np.log(p + 1e-12)
        ) / np.log(len(p))
    )

    return (
        dominant_freq,
        centroid,
        entropy,
    )


def extract_features(x):
    # x is already filtered + z-normalized

    dx = np.diff(x)

    dominant_freq, centroid, spec_entropy = (
        spectral_features(x)
    )

    zero_cross = np.mean(
        np.diff(np.signbit(x)) != 0
    )

    # Shape/morphology statistics
    skew = stats.skew(
        x,
        bias=False,
    )

    kurt = stats.kurtosis(
        x,
        fisher=True,
        bias=False,
    )

    derivative_std = np.std(dx)

    derivative_mad = np.median(
        np.abs(
            dx - np.median(dx)
        )
    )

    # Autocorrelation morphology
    ac = signal.correlate(
        x,
        x,
        mode="full",
        method="fft",
    )

    ac = ac[len(x)-1:]

    if ac[0] != 0:
        ac = ac / ac[0]

    # physiological lag range:
    # 0.3 s – 2.0 s
    lo = int(0.3 * FS)
    hi = int(2.0 * FS)

    ac_region = ac[lo:hi]

    if len(ac_region):
        ac_peak = float(
            np.max(ac_region)
        )
    else:
        ac_peak = np.nan

    return {
        "skew": float(skew),
        "kurtosis": float(kurt),

        "derivative_std":
            float(derivative_std),

        "derivative_mad":
            float(derivative_mad),

        "zero_cross_rate":
            float(zero_cross),

        "dominant_freq":
            dominant_freq,

        "spectral_centroid":
            centroid,

        "spectral_entropy":
            spec_entropy,

        "autocorr_peak":
            ac_peak,
    }


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

# ------------------------------------------------------------
# CRITICAL CONTROL:
# NSR ONLY
#
# Exclude 4-col because it only has 2 NSR subjects.
# ------------------------------------------------------------

meta = meta[
    (meta["class_name"] == "NSR")
    &
    (
        meta["format"]
        .isin(
            [
                "1col_legacy",
                "3col",
            ]
        )
    )
].copy()

meta["domain_label"] = (
    meta["format"]
    .map(
        {
            "1col_legacy": 0,
            "3col": 1,
        }
    )
)

print("=" * 90)
print("06A.4.1 PULSEWATCH CONTROLLED DOMAIN PROBE")
print("=" * 90)

print("NSR segments:", len(meta))
print("Subjects:", meta["uid"].nunique())

print()
print("Segments by format:")
print(
    meta["format"]
    .value_counts()
)

print()
print("Subjects by format:")
print(
    meta.groupby("format")["uid"]
    .nunique()
)


# ============================================================
# READ RAW PPG + PREPROCESS + FEATURES
# ============================================================

rows = []

done = 0

for tar_path, g in meta.groupby(
    "tar_path",
    sort=True,
):

    print(
        f"[{done:6d}/{len(meta)}] "
        f"{Path(tar_path).name} "
        f"n={len(g)}"
    )

    with tarfile.open(
        tar_path,
        "r",
    ) as tf:

        for _, r in g.iterrows():

            fobj = tf.extractfile(
                r["member"]
            )

            if fobj is None:
                continue

            text = fobj.read().decode(
                "utf-8",
                errors="strict",
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

            n_cols = arr.shape[1]

            if n_cols == 1:
                ppg = arr[:, 0]

            elif n_cols in (3, 4):
                # Official Pulsewatch convention:
                # raw PPG = second column.
                ppg = arr[:, 1]

            else:
                continue

            if len(ppg) != 1500:
                continue

            if not np.isfinite(ppg).all():
                continue

            ppg_f = bandpass(
                ppg,
                FS,
            )

            ppg_z = zscore_segment(
                ppg_f
            )

            if ppg_z is None:
                continue

            feats = extract_features(
                ppg_z
            )

            row = {
                "segment":
                    r["segment"],

                "uid":
                    r["uid"],

                "format":
                    r["format"],

                "domain_label":
                    int(
                        r["domain_label"]
                    ),
            }

            row.update(feats)

            rows.append(row)

    done += len(g)


seg = pd.DataFrame(rows)

seg.to_csv(
    OUT_SEG,
    index=False,
)

print()
print("=" * 90)
print("SEGMENT FEATURES")
print("=" * 90)

print("Extracted:", len(seg))
print("Subjects :", seg["uid"].nunique())

print()
print(
    seg["format"]
    .value_counts()
)


# ============================================================
# SUBJECT-LEVEL AGGREGATION
#
# Each subject becomes ONE sample.
# Prevents subjects with thousands of segments dominating.
# ============================================================

FEATURES = [
    "skew",
    "kurtosis",
    "derivative_std",
    "derivative_mad",
    "zero_cross_rate",
    "dominant_freq",
    "spectral_centroid",
    "spectral_entropy",
    "autocorr_peak",
]

subject = (
    seg.groupby(
        [
            "uid",
            "format",
            "domain_label",
        ],
        as_index=False,
    )[FEATURES]
    .median()
)

subject.to_csv(
    OUT_SUBJ,
    index=False,
)

print()
print("=" * 90)
print("SUBJECT-LEVEL DATA")
print("=" * 90)

print("Subjects:", len(subject))

print()
print(
    subject["format"]
    .value_counts()
)


# ============================================================
# CLEAN
# ============================================================

clean = subject.dropna(
    subset=FEATURES
).copy()

X = clean[FEATURES]
y = clean["domain_label"]


# ============================================================
# CLASSIFIER
#
# Diagnostic probe only.
# Logistic regression intentionally simple.
# ============================================================

model = Pipeline(
    [
        (
            "scale",
            StandardScaler(),
        ),
        (
            "clf",
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# Repeated CV at SUBJECT level.
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=20,
    random_state=RANDOM_STATE,
)

scoring = {
    "balanced_accuracy":
        make_scorer(
            balanced_accuracy_score
        ),

    "macro_f1":
        make_scorer(
            f1_score,
            average="macro",
        ),

    "roc_auc":
        "roc_auc",
}

result = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for metric in [
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
]:

    values = result[
        "test_" + metric
    ]

    summary_rows.append(
        {
            "metric": metric,
            "mean": np.mean(values),
            "std": np.std(values),
            "median": np.median(values),
            "p05": np.quantile(
                values,
                0.05,
            ),
            "p95": np.quantile(
                values,
                0.95,
            ),
        }
    )

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT_SUM,
    index=False,
)

print()
print("=" * 90)
print("CONTROLLED DOMAIN PROBE RESULT")
print("=" * 90)

print(
    summary.to_string(
        index=False
    )
)

print()
print("Chance balanced accuracy = 0.50")
print("Chance ROC-AUC           = 0.50")

print()
print("Saved:")
print(" ", OUT_SEG)
print(" ", OUT_SUBJ)
print(" ", OUT_SUM)
