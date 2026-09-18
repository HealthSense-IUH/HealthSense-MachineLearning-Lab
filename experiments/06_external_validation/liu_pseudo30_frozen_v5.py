import sys
sys.path.insert(0, "src")

from pathlib import Path
import ast

import joblib
import numpy as np
import pandas as pd
import scipy.io as sio

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
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

ROOT = Path(
    "data/raw/PPGArrhythmiaDetection-main"
)

DATA = ROOT / "valid_testDataset"

OUT = Path(
    "experiments/06_external_validation"
)

MODEL = Path(
    "models/mimic/"
    "healthsense_af_v5_rf_ac_frozen.pkl"
)

FS = 100.0
MIN_NN = 10

LABELS = {
    0: "SR",
    1: "PVC",
    2: "PAC",
    3: "VT",
    4: "SVT",
    5: "AF",
}

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

OUT_PRED = (
    OUT /
    "liu_pseudo30_frozen_v5_predictions.csv"
)

OUT_SUMMARY = (
    OUT /
    "liu_pseudo30_frozen_v5_summary.csv"
)

OUT_SUBJECT = (
    OUT /
    "liu_pseudo30_frozen_v5_subject_rhythm.csv"
)


# ============================================================
# PPG AC — same definition as frozen v5
# ============================================================

def autocorrelation_feature(x, fs):

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
        spectrum *
        np.conj(spectrum),
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
# TEST SUBJECTS
# ============================================================

subjects = []

for line in (
    ROOT /
    "testsubjects.txt"
).read_text(
    encoding="utf-8",
    errors="replace",
).splitlines():

    line = line.strip()

    if not line:
        continue

    try:
        name = ast.literal_eval(
            line
        )
    except Exception:
        name = line.strip("'\"")

    subjects.append(name)


print("=" * 105)
print("LIU PSEUDO-30s — FROZEN HEALTHSENSE V5")
print("=" * 105)

print(
    "Official test subjects:",
    len(subjects)
)


# ============================================================
# BUILD NON-OVERLAPPING 3x10s WINDOWS
# ============================================================

windows = []

for si, fname in enumerate(
    subjects,
    1,
):

    p = DATA / fname

    m = sio.loadmat(
        p,
        squeeze_me=True,
        struct_as_record=False,
    )

    ppg = np.asarray(
        m["ppgseg"],
        dtype=float,
    )

    labels = np.asarray(
        m["labels"]
    ).reshape(-1).astype(int)

    if ppg.ndim == 1:
        ppg = ppg.reshape(
            1,
            -1,
        )

    if (
        ppg.shape[0]
        != len(labels)
        and
        ppg.shape[1]
        == len(labels)
    ):
        ppg = ppg.T

    uid = fname.replace(
        ".mat",
        "",
    )

    # ----------------------------------------------
    # Find same-label stored-order runs
    # ----------------------------------------------

    start = 0

    for i in range(
        1,
        len(labels) + 1,
    ):

        boundary = (
            i == len(labels)
            or
            labels[i]
            != labels[start]
        )

        if not boundary:
            continue

        lab = int(
            labels[start]
        )

        run_end = i

        # non-overlapping triplets
        j = start

        while (
            j + 2
            < run_end
        ):

            x = np.concatenate([
                ppg[j],
                ppg[j + 1],
                ppg[j + 2],
            ])

            windows.append({
                "subject_id":
                    uid,

                "label_original":
                    lab,

                "rhythm":
                    LABELS[lab],

                "binary_label":
                    int(
                        lab == 5
                    ),

                "idx_0":
                    j,

                "idx_1":
                    j + 1,

                "idx_2":
                    j + 2,

                "ppg":
                    x,
            })

            j += 3

        start = i

    print(
        f"[{si:02d}/{len(subjects)}] "
        f"{uid}"
    )


print()
print(
    "Pseudo-30s windows:",
    len(windows)
)

rhythm_counts = pd.Series([
    x["rhythm"]
    for x in windows
]).value_counts()

print()
print("WINDOWS BY RHYTHM")
print(
    rhythm_counts.to_string()
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

rows = []

for i, w in enumerate(
    windows,
    1,
):

    if (
        i == 1
        or
        i % 500 == 0
        or
        i == len(windows)
    ):

        print(
            f"[feature "
            f"{i:05d}/"
            f"{len(windows):05d}]"
        )

    x = np.asarray(
        w["ppg"],
        dtype=float,
    )

    base = {
        k: v
        for k, v in w.items()
        if k != "ppg"
    }

    try:

        if (
            len(x) != 3000
            or
            not np.all(
                np.isfinite(x)
            )
        ):

            raise ValueError(
                "BAD_PPG"
            )

        nn_ms, nn_times = (
            extract_nn_series(
                x,
                fs=FS,
            )
        )

        if (
            len(nn_ms)
            < MIN_NN
        ):

            rows.append({
                **base,

                "feature_ok":
                    False,

                "error":
                    "TOO_FEW_NN",

                "n_nn":
                    len(nn_ms),
            })

            continue

        feats = (
            compute_hrv_features(
                nn_ms,
                nn_times,
            )
        )

        feats["PPG_AC"] = (
            autocorrelation_feature(
                x,
                FS,
            )
        )

        finite = all(
            np.isfinite(
                feats[f]
            )
            for f in FEATURES
        )

        if not finite:

            rows.append({
                **base,

                "feature_ok":
                    False,

                "error":
                    "NONFINITE_FEATURE",

                "n_nn":
                    len(nn_ms),

                **feats,
            })

            continue

        rows.append({
            **base,

            "feature_ok":
                True,

            "error":
                "",

            "n_nn":
                len(nn_ms),

            **feats,
        })

    except Exception as e:

        rows.append({
            **base,

            "feature_ok":
                False,

            "error":
                repr(e),
        })


df = pd.DataFrame(
    rows
)


# ============================================================
# COVERAGE
# ============================================================

print()
print("=" * 105)
print("COVERAGE")
print("=" * 105)

coverage = (
    df.groupby("rhythm")
      ["feature_ok"]
      .agg(
          ["count", "sum", "mean"]
      )
)

print(
    coverage.to_string()
)

ok = df[
    df.feature_ok == True
].copy()

print()
print(
    "Overall:",
    len(ok),
    "/",
    len(df),
    "=",
    len(ok) / len(df),
)


# ============================================================
# FROZEN MODEL
# ============================================================

model = joblib.load(
    MODEL
)

ok["af_probability"] = (
    model.predict_proba(
        ok[FEATURES]
    )[:, 1]
)

ok[
    "fixed_090_positive"
] = (
    ok.af_probability
    >= 0.90
)

ok[
    "fixed_085_positive"
] = (
    ok.af_probability
    >= 0.85
)

ok[
    "adaptive_threshold"
] = np.where(
    ok.HR_mean < 60,
    0.80,
    0.90,
)

ok[
    "adaptive_positive"
] = (
    ok.af_probability
    >=
    ok.adaptive_threshold
)

ok[
    "hr_lt60"
] = (
    ok.HR_mean < 60
)

ok.to_csv(
    OUT_PRED,
    index=False,
)


# ============================================================
# BINARY AF-vs-ALL
# ============================================================

y = ok[
    "binary_label"
].astype(int)

p = ok[
    "af_probability"
]


print()
print("=" * 105)
print("AF vs ALL")
print("=" * 105)

roc = roc_auc_score(
    y,
    p,
)

pr = average_precision_score(
    y,
    p,
)

print(
    f"ROC-AUC : {roc:.6f}"
)

print(
    f"PR-AUC  : {pr:.6f}"
)


RULES = {
    "fixed_090":
        "fixed_090_positive",

    "fixed_085":
        "fixed_085_positive",

    "adaptive":
        "adaptive_positive",
}


summary_rows = []


def binary_metrics(data, col):

    yt = (
        data.binary_label
        .astype(int)
    )

    yp = (
        data[col]
        .astype(int)
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            yt,
            yp,
            labels=[0, 1],
        ).ravel()
    )

    sens = (
        tp / (tp + fn)
        if tp + fn
        else np.nan
    )

    spec = (
        tn / (tn + fp)
        if tn + fp
        else np.nan
    )

    return (
        tn,
        fp,
        fn,
        tp,
        sens,
        spec,
    )


for name, col in (
    RULES.items()
):

    (
        tn,
        fp,
        fn,
        tp,
        sens,
        spec,
    ) = binary_metrics(
        ok,
        col,
    )

    print()
    print(name)

    print(
        f"  Sensitivity : "
        f"{sens * 100:.3f}%"
    )

    print(
        f"  Specificity : "
        f"{spec * 100:.3f}%"
    )

    print(
        f"  TP={tp} FN={fn} "
        f"TN={tn} FP={fp}"
    )

    summary_rows.append({
        "analysis":
            "AF_vs_ALL",

        "rhythm":
            "ALL",

        "rule":
            name,

        "n":
            len(ok),

        "positive_rate":
            np.mean(
                ok[col]
            ),

        "sensitivity":
            sens,

        "specificity":
            spec,
    })


# ============================================================
# PER-RHYTHM CONFUSION CHALLENGE
# ============================================================

print()
print("=" * 105)
print("PER-RHYTHM POSITIVE RATE")
print("=" * 105)

for rhythm in [
    "AF",
    "SR",
    "PAC",
    "PVC",
    "SVT",
    "VT",
]:

    d = ok[
        ok.rhythm == rhythm
    ]

    print(
        f"\n{rhythm} "
        f"n={len(d)}"
    )

    for name, col in (
        RULES.items()
    ):

        rate = float(
            d[col].mean()
        )

        print(
            f"  {name:12s} "
            f"{rate * 100:7.3f}%"
        )

        summary_rows.append({
            "analysis":
                (
                    "AF_sensitivity"
                    if rhythm == "AF"
                    else "false_positive"
                ),

            "rhythm":
                rhythm,

            "rule":
                name,

            "n":
                len(d),

            "positive_rate":
                rate,

            "sensitivity":
                (
                    rate
                    if rhythm == "AF"
                    else np.nan
                ),

            "specificity":
                (
                    np.nan
                    if rhythm == "AF"
                    else 1.0 - rate
                ),
        })


# ============================================================
# LOW-HR
# ============================================================

print()
print("=" * 105)
print("LOW HR")
print("=" * 105)

for name_hr, mask in [
    (
        "HR_LT60",
        ok.hr_lt60,
    ),
    (
        "HR_GE60",
        ~ok.hr_lt60,
    ),
]:

    d = ok[mask]

    print(
        f"\n{name_hr}: "
        f"n={len(d)} "
        f"AF="
        f"{int((d.rhythm == 'AF').sum())}"
    )

    for name, col in (
        RULES.items()
    ):

        af = d[
            d.rhythm == "AF"
        ]

        non = d[
            d.rhythm != "AF"
        ]

        sens = (
            af[col].mean()
            if len(af)
            else np.nan
        )

        fpr = (
            non[col].mean()
            if len(non)
            else np.nan
        )

        print(
            f"  {name:12s}"
            f" AF={sens:.4f}"
            f" nonAF_FP={fpr:.4f}"
        )


# ============================================================
# SUBJECT x RHYTHM
# ============================================================

subject_rows = []

for (
    uid,
    rhythm,
), g in ok.groupby([
    "subject_id",
    "rhythm",
]):

    row = {
        "subject_id":
            uid,

        "rhythm":
            rhythm,

        "n":
            len(g),

        "hr_median":
            g.HR_mean.median(),

        "p_af_median":
            g.af_probability.median(),

        "low_hr_fraction":
            g.hr_lt60.mean(),
    }

    for name, col in (
        RULES.items()
    ):

        row[
            f"{name}_positive_rate"
        ] = float(
            g[col].mean()
        )

    subject_rows.append(
        row
    )


subject_df = pd.DataFrame(
    subject_rows
)

subject_df.to_csv(
    OUT_SUBJECT,
    index=False,
)


print()
print("=" * 105)
print("SUBJECT-MACRO BY RHYTHM")
print("=" * 105)

for rhythm in [
    "AF",
    "SR",
    "PAC",
    "PVC",
    "SVT",
    "VT",
]:

    d = subject_df[
        subject_df.rhythm
        == rhythm
    ]

    if len(d) == 0:
        continue

    print(
        f"\n{rhythm}: "
        f"{len(d)} subjects"
    )

    for name in RULES:

        macro = (
            d[
                f"{name}_positive_rate"
            ].mean()
        )

        print(
            f"  {name:12s}"
            f" {macro * 100:7.3f}%"
        )


# ============================================================
# SAVE
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary[
    "ROC_AUC_AF_vs_ALL"
] = roc

summary[
    "PR_AUC_AF_vs_ALL"
] = pr

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)


print()
print("=" * 105)
print("INTERPRETATION")
print("=" * 105)

print(
    "These are pseudo-30s windows created "
    "from three adjacent stored 10-s segments."
)

print(
    "The public Liu dataset does not provide "
    "timestamps proving that every triplet is "
    "truly continuous after artifact removal."
)

print(
    "Therefore use this ONLY as an external "
    "multiclass sensitivity / robustness analysis."
)

print(
    "Do NOT report it as exact 30-s "
    "confirmatory validation."
)

print(
    "Do NOT evaluate production 3-of-3 "
    "alert persistence from these windows."
)

print()
print("SAVED:")
print(OUT_PRED)
print(OUT_SUMMARY)
print(OUT_SUBJECT)
