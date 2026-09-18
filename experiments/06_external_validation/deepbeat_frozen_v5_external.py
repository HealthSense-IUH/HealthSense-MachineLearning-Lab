import sys
sys.path.insert(0, "src")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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

NPZ = Path(
    "data/raw/test.npz"
)

MODEL_PATH = Path(
    "models/mimic/"
    "healthsense_af_v5_rf_ac_frozen.pkl"
)

OUT = Path(
    "experiments/06_external_validation"
)

OUT_FEATURES = (
    OUT /
    "deepbeat_frozen_v5_features.csv"
)

OUT_PRED = (
    OUT /
    "deepbeat_frozen_v5_predictions.csv"
)

OUT_SUBJECT = (
    OUT /
    "deepbeat_frozen_v5_subject_summary.csv"
)

OUT_QUALITY = (
    OUT /
    "deepbeat_frozen_v5_quality_summary.csv"
)

OUT_SUMMARY = (
    OUT /
    "deepbeat_frozen_v5_summary.csv"
)


# DeepBeat public test signal:
# 25 s × 32 Hz = 800 samples
FS = 32.0
WINDOW_S = 25.0

MIN_NN = 10


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


QUALITY_NAMES = {
    0: "poor",
    1: "acceptable",
    2: "excellent",
}


# ============================================================
# PPG AC — SAME DEFINITION AS FROZEN V5
# ============================================================

def autocorrelation_feature(
    x,
    fs,
):

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
# LOAD DATA
# ============================================================

print("=" * 105)
print("DEEPBEAT TEST — FROZEN HEALTHSENSE V5")
print("=" * 105)

with np.load(
    NPZ,
    allow_pickle=True,
) as z:

    signal = (
        z["signal"][:, :, 0]
        .astype(float)
    )

    rhythm = z["rhythm"]

    qa = z["qa_label"]

    params = z["parameters"]


y = np.argmax(
    rhythm,
    axis=1,
).astype(int)

quality = np.argmax(
    qa,
    axis=1,
).astype(int)

timestamps = pd.to_datetime(
    params[:, 0]
)

sources = np.array([
    str(x).strip()
    for x in params[:, 1]
])

subjects = (
    params[:, 2]
    .astype(int)
)


print(
    "\nSegments:",
    len(signal)
)

print(
    "Subjects:",
    len(
        np.unique(subjects)
    )
)

print(
    "AF:",
    int(
        np.sum(y == 1)
    )
)

print(
    "Non-AF:",
    int(
        np.sum(y == 0)
    )
)

print(
    "\nQuality:"
)

for q in sorted(
    np.unique(quality)
):

    print(
        q,
        QUALITY_NAMES.get(
            int(q),
            str(q)
        ),
        int(
            np.sum(
                quality == q
            )
        )
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

rows = []

for i, x in enumerate(
    signal
):

    if (
        i == 0
        or
        (i + 1) % 1000 == 0
        or
        i + 1 == len(signal)
    ):

        print(
            f"[{i + 1:05d}/"
            f"{len(signal):05d}]"
        )

    base = {
        "index":
            i,

        "subject_id":
            int(
                subjects[i]
            ),

        "timestamp":
            timestamps[i],

        "source":
            sources[i],

        "label":
            int(
                y[i]
            ),

        "rhythm":
            (
                "AF"
                if y[i] == 1
                else "NON_AF"
            ),

        "quality":
            int(
                quality[i]
            ),

        "quality_name":
            QUALITY_NAMES.get(
                int(
                    quality[i]
                ),
                str(
                    quality[i]
                ),
            ),
    }

    try:

        nn_ms, nn_times = (
            extract_nn_series(
                x,
                fs=FS,
            )
        )

        n_nn = len(
            nn_ms
        )

        if n_nn < MIN_NN:

            rows.append({
                **base,

                "feature_ok":
                    False,

                "error":
                    "TOO_FEW_NN",

                "n_nn":
                    n_nn,
            })

            continue

        feats = (
            compute_hrv_features(
                nn_ms,
                nn_times,
            )
        )

        feats[
            "PPG_AC"
        ] = (
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
                    n_nn,

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
                n_nn,

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


features_df = pd.DataFrame(
    rows
)

features_df.to_csv(
    OUT_FEATURES,
    index=False,
)


# ============================================================
# FROZEN INFERENCE
# ============================================================

ok = features_df[
    features_df[
        "feature_ok"
    ] == True
].copy()


print()
print("=" * 105)
print("COVERAGE")
print("=" * 105)

print(
    "Total       :",
    len(features_df)
)

print(
    "Feature OK  :",
    len(ok)
)

print(
    "No decision :",
    len(features_df)
    -
    len(ok)
)

print(
    "Coverage    :",
    len(ok)
    /
    len(features_df)
)


print("\nCOVERAGE BY RHYTHM")

coverage_rhythm = (
    features_df
    .groupby("rhythm")
    ["feature_ok"]
    .agg(
        [
            "count",
            "sum",
            "mean",
        ]
    )
)

print(
    coverage_rhythm
    .to_string()
)


print("\nCOVERAGE BY QUALITY")

coverage_quality = (
    features_df
    .groupby(
        "quality_name"
    )["feature_ok"]
    .agg(
        [
            "count",
            "sum",
            "mean",
        ]
    )
)

print(
    coverage_quality
    .to_string()
)


if len(ok) == 0:

    raise RuntimeError(
        "No usable DeepBeat windows"
    )


model = joblib.load(
    MODEL_PATH
)

X = ok[
    FEATURES
]

ok[
    "af_probability"
] = (
    model.predict_proba(
        X
    )[:, 1]
)


# ============================================================
# FROZEN POLICIES
# ============================================================

ok[
    "fixed_090_positive"
] = (
    ok[
        "af_probability"
    ]
    >= 0.90
)

ok[
    "fixed_085_positive"
] = (
    ok[
        "af_probability"
    ]
    >= 0.85
)

ok[
    "adaptive_threshold"
] = np.where(
    ok[
        "HR_mean"
    ] < 60.0,
    0.80,
    0.90,
)

ok[
    "adaptive_positive"
] = (
    ok[
        "af_probability"
    ]
    >=
    ok[
        "adaptive_threshold"
    ]
)

ok[
    "hr_lt60"
] = (
    ok[
        "HR_mean"
    ]
    < 60.0
)


ok.to_csv(
    OUT_PRED,
    index=False,
)


# ============================================================
# GLOBAL DISCRIMINATION
# ============================================================

truth = (
    ok[
        "label"
    ]
    .astype(int)
    .to_numpy()
)

prob = (
    ok[
        "af_probability"
    ]
    .to_numpy()
)


roc = roc_auc_score(
    truth,
    prob,
)

pr = average_precision_score(
    truth,
    prob,
)


print()
print("=" * 105)
print("GLOBAL DISCRIMINATION")
print("=" * 105)

print(
    f"ROC-AUC : {roc:.6f}"
)

print(
    f"PR-AUC  : {pr:.6f}"
)


# ============================================================
# THRESHOLD METRICS
# ============================================================

rules = {
    "fixed_090":
        "fixed_090_positive",

    "fixed_085":
        "fixed_085_positive",

    "hr_lt60_080_else090":
        "adaptive_positive",
}


summary_rows = []


def threshold_metrics(
    df,
    col,
):

    yt = (
        df["label"]
        .astype(int)
        .to_numpy()
    )

    yp = (
        df[col]
        .astype(int)
        .to_numpy()
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            yt,
            yp,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )

    sensitivity = (
        tp /
        (tp + fn)
        if tp + fn
        else np.nan
    )

    specificity = (
        tn /
        (tn + fp)
        if tn + fp
        else np.nan
    )

    fpr = (
        fp /
        (fp + tn)
        if fp + tn
        else np.nan
    )

    return {
        "TP":
            tp,

        "FN":
            fn,

        "TN":
            tn,

        "FP":
            fp,

        "Sensitivity":
            sensitivity,

        "Specificity":
            specificity,

        "FPR":
            fpr,
    }


print()
print("=" * 105)
print("SEGMENT-LEVEL FROZEN POLICY METRICS")
print("=" * 105)

for rule, col in (
    rules.items()
):

    m = threshold_metrics(
        ok,
        col,
    )

    summary_rows.append({
        "subset":
            "ALL",

        "rule":
            rule,

        "n":
            len(ok),

        **m,
    })

    print(
        f"\n{rule}"
    )

    print(
        f"  Sensitivity : "
        f"{m['Sensitivity'] * 100:.3f}%"
    )

    print(
        f"  Specificity : "
        f"{m['Specificity'] * 100:.3f}%"
    )

    print(
        f"  FPR         : "
        f"{m['FPR'] * 100:.3f}%"
    )

    print(
        "  "
        f"TP={m['TP']} "
        f"FN={m['FN']} "
        f"TN={m['TN']} "
        f"FP={m['FP']}"
    )


# ============================================================
# LOW-HR STRATIFICATION
# ============================================================

print()
print("=" * 105)
print("LOW-HR STRATIFICATION")
print("=" * 105)

for low_hr_value, name in [
    (
        True,
        "HR_LT60",
    ),
    (
        False,
        "HR_GE60",
    ),
]:

    d = ok[
        ok[
            "hr_lt60"
        ]
        ==
        low_hr_value
    ]

    print(
        f"\n{name}: "
        f"{len(d)} segments"
    )

    print(
        "  AF:",
        int(
            (
                d.label
                == 1
            ).sum()
        ),
        " Non-AF:",
        int(
            (
                d.label
                == 0
            ).sum()
        ),
    )

    if (
        d.label.nunique()
        < 2
    ):
        print(
            "  Only one class; "
            "threshold metrics limited."
        )

    for rule, col in (
        rules.items()
    ):

        m = threshold_metrics(
            d,
            col,
        )

        summary_rows.append({
            "subset":
                name,

            "rule":
                rule,

            "n":
                len(d),

            **m,
        })

        print(
            f"  {rule:24s}"
            f" sens="
            f"{m['Sensitivity']:.4f}"
            f" spec="
            f"{m['Specificity']:.4f}"
            f" fpr="
            f"{m['FPR']:.4f}"
        )


# ============================================================
# QUALITY STRATIFICATION
# ============================================================

quality_rows = []

print()
print("=" * 105)
print("QUALITY STRATIFICATION")
print("=" * 105)

for q in sorted(
    ok[
        "quality"
    ].unique()
):

    d = ok[
        ok[
            "quality"
        ] == q
    ]

    qname = (
        QUALITY_NAMES.get(
            int(q),
            str(q),
        )
    )

    print(
        f"\nQUALITY {q} "
        f"({qname}) "
        f"n={len(d)}"
    )

    if (
        d[
            "label"
        ].nunique()
        == 2
    ):

        q_roc = (
            roc_auc_score(
                d["label"],
                d[
                    "af_probability"
                ],
            )
        )

        q_pr = (
            average_precision_score(
                d["label"],
                d[
                    "af_probability"
                ],
            )
        )

    else:

        q_roc = np.nan
        q_pr = np.nan

    print(
        f"  ROC-AUC={q_roc:.6f}"
    )

    print(
        f"  PR-AUC ={q_pr:.6f}"
    )

    for rule, col in (
        rules.items()
    ):

        m = threshold_metrics(
            d,
            col,
        )

        quality_rows.append({
            "quality":
                int(q),

            "quality_name":
                qname,

            "rule":
                rule,

            "n":
                len(d),

            "ROC_AUC":
                q_roc,

            "PR_AUC":
                q_pr,

            **m,
        })

        print(
            f"  {rule:24s}"
            f" sens="
            f"{m['Sensitivity']:.4f}"
            f" spec="
            f"{m['Specificity']:.4f}"
        )


quality_df = pd.DataFrame(
    quality_rows
)

quality_df.to_csv(
    OUT_QUALITY,
    index=False,
)


# ============================================================
# SUBJECT LEVEL
# ============================================================

subject_rows = []

for uid, g in ok.groupby(
    "subject_id"
):

    label = int(
        g[
            "label"
        ].iloc[0]
    )

    row = {
        "subject_id":
            uid,

        "label":
            label,

        "rhythm":
            (
                "AF"
                if label == 1
                else "NON_AF"
            ),

        "n_segments":
            len(g),

        "coverage":
            len(g)
            /
            int(
                (
                    features_df[
                        "subject_id"
                    ] == uid
                ).sum()
            ),

        "hr_median":
            g[
                "HR_mean"
            ].median(),

        "low_hr_fraction":
            g[
                "hr_lt60"
            ].mean(),

        "p_af_mean":
            g[
                "af_probability"
            ].mean(),

        "p_af_median":
            g[
                "af_probability"
            ].median(),

        "p_af_p90":
            g[
                "af_probability"
            ].quantile(
                0.90
            ),
    }

    for rule, col in (
        rules.items()
    ):

        row[
            f"{rule}_positive_rate"
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
print("SUBJECT-MACRO")
print("=" * 105)

for rule in rules:

    col = (
        f"{rule}_positive_rate"
    )

    af_subjects = (
        subject_df[
            subject_df.label == 1
        ]
    )

    non_subjects = (
        subject_df[
            subject_df.label == 0
        ]
    )

    af_macro = (
        af_subjects[
            col
        ].mean()
    )

    non_fp_macro = (
        non_subjects[
            col
        ].mean()
    )

    print(
        f"{rule:24s}"
        f" AF sensitivity="
        f"{af_macro * 100:7.3f}%"
        f" | non-AF FP="
        f"{non_fp_macro * 100:7.3f}%"
    )

    summary_rows.append({
        "subset":
            "SUBJECT_MACRO",

        "rule":
            rule,

        "n":
            len(subject_df),

        "TP":
            np.nan,

        "FN":
            np.nan,

        "TN":
            np.nan,

        "FP":
            np.nan,

        "Sensitivity":
            af_macro,

        "Specificity":
            1.0
            -
            non_fp_macro,

        "FPR":
            non_fp_macro,
    })


print()
print("WORST AF SUBJECTS")

print(
    subject_df[
        subject_df.label == 1
    ]
    .sort_values(
        "fixed_090_positive_rate"
    )
    [
        [
            "subject_id",
            "n_segments",
            "hr_median",
            "low_hr_fraction",
            "p_af_median",
            "fixed_090_positive_rate",
            "fixed_085_positive_rate",
            "hr_lt60_080_else090_positive_rate",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("WORST NON-AF SUBJECTS")

print(
    subject_df[
        subject_df.label == 0
    ]
    .sort_values(
        "fixed_090_positive_rate",
        ascending=False,
    )
    [
        [
            "subject_id",
            "n_segments",
            "hr_median",
            "low_hr_fraction",
            "p_af_median",
            "fixed_090_positive_rate",
            "fixed_085_positive_rate",
            "hr_lt60_080_else090_positive_rate",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_df[
    "ROC_AUC"
] = roc

summary_df[
    "PR_AUC"
] = pr

summary_df.to_csv(
    OUT_SUMMARY,
    index=False,
)


print()
print("=" * 105)
print("IMPORTANT INTERPRETATION")
print("=" * 105)

print(
    "DeepBeat windows are 25 s, "
    "whereas frozen HealthSense v5 "
    "was developed on 30-s windows."
)

print(
    "Therefore this is an independent "
    "cross-domain / duration-shift stress test."
)

print(
    "Do NOT interpret these segment results "
    "as exact production 30-s performance."
)

print(
    "Do NOT compute the 3-of-3 alert rule "
    "from DeepBeat stored order."
)

print()
print("SAVED:")
print(OUT_FEATURES)
print(OUT_PRED)
print(OUT_SUBJECT)
print(OUT_QUALITY)
print(OUT_SUMMARY)
