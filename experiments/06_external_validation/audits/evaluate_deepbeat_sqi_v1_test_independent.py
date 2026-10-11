import sys
sys.path.insert(0, "src")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis
from sklearn.metrics import roc_auc_score

from healthsense_ml.signal_processing import (
    bandpass_filter,
)


# ============================================================
# CONFIG
# ============================================================

TEST_NPZ = Path(
    "data/raw/test.npz"
)

TEST_MANIFEST = Path(
    "experiments/06_external_validation/"
    "deepbeat_test_subject_independent_manifest.csv"
)

SQI_MODEL = Path(
    "experiments/06_external_validation/"
    "deepbeat_sqi_v1_frozen.joblib"
)

AF_PRED = Path(
    "experiments/06_external_validation/"
    "deepbeat_frozen_v5_predictions.csv"
)

OUT = Path(
    "experiments/06_external_validation"
)

OUT_SQI_FEATURES = (
    OUT /
    "deepbeat_test_independent_sqi_v1_features.csv"
)

OUT_PRED = (
    OUT /
    "deepbeat_test_independent_sqi_v1_predictions.csv"
)

OUT_QUALITY = (
    OUT /
    "deepbeat_test_independent_sqi_v1_quality_summary.csv"
)

OUT_SUBJECT = (
    OUT /
    "deepbeat_test_independent_sqi_v1_subject_summary.csv"
)

OUT_PIPELINE = (
    OUT /
    "deepbeat_test_independent_sqi_v1_pipeline_summary.csv"
)


FS = 32.0
EPS = 1e-12
AF_THRESHOLD = 0.90


# ============================================================
# EXACT SAME SQI FEATURE DEFINITIONS USED DURING FREEZE
# ============================================================

def morphology_features(x):

    x = np.asarray(
        x,
        dtype=float,
    ).reshape(-1)

    if (
        len(x) != 800
        or
        not np.all(
            np.isfinite(x)
        )
    ):
        raise ValueError(
            "Invalid waveform"
        )

    mean_x = float(
        np.mean(x)
    )

    std_x = float(
        np.std(x)
    )

    xmin = float(
        np.min(x)
    )

    xmax = float(
        np.max(x)
    )

    amplitude_range = (
        xmax - xmin
    )

    dx = np.diff(x)

    ddx = np.diff(
        x,
        n=2,
    )

    mad_dx = float(
        np.median(
            np.abs(
                dx
                -
                np.median(dx)
            )
        )
    )

    mad_ddx = float(
        np.median(
            np.abs(
                ddx
                -
                np.median(ddx)
            )
        )
    )

    derivative_noise = (
        mad_dx
        /
        (std_x + EPS)
    )

    second_derivative_noise = (
        mad_ddx
        /
        (std_x + EPS)
    )

    flat_fraction = float(
        np.mean(
            np.abs(dx)
            < 1e-4
        )
    )

    clipping_low_fraction = float(
        np.mean(
            x <= 0.01
        )
    )

    clipping_high_fraction = float(
        np.mean(
            x >= 0.99
        )
    )

    clipping_fraction = (
        clipping_low_fraction
        +
        clipping_high_fraction
    )

    skewness = float(
        skew(
            x,
            bias=False,
        )
    )

    kurt = float(
        kurtosis(
            x,
            fisher=True,
            bias=False,
        )
    )

    # --------------------------------------------------------
    # Spectrum
    # --------------------------------------------------------

    centered = (
        x - mean_x
    )

    windowed = (
        centered
        *
        np.hanning(
            len(centered)
        )
    )

    spectrum = np.fft.rfft(
        windowed
    )

    psd = (
        np.abs(
            spectrum
        ) ** 2
    )

    freq = np.fft.rfftfreq(
        len(centered),
        d=1.0 / FS,
    )

    band = (
        (freq >= 0.5)
        &
        (freq <= 8.0)
    )

    phys = (
        (freq >= 0.5)
        &
        (freq <= 5.0)
    )

    hf = (
        (freq > 5.0)
        &
        (freq <= 8.0)
    )

    band_power = float(
        np.sum(
            psd[band]
        )
    )

    phys_power = float(
        np.sum(
            psd[phys]
        )
    )

    hf_power = float(
        np.sum(
            psd[hf]
        )
    )

    physio_band_ratio = (
        phys_power
        /
        (band_power + EPS)
    )

    hf_noise_ratio = (
        hf_power
        /
        (band_power + EPS)
    )

    p = psd[band]

    if (
        len(p) > 1
        and
        np.sum(p) > EPS
    ):

        pn = (
            p
            /
            np.sum(p)
        )

        spectral_entropy = float(
            -np.sum(
                pn
                *
                np.log(
                    pn + EPS
                )
            )
            /
            np.log(
                len(p)
            )
        )

        dominant_power_fraction = float(
            np.max(pn)
        )

    else:

        spectral_entropy = np.nan
        dominant_power_fraction = np.nan

    # --------------------------------------------------------
    # Autocorrelation
    # --------------------------------------------------------

    xf = bandpass_filter(
        x,
        fs=FS,
    )

    xf = (
        xf
        -
        np.mean(xf)
    )

    n = len(xf)

    nfft = 1 << (
        (2 * n - 1).bit_length()
    )

    sp = np.fft.rfft(
        xf,
        n=nfft,
    )

    ac = np.fft.irfft(
        sp
        *
        np.conj(sp),
        n=nfft,
    )[:n]

    ac = ac / n

    if (
        np.isfinite(
            ac[0]
        )
        and
        abs(ac[0]) > EPS
    ):

        ac = (
            ac
            /
            ac[0]
        )

        ppg_ac = float(
            np.mean(
                np.abs(
                    ac[1:]
                )
            )
        )

        lag_min = max(
            1,
            int(
                round(
                    0.30 * FS
                )
            )
        )

        lag_max = min(
            len(ac) - 1,
            int(
                round(
                    2.00 * FS
                )
            )
        )

        ac_peak = float(
            np.max(
                ac[
                    lag_min:
                    lag_max + 1
                ]
            )
        )

        ac_abs_peak = float(
            np.max(
                np.abs(
                    ac[
                        lag_min:
                        lag_max + 1
                    ]
                )
            )
        )

    else:

        ppg_ac = np.nan
        ac_peak = np.nan
        ac_abs_peak = np.nan

    return {
        "raw_mean":
            mean_x,

        "raw_std":
            std_x,

        "amplitude_range":
            amplitude_range,

        "derivative_noise":
            derivative_noise,

        "second_derivative_noise":
            second_derivative_noise,

        "flat_fraction":
            flat_fraction,

        "clipping_low_fraction":
            clipping_low_fraction,

        "clipping_high_fraction":
            clipping_high_fraction,

        "clipping_fraction":
            clipping_fraction,

        "skewness":
            skewness,

        "kurtosis":
            kurt,

        "physio_band_ratio":
            physio_band_ratio,

        "hf_noise_ratio":
            hf_noise_ratio,

        "spectral_entropy":
            spectral_entropy,

        "dominant_power_fraction":
            dominant_power_fraction,

        "PPG_AC":
            ppg_ac,

        "ac_peak":
            ac_peak,

        "ac_abs_peak":
            ac_abs_peak,
    }


# ============================================================
# LOAD FROZEN SQI
# ============================================================

artifact = joblib.load(
    SQI_MODEL
)

sqi_pipeline = artifact[
    "pipeline"
]

SQI_FEATURES = artifact[
    "features"
]

SQI_THRESHOLD = float(
    artifact[
        "quality_score_threshold"
    ]
)

development_subjects = set(
    int(x)
    for x in artifact[
        "development_subjects"
    ]
)


print("=" * 110)
print("DEEPBEAT SUBJECT-INDEPENDENT TEST — FROZEN SQI V1")
print("=" * 110)

print(
    "SQI feature set:",
    artifact[
        "feature_set"
    ]
)

print(
    "SQI features:",
    SQI_FEATURES
)

print(
    "Frozen SQI threshold:",
    f"{SQI_THRESHOLD:.12f}"
)

print(
    "SQI development subjects:",
    sorted(
        development_subjects
    )
)


# ============================================================
# LOAD INDEPENDENT TEST MANIFEST
# ============================================================

manifest = pd.read_csv(
    TEST_MANIFEST
)

manifest[
    "global_index"
] = (
    manifest[
        "global_index"
    ].astype(int)
)

manifest[
    "subject_id"
] = (
    manifest[
        "subject_id"
    ].astype(int)
)

test_subjects = set(
    manifest[
        "subject_id"
    ].unique()
)


overlap = (
    development_subjects
    &
    test_subjects
)


print()
print(
    "Test windows:",
    len(manifest)
)

print(
    "Test subjects:",
    len(test_subjects)
)

print(
    "Test subject IDs:",
    sorted(
        test_subjects
    )
)

print(
    "Development/test subject overlap:",
    sorted(
        overlap
    )
)


if overlap:

    raise RuntimeError(
        "SUBJECT LEAKAGE DETECTED"
    )


# ============================================================
# LOAD TEST SIGNAL
#
# test.npz is small enough (~17k × 800) to load safely.
# ============================================================

print()
print(
    "Loading DeepBeat test signal..."
)

with np.load(
    TEST_NPZ,
    allow_pickle=False,
) as z:

    signal = (
        z["signal"]
        [:, :, 0]
        .astype(
            float,
            copy=False,
        )
    )


print(
    "Signal shape:",
    signal.shape
)


# ============================================================
# EXTRACT FROZEN SQI FEATURES
# ============================================================

feature_rows = []

for j, r in enumerate(
    manifest.itertuples(
        index=False
    ),
    1,
):

    idx = int(
        r.global_index
    )

    x = signal[
        idx
    ]

    feats = morphology_features(
        x
    )

    feature_rows.append({
        "global_index":
            idx,

        "subject_id":
            int(
                r.subject_id
            ),

        "label":
            int(
                r.label
            ),

        "rhythm":
            r.rhythm,

        "quality":
            int(
                r.quality
            ),

        "quality_name":
            r.quality_name,

        "quality_good":
            int(
                int(
                    r.quality
                )
                > 0
            ),

        **feats,
    })

    if (
        j == 1
        or
        j % 500 == 0
        or
        j == len(manifest)
    ):

        print(
            f"[SQI "
            f"{j:05d}/"
            f"{len(manifest):05d}]"
        )


sqi = pd.DataFrame(
    feature_rows
)


# ============================================================
# APPLY FROZEN SQI — NO FITTING
# ============================================================

sqi[
    "sqi_probability"
] = (
    sqi_pipeline.predict_proba(
        sqi[
            SQI_FEATURES
        ]
    )[:, 1]
)

sqi[
    "sqi_accepted"
] = (
    sqi[
        "sqi_probability"
    ]
    >=
    SQI_THRESHOLD
)


sqi.to_csv(
    OUT_SQI_FEATURES,
    index=False,
)


# ============================================================
# TEST SQI ITSELF
# ============================================================

print()
print("=" * 110)
print("SQI QUALITY GENERALIZATION")
print("=" * 110)


quality_rows = []


for subset_name, d in [
    (
        "ALL",
        sqi,
    ),
    (
        "AF",
        sqi[
            sqi.rhythm
            == "AF"
        ],
    ),
    (
        "NON_AF",
        sqi[
            sqi.rhythm
            == "NON_AF"
        ],
    ),
]:

    auc = roc_auc_score(
        d[
            "quality_good"
        ],
        d[
            "sqi_probability"
        ],
    )

    good = d[
        d.quality_good
        == 1
    ]

    poor = d[
        d.quality_good
        == 0
    ]

    good_retention = float(
        good[
            "sqi_accepted"
        ].mean()
    )

    poor_rejection = float(
        1.0
        -
        poor[
            "sqi_accepted"
        ].mean()
    )

    acceptance = float(
        d[
            "sqi_accepted"
        ].mean()
    )

    quality_rows.append({
        "subset":
            subset_name,

        "n":
            len(d),

        "quality_auc":
            auc,

        "good_retention":
            good_retention,

        "poor_rejection":
            poor_rejection,

        "overall_acceptance":
            acceptance,
    })

    print()
    print(subset_name)

    print(
        f"  Quality AUC      : "
        f"{auc:.6f}"
    )

    print(
        f"  Good retention   : "
        f"{good_retention*100:.3f}%"
    )

    print(
        f"  Poor rejection   : "
        f"{poor_rejection*100:.3f}%"
    )

    print(
        f"  Overall accepted : "
        f"{acceptance*100:.3f}%"
    )


quality_df = pd.DataFrame(
    quality_rows
)


# ============================================================
# QUALITY × RHYTHM RETENTION
# ============================================================

print()
print("=" * 110)
print("SQI RETENTION BY RHYTHM × QUALITY")
print("=" * 110)


detail_rows = []

for rhythm in [
    "AF",
    "NON_AF",
]:

    for q in [
        0,
        1,
        2,
    ]:

        d = sqi[
            (sqi.rhythm == rhythm)
            &
            (sqi.quality == q)
        ]

        if len(d) == 0:
            continue

        rate = float(
            d[
                "sqi_accepted"
            ].mean()
        )

        detail_rows.append({
            "subset":
                f"{rhythm}_Q{q}",

            "n":
                len(d),

            "quality_auc":
                np.nan,

            "good_retention":
                (
                    rate
                    if q > 0
                    else np.nan
                ),

            "poor_rejection":
                (
                    1.0 - rate
                    if q == 0
                    else np.nan
                ),

            "overall_acceptance":
                rate,
        })

        print(
            f"{rhythm:7s} "
            f"quality={q} "
            f"n={len(d):5d} "
            f"accepted="
            f"{rate*100:7.3f}%"
        )


quality_df = pd.concat(
    [
        quality_df,
        pd.DataFrame(
            detail_rows
        ),
    ],
    ignore_index=True,
)

quality_df.to_csv(
    OUT_QUALITY,
    index=False,
)


# ============================================================
# PER-SUBJECT SQI
# ============================================================

print()
print("=" * 110)
print("SQI PER SUBJECT")
print("=" * 110)


for uid, d in sqi.groupby(
    "subject_id"
):

    good = d[
        d.quality_good == 1
    ]

    poor = d[
        d.quality_good == 0
    ]

    good_ret = (
        float(
            good.sqi_accepted.mean()
        )
        if len(good)
        else np.nan
    )

    poor_rej = (
        float(
            1.0
            -
            poor.sqi_accepted.mean()
        )
        if len(poor)
        else np.nan
    )

    print(
        f"subject={uid:3d} "
        f"{d.rhythm.iloc[0]:7s} "
        f"n={len(d):4d} "
        f"good_n={len(good):4d} "
        f"good_ret="
        f"{good_ret:7.4f} "
        f"poor_rej="
        f"{poor_rej:7.4f}"
    )


# ============================================================
# MERGE EXISTING FROZEN AF V5 PREDICTIONS
#
# These predictions were generated before SQI development.
# No AF model retraining or threshold change occurs here.
# ============================================================

af_pred = pd.read_csv(
    AF_PRED
)

af_pred = af_pred.rename(
    columns={
        "index":
            "global_index"
    }
)


needed = [
    "global_index",
    "af_probability",
]

if "HR_mean" in af_pred.columns:
    needed.append(
        "HR_mean"
    )


af_pred = af_pred[
    needed
].copy()


merged = sqi.merge(
    af_pred,
    on="global_index",
    how="left",
    validate="one_to_one",
)


merged[
    "af_decision"
] = (
    merged[
        "af_probability"
    ].notna()
)

merged[
    "baseline_af_positive"
] = (
    merged[
        "af_decision"
    ]
    &
    (
        merged[
            "af_probability"
        ]
        >=
        AF_THRESHOLD
    )
)

merged[
    "pipeline_decision"
] = (
    merged[
        "sqi_accepted"
    ]
    &
    merged[
        "af_decision"
    ]
)

merged[
    "pipeline_af_positive"
] = (
    merged[
        "pipeline_decision"
    ]
    &
    (
        merged[
            "af_probability"
        ]
        >=
        AF_THRESHOLD
    )
)


merged.to_csv(
    OUT_PRED,
    index=False,
)


# ============================================================
# PIPELINE METRICS
#
# Two different concepts are reported:
#
# SELECTIVE sensitivity/FPR:
#   performance only where the pipeline makes a decision.
#
# RAW positive fraction:
#   alerts divided by ALL raw windows, so abstention/no-decision
#   remains visible instead of being hidden.
# ============================================================

def evaluate_stage(
    data,
    stage,
):

    if stage == "BASELINE":

        decision = (
            data[
                "af_decision"
            ].to_numpy(
                dtype=bool
            )
        )

        positive = (
            data[
                "baseline_af_positive"
            ].to_numpy(
                dtype=bool
            )
        )

    elif stage == "SQI_V1":

        decision = (
            data[
                "pipeline_decision"
            ].to_numpy(
                dtype=bool
            )
        )

        positive = (
            data[
                "pipeline_af_positive"
            ].to_numpy(
                dtype=bool
            )
        )

    else:

        raise ValueError(
            stage
        )

    y = (
        data[
            "label"
        ].to_numpy(
            dtype=int
        )
    )

    af = (
        y == 1
    )

    non = (
        y == 0
    )

    af_decision = (
        af
        &
        decision
    )

    non_decision = (
        non
        &
        decision
    )

    selective_sensitivity = (
        positive[
            af_decision
        ].mean()
        if af_decision.sum()
        else np.nan
    )

    selective_fpr = (
        positive[
            non_decision
        ].mean()
        if non_decision.sum()
        else np.nan
    )

    af_decision_coverage = (
        af_decision.sum()
        /
        af.sum()
    )

    non_decision_coverage = (
        non_decision.sum()
        /
        non.sum()
    )

    overall_coverage = float(
        decision.mean()
    )

    # Not called sensitivity/FPR because abstentions
    # are included in denominators.
    raw_af_positive_fraction = (
        positive[
            af
        ].sum()
        /
        af.sum()
    )

    raw_non_af_positive_fraction = (
        positive[
            non
        ].sum()
        /
        non.sum()
    )

    auc = np.nan

    if (
        len(
            np.unique(
                y[
                    decision
                ]
            )
        )
        == 2
    ):

        auc = roc_auc_score(
            y[
                decision
            ],
            data.loc[
                decision,
                "af_probability",
            ],
        )

    return {
        "stage":
            stage,

        "n_raw":
            len(data),

        "n_decision":
            int(
                decision.sum()
            ),

        "overall_decision_coverage":
            overall_coverage,

        "AF_decision_coverage":
            af_decision_coverage,

        "NON_AF_decision_coverage":
            non_decision_coverage,

        "selective_AF_sensitivity":
            selective_sensitivity,

        "selective_NON_AF_FPR":
            selective_fpr,

        "AF_positive_fraction_all_raw":
            raw_af_positive_fraction,

        "NON_AF_positive_fraction_all_raw":
            raw_non_af_positive_fraction,

        "ROC_AUC_decision_windows":
            auc,
    }


pipeline_rows = []


print()
print("=" * 110)
print("AF V5 BEFORE vs AFTER FROZEN SQI")
print("=" * 110)


for stage in [
    "BASELINE",
    "SQI_V1",
]:

    m = evaluate_stage(
        merged,
        stage,
    )

    pipeline_rows.append(
        m
    )

    print()
    print(stage)

    print(
        f"  Decision coverage       : "
        f"{m['overall_decision_coverage']*100:.3f}%"
    )

    print(
        f"  AF decision coverage    : "
        f"{m['AF_decision_coverage']*100:.3f}%"
    )

    print(
        f"  Non-AF decision coverage: "
        f"{m['NON_AF_decision_coverage']*100:.3f}%"
    )

    print(
        f"  Selective sensitivity   : "
        f"{m['selective_AF_sensitivity']*100:.3f}%"
    )

    print(
        f"  Selective non-AF FPR    : "
        f"{m['selective_NON_AF_FPR']*100:.3f}%"
    )

    print(
        f"  AF positive / all raw   : "
        f"{m['AF_positive_fraction_all_raw']*100:.3f}%"
    )

    print(
        f"  Non-AF positive/all raw : "
        f"{m['NON_AF_positive_fraction_all_raw']*100:.3f}%"
    )

    print(
        f"  ROC-AUC decision windows: "
        f"{m['ROC_AUC_decision_windows']:.6f}"
    )


# ============================================================
# AF PERFORMANCE BY TRUE QUALITY
# ============================================================

print()
print("=" * 110)
print("AF MODEL BEFORE/AFTER SQI BY TRUE QUALITY")
print("=" * 110)


for q in [
    0,
    1,
    2,
]:

    d = merged[
        merged.quality == q
    ]

    print(
        f"\nQUALITY={q} "
        f"n={len(d)}"
    )

    for stage in [
        "BASELINE",
        "SQI_V1",
    ]:

        m = evaluate_stage(
            d,
            stage,
        )

        pipeline_rows.append({
            **m,

            "stage":
                f"{stage}_Q{q}",
        })

        print(
            f"  {stage:8s} "
            f"coverage="
            f"{m['overall_decision_coverage']:.4f} "
            f"sens="
            f"{m['selective_AF_sensitivity']:.4f} "
            f"FPR="
            f"{m['selective_NON_AF_FPR']:.4f} "
            f"AUC="
            f"{m['ROC_AUC_decision_windows']:.4f}"
        )


# ============================================================
# SUBJECT SUMMARY
# ============================================================

subject_rows = []


for uid, g in merged.groupby(
    "subject_id"
):

    label = int(
        g[
            "label"
        ].iloc[0]
    )

    baseline_dec = g[
        g.af_decision
    ]

    after_dec = g[
        g.pipeline_decision
    ]

    subject_rows.append({
        "subject_id":
            int(uid),

        "label":
            label,

        "rhythm":
            g.rhythm.iloc[0],

        "n_raw":
            len(g),

        "good_quality_fraction":
            float(
                g.quality_good.mean()
            ),

        "sqi_acceptance":
            float(
                g.sqi_accepted.mean()
            ),

        "baseline_decision_coverage":
            float(
                g.af_decision.mean()
            ),

        "after_sqi_decision_coverage":
            float(
                g.pipeline_decision.mean()
            ),

        "baseline_positive_rate_decisions":
            (
                float(
                    baseline_dec[
                        "baseline_af_positive"
                    ].mean()
                )
                if len(
                    baseline_dec
                )
                else np.nan
            ),

        "after_sqi_positive_rate_decisions":
            (
                float(
                    after_dec[
                        "pipeline_af_positive"
                    ].mean()
                )
                if len(
                    after_dec
                )
                else np.nan
            ),

        "baseline_positive_fraction_all_raw":
            float(
                g[
                    "baseline_af_positive"
                ].sum()
                /
                len(g)
            ),

        "after_sqi_positive_fraction_all_raw":
            float(
                g[
                    "pipeline_af_positive"
                ].sum()
                /
                len(g)
            ),
    })


subject_df = pd.DataFrame(
    subject_rows
)

subject_df.to_csv(
    OUT_SUBJECT,
    index=False,
)


print()
print("=" * 110)
print("SUBJECT LEVEL")
print("=" * 110)

print(
    subject_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# SUBJECT-MACRO
# ============================================================

print()
print("=" * 110)
print("SUBJECT-MACRO BEFORE / AFTER SQI")
print("=" * 110)


for label, rhythm in [
    (
        1,
        "AF",
    ),
    (
        0,
        "NON_AF",
    ),
]:

    d = subject_df[
        subject_df.label
        == label
    ]

    print(
        f"\n{rhythm}: "
        f"{len(d)} subjects"
    )

    before = (
        d[
            "baseline_positive_rate_decisions"
        ].mean()
    )

    after = (
        d[
            "after_sqi_positive_rate_decisions"
        ].mean()
    )

    before_raw = (
        d[
            "baseline_positive_fraction_all_raw"
        ].mean()
    )

    after_raw = (
        d[
            "after_sqi_positive_fraction_all_raw"
        ].mean()
    )

    print(
        f"  selective positive rate "
        f"before={before*100:.3f}% "
        f"after={after*100:.3f}%"
    )

    print(
        f"  all-raw positive fraction "
        f"before={before_raw*100:.3f}% "
        f"after={after_raw*100:.3f}%"
    )


# ============================================================
# SAVE PIPELINE SUMMARY
# ============================================================

pipeline_df = pd.DataFrame(
    pipeline_rows
)

pipeline_df.to_csv(
    OUT_PIPELINE,
    index=False,
)


print()
print("=" * 110)
print("FROZEN TEST COMPLETE")
print("=" * 110)

print(
    "SQI v1 was developed on subjects 138-153."
)

print(
    "This evaluation used only subjects 154-167."
)

print(
    "No SQI fitting, threshold selection, "
    "or AF-model tuning was performed here."
)

print(
    "Do NOT modify SQI v1 after viewing these results."
)

print()
print("SAVED:")
print(OUT_SQI_FEATURES)
print(OUT_PRED)
print(OUT_QUALITY)
print(OUT_SUBJECT)
print(OUT_PIPELINE)
