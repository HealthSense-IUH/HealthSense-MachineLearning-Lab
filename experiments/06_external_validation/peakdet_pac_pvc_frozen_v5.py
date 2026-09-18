import sys
sys.path.insert(0, "src")

from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import scipy.io as sio

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

ZIP_PATH = Path(
    "data/raw/PPG_PeakDet_MIMICIII.zip"
)

MANIFEST = ROOT / (
    "peakdet_subject_manifest_no_mimic_overlap.csv"
)

MODEL_PATH = Path(
    "models/mimic/"
    "healthsense_af_v5_rf_ac_frozen.pkl"
)

OUT_PRED = ROOT / (
    "peakdet_pac_pvc_frozen_v5_predictions.csv"
)

OUT_SUBJ = ROOT / (
    "peakdet_pac_pvc_frozen_v5_subject_summary.csv"
)

OUT_SUM = ROOT / (
    "peakdet_pac_pvc_frozen_v5_summary.csv"
)

# PeakDet:
# 1500 PPG samples / 30 s = 50 Hz
FS = 50.0

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


# ============================================================
# AC FEATURE — SAME DEFINITION AS V5
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
            np.abs(ac[1:])
        )
    )


# ============================================================
# LOAD MANIFEST
# ============================================================

manifest = pd.read_csv(
    MANIFEST
)

# Guard against accidental leakage
manifest = manifest[
    manifest["class_name"] == "PAC_PVC"
].copy()

print("=" * 100)
print("PEAKDET PAC/PVC — FROZEN HEALTHSENSE V5")
print("=" * 100)

print(
    "\nSegments in independent manifest:",
    len(manifest)
)

print(
    "Subjects:",
    manifest["subject_id"].nunique()
)

print(
    "\nSubjects:",
    sorted(
        manifest["subject_id"]
        .dropna()
        .unique()
    )
)

if len(manifest) != 50:
    print(
        "\nWARNING:"
        f" expected 50 PAC/PVC segments,"
        f" got {len(manifest)}"
    )

# ============================================================
# LOAD FROZEN MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)

print(
    "\nFrozen model:",
    MODEL_PATH
)

# ============================================================
# INDEX ZIP
# ============================================================

with ZipFile(ZIP_PATH) as z:

    mat_members = {
        Path(name).name: name
        for name in z.namelist()
        if name.lower().endswith(".mat")
    }

    rows = []

    for i, r in enumerate(
        manifest.itertuples(),
        1,
    ):

        filename = r.file

        print(
            f"[{i:02d}/{len(manifest)}] "
            f"{r.subject_id} "
            f"{filename}"
        )

        if filename not in mat_members:

            rows.append({
                "file": filename,
                "subject_id":
                    r.subject_id,
                "class_name":
                    "PAC_PVC",
                "feature_ok":
                    False,
                "error":
                    "MAT_NOT_FOUND",
            })

            continue

        raw = z.read(
            mat_members[filename]
        )

        try:

            m = sio.loadmat(
                BytesIO(raw),
                squeeze_me=True,
                struct_as_record=False,
            )

            ppg = np.asarray(
                m["PPG_raw_buffer"],
                dtype=float,
            ).reshape(-1)

            if len(ppg) != 1500:
                raise ValueError(
                    f"Unexpected PPG length "
                    f"{len(ppg)}"
                )

            # --------------------------------------------
            # PPG -> NN
            # --------------------------------------------

            nn_ms, nn_times = (
                extract_nn_series(
                    ppg,
                    fs=FS,
                )
            )

            n_nn = len(nn_ms)

            # ECG metadata for later audit
            r_locs = np.asarray(
                m.get(
                    "R_locs",
                    []
                )
            ).reshape(-1)

            wbwref_hr = np.asarray(
                m.get(
                    "wbwrefHR",
                    []
                ),
                dtype=float,
            ).reshape(-1)

            base = {
                "file":
                    filename,

                "subject_id":
                    r.subject_id,

                "subject_info":
                    r.subject_info,

                "class_name":
                    "PAC_PVC",

                "n_samples_ppg":
                    len(ppg),

                "n_nn":
                    n_nn,

                "n_ecg_rpeaks":
                    len(r_locs),

                "ecg_hr_median":
                    (
                        float(
                            np.nanmedian(
                                wbwref_hr
                            )
                        )
                        if len(wbwref_hr)
                        else np.nan
                    ),
            }

            if n_nn < MIN_NN:

                rows.append({
                    **base,
                    "feature_ok":
                        False,
                    "error":
                        "TOO_FEW_NN",
                })

                continue

            # --------------------------------------------
            # HRV features
            # --------------------------------------------

            feats = compute_hrv_features(
                nn_ms,
                nn_times,
            )

            feats["PPG_AC"] = (
                autocorrelation_feature(
                    ppg,
                    FS,
                )
            )

            X = pd.DataFrame(
                [{
                    f: feats[f]
                    for f in FEATURES
                }]
            )

            if (
                X[FEATURES]
                .isna()
                .any(axis=None)
            ):
                rows.append({
                    **base,
                    "feature_ok":
                        False,
                    "error":
                        "NAN_FEATURE",
                    **feats,
                })

                continue

            # --------------------------------------------
            # frozen inference
            # --------------------------------------------

            p_af = float(
                model.predict_proba(
                    X[FEATURES]
                )[0, 1]
            )

            hr = float(
                feats["HR_mean"]
            )

            # frozen candidate rules
            fixed_090 = (
                p_af >= 0.90
            )

            fixed_085 = (
                p_af >= 0.85
            )

            adaptive_threshold = (
                0.80
                if hr < 60.0
                else 0.90
            )

            adaptive = (
                p_af
                >= adaptive_threshold
            )

            rows.append({
                **base,

                "feature_ok":
                    True,

                "error":
                    "",

                **feats,

                "af_probability":
                    p_af,

                "fixed_090_positive":
                    fixed_090,

                "fixed_085_positive":
                    fixed_085,

                "adaptive_threshold":
                    adaptive_threshold,

                "hr_lt60":
                    hr < 60.0,

                "adaptive_positive":
                    adaptive,
            })

        except Exception as e:

            rows.append({
                "file":
                    filename,

                "subject_id":
                    r.subject_id,

                "subject_info":
                    r.subject_info,

                "class_name":
                    "PAC_PVC",

                "feature_ok":
                    False,

                "error":
                    repr(e),
            })


pred = pd.DataFrame(rows)

pred.to_csv(
    OUT_PRED,
    index=False,
)

# ============================================================
# VALID ROWS
# ============================================================

ok = pred[
    pred["feature_ok"] == True
].copy()

print()
print("=" * 100)
print("COVERAGE")
print("=" * 100)

print(
    "Total        :",
    len(pred)
)

print(
    "Feature OK   :",
    len(ok)
)

print(
    "No decision  :",
    len(pred) - len(ok)
)

print(
    "Coverage     :",
    (
        len(ok) / len(pred)
        if len(pred)
        else np.nan
    )
)

if len(ok) == 0:
    raise RuntimeError(
        "No usable PAC/PVC segments"
    )

# ============================================================
# SEGMENT LEVEL
# ============================================================

print()
print("=" * 100)
print("SEGMENT-LEVEL FALSE POSITIVE CHALLENGE")
print("=" * 100)

rules = {
    "fixed_090":
        "fixed_090_positive",

    "fixed_085":
        "fixed_085_positive",

    "hr_lt60_080_else090":
        "adaptive_positive",
}

summary_rows = []

for rule, col in rules.items():

    n_pos = int(
        ok[col].sum()
    )

    rate = (
        n_pos / len(ok)
    )

    summary_rows.append({
        "rule":
            rule,

        "level":
            "segment",

        "n":
            len(ok),

        "false_positive":
            n_pos,

        "false_positive_rate":
            rate,
    })

    print(
        f"{rule:24s} "
        f"{n_pos:3d}/{len(ok):3d} "
        f"= {rate * 100:7.3f}%"
    )

print()
print(
    "AF probability:"
)

print(
    ok["af_probability"]
    .describe(
        percentiles=[
            .10,
            .25,
            .50,
            .75,
            .90,
            .95,
            .99,
        ]
    )
)

print()
print(
    "HR_mean:"
)

print(
    ok["HR_mean"]
    .describe(
        percentiles=[
            .10,
            .25,
            .50,
            .75,
            .90,
        ]
    )
)

print()
print(
    "Low-HR segments:",
    int(
        ok["hr_lt60"].sum()
    ),
    "/",
    len(ok)
)

# ============================================================
# SUBJECT MACRO
# ============================================================

subject_rows = []

for uid, g in ok.groupby(
    "subject_id"
):

    row = {
        "subject_id":
            uid,

        "n_segments":
            len(g),

        "hr_median":
            g["HR_mean"].median(),

        "p_af_median":
            g[
                "af_probability"
            ].median(),

        "p_af_max":
            g[
                "af_probability"
            ].max(),

        "low_hr_fraction":
            g[
                "hr_lt60"
            ].mean(),
    }

    for rule, col in rules.items():

        row[
            f"{rule}_positive_n"
        ] = int(
            g[col].sum()
        )

        row[
            f"{rule}_positive_rate"
        ] = float(
            g[col].mean()
        )

    subject_rows.append(
        row
    )

subj = pd.DataFrame(
    subject_rows
)

subj.to_csv(
    OUT_SUBJ,
    index=False,
)

print()
print("=" * 100)
print("SUBJECT LEVEL")
print("=" * 100)

print(
    subj.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)

print()
print("=" * 100)
print("SUBJECT-MACRO POSITIVE RATE")
print("=" * 100)

for rule, _ in rules.items():

    col = (
        f"{rule}_positive_rate"
    )

    macro = (
        subj[col].mean()
    )

    summary_rows.append({
        "rule":
            rule,

        "level":
            "subject_macro",

        "n":
            len(subj),

        "false_positive":
            np.nan,

        "false_positive_rate":
            macro,
    })

    print(
        f"{rule:24s} "
        f"{macro * 100:7.3f}%"
    )

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT_SUM,
    index=False,
)

# ============================================================
# IMPORTANT NOTE
# ============================================================

print()
print("=" * 100)
print("INTERPRETATION NOTE")
print("=" * 100)

print(
    "These are SEGMENT positive rates, "
    "NOT 3-of-3 alert rates."
)

print(
    "PeakDet files do not provide enough "
    "validated temporal continuity to "
    "reconstruct the production alert rule."
)

print()
print("SAVED:")
print(OUT_PRED)
print(OUT_SUBJ)
print(OUT_SUM)
