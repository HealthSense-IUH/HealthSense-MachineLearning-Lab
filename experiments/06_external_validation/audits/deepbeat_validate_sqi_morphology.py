import sys
sys.path.insert(0, "src")

from pathlib import Path
from zipfile import ZipFile
import numpy.lib.format as fmt

import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

from healthsense_ml.signal_processing import (
    bandpass_filter,
)


# ============================================================
# CONFIG
# ============================================================

NPZ = Path(
    "data/raw/validate.npz"
)

MANIFEST = Path(
    "experiments/06_external_validation/"
    "deepbeat_validate_sqi_sample_manifest.csv"
)

OUT = Path(
    "experiments/06_external_validation"
)

OUT_FEATURES = (
    OUT /
    "deepbeat_validate_sqi_morphology_features.csv"
)

OUT_UNIVARIATE = (
    OUT /
    "deepbeat_validate_sqi_univariate.csv"
)

OUT_LOSO = (
    OUT /
    "deepbeat_validate_sqi_loso_summary.csv"
)

OUT_OOF = (
    OUT /
    "deepbeat_validate_sqi_loso_predictions.csv"
)

FS = 32.0
EPS = 1e-12


# ============================================================
# FEATURE EXTRACTION
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

    # ----------------------------------------
    # Raw/basic waveform statistics
    # ----------------------------------------

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

    # Scale-normalized roughness
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

    # ----------------------------------------
    # Spectrum
    # ----------------------------------------

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

    # ----------------------------------------
    # V5-compatible autocorrelation +
    # periodicity
    # ----------------------------------------

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

        # Same PPG_AC definition used
        # by frozen HealthSense v5
        ppg_ac = float(
            np.mean(
                np.abs(
                    ac[1:]
                )
            )
        )

        # Physiological pulse lag:
        # 0.30–2.0 s ~= 30–200 bpm
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
# LOAD LOCKED SAMPLE MANIFEST
# ============================================================

manifest = pd.read_csv(
    MANIFEST
)

manifest[
    "global_index"
] = manifest[
    "global_index"
].astype(int)

manifest = (
    manifest
    .sort_values(
        "global_index"
    )
    .reset_index(
        drop=True
    )
)

print("=" * 105)
print("DEEPBEAT VALIDATION SQI MORPHOLOGY")
print("=" * 105)

print(
    "Selected windows:",
    len(manifest)
)

print(
    "Subjects:",
    manifest.subject_id.nunique()
)

print()
print(
    pd.crosstab(
        manifest.rhythm,
        manifest.quality_name,
    ).to_string()
)


# ============================================================
# STREAM SELECTED SIGNAL ROWS DIRECTLY
# FROM signal.npy INSIDE validate.npz
# ============================================================

rows = []

with ZipFile(
    NPZ
) as z:

    info = z.getinfo(
        "signal.npy"
    )

    print()
    print(
        "ZIP compression type:",
        info.compress_type,
        "(0 means stored/uncompressed)"
    )

    with z.open(
        info
    ) as f:

        version = fmt.read_magic(
            f
        )

        if version == (1, 0):

            (
                shape,
                fortran,
                dtype,
            ) = (
                fmt.read_array_header_1_0(
                    f
                )
            )

        else:

            (
                shape,
                fortran,
                dtype,
            ) = (
                fmt.read_array_header_2_0(
                    f
                )
            )

        print(
            "signal shape:",
            shape
        )

        print(
            "dtype:",
            dtype
        )

        print(
            "fortran:",
            fortran
        )

        if fortran:
            raise RuntimeError(
                "Unexpected Fortran-order signal array"
            )

        data_offset = f.tell()

        row_elements = int(
            np.prod(
                shape[1:]
            )
        )

        row_bytes = (
            row_elements
            *
            dtype.itemsize
        )

        for j, r in enumerate(
            manifest.itertuples(
                index=False
            ),
            1,
        ):

            idx = int(
                r.global_index
            )

            offset = (
                data_offset
                +
                idx
                *
                row_bytes
            )

            f.seek(
                offset
            )

            raw = f.read(
                row_bytes
            )

            if len(raw) != row_bytes:

                raise RuntimeError(
                    f"Short read at index {idx}"
                )

            x = np.frombuffer(
                raw,
                dtype=dtype,
                count=row_elements,
            ).astype(
                float,
                copy=True,
            )

            x = x.reshape(
                shape[1:]
            ).squeeze()

            feats = (
                morphology_features(
                    x
                )
            )

            row = {
                "global_index":
                    idx,

                "subject_id":
                    int(
                        r.subject_id
                    ),

                "source":
                    r.source,

                "timestamp":
                    r.timestamp,

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
                        ) > 0
                    ),

                **feats,
            }

            rows.append(
                row
            )

            if (
                j == 1
                or
                j % 500 == 0
                or
                j == len(manifest)
            ):

                print(
                    f"[{j:05d}/"
                    f"{len(manifest):05d}]"
                )


df = pd.DataFrame(
    rows
)

df.to_csv(
    OUT_FEATURES,
    index=False,
)


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [
    "raw_std",
    "amplitude_range",
    "derivative_noise",
    "second_derivative_noise",
    "flat_fraction",
    "clipping_low_fraction",
    "clipping_high_fraction",
    "clipping_fraction",
    "skewness",
    "kurtosis",
    "physio_band_ratio",
    "hf_noise_ratio",
    "spectral_entropy",
    "dominant_power_fraction",
    "PPG_AC",
    "ac_peak",
    "ac_abs_peak",
]


# ============================================================
# UNIVARIATE QUALITY AUC
#
# Target:
# quality 0 = poor
# quality 1/2 = usable/good
#
# Evaluate separately inside AF/non-AF
# to detect rhythm confounding.
# ============================================================

univariate_rows = []

print()
print("=" * 105)
print("UNIVARIATE QUALITY SEPARATION")
print("=" * 105)

for subset_name, d in [
    (
        "ALL",
        df,
    ),
    (
        "AF_ONLY",
        df[
            df.rhythm
            == "AF"
        ],
    ),
    (
        "NON_AF_ONLY",
        df[
            df.rhythm
            == "NON_AF"
        ],
    ),
]:

    print(
        f"\n{subset_name}"
    )

    local = []

    for feature in FEATURES:

        x = d[
            feature
        ].replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        valid = (
            x.notna()
        )

        if (
            valid.sum() == 0
            or
            d.loc[
                valid,
                "quality_good",
            ].nunique()
            < 2
        ):
            continue

        auc = roc_auc_score(
            d.loc[
                valid,
                "quality_good",
            ],
            x[
                valid
            ],
        )

        if auc >= 0.5:

            directionless = auc

            direction = (
                "higher=better"
            )

        else:

            directionless = (
                1.0 - auc
            )

            direction = (
                "lower=better"
            )

        row = {
            "subset":
                subset_name,

            "feature":
                feature,

            "n":
                int(
                    valid.sum()
                ),

            "auc_raw":
                float(auc),

            "auc_directionless":
                float(
                    directionless
                ),

            "direction":
                direction,

            "median_poor":
                float(
                    d.loc[
                        valid
                        &
                        (
                            d.quality
                            == 0
                        ),
                        feature,
                    ].median()
                ),

            "median_acceptable":
                float(
                    d.loc[
                        valid
                        &
                        (
                            d.quality
                            == 1
                        ),
                        feature,
                    ].median()
                ),

            "median_excellent":
                float(
                    d.loc[
                        valid
                        &
                        (
                            d.quality
                            == 2
                        ),
                        feature,
                    ].median()
                ),
        }

        local.append(
            row
        )

        univariate_rows.append(
            row
        )

    local_df = pd.DataFrame(
        local
    ).sort_values(
        "auc_directionless",
        ascending=False,
    )

    print(
        local_df[
            [
                "feature",
                "auc_directionless",
                "direction",
                "median_poor",
                "median_acceptable",
                "median_excellent",
            ]
        ]
        .head(12)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )


univariate = pd.DataFrame(
    univariate_rows
)

univariate.to_csv(
    OUT_UNIVARIATE,
    index=False,
)


# ============================================================
# LOSO QUALITY MODELS
#
# Only validation subjects are used.
# No DeepBeat test rows are touched.
# ============================================================

QUALITY_FEATURE_SETS = {

    "PPG_AC_ONLY": [
        "PPG_AC",
    ],

    "PERIODICITY": [
        "PPG_AC",
        "ac_peak",
        "ac_abs_peak",
        "spectral_entropy",
        "dominant_power_fraction",
    ],

    "MORPHOLOGY": [
        "derivative_noise",
        "second_derivative_noise",
        "flat_fraction",
        "clipping_fraction",
        "skewness",
        "kurtosis",
        "physio_band_ratio",
        "hf_noise_ratio",
        "spectral_entropy",
        "dominant_power_fraction",
        "PPG_AC",
        "ac_peak",
        "ac_abs_peak",
    ],
}


logo = LeaveOneGroupOut()

y = (
    df[
        "quality_good"
    ]
    .astype(int)
    .to_numpy()
)

groups = (
    df[
        "subject_id"
    ]
    .astype(int)
    .to_numpy()
)

oof_frames = []
summary_rows = []


print()
print("=" * 105)
print("LEAVE-ONE-SUBJECT-OUT SQI")
print("=" * 105)


for set_name, columns in (
    QUALITY_FEATURE_SETS.items()
):

    X = df[
        columns
    ].copy()

    oof = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        logo.split(
            X,
            y,
            groups,
        ),
        1,
    ):

        held_subject = int(
            groups[
                test_idx
            ][0]
        )

        pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=20260916,
                ),
            ),
        ])

        pipe.fit(
            X.iloc[
                train_idx
            ],
            y[
                train_idx
            ],
        )

        oof[
            test_idx
        ] = (
            pipe.predict_proba(
                X.iloc[
                    test_idx
                ]
            )[:, 1]
        )

        print(
            f"{set_name:15s} "
            f"fold={fold:02d} "
            f"subject={held_subject}"
        )

    # ----------------------------------------
    # Overall + rhythm strata
    # ----------------------------------------

    print(
        f"\n{set_name}"
    )

    for subset_name, mask in [
        (
            "ALL",
            np.ones(
                len(df),
                dtype=bool,
            ),
        ),
        (
            "AF_ONLY",
            (
                df.rhythm
                == "AF"
            ).to_numpy(),
        ),
        (
            "NON_AF_ONLY",
            (
                df.rhythm
                == "NON_AF"
            ).to_numpy(),
        ),
    ]:

        auc = roc_auc_score(
            y[mask],
            oof[mask],
        )

        summary_rows.append({
            "feature_set":
                set_name,

            "subset":
                subset_name,

            "n":
                int(
                    mask.sum()
                ),

            "loso_auc":
                float(auc),
        })

        print(
            f"  {subset_name:12s} "
            f"AUC={auc:.6f}"
        )

    # ----------------------------------------
    # Per-subject AUC
    # ----------------------------------------

    subject_aucs = []

    for uid in sorted(
        np.unique(groups)
    ):

        mask = (
            groups == uid
        )

        if (
            len(
                np.unique(
                    y[mask]
                )
            )
            < 2
        ):
            continue

        auc = roc_auc_score(
            y[mask],
            oof[mask],
        )

        subject_aucs.append(
            auc
        )

    print(
        "  per-subject AUC:"
        f" median="
        f"{np.median(subject_aucs):.6f}"
        f" min="
        f"{np.min(subject_aucs):.6f}"
        f" max="
        f"{np.max(subject_aucs):.6f}"
    )

    summary_rows.append({
        "feature_set":
            set_name,

        "subset":
            "SUBJECT_AUC_MEDIAN",

        "n":
            len(
                subject_aucs
            ),

        "loso_auc":
            float(
                np.median(
                    subject_aucs
                )
            ),
    })

    oof_frames.append(
        pd.DataFrame({
            "global_index":
                df.global_index,

            "subject_id":
                df.subject_id,

            "rhythm":
                df.rhythm,

            "quality":
                df.quality,

            "quality_name":
                df.quality_name,

            "quality_good":
                y,

            "feature_set":
                set_name,

            "oof_quality_probability":
                oof,
        })
    )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT_LOSO,
    index=False,
)

oof_df = pd.concat(
    oof_frames,
    ignore_index=True,
)

oof_df.to_csv(
    OUT_OOF,
    index=False,
)


# ============================================================
# AF / NON-AF DIRECTION CONSISTENCY
# ============================================================

print()
print("=" * 105)
print("UNIVARIATE RHYTHM CONSISTENCY")
print("=" * 105)

af = (
    univariate[
        univariate.subset
        == "AF_ONLY"
    ]
    [
        [
            "feature",
            "auc_directionless",
            "direction",
        ]
    ]
    .rename(
        columns={
            "auc_directionless":
                "auc_AF",

            "direction":
                "direction_AF",
        }
    )
)

non = (
    univariate[
        univariate.subset
        == "NON_AF_ONLY"
    ]
    [
        [
            "feature",
            "auc_directionless",
            "direction",
        ]
    ]
    .rename(
        columns={
            "auc_directionless":
                "auc_NON_AF",

            "direction":
                "direction_NON_AF",
        }
    )
)

consistency = af.merge(
    non,
    on="feature",
)

consistency[
    "same_direction"
] = (
    consistency.direction_AF
    ==
    consistency.direction_NON_AF
)

consistency[
    "worst_stratum_auc"
] = consistency[
    [
        "auc_AF",
        "auc_NON_AF",
    ]
].min(
    axis=1
)

consistency = consistency.sort_values(
    [
        "same_direction",
        "worst_stratum_auc",
    ],
    ascending=[
        False,
        False,
    ],
)

print(
    consistency.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.5f}",
    )
)


print()
print("=" * 105)
print("IMPORTANT")
print("=" * 105)

print(
    "No SQI threshold has been selected."
)

print(
    "DeepBeat TEST has not been used "
    "to fit or select the quality model."
)

print(
    "Next step: choose the smallest robust "
    "SQI model using validation LOSO only, "
    "freeze it, then evaluate once on "
    "test-only subjects 154-167."
)

print()
print("SAVED:")
print(OUT_FEATURES)
print(OUT_UNIVARIATE)
print(OUT_LOSO)
print(OUT_OOF)
