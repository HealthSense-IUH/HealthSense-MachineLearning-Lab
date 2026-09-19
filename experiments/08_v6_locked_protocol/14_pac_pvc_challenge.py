from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(
    "/home/phuc/Documents/HealthSense_rungtamnhi"
)

ART = (
    ROOT /
    "experiments/08_v6_locked_protocol/artifacts"
)

MODEL = (
    ROOT /
    "models/multidomain/"
    "healthsense_af_v6c_stacking.pkl"
)


bundle = joblib.load(
    MODEL
)

FEATURES = list(
    bundle["features"]
)

BASE = bundle[
    "base_models"
]

META = bundle[
    "meta"
]


def predict(df):

    valid = (
        df[FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .notna()
        .all(axis=1)
    )


    x = df.loc[
        valid
    ].copy()


    X = x[
        FEATURES
    ]


    names = list(
        BASE.keys()
    )


    base_prob = np.column_stack([
        BASE[name]
        .predict_proba(X)[:, 1]

        for name
        in names
    ])


    x["ensemble_mean"] = (
        base_prob.mean(
            axis=1
        )
    )

    x["ensemble_std"] = (
        base_prob.std(
            axis=1
        )
    )


    x["meta_probability"] = (
        META.predict_proba(
            base_prob
        )[:, 1]
    )


    return x


def summarize(
    name,
    df
):

    p = (
        df[
            "meta_probability"
        ]
        .to_numpy()
    )


    row = {
        "challenge":
            name,

        "windows":
            len(df),

        "mean_probability":
            float(
                np.mean(p)
            ),

        "median_probability":
            float(
                np.median(p)
            ),

        "p90_probability":
            float(
                np.quantile(
                    p,
                    .90
                )
            ),

        "p95_probability":
            float(
                np.quantile(
                    p,
                    .95
                )
            ),

        "max_probability":
            float(
                np.max(p)
            ),

        "mean_ensemble_std":
            float(
                df[
                    "ensemble_std"
                ].mean()
            ),
    }


    for threshold in [
        .50,
        .70,
        .80,
        .90,
        .95,
    ]:

        tag = str(
            threshold
        ).replace(
            ".",
            "_"
        )

        count = int(
            (
                p
                >=
                threshold
            ).sum()
        )


        row[
            f"FP_count_ge_{tag}"
        ] = count

        row[
            f"FPR_ge_{tag}"
        ] = (
            count
            /
            len(df)
        )


    return row


# ============================================================
# PulseWatch held-out PAC/PVC subset
# ============================================================

pulse_file = (
    ROOT /
    "experiments/07_v6_multidomain/"
    "artifacts/hard_domain_splits/"
    "pulsewatch_test.csv"
)


pulse = pd.read_csv(
    pulse_file,
    low_memory=False
)


pulse_pac = pulse[
    pulse[
        "class_name"
    ]
    .astype(str)
    .str.upper()
    .eq("PAC_PVC")
].copy()


pulse_pred = predict(
    pulse_pac
)


pulse_pred.to_csv(
    ART /
    "challenge_pulsewatch_pac_pvc_predictions.csv",
    index=False
)


# ============================================================
# Auto-discover PeakDet 14-feature file
# ============================================================

candidates = []


for f in ROOT.rglob(
    "*peakdet*.csv"
):

    try:

        head = pd.read_csv(
            f,
            nrows=5,
            low_memory=False
        )

    except Exception:

        continue


    if all(
        feature in head.columns
        for feature in FEATURES
    ):

        try:

            full = pd.read_csv(
                f,
                low_memory=False
            )

        except Exception:

            continue


        candidates.append(
            (
                f,
                full
            )
        )


print(
    "\nPeakDet candidates:"
)


for f, x in candidates:

    print(
        len(x),
        f
    )


if not candidates:

    raise RuntimeError(
        "No PeakDet CSV with all "
        "14 features found."
    )


# Prefer exactly 50-row file.
exact_50 = [
    item
    for item in candidates
    if len(item[1]) == 50
]


if exact_50:

    peak_file, peak = (
        exact_50[0]
    )

else:

    # Otherwise choose candidate
    # closest to 50 rows.
    peak_file, peak = min(
        candidates,
        key=lambda item:
            abs(
                len(item[1])
                -
                50
            )
    )


print(
    "\nSelected PeakDet:"
)

print(
    peak_file
)

print(
    "rows:",
    len(peak)
)


peak_pred = predict(
    peak
)


peak_pred.to_csv(
    ART /
    "challenge_peakdet_pac_pvc_predictions.csv",
    index=False
)


# ============================================================
# Summaries
# ============================================================

rows = [
    summarize(
        "PULSEWATCH_PAC_PVC_TEST",
        pulse_pred
    ),

    summarize(
        "PEAKDET_PAC_PVC",
        peak_pred
    ),
]


summary = pd.DataFrame(
    rows
)


summary.to_csv(
    ART /
    "pac_pvc_challenge_summary.csv",
    index=False
)


print(
    "\n" +
    "=" * 100
)

print(
    "PAC/PVC CHALLENGE"
)

print(
    summary.to_string(
        index=False
    )
)
