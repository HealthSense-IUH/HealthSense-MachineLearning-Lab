from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ROOT = Path(
    "experiments/06_external_validation"
)

FEATURE_FILE = (
    ROOT /
    "deepbeat_validate_sqi_morphology_features.csv"
)

OOF_FILE = (
    ROOT /
    "deepbeat_validate_sqi_loso_predictions.csv"
)

MODEL_OUT = (
    ROOT /
    "deepbeat_sqi_v1_frozen.joblib"
)

META_OUT = (
    ROOT /
    "deepbeat_sqi_v1_frozen.json"
)

POLICY_OUT = (
    ROOT /
    "deepbeat_sqi_v1_policy_selection.csv"
)


SEED = 20260916

TARGET_AF_GOOD_RETENTION = 0.95

# With 20 observations, one sample corresponds to 5%.
# Subjects with fewer than 20 good-AF windows are
# reported but not used to estimate a 5% lower tail.
MIN_AF_GOOD_WINDOWS = 20


FEATURE_SETS = {

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


def sha256(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def subject_safe_threshold(
    d,
):

    af_good = d[
        (d.rhythm == "AF")
        &
        (d.quality_good == 1)
    ].copy()

    subject_rows = []

    for uid, g in af_good.groupby(
        "subject_id"
    ):

        n = len(g)

        if n < MIN_AF_GOOD_WINDOWS:

            subject_rows.append({
                "subject_id":
                    int(uid),

                "n_good":
                    n,

                "eligible":
                    False,

                "subject_threshold":
                    np.nan,
            })

            continue

        values = np.sort(
            g[
                "oof_quality_probability"
            ].to_numpy()
        )

        # Largest empirical threshold that retains
        # at least 95% of this subject's good windows.
        k = int(
            np.floor(
                (
                    1.0
                    -
                    TARGET_AF_GOOD_RETENTION
                )
                *
                n
            )
        )

        k = min(
            max(
                k,
                0,
            ),
            n - 1,
        )

        threshold = float(
            values[k]
        )

        subject_rows.append({
            "subject_id":
                int(uid),

            "n_good":
                n,

            "eligible":
                True,

            "subject_threshold":
                threshold,
        })

    subject_df = pd.DataFrame(
        subject_rows
    )

    eligible = subject_df[
        subject_df.eligible
    ]

    if len(eligible) == 0:

        raise RuntimeError(
            "No eligible AF subjects"
        )

    # Conservative threshold:
    # satisfy the worst eligible AF subject.
    threshold = float(
        eligible[
            "subject_threshold"
        ].min()
    )

    return (
        threshold,
        subject_df,
    )


features = pd.read_csv(
    FEATURE_FILE
)

oof = pd.read_csv(
    OOF_FILE
)


print("=" * 105)
print("FREEZE DEEPBEAT SQI V1")
print("=" * 105)

print(
    "Validation windows:",
    len(features)
)

print(
    "Validation subjects:",
    features.subject_id.nunique()
)


policy_rows = []

subject_constraint_tables = {}


for set_name, columns in (
    FEATURE_SETS.items()
):

    d = oof[
        oof.feature_set
        == set_name
    ].copy()

    threshold, subject_table = (
        subject_safe_threshold(
            d
        )
    )

    subject_constraint_tables[
        set_name
    ] = subject_table

    d[
        "accepted"
    ] = (
        d[
            "oof_quality_probability"
        ]
        >=
        threshold
    )

    af_good = d[
        (d.rhythm == "AF")
        &
        (d.quality_good == 1)
    ]

    af_poor = d[
        (d.rhythm == "AF")
        &
        (d.quality_good == 0)
    ]

    non_good = d[
        (d.rhythm == "NON_AF")
        &
        (d.quality_good == 1)
    ]

    non_poor = d[
        (d.rhythm == "NON_AF")
        &
        (d.quality_good == 0)
    ]

    all_good = d[
        d.quality_good == 1
    ]

    all_poor = d[
        d.quality_good == 0
    ]

    constrained_subjects = (
        subject_table[
            subject_table.eligible
        ]
        .subject_id
        .astype(int)
        .tolist()
    )

    constrained_rates = (
        af_good[
            af_good.subject_id.isin(
                constrained_subjects
            )
        ]
        .groupby(
            "subject_id"
        )
        .accepted
        .mean()
    )

    min_constrained_retention = float(
        constrained_rates.min()
    )

    row = {
        "feature_set":
            set_name,

        "n_features":
            len(columns),

        "quality_score_threshold":
            threshold,

        "min_constrained_AF_good_retention":
            min_constrained_retention,

        "AF_good_retention":
            float(
                af_good.accepted.mean()
            ),

        "NON_AF_good_retention":
            float(
                non_good.accepted.mean()
            ),

        "ALL_good_retention":
            float(
                all_good.accepted.mean()
            ),

        "AF_poor_rejection":
            float(
                1.0
                -
                af_poor.accepted.mean()
            ),

        "NON_AF_poor_rejection":
            float(
                1.0
                -
                non_poor.accepted.mean()
            ),

        "ALL_poor_rejection":
            float(
                1.0
                -
                all_poor.accepted.mean()
            ),

        "overall_acceptance":
            float(
                d.accepted.mean()
            ),
    }

    policy_rows.append(
        row
    )


policy = pd.DataFrame(
    policy_rows
)


# ============================================================
# PREDEFINED SELECTION RULE
#
# 1. >=95% retention in every adequately represented
#    AF subject.
# 2. >=95% good non-AF retention.
# 3. Among passing models, maximize poor rejection.
# 4. Fewer features only breaks a tie.
# ============================================================

eligible = policy[
    (
        policy[
            "min_constrained_AF_good_retention"
        ]
        >=
        TARGET_AF_GOOD_RETENTION
    )
    &
    (
        policy[
            "NON_AF_good_retention"
        ]
        >=
        0.95
    )
].copy()


if len(eligible) == 0:

    raise RuntimeError(
        "No SQI candidate satisfies safety constraints"
    )


eligible = eligible.sort_values(
    [
        "ALL_poor_rejection",
        "n_features",
    ],
    ascending=[
        False,
        True,
    ],
)


selected = (
    eligible.iloc[0]
)

selected_name = (
    selected[
        "feature_set"
    ]
)

selected_features = (
    FEATURE_SETS[
        selected_name
    ]
)

selected_threshold = float(
    selected[
        "quality_score_threshold"
    ]
)


policy[
    "selected"
] = (
    policy.feature_set
    ==
    selected_name
)

policy.to_csv(
    POLICY_OUT,
    index=False,
)


print()
print("=" * 105)
print("POLICY COMPARISON")
print("=" * 105)

print(
    policy.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


print()
print("=" * 105)
print("SELECTED")
print("=" * 105)

print(
    "Feature set:",
    selected_name
)

print(
    "Features:",
    selected_features
)

print(
    "OOF quality-score threshold:",
    f"{selected_threshold:.12f}"
)


print()
print("AF SUBJECT CONSTRAINTS")

print(
    subject_constraint_tables[
        selected_name
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# FINAL FIT ON ALL VALIDATION SAMPLE
# ============================================================

X = features[
    selected_features
]

y = (
    features[
        "quality_good"
    ]
    .astype(int)
)


pipeline = Pipeline([
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
            random_state=SEED,
        ),
    ),
])


pipeline.fit(
    X,
    y,
)


artifact = {
    "name":
        "HealthSense DeepBeat SQI v1",

    "pipeline":
        pipeline,

    "feature_set":
        selected_name,

    "features":
        selected_features,

    # Threshold selected entirely from validation LOSO.
    "quality_score_threshold":
        selected_threshold,

    "quality_definition":
        "quality 1/2 accepted target; quality 0 poor",

    "development_dataset":
        "DeepBeat validate.npz",

    "development_windows":
        int(
            len(features)
        ),

    "development_subjects":
        sorted(
            features.subject_id
            .astype(int)
            .unique()
            .tolist()
        ),

    "target_AF_good_retention":
        TARGET_AF_GOOD_RETENTION,

    "min_AF_good_windows_for_constraint":
        MIN_AF_GOOD_WINDOWS,

    "random_seed":
        SEED,
}


joblib.dump(
    artifact,
    MODEL_OUT,
)


metadata = {
    k: v
    for k, v in artifact.items()
    if k != "pipeline"
}

metadata[
    "validation_feature_sha256"
] = sha256(
    FEATURE_FILE
)

metadata[
    "validation_oof_sha256"
] = sha256(
    OOF_FILE
)

metadata[
    "policy_selection_sha256"
] = sha256(
    POLICY_OUT
)


with open(
    META_OUT,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


print()
print("=" * 105)
print("FROZEN")
print("=" * 105)

print(MODEL_OUT)
print(META_OUT)
print(POLICY_OUT)

print()
print(
    "IMPORTANT: Do not modify feature set, "
    "model or threshold after examining "
    "DeepBeat test-only subjects."
)
