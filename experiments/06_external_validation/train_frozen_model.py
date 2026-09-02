import sys
sys.path.insert(0, "src")

import json
import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    GroupKFold,
    GridSearchCV,
)

from healthsense_ml.training import (
    build_model_registry,
)


DATA = (
    "experiments/04_feature_model/"
    "af_features_cosen_ac.csv"
)

OUT_MODEL = (
    "models/mimic/"
    "healthsense_af_v5_rf_ac_frozen.pkl"
)

OUT_META = (
    "models/mimic/"
    "healthsense_af_v5_rf_ac_frozen.json"
)


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


df = pd.read_csv(DATA)

X = df[FEATURES]
y = df["status"].to_numpy()
groups = df["record_id"].to_numpy()


rf_spec = (
    build_model_registry()[
        "Random Forest"
    ]
)


search = GridSearchCV(
    clone(
        rf_spec["pipeline"]
    ),
    rf_spec["param_grid"],
    cv=GroupKFold(
        n_splits=3
    ),
    scoring="roc_auc",
    n_jobs=-1,
    refit=True,
)

search.fit(
    X,
    y,
    groups=groups,
)


print(
    "Best params:",
    search.best_params_,
)

print(
    "Best grouped-CV AUC:",
    search.best_score_,
)


import os

os.makedirs(
    "models/mimic",
    exist_ok=True,
)


joblib.dump(
    search.best_estimator_,
    OUT_MODEL,
)


metadata = {
    "version":
        "healthsense-af-v5",

    "development_dataset":
        "MIMIC PERform AF",

    "subjects":
        int(
            df.record_id.nunique()
        ),

    "features":
        FEATURES,

    "n_features":
        len(FEATURES),

    "window_s":
        30,

    "step_s":
        10,

    "iqr_filter":
        False,

    "calibration":
        None,

    "alert_threshold":
        0.90,

    "alert_policy":
        "3-of-3",

    "best_params":
        search.best_params_,

    "internal_nested_subject_sensitivity":
        1.0,

    "internal_nested_subject_specificity":
        0.75,

    "note":
        (
            "Internal metrics are development estimates. "
            "External validation required before deployment."
        ),
}


with open(
    OUT_META,
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


print(
    "Saved:",
    OUT_MODEL,
)

print(
    "Saved:",
    OUT_META,
)
