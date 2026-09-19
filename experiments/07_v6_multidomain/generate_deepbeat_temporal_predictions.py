from pathlib import Path

import joblib
import pandas as pd


MODEL = (
    "models/multidomain/"
    "healthsense_af_v6a_rf_ac_frozen.pkl"
)

INPUT = (
    "data/features/v6/"
    "deepbeat_test_independent_v6_14f.csv"
)

OUTPUT = (
    "experiments/07_v6_multidomain/"
    "artifacts/"
    "deepbeat_test_v6a_temporal_predictions.csv"
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


model = joblib.load(
    MODEL
)


df = pd.read_csv(
    INPUT
)


df = df[
    df.feature_ok == True
].copy()


df["probability"] = (
    model.predict_proba(
        df[FEATURES]
    )[:,1]
)


df = (
    df
    .sort_values(
        [
            "subject_id",
            "timestamp",
            "global_index",
        ]
    )
    .reset_index(drop=True)
)


df.to_csv(
    OUTPUT,
    index=False,
)


print("="*100)
print("TEMPORAL PREDICTION")
print("="*100)

print(
    "windows:",
    len(df)
)

print(
    "subjects:",
    df.subject_id.nunique()
)

print(
    "saved:",
    OUTPUT
)

print()

print(
    df[
        [
            "subject_id",
            "timestamp",
            "label",
            "probability",
        ]
    ]
    .head(20)
)
