from pathlib import Path
import hashlib

import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


DATA = Path(
    "data/features/v7/"
    "healthsense_af_v6c_pacaware_train.csv"
)


OUT = Path(
    "models/multidomain/"
    "healthsense_af_v6c_pacaware_rf.pkl"
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


df = pd.read_csv(
    DATA,
    low_memory=False
)


print("="*100)
print("HEALTHSENSE AF V6-C PAC AWARE TRAINING")
print("="*100)


print(
df.groupby(
    [
        "domain",
        "class_name"
    ]
).size()
)


# =====================================================
# Weight design
# =====================================================

df["weight_group"] = df["class_name"]


# hard negative emphasis
weight_map = {

    # positive AF
    "AF": 1.5,

    # normal rhythm
    "NSR": 1.0,

    # hard negatives
    "PAC_PVC": 3.0,

    "OTHER": 1.5,

    "NON_AF": 1.0
}


df["sample_weight"] = (
    df.weight_group
    .map(weight_map)
    .fillna(1.0)
)


# normalize

df["sample_weight"] /= (
    df.sample_weight.mean()
)


print()
print("WEIGHT AUDIT")

print(
df.groupby(
    [
        "class_name"
    ]
)
.sample_weight.mean()
)


# =====================================================
# Model
# =====================================================

model = Pipeline(
[
(
"rf",
RandomForestClassifier(
    n_estimators=500,
    max_depth=14,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
)
]
)


print()
print("Training...")


model.fit(
    df[FEATURES],
    df.label_v7.astype(int),
    rf__sample_weight=df.sample_weight
)


OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


joblib.dump(
    model,
    OUT
)


sha = hashlib.sha256(
    OUT.read_bytes()
).hexdigest()


print()
print("saved:", OUT)
print("sha256:", sha)

