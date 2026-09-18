import json
import hashlib
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


DATA = Path(
    "data/features/v7/"
    "healthsense_af_v6b_clean_train.csv"
)

OUT = Path(
    "models/multidomain/"
    "healthsense_af_v6b_clean_rf.pkl"
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


print("="*100)
print("V6-B CLEAN TRAINING")
print("="*100)

print(
df.groupby(
    [
        "domain",
        "label_v7"
    ]
).size()
)


X=df[FEATURES]
y=df.label_v7.astype(int)


# ==================================================
# domain/class balanced weight
# ==================================================

group_count = (
    df.groupby(
        [
            "domain",
            "label_v7"
        ]
    )
    .size()
)


weights = {}

for idx,n in group_count.items():
    weights[idx]=1.0/(8*n)


sample_weight=np.array(
    [
        weights[
            (
                row.domain,
                row.label_v7
            )
        ]
        for _,row in df.iterrows()
    ]
)


sample_weight = (
    sample_weight /
    sample_weight.mean()
)


print()
print("WEIGHT AUDIT")

print(
pd.DataFrame(
    {
        "domain":df.domain,
        "label":df.label_v7,
        "weight":sample_weight
    }
)
.groupby(
    [
        "domain",
        "label"
    ]
)
.weight.mean()
)


model = Pipeline(
    [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
        )
    ]
)


print()
print("Training...")

model.fit(
    X,
    y,
    rf__sample_weight=sample_weight
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


meta={
    "model":"V6-B-clean",
    "features":FEATURES,
    "training_rows":len(df),
    "subjects":int(df.subject_id.nunique()),
    "sha256":sha,
}


with open(
    str(OUT).replace(
        ".pkl",
        ".json"
    ),
    "w"
) as f:
    json.dump(
        meta,
        f,
        indent=2
    )


print()
print("saved:",OUT)
print("sha256:",sha)

