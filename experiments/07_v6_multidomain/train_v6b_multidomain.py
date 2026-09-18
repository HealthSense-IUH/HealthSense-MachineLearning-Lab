import sys
sys.path.insert(0, "src")

from pathlib import Path
import json
import hashlib

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


DATA = Path(
    "data/features/v7/"
    "healthsense_af_v6b_train.csv"
)


MODEL_DIR = Path(
    "models/multidomain"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
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


def sha256(path):

    h = hashlib.sha256()

    with open(path,"rb") as f:
        for chunk in iter(
            lambda:f.read(1024*1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()



df=pd.read_csv(DATA)


X=df[FEATURES]
y=df["label_v7"]


print("="*100)
print("HEALTHSENSE AF V6-B MULTIDOMAIN TRAINING")
print("="*100)


print()
print(df.groupby(
    [
        "domain",
        "label_v7"
    ]
).size())


configs = {

"v6b0_none":
None,

"v6b1_balanced":
"balanced",

}


results=[]


for name,weight in configs.items():

    print()
    print("-"*100)
    print(name)

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=42,
                    n_jobs=-1,
                    class_weight=weight
                )
            )
        ]
    )


    model.fit(
        X,
        y
    )


    pred=model.predict_proba(X)[:,1]


    auc=roc_auc_score(
        y,
        pred
    )

    pr=average_precision_score(
        y,
        pred
    )

    brier=brier_score_loss(
        y,
        pred
    )


    print(
        "Train AUROC:",
        auc
    )

    print(
        "Train PR-AUC:",
        pr
    )


    out=MODEL_DIR / (
        "healthsense_af_"
        +name
        +".pkl"
    )


    joblib.dump(
        model,
        out
    )


    meta={
        "model":name,
        "features":FEATURES,
        "train_rows":len(df),
        "sha256":sha256(out)
    }


    with open(
        out.with_suffix(".json"),
        "w"
    ) as f:

        json.dump(
            meta,
            f,
            indent=2
        )


    results.append(
        {
            "model":name,
            "auroc":auc,
            "pr_auc":pr,
            "brier":brier
        }
    )


print()
print("="*100)
print(
    pd.DataFrame(results)
)

