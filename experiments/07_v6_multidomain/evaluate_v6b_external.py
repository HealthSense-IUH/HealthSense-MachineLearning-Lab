from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


FEATURES=[
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


MODELS=[
    "v6b0_none",
    "v6b1_balanced",
]


def evaluate(name, model, df, label):

    x=df[FEATURES]

    y=df[label]

    p=model.predict_proba(x)[:,1]

    print()
    print("-"*80)
    print(name)

    print("windows:",len(df))

    print(
        "AUROC:",
        roc_auc_score(y,p)
    )

    print(
        "PR-AUC:",
        average_precision_score(y,p)
    )

    print(
        "Brier:",
        brier_score_loss(y,p)
    )

    print(
        confusion_matrix(
            y,
            p>=0.5
        )
    )


models={}

for m in MODELS:

    models[m]=joblib.load(
        "models/multidomain/"
        f"healthsense_af_{m}.pkl"
    )


# DeepBeat independent

deep=pd.read_csv(
"data/features/v6/"
"deepbeat_test_independent_v6_14f.csv"
)

deep["label"]=deep["label"].astype(int)


print("="*100)
print("DEEPBEAT TEST")
print("="*100)


for n,m in models.items():

    evaluate(
        n,
        m,
        deep,
        "label"
    )


