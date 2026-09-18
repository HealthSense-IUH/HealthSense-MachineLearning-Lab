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


def evaluate(name, model, df, title):

    x=df[FEATURES]

    y=df["label"]

    p=model.predict_proba(x)[:,1]


    print()
    print("="*100)
    print(title)
    print(name)

    print(
        "windows:",
        len(df)
    )

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


df=pd.read_csv(
"experiments/06_external_validation/"
"pulsewatch_frozen_v5_features.csv"
)


df=df[
    df.feature_ok==True
]


print("="*100)
print("PULSEWATCH LABEL DISTRIBUTION")
print("="*100)

print(
df.class_name.value_counts()
)


# =====================================================
# 1. AF vs NSR
# =====================================================

binary=df[
    df.class_name.isin(
        [
            "AF",
            "NSR"
        ]
    )
].copy()


binary["label"]=(
    binary.class_name=="AF"
).astype(int)


print()
print("AF VS NSR")


for n,m in models.items():

    evaluate(
        n,
        m,
        binary,
        "AF VS NSR"
    )



# =====================================================
# 2. AF vs ALL NON AF
# =====================================================


all_binary=df.copy()

all_binary["label"]=(
    all_binary.class_name=="AF"
).astype(int)


print()
print("AF VS ALL NON AF")


for n,m in models.items():

    evaluate(
        n,
        m,
        all_binary,
        "AF VS ALL NON AF"
    )

