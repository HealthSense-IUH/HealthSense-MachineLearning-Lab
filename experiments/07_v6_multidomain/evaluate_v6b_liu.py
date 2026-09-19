import joblib
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
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


models={
    m:joblib.load(
        f"models/multidomain/healthsense_af_{m}.pkl"
    )
    for m in MODELS
}


df=pd.read_csv(
"experiments/06_external_validation/"
"liu_pseudo30_frozen_v5_predictions.csv"
)


df=df[df.feature_ok==True]


df["label"]=df.binary_label.astype(int)


print("="*100)
print("LIU DISTRIBUTION")
print("="*100)

print(df.rhythm.value_counts())


for name,model in models.items():

    p=model.predict_proba(
        df[FEATURES]
    )[:,1]

    print()
    print("="*100)
    print(name)

    print(
        "AUROC:",
        roc_auc_score(
            df.label,
            p
        )
    )

    print(
        "PR-AUC:",
        average_precision_score(
            df.label,
            p
        )
    )


    df["pred"]=(p>=0.5).astype(int)


    print()
    print(
        df.groupby("rhythm")
        .apply(
            lambda x:
            pd.Series(
            {
            "n":len(x),
            "mean_prob":x.iloc[:, -2].mean(),
            "positive_rate":x.pred.mean()
            }
            )
        )
    )

