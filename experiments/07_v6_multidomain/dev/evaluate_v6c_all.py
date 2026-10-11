import joblib
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


MODEL = (
    "models/multidomain/"
    "healthsense_af_v6c_pacaware_rf.pkl"
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


model=joblib.load(MODEL)


def evaluate(name, df, y):

    X=df[FEATURES]

    p=model.predict_proba(X)[:,1]

    print()
    print("="*80)
    print(name)
    print("="*80)

    print("windows:",len(df))

    if len(set(y))==2:

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


    print()

    tmp=df.copy()
    tmp["prob"]=p

    if "class_name" in tmp.columns:

        print(
            tmp.groupby("class_name")
            .prob
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "max"
                ]
            )
        )


# ======================================================
# DeepBeat test
# ======================================================

deep=pd.read_csv(
"data/features/v6/"
"deepbeat_test_independent_v6_14f.csv"
)


evaluate(
"DEEPBEAT TEST",
deep,
deep.label.astype(int)
)



# ======================================================
# PulseWatch TEST
# ======================================================

pulse=pd.read_csv(
"experiments/07_v6_multidomain/artifacts/"
"hard_domain_splits/"
"pulsewatch_test.csv"
)

pulse["label_v7"]=(
    pulse.class_name=="AF"
).astype(int)


evaluate(
"PULSEWATCH TEST AF VS ALL",
pulse,
pulse.label_v7
)



# ======================================================
# LIU TEST
# ======================================================

liu=pd.read_csv(
"experiments/07_v6_multidomain/artifacts/"
"hard_domain_splits/"
"liu_test.csv"
)


liu["label_v7"]=(
    liu.rhythm=="AF"
).astype(int)


liu["class_name"]=liu.rhythm


evaluate(
"LIU TEST AF VS ALL",
liu,
liu.label_v7
)



# ======================================================
# PEAKDET PAC PVC
# ======================================================

peak=pd.read_csv(
"experiments/06_external_validation/"
"peakdet_pac_pvc_frozen_v5_predictions.csv"
)

print()
print("="*80)
print("PEAKDET PAC/PVC")
print("="*80)

pp=model.predict_proba(
    peak[FEATURES]
)[:,1]


print(
"mean:",
pp.mean()
)

print(
"median:",
pd.Series(pp).median()
)

print(
">=0.5:",
(pp>=0.5).sum()
)

print(
">=0.8:",
(pp>=0.8).sum()
)

print(
"max:",
pp.max()
)

