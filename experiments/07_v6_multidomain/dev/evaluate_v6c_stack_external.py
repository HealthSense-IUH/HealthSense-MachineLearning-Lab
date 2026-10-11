import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    confusion_matrix
)


MODEL_PATH = (
    "models/multidomain/"
    "healthsense_af_v6c_stacking.pkl"
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
"PPG_AC"
]


bundle = joblib.load(
    MODEL_PATH
)


base_models = bundle["base_models"]
meta = bundle["meta"]


def predict_stack(df):

    X=df[FEATURES]

    base=[]

    for name,model in base_models.items():

        p=model.predict_proba(X)[:,1]

        base.append(p)


    base=np.vstack(base).T


    return meta.predict_proba(base)[:,1]



def evaluate(name,df):

    y=df.label_v7.values

    p=predict_stack(df)


    print()
    print("="*80)
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
        "F1:",
        f1_score(
            y,
            p>=0.5
        )
    )

    print(
        confusion_matrix(
            y,
            p>=0.5
        )
    )



# =====================================================
# DeepBeat
# =====================================================

deep=pd.read_csv(
"data/features/v6/"
"deepbeat_test_independent_v6_14f.csv"
)

deep["label_v7"]=deep.label

evaluate(
    "DEEPBEAT",
    deep
)



# =====================================================
# PulseWatch
# =====================================================

pulse=pd.read_csv(
"experiments/07_v6_multidomain/artifacts/"
"hard_domain_splits/"
"pulsewatch_test.csv"
)

pulse["label_v7"]=(
    pulse.class_name=="AF"
).astype(int)


evaluate(
    "PULSEWATCH",
    pulse
)



# =====================================================
# LIU
# =====================================================

liu=pd.read_csv(
"experiments/07_v6_multidomain/artifacts/"
"hard_domain_splits/"
"liu_test.csv"
)

liu["label_v7"]=(
    liu.rhythm=="AF"
).astype(int)


evaluate(
    "LIU",
    liu
)


