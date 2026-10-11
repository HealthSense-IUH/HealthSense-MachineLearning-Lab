from pathlib import Path

import joblib
import pandas as pd
import numpy as np


MODEL=(
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
"PPG_AC",
]


ART=Path(
"experiments/07_v6_multidomain/artifacts"
)


def predict_uncertainty(df):

    stack=joblib.load(MODEL)

    X=df[FEATURES]


    probs=[]


    for name,model in stack["base_models"].items():

        p=model.predict_proba(X)[:,1]

        probs.append(p)


    probs=np.vstack(probs)


    df["p_et"]=probs[0]
    df["p_rf"]=probs[1]
    df["p_xgb"]=probs[2]


    df["ensemble_mean"]=(
        probs.mean(axis=0)
    )


    df["ensemble_std"]=(
        probs.std(axis=0)
    )


    return df



datasets=[

(
"pulsewatch",
"experiments/07_v6_multidomain/artifacts/"
"pulsewatch_v6c_predictions.csv"
),

(
"deepbeat",
"experiments/07_v6_multidomain/artifacts/"
"deepbeat_v6c_predictions.csv"
),

(
"liu",
"experiments/07_v6_multidomain/artifacts/"
"liu_v6c_predictions.csv"
)

]


for name,path in datasets:

    # cần load feature gốc vì prediction file chỉ có probability
    print(
        "processing",
        name
    )


print("DONE")
