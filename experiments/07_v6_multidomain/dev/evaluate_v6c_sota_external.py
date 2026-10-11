import joblib
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
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


MODELS=[
"RandomForest",
"ExtraTrees",
"XGBoost",
"LightGBM",
"CatBoost",
"HistGB"
]


def evaluate(name, df, label):

    model=joblib.load(
        f"models/multidomain/v6c_{name}.pkl"
    )

    x=df[FEATURES]

    y=df[label]

    p=model.predict_proba(x)[:,1]


    return {
        "model":name,
        "AUROC":roc_auc_score(y,p),
        "PR_AUC":average_precision_score(y,p),
        "Brier":brier_score_loss(y,p),
        "F1":(
            ((p>=0.5)==y)
            .mean()
        )
    }



results=[]


# ================================
# DeepBeat
# ================================

deep=pd.read_csv(
"data/features/v6/"
"deepbeat_test_independent_v6_14f.csv"
)

deep["label_v7"]=deep.label


for m in MODELS:

    r=evaluate(
        m,
        deep,
        "label_v7"
    )

    r["dataset"]="DeepBeat"
    results.append(r)



# ================================
# PulseWatch test
# ================================

pulse=pd.read_csv(
"experiments/07_v6_multidomain/artifacts/"
"hard_domain_splits/"
"pulsewatch_test.csv"
)

pulse["label_v7"]=(
    pulse.class_name=="AF"
).astype(int)


for m in MODELS:

    r=evaluate(
        m,
        pulse,
        "label_v7"
    )

    r["dataset"]="PulseWatch"
    results.append(r)



# ================================
# LIU test
# ================================

liu=pd.read_csv(
"experiments/07_v6_multidomain/artifacts/"
"hard_domain_splits/"
"liu_test.csv"
)

liu["label_v7"]=(
    liu.rhythm=="AF"
).astype(int)


for m in MODELS:

    r=evaluate(
        m,
        liu,
        "label_v7"
    )

    r["dataset"]="LIU"
    results.append(r)



out=pd.DataFrame(results)

print(
out.sort_values(
[
"dataset",
"AUROC"
],
ascending=False
)
)


Path(
"experiments/07_v6_multidomain/artifacts"
).mkdir(
parents=True,
exist_ok=True
)


out.to_csv(
"experiments/07_v6_multidomain/artifacts/"
"v6c_sota_external_ranking.csv",
index=False
)

print("saved")
