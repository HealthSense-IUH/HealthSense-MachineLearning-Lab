import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier
)

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


DATA = (
    "data/features/v7/"
    "healthsense_af_v6c_pacaware_train.csv"
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


df=pd.read_csv(
    DATA,
    low_memory=False
)


X=df[FEATURES]
y=df.label_v7


models={

"RandomForest":
RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
),


"ExtraTrees":
ExtraTreesClassifier(
    n_estimators=400,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
),


"XGBoost":
XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist"
),


"LightGBM":
LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05
),


"CatBoost":
CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    verbose=False
),


"HistGB":
HistGradientBoostingClassifier(
    max_iter=300
)

}


results=[]


for name,model in models.items():

    print("="*80)
    print(name)

    t=time.time()

    model.fit(
        X,
        y
    )

    p=model.predict_proba(X)[:,1]

    results.append({

        "model":name,

        "AUROC":
        roc_auc_score(y,p),

        "PR_AUC":
        average_precision_score(y,p),

        "Brier":
        brier_score_loss(y,p),

        "F1":
        f1_score(
            y,
            p>=0.5
        ),

        "train_sec":
        time.time()-t

    })


result=pd.DataFrame(results)

print(result.sort_values(
    "AUROC",
    ascending=False
))


Path(
"experiments/07_v6_multidomain/artifacts"
).mkdir(
    parents=True,
    exist_ok=True
)



import joblib

for name, model in models.items():
    joblib.dump(
        model,
        f"models/multidomain/v6c_{name}.pkl"
    )

print("saved models")

result.to_csv(

"experiments/07_v6_multidomain/artifacts/"
"v6c_sota_benchmark.csv",
index=False
)

