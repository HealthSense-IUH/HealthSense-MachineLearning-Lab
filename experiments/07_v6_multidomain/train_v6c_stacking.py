from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier
)

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA = (
    "data/features/v7/"
    "healthsense_af_v6c_pacaware_train.csv"
)


OUT = Path(
    "models/multidomain/"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
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
y=df.label_v7.astype(int)

groups=df.subject_id.astype(str)


models={

"ExtraTrees":
ExtraTreesClassifier(
    n_estimators=500,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
),


"RandomForest":
RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
),


"XGBoost":
XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42
)

}


gkf=GroupKFold(
    n_splits=5
)


oof=np.zeros(
    (len(df), len(models))
)


for fold,(train_idx,val_idx) in enumerate(
    gkf.split(
        X,
        y,
        groups
    )
):

    print(
        "FOLD",
        fold+1
    )

    for j,(name,model) in enumerate(
        models.items()
    ):

        print(
            " ",
            name
        )

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )


        oof[val_idx,j]=(
            model.predict_proba(
                X.iloc[val_idx]
            )[:,1]
        )


meta=Pipeline(
[
(
"scale",
StandardScaler()
),
(
"lr",
LogisticRegression(
    max_iter=1000
)
)
]
)


meta.fit(
    oof,
    y
)


# train final base models

for name,model in models.items():

    print(
        "FINAL TRAIN",
        name
    )

    model.fit(
        X,
        y
    )


joblib.dump(
{
"base_models":models,
"meta":meta,
"features":FEATURES
},
OUT /
"healthsense_af_v6c_stacking.pkl"
)


print(
"saved stacking"
)

