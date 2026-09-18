import time
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score
)


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



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


DATA=Path(
"data/features/v7/"
"healthsense_af_v6c_pacaware_train.csv"
)


OUT=Path(
"experiments/07_v6_multidomain/artifacts/"
"v6c_model_benchmark.csv"
)


df=pd.read_csv(
DATA,
low_memory=False
)


X=df[FEATURES]
y=df.label_v7.astype(int)


sample_weight=df.class_name.map(
{
"PAC_PVC":3,
"AF":1.5,
"OTHER":1.5,
"NSR":1,
"NON_AF":1
}
)


models={

"RandomForest":
RandomForestClassifier(
n_estimators=500,
max_depth=14,
n_jobs=-1,
random_state=42
),


"ExtraTrees":
ExtraTreesClassifier(
n_estimators=500,
max_depth=14,
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
eval_metric="logloss"
),


"LightGBM":
LGBMClassifier(
n_estimators=500,
learning_rate=0.05,
num_leaves=31
),


"LogisticRegression":
Pipeline(
[
("scale",StandardScaler()),
("lr",LogisticRegression(
max_iter=1000
))
]
),


"SVM":
Pipeline(
[
("scale",StandardScaler()),
("svm",SVC(
probability=True
))
]
),


"KNN":
Pipeline(
[
("scale",StandardScaler()),
("knn",KNeighborsClassifier(
n_neighbors=15
))
]
),


"MLP":
Pipeline(
[
("scale",StandardScaler()),
("mlp",MLPClassifier(
hidden_layer_sizes=(64,32),
max_iter=100
))
]
),

}


results=[]


for name,model in models.items():

    print("="*80)
    print(name)

    start=time.time()


    try:

        model.fit(
            X,
            y,
            **(
                {
                "sample_weight":sample_weight
                }
                if name not in
                [
                "LogisticRegression",
                "SVM",
                "KNN",
                "MLP"
                ]
                else {}
            )
        )


        t=time.time()-start


        p=model.predict_proba(X)[:,1]


        pred=p>=0.5


        tn,fp,fn,tp=confusion_matrix(
            y,
            pred
        ).ravel()


        results.append(
        {
        "model":name,
        "AUROC":roc_auc_score(y,p),
        "PR_AUC":average_precision_score(y,p),
        "Brier":brier_score_loss(y,p),
        "Sensitivity":tp/(tp+fn),
        "Specificity":tn/(tn+fp),
        "F1":f1_score(y,pred),
        "train_sec":t
        }
        )


    except Exception as e:

        print(e)



# ensemble

ensemble=VotingClassifier(
[
("rf",models["RandomForest"]),
("et",models["ExtraTrees"]),
("xgb",models["XGBoost"]),
("lgbm",models["LightGBM"])
],
voting="soft"
)


ensemble.fit(
X,
y
)


p=ensemble.predict_proba(X)[:,1]


results.append(
{
"model":"SoftVoting",
"AUROC":roc_auc_score(y,p),
"PR_AUC":average_precision_score(y,p),
"Brier":brier_score_loss(y,p),
"F1":f1_score(y,p>=0.5)
}
)



out=pd.DataFrame(results)

out=out.sort_values(
"AUROC",
ascending=False
)


OUT.parent.mkdir(
parents=True,
exist_ok=True
)

out.to_csv(
OUT,
index=False
)


print(out)

print()

print(
"saved:",
OUT
)
