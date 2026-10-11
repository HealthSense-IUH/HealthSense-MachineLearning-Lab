from pathlib import Path
import joblib
import pandas as pd
import numpy as np


MODEL="models/multidomain/healthsense_af_v6c_stacking.pkl"


ART=Path(
"experiments/07_v6_multidomain/artifacts"
)

ART.mkdir(
    parents=True,
    exist_ok=True
)


stack=joblib.load(MODEL)

base_models=stack["base_models"]
meta=stack["meta"]
FEATURES=stack["features"]


def run(name,path,label_col):

    print("="*80)
    print(name)


    df=pd.read_csv(path)


    if label_col in df.columns:
        df["label"] = df[label_col].astype(int)

    X=df[FEATURES]


    base=[]

    for n,m in base_models.items():

        print("base:",n)

        p=m.predict_proba(X)[:,1]

        base.append(p)

        df[n+"_prob"]=p


    base=np.vstack(base).T


    df["ensemble_mean"]=base.mean(axis=1)

    df["ensemble_std"]=base.std(axis=1)


    df["meta_probability"]=(
        meta.predict_proba(base)[:,1]
    )


    out=ART/f"{name}_v6e_predictions.csv"


    cols=[
        c for c in [
            "subject_id",
            "uid",
            "label",
            "window_index",
            "ExtraTrees_prob",
            "RandomForest_prob",
            "XGBoost_prob",
            "ensemble_mean",
            "ensemble_std",
            "meta_probability"
        ]
        if c in df.columns
    ]


    df[cols].to_csv(
        out,
        index=False
    )


    print(df[cols].head())
    print("rows:",len(df))
    print("uncertainty:",df.ensemble_std.mean())
    print("saved:",out)



run(
"deepbeat",
"data/features/v6/deepbeat_test_independent_v6_14f.csv",
"label"
)


run(
"pulsewatch",
"experiments/07_v6_multidomain/artifacts/hard_domain_splits/pulsewatch_test.csv",
"binary_label_clean"
)


run(
"liu",
"experiments/07_v6_multidomain/artifacts/hard_domain_splits/liu_test.csv",
"binary_label_clean"
)

