from pathlib import Path

import joblib
import pandas as pd


MODEL = (
    "models/multidomain/"
    "healthsense_af_v6c_stacking.pkl"
)


OUT = Path(
    "experiments/07_v6_multidomain/artifacts"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


stack = joblib.load(
    MODEL
)


base_models = stack["base_models"]
meta_model = stack["meta"]
FEATURES = stack["features"]


print(
    "STACK:",
    list(base_models.keys())
)


def predict_stack(df):

    X=df[FEATURES]


    preds={}


    for model_name,model in base_models.items():

        preds[model_name]=(
            model.predict_proba(
                X.values
            )[:,1]
        )


    meta_X=pd.DataFrame(
        preds
    )


    return meta_model.predict_proba(
        meta_X
    )[:,1]



def process(
    name,
    path,
    subject_col,
    label_col=None,
    label_func=None,
    timestamp_col=None,
):

    print()
    print("="*80)
    print(name)
    print("="*80)


    df=pd.read_csv(
        path
    )


    if label_func:

        df["label"]=label_func(df)


    else:

        df=df.rename(
            columns={
                label_col:"label"
            }
        )


    df["subject_id"]=(
        df[subject_col]
        .astype(str)
    )


    df["probability"]=predict_stack(
        df
    )


    # preserve temporal order
    df["window_index"] = (
        df.groupby("subject_id")
        .cumcount()
    )


    keep=[
        "subject_id",
        "label",
        "window_index",
        "probability"
    ]


    if timestamp_col and timestamp_col in df.columns:

        keep.insert(
            2,
            timestamp_col
        )


    out=OUT / (
        name.lower()
        +
        "_v6c_predictions.csv"
    )


    df[keep].to_csv(
        out,
        index=False
    )


    print(df[keep].head())
    print("rows:",len(df))
    print("saved:",out)



# ==========================================================
# DeepBeat
# ==========================================================

process(
    "DeepBeat",
    "data/features/v6/"
    "deepbeat_test_independent_v6_14f.csv",
    "subject_id",
    label_col="label",
    timestamp_col="timestamp"
)


# ==========================================================
# PulseWatch
# ==========================================================

process(
    "PulseWatch",
    "experiments/07_v6_multidomain/artifacts/"
    "hard_domain_splits/"
    "pulsewatch_test.csv",
    "uid",
    label_func=lambda x:
        (
            x.class_name=="AF"
        ).astype(int),
)



# ==========================================================
# LIU
# ==========================================================

process(
    "LIU",
    "experiments/07_v6_multidomain/artifacts/"
    "hard_domain_splits/"
    "liu_test.csv",
    "subject_id",
    label_func=lambda x:
        (
            x.rhythm=="AF"
        ).astype(int),
)


print()
print("DONE")
