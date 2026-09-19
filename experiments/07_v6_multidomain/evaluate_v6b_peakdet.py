import joblib
import pandas as pd


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
"peakdet_pac_pvc_frozen_v5_predictions.csv"
)


df=df[
    df.feature_ok==True
]


print("="*100)
print("PEAKDET PAC/PVC")
print("="*100)

print(
"windows:",
len(df)
)


for name,model in models.items():

    p=model.predict_proba(
        df[FEATURES]
    )[:,1]


    print()
    print("="*80)
    print(name)

    print(
        "mean probability:",
        p.mean()
    )

    print(
        "median probability:",
        pd.Series(p).median()
    )

    print(
        ">=0.5:",
        (p>=0.5).mean()
    )

    print(
        ">=0.8:",
        (p>=0.8).mean()
    )

    print(
        "max:",
        p.max()
    )

