from pathlib import Path
import pandas as pd


OUT = Path(
    "data/features/v7/"
    "healthsense_af_v6b_train.csv"
)


FEATURES = [
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


# DeepBeat
deep = pd.read_csv(
"data/features/v6/deepbeat_train_v6_14f.csv"
)

deep["domain"]="DEEPBEAT"
deep["label_v7"]=deep.label


# MIMIC
mimic=pd.read_csv(
"data/features/v6/mimic_perform_v6_14f.csv"
)

mimic["domain"]="MIMIC_PERFORM"
mimic["label_v7"] = (
    mimic.status.astype(int)
)


# V7
v7=pd.read_csv(
"data/features/v7/hard_negative_v7_v2.csv"
)


cols=FEATURES+[
"label_v7",
"domain"
]


deep=deep[cols]
mimic=mimic[cols]
v7=v7[cols]


df=pd.concat(
[
deep,
mimic,
v7
],
ignore_index=True
)


df=df.sample(
frac=1,
random_state=42
)


df.to_csv(
OUT,
index=False
)


print("="*100)
print("V6-B MANIFEST")
print("="*100)

print(df.shape)

print()

print(
df.groupby(
[
"domain",
"label_v7"
]
).size()
)

print()

print(
"saved:",
OUT
)
