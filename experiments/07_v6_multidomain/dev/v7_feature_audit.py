from pathlib import Path

import pandas as pd
import numpy as np


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


DEEP = (
    "data/features/v6/"
    "deepbeat_train_v6_14f.csv"
)

V7 = (
    "data/features/v7/"
    "hard_negative_v7_v2.csv"
)


print("="*100)
print("LOAD DATA")
print("="*100)


deep = pd.read_csv(DEEP)

v7 = pd.read_csv(V7)


print(
    "DeepBeat:",
    deep.shape
)

print(
    "V7:",
    v7.shape
)


# =====================================================
# Label mapping
# =====================================================

deep_af = deep[
    deep.label == 1
]

deep_non = deep[
    deep.label == 0
]


v7_af = v7[
    v7.label_v7 == 1
]

v7_non = v7[
    v7.label_v7 == 0
]


# =====================================================
# Statistics
# =====================================================

def summary(name, df):

    print()
    print("-"*100)
    print(name)

    out=[]

    for f in FEATURES:

        out.append(
            {
                "feature":f,

                "mean":
                    df[f].mean(),

                "std":
                    df[f].std(),

                "median":
                    df[f].median(),

            }
        )


    print(
        pd.DataFrame(out)
        .to_string(
            index=False
        )
    )



summary(
    "DEEPBEAT AF",
    deep_af
)


summary(
    "V7 AF",
    v7_af
)


summary(
    "V7 HARD NEGATIVE",
    v7_non
)


# =====================================================
# Domain distribution
# =====================================================

print()
print("="*100)
print("V7 NON AF DOMAIN")
print("="*100)


print(
    v7_non.domain.value_counts()
)


# =====================================================
# Missing
# =====================================================

print()
print("="*100)
print("NaN")
print("="*100)


print(
    v7.isna()
    .sum()
)

