from pathlib import Path
import pandas as pd
import numpy as np


OUT = Path(
    "data/features/v7"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
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


# =====================================================
# Pulsewatch
# =====================================================

pulse = pd.read_csv(
    "experiments/06_external_validation/"
    "pulsewatch_frozen_v5_features.csv"
)

pulse = pulse[
    pulse.feature_ok == True
].copy()


pulse["label_v7"] = (
    pulse.class_name == "AF"
).astype(int)


pulse["domain"] = "PULSEWATCH"


pulse["subject_id_v7"] = (
    pulse.uid.astype(str)
)


pulse_hn = pulse[
    pulse.class_name == "PAC_PVC"
]


pulse_af = pulse[
    pulse.class_name == "AF"
]


pulse_nsr = pulse[
    pulse.class_name == "NSR"
]


# sample
pulse_hn = pulse_hn.sample(
    n=min(
        5000,
        len(pulse_hn)
    ),
    random_state=42,
)

pulse_nsr = pulse_nsr.sample(
    n=5000,
    random_state=42,
)


# =====================================================
# Liu
# =====================================================

liu = pd.read_csv(
    "experiments/06_external_validation/"
    "liu_pseudo30_frozen_v5_predictions.csv"
)


liu = liu[
    liu.feature_ok == True
].copy()


liu["label_v7"] = (
    liu.rhythm == "AF"
).astype(int)


liu["domain"]="LIU"

liu["subject_id_v7"] = (
    liu.subject_id.astype(str)
)


liu_hn = liu[
    liu.label_v7 == 0
]


liu_hn = liu_hn.sample(
    n=min(
        5000,
        len(liu_hn)
    ),
    random_state=42,
)


# =====================================================
# PeakDet
# =====================================================

peak = pd.read_csv(
    "experiments/06_external_validation/"
    "peakdet_pac_pvc_frozen_v5_predictions.csv"
)


peak = peak[
    peak.feature_ok == True
].copy()


peak["label_v7"]=0

peak["domain"]="PEAKDET"

peak["subject_id_v7"]=(
    peak.subject_id.astype(str)
)


# =====================================================
# Combine
# =====================================================

cols = FEATURES + [
    "label_v7",
    "domain",
    "subject_id_v7",
]


parts = [
    pulse_af[cols],
    pulse_hn[cols],
    pulse_nsr[cols],
    liu_hn[cols],
    peak[cols],
]


v7 = pd.concat(
    parts,
    ignore_index=True
)


v7 = (
    v7
    .sample(
        frac=1,
        random_state=42
    )
    .reset_index(drop=True)
)


v7.to_csv(
    OUT /
    "hard_negative_v7_train.csv",
    index=False
)


print("="*100)
print("V7 HARD NEGATIVE DATASET")
print("="*100)

print(
    v7.domain.value_counts()
)

print()

print(
    v7.label_v7.value_counts()
)

print()

print(
    v7.groupby(
        [
            "domain",
            "label_v7"
        ]
    ).size()
)

print()

print(
    "rows:",
    len(v7)
)

