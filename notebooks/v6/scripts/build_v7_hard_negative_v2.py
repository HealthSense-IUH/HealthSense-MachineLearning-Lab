from pathlib import Path
import pandas as pd
import numpy as np


OUT = Path(
    "data/features/v7"
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


# ============================
# Pulsewatch
# ============================

pulse = pd.read_csv(
    "results/v5/"
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


af = pulse[
    pulse.class_name == "AF"
]


pacpvc = pulse[
    pulse.class_name == "PAC_PVC"
]


nsr = pulse[
    pulse.class_name == "NSR"
]


# AF diversity
# ============================================
# AF balanced subject sampling
# ============================================

af_balanced = (
    af
    .groupby(
        "uid",
        group_keys=False
    )
    .apply(
        lambda x:
        x.sample(
            min(
                len(x),
                300
            ),
            random_state=42
        )
    )
)


# nếu còn thiếu thì lấy thêm từ phần còn lại
remaining = af.drop(
    af_balanced.index
)


need = 5000 - len(af_balanced)


if need > 0:

    extra = remaining.sample(
        n=min(
            need,
            len(remaining)
        ),
        random_state=42
    )

    af_balanced = pd.concat(
        [
            af_balanced,
            extra
        ]
    )


af = af_balanced


# keep ALL ectopy
pacpvc = pacpvc.copy()


# hard NSR:
# high irregularity proxy
nsr["hard_score"] = (
    nsr.RMSSD.rank(pct=True)
    +
    nsr.SDNN.rank(pct=True)
    -
    nsr.SampEn.rank(pct=True)
)


nsr = (
    nsr
    .sort_values(
        "hard_score",
        ascending=False
    )
    .head(5000)
)


# ============================
# Liu
# ============================

liu = pd.read_csv(
    "results/v5/"
    "liu_pseudo30_frozen_v5_predictions.csv"
)


liu = liu[
    liu.feature_ok == True
].copy()


liu = liu[
    liu.rhythm.isin(
        [
            "PAC",
            "PVC",
            "SVT",
            "VT",
        ]
    )
]


liu["label_v7"]=0
liu["domain"]="LIU"
liu["subject_id_v7"]=(
    liu.subject_id.astype(str)
)


# ============================
# PeakDet
# ============================

peak=pd.read_csv(
    "results/v5/"
    "peakdet_pac_pvc_frozen_v5_predictions.csv"
)

peak=peak[
    peak.feature_ok==True
]

peak["label_v7"]=0
peak["domain"]="PEAKDET"
peak["subject_id_v7"]=(
    peak.subject_id.astype(str)
)


# ============================
# Combine
# ============================

cols = FEATURES + [
    "label_v7",
    "domain",
    "subject_id_v7",
]


v7=pd.concat(
    [
        af[cols],
        pacpvc[cols],
        nsr[cols],
        liu[cols],
        peak[cols],
    ],
    ignore_index=True
)


v7=v7.sample(
    frac=1,
    random_state=42
)


v7.to_csv(
    OUT /
    "hard_negative_v7_v2.csv",
    index=False
)


print("="*100)
print("V7 HARD NEGATIVE V2")
print("="*100)

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
    v7.label_v7.value_counts()
)

print()

print(
    "TOTAL:",
    len(v7)
)
