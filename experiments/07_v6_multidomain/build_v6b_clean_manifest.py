from pathlib import Path
import pandas as pd


OUT = Path(
    "data/features/v7/"
    "healthsense_af_v6c_pacaware_train.csv"
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


frames=[]


# =====================================================
# DeepBeat
# =====================================================

deep = pd.read_csv(
    "data/features/v6/"
    "deepbeat_train_v6_14f.csv"
)

deep["domain"]="DEEPBEAT"

deep["label_v7"] = (
    deep["label"]
    .astype(int)
)

deep["class_name"] = (
    deep["label_v7"]
    .map(
        {
            0:"NON_AF",
            1:"AF"
        }
    )
)

frames.append(
    deep[
        FEATURES
        +
        [
            "label_v7",
            "domain",
            "subject_id",
            "class_name"
        ]
    ]
)



# =====================================================
# MIMIC
# =====================================================

mimic=pd.read_csv(
    "data/features/v6/"
    "mimic_perform_v6_14f.csv"
)

mimic["domain"]="MIMIC_PERFORM"

mimic["label_v7"] = (
    mimic.status
    .astype(int)
)

mimic["subject_id"] = (
    "mimic_"
    +
    mimic.record_id.astype(str)
)

mimic["class_name"] = (
    mimic["label_v7"]
    .map(
        {
            0:"NON_AF",
            1:"AF"
        }
    )
)


frames.append(
    mimic[
        FEATURES
        +
        [
            "label_v7",
            "domain",
            "subject_id",
            "class_name"
        ]
    ]
)



# =====================================================
# PulseWatch TRAIN
# =====================================================

pulse=pd.read_csv(
    "experiments/07_v6_multidomain/artifacts/"
    "hard_domain_splits/"
    "pulsewatch_train.csv"
)


pulse["domain"]="PULSEWATCH"


pulse["label_v7"] = (
    pulse.class_name=="AF"
).astype(int)


pulse["subject_id"] = (
    pulse.uid.astype(str)
)


# KEEP ORIGINAL RHYTHM CLASS
# AF -> positive
# NSR/PAC_PVC -> negative

frames.append(
    pulse[
        FEATURES
        +
        [
            "label_v7",
            "domain",
            "subject_id",
            "class_name"
        ]
    ]
)



# =====================================================
# Liu TRAIN
# =====================================================

liu=pd.read_csv(
    "experiments/07_v6_multidomain/artifacts/"
    "hard_domain_splits/"
    "liu_train.csv"
)


liu["domain"]="LIU"


liu["label_v7"] = (
    liu.rhythm=="AF"
).astype(int)


liu["subject_id"] = (
    liu.subject_id.astype(str)
)


liu["class_name"] = (
    liu["label_v7"]
    .map(
        {
            0:"OTHER",
            1:"AF"
        }
    )
)


frames.append(
    liu[
        FEATURES
        +
        [
            "label_v7",
            "domain",
            "subject_id",
            "class_name"
        ]
    ]
)



# =====================================================
# CONCAT
# =====================================================

df=pd.concat(
    frames,
    ignore_index=True
)


print("="*100)
print("V6-C PAC AWARE MANIFEST")
print("="*100)


print()
print("shape:")
print(df.shape)


print()
print("domain / class")


print(
df.groupby(
    [
        "domain",
        "class_name"
    ]
).size()
)



print()
print("binary label")


print(
df.label_v7.value_counts()
)



print()
print(
"subjects:",
df.subject_id.nunique()
)



OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


df.to_csv(
    OUT,
    index=False
)


print()
print(
"saved:",
OUT
)

