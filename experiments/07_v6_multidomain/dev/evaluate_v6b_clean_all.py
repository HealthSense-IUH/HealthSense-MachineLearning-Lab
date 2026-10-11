import joblib
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


MODEL = (
    "models/multidomain/"
    "healthsense_af_v6b_clean_rf.pkl"
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
"PPG_AC",
]


model = joblib.load(MODEL)



def evaluate(name, df, y):

    X=df[FEATURES]

    p=model.predict_proba(X)[:,1]

    print()
    print("="*100)
    print(name)
    print("="*100)

    print("windows:",len(df))

    print(
        "AUROC:",
        roc_auc_score(y,p)
    )

    print(
        "PR-AUC:",
        average_precision_score(y,p)
    )

    print(
        "Brier:",
        brier_score_loss(y,p)
    )


    pred=(p>=0.5).astype(int)

    print(
        confusion_matrix(
            y,
            pred
        )
    )



# =====================================================
# DeepBeat
# =====================================================

deep=pd.read_csv(
    "data/features/v6/"
    "deepbeat_test_independent_v6_14f.csv"
)

evaluate(
    "DEEPBEAT TEST",
    deep,
    deep.label.astype(int)
)



# =====================================================
# PulseWatch TEST
# =====================================================

pulse=pd.read_csv(
    "experiments/07_v6_multidomain/artifacts/"
    "hard_domain_splits/"
    "pulsewatch_test.csv"
)

pulse=pulse[
    pulse.feature_ok==True
].copy()

pulse["y"]=(
    pulse.class_name=="AF"
).astype(int)


evaluate(
    "PULSEWATCH TEST AF vs ALL",
    pulse,
    pulse.y
)



# =====================================================
# Liu TEST
# =====================================================

liu=pd.read_csv(
    "experiments/07_v6_multidomain/artifacts/"
    "hard_domain_splits/"
    "liu_test.csv"
)

liu["y"]=(
    liu.rhythm=="AF"
).astype(int)


evaluate(
    "LIU TEST AF vs ALL",
    liu,
    liu.y
)



# =====================================================
# PeakDet
# =====================================================

peak=pd.read_csv(
    "experiments/06_external_validation/"
    "peakdet_pac_pvc_frozen_v5_predictions.csv"
)

p=model.predict_proba(
    peak[FEATURES]
)[:,1]


print()
print("="*100)
print("PEAKDET PAC/PVC")
print("="*100)

print("windows:",len(p))
print("mean:",p.mean())
print("median:",pd.Series(p).median())
print("max:",p.max())
print(">=0.5:",(p>=0.5).sum())

