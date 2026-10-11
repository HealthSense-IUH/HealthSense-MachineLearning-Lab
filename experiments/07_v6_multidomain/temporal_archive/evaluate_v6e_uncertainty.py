from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)


ART=Path(
"experiments/07_v6_multidomain/artifacts"
)


for name in [
    "deepbeat",
    "pulsewatch",
    "liu"
]:

    print("="*80)
    print(name)


    df=pd.read_csv(
        ART/f"{name}_v6e_predictions.csv"
    )


    y=df.label.copy()

    if name=="pulsewatch":
        # PulseWatch: 0=NSR, 1=AF, 2=PAC/PVC
        # binary AF screening
        y=(y==1).astype(int)

    p=df.meta_probability
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


    print(
        "uncertainty:"
    )

    print(
        df.ensemble_std.describe()
    )


    print(
        "high uncertainty (>0.2):",
        (df.ensemble_std>0.2).sum()
    )
