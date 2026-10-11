from pathlib import Path
import json

import pandas as pd


ROOT=Path(__file__).resolve().parents[3]

ART=(
ROOT/
"results/v6"
)


rows=[]


for domain in [
"deepbeat",
"pulsewatch",
"liu",
"mimic_perform"
]:

    f=(
        ART /
        f"lodo_{domain}_summary.json"
    )


    if not f.exists():
        print(
            "missing:",
            f
        )
        continue


    rows.append(
        json.loads(
            f.read_text()
        )
    )


out=pd.DataFrame(
    rows
)


out.to_csv(
    ART /
    "lodo_domain_generalization_summary.csv",
    index=False
)


print(
    out[
        [
            "heldout_domain",
            "eval_rows",
            "eval_subjects",
            "AUROC",
            "PR_AUC",
            "Brier",
            "sensitivity_at_0.5",
            "specificity_at_0.5",
        ]
    ].to_string(
        index=False
    )
)
