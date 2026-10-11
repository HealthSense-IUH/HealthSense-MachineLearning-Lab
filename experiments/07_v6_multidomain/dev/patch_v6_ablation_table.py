from pathlib import Path
import pandas as pd


ART=Path(
"experiments/07_v6_multidomain/artifacts"
)


ab=pd.read_csv(
ART/"v6_ablation_summary.csv"
)


temp=pd.read_csv(
ART/"v6_temporal_comparison_final.csv"
)


for c in [
    "TP",
    "TN",
    "FP",
    "FN"
]:
    if c not in ab.columns:
        ab[c]=None


for _,r in temp.iterrows():

    mask=(
        (ab.stage==r.method)
        &
        (ab.dataset==r.dataset)
    )

    ab.loc[
        mask,
        [
        "TP",
        "TN",
        "FP",
        "FN"
        ]
    ]=[
        r.TP,
        r.TN,
        r.FP,
        r.FN
    ]



ab.to_csv(
ART/"v6_ablation_summary_final.csv",
index=False
)


with pd.ExcelWriter(
ART/"v6_complete_summary_tables_final.xlsx"
) as writer:

    ab.to_excel(
        writer,
        sheet_name="ablation",
        index=False
    )

    pd.read_csv(
        ART/"v6_generalization_summary.csv"
    ).to_excel(
        writer,
        sheet_name="generalization",
        index=False
    )


print(ab)

