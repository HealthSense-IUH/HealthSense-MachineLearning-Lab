from pathlib import Path
import pandas as pd


ART=Path(
"experiments/07_v6_multidomain/artifacts"
)


# =====================================================
# 1. Ablation summary
# =====================================================

rows=[]


# v6e final model performance
perf=pd.read_csv(
    ART/"v6_model_performance_summary.csv"
)


for _,r in perf.iterrows():

    rows.append(
        {
        "stage":"v6e",
        "component":"ensemble+uncertainty",
        "dataset":r.dataset,
        "AUROC":r.AUROC,
        "PR_AUC":r.PR_AUC,
        "Brier":r.Brier
        }
    )


# temporal stages
temp=pd.read_csv(
    ART/"v6_temporal_comparison_final.csv"
)


for _,r in temp.iterrows():

    rows.append(
        {
        "stage":r.method,
        "component":
            {
            "v6f":"uncertainty+persistence",
            "v6g":"episode temporal aggregation",
            "v6h":"final alert operating point"
            }.get(
                r.method,
                r.method
            ),

        "dataset":r.dataset,

        "Sensitivity":
            r.Sensitivity,

        "Specificity":
            r.Specificity,

        "Accuracy":
            r.Accuracy,

        "F1":
            r.F1
        }
    )


ab=pd.DataFrame(rows)


ab.to_csv(
    ART/"v6_ablation_summary.csv",
    index=False
)



# =====================================================
# 2. Generalization summary
# =====================================================


gen=[]


# window level external domains

for _,r in perf.iterrows():

    gen.append(
        {
        "dataset":r.dataset,
        "evaluation":
            "independent test",
        "AUROC":r.AUROC,
        "PR_AUC":r.PR_AUC,
        "Brier":r.Brier
        }
    )



# temporal final

for _,r in temp[
    temp.method=="v6h"
].iterrows():

    gen.append(
        {
        "dataset":r.dataset,
        "evaluation":
            "subject-level temporal alert",

        "Sensitivity":
            r.Sensitivity,

        "Specificity":
            r.Specificity,

        "Accuracy":
            r.Accuracy,

        "F1":
            r.F1
        }
    )



# MIMIC LOSO retention if available

mimic=ART/"mimic_loso_v6a_retention_summary.csv"

if mimic.exists():

    m=pd.read_csv(mimic)

    for _,r in m.iterrows():

        gen.append(
            {
            "dataset":"MIMIC-III-Ext-PPG",
            "evaluation":
                "LOSO retention",

            **r.to_dict()
            }
        )


general=pd.DataFrame(gen)


general.to_csv(
    ART/"v6_generalization_summary.csv",
    index=False
)



with pd.ExcelWriter(
    ART/"v6_complete_summary_tables.xlsx"
) as writer:

    ab.to_excel(
        writer,
        sheet_name="ablation",
        index=False
    )

    general.to_excel(
        writer,
        sheet_name="generalization",
        index=False
    )



print("generated:")
print("v6_ablation_summary.csv")
print("v6_generalization_summary.csv")
print("v6_complete_summary_tables.xlsx")


