from pathlib import Path
import pandas as pd


ART=Path(
"experiments/07_v6_multidomain/artifacts"
)


# =====================================================
# 1. Temporal comparison
# =====================================================

v6f=pd.read_csv(
    ART/"v6f_uncertainty_temporal_subject_results.csv"
)

v6f=v6f[
    v6f.method=="v6f"
]


v6f=v6f.groupby(
"dataset"
).first().reset_index()


v6h=pd.read_csv(
    ART/"v6h_selected_operating_points.csv"
)


temporal=[]


for _,r in v6f.iterrows():

    temporal.append(
        {
        "dataset":r.dataset,
        "method":"v6f",
        "TP":None,
        "TN":None,
        "FP":None,
        "FN":None,
        "Sensitivity":None,
        "Specificity":None
        }
    )


for _,r in v6h.iterrows():

    temporal.append(
        {
        "dataset":r.dataset,
        "method":"v6h",
        "TP":r.TP,
        "TN":r.TN,
        "FP":r.FP,
        "FN":r.FN,
        "Sensitivity":r.sensitivity,
        "Specificity":r.specificity
        }
    )


temporal=pd.DataFrame(temporal)


temporal.to_csv(
ART/"v6_temporal_comparison.csv",
index=False
)


# =====================================================
# 2. Model performance summary
# =====================================================

perf=pd.read_csv(
ART/"v6_final_thesis_table.csv"
)


perf.to_csv(
ART/"v6_model_performance_summary.csv",
index=False
)


# =====================================================
# 3. Uncertainty summary
# =====================================================

rows=[]


for name in [
    "deepbeat",
    "pulsewatch",
    "liu"
]:

    df=pd.read_csv(
        ART/f"{name}_v6e_predictions.csv"
    )

    rows.append(
        {
        "dataset":name,
        "mean_uncertainty":
            df.ensemble_std.mean(),
        "std_uncertainty":
            df.ensemble_std.std(),
        "max_uncertainty":
            df.ensemble_std.max(),
        "high_uncertainty_gt_0.2":
            (df.ensemble_std>0.2).sum(),
        "samples":
            len(df)
        }
    )


unc=pd.DataFrame(rows)


unc.to_csv(
ART/"v6_uncertainty_summary.csv",
index=False
)


# =====================================================
# 4. Excel workbook
# =====================================================

with pd.ExcelWriter(
    ART/"v6_thesis_tables.xlsx",
    engine="openpyxl"
) as writer:

    perf.to_excel(
        writer,
        sheet_name="performance",
        index=False
    )

    temporal.to_excel(
        writer,
        sheet_name="temporal",
        index=False
    )

    unc.to_excel(
        writer,
        sheet_name="uncertainty",
        index=False
    )


print("generated:")
print("v6_temporal_comparison.csv")
print("v6_model_performance_summary.csv")
print("v6_uncertainty_summary.csv")
print("v6_thesis_tables.xlsx")

