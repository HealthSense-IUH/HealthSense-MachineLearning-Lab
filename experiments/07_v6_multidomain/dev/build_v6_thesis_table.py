from pathlib import Path
import pandas as pd


ART=Path(
"experiments/07_v6_multidomain/artifacts"
)


metrics=[]


# v6e window-level
v6e={
"deepbeat":{
"AUROC":0.9903911446181703,
"PR_AUC":0.9889661135717717,
"Brier":0.058315144226885586
},

"pulsewatch":{
"AUROC":0.9984653083599642,
"PR_AUC":0.9889539696985515,
"Brier":0.009482898364224656
},

"liu":{
"AUROC":0.9866240530303031,
"PR_AUC":0.9779764983188317,
"Brier":0.032217392349941594
}
}


decision=pd.read_csv(
ART/"v6h_selected_operating_points.csv"
)


for _,r in decision.iterrows():

    d=r.dataset

    metrics.append(
        {
        "dataset":d,

        "AUROC":
            v6e[d]["AUROC"],

        "PR_AUC":
            v6e[d]["PR_AUC"],

        "Brier":
            v6e[d]["Brier"],

        "TP":
            r.TP,

        "TN":
            r.TN,

        "FP":
            r.FP,

        "FN":
            r.FN,

        "Sensitivity":
            r.sensitivity,

        "Specificity":
            r.specificity,

        "consecutive":
            r.consecutive,

        "positive_windows":
            r.positive,

        "uncertainty":
            r.uncertainty
        }
    )


out=pd.DataFrame(metrics)


out.to_csv(
ART/"v6_final_thesis_table.csv",
index=False
)


print(out)

