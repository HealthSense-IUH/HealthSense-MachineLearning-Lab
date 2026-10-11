from pathlib import Path
import pandas as pd


ART=Path(
"experiments/07_v6_multidomain/artifacts"
)


GT={
"deepbeat":
(
"data/features/v6/deepbeat_test_independent_v6_14f.csv",
"subject_id",
"label"
),

"pulsewatch":
(
"experiments/07_v6_multidomain/artifacts/hard_domain_splits/pulsewatch_test.csv",
"uid",
"label"
),

"liu":
(
"experiments/07_v6_multidomain/artifacts/hard_domain_splits/liu_test.csv",
"subject_id",
"binary_label_clean"
)
}



def calc(gt,pred):

    m=pd.concat(
        [
            gt.rename("gt"),
            pred.rename("pred")
        ],
        axis=1
    ).fillna(0)


    TP=((m["gt"]==1)&(m["pred"]==1)).sum()
    TN=((m["gt"]==0)&(m["pred"]==0)).sum()
    FP=((m["gt"]==0)&(m["pred"]==1)).sum()
    FN=((m["gt"]==1)&(m["pred"]==0)).sum()


    return {
        "TP":TP,
        "TN":TN,
        "FP":FP,
        "FN":FN,

        "Sensitivity":
            TP/(TP+FN) if TP+FN else 0,

        "Specificity":
            TN/(TN+FP) if TN+FP else 0,

        "PPV":
            TP/(TP+FP) if TP+FP else 0,

        "NPV":
            TN/(TN+FN) if TN+FN else 0,

        "Accuracy":
            (TP+TN)/(TP+TN+FP+FN)
            if TP+TN+FP+FN else 0,

        "F1":
            2*TP/(2*TP+FP+FN)
            if 2*TP+FP+FN else 0
    }



def load_gt(name):

    path,id_col,label_col=GT[name]

    df=pd.read_csv(path)

    df[id_col]=df[id_col].astype(str)


    if name=="pulsewatch":
        df[label_col]=(df[label_col]==1).astype(int)


    return (
        df.groupby(id_col)[label_col]
        .max()
    )



rows=[]



# ==========================
# v6h
# ==========================

v6h=pd.read_csv(
ART/"v6h_selected_operating_points.csv"
)


for _,r in v6h.iterrows():

    rows.append(
        {
        "dataset":r.dataset,
        "method":"v6h",
        **calc(
            load_gt(r.dataset),
            pd.Series(
                {
                str(x):1
                for x in []
                }
            )
        )
        }
    )



# replace v6h directly using saved confusion
rows=[]

for _,r in v6h.iterrows():

    TP=r.TP
    TN=r.TN
    FP=r.FP
    FN=r.FN

    rows.append(
        {
        "dataset":r.dataset,
        "method":"v6h",
        "TP":TP,
        "TN":TN,
        "FP":FP,
        "FN":FN,
        "Sensitivity":r.sensitivity,
        "Specificity":r.specificity,
        "PPV":TP/(TP+FP) if TP+FP else 0,
        "NPV":TN/(TN+FN) if TN+FN else 0,
        "Accuracy":
            (TP+TN)/(TP+TN+FP+FN),
        "F1":
            2*TP/(2*TP+FP+FN)
        }
    )



# ==========================
# v6f
# ==========================

v6f=pd.read_csv(
ART/"v6f_uncertainty_temporal_subject_results.csv"
)


for name in GT:

    gt=load_gt(name)
    gt.index=gt.index.astype(str)


    pred=v6f[
        (v6f.dataset==name)
        &
        (v6f.method=="v6f")
    ]


    pred=pred.groupby(
        "subject_id"
    ).alert.max()


    pred.index=pred.index.astype(str)


    rows.append(
        {
        "dataset":name,
        "method":"v6f",
        **calc(gt,pred)
        }
    )



# ==========================
# v6g
# ==========================

v6g=pd.read_csv(
ART/"v6g_episode_decision_results.csv"
)


for name in GT:

    x=v6g[
        v6g.dataset==name
    ].sort_values(
        [
        "sensitivity",
        "specificity"
        ],
        ascending=False
    ).iloc[0]


    rows.append(
        {
        "dataset":name,
        "method":"v6g",
        "TP":x.TP,
        "TN":x.TN,
        "FP":x.FP,
        "FN":x.FN,
        "Sensitivity":x.sensitivity,
        "Specificity":x.specificity,
        "PPV":x.TP/(x.TP+x.FP),
        "NPV":x.TN/(x.TN+x.FN),
        "Accuracy":
            (x.TP+x.TN)/(x.TP+x.TN+x.FP+x.FN),
        "F1":
            2*x.TP/(2*x.TP+x.FP+x.FN)
        }
    )


out=pd.DataFrame(rows)


out.to_csv(
ART/"v6_temporal_comparison_final.csv",
index=False
)


print(out)

