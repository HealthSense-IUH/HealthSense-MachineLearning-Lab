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



decision=pd.read_csv(
ART/"v6f_uncertainty_temporal_subject_results.csv"
)



results=[]


for dataset,(path,id_col,label_col) in GT.items():

    print("="*80)
    print(dataset)


    gt=pd.read_csv(path)


    gt[id_col]=gt[id_col].astype(str)


    if dataset=="pulsewatch":
        gt[label_col]=(gt[label_col]==1).astype(int)


    gt=gt.groupby(
        id_col
    )[label_col].max()


    gt.index=(
        gt.index
        .astype(str)
    )


    pred=decision[
        (decision.dataset==dataset)
        &
        (decision.method=="v6f")
    ]


    pred=pred.copy()

    pred["subject_id"]=pred.subject_id.astype(str)


    pred=pred.groupby(
        "subject_id"
    ).alert.max()



    gt.index=gt.index.astype(str)


    merged=pd.concat(
        [
            gt.rename("gt"),
            pred.rename("pred")
        ],
        axis=1
    ).fillna(0)

    print("MERGED")
    print(merged)
    print("shape:", merged.shape)
    print("gt count")
    print(merged["gt"].value_counts())
    print("pred count")
    print(merged["pred"].value_counts())



    TP=((merged["gt"]==1)&(merged["pred"]==1)).sum()
    TN=((merged["gt"]==0)&(merged["pred"]==0)).sum()
    FP=((merged["gt"]==0)&(merged["pred"]==1)).sum()
    FN=((merged["gt"]==1)&(merged["pred"]==0)).sum()

    results.append(
        {
        "dataset":dataset,
        "TP":TP,
        "TN":TN,
        "FP":FP,
        "FN":FN,
        "sensitivity":TP/(TP+FN) if TP+FN else 0,
        "specificity":TN/(TN+FP) if TN+FP else 0,
        }
    )
print(
    pd.DataFrame(results)
)
