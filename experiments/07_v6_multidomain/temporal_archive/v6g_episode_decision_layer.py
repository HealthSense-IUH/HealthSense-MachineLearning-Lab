from pathlib import Path
import pandas as pd
import numpy as np


ART=Path(
    "experiments/07_v6_multidomain/artifacts"
)


DATASETS=[
    "deepbeat",
    "pulsewatch",
    "liu"
]


def longest_run(x):

    best=0
    cur=0

    for v in x:

        if v:
            cur+=1
            best=max(best,cur)

        else:
            cur=0

    return best



def extract_episode_features(df):

    rows=[]

    if "uid" in df.columns:
        df["subject_id"]=df.uid.astype(str)
    else:
        df["subject_id"]=df.subject_id.astype(str)


    for sid,g in df.groupby("subject_id"):

        p=g.meta_probability.values
        u=g.ensemble_std.values


        for threshold in [
            0.5,
            0.7,
            0.8
        ]:

            positive=p>=threshold


            rows.append(
                {
                    "subject_id":sid,

                    "max_prob":
                        p.max(),

                    "mean_prob":
                        p.mean(),

                    "positive_windows":
                        positive.sum(),

                    "positive_ratio":
                        positive.mean(),

                    "max_consecutive":
                        longest_run(
                            positive
                        ),

                    "uncertainty":
                        u.mean(),

                    "threshold":
                        threshold
                }
            )


    return pd.DataFrame(rows)



def evaluate_rule(
    feat,
    min_consecutive,
    min_positive,
    uncertainty_limit
):

    alert=(

        (
            feat.max_consecutive
            >=min_consecutive
        )

        |

        (
            feat.positive_windows
            >=min_positive
        )

    ) & (

        feat.uncertainty
        <=uncertainty_limit

    )


    return alert.astype(int)



def load_gt(name):

    if name=="deepbeat":

        df=pd.read_csv(
            "data/features/v6/deepbeat_test_independent_v6_14f.csv"
        )

        df.subject_id=df.subject_id.astype(str)

        return (
            df.groupby(
                "subject_id"
            ).label.max()
        )


    if name=="pulsewatch":

        df=pd.read_csv(
            "experiments/07_v6_multidomain/artifacts/hard_domain_splits/pulsewatch_test.csv"
        )

        df.uid=df.uid.astype(str)

        df.label=(df.label==1).astype(int)

        return (
            df.groupby(
                "uid"
            ).label.max()
        )


    if name=="liu":

        df=pd.read_csv(
            "experiments/07_v6_multidomain/artifacts/hard_domain_splits/liu_test.csv"
        )

        df.subject_id=df.subject_id.astype(str)

        return (
            df.groupby(
                "subject_id"
            ).binary_label_clean.max()
        )



results=[]


for name in DATASETS:


    pred=pd.read_csv(
        ART/f"{name}_v6e_predictions.csv"
    )


    feat=extract_episode_features(
        pred
    )


    gt=load_gt(name)

    gt.index=gt.index.astype(str)



    for c in [
        1,2,3,5
    ]:

        for n in [
            1,3,5,10
        ]:

            for u in [
                0.05,
                0.1,
                0.2
            ]:


                tmp=feat[
                    feat.threshold==0.7
                ].copy()


                tmp["alert"]=evaluate_rule(
                    tmp,
                    c,
                    n,
                    u
                )


                tmp=tmp.groupby(
                    "subject_id"
                ).alert.max()


                gt.index=gt.index.astype(str)

                tmp.index=tmp.index.astype(str)


                merged=pd.concat(
                [
                gt.rename("gt"),
                tmp.rename("pred")
                ],
                axis=1
                ).fillna(0)

                TP=((merged["gt"]==1)&(merged["pred"]==1)).sum()

                TN=((merged["gt"]==0)&(merged["pred"]==0)).sum()

                FP=((merged["gt"]==0)&(merged["pred"]==1)).sum()

                FN=((merged["gt"]==1)&(merged["pred"]==0)).sum()


                results.append(
                    {
                        "dataset":name,
                        "consecutive":c,
                        "positive":n,
                        "uncertainty":u,
                        "TP":TP,
                        "TN":TN,
                        "FP":FP,
                        "FN":FN,
                        "sensitivity":
                            TP/(TP+FN)
                            if TP+FN else 0,
                        "specificity":
                            TN/(TN+FP)
                            if TN+FP else 0
                    }
                )


out=pd.DataFrame(results)


out.to_csv(
    ART/"v6g_episode_decision_results.csv",
    index=False
)


print(out)

