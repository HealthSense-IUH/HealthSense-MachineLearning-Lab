from pathlib import Path
import pandas as pd
import numpy as np


ART=Path(
    "experiments/07_v6_multidomain/artifacts"
)


WINDOW_SEC=30


def moving_average(x,n):

    if n<=1:
        return x

    return np.convolve(
        x,
        np.ones(n)/n,
        mode="same"
    )



def persistence_alert(
    x,
    threshold,
    persistence
):

    count=0
    alert=[]

    for v in x:

        if v>=threshold:
            count+=1
        else:
            count=0

        alert.append(
            count>=persistence
        )

    return np.array(alert)



def evaluate(
    df,
    method,
    alpha,
    smooth,
    threshold,
    persistence
):

    results=[]


    for sid,g in df.groupby(
        "subject_id"
    ):

        if "window_index" in g.columns:
            g=g.sort_values(
                "window_index"
            )


        p=g.meta_probability.values
        u=g.ensemble_std.values


        if method=="baseline":

            score=p


        elif method=="v6d":

            score=moving_average(
                p,
                smooth
            )


        elif method=="v6f":

            risk=p*(1-alpha*u)

            score=moving_average(
                risk,
                smooth
            )


        alert=persistence_alert(
            score,
            threshold,
            persistence
        )


        results.append(
            {
                "subject_id":sid,
                "positive_windows":alert.sum(),
                "alert":int(alert.any()),
                "method":method
            }
        )


    return pd.DataFrame(results)



configs=[]


for method in [
    "baseline",
    "v6d",
    "v6f"
]:

    for smooth in [
        1,
        3,
        5
    ]:

        for persistence in [
            3,
            5,
            7
        ]:

            configs.append(
                (
                    method,
                    smooth,
                    persistence
                )
            )



all_results=[]


for dataset in [
    "deepbeat",
    "pulsewatch",
    "liu"
]:

    print("="*80)
    print(dataset)


    df=pd.read_csv(
        ART/f"{dataset}_v6e_predictions.csv"
    )


    # normalize id
    if "uid" in df.columns:
        df["subject_id"]=df.uid


    for method,smooth,persistence in configs:

        r=evaluate(
            df,
            method,
            alpha=1.0,
            smooth=smooth,
            threshold=0.5,
            persistence=persistence
        )


        r["dataset"]=dataset
        r["smooth"]=smooth
        r["persistence"]=persistence


        all_results.append(r)



result=pd.concat(
    all_results,
    ignore_index=True
)


result.to_csv(
    ART/"v6f_uncertainty_temporal_subject_results.csv",
    index=False
)


print(result.head())

print(
"saved:",
ART/"v6f_uncertainty_temporal_subject_results.csv"
)
