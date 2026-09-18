from pathlib import Path
import pandas as pd
import numpy as np


ART=Path(
"experiments/07_v6_multidomain/artifacts"
)


WINDOW_SEC=30


def smooth(x,n):

    if n<=1:
        return x

    return (
        pd.Series(x)
        .rolling(
            n,
            center=True,
            min_periods=1
        )
        .mean()
        .values
    )


def persistence(
    p,
    threshold,
    length
):

    active=p>=threshold

    if length<=1:
        return active.any()


    count=0

    for x in active:

        if x:
            count+=1

            if count>=length:
                return True
        else:
            count=0

    return False



def evaluate(
    name,
    file,
    smooth_n,
    persist_n
):

    df=pd.read_csv(file)


    total_af=0
    detected_af=0

    total_nonaf=0
    false_alert=0


    for sid,g in df.groupby(
        "subject_id"
    ):

        g=g.sort_values(
            "window_index"
        )


        p=smooth(
            g.probability.values,
            smooth_n
        )


        alert=persistence(
            p,
            0.5,
            persist_n
        )


        gt=int(
            g.label.max()
        )


        if gt:
            total_af+=1

            if alert:
                detected_af+=1

        else:

            total_nonaf+=1

            if alert:
                false_alert+=1


    return {

        "dataset":name,
        "smooth":smooth_n,
        "persistence":persist_n,
        "sensitivity":
            detected_af/max(total_af,1),
        "specificity":
            1-false_alert/max(total_nonaf,1),
        "false_alert_subject":
            false_alert
    }



configs=[
    ("A0",1,1),
    ("A1",5,1),
    ("A2",1,7),
    ("A3",5,7),
]


results=[]


for name in [
    "DeepBeat",
    "PulseWatch",
    "LIU"
]:

    file=ART/f"{name.lower()}_v6c_predictions.csv"


    for method,s,p in configs:

        r=evaluate(
            name,
            file,
            s,
            p
        )

        r["method"]=method

        results.append(r)



out=pd.DataFrame(results)


print(out)


out.to_csv(
    ART/"v6d_ablation_results.csv",
    index=False
)


print()
print(
"saved:",
ART/"v6d_ablation_results.csv"
)
