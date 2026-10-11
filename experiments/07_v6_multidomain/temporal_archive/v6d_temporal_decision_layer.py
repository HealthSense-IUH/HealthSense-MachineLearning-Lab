from pathlib import Path
import joblib
import numpy as np
import pandas as pd


from sklearn.metrics import (
    confusion_matrix,
)


MODEL = (
    "models/multidomain/"
    "healthsense_af_v6c_stacking.pkl"
)


FEATURES=[
"HR_mean",
"Mean_NN",
"SDNN",
"RMSSD",
"NN50",
"pNN50",
"CV",
"HF",
"Total_Power",
"HF_norm",
"SD1",
"SD2",
"SampEn",
"PPG_AC",
]


OUT = Path(
    "experiments/07_v6_multidomain/artifacts/"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)



# ============================================================
# Temporal functions
# ============================================================

def moving_average(
    x,
    k
):
    return (
        pd.Series(x)
        .rolling(
            k,
            min_periods=1
        )
        .mean()
        .values
    )



def persistence_alert(
    prob,
    threshold,
    persistence
):

    high = prob >= threshold

    result=[]

    for i in range(len(high)):

        if i+1 < persistence:
            result.append(False)
        else:
            result.append(
                high[
                    i+1-persistence:i+1
                ].all()
            )

    return np.array(result)



def evaluate_subjects(
    df,
    threshold,
    persistence,
    smooth
):

    subject_results=[]


    for sid,g in df.groupby(
        "subject_id"
    ):

        if "timestamp" in g.columns:
            g = g.sort_values(
                "timestamp"
            )
        elif "window_index" in g.columns:
            g = g.sort_values(
                "window_index"
            )
        else:
            raise ValueError(
                "No temporal ordering column found"
            )


        p=moving_average(
            g.probability.values,
            smooth
        )


        alert=persistence_alert(
            p,
            threshold,
            persistence
        )


        subject_results.append(
            {
                "subject_id":sid,
                "label":g.label.iloc[0],
                "alert":alert.any(),
                "windows":len(g),
                "positive_windows":alert.sum()
            }
        )


    r=pd.DataFrame(
        subject_results
    )


    tn,fp,fn,tp=confusion_matrix(
        r.label,
        r.alert
    ).ravel()


    return {
        "threshold":threshold,
        "persistence":persistence,
        "smooth":smooth,
        "subjects":len(r),
        "TP":tp,
        "TN":tn,
        "FP":fp,
        "FN":fn,
        "sensitivity":tp/(tp+fn)
            if tp+fn else 0,
        "specificity":tn/(tn+fp)
            if tn+fp else 0,
        "false_alert_rate":fp/(tn+fp)
            if tn+fp else 0
    }



# ============================================================
# Load predictions
# ============================================================

files={

"DeepBeat":
"experiments/07_v6_multidomain/artifacts/"
"deepbeat_v6c_predictions.csv",

"PulseWatch":
"experiments/07_v6_multidomain/artifacts/"
"pulsewatch_v6c_predictions.csv",

"LIU":
"experiments/07_v6_multidomain/artifacts/"
"liu_v6c_predictions.csv"

}



all_results=[]


for name,path in files.items():

    print()
    print("="*80)
    print(name)
    print("="*80)


    df=pd.read_csv(path)


    for threshold in [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9
    ]:

        for persistence in [
            3,
            5,
            7,
            10
        ]:

            for smooth in [
                1,
                3,
                5
            ]:


                r=evaluate_subjects(
                    df,
                    threshold,
                    persistence,
                    smooth
                )

                r["dataset"]=name

                all_results.append(
                    r
                )



result=pd.DataFrame(
    all_results
)


print(
result.sort_values(
    [
        "dataset",
        "false_alert_rate",
        "sensitivity"
    ],
    ascending=[
        True,
        True,
        False
    ]
)
.head(30)
)


result.to_csv(
    OUT /
    "v6d_temporal_decision_results.csv",
    index=False
)


print()
print(
"saved:",
OUT /
"v6d_temporal_decision_results.csv"
)
