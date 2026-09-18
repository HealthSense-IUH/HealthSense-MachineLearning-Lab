from pathlib import Path
import pandas as pd
import numpy as np


ARTIFACTS = Path(
    "experiments/07_v6_multidomain/artifacts"
)


WINDOW_SEC = 30


def moving_average(
    x,
    n=5
):
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


def detect_episode(
    prob,
    threshold=0.5,
    persistence=7
):

    active = (
        prob >= threshold
    )

    episodes=[]

    start=None
    count=0

    for i,v in enumerate(active):

        if v:
            count+=1

            if count>=persistence and start is None:
                start=i-persistence+1

        else:

            if start is not None:
                episodes.append(
                    (
                        start,
                        i-1
                    )
                )

            start=None
            count=0


    if start is not None:
        episodes.append(
            (
                start,
                len(active)-1
            )
        )

    return episodes



def analyze_dataset(
    name,
    file
):

    df=pd.read_csv(file)

    print()
    print("="*100)
    print(name)

    results=[]


    for sid,g in df.groupby(
        "subject_id"
    ):

        g=g.sort_values(
            "window_index"
        )


        p=moving_average(
            g.probability.values,
            5
        )


        episodes=detect_episode(
            p,
            threshold=0.5,
            persistence=7
        )


        gt=int(
            g.label.max()
        )


        duration=len(g)*WINDOW_SEC/3600


        af_windows=(
            g.label==1
        ).sum()


        results.append(
            {
                "subject_id":sid,
                "gt_af":gt,
                "episodes":len(episodes),
                "recording_hours":duration,
                "af_burden":
                    af_windows/len(g),
                "detected":
                    int(len(episodes)>0)
            }
        )


    out=pd.DataFrame(results)


    print(out)


    print()

    print(
        "Sensitivity:",
        (
            out[
                out.gt_af==1
            ].detected.mean()
        )
    )

    print(
        "False alert subjects:",
        (
            out[
                (out.gt_af==0)
                &
                (out.detected==1)
            ].shape[0]
        )
    )


    out.to_csv(
        ARTIFACTS /
        f"{name.lower()}_episode_analysis.csv",
        index=False
    )



for name in [
    "DeepBeat",
    "PulseWatch",
    "LIU"
]:

    analyze_dataset(
        name,
        ARTIFACTS /
        f"{name.lower()}_v6c_predictions.csv"
    )


print()
print("DONE")
