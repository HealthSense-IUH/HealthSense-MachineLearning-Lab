from pathlib import Path

import pandas as pd
import numpy as np


INPUT = Path(
    "experiments/07_v6_multidomain/"
    "artifacts/"
    "deepbeat_test_v6a_temporal_predictions.csv"
)


WINDOW_SECONDS = 25


THRESHOLD = 0.8
PERSISTENCE = 3


df = pd.read_csv(INPUT)


def evaluate(
    name,
    data,
):

    rows=[]

    for sid,g in data.groupby(
        "subject_id"
    ):

        g = (
            g.sort_values(
                [
                    "timestamp",
                    "global_index",
                ]
            )
        )

        positive = (
            g.probability
            >= THRESHOLD
        )

        alert = (
            positive
            .astype(int)
            .rolling(
                PERSISTENCE,
                min_periods=PERSISTENCE,
            )
            .sum()
            >= PERSISTENCE
        )


        rows.append(
            {
                "subject_id":
                    sid,

                "label":
                    g.label.iloc[0],

                "windows":
                    len(g),

                "signal_hours":
                    (
                        len(g)
                        *
                        WINDOW_SECONDS
                        /
                        3600
                    ),

                "alert":
                    bool(
                        alert.any()
                    ),

                "positive_windows":
                    int(
                        positive.sum()
                    ),
            }
        )


    r=pd.DataFrame(rows)


    af=r[
        r.label==1
    ]

    non=r[
        r.label==0
    ]


    return {

        "strategy":
            name,

        "windows":
            len(data),

        "coverage":
            len(data)
            /
            len(df),

        "AF_subject_sensitivity":
            af.alert.mean(),

        "NON_AF_false_alert_subject_rate":
            non.alert.mean(),

        "NON_AF_false_alerts":
            non.alert.sum(),

        "NON_AF_hours":
            non.signal_hours.sum(),

        "false_alarm_hour":
            (
                non.alert.sum()
                /
                non.signal_hours.sum()
            ),

        "AF_windows_retained":
            af.windows.sum(),

        "AF_window_retention":
            (
                af.windows.sum()
                /
                df[
                    df.label==1
                ].shape[0]
            ),

    }



results=[]


# no SQI
results.append(
    evaluate(
        "ALL_WINDOWS",
        df,
    )
)


# excellent only
results.append(
    evaluate(
        "EXCELLENT_ONLY",
        df[
            df.quality_name=="excellent"
        ],
    )
)


# acceptable + excellent
results.append(
    evaluate(
        "EXCELLENT_ACCEPTABLE",
        df[
            df.quality_name.isin(
                [
                    "excellent",
                    "acceptable",
                ]
            )
        ],
    )
)


out=pd.DataFrame(
    results
)


print(
    out.to_string(
        index=False
    )
)


out.to_csv(
    "experiments/07_v6_multidomain/artifacts/"
    "sqi_temporal_ablation.csv",
    index=False,
)
