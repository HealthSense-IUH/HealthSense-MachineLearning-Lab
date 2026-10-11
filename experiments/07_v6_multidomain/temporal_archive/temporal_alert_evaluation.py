from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "experiments/07_v6_multidomain/"
    "artifacts/"
    "deepbeat_test_v6a_temporal_predictions.csv"
)


OUTPUT = Path(
    "experiments/07_v6_multidomain/"
    "artifacts/"
    "temporal_alert_results.csv"
)


WINDOW_SECONDS = 25


THRESHOLDS = [
    0.5,
    0.7,
    0.8,
    0.9,
]


WINDOWS = [
    3,
    5,
    7,
]


def evaluate_subject(
    df,
    threshold,
    persistence,
):

    x = df.copy()


    x["positive"] = (
        x.probability
        >= threshold
    )


    alerts = []


    pos = (
        x.positive
        .astype(int)
        .rolling(
            persistence
        )
        .sum()
        >= persistence
    )


    x["alert"] = pos


    # first alert
    if x.alert.any():

        first_alert = (
            x.loc[
                x.alert,
                "timestamp"
            ]
            .iloc[0]
        )

    else:

        first_alert = None


    y_true = (
        x.label
        .iloc[0]
    )


    return {

        "subject_id":
            x.subject_id.iloc[0],

        "label":
            y_true,

        "threshold":
            threshold,

        "persistence":
            persistence,

        "windows":
            len(x),

        "duration_hours":
            (
                len(x)
                *
                WINDOW_SECONDS
                /
                3600
            ),

        "alerts":
            int(
                x.alert.sum()
            ),

        "detected":
            bool(
                x.alert.any()
            ),

        "first_alert":
            first_alert,

    }



df = pd.read_csv(
    INPUT
)


rows = []


for threshold in THRESHOLDS:

    for persistence in WINDOWS:

        for sid, g in (
            df
            .groupby(
                "subject_id"
            )
        ):

            rows.append(
                evaluate_subject(
                    g,
                    threshold,
                    persistence,
                )
            )


result = pd.DataFrame(
    rows
)


result.to_csv(
    OUTPUT,
    index=False,
)


print("="*100)
print("TEMPORAL ALERT RESULTS")
print("="*100)


for (
    threshold,
    persistence
), g in (
    result
    .groupby(
        [
            "threshold",
            "persistence",
        ]
    )
):

    af = g[
        g.label == 1
    ]

    non = g[
        g.label == 0
    ]


    sensitivity = (
        af.detected.mean()
        if len(af)
        else np.nan
    )


    false_alarm_rate = (
        non.detected.mean()
        if len(non)
        else np.nan
    )


    print(
        f"T={threshold}, "
        f"P={persistence}"
    )

    print(
        "  AF sensitivity:",
        sensitivity
    )

    print(
        "  non-AF false alert:",
        false_alarm_rate
    )


print()
print(
    "saved:",
    OUTPUT
)
