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
    "temporal_alert_v2_results.csv"
)


WINDOW_SECONDS = 25


THRESHOLDS = [
    0.5,
    0.7,
    0.8,
    0.9,
]


PERSISTENCES = [
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

    x = x.sort_values(
        [
            "timestamp",
            "global_index",
        ]
    )


    x["positive"] = (
        x.probability
        >= threshold
    )


    x["alert"] = (
        x["positive"]
        .astype(int)
        .rolling(
            persistence,
            min_periods=persistence,
        )
        .sum()
        >= persistence
    )


    signal_hours = (
        len(x)
        *
        WINDOW_SECONDS
        /
        3600
    )


    alerts = int(
        x.alert.sum()
    )


    window_positive = int(
        x.positive.sum()
    )


    if x.label.iloc[0] == 1:

        detected = bool(
            x.alert.any()
        )

    else:

        detected = False


    return {

        "subject_id":
            x.subject_id.iloc[0],

        "label":
            int(
                x.label.iloc[0]
            ),

        "threshold":
            threshold,

        "persistence":
            persistence,

        "windows":
            len(x),

        "signal_hours":
            signal_hours,

        "window_positive":
            window_positive,

        "window_positive_rate":
            (
                window_positive
                /
                len(x)
            ),

        "alerts":
            alerts,

        "detected":
            detected,

    }



df = pd.read_csv(
    INPUT
)


rows=[]


for threshold in THRESHOLDS:

    for persistence in PERSISTENCES:

        for sid,g in df.groupby(
            "subject_id"
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


# ============================================================
# AGGREGATE
# ============================================================

summary=[]


for (
    threshold,
    persistence
),g in result.groupby(
    [
        "threshold",
        "persistence",
    ]
):


    af = g[
        g.label==1
    ]


    non = g[
        g.label==0
    ]


    af_sensitivity = (
        af.detected.mean()
    )


    non_false_subject = (
        (~non.detected)
        .count()
    )


    false_alerts = (
        non.alerts.sum()
    )


    non_hours = (
        non.signal_hours.sum()
    )


    far_hour = (
        false_alerts
        /
        non_hours
        if non_hours>0
        else np.nan
    )


    window_fp = (
        non.window_positive.sum()
    )


    window_total = (
        non.windows.sum()
    )


    summary.append({

        "threshold":
            threshold,

        "persistence":
            persistence,


        "AF_subject_sensitivity":
            af_sensitivity,


        "NON_AF_subject_false_alert_rate":
            (
                non.detected.mean()
            ),


        "false_alerts":
            false_alerts,


        "non_af_signal_hours":
            non_hours,


        "false_alarm_per_hour":
            far_hour,


        "NON_AF_window_FPR":
            (
                window_fp
                /
                window_total
            ),

    })


summary = pd.DataFrame(
    summary
)


summary.to_csv(
    OUTPUT,
    index=False,
)


print("="*100)
print(
    "TEMPORAL ALERT V2 SUMMARY"
)
print("="*100)


print(
    summary.to_string(
        index=False
    )
)


print()

print(
    "Saved:",
    OUTPUT
)
