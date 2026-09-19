from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import beta


ROOT = Path(
    "/home/phuc/Documents/HealthSense_rungtamnhi"
)

ART = (
    ROOT /
    "experiments/08_v6_locked_protocol/artifacts"
)


DATASETS = [
    "deepbeat",
    "pulsewatch",
    "liu",
]


ALPHA = 0.05


def wilson_ci(
    success,
    total,
    z=1.959963984540054
):

    if total == 0:
        return (
            np.nan,
            np.nan
        )


    p = (
        success
        /
        total
    )


    denom = (
        1
        +
        z**2
        /
        total
    )


    center = (
        p
        +
        z**2
        /
        (
            2
            *
            total
        )
    )


    radius = (
        z
        *
        np.sqrt(
            p
            *
            (
                1
                -
                p
            )
            /
            total
            +
            z**2
            /
            (
                4
                *
                total**2
            )
        )
    )


    low = (
        center
        -
        radius
    ) / denom


    high = (
        center
        +
        radius
    ) / denom


    return (
        float(low),
        float(high)
    )


def clopper_pearson(
    success,
    total,
    alpha=0.05
):

    if total == 0:
        return (
            np.nan,
            np.nan
        )


    if success == 0:

        low = 0.0

    else:

        low = beta.ppf(
            alpha / 2,
            success,
            total
            -
            success
            +
            1
        )


    if success == total:

        high = 1.0

    else:

        high = beta.ppf(
            1
            -
            alpha / 2,
            success
            +
            1,
            total
            -
            success
        )


    return (
        float(low),
        float(high)
    )


rows = []


for dataset in DATASETS:

    f = (
        ART /
        f"{dataset}_global_rule_v2_subject_results.csv"
    )


    df = pd.read_csv(
        f
    )


    gt = (
        df["gt"]
        .astype(int)
        .to_numpy()
    )

    pred = (
        df["alert"]
        .astype(int)
        .to_numpy()
    )


    TP = int(
        (
            (gt == 1)
            &
            (pred == 1)
        ).sum()
    )

    TN = int(
        (
            (gt == 0)
            &
            (pred == 0)
        ).sum()
    )

    FP = int(
        (
            (gt == 0)
            &
            (pred == 1)
        ).sum()
    )

    FN = int(
        (
            (gt == 1)
            &
            (pred == 0)
        ).sum()
    )


    measures = {
        "Sensitivity": (
            TP,
            TP + FN
        ),

        "Specificity": (
            TN,
            TN + FP
        ),

        "PPV": (
            TP,
            TP + FP
        ),

        "NPV": (
            TN,
            TN + FN
        ),
    }


    for metric, (
        success,
        total
    ) in measures.items():

        point = (
            success / total
            if total
            else np.nan
        )


        w_low, w_high = (
            wilson_ci(
                success,
                total
            )
        )


        cp_low, cp_high = (
            clopper_pearson(
                success,
                total,
                ALPHA
            )
        )


        rows.append({
            "dataset":
                dataset,

            "metric":
                metric,

            "success":
                success,

            "total":
                total,

            "point_estimate":
                point,

            "wilson_95_low":
                w_low,

            "wilson_95_high":
                w_high,

            "exact_95_low":
                cp_low,

            "exact_95_high":
                cp_high,

            "evaluation":
                "GLOBAL_RULE_V2_REUSED_TEST_EXPLORATORY",
        })


out = pd.DataFrame(
    rows
)


out.to_csv(
    ART /
    "global_rule_v2_exact_subject_ci.csv",
    index=False
)


print(
    out.to_string(
        index=False
    )
)
