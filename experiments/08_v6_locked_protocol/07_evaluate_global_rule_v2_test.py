from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path("/home/phuc/Documents/HealthSense_rungtamnhi")
ART = ROOT / "experiments/08_v6_locked_protocol/artifacts"


DATASETS = [
    "deepbeat",
    "pulsewatch",
    "liu",
]


rule = json.loads(
    (
        ART /
        "FROZEN_GLOBAL_RULE_V2.json"
    ).read_text()
)


BP = rule[
    "base"
][
    "probability_threshold"
]

BU = rule[
    "base"
][
    "uncertainty_threshold"
]

BK = rule[
    "base"
][
    "min_consecutive"
]


RP = rule[
    "rescue"
][
    "probability_threshold"
]

RU = rule[
    "rescue"
][
    "uncertainty_threshold"
]

RK = rule[
    "rescue"
][
    "min_consecutive"
]


def longest_true_run(values):

    best = 0
    cur = 0

    for value in values:

        if bool(value):

            cur += 1

            best = max(
                best,
                cur
            )

        else:

            cur = 0

    return best


def subject_results(df):

    rows = []


    for subject_id, subject in df.groupby(
        "subject_id",
        sort=False
    ):

        base_max_run = 0
        rescue_max_run = 0


        for _, stream in subject.groupby(
            "stream_id",
            sort=False
        ):

            stream = stream.sort_values(
                "order_in_stream"
            )


            base_candidate = (
                (
                    stream[
                        "meta_probability"
                    ]
                    >= BP
                )
                &
                (
                    stream[
                        "ensemble_std"
                    ]
                    <= BU
                )
            )


            rescue_candidate = (
                (
                    stream[
                        "meta_probability"
                    ]
                    >= RP
                )
                &
                (
                    stream[
                        "ensemble_std"
                    ]
                    <= RU
                )
            )


            base_max_run = max(
                base_max_run,
                longest_true_run(
                    base_candidate
                    .to_numpy()
                )
            )


            rescue_max_run = max(
                rescue_max_run,
                longest_true_run(
                    rescue_candidate
                    .to_numpy()
                )
            )


        base_alert = (
            base_max_run >= BK
        )

        rescue_alert = (
            rescue_max_run >= RK
        )


        alert = int(
            base_alert
            or
            rescue_alert
        )


        rows.append({
            "subject_id":
                str(subject_id),

            "gt":
                int(
                    subject[
                        "label"
                    ].max()
                ),

            "alert":
                alert,

            "base_max_run":
                base_max_run,

            "rescue_max_run":
                rescue_max_run,

            "base_alert":
                int(
                    base_alert
                ),

            "rescue_alert":
                int(
                    rescue_alert
                ),

            "total_windows":
                len(subject),

            "max_probability":
                float(
                    subject[
                        "meta_probability"
                    ].max()
                ),

            "mean_probability":
                float(
                    subject[
                        "meta_probability"
                    ].mean()
                ),

            "mean_uncertainty":
                float(
                    subject[
                        "ensemble_std"
                    ].mean()
                ),
        })


    return pd.DataFrame(
        rows
    )


def metrics(df):

    gt = df["gt"].to_numpy()
    pred = df["alert"].to_numpy()


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


    sens = (
        TP / (TP + FN)
        if TP + FN
        else np.nan
    )

    spec = (
        TN / (TN + FP)
        if TN + FP
        else np.nan
    )

    ppv = (
        TP / (TP + FP)
        if TP + FP
        else np.nan
    )

    npv = (
        TN / (TN + FN)
        if TN + FN
        else np.nan
    )


    total = (
        TP
        + TN
        + FP
        + FN
    )


    return {
        "TP":
            TP,

        "TN":
            TN,

        "FP":
            FP,

        "FN":
            FN,

        "sensitivity":
            sens,

        "specificity":
            spec,

        "PPV":
            ppv,

        "NPV":
            npv,

        "accuracy":
            (
                (TP + TN) / total
                if total
                else np.nan
            ),

        "F1":
            (
                2 * TP
                /
                (
                    2 * TP
                    + FP
                    + FN
                )
                if (
                    2 * TP
                    + FP
                    + FN
                )
                else 0
            ),

        "balanced_accuracy":
            (
                sens + spec
            ) / 2,
    }


summary = []


for dataset in DATASETS:

    df = pd.read_csv(
        ART /
        f"{dataset}_test_verified_streams.csv",
        low_memory=False
    )

    df["subject_id"] = (
        df["subject_id"]
        .astype(str)
    )


    sr = subject_results(
        df
    )


    sr.insert(
        0,
        "dataset",
        dataset
    )


    sr.to_csv(
        ART /
        f"{dataset}_global_rule_v2_subject_results.csv",
        index=False
    )


    m = metrics(
        sr
    )


    m.update({
        "dataset":
            dataset,

        "evaluation":
            "REUSED_TEST_EXPLORATORY",

        "base_probability":
            BP,

        "base_uncertainty":
            BU,

        "base_run":
            BK,

        "rescue_probability":
            RP,

        "rescue_uncertainty":
            RU,

        "rescue_run":
            RK,
    })


    summary.append(
        m
    )


    print(
        "\n" +
        "=" * 100
    )

    print(
        dataset
    )

    print(
        pd.Series(m)
    )

    print(
        "\nSubjects:"
    )

    print(
        sr.to_string(
            index=False
        )
    )


summary = pd.DataFrame(
    summary
)


summary.to_csv(
    ART /
    "global_rule_v2_reused_test_summary.csv",
    index=False
)


print(
    "\n" +
    "=" * 100
)

print(
    "GLOBAL RULE V2 — REUSED TEST"
)

print(
    summary.to_string(
        index=False
    )
)
