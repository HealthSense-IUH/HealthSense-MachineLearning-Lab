from pathlib import Path
import hashlib
import json
import itertools

import numpy as np
import pandas as pd


ROOT = Path("/home/phuc/Documents/HealthSense_rungtamnhi")
ART = ROOT / "experiments/08_v6_locked_protocol/artifacts"


DATASETS = [
    "deepbeat",
    "pulsewatch",
    "liu",
]


PROB_THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]

UNC_THRESHOLDS = [
    0.05,
    0.10,
    0.20,
    1.00,   # practically no uncertainty filtering
]

MIN_CONSECUTIVE = [
    1,
    2,
    3,
    5,
]

MIN_POSITIVE = [
    1,
    2,
    3,
    5,
    10,
]

LOGICS = [
    "OR",
    "AND",
]


def longest_true_run(x):
    best = 0
    current = 0

    for value in x:
        if bool(value):
            current += 1
            best = max(
                best,
                current
            )
        else:
            current = 0

    return best


def subject_features(
    df,
    probability_threshold,
    uncertainty_threshold
):
    df = df.copy()

    df["eligible"] = (
        (
            df["meta_probability"]
            >= probability_threshold
        )
        &
        (
            df["ensemble_std"]
            <= uncertainty_threshold
        )
    )

    rows = []

    for subject_id, sg in df.groupby(
        "subject_id",
        sort=False
    ):

        max_run = 0

        for _, stream in sg.groupby(
            "stream_id",
            sort=False
        ):
            stream = stream.sort_values(
                "order_in_stream"
            )

            run = longest_true_run(
                stream["eligible"].to_numpy()
            )

            max_run = max(
                max_run,
                run
            )

        rows.append({
            "subject_id":
                str(subject_id),

            "gt":
                int(
                    sg["label"].max()
                ),

            "eligible_windows":
                int(
                    sg["eligible"].sum()
                ),

            "max_consecutive":
                int(max_run),

            "total_windows":
                len(sg),

            "eligible_ratio":
                float(
                    sg["eligible"].mean()
                ),
        })

    return pd.DataFrame(rows)


def metrics(gt, pred):
    gt = np.asarray(gt)
    pred = np.asarray(pred)

    TP = int(
        ((gt == 1) & (pred == 1)).sum()
    )

    TN = int(
        ((gt == 0) & (pred == 0)).sum()
    )

    FP = int(
        ((gt == 0) & (pred == 1)).sum()
    )

    FN = int(
        ((gt == 1) & (pred == 0)).sum()
    )

    sensitivity = (
        TP / (TP + FN)
        if TP + FN
        else np.nan
    )

    specificity = (
        TN / (TN + FP)
        if TN + FP
        else np.nan
    )

    balanced_accuracy = (
        (sensitivity + specificity) / 2
    )

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "sensitivity":
            sensitivity,
        "specificity":
            specificity,
        "balanced_accuracy":
            balanced_accuracy,
    }


dev = {}

for dataset in DATASETS:

    f = (
        ART /
        f"{dataset}_dev_verified_streams.csv"
    )

    x = pd.read_csv(
        f,
        low_memory=False
    )

    x["subject_id"] = (
        x["subject_id"].astype(str)
    )

    dev[dataset] = x

    gt = (
        x.groupby("subject_id")["label"]
        .max()
    )

    print(
        dataset,
        "subjects=",
        len(gt),
        "AF subjects=",
        int(gt.sum()),
        "non-AF subjects=",
        int((gt == 0).sum())
    )


all_rows = []


for (
    probability_threshold,
    uncertainty_threshold
) in itertools.product(
    PROB_THRESHOLDS,
    UNC_THRESHOLDS
):

    prepared = {}

    for dataset in DATASETS:
        prepared[dataset] = (
            subject_features(
                dev[dataset],
                probability_threshold,
                uncertainty_threshold
            )
        )


    for (
        min_consecutive,
        min_positive,
        logic
    ) in itertools.product(
        MIN_CONSECUTIVE,
        MIN_POSITIVE,
        LOGICS
    ):

        dataset_metrics = []

        for dataset in DATASETS:

            sf = prepared[dataset]

            persistent = (
                sf["max_consecutive"]
                >= min_consecutive
            )

            repeated = (
                sf["eligible_windows"]
                >= min_positive
            )

            if logic == "OR":
                pred = (
                    persistent
                    |
                    repeated
                ).astype(int)

            elif logic == "AND":
                pred = (
                    persistent
                    &
                    repeated
                ).astype(int)

            else:
                raise RuntimeError(
                    logic
                )


            m = metrics(
                sf["gt"],
                pred
            )

            dataset_metrics.append({
                "dataset":
                    dataset,
                **m
            })


        md = pd.DataFrame(
            dataset_metrics
        )


        row = {
            "probability_threshold":
                probability_threshold,

            "uncertainty_threshold":
                uncertainty_threshold,

            "min_consecutive":
                min_consecutive,

            "min_positive":
                min_positive,

            "logic":
                logic,

            "macro_sensitivity":
                md["sensitivity"].mean(),

            "macro_specificity":
                md["specificity"].mean(),

            "macro_balanced_accuracy":
                md[
                    "balanced_accuracy"
                ].mean(),

            "min_dataset_sensitivity":
                md["sensitivity"].min(),

            "min_dataset_specificity":
                md["specificity"].min(),
        }


        for _, r in md.iterrows():

            d = r["dataset"]

            for metric in [
                "TP",
                "TN",
                "FP",
                "FN",
                "sensitivity",
                "specificity",
                "balanced_accuracy",
            ]:
                row[
                    f"{d}_{metric}"
                ] = r[metric]


        all_rows.append(row)


results = pd.DataFrame(
    all_rows
)


results.to_csv(
    ART / "dev_global_rule_grid.csv",
    index=False
)


# ------------------------------------------------------------
# PREDECLARED SELECTION
#
# Screening constraint:
#   sensitivity >= 0.80 on EVERY development domain.
#
# Among eligible rules:
#   1. maximize macro balanced accuracy
#   2. maximize macro specificity
#   3. maximize minimum dataset sensitivity
#   4. maximize macro sensitivity
#
# TEST IS NOT USED.
# ------------------------------------------------------------

eligible = results[
    results[
        "min_dataset_sensitivity"
    ] >= 0.80
].copy()


if len(eligible) == 0:

    print(
        "WARNING: no rule satisfied "
        "minimum sensitivity >= 0.80."
    )

    eligible = results.copy()


eligible = eligible.sort_values(
    [
        "macro_balanced_accuracy",
        "macro_specificity",
        "min_dataset_sensitivity",
        "macro_sensitivity",
    ],
    ascending=[
        False,
        False,
        False,
        False,
    ]
)


winner = eligible.iloc[0]


print("\n" + "=" * 100)
print("TOP 20 DEV RULES")

cols = [
    "probability_threshold",
    "uncertainty_threshold",
    "min_consecutive",
    "min_positive",
    "logic",
    "macro_sensitivity",
    "macro_specificity",
    "macro_balanced_accuracy",
    "min_dataset_sensitivity",
]

print(
    eligible[
        cols
    ]
    .head(20)
    .to_string(index=False)
)


print("\n" + "=" * 100)
print("SELECTED GLOBAL RULE")

print(
    winner.to_string()
)


rule = {
    "version":
        "v6_locked_global_rule_1",

    "probability_threshold":
        float(
            winner[
                "probability_threshold"
            ]
        ),

    "uncertainty_threshold":
        float(
            winner[
                "uncertainty_threshold"
            ]
        ),

    "min_consecutive":
        int(
            winner[
                "min_consecutive"
            ]
        ),

    "min_positive":
        int(
            winner[
                "min_positive"
            ]
        ),

    "logic":
        winner["logic"],

    "selection_dataset":
        "DEV_ONLY",

    "test_used_for_selection":
        False,

    "selection_constraint":
        (
            "subject-level sensitivity >= 0.80 "
            "on every development dataset"
        ),

    "selection_objective":
        (
            "maximize macro balanced accuracy; "
            "then macro specificity; "
            "then minimum dataset sensitivity; "
            "then macro sensitivity"
        ),

    "development_metrics": {
        "macro_sensitivity":
            float(
                winner[
                    "macro_sensitivity"
                ]
            ),

        "macro_specificity":
            float(
                winner[
                    "macro_specificity"
                ]
            ),

        "macro_balanced_accuracy":
            float(
                winner[
                    "macro_balanced_accuracy"
                ]
            ),

        "min_dataset_sensitivity":
            float(
                winner[
                    "min_dataset_sensitivity"
                ]
            ),
    }
}


rule_path = (
    ART /
    "FROZEN_GLOBAL_RULE.json"
)


rule_path.write_text(
    json.dumps(
        rule,
        indent=2
    )
)


digest = hashlib.sha256(
    rule_path.read_bytes()
).hexdigest()


(
    ART /
    "FROZEN_GLOBAL_RULE_SHA256.txt"
).write_text(
    digest
    + "  "
    + rule_path.name
    + "\n"
)


print("\nFrozen:")
print(rule_path)
print("SHA256:", digest)
