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
        "FROZEN_GLOBAL_RULE.json"
    ).read_text()
)


P = rule["probability_threshold"]
U = rule["uncertainty_threshold"]
K = rule["min_consecutive"]
N = rule["min_positive"]
LOGIC = rule["logic"]


print("FROZEN RULE")
print(json.dumps(rule, indent=2))


def longest_true_run(x):
    best = 0
    cur = 0

    for v in x:
        if bool(v):
            cur += 1
            best = max(
                best,
                cur
            )
        else:
            cur = 0

    return best


def evaluate_subjects(df):
    df = df.copy()

    df["eligible"] = (
        (df["meta_probability"] >= P)
        &
        (df["ensemble_std"] <= U)
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

            max_run = max(
                max_run,
                longest_true_run(
                    stream[
                        "eligible"
                    ].to_numpy()
                )
            )


        positive = int(
            sg["eligible"].sum()
        )

        a = (
            max_run >= K
        )

        b = (
            positive >= N
        )


        if LOGIC == "OR":
            alert = int(
                a or b
            )

        elif LOGIC == "AND":
            alert = int(
                a and b
            )

        else:
            raise RuntimeError(
                LOGIC
            )


        rows.append({
            "subject_id":
                str(subject_id),

            "gt":
                int(
                    sg["label"].max()
                ),

            "alert":
                alert,

            "eligible_windows":
                positive,

            "max_consecutive":
                max_run,

            "total_windows":
                len(sg),

            "max_probability":
                float(
                    sg[
                        "meta_probability"
                    ].max()
                ),

            "mean_probability":
                float(
                    sg[
                        "meta_probability"
                    ].mean()
                ),

            "mean_uncertainty":
                float(
                    sg[
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

    acc = (
        (TP + TN)
        /
        (TP + TN + FP + FN)
    )

    f1 = (
        2 * TP
        /
        (2 * TP + FP + FN)
        if 2 * TP + FP + FN
        else 0
    )

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "sensitivity": sens,
        "specificity": spec,
        "PPV": ppv,
        "NPV": npv,
        "accuracy": acc,
        "F1": f1,
        "balanced_accuracy":
            (sens + spec) / 2,
    }


summary = []


for dataset in DATASETS:

    f = (
        ART /
        f"{dataset}_test_verified_streams.csv"
    )

    df = pd.read_csv(
        f,
        low_memory=False
    )

    df["subject_id"] = (
        df["subject_id"]
        .astype(str)
    )

    subjects = evaluate_subjects(
        df
    )

    subjects.insert(
        0,
        "dataset",
        dataset
    )


    subjects.to_csv(
        ART /
        f"{dataset}_frozen_rule_subject_results.csv",
        index=False
    )


    m = metrics(subjects)

    m.update({
        "dataset":
            dataset,

        "evaluation":
            "REUSED_SUBJECT_HELD_OUT_TEST",

        "probability_threshold":
            P,

        "uncertainty_threshold":
            U,

        "min_consecutive":
            K,

        "min_positive":
            N,

        "logic":
            LOGIC,
    })

    summary.append(m)


    print("\n" + "=" * 100)
    print(dataset)
    print(pd.Series(m))
    print("\nSubject results:")
    print(
        subjects.to_string(
            index=False
        )
    )


summary = pd.DataFrame(
    summary
)

summary.to_csv(
    ART /
    "frozen_global_rule_reused_test_summary.csv",
    index=False
)


print("\n" + "=" * 100)
print("REUSED TEST SUMMARY")
print(
    summary.to_string(
        index=False
    )
)
