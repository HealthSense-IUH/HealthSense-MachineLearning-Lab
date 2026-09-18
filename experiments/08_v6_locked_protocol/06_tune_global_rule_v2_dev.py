from pathlib import Path
import hashlib
import itertools
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


# ------------------------------------------------------------
# BASE branch:
# lower probability, requires longer persistence.
# ------------------------------------------------------------

BASE_P = [
    0.50,
    0.60,
    0.70,
    0.80,
]

BASE_U = [
    0.05,
    0.10,
    0.20,
    1.00,
]

BASE_RUN = [
    2,
    3,
    5,
]


# ------------------------------------------------------------
# RESCUE branch:
# very high probability, allows shorter AF episode.
# Still requires >= 2 consecutive windows.
# ------------------------------------------------------------

RESCUE_P = [
    0.85,
    0.90,
    0.95,
]

RESCUE_U = [
    0.05,
    0.10,
    0.20,
]

RESCUE_RUN = [
    2,
    3,
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


def build_run_table(
    df,
    probability_threshold,
    uncertainty_threshold
):

    x = df.copy()

    x["candidate"] = (
        (
            x["meta_probability"]
            >= probability_threshold
        )
        &
        (
            x["ensemble_std"]
            <= uncertainty_threshold
        )
    )


    rows = []


    for subject_id, subject in x.groupby(
        "subject_id",
        sort=False
    ):

        max_run = 0
        candidate_windows = 0


        for _, stream in subject.groupby(
            "stream_id",
            sort=False
        ):

            stream = stream.sort_values(
                "order_in_stream"
            )

            values = (
                stream["candidate"]
                .to_numpy()
            )

            candidate_windows += int(
                values.sum()
            )

            max_run = max(
                max_run,
                longest_true_run(
                    values
                )
            )


        rows.append({
            "subject_id":
                str(subject_id),

            "gt":
                int(
                    subject["label"].max()
                ),

            "max_run":
                int(max_run),

            "candidate_windows":
                int(candidate_windows),

            "total_windows":
                len(subject),
        })


    return pd.DataFrame(rows)


def confusion(gt, pred):

    gt = np.asarray(
        gt,
        dtype=int
    )

    pred = np.asarray(
        pred,
        dtype=int
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


    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,

        "sensitivity":
            sens,

        "specificity":
            spec,

        "balanced_accuracy":
            (sens + spec) / 2,
    }


# ============================================================
# Load DEV ONLY
# ============================================================

dev = {}


for dataset in DATASETS:

    f = (
        ART /
        f"{dataset}_dev_verified_streams.csv"
    )

    df = pd.read_csv(
        f,
        low_memory=False
    )

    df["subject_id"] = (
        df["subject_id"]
        .astype(str)
    )

    dev[dataset] = df


    gt = (
        df.groupby(
            "subject_id"
        )["label"]
        .max()
    )


    print(
        dataset,
        "subjects=",
        len(gt),
        "AF=",
        int(gt.sum()),
        "nonAF=",
        int((gt == 0).sum())
    )


# ============================================================
# Precompute max-run tables.
# ============================================================

threshold_pairs = set()


for p, u in itertools.product(
    BASE_P,
    BASE_U
):

    threshold_pairs.add(
        (p, u)
    )


for p, u in itertools.product(
    RESCUE_P,
    RESCUE_U
):

    threshold_pairs.add(
        (p, u)
    )


cache = {}


for dataset in DATASETS:

    for p, u in sorted(
        threshold_pairs
    ):

        cache[
            (
                dataset,
                p,
                u
            )
        ] = build_run_table(
            dev[dataset],
            p,
            u
        )


# ============================================================
# Grid
#
# Alert if:
#
#   BASE persistent episode
#          OR
#   HIGH-CONFIDENCE short episode
#
# No subject-wide scattered-window counting.
# ============================================================

rows = []


for (
    base_p,
    base_u,
    base_run,
    rescue_p,
    rescue_u,
    rescue_run
) in itertools.product(
    BASE_P,
    BASE_U,
    BASE_RUN,
    RESCUE_P,
    RESCUE_U,
    RESCUE_RUN
):

    dataset_metrics = []


    for dataset in DATASETS:

        base = cache[
            (
                dataset,
                base_p,
                base_u
            )
        ].copy()


        rescue = cache[
            (
                dataset,
                rescue_p,
                rescue_u
            )
        ][
            [
                "subject_id",
                "max_run"
            ]
        ].rename(
            columns={
                "max_run":
                    "rescue_max_run"
            }
        )


        merged = base.merge(
            rescue,
            on="subject_id",
            how="inner",
            validate="one_to_one"
        )


        base_alert = (
            merged["max_run"]
            >= base_run
        )


        rescue_alert = (
            merged["rescue_max_run"]
            >= rescue_run
        )


        pred = (
            base_alert
            |
            rescue_alert
        ).astype(int)


        m = confusion(
            merged["gt"],
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
        "base_probability":
            base_p,

        "base_uncertainty":
            base_u,

        "base_run":
            base_run,

        "rescue_probability":
            rescue_p,

        "rescue_uncertainty":
            rescue_u,

        "rescue_run":
            rescue_run,

        "macro_sensitivity":
            md[
                "sensitivity"
            ].mean(),

        "macro_specificity":
            md[
                "specificity"
            ].mean(),

        "macro_balanced_accuracy":
            md[
                "balanced_accuracy"
            ].mean(),

        "min_dataset_sensitivity":
            md[
                "sensitivity"
            ].min(),

        "min_dataset_specificity":
            md[
                "specificity"
            ].min(),
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


    rows.append(row)


grid = pd.DataFrame(
    rows
)


grid.to_csv(
    ART /
    "dev_global_rule_v2_grid.csv",
    index=False
)


# ============================================================
# Selection rule
#
# DEV ONLY.
#
# Tiny development cohorts mean:
#
# DeepBeat: 4 AF subjects
# Liu:      4 AF subjects
#
# Requiring >= 0.80 per dataset forces 4/4 = 1.00.
#
# V2 therefore permits at most one miss in such a dataset:
#
# min dataset sensitivity >= 0.75
#
# while requiring:
#
# macro sensitivity >= 0.85
#
# Then maximize balanced accuracy / specificity.
# ============================================================

eligible = grid[
    (
        grid[
            "min_dataset_sensitivity"
        ]
        >= 0.75
    )
    &
    (
        grid[
            "macro_sensitivity"
        ]
        >= 0.85
    )
].copy()


if len(eligible) == 0:

    raise RuntimeError(
        "No V2 rule satisfies "
        "the predefined DEV constraints."
    )


eligible = eligible.sort_values(
    [
        "macro_balanced_accuracy",
        "macro_specificity",
        "min_dataset_specificity",
        "macro_sensitivity",

        # deterministic tie-breakers:
        "base_probability",
        "rescue_probability",
        "base_run",
        "rescue_run",
    ],
    ascending=[
        False,
        False,
        False,
        False,

        False,
        False,
        False,
        False,
    ]
)


winner = eligible.iloc[0]


print(
    "\n" +
    "=" * 100
)

print(
    "TOP 30 ELIGIBLE DEV RULES"
)


show = [
    "base_probability",
    "base_uncertainty",
    "base_run",
    "rescue_probability",
    "rescue_uncertainty",
    "rescue_run",

    "macro_sensitivity",
    "macro_specificity",
    "macro_balanced_accuracy",

    "min_dataset_sensitivity",
    "min_dataset_specificity",

    "deepbeat_sensitivity",
    "deepbeat_specificity",

    "pulsewatch_sensitivity",
    "pulsewatch_specificity",

    "liu_sensitivity",
    "liu_specificity",
]


print(
    eligible[
        show
    ]
    .head(30)
    .to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "SELECTED GLOBAL RULE V2"
)

print(
    winner.to_string()
)


rule = {
    "version":
        "v6_locked_global_rule_v2",

    "decision":
        (
            "(base persistent episode) "
            "OR "
            "(high-confidence short episode)"
        ),

    "base": {
        "probability_threshold":
            float(
                winner[
                    "base_probability"
                ]
            ),

        "uncertainty_threshold":
            float(
                winner[
                    "base_uncertainty"
                ]
            ),

        "min_consecutive":
            int(
                winner[
                    "base_run"
                ]
            ),
    },

    "rescue": {
        "probability_threshold":
            float(
                winner[
                    "rescue_probability"
                ]
            ),

        "uncertainty_threshold":
            float(
                winner[
                    "rescue_uncertainty"
                ]
            ),

        "min_consecutive":
            int(
                winner[
                    "rescue_run"
                ]
            ),
    },

    "stream_policy":
        (
            "consecutive windows must belong "
            "to the same verified stream; "
            "gaps reset persistence"
        ),

    "selection_data":
        "DEV_ONLY",

    "test_used_numerically_for_selection":
        False,

    "important_protocol_note":
        (
            "The reused test sets had already been "
            "inspected before V2 rule design. "
            "Therefore V2 remains exploratory and "
            "requires confirmation on a future "
            "untouched external dataset."
        ),

    "constraints": {
        "macro_sensitivity_min":
            0.85,

        "per_dataset_sensitivity_min":
            0.75,
    },

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

        "min_dataset_specificity":
            float(
                winner[
                    "min_dataset_specificity"
                ]
            ),
    },
}


rule_path = (
    ART /
    "FROZEN_GLOBAL_RULE_V2.json"
)


rule_path.write_text(
    json.dumps(
        rule,
        indent=2
    )
)


sha = hashlib.sha256(
    rule_path.read_bytes()
).hexdigest()


(
    ART /
    "FROZEN_GLOBAL_RULE_V2_SHA256.txt"
).write_text(
    sha
    + "  "
    + rule_path.name
    + "\n"
)


print(
    "\nFrozen:",
    rule_path
)

print(
    "SHA256:",
    sha
)
