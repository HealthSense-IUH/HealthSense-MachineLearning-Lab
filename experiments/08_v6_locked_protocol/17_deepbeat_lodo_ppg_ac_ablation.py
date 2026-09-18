from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


ROOT = Path(
    "/home/phuc/Documents/HealthSense_rungtamnhi"
)

ART = (
    ROOT /
    "experiments/08_v6_locked_protocol/artifacts"
)

MANIFEST = (
    ROOT /
    "data/features/v7/"
    "healthsense_af_v6c_pacaware_train.csv"
)

MODEL = (
    ROOT /
    "models/multidomain/"
    "healthsense_af_v6c_stacking.pkl"
)


bundle = joblib.load(
    MODEL
)

BASE_TEMPLATE = (
    bundle["base_models"]
)

META_TEMPLATE = (
    bundle["meta"]
)

FEATURES_14 = list(
    bundle["features"]
)

FEATURES_13 = [
    f
    for f in FEATURES_14
    if f != "PPG_AC"
]


def valid_mask(
    df
):

    return (
        df[
            FEATURES_14
        ]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .notna()
        .all(axis=1)
    )


def evaluate(
    y,
    p
):

    pred = (
        p >= 0.5
    ).astype(int)


    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            pred,
            labels=[
                0,
                1
            ]
        ).ravel()
    )


    return {
        "AUROC":
            float(
                roc_auc_score(
                    y,
                    p
                )
            ),

        "PR_AUC":
            float(
                average_precision_score(
                    y,
                    p
                )
            ),

        "Brier":
            float(
                brier_score_loss(
                    y,
                    p
                )
            ),

        "TP":
            int(tp),

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "sensitivity":
            float(
                tp /
                (
                    tp + fn
                )
            ),

        "specificity":
            float(
                tn /
                (
                    tn + fp
                )
            ),
    }


# ============================================================
# NON-DEEPBEAT training cohort
# ============================================================

manifest = pd.read_csv(
    MANIFEST,
    low_memory=False
)


manifest["domain"] = (
    manifest[
        "domain"
    ].astype(str)
)

manifest["subject_id"] = (
    manifest[
        "subject_id"
    ].astype(str)
)


train = manifest[
    manifest[
        "domain"
    ]
    !=
    "DEEPBEAT"
].copy()


train = (
    train.loc[
        valid_mask(
            train
        )
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


train["group_id"] = (
    train[
        "domain"
    ]
    +
    ":"
    +
    train[
        "subject_id"
    ]
)


y_train = (
    pd.to_numeric(
        train[
            "label_v7"
        ],
        errors="raise"
    )
    .astype(int)
    .to_numpy()
)


groups = (
    train[
        "group_id"
    ]
    .to_numpy()
)


print(
    "train rows:",
    len(train)
)

print(
    "train groups:",
    train[
        "group_id"
    ].nunique()
)

print(
    train[
        "domain"
    ].value_counts()
)


# ============================================================
# ALL DeepBeat is valid holdout domain
# ============================================================

parts = []


for split, file in [
    (
        "train",
        ROOT /
        "data/features/v6/"
        "deepbeat_train_v6_14f.csv"
    ),
    (
        "dev",
        ROOT /
        "data/features/v6/"
        "deepbeat_validate_v6_14f.csv"
    ),
    (
        "test",
        ROOT /
        "data/features/v6/"
        "deepbeat_test_independent_v6_14f.csv"
    ),
]:

    x = pd.read_csv(
        file,
        low_memory=False
    )


    x = (
        x.loc[
            valid_mask(
                x
            )
        ]
        .copy()
    )


    x[
        "source_split"
    ] = split


    x[
        "label_eval"
    ] = (
        pd.to_numeric(
            x["label"],
            errors="raise"
        )
        .astype(int)
    )


    x[
        "subject_eval"
    ] = (
        x[
            "subject_id"
        ].astype(str)
    )


    parts.append(
        x
    )


test = pd.concat(
    parts,
    ignore_index=True
)


y_test = (
    test[
        "label_eval"
    ]
    .astype(int)
    .to_numpy()
)


print(
    "\nDeepBeat eval:",
    len(test),
    "subjects=",
    test[
        "subject_eval"
    ].nunique()
)


# ============================================================
# Matched folds
# ============================================================

gkf = GroupKFold(
    n_splits=5
)


splits = list(
    gkf.split(
        train,
        y_train,
        groups
    )
)


summary_rows = []
prediction_parts = []


for variant, features in [
    (
        "13F_NO_PPG_AC",
        FEATURES_13
    ),
    (
        "14F_WITH_PPG_AC",
        FEATURES_14
    ),
]:

    print(
        "\n" +
        "=" * 100
    )

    print(
        variant
    )


    X = train[
        features
    ]


    names = list(
        BASE_TEMPLATE.keys()
    )


    oof = np.zeros(
        (
            len(train),
            len(names)
        )
    )


    start = time.time()


    for fold, (
        tr,
        va
    ) in enumerate(
        splits,
        start=1
    ):

        print(
            "fold",
            fold
        )


        for j, name in enumerate(
            names
        ):

            print(
                " fitting",
                name
            )


            model = clone(
                BASE_TEMPLATE[
                    name
                ]
            )


            model.fit(
                X.iloc[
                    tr
                ],
                y_train[
                    tr
                ]
            )


            oof[
                va,
                j
            ] = (
                model.predict_proba(
                    X.iloc[
                        va
                    ]
                )[:, 1]
            )


    meta = clone(
        META_TEMPLATE
    )


    meta.fit(
        oof,
        y_train
    )


    final_models = {}


    for name in names:

        print(
            "final",
            name
        )


        model = clone(
            BASE_TEMPLATE[
                name
            ]
        )


        model.fit(
            X,
            y_train
        )


        final_models[
            name
        ] = model


    Xt = test[
        features
    ]


    base_test = np.column_stack([
        final_models[
            name
        ]
        .predict_proba(
            Xt
        )[:, 1]

        for name
        in names
    ])


    p = (
        meta.predict_proba(
            base_test
        )[:, 1]
    )


    # --------------------------------------------------------
    # ALL DeepBeat
    # --------------------------------------------------------

    m = evaluate(
        y_test,
        p
    )


    summary_rows.append({
        "variant":
            variant,

        "split":
            "ALL_DEEPBEAT",

        "rows":
            len(test),

        "subjects":
            test[
                "subject_eval"
            ].nunique(),

        **m,
    })


    # --------------------------------------------------------
    # Also break down by source split
    # --------------------------------------------------------

    for split in [
        "train",
        "dev",
        "test"
    ]:

        mask = (
            test[
                "source_split"
            ]
            ==
            split
        )


        mm = evaluate(
            y_test[
                mask
            ],
            p[
                mask
            ]
        )


        summary_rows.append({
            "variant":
                variant,

            "split":
                split,

            "rows":
                int(
                    mask.sum()
                ),

            "subjects":
                test.loc[
                    mask,
                    "subject_eval"
                ].nunique(),

            **mm,
        })


    prediction_parts.append(
        pd.DataFrame({
            "variant":
                variant,

            "split":
                test[
                    "source_split"
                ].to_numpy(),

            "subject_id":
                test[
                    "subject_eval"
                ].to_numpy(),

            "label":
                y_test,

            "probability":
                p,
        })
    )


    print(
        "ALL:",
        m
    )

    print(
        "elapsed:",
        time.time()
        -
        start
    )


# ============================================================
# Save
# ============================================================

summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    ART /
    "deepbeat_lodo_ppg_ac_ablation_summary.csv",
    index=False
)


pd.concat(
    prediction_parts,
    ignore_index=True
).to_csv(
    ART /
    "deepbeat_lodo_ppg_ac_ablation_predictions.csv",
    index=False
)


# ============================================================
# Delta 14F - 13F
# ============================================================

s13 = (
    summary[
        summary[
            "variant"
        ]
        ==
        "13F_NO_PPG_AC"
    ]
    .set_index(
        "split"
    )
)


s14 = (
    summary[
        summary[
            "variant"
        ]
        ==
        "14F_WITH_PPG_AC"
    ]
    .set_index(
        "split"
    )
)


rows = []


for split in [
    "ALL_DEEPBEAT",
    "train",
    "dev",
    "test"
]:

    rows.append({
        "split":
            split,

        "delta_AUROC":
            s14.loc[
                split,
                "AUROC"
            ]
            -
            s13.loc[
                split,
                "AUROC"
            ],

        "delta_PR_AUC":
            s14.loc[
                split,
                "PR_AUC"
            ]
            -
            s13.loc[
                split,
                "PR_AUC"
            ],

        "brier_improvement":
            s13.loc[
                split,
                "Brier"
            ]
            -
            s14.loc[
                split,
                "Brier"
            ],

        "delta_sensitivity":
            s14.loc[
                split,
                "sensitivity"
            ]
            -
            s13.loc[
                split,
                "sensitivity"
            ],

        "delta_specificity":
            s14.loc[
                split,
                "specificity"
            ]
            -
            s13.loc[
                split,
                "specificity"
            ],
    })


delta = pd.DataFrame(
    rows
)


delta.to_csv(
    ART /
    "deepbeat_lodo_ppg_ac_ablation_delta.csv",
    index=False
)


print(
    "\n" +
    "=" * 100
)

print(
    "SUMMARY"
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "14F - 13F"
)

print(
    delta.to_string(
        index=False
    )
)
