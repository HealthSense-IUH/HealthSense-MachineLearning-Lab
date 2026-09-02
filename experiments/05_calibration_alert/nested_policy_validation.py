import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    LeaveOneGroupOut,
    GroupKFold,
    GridSearchCV,
    StratifiedGroupKFold,
)

from healthsense_ml.training import build_model_registry


DATA = (
    "experiments/04_feature_model/"
    "af_features_cosen_ac.csv"
)

df = pd.read_csv(DATA)


FEATURES = [
    "HR_mean",
    "Mean_NN",
    "SDNN",
    "RMSSD",
    "NN50",
    "pNN50",
    "CV",
    "HF",
    "Total_Power",
    "HF_norm",
    "SD1",
    "SD2",
    "SampEn",
    "PPG_AC",
]


THRESHOLDS = np.arange(
    0.30,
    0.91,
    0.05,
)


POLICIES = [
    (1, 1),
    (2, 3),
    (3, 3),
    (3, 5),
]


rf_spec = (
    build_model_registry()[
        "Random Forest"
    ]
)


def apply_policy(
    prob,
    threshold,
    k,
    n,
):
    positive = (
        np.asarray(prob)
        >= threshold
    ).astype(int)

    alert = np.zeros(
        len(positive),
        dtype=int,
    )

    for i in range(
        len(positive)
    ):
        if i < n - 1:
            continue

        recent = positive[
            i - n + 1:
            i + 1
        ]

        if recent.sum() >= k:
            alert[i] = 1

    return alert


def count_episodes(alert):

    alert = np.asarray(
        alert,
        dtype=int,
    )

    if len(alert) == 0:
        return 0

    starts = (
        (alert == 1)
        &
        np.r_[
            True,
            alert[:-1] == 0
        ]
    )

    return int(
        starts.sum()
    )


def fit_tuned_rf(
    train_df,
):
    X = train_df[FEATURES]
    y = train_df[
        "status"
    ].to_numpy()

    groups = train_df[
        "record_id"
    ].to_numpy()

    search = GridSearchCV(
        clone(
            rf_spec[
                "pipeline"
            ]
        ),
        rf_spec[
            "param_grid"
        ],
        cv=GroupKFold(
            n_splits=3
        ),
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )

    search.fit(
        X,
        y,
        groups=groups,
    )

    return (
        search.best_estimator_,
        search.best_params_,
    )


def generate_inner_oof(
    outer_train,
):

    X = outer_train[
        FEATURES
    ]

    y = outer_train[
        "status"
    ].to_numpy()

    groups = outer_train[
        "record_id"
    ].to_numpy()

    splitter = (
        StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )
    )

    rows = []

    for inner_fold, (
        inner_train_idx,
        inner_val_idx,
    ) in enumerate(
        splitter.split(
            X,
            y,
            groups,
        ),
        1,
    ):

        inner_train = (
            outer_train
            .iloc[
                inner_train_idx
            ]
            .copy()
        )

        inner_val = (
            outer_train
            .iloc[
                inner_val_idx
            ]
            .copy()
        )

        model, params = (
            fit_tuned_rf(
                inner_train
            )
        )

        prob = (
            model.predict_proba(
                inner_val[
                    FEATURES
                ]
            )[:, 1]
        )

        temp = inner_val[
            [
                "record_id",
                "status",
                "t_start",
            ]
        ].copy()

        temp[
            "prob"
        ] = prob

        temp[
            "inner_fold"
        ] = inner_fold

        rows.append(
            temp
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def evaluate_policy(
    pred,
    threshold,
    k,
    n,
):

    subject_rows = []

    all_truth = []
    all_alert = []

    for rid, x in (
        pred.groupby(
            "record_id"
        )
    ):

        x = x.sort_values(
            "t_start"
        )

        truth = int(
            x.status.iloc[0]
        )

        alert = apply_policy(
            x.prob.to_numpy(),
            threshold,
            k,
            n,
        )

        duration_s = (
            x.t_start.max()
            - x.t_start.min()
            + 30.0
        )

        duration_h = (
            duration_s
            / 3600.0
        )

        episodes = (
            count_episodes(
                alert
            )
        )

        first = (
            np.flatnonzero(
                alert
            )
        )

        first_alert_s = (
            float(
                x.iloc[
                    first[0]
                ].t_start
                + 30.0
            )
            if len(first)
            else np.nan
        )

        subject_rows.append({
            "record_id":
                rid,

            "status":
                truth,

            "any_alert":
                int(
                    alert.any()
                ),

            "episodes":
                episodes,

            "episodes_per_hour":
                (
                    episodes
                    / duration_h
                    if duration_h > 0
                    else np.nan
                ),

            "first_alert_s":
                first_alert_s,
        })

        all_truth.extend(
            [truth]
            * len(alert)
        )

        all_alert.extend(
            alert.tolist()
        )

    subject = pd.DataFrame(
        subject_rows
    )

    af = subject[
        subject.status == 1
    ]

    non = subject[
        subject.status == 0
    ]

    all_truth = np.asarray(
        all_truth
    )

    all_alert = np.asarray(
        all_alert
    )

    tp = np.sum(
        (all_truth == 1)
        &
        (all_alert == 1)
    )

    fn = np.sum(
        (all_truth == 1)
        &
        (all_alert == 0)
    )

    tn = np.sum(
        (all_truth == 0)
        &
        (all_alert == 0)
    )

    fp = np.sum(
        (all_truth == 0)
        &
        (all_alert == 1)
    )

    subject_sensitivity = (
        af.any_alert.mean()
    )

    subject_specificity = (
        1.0
        - non.any_alert.mean()
    )

    window_sensitivity = (
        tp / (tp + fn)
        if tp + fn
        else np.nan
    )

    window_specificity = (
        tn / (tn + fp)
        if tn + fp
        else np.nan
    )

    detected_af = af[
        af.any_alert == 1
    ]

    return {
        "threshold":
            float(threshold),

        "k":
            k,

        "n":
            n,

        "policy":
            f"{k}-of-{n}",

        "subject_sensitivity":
            subject_sensitivity,

        "subject_specificity":
            subject_specificity,

        "window_sensitivity":
            window_sensitivity,

        "window_specificity":
            window_specificity,

        "mean_false_alerts_per_hour":
            non[
                "episodes_per_hour"
            ].mean(),

        "median_first_alert_s":
            (
                detected_af[
                    "first_alert_s"
                ].median()
                if len(detected_af)
                else np.nan
            ),
    }


def select_policy(
    inner_oof,
):

    rows = []

    for threshold in (
        THRESHOLDS
    ):

        for k, n in POLICIES:

            rows.append(
                evaluate_policy(
                    inner_oof,
                    threshold,
                    k,
                    n,
                )
            )

    result = pd.DataFrame(
        rows
    )

    eligible = result[
        result[
            "subject_sensitivity"
        ] >= 0.95
    ].copy()

    if len(eligible) == 0:

        best_sens = result[
            "subject_sensitivity"
        ].max()

        eligible = result[
            result[
                "subject_sensitivity"
            ] == best_sens
        ].copy()

    # Pre-declared selection rule:
    #
    # screening sensitivity first,
    # then subject specificity,
    # then fewer false alert episodes,
    # then window specificity,
    # then window sensitivity,
    # then faster alert.
    eligible = (
        eligible.sort_values(
            [
                "subject_specificity",
                "mean_false_alerts_per_hour",
                "window_specificity",
                "window_sensitivity",
                "median_first_alert_s",
                "threshold",
            ],
            ascending=[
                False,
                True,
                False,
                False,
                True,
                False,
            ],
        )
    )

    return (
        eligible.iloc[0],
        result,
    )


outer = LeaveOneGroupOut()

X_all = df[
    FEATURES
]

y_all = df[
    "status"
].to_numpy()

groups_all = df[
    "record_id"
].to_numpy()


outer_subject_rows = []
outer_window_rows = []
selection_rows = []


for outer_fold, (
    train_idx,
    test_idx,
) in enumerate(
    outer.split(
        X_all,
        y_all,
        groups_all,
    ),
    1,
):

    outer_train = (
        df.iloc[
            train_idx
        ].copy()
    )

    outer_test = (
        df.iloc[
            test_idx
        ].copy()
    )

    test_subject = (
        outer_test[
            "record_id"
        ].iloc[0]
    )

    truth = int(
        outer_test[
            "status"
        ].iloc[0]
    )

    print()
    print(
        "=" * 90
    )

    print(
        f"[{outer_fold:02d}/35] "
        f"OUTER TEST: "
        f"{test_subject}"
    )

    # --------------------------------------------------
    # POLICY SELECTION USING OUTER-TRAIN ONLY
    # --------------------------------------------------

    inner_oof = (
        generate_inner_oof(
            outer_train
        )
    )

    selected, candidates = (
        select_policy(
            inner_oof
        )
    )

    threshold = float(
        selected[
            "threshold"
        ]
    )

    k = int(
        selected["k"]
    )

    n = int(
        selected["n"]
    )

    print(
        "Selected:",
        f"threshold={threshold:.2f}",
        f"policy={k}-of-{n}",
        f"inner sensitivity="
        f"{selected['subject_sensitivity']:.3f}",
        f"inner specificity="
        f"{selected['subject_specificity']:.3f}",
    )

    selection_rows.append({
        "outer_subject":
            test_subject,

        "outer_status":
            truth,

        **selected.to_dict(),
    })

    # --------------------------------------------------
    # FINAL MODEL ON ALL OUTER-TRAIN SUBJECTS
    # --------------------------------------------------

    final_model, best_params = (
        fit_tuned_rf(
            outer_train
        )
    )

    outer_test = (
        outer_test.sort_values(
            "t_start"
        )
        .copy()
    )

    prob = (
        final_model.predict_proba(
            outer_test[
                FEATURES
            ]
        )[:, 1]
    )

    alert = apply_policy(
        prob,
        threshold,
        k,
        n,
    )

    episodes = (
        count_episodes(
            alert
        )
    )

    alert_idx = (
        np.flatnonzero(
            alert
        )
    )

    first_alert_s = (
        float(
            outer_test.iloc[
                alert_idx[0]
            ].t_start
            + 30.0
        )
        if len(alert_idx)
        else np.nan
    )

    duration_s = (
        outer_test.t_start.max()
        - outer_test.t_start.min()
        + 30.0
    )

    duration_h = (
        duration_s / 3600.0
    )

    outer_subject_rows.append({
        "record_id":
            test_subject,

        "status":
            truth,

        "threshold":
            threshold,

        "policy":
            f"{k}-of-{n}",

        "any_alert":
            int(
                alert.any()
            ),

        "alert_episodes":
            episodes,

        "episodes_per_hour":
            episodes
            / duration_h,

        "first_alert_s":
            first_alert_s,

        "best_params":
            str(
                best_params
            ),
    })

    for (
        t_start,
        p,
        a,
    ) in zip(
        outer_test[
            "t_start"
        ],
        prob,
        alert,
    ):

        outer_window_rows.append({
            "record_id":
                test_subject,

            "status":
                truth,

            "t_start":
                t_start,

            "prob":
                p,

            "alert":
                int(a),

            "threshold":
                threshold,

            "policy":
                f"{k}-of-{n}",
        })


subject_result = pd.DataFrame(
    outer_subject_rows
)

window_result = pd.DataFrame(
    outer_window_rows
)

selection_result = pd.DataFrame(
    selection_rows
)


subject_result.to_csv(
    "experiments/05_calibration_alert/"
    "nested_policy_subject_results.csv",
    index=False,
)

window_result.to_csv(
    "experiments/05_calibration_alert/"
    "nested_policy_window_results.csv",
    index=False,
)

selection_result.to_csv(
    "experiments/05_calibration_alert/"
    "nested_policy_selections.csv",
    index=False,
)


# ======================================================
# FINAL OUTER TEST METRICS
# ======================================================

af = subject_result[
    subject_result.status == 1
]

non = subject_result[
    subject_result.status == 0
]


subject_sens = (
    af.any_alert.mean()
)

subject_spec = (
    1.0
    - non.any_alert.mean()
)


y = window_result[
    "status"
].to_numpy()

a = window_result[
    "alert"
].to_numpy()


tp = np.sum(
    (y == 1)
    &
    (a == 1)
)

fn = np.sum(
    (y == 1)
    &
    (a == 0)
)

tn = np.sum(
    (y == 0)
    &
    (a == 0)
)

fp = np.sum(
    (y == 0)
    &
    (a == 1)
)


window_sens = (
    tp / (tp + fn)
)

window_spec = (
    tn / (tn + fp)
)


print()
print(
    "=" * 100
)

print(
    "STRICT NESTED POLICY RESULTS"
)

print(
    "=" * 100
)

print(
    "Subject sensitivity:",
    f"{subject_sens:.4f}",
    f"({af.any_alert.sum()}/{len(af)})",
)

print(
    "Subject specificity:",
    f"{subject_spec:.4f}",
    f"({len(non) - non.any_alert.sum()}/{len(non)})",
)

print(
    "Window sensitivity:",
    f"{window_sens:.4f}",
)

print(
    "Window specificity:",
    f"{window_spec:.4f}",
)

print(
    "Non-AF false subjects:",
    int(
        non.any_alert.sum()
    ),
)

print(
    "Median false alerts/hour:",
    f"{non.episodes_per_hour.median():.4f}",
)

print(
    "Mean false alerts/hour:",
    f"{non.episodes_per_hour.mean():.4f}",
)

detected = af[
    af.any_alert == 1
]

print(
    "Median first-alert time:",
    f"{detected.first_alert_s.median():.1f}s",
)


print()
print(
    "=" * 100
)

print(
    "SELECTED POLICY FREQUENCY"
)

print(
    "=" * 100
)

print(
    subject_result[
        [
            "threshold",
            "policy",
        ]
    ]
    .value_counts()
    .sort_values(
        ascending=False
    )
    .to_string()
)


print()
print(
    "=" * 100
)

print(
    "OUTER FALSE POSITIVES"
)

print(
    "=" * 100
)

print(
    non[
        non.any_alert == 1
    ][
        [
            "record_id",
            "threshold",
            "policy",
            "alert_episodes",
            "episodes_per_hour",
            "first_alert_s",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)
