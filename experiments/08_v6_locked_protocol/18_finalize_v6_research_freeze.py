from pathlib import Path
import hashlib
import json

import pandas as pd


ROOT = Path(
    "/home/phuc/Documents/HealthSense_rungtamnhi"
)

EXP = (
    ROOT /
    "experiments/08_v6_locked_protocol"
)

ART = (
    EXP /
    "artifacts"
)

MODEL = (
    ROOT /
    "models/multidomain/"
    "healthsense_af_v6c_stacking.pkl"
)


def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def read_csv(name):

    f = ART / name

    if not f.exists():

        raise RuntimeError(
            f"Missing required artifact: {f}"
        )

    return pd.read_csv(
        f
    )


def read_json(name):

    f = ART / name

    if not f.exists():

        raise RuntimeError(
            f"Missing required artifact: {f}"
        )

    return json.loads(
        f.read_text()
    )


# ============================================================
# Load major evidence
# ============================================================

lodo = read_csv(
    "lodo_domain_generalization_summary.csv"
)

ppg_macro = read_csv(
    "ppg_ac_ablation_dev_macro.csv"
)

ppg_delta = read_csv(
    "ppg_ac_ablation_dev_delta.csv"
)

pac = read_csv(
    "pac_pvc_challenge_summary.csv"
)

cal = read_csv(
    "calibration_summary.csv"
)

ci = read_csv(
    "global_rule_v2_exact_subject_ci.csv"
)

v2 = read_csv(
    "global_rule_v2_reused_test_summary.csv"
)

exposure = read_csv(
    "deepbeat_domain_exposure_summary.csv"
)

deepbeat_ablation = read_csv(
    "deepbeat_lodo_ppg_ac_ablation_summary.csv"
)

deepbeat_delta = read_csv(
    "deepbeat_lodo_ppg_ac_ablation_delta.csv"
)

rule_v2 = read_json(
    "FROZEN_GLOBAL_RULE_V2.json"
)


# ============================================================
# Final methodological status
# ============================================================

status = {
    "project":
        "HealthSense AF PPG Screening",

    "freeze_version":
        "V6_RESEARCH_FREEZE_2026_09",

    "primary_model":
        str(
            MODEL.relative_to(ROOT)
        ),

    "feature_schema": {
        "n_features":
            14,

        "features": [
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
        ],

        "PPG_AC":
            "RETAIN",
    },

    "probability_output": {
        "primary":
            "raw meta_probability",

        "platt_calibration":
            "ANALYSIS_ONLY_NOT_PROMOTED",

        "reason":
            (
                "Platt calibration did not improve "
                "development performance consistently "
                "across domains."
            ),
    },

    "subject_alert_rule": {
        "candidate":
            rule_v2,

        "status":
            "EXPLORATORY_CANDIDATE_FOR_SEALED_VALIDATION",

        "reason":
            (
                "The test datasets had already been "
                "inspected before V2 rule design."
            ),
    },

    "external_validation": {
        "MIMIC_III_Ext_PPG":
            "PENDING_ACCESS_NOT_USED",

        "TriggersAF":
            "PENDING_ACCESS_NOT_USED",
    },

    "evaluation_policy": {
        "current_subject_tests":
            "REUSED_SUBJECT_HELD_OUT_BENCHMARK",

        "LODO":
            "RETROSPECTIVE_DOMAIN_GENERALIZATION",

        "future_external":
            "SEALED_CONFIRMATORY_VALIDATION",
    },

    "development_status":
        (
            "STOP MODEL TUNING ON CURRENT BENCHMARKS"
        ),
}


status_path = (
    ART /
    "V6_RESEARCH_FREEZE_STATUS.json"
)


status_path.write_text(
    json.dumps(
        status,
        indent=2
    )
)


# ============================================================
# Evidence table
# ============================================================

rows = []


# ------------------------------------------------------------
# LODO
# ------------------------------------------------------------

for _, r in lodo.iterrows():

    rows.append({
        "section":
            "LODO",

        "dataset":
            r[
                "heldout_domain"
            ],

        "metric":
            "AUROC",

        "value":
            r["AUROC"],

        "evidence_level":
            "RETROSPECTIVE_DOMAIN_GENERALIZATION",

        "interpretation":
            (
                "Entire domain excluded from training "
                "and meta-learning."
            ),
    })


    rows.append({
        "section":
            "LODO",

        "dataset":
            r[
                "heldout_domain"
            ],

        "metric":
            "Sensitivity@0.5",

        "value":
            r[
                "sensitivity_at_0.5"
            ],

        "evidence_level":
            "RETROSPECTIVE_DOMAIN_GENERALIZATION",

        "interpretation":
            (
                "Fixed 0.5 threshold diagnostic; "
                "not threshold tuning."
            ),
    })


# ------------------------------------------------------------
# PPG_AC matched ablation
# ------------------------------------------------------------

for _, r in ppg_delta.iterrows():

    rows.append({
        "section":
            "PPG_AC_ABLATION",

        "dataset":
            r["dataset"],

        "metric":
            "delta_AUROC_14_minus_13",

        "value":
            r[
                "delta_AUROC_14_minus_13"
            ],

        "evidence_level":
            "MATCHED_DEV_ABLATION",

        "interpretation":
            (
                "Positive value favors retaining PPG_AC."
            ),
    })


# ------------------------------------------------------------
# DeepBeat LODO PPG_AC ablation
# ------------------------------------------------------------

db_all = deepbeat_delta[
    deepbeat_delta[
        "split"
    ]
    ==
    "ALL_DEEPBEAT"
].iloc[0]


for metric, source in [
    (
        "delta_AUROC_14_minus_13",
        "delta_AUROC"
    ),
    (
        "delta_PR_AUC_14_minus_13",
        "delta_PR_AUC"
    ),
    (
        "Brier_improvement_14_vs_13",
        "brier_improvement"
    ),
    (
        "delta_sensitivity_at_0.5",
        "delta_sensitivity"
    ),
    (
        "delta_specificity_at_0.5",
        "delta_specificity"
    ),
]:

    rows.append({
        "section":
            "DEEPBEAT_LODO_PPG_AC",

        "dataset":
            "DEEPBEAT",

        "metric":
            metric,

        "value":
            db_all[
                source
            ],

        "evidence_level":
            "RETROSPECTIVE_LODO_ABLATION",

        "interpretation":
            (
                "Matched 13F vs 14F with DeepBeat "
                "completely excluded from training."
            ),
    })


# ------------------------------------------------------------
# PAC/PVC
# ------------------------------------------------------------

for _, r in pac.iterrows():

    for threshold in [
        "0_5",
        "0_9"
    ]:

        rows.append({
            "section":
                "PAC_PVC_CHALLENGE",

            "dataset":
                r[
                    "challenge"
                ],

            "metric":
                f"window_FPR_ge_{threshold}",

            "value":
                r[
                    f"FPR_ge_{threshold}"
                ],

            "evidence_level":
                "HARD_NEGATIVE_CHALLENGE",

            "interpretation":
                (
                    "Window-level false-positive rate; "
                    "not temporal subject-alert rate."
                ),
        })


evidence = pd.DataFrame(
    rows
)


evidence.to_csv(
    ART /
    "v6_locked_evidence_summary.csv",
    index=False
)


# ============================================================
# Artifacts that must NOT be presented as final confirmation
# ============================================================

deprecated = [
    {
        "artifact":
            "v6h_selected_operating_points.csv",

        "status":
            "DEPRECATED_FOR_CONFIRMATORY_REPORTING",

        "reason":
            (
                "Per-dataset operating points were "
                "selected after test inspection."
            ),
    },

    {
        "artifact":
            "v6_final_thesis_table.csv",

        "status":
            "DEPRECATED_FOR_CONFIRMATORY_REPORTING",

        "reason":
            (
                "Contains test-tuned V6H subject-level "
                "operating points."
            ),
    },

    {
        "artifact":
            "v6_temporal_comparison_final.csv",

        "status":
            "HISTORICAL_EXPLORATORY",

        "reason":
            (
                "Includes V6F pooled configuration-union "
                "evaluation and pre-audit temporal logic."
            ),
    },

    {
        "artifact":
            "v6_generalization_summary.csv",

        "status":
            "HISTORICAL_PROVENANCE_WARNING",

        "reason":
            (
                "Earlier summary may contain obsolete "
                "MIMIC-III-Ext-PPG naming. "
                "The actual source used was "
                "MIMIC PERform AF."
            ),
    },
]


deprecated_df = pd.DataFrame(
    deprecated
)


deprecated_df.to_csv(
    ART /
    "deprecated_exploratory_artifacts.csv",
    index=False
)


# ============================================================
# Provenance corrections
# ============================================================

provenance = """# Provenance corrections

## MIMIC

Any previous V6 artifact that labels the existing 4,130-window
MIMIC dataset as `MIMIC-III-Ext-PPG` must not be reported under
that name.

Correct source:

`MIMIC PERform AF`

MIMIC-III-Ext-PPG has not yet been used.

## Test terminology

The existing DeepBeat, PulseWatch and Liu test results should be
described as:

`reused subject-held-out benchmarks`

They are subject-disjoint from training, but the datasets and test
results have been inspected during model development.

## LODO terminology

The leave-one-domain-out experiments should be described as:

`retrospective leave-one-domain-out domain-generalization analysis`

They are not pristine external validation because these datasets
were already known during the project.

## Subject alert rule

V6 Global Rule V2 is an exploratory frozen candidate.

It must not be described as a confirmed deployment threshold before
evaluation on a future untouched dataset.

## Calibration

Raw `meta_probability` remains the primary probability output.

The DEV-fitted Platt calibrator is retained only as a calibration
analysis artifact and is not promoted to the primary model output.
"""


(
    ART /
    "PROVENANCE_CORRECTIONS.md"
).write_text(
    provenance
)


# ============================================================
# Research-ready summary
# ============================================================

md = """# HealthSense AF — V6 Research Freeze

## Frozen model

The current frozen model candidate is the V6C stacked ensemble
using ExtraTrees, Random Forest and XGBoost base learners with
a logistic-regression meta learner.

The feature schema contains 14 features and retains `PPG_AC`.

## PPG_AC

Matched development ablation supports retaining PPG_AC.

Across the three development domains, the 14-feature configuration
improved macro discrimination and Brier score relative to the
13-feature configuration.

A separate DeepBeat leave-one-domain-out ablation also showed better
overall AUROC, PR-AUC, Brier score and specificity with PPG_AC,
although sensitivity at the fixed 0.5 threshold was slightly lower.

Therefore PPG_AC is retained.

## Domain generalization

Retrospective leave-one-domain-out evaluation showed strong
generalization to PulseWatch, Liu and MIMIC PERform AF.

DeepBeat was substantially harder when the entire DeepBeat domain
was excluded from training.

This failure cannot be explained by the stacking layer alone.
The DeepBeat false-negative phenotype had lower beat-to-beat
variability and substantially higher PPG autocorrelation than
correctly detected AF windows.

## DeepBeat domain exposure

Adding DeepBeat training subjects improved ranking and AF sensitivity
on subject-disjoint DeepBeat DEV/TEST data, but also increased
non-AF probabilities and reduced specificity at a fixed threshold.

This indicates a domain/phenotype transfer problem rather than a
simple threshold problem.

## PAC/PVC

PulseWatch held-out PAC/PVC windows showed relatively low
false-positive rates.

PeakDet remained a difficult PAC/PVC hard-negative challenge.

PeakDet results are retained as a failure-mode analysis and are not
used for further model tuning.

## Calibration

Raw meta probabilities remain the primary model output.

Platt calibration improved some reused-test Brier scores but did not
produce consistent improvement on development data, particularly
DeepBeat.

The Platt calibrator is therefore not promoted.

## Temporal alert

Global Rule V2 uses verified temporal streams and a common rule
across datasets.

Because the reused test sets were already inspected before V2 was
designed, V2 remains an exploratory candidate for future sealed
validation.

## Statistical uncertainty

Window-level discrimination metrics use subject-cluster bootstrap
confidence intervals.

Subject-level sensitivity, specificity, PPV and NPV should use
Wilson or exact binomial confidence intervals because subject
counts are small.

## External validation

MIMIC-III-Ext-PPG and TriggersAF have not been used.

When access becomes available, they should be treated as sealed
external validation datasets.

No threshold, calibration parameter, feature set or model
hyperparameter should be changed after inspecting their results.
"""


(
    ART /
    "V6_RESEARCH_FREEZE.md"
).write_text(
    md
)


# ============================================================
# SHA256 manifest
# ============================================================

important_files = [
    MODEL,

    ART /
    "FROZEN_GLOBAL_RULE_V2.json",

    ART /
    "FROZEN_PLATT_CALIBRATOR.json",

    ART /
    "lodo_domain_generalization_summary.csv",

    ART /
    "ppg_ac_ablation_dev_summary.csv",

    ART /
    "ppg_ac_ablation_dev_delta.csv",

    ART /
    "deepbeat_lodo_ppg_ac_ablation_summary.csv",

    ART /
    "deepbeat_lodo_ppg_ac_ablation_delta.csv",

    ART /
    "pac_pvc_challenge_summary.csv",

    ART /
    "global_rule_v2_exact_subject_ci.csv",

    ART /
    "window_metrics_subject_cluster_bootstrap.csv",

    ART /
    "V6_RESEARCH_FREEZE_STATUS.json",

    ART /
    "PROVENANCE_CORRECTIONS.md",

    ART /
    "V6_RESEARCH_FREEZE.md",
]


manifest_rows = []


for f in important_files:

    if not f.exists():

        print(
            "WARNING missing:",
            f
        )

        continue


    manifest_rows.append({
        "file":
            str(
                f.relative_to(ROOT)
            ),

        "bytes":
            f.stat().st_size,

        "sha256":
            sha256_file(
                f
            ),
    })


manifest_df = pd.DataFrame(
    manifest_rows
)


manifest_df.to_csv(
    ART /
    "V6_RESEARCH_FREEZE_SHA256.csv",
    index=False
)


# ============================================================
# Print final status
# ============================================================

print(
    "\n" +
    "=" * 100
)

print(
    "V6 RESEARCH FREEZE"
)

print(
    json.dumps(
        status,
        indent=2
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "LODO"
)

print(
    lodo[
        [
            "heldout_domain",
            "AUROC",
            "PR_AUC",
            "Brier",
            "sensitivity_at_0.5",
            "specificity_at_0.5",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "PPG_AC MACRO DEV"
)

print(
    ppg_macro.to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "DEEPBEAT LODO PPG_AC"
)

print(
    deepbeat_delta.to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "PAC/PVC"
)

print(
    pac.to_string(
        index=False
    )
)


print(
    "\nFreeze artifacts written to:"
)

print(
    ART /
    "V6_RESEARCH_FREEZE.md"
)

print(
    ART /
    "V6_RESEARCH_FREEZE_STATUS.json"
)

print(
    ART /
    "V6_RESEARCH_FREEZE_SHA256.csv"
)
