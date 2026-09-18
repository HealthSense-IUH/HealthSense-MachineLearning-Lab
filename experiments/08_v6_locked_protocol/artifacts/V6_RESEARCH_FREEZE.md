# HealthSense AF — V6 Research Freeze

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
