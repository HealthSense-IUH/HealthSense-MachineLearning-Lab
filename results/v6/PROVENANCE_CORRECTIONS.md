# Provenance corrections

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
