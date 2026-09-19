# Experiment 03 — PPG-only SQI Feasibility

A PPG-only Random Forest SQI was evaluated using LOSO.

Result:
- ROC-AUC: 0.8112

At SQI threshold 0.30:
- AF coverage: 0.905
- Non-AF coverage: 0.904
- AF bad-window leakage: 0.245
- Non-AF bad-window leakage: 0.043

Increasing the SQI threshold reduced AF coverage substantially without
meaningfully reducing bad-window leakage in AF.

Decision:
Do not use this SQI as a hard quality gate in the current AF pipeline.

The SQI problem may be revisited with additional motion/sensor information.
