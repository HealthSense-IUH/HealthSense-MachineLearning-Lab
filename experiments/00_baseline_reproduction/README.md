# Experiment 00 — v4 Baseline Reproduction

Environment:
- Python 3.12.14
- scikit-learn 1.9.0
- scipy 1.18.1
- xgboost 3.4.1

## MIMIC PERform AF

35 subjects:
- 19 AF
- 16 Non-AF
- 4130 PPG windows
- 30 s window / 10 s step

MIMIC-only LOSO subject:
- Accuracy: 0.9429
- Recall: 1.0000
- Specificity: 0.8750

Best MIMIC window ROC-AUC:
- Random Forest: 0.9398

## Cross-dataset

MIMIC -> AFDB:
- XGBoost ROC-AUC: 0.9870

AFDB -> MIMIC:
- Random Forest ROC-AUC: 0.9757
- XGBoost ROC-AUC: 0.9685

## Pooled

XGBoost:
- Accuracy: 0.9589
- ROC-AUC: 0.9883

Target-domain MIMIC:
- Accuracy: 0.9039
- ROC-AUC: 0.9313

## Beat validation

AF:
- median beat F1: 0.855

Non-AF:
- median beat F1: 0.991

Baseline reproduced successfully.
