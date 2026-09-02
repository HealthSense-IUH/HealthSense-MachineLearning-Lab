# Experiment 01 — AFDB Label Audit

## Hypothesis

The previous pipeline grouped AFIB and AFL into the same positive class.

For a binary AF classification task this creates inconsistent target
semantics.

## Baseline

AFDB mapping:

- AFIB -> AF
- AFL -> AF
- N -> Non-AF
- J/Other -> excluded

AFDB windows:

28,903

## Proposed mapping

- AFIB -> AF
- N -> Non-AF
- AFL -> excluded
- J/Other -> excluded

AFDB windows:

28,734

169 AFL windows were removed.

## Cross-dataset result

The AFIB-only labels improved cross-dataset generalization.

Most notably, AFDB -> MIMIC with Random Forest:

- Accuracy: 0.9186 -> 0.9334
- Specificity: 0.8533 -> 0.8877
- F1: 0.9285 -> 0.9406
- ROC-AUC: 0.9757 -> 0.9788

This experiment supports keeping AFIB and AFL as distinct rhythm classes.

## Decision

Use AFIB-only positive labels for binary AF classification in v5.

AFL may later be introduced as a separate class in a multiclass model.
