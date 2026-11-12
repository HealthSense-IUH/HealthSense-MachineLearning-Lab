# 1. Prepare data

Turns the raw datasets into the feature tables and subject-level splits used by the model.

- `data/`: raw MIMIC PERform AF recordings (`raw/`), feature tables (`features/`), splits (`splits/`). The other raw datasets are not stored here because of their licences.
- `scripts/`: DeepBeat manifests and memmap, feature extraction, subject-level splits, the PAC-aware training table (`build_v6b_*`, `build_v7_hard_negative_v2.py`).
- `scripts/v5_extractors/`: PulseWatch, Liu 2022 and PeakDet extractors from V5, whose outputs (`v5_outputs/`, not in git) feed V6.
- `00_extract_features_and_splits.ipynb`, `01_prepare_training_table.ipynb`: run the steps in order.

The feature computation itself is in `healthsense_core/` at the repository root.
