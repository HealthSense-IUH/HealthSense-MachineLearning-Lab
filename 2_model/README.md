# 2. Model

- `train_v6c_stacking.py`, `02_train_stacking.ipynb`: train the stacked classifier (Extra Trees, Random Forest and XGBoost, with a logistic-regression meta-learner) on the training table from step 1.
- `models/`: model cards (JSON) with features, architecture and SHA-256. `healthsense_af_v6c_stacking.json` describes the published model. The `.pkl` file (about 1.9 GB) is not stored in git.
