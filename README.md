# HealthSense ML Lab

Code and results behind the atrial fibrillation (AF) classifier of the HealthSense wearable system: a fixed
stacked classifier (Extra Trees, Random Forest and XGBoost with a logistic-regression meta-learner) on 14
features of 30-s photoplethysmography (PPG) windows, plus a temporal persistence rule. The classifier and
its evaluation are described in *"HealthSense: An AI- and IoT-Based System for Atrial Fibrillation Screening
from Wearable Photoplethysmography"* (submitted to SN Computer Science).

**[PAPER.md](PAPER.md)** maps every table, figure and supplementary analysis of the paper to the script that
produces it and the file that stores the result.

## Layout

The folders are numbered in the order of the work.

| Folder | Content |
|---|---|
| [`healthsense_core/`](healthsense_core) | Core code shared by all steps: PPG filtering, beat detection, NN intervals, the 14 features |
| [`1_prepare_data/`](1_prepare_data) | Data (`data/`) and the scripts that turn the raw datasets into feature tables and subject-level train/DEV/TEST splits |
| [`2_model/`](2_model) | Training of the stacked classifier and the model cards (`models/`) |
| [`3_evaluation/`](3_evaluation) | Locked evaluation protocol (steps 01–18) and its results: TEST metrics, temporal rule, LODO, ablation, calibration, PAC/PVC |
| [`4_paper/`](4_paper) | Supplementary analyses and figure scripts for the paper, with their outputs |
| [`archive/`](archive) | Earlier versions (V4, V5), kept for history |
| [`docs/`](docs) | Project history in Vietnamese ([`HISTORY_vi.md`](docs/HISTORY_vi.md)) and the map of old to new paths ([`PATH_MAP.md`](docs/PATH_MAP.md)) |

## Setup

```bash
python -m venv venv
# activate the environment, then:
pip install -r requirements-v6-repro.txt   # environment verified to reload the frozen V6 model
```

Run every script **from the repository root** with the root on the Python path, for example:

```bash
PYTHONPATH=. python 3_evaluation/protocol/09_subject_cluster_bootstrap.py
```

## Reproducing the results

1. `1_prepare_data/00_extract_features_and_splits.ipynb` and `01_prepare_training_table.ipynb`: feature tables and splits.
2. `2_model/02_train_stacking.ipynb`: trains the stacked classifier.
3. `3_evaluation/03_locked_protocol.ipynb`: runs the locked protocol 01–18.
4. `4_paper/`: supplementary analyses and figures (see [PAPER.md](PAPER.md)).

Steps 1 and 2 need the raw datasets, which are not stored here because of their licences. MIMIC PERform AF
(ODbL) is included under `1_prepare_data/data/raw/mimic_perform/`. DeepBeat, PulseWatch and Liu 2022 must be
obtained from their providers (see the Data availability statement of the paper). The step 3 and 4 results
are committed, so they can be inspected without the raw data.

## Frozen model

The published model is `healthsense_af_v6c_stacking.pkl` (about 1.9 GB, not stored in git), SHA-256
`e27370c51c8bbaf624abcbdabae40af5528d28bb2bd9e9249d1d5bf9cbb9b77c`. Its model card is
`2_model/models/healthsense_af_v6c_stacking.json`. The model file is available from the authors on request.
The deployed inference service is in the HealthSense-AI-AFib-Prediction-Service repository.

## Licence

MIT (see [LICENSE](LICENSE)).
