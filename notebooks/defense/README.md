# HealthSense | Reproducible evidence notebooks for defense

**Purpose:** turn saved, real research artifacts into auditable visual evidence. This is a companion to the two existing notebooks in the repository, not a replacement for the original experiment scripts. Files here **do not train models, choose new operating points, or overwrite frozen artifacts**.

Source snapshot checked: `HealthSense-IUH/HealthSense-MachineLearning-Lab`, `main` at `23773e9597dce439575ec4d2fa32038d2d598fea` (the notebooks record the *actual* local Git HEAD at execution). Reference: [repo](https://github.com/HealthSense-IUH/HealthSense-MachineLearning-Lab).

## The 3 notebooks

| Notebook | Figures / evidence |
|---|---|
| `01_Data_Label_and_Signal_Audit.ipynb` | Frozen prediction inventory (counts by dataset/split), subject-overlap audit, historical baseline, AFIB/AFL label-audit deltas, PPG beat detector, SQI ROC computed from tracked SQI prediction CSV. |
| `02_Frozen_Metrics_LODO_and_Ablations.ipynb` | Reused-test AUROC / PR-AUC / Brier with subject-cluster CI; all 4 LODO domains, TP/TN/FP/FN and derived sensitivity/specificity/accuracy/F1; confusion matrices; 13F/14F ablation; PAC/PVC FPR; DeepBeat base-vs-stack and FN-vs-TP forensics. |
| `03_Temporal_Calibration_and_Freeze.ipynb` | Selected DEV-only alert rule and selection constraints; verified stream coverage; exploratory subject-level confusion; exact sensitivity confidence intervals; DEV Brier raw versus Platt; feature and SHA256 integrity gate; limitations and freeze status. |

`healthsense_evidence_utils.py` is required, and must stay alongside the notebooks.

## Install / run on Ubuntu

Your full research repository must be present locally. The artifact CSVs used by these notebooks are already tracked in the repository, except the model binary and some large prediction CSVs.

```bash
# Use your existing research environment (or equivalent Python >=3.10)
conda activate healthsense-af-v5
python -m pip install pandas numpy matplotlib scikit-learn jupyterlab ipykernel nbformat
python -m ipykernel install --user --name healthsense-af-v5 --display-name 'Python (HealthSense evidence)'

cd /home/phuc/Documents/HealthSense_rungtamnhi
mkdir -p notebooks
unzip ~/Downloads/HealthSense_Defense_Evidence_Notebooks.zip -d notebooks/
export HEALTHSENSE_REPO="$PWD"
jupyter lab notebooks/HealthSense_Defense_Evidence_Notebooks/
```

Run `01` → `02` → `03` using **Kernel → Restart Kernel and Run All Cells** for each. If you placed the archive elsewhere, set `HEALTHSENSE_REPO` to the full repository root and launch Jupyter from the extracted notebook directory. In VSCode with a different kernel working directory, set `HEALTHSENSE_NOTEBOOK_DIR` to the extracted folder before launching VSCode/Jupyter.

No extra dependency such as XGBoost is required for *artifact-only visualizations*. The original training scripts have separate requirements.

## Artifacts and provenance output

Each notebook writes to a new UTC-timestamped folder under:

`<repo>/outputs/healthsense_defense_evidence/<timestamp>/`

It exports figure pairs (`.png`, `.pdf`), compact `.csv` tables, and a notebook-specific `*_input_manifest.json` with local Git HEAD/status, source file paths, byte sizes, SHA256 (for files <=64 MiB), and plot inventory. It never writes inside `experiments/.../artifacts` or `models`.

If these generated outputs are intended to be committed to GitHub, manually review them and verify that subject identifiers/raw patient data are not exposed. The supplied charts are aggregate-level. Do not commit local raw ECG/PPG or the ~1.88 GB model binary via regular Git.

## Distinguish evidence levels

* **Replot existing CSV**: original experiment *result tables*; graphic itself is new, not a newly conducted experiment.
* **Recomputed from saved confusion counts**: accuracy, precision, F1, balanced accuracy; counts are the source data, not new model inference.
* **Recomputed from saved predictions**: actual ROC/PR curves or threshold metrics, only where those CSVs exist locally.
* **Reused subject-held-out TEST**: splits are subject-disjoint within each dataset, but earlier benchmark inspection occurred during model development.
* **Retrospective LODO**: a leave-one-domain-out refitting experiment, not a pristine external clinical validation.
* **Exploratory temporal subject alert**: DEV-only rule-selection procedure; current reused TEST had been inspected earlier, so no confirmatory claims.

### Crucial missing objects

1. `models/multidomain/healthsense_af_v6c_stacking.pkl` is not in normal Git. Its declared hash is in `locked_inputs_SHA256.txt`.
2. `*_test_locked_predictions.csv` are not committed. Notebook 02 **never reconstructs an ROC curve from an aggregate AUROC**. If present locally, it validates each prediction CSV against the frozen SHA before recomputing ROC, PR and 0.5 confusion metrics. Otherwise it explicitly marks the missing file.
3. The complete `dev_global_rule_v2_grid.csv` is not in the checked tree. Notebook 03 shows the selected DEV rule and its recorded metrics, **not invented grid-search results**.
4. Current repo alone does not show clinical validation with reference ECG on real wearable users or that the deployed AI Service is running the same frozen model and 14-feature schema.

Optional 1.88 GB model hashing is disabled by default for responsiveness. To verify it, run with `HEALTHSENSE_HASH_LARGE=1` before notebook 03. The hash check raises on a mismatch; missing files are reported, not silently replaced. Results from missing artifacts must not be claimed.

## The original experiment scripts

| Evidence | Original script / source |
|---|---|
| Locked window predictions | `experiments/08_v6_locked_protocol/01_generate_locked_predictions.py` |
| Audit split/time and verified streams | `02_audit_temporal_and_splits.py` / `03_build_verified_streams.py` |
| DEV temporal rule | `06_tune_global_rule_v2_dev.py` |
| Reused TEST subject alert | `07_evaluate_global_rule_v2_test.py` |
| Calibration / bootstrap / CI | `08_calibration_diagnostics.py` / `09_subject_cluster_bootstrap.py` / `10_exact_subject_ci.py` |
| Retrospective LODO and aggregation | `11_lodo_domain_generalization.py` / `12_collect_lodo_results.py` |
| Matched ablation and PAC/PVC | `13_ppg_ac_ablation_dev.py` / `14_pac_pvc_challenge.py` |
| DeepBeat forensic and freeze | `15_deepbeat_lodo_forensics.py` to `18_finalize_v6_research_freeze.py` |

Historical exploratory artifacts listed in `deprecated_exploratory_artifacts.csv` must not be promoted to confirmatory results.

## Validation performed for this delivered bundle

All three notebooks were structurally validated with `nbformat` and smoke-executed on a small *local test fixture* consisting of representative artifact schemas and values. This checks plotting and control flow; it **does not mean the actual full repository was executed in this environment**, which has no direct GitHub clone access. Run the full notebooks against your Ubuntu repo to generate actual defense figures and manifests.
