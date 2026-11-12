# Paper map

Where each part of the paper comes from. Paths are relative to the repository root. `protocol/NN` means
`3_evaluation/protocol/NN_*.py`.

## Tables

| Paper | Content | Script | Result file |
|---|---|---|---|
| Table 1 | The 14 input features | `healthsense_core/hrv_features.py`, PPG_AC in `1_prepare_data/scripts/extract_deepbeat_v6_features.py` | — |
| Table 2 | Datasets, subject-level Train/DEV/TEST | `1_prepare_data/` notebooks 00–01, `protocol/01`, `protocol/03` | `1_prepare_data/data/splits/`, `3_evaluation/results/prediction_generation_audit.csv`, `verified_stream_summary.csv` |
| Table 3 | Contribution of PPG_AC (DEV) | `protocol/13` | `3_evaluation/results/ppg_ac_ablation_dev_*.csv` |
| Table 4 | Stacked classifier vs single learners and baselines (DEV) | `4_paper/analyses/supp_stacking_vs_single_dev.py` | `4_paper/results/stacking_vs_single_dev_*.csv` |
| Table 5 | False-positive rate on PAC/PVC datasets | `protocol/14` | `3_evaluation/results/pac_pvc_challenge_summary.csv` |
| Table 7 | Model and analysis configuration | `2_model/train_v6c_stacking.py` | `2_model/models/healthsense_af_v6c_stacking.json` |

## Figures

Figures 1 and 2 are drawn in TikZ in the manuscript.

| Paper | Script | Inputs | Output |
|---|---|---|---|
| Fig. 3 Reliability diagrams (DEV) | `4_paper/figures/supp_fig3_reliability_dev.py` | `protocol/08` (`dev_platt_oof_predictions.csv`) | `4_paper/results/fig3_reliability_dev.pdf` |
| Fig. 4 ROC and precision–recall (TEST) | `4_paper/figures/supp_fig4_roc_pr_test.py` | `protocol/01`, `protocol/09` | `4_paper/results/fig4_roc_pr_test.pdf` |
| Fig. 5 Temporal persistence rule example | `4_paper/figures/supp_fig5_temporal_rule_timeline.py` | `protocol/03` | `4_paper/results/fig5_temporal_rule_timeline.pdf` |

## Sections

| Paper | Content | Script | Result file |
|---|---|---|---|
| §5.3 | Spectral-concentration threshold of the quality gate | `4_paper/analyses/supp_sqi_spectral_threshold_check.py` | `4_paper/results/sqi_spectral_*` |
| §5.3 | DEV/TEST re-evaluated with the inference service's quality gate | `4_paper/analyses/supp_service_sqi_gate_dev_test.py` | `4_paper/results/service_sqi_gate_summary.json` |
| §5.6 | Platt-scaling diagnostic | `protocol/08` | `3_evaluation/results/calibration_summary.csv`, `FROZEN_PLATT_CALIBRATOR.json` |
| §5.7, §6.8, App. B | Temporal persistence rule: selection on DEV, evaluation on TEST | `protocol/04`–`07`, `protocol/10` | `3_evaluation/results/FROZEN_GLOBAL_RULE_V2.json`, `global_rule_v2_*.csv`, `*_global_rule_v2_subject_results.csv` |
| §6.4 | Window-level TEST metrics with subject-clustered bootstrap CIs | `protocol/09` | `3_evaluation/results/window_metrics_subject_cluster_bootstrap.csv` |
| §6.4, §7.3 | Leave-one-domain-out (LODO) analysis | `protocol/11`, `protocol/12`, `protocol/15`–`17` | `3_evaluation/results/lodo_*`, `deepbeat_lodo_*`, `deepbeat_domain_exposure_*` |
| §6.6 | Comparison with the published DeepBeat CNNs | `4_paper/analyses/supp_deepbeat_cnn_{export,run,compare}.py` | `4_paper/results/deepbeat_cnn_comparison_summary.json` |
| §6.9 | Computational cost of the inference service | `scripts/benchmark_inference.py` in the AI Service repository | `docs/benchmark_v6_mimic_perform_af.json` in the AI Service repository |
| App. B | Software environment that reloads the frozen model | — | `requirements-v6-repro.txt` |
| — | Research freeze (hashes of the frozen model, rule and calibrator) | `protocol/18` | `3_evaluation/results/V6_RESEARCH_FREEZE*` |

The feature-parity test between this pipeline and the inference service (§6.3) is
`tests/test_parity_v6.py` in the AI Service repository.
