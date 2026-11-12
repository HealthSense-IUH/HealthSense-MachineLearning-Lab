# 3. Evaluation

- `protocol/01`–`18`: the locked evaluation protocol. It covers predictions on DEV/TEST (01–03), the temporal persistence rule chosen on DEV and tested on TEST (04–07, 10), calibration (08), subject-clustered bootstrap CIs (09), leave-one-domain-out (11–12, 15–17), the PPG_AC ablation (13), the PAC/PVC challenge (14) and the research freeze (18).
- `results/`: their outputs. `FROZEN_*` and `V6_RESEARCH_FREEZE*` are frozen records; do not edit them. Window-level predictions are not committed because they derive from licensed data.
- `03_locked_protocol.ipynb` runs the protocol; `review_visual.ipynb` displays the results.
