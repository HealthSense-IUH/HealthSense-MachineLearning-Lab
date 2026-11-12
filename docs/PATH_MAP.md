# Map of old to new paths

On 2026-11-12 the repository was reorganised into numbered folders. Files were moved with `git mv`, so
`git log --follow` shows their full history. File contents were not changed, apart from the paths inside
code, notebooks and docs.

The research freeze records in `3_evaluation/results/` were **not** edited, so their SHA-256 hashes stay
valid. These records are `V6_RESEARCH_FREEZE_SHA256.csv`, `V6_RESEARCH_FREEZE_STATUS.json`,
`locked_inputs_SHA256.txt`, `protocol_metadata.json` and the `*_SHA256.txt` files. They still list the old
paths, and this table translates them. The same applies to `1_prepare_data/data/MANIFEST.csv` and the SHA
files in `1_prepare_data/data/splits/`.

| Old path | New path |
|---|---|
| `src/healthsense_core/` | `healthsense_core/` |
| `data/` | `1_prepare_data/data/` |
| `notebooks/v6/00_extract_features_and_splits.ipynb`, `01_prepare_training_table.ipynb` | `1_prepare_data/` |
| `notebooks/v6/scripts/` (manifests, memmap, feature extraction, splits, `build_v6b_*`, `build_v7_hard_negative_v2.py`) | `1_prepare_data/scripts/` |
| `notebooks/v5/scripts/` (PulseWatch, Liu, PeakDet extractors, whose outputs feed V6) | `1_prepare_data/scripts/v5_extractors/` |
| `results/v5/` (generated, not in git) | `1_prepare_data/v5_outputs/` |
| `notebooks/v6/02_train_stacking.ipynb`, `notebooks/v6/scripts/train_v6c_stacking.py` | `2_model/` |
| `models/v6/` | `2_model/models/` |
| `notebooks/v6/protocol/` | `3_evaluation/protocol/` |
| `notebooks/v6/03_locked_protocol.ipynb`, `review_visual.ipynb` | `3_evaluation/` |
| `results/v6/` | `3_evaluation/results/` |
| `notebooks/v6/scripts/supp_*.py` (analyses) | `4_paper/analyses/` |
| `notebooks/v6/scripts/supp_fig*.py` | `4_paper/figures/` |
| `results/v6/supplementary/` | `4_paper/results/` |
| `notebooks/v4/`, `results/v4/`, `models/v4/` | `archive/v4/` |
| `notebooks/v5/` (notebooks and the other scripts), `models/v5/` | `archive/v5/` |
| `README.md` (Vietnamese) | `docs/HISTORY_vi.md` |
