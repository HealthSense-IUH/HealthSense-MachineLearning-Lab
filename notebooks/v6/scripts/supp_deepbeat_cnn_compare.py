"""So sánh với mạng CNN DeepBeat đã công bố, bước 3: chỉ số trên cùng các cửa sổ DeepBeat TEST.

So sánh chính là DeepBeat TEST: bản test của bản phát hành DeepBeat, nên cả mạng DeepBeat lẫn bộ phân loại V6
đều không dùng các người tham gia này khi huấn luyện hay chọn mô hình. DeepBeat DEV (bản validate) được báo cáo
kèm nhưng không dùng để kết luận, vì DeepBeat đã dùng nó khi phát triển mạng.

Chỉ số: AUROC, AP, độ nhạy và độ đặc hiệu ở ngưỡng 0,5; khoảng tin cậy 95 % và hiệu AUROC ghép cặp
(V6 trừ CNN) bằng bootstrap theo người tham gia (1000 lượt, seed cố định; bỏ lượt chỉ có một lớp).

Chạy từ gốc repo:
  python notebooks/v6/scripts/supp_deepbeat_cnn_compare.py --work <thư mục làm việc>
Đầu ra: results/v6/supplementary/deepbeat_cnn_comparison_summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/v6/supplementary/deepbeat_cnn_comparison_summary.json"
SEED, DRAWS = 20261112, 1000
V6 = "V6 fixed stacked classifier"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()

    meta = pd.read_csv(args.work / "deepbeat_devtest_meta.csv")
    pred = np.load(args.work / "cnn_predictions.npz")
    models = {V6: meta["meta_probability"].to_numpy(), **{k: pred[k] for k in pred.files}}
    summary = {"note": "DeepBeat TEST is the primary comparison; DEV was used to develop the DeepBeat networks."}
    for split in ("TEST", "DEV"):
        m = meta["split"].to_numpy() == split
        y = meta.loc[m, "label"].to_numpy()
        codes, uniques = pd.factorize(meta.loc[m, "subject_id"])
        order = np.argsort(codes, kind="stable")
        groups = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
        rng = np.random.default_rng(SEED)
        draws = [np.concatenate([groups[k] for k in rng.integers(len(uniques), size=len(uniques))]) for _ in range(DRAWS)]
        draws = [ii for ii in draws if len(np.unique(y[ii])) == 2]
        v6 = models[V6][m]
        rows = {}
        for name, p_all in models.items():
            p = p_all[m]
            aucs = [roc_auc_score(y[ii], p[ii]) for ii in draws]
            row = {"AUROC": round(float(roc_auc_score(y, p)), 4),
                   "AUROC_CI95": [round(float(np.percentile(aucs, q)), 4) for q in (2.5, 97.5)],
                   "AP": round(float(average_precision_score(y, p)), 4),
                   "sensitivity@0.5": round(float((p[y == 1] >= 0.5).mean()), 4),
                   "specificity@0.5": round(float((p[y == 0] < 0.5).mean()), 4)}
            if name != V6:
                deltas = [roc_auc_score(y[ii], v6[ii]) - roc_auc_score(y[ii], p[ii]) for ii in draws]
                row["dAUROC_V6_minus_model"] = round(float(roc_auc_score(y, v6) - roc_auc_score(y, p)), 4)
                row["dAUROC_CI95"] = [round(float(np.percentile(deltas, q)), 4) for q in (2.5, 97.5)]
            rows[name] = row
        summary[split] = {"windows": int(m.sum()), "af_windows": int(y.sum()), "participants": int(len(uniques)),
                          "valid_bootstrap_draws": len(draws), "models": rows}
    OUT.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for split in ("TEST", "DEV"):
        print(split, json.dumps(summary[split]["models"], indent=None))


if __name__ == "__main__":
    main()
