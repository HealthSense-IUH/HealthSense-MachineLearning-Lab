"""Hình 4 của bản thảo: đường ROC và precision-recall của bộ phân loại cố định trên các tập TEST dùng lại.

Vẽ theo quy định hình của SN Computer Science: chữ không chân, phân biệt ba dataset bằng kiểu nét ngoài
màu, không đặt tiêu đề trong hình, gọi diện tích dưới đường PR là "AP" (average precision) như bản thảo.
Khi in ở 0.72 bề rộng chữ (108 mm), chữ trong hình cỡ 8-9 pt.

Đầu vào: dự đoán khoá 3_evaluation/results/<dataset>_test_locked_predictions.csv (protocol 03; gitignore vì dẫn xuất
từ dữ liệu Synapse) và khoảng tin cậy 3_evaluation/results/window_metrics_subject_cluster_bootstrap.csv (protocol 09).
AUROC/AP tính lại phải trùng bảng bootstrap: PulseWatch 0.9985/0.9890, DeepBeat 0.9904/0.9890,
Liu 0.9866/0.9780.

Chạy từ gốc repo:  PYTHONPATH=. python 4_paper/figures/supp_fig4_roc_pr_test.py
Đầu ra: 4_paper/results/fig4_roc_pr_test.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "3_evaluation/results"
OUT = ROOT / "4_paper/results"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.linewidth": 0.6})

DATASETS = [  # màu Okabe-Ito (an toàn cho người mù màu) và kiểu nét riêng
    ("pulsewatch", "PulseWatch", "#0072B2", "-"),
    ("deepbeat", "DeepBeat", "#D55E00", "--"),
    ("liu", "Liu 2022", "#009E73", "-."),
]

ci = pd.read_csv(ART / "window_metrics_subject_cluster_bootstrap.csv")
ci = ci[(ci["split"] == "REUSED_TEST") & (ci["method"] == "RAW")]


def interval(dataset, metric):
    row = ci[(ci["dataset"] == dataset) & (ci["metric"] == metric)].iloc[0]
    return row["CI95_low"], row["CI95_high"]


fig, (ax_roc, ax_pr) = plt.subplots(2, 1, figsize=(3.6, 5.6))
ax_roc.plot([0, 1], [0, 1], color="0.6", linewidth=0.7, linestyle=":")

for key, name, colour, style in DATASETS:
    d = pd.read_csv(ART / f"{key}_test_locked_predictions.csv")
    y, p = d["label"].to_numpy(), d["meta_probability"].to_numpy()

    fpr, tpr, _ = roc_curve(y, p)
    auroc = roc_auc_score(y, p)
    lo, hi = interval(key, "AUROC")
    ax_roc.plot(fpr, tpr, color=colour, linestyle=style, linewidth=1.2,
                label=f"{name}: AUROC {auroc:.4f} [{lo:.4f}, {hi:.4f}]")

    precision, recall, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    lo, hi = interval(key, "PR_AUC")
    ax_pr.step(recall, precision, where="post", color=colour, linestyle=style, linewidth=1.2,
               label=f"{name}: AP {ap:.4f} [{lo:.4f}, {hi:.4f}]")
    ax_pr.axhline(y.mean(), color=colour, linestyle=":", linewidth=0.8)  # tỉ lệ AF (mốc của AP)
    print(name, "AUROC", round(auroc, 4), "AP", round(ap, 4))

ax_roc.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate (1 − specificity)", ylabel="Sensitivity")
ax_pr.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
for ax, letter, legend_kw in (
    (ax_roc, "a", {"loc": "lower right"}),
    (ax_pr, "b", {"loc": "center left", "bbox_to_anchor": (0.0, 0.70)}),
):
    ax.text(-0.14, 1.06, f"({letter})", transform=ax.transAxes, fontweight="bold", va="top")
    ax.legend(frameon=False, handlelength=2.6, **legend_kw)
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(h_pad=1.2)
fig.savefig(OUT / "fig4_roc_pr_test.pdf")
print("Đã lưu", OUT / "fig4_roc_pr_test.pdf")
