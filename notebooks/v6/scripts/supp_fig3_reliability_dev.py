"""Hình 3 của bản thảo: reliability diagram của bộ phân loại cố định trên DEV (xác suất thô và Platt OOF).

Vẽ theo quy định hình của SN Computer Science: chữ không chân 8 pt ở kích thước in (rộng 135 mm),
phân biệt hai đường bằng marker và kiểu nét ngoài màu, đánh dấu ô (a)/(b) thay cho tiêu đề trong hình.

Đầu vào: results/v6/dev_platt_oof_predictions.csv, do protocol 08_calibration_diagnostics.py sinh ra
(file dẫn xuất từ dữ liệu Synapse nên được gitignore). Brier trong chú giải phải trùng
results/v6/calibration_summary.csv (DEV, RAW và PLATT_OOF).

Chạy từ gốc repo:  PYTHONPATH=src python notebooks/v6/scripts/supp_fig3_reliability_dev.py
Đầu ra: results/v6/supplementary/fig3_reliability_dev.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "results/v6/dev_platt_oof_predictions.csv"
OUT = ROOT / "results/v6/supplementary"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "pdf.fonttype": 42, "axes.linewidth": 0.6})

METHODS = [  # màu Okabe-Ito, thêm marker và kiểu nét để đọc được khi in đen trắng
    ("meta_probability", "Raw", "#0072B2", "o", "-"),
    ("calibrated_probability_oof", "Platt (OOF)", "#D55E00", "s", "--"),
]
PANELS = [("pulsewatch", "a"), ("deepbeat", "b")]

d = pd.read_csv(SRC, low_memory=False)
fig, axes = plt.subplots(1, 2, figsize=(5.3, 2.55), sharey=True)
for ax, (dataset, letter) in zip(axes, PANELS):
    x = d[d["dataset"] == dataset]
    y = x["label"].to_numpy()
    ax.plot([0, 1], [0, 1], color="0.6", linewidth=0.7, linestyle=":")
    for col, name, colour, marker, style in METHODS:
        p = x[col].to_numpy()
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="uniform")
        ax.plot(predicted, observed, color=colour, marker=marker, markersize=3.5, linestyle=style,
                linewidth=1.1, label=f"{name}, Brier {brier_score_loss(y, p):.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability (bin mean)")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.text(-0.16 if letter == "a" else -0.06, 1.07, f"({letter})", transform=ax.transAxes,
            fontweight="bold", va="top")
    ax.legend(loc="upper left", frameon=False, handlelength=2.4)
    ax.spines[["top", "right"]].set_visible(False)
    print(dataset, x["subject_id"].nunique(), "người tham gia,", len(x), "cửa sổ")
axes[0].set_ylabel("Observed AF fraction in bin")
fig.tight_layout(w_pad=1.0)
fig.savefig(OUT / "fig3_reliability_dev.pdf")
print("Đã lưu", OUT / "fig3_reliability_dev.pdf")
