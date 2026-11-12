"""Hình 5 của bản thảo: quy tắc temporal persistence trên hai người tham gia PulseWatch TEST.

Vẽ lại từ dự đoán khoá (locked predictions) của model V6C đã đóng băng
(2_model/models/healthsense_af_v6c_stacking.pkl, SHA-256 e27370c5...). Trục thời gian là thời gian
thật trong stream: các đoạn PulseWatch liên tiếp, không chồng lấn, dài khoảng 30 s.

Cần chạy trước protocol 01 và 03 để có 3_evaluation/results/pulsewatch_test_verified_streams.csv
(file này dẫn xuất từ dữ liệu Synapse nên được gitignore). Kiểm tra: kết quả theo người tạo lại
phải trùng 3_evaluation/results/pulsewatch_global_rule_v2_subject_results.csv.

Chọn stream:
(a) người 419 (AF): stream có run dài nhất của nhánh hai cửa sổ;
(b) người 90 (không AF): stream dài nhất có chứa một cửa sổ P_meta >= 0.90.

Chạy từ gốc repo:  PYTHONPATH=. python 4_paper/figures/supp_fig5_temporal_rule_timeline.py
Đầu ra: 4_paper/results/fig5_temporal_rule_timeline.pdf và fig5_temporal_rule_timeline.json
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "3_evaluation/results/pulsewatch_test_verified_streams.csv"
OUT = ROOT / "4_paper/results"
OUT.mkdir(parents=True, exist_ok=True)

P5, P2, SIG = 0.80, 0.90, 0.05  # như FROZEN_GLOBAL_RULE_V2.json
K5, K2 = 5, 2

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "pdf.fonttype": 42})


def stream_table(g):
    g = g.sort_values("order_in_stream").copy()
    dur = g["duration_ms"].to_numpy(float) / 1000.0
    g["t_end"] = np.cumsum(dur)
    g["q2"] = (g["meta_probability"] >= P2) & (g["ensemble_std"] <= SIG)
    g["q5"] = (g["meta_probability"] >= P5) & (g["ensemble_std"] <= SIG)
    return g


def longest_run(q):
    best = cur = 0
    for v in q:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def first_alert(g):
    r2 = r5 = 0
    for i, (q2, q5) in enumerate(zip(g["q2"], g["q5"])):
        r2 = r2 + 1 if q2 else 0
        r5 = r5 + 1 if q5 else 0
        if r2 >= K2:
            return i, "two-window high-confidence branch"
        if r5 >= K5:
            return i, "five-window persistence branch"
    return None, None


d = pd.read_csv(SRC, low_memory=False)
d["subject_id"] = d["subject_id"].astype(str)

s419 = {sid: stream_table(g) for sid, g in d[d["subject_id"] == "419"].groupby("stream_id", sort=False)}
sid_a = max(s419, key=lambda s: (longest_run(s419[s]["q2"]), len(s419[s])))
ga = s419[sid_a]
ia, branch_a = first_alert(ga)

s90 = {sid: stream_table(g) for sid, g in d[d["subject_id"] == "90"].groupby("stream_id", sort=False)}
cands = [s for s, g in s90.items() if (g["meta_probability"] >= P2).any()]
sid_b = max(cands, key=lambda s: (len(s90[s]), s90[s]["meta_probability"].max()))
gb = s90[sid_b]
imax = gb["meta_probability"].idxmax()
ib, _ = first_alert(gb)

fig, axes = plt.subplots(1, 2, figsize=(5.9, 3.2), sharey=True)
for ax, g, tag in ((axes[0], ga, "(a)"), (axes[1], gb, "(b)")):
    t = g["t_end"].to_numpy()
    p = g["meta_probability"].to_numpy()
    low = (g["ensemble_std"] <= SIG).to_numpy()
    ax.plot(t, p, color="0.55", linewidth=0.8, zorder=1)
    ax.scatter(t[low], p[low], marker="o", s=22, color="#1f4e79", zorder=3,
               label=r"$\sigma_{\mathrm{dis}} \leq 0.05$")
    ax.scatter(t[~low], p[~low], marker="x", s=24, color="#c55a11", zorder=3,
               label=r"$\sigma_{\mathrm{dis}} > 0.05$")
    ax.axhline(P5, color="0.3", linestyle="--", linewidth=0.8, label="0.80: five-window persistence branch")
    ax.axhline(P2, color="0.3", linestyle=":", linewidth=0.9, label="0.90: two-window high-confidence branch")
    ax.set_xlabel("Time within stream (s)")
    ax.set_ylim(-0.02, 1.05)
    ax.text(0.01, 1.02, tag, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
axes[0].set_ylabel(r"AF probability $P_{\mathrm{meta}}$")
t_alert = float(ga["t_end"].iloc[ia])
axes[0].axvline(t_alert, color="#7b2c8f", linewidth=1.0)
axes[0].text(t_alert, 0.08, f"  alert at {t_alert:.0f} s", color="#7b2c8f", fontsize=8, ha="left")
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.16, 1, 1))
fig.savefig(OUT / "fig5_temporal_rule_timeline.pdf")

info = {
    "panel_a": {"subject": "419", "stream_id": str(sid_a), "windows": int(len(ga)),
                "alert_branch": branch_a, "alert_time_s": round(t_alert, 1),
                "first_windows_t_p_sigma": [[round(float(x), 1), round(float(y), 4), round(float(z), 4)]
                                            for x, y, z in zip(ga["t_end"][:4], ga["meta_probability"][:4],
                                                               ga["ensemble_std"][:4])]},
    "panel_b": {"subject": "90", "stream_id": str(sid_b), "windows": int(len(gb)), "alert": ib is not None,
                "peak_p": round(float(gb.loc[imax, "meta_probability"]), 4),
                "peak_sigma": round(float(gb.loc[imax, "ensemble_std"]), 4),
                "peak_time_s": round(float(gb.loc[imax, "t_end"]), 1)},
}
(OUT / "fig5_temporal_rule_timeline.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
print(json.dumps(info, indent=1))
