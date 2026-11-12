"""Phân tích bổ sung: stacking so với từng base learner và baseline đơn giản trên DEV.

Trả lời câu hỏi phản biện "vì sao cần stacking?". Thiết lập giữ nguyên như
3_evaluation/protocol/13_ppg_ac_ablation_dev.py:
- cùng bảng huấn luyện 1_prepare_data/data/features/v7/healthsense_af_v6c_pacaware_train.csv,
  cùng bộ lọc dòng hợp lệ trên 14 đặc trưng;
- OOF cho meta-learner bằng GroupKFold(5) nhóm theo domain:subject_id;
- siêu tham số base learner/meta-learner như 2_model/train_v6c_stacking.py;
- đánh giá trên cùng ba cohort DEV (DeepBeat, PulseWatch, Liu). KHÔNG chạm TEST.

So sánh: stacking, soft voting (trung bình xác suất ba base learner), từng base
learner, logistic regression 14 đặc trưng, logistic regression đơn biến trên
RMSSD và CV. Khác biệt macro được ước lượng bằng paired subject-cluster bootstrap.

Chạy từ gốc repo:  PYTHONPATH=. python 4_paper/analyses/supp_stacking_vs_single_dev.py
Đầu ra: 4_paper/results/stacking_vs_single_dev_*.csv và *_run_info.json
(file dự đoán theo cửa sổ dẫn xuất từ dữ liệu Synapse nên không commit).
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "4_paper/results"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "HR_mean", "Mean_NN", "SDNN", "RMSSD", "NN50", "pNN50", "CV",
    "HF", "Total_Power", "HF_norm", "SD1", "SD2", "SampEn", "PPG_AC",
]
BASE = {
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss", tree_method="hist",
        random_state=42),
}
META = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
LINEAR = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
REF = "Stacking (ET+RF+XGB, LR meta)"
N_BOOT = 1000
SEED = 20261111


def common_valid_mask(df):
    mask = df[FEATURES].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    if "feature_ok" in df.columns:
        mask &= df["feature_ok"].astype(str).str.lower().isin(["1", "true", "yes"])
    return mask


def metrics(y, p):
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "PR_AUC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
    }


def load_dev():
    """Ba cohort DEV, đọc giống hệt 13_ppg_ac_ablation_dev.py."""
    out = {}
    x = pd.read_csv(ROOT / "1_prepare_data/data/features/v6/deepbeat_validate_v6_14f.csv", low_memory=False)
    x["label_eval"] = pd.to_numeric(x["label"], errors="raise").astype(int)
    x["subject_eval"] = x["subject_id"].astype(str)
    out["deepbeat"] = x

    x = pd.read_csv(ROOT / "1_prepare_data/data/splits/pulsewatch_dev.csv", low_memory=False)
    if "class_name" in x.columns:
        x["label_eval"] = x["class_name"].astype(str).str.upper().eq("AF").astype(int)
    else:
        x["label_eval"] = pd.to_numeric(x["label"], errors="raise").eq(1).astype(int)
    x["subject_eval"] = x["uid"].astype(str)
    out["pulsewatch"] = x

    x = pd.read_csv(ROOT / "1_prepare_data/data/splits/liu_dev.csv", low_memory=False)
    if "binary_label_clean" in x.columns:
        x["label_eval"] = pd.to_numeric(x["binary_label_clean"], errors="raise").astype(int)
    elif "binary_label" in x.columns:
        x["label_eval"] = pd.to_numeric(x["binary_label"], errors="raise").astype(int)
    else:
        x["label_eval"] = x["rhythm"].astype(str).str.upper().eq("AF").astype(int)
    x["subject_eval"] = x["subject_id"].astype(str)
    out["liu"] = x
    return {k: v.loc[common_valid_mask(v)].reset_index(drop=True) for k, v in out.items()}


t0 = time.time()

# ---------- Huấn luyện ----------
train_raw = pd.read_csv(ROOT / "1_prepare_data/data/features/v7/healthsense_af_v6c_pacaware_train.csv", low_memory=False)
train = train_raw.loc[common_valid_mask(train_raw)].reset_index(drop=True)
train["group_id"] = train["domain"].astype(str) + ":" + train["subject_id"].astype(str)
y = pd.to_numeric(train["label_v7"], errors="raise").astype(int).to_numpy()
groups = train["group_id"].to_numpy()
X = train[FEATURES]
print(f"train rows={len(train)} groups={train['group_id'].nunique()} labels={np.bincount(y).tolist()}", flush=True)

dev = load_dev()
for k, d in dev.items():
    print(f"DEV {k}: rows={len(d)} subjects={d['subject_eval'].nunique()} AF={int(d['label_eval'].sum())}", flush=True)

names = list(BASE)
splits = list(GroupKFold(n_splits=5).split(X, y, groups))
oof = np.zeros((len(train), len(names)))
for f, (tr, va) in enumerate(splits, 1):
    for j, n in enumerate(names):
        oof[va, j] = clone(BASE[n]).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    print(f"fold {f}/5 ({time.time() - t0:.0f}s)", flush=True)
meta = clone(META).fit(oof, y)

final = {n: clone(BASE[n]).fit(X, y) for n in names}
linear = clone(LINEAR).fit(X, y)
uni = {f: clone(LINEAR).fit(train[[f]], y) for f in ["RMSSD", "CV"]}

# ---------- Đánh giá DEV ----------
rows, preds = [], []
for k, d in dev.items():
    yd = d["label_eval"].to_numpy()
    base_p = np.column_stack([final[n].predict_proba(d[FEATURES])[:, 1] for n in names])
    cand = {
        REF: meta.predict_proba(base_p)[:, 1],
        "Soft voting (mean of ET, RF, XGB)": base_p.mean(axis=1),
        "ExtraTrees only": base_p[:, 0],
        "RandomForest only": base_p[:, 1],
        "XGBoost only": base_p[:, 2],
        "Logistic regression (14 features)": linear.predict_proba(d[FEATURES])[:, 1],
        "RMSSD only (univariate LR)": uni["RMSSD"].predict_proba(d[["RMSSD"]])[:, 1],
        "CV only (univariate LR)": uni["CV"].predict_proba(d[["CV"]])[:, 1],
    }
    for model, p in cand.items():
        rows.append({"model": model, "dataset": k, "rows": len(d),
                     "subjects": d["subject_eval"].nunique(), **metrics(yd, p)})
        preds.append(pd.DataFrame({"model": model, "dataset": k, "subject_id": d["subject_eval"].to_numpy(),
                                   "label": yd, "probability": p}))

summary = pd.DataFrame(rows)
macro = summary.groupby("model", sort=False)[["AUROC", "PR_AUC", "Brier"]].mean().reset_index()
pred_df = pd.concat(preds, ignore_index=True)

# ---------- Paired subject-cluster bootstrap (macro qua ba DEV) ----------
others = [m for m in macro["model"] if m != REF]
arr = {}
for k in dev:
    g = pred_df[pred_df["dataset"] == k]
    gs = g[g["model"] == REF]
    sid = gs["subject_id"].to_numpy()
    arr[k] = {
        "label": gs["label"].to_numpy(),
        "subjects": np.unique(sid),
        "rows_of": {s: np.flatnonzero(sid == s) for s in np.unique(sid)},
        "p": {m: g[g["model"] == m]["probability"].to_numpy() for m in [REF, *others]},
    }
rng = np.random.default_rng(SEED)
boot = {m: {"AUROC": [], "PR_AUC": [], "Brier": []} for m in others}
for _ in range(N_BOOT):
    per = {m: {"AUROC": [], "PR_AUC": [], "Brier": []} for m in others}
    valid = True
    for a in arr.values():
        pick = rng.choice(a["subjects"], size=len(a["subjects"]), replace=True)
        idx = np.concatenate([a["rows_of"][s] for s in pick])
        yb = a["label"][idx]
        if yb.min() == yb.max():  # bỏ lượt chỉ có một lớp ở một domain
            valid = False
            break
        mr = metrics(yb, a["p"][REF][idx])
        for m in others:
            mm = metrics(yb, a["p"][m][idx])
            per[m]["AUROC"].append(mr["AUROC"] - mm["AUROC"])
            per[m]["PR_AUC"].append(mr["PR_AUC"] - mm["PR_AUC"])
            per[m]["Brier"].append(mm["Brier"] - mr["Brier"])  # dương = stacking tốt hơn
    if valid:
        for m in others:
            for key in boot[m]:
                boot[m][key].append(float(np.mean(per[m][key])))

ci_rows = []
for m, d in boot.items():
    r = {"comparator": m, "valid_draws": len(d["AUROC"])}
    for key in ["AUROC", "PR_AUC", "Brier"]:
        a = np.array(d[key])
        r[f"d{key}_mean"] = float(a.mean())
        r[f"d{key}_lo"] = float(np.percentile(a, 2.5))
        r[f"d{key}_hi"] = float(np.percentile(a, 97.5))
    ci_rows.append(r)
ci = pd.DataFrame(ci_rows)

# ---------- Ghi ----------
summary.to_csv(OUT / "stacking_vs_single_dev_summary.csv", index=False)
macro.to_csv(OUT / "stacking_vs_single_dev_macro.csv", index=False)
ci.to_csv(OUT / "stacking_vs_single_dev_bootstrap_delta.csv", index=False)
pred_df.to_csv(OUT / "stacking_vs_single_dev_predictions.csv", index=False)  # gitignored
run_info = {
    "train_rows": int(len(train)),
    "train_groups": int(train["group_id"].nunique()),
    "dev_rows": {k: int(len(d)) for k, d in dev.items()},
    "bootstrap_draws": N_BOOT,
    "bootstrap_valid_draws": int(ci["valid_draws"].iloc[0]) if len(ci) else 0,
    "bootstrap_seed": SEED,
    "meta_coefficients_standardized_ET_RF_XGB": [float(c) for c in meta.named_steps["lr"].coef_[0]],
    "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                 "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
    "seconds": round(time.time() - t0),
}
(OUT / "stacking_vs_single_dev_run_info.json").write_text(json.dumps(run_info, indent=1), encoding="utf-8")

pd.set_option("display.width", 200)
fmt = lambda v: f"{v:.4f}"
print("\nPER DATASET\n", summary.to_string(index=False, float_format=fmt))
print("\nMACRO\n", macro.to_string(index=False, float_format=fmt))
print("\nSTACKING - COMPARATOR (macro, 95% bootstrap CI; Brier dương = stacking tốt hơn)\n",
      ci.to_string(index=False, float_format=fmt))
print(json.dumps(run_info, indent=1))
