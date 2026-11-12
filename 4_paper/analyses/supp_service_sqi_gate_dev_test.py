"""Phân tích bổ sung: đánh giá lại DEV/TEST khi lọc cửa sổ bằng đúng cổng SQI của dịch vụ suy luận.

Bối cảnh: phân tích hồi cứu trong bài giữ cửa sổ có >= 10 khoảng NN và 14 đặc trưng hữu hạn, còn dịch vụ
(AI Service) dùng cổng SQI: spectral concentration >= 0,70, >= 10 nhịp hợp lệ, >= 80 % khoảng NN hợp lệ,
nhịp tim 30-220 bpm. Script này áp đúng cổng của dịch vụ (app/services/hrv_v4.py, hàm
extract_features_from_ppg) lên tín hiệu thô của từng cửa sổ DEV/TEST đã khoá, rồi tính lại chỉ số.

Mô hình và dự đoán giữ nguyên: P_meta và sigma_dis lấy từ dự đoán khoá (3_evaluation/results/*_verified_streams.csv).
Đặc trưng của dịch vụ trùng với pipeline tới sai số tương đối 1e-9 (tests/test_parity_v6.py của AI Service),
nên chỉ cần quyết định qua/không qua cổng cho từng cửa sổ.

Không tái tạo được cách cắt cửa sổ 30 s / bước 10 s của dịch vụ: dữ liệu thô chỉ có đoạn cắt sẵn (DeepBeat
25 s ở 32 Hz, PulseWatch 30 s ở 50 Hz, Liu 10 s ở 100 Hz). Tín hiệu mỗi cửa sổ được dựng lại đúng như lúc
trích đặc trưng: DeepBeat lấy signal[global_index]; PulseWatch đọc file đoạn trong tar rồi chuẩn hoá thời
gian bằng chính hàm của 1_prepare_data/scripts/v5_extractors/pulsewatch_frozen_v5_external.py; Liu ghép ba đoạn 10 s
(idx_0, idx_1, idx_2).

Kiểm tra: khi không áp cổng, script phải tái lập AUROC/AP và kết quả quy tắc theo người của bài.

Chạy từ gốc repo (cần bản AI Service và file raw-001.zip của nhóm):
  PYTHONPATH=. python 4_paper/analyses/supp_service_sqi_gate_dev_test.py \
      --raw-zip <raw-001.zip> --ai-service <thư mục AI Service>
Đầu ra: 4_paper/results/service_sqi_gate_summary.json (tổng hợp, commit được) và
4_paper/results/service_sqi_gate_window_predictions.csv (theo cửa sổ, gitignore).
"""

import argparse
import ast
import io
import json
import os
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from numpy.lib import format as npf
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "3_evaluation/results"
OUT = ROOT / "4_paper/results"
RULE = json.loads((ART / "FROZEN_GLOBAL_RULE_V2.json").read_text(encoding="utf-8"))
SEED, DRAWS = 20261112, 1000
THRESHOLDS = (0.5, 0.8, 0.9)


def load_v5_pulsewatch_timebase():
    """Lấy nguyên văn các hàm chuẩn hoá thời gian của pipeline PulseWatch (không chạy phần còn lại)."""
    src = (ROOT / "1_prepare_data/scripts/v5_extractors/pulsewatch_frozen_v5_external.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ("parse_ppg", "nominal_timebase", "normalize_timebase")
            or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("N_SAMPLES", "TARGET_DT_MS", "FS")
                                                for t in n.targets)]
    ns = {"np": np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "pulsewatch_v5_timebase", "exec"), ns)
    return ns


def read_npy_rows(zf: zipfile.ZipFile, member: str, rows: np.ndarray) -> np.ndarray:
    """Đọc mảng signal.npy trong file npz (nằm trong raw-001.zip) rồi lấy các hàng cần."""
    inner = zipfile.ZipFile(zf.open(member))
    with inner.open("signal.npy") as fh:
        version = npf.read_magic(fh)
        shape, fortran, dtype = (npf.read_array_header_1_0(fh) if version == (1, 0)
                                 else npf.read_array_header_2_0(fh))
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        header_end = fh.tell()
        out = np.empty((len(rows),) + tuple(shape[1:]), dtype=dtype)
        for k in np.argsort(rows):  # tăng dần, nên chỉ nhảy tới phía trước
            fh.seek(header_end + int(rows[k]) * row_bytes)
            out[k] = np.frombuffer(fh.read(row_bytes), dtype=dtype).reshape(shape[1:])
    return out


def gate(ppg: np.ndarray, fs: float, extract) -> str:
    """'' nếu qua cổng, ngược lại là danh sách lý do (spectral, beats, valid_ratio, hr, other)."""
    ppg = np.asarray(ppg, dtype=float)
    time_ms = np.arange(len(ppg)) * 1000.0 / fs
    try:
        extract(time_ms, ppg, fs=fs)
        return ""
    except ValueError as err:  # PoorSignalQualityError kế thừa ValueError
        msg = str(err)
        reasons = [name for name, key in (("spectral", "spectral concentration"), ("beats", "nhịp hợp lệ"),
                                          ("valid_ratio", "phi sinh lý"), ("hr", "ngoài dải sinh lý"))
                   if key in msg]
        return "+".join(reasons) or "other"
    except Exception as err:  # lỗi tính toán khác: coi như không qua cổng
        return "error:" + type(err).__name__


def deepbeat_signals(zf, df):
    out = {}
    for split, npz in (("DEV", "validate.npz"), ("TEST", "test.npz")):
        part = df[df["split"] == split]
        member = next(n for n in zf.namelist() if n.endswith("/" + npz))
        sig = read_npy_rows(zf, member, part["global_index"].to_numpy())
        for idx, s in zip(part.index, sig):
            out[idx] = (s.reshape(-1), 32.0)
    return out


def pulsewatch_signals(zf, df, tb):
    wanted = {str(s): idx for idx, s in zip(df.index, df["segment"])}
    tars = [n for n in zf.namelist() if n.endswith("_final.tar")]
    subjects = {str(int(s)).zfill(3) for s in df["subject_id"].unique()}
    out = {}
    for tar_name in sorted(tars):
        if tar_name.split("/")[-1].split("_")[0] not in subjects:
            continue
        with zf.open(tar_name) as fh, tarfile.open(fileobj=fh, mode="r|*") as tf:
            for mem in tf:
                if not mem.isfile():
                    continue
                key = os.path.splitext(os.path.basename(mem.name))[0]
                if key not in wanted:
                    continue
                arr = np.loadtxt(io.StringIO(tf.extractfile(mem).read().decode("utf-8")), dtype=np.float64)
                ppg, timestamp, _ = tb["parse_ppg"](arr)
                y, _info = tb["nominal_timebase"](ppg) if timestamp is None else tb["normalize_timebase"](ppg, timestamp)
                out[wanted[key]] = (y, 50.0)
    return out


def liu_signals(zf, df):
    mats = {n.split("/")[-1].replace(".mat", ""): n for n in zf.namelist()
            if n.endswith(".mat") and "PPGArrhythmiaDetection-main/" in n and "/PPGArrhythmiaDetection-main.zip" not in n}
    out = {}
    for subject, part in df.groupby("subject_id"):
        m = sio.loadmat(io.BytesIO(zf.read(mats[str(subject)])), squeeze_me=True, struct_as_record=False)
        ppg = np.asarray(m["ppgseg"], dtype=float)
        labels = np.asarray(m["labels"]).reshape(-1)
        if ppg.ndim == 1:
            ppg = ppg.reshape(1, -1)
        if ppg.shape[0] != len(labels) and ppg.shape[1] == len(labels):
            ppg = ppg.T
        for idx, r in part.iterrows():
            out[idx] = (np.concatenate([ppg[int(r["idx_0"])], ppg[int(r["idx_1"])], ppg[int(r["idx_2"])]]), 100.0)
    return out


def bootstrap_ci(df: pd.DataFrame, metric) -> list:
    rng = np.random.default_rng(SEED)
    codes, uniques = pd.factorize(df["subject_id"])
    order = np.argsort(codes, kind="stable")
    by_subject = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
    y, p = df["label"].to_numpy(), df["meta_probability"].to_numpy()
    vals = []
    for _ in range(DRAWS):
        ii = np.concatenate([by_subject[k] for k in rng.integers(len(uniques), size=len(uniques))])
        if len(np.unique(y[ii])) == 2:
            vals.append(metric(y[ii], p[ii]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def window_metrics(df: pd.DataFrame) -> dict:
    y, p = df["label"].to_numpy(), df["meta_probability"].to_numpy()
    out = {"windows": int(len(df)), "af_windows": int(y.sum()), "subjects": int(df["subject_id"].nunique())}
    if len(np.unique(y)) == 2:
        out.update(AUROC=round(float(roc_auc_score(y, p)), 4), AUROC_CI95=bootstrap_ci(df, roc_auc_score),
                   AP=round(float(average_precision_score(y, p)), 4), AP_CI95=bootstrap_ci(df, average_precision_score))
        for t in THRESHOLDS:
            out[f"sensitivity@{t}"] = round(float((p[y == 1] >= t).mean()), 4)
            out[f"specificity@{t}"] = round(float((p[y == 0] < t).mean()), 4)
    return out


def longest_run(values) -> int:
    best = cur = 0
    for v in values:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def subject_rule(df: pd.DataFrame, use_gate: bool) -> dict:
    b, r = RULE["base"], RULE["rescue"]
    tp = fn = tn = fp = 0
    for _, subject in df.groupby("subject_id", sort=False):
        base_run = rescue_run = 0
        for _, stream in subject.groupby("stream_id", sort=False):
            s = stream.sort_values("order_in_stream")
            ok = s["gate"].eq("").to_numpy() if use_gate else np.ones(len(s), bool)
            p, sd = s["meta_probability"].to_numpy(), s["ensemble_std"].to_numpy()
            base_run = max(base_run, longest_run(ok & (p >= b["probability_threshold"]) & (sd <= b["uncertainty_threshold"])))
            rescue_run = max(rescue_run, longest_run(ok & (p >= r["probability_threshold"]) & (sd <= r["uncertainty_threshold"])))
        alert = base_run >= b["min_consecutive"] or rescue_run >= r["min_consecutive"]
        if subject["label"].max() == 1:
            tp += alert; fn += not alert
        else:
            fp += alert; tn += not alert
    return {"af_subjects": tp + fn, "alerted_af": tp, "non_af_subjects": tn + fp, "alerted_non_af": fp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-zip", type=Path, required=True)
    ap.add_argument("--ai-service", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.ai_service))
    from app.services.hrv_v4 import extract_features_from_ppg  # cổng SQI của dịch vụ

    zf = zipfile.ZipFile(args.raw_zip)
    tb = load_v5_pulsewatch_timebase()
    summary, frames = {}, []
    for ds in ("deepbeat", "pulsewatch", "liu"):
        df = pd.concat([pd.read_csv(ART / f"{ds}_{sp}_verified_streams.csv", low_memory=False) for sp in ("dev", "test")],
                       ignore_index=True)
        df["split"] = df["split"].astype(str).str.upper().str.replace("REUSED_", "").str.replace("TEST_REUSED", "TEST")
        df.loc[df["split"].str.contains("TEST"), "split"] = "TEST"
        signals = (deepbeat_signals(zf, df) if ds == "deepbeat" else
                   pulsewatch_signals(zf, df, tb) if ds == "pulsewatch" else liu_signals(zf, df))
        df["gate"] = [gate(*signals[i], extract_features_from_ppg) if i in signals else "missing_raw" for i in df.index]
        frames.append(df[["dataset", "split", "subject_id", "label", "meta_probability", "ensemble_std", "gate"]])
        summary[ds] = {}
        for sp, part in df.groupby("split"):
            gated = part[part["gate"] == ""]
            reasons = part.loc[part["gate"] != "", "gate"].value_counts().to_dict()
            summary[ds][sp] = {
                "windows": int(len(part)),
                "passed_gate": int(len(gated)),
                "rejected_af_windows": int(((part["gate"] != "") & (part["label"] == 1)).sum()),
                "rejected_non_af_windows": int(((part["gate"] != "") & (part["label"] == 0)).sum()),
                "rejection_reasons": reasons,
                "all_windows": window_metrics(part),
                "gated_windows": window_metrics(gated),
                "rule_v2_all_windows": subject_rule(part, use_gate=False),
                "rule_v2_service_gate": subject_rule(part, use_gate=True),
            }
            print(ds, sp, json.dumps({k: summary[ds][sp][k] for k in ("windows", "passed_gate", "rejection_reasons")}))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(OUT / "service_sqi_gate_window_predictions.csv", index=False)
    (OUT / "service_sqi_gate_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", OUT / "service_sqi_gate_summary.json")


if __name__ == "__main__":
    main()
