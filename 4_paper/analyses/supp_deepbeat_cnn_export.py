"""So sánh với mạng CNN DeepBeat đã công bố, bước 1: trích tín hiệu các cửa sổ DeepBeat DEV/TEST.

Tín hiệu lấy đúng như lúc trích đặc trưng V6 (signal[global_index] trong validate.npz / test.npz của bản phát
hành DeepBeat): đoạn 25 s, 32 Hz, đã được DeepBeat chuẩn hoá về [0, 1], lọc thông dải và hạ mẫu. Đây cũng là
đầu vào của các mạng DeepBeat, nên hai mô hình được so trên đúng cùng cửa sổ và cùng tín hiệu.

Chạy từ gốc repo:
  PYTHONPATH=. python 4_paper/analyses/supp_deepbeat_cnn_export.py --raw-zip <raw-001.zip> --work <thư mục làm việc>
Đầu ra (ngoài repo): <work>/deepbeat_devtest_X.npy và <work>/deepbeat_devtest_meta.csv.
"""

import argparse
import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib import format as npf

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("gate", ROOT / "4_paper/analyses/supp_service_sqi_gate_dev_test.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-zip", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.raw_zip)
    df = pd.concat([pd.read_csv(ROOT / f"3_evaluation/results/deepbeat_{s}_verified_streams.csv", low_memory=False)
                    for s in ("dev", "test")], ignore_index=True)
    df["split"] = np.where(df["split"].astype(str).str.upper().str.contains("TEST"), "TEST", "DEV")
    signals = gate.deepbeat_signals(zf, df)
    X = np.stack([signals[i][0] for i in df.index]).astype(np.float32)[:, :, None]

    # Nhãn của bản phát hành DeepBeat (rhythm one-hot) phải trùng nhãn của V6: cột 1 là AF
    rhythm = {}
    for split, npz in (("DEV", "validate.npz"), ("TEST", "test.npz")):
        part = df[df["split"] == split]
        inner = zipfile.ZipFile(zf.open(next(n for n in zf.namelist() if n.endswith("/" + npz))))
        with inner.open("rhythm.npy") as fh:
            version = npf.read_magic(fh)
            shape, _, dtype = (npf.read_array_header_1_0(fh) if version == (1, 0) else npf.read_array_header_2_0(fh))
            arr = np.frombuffer(fh.read(), dtype=dtype).reshape(shape)
        for idx, gi in zip(part.index, part["global_index"]):
            rhythm[idx] = arr[int(gi), 1]
    agree = float((np.array([rhythm[i] for i in df.index]).round() == df["label"].to_numpy()).mean())
    if agree != 1.0:
        raise SystemExit(f"Nhãn DeepBeat và nhãn V6 không trùng ({agree:.3f})")

    args.work.mkdir(parents=True, exist_ok=True)
    np.save(args.work / "deepbeat_devtest_X.npy", X)
    df[["split", "subject_id", "label", "meta_probability", "ensemble_std"]].to_csv(
        args.work / "deepbeat_devtest_meta.csv", index=False)
    print("windows", X.shape, "label agreement", agree)


if __name__ == "__main__":
    main()
