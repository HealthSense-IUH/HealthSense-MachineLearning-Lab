"""So sánh với mạng CNN DeepBeat đã công bố, bước 2: chạy các mạng DeepBeat (trong container TensorFlow).

Mô hình lấy từ repo công khai github.com/AshleyLab/deepbeat (commit f54004a, giấy phép GPL-3.0; không chép
vào repo này): deepbeat.h5 (đa nhiệm: chất lượng + nhịp) và deepbeat_singletask_pretrained.h5 (chỉ nhịp).
Có thể thêm mô hình khác bằng --models. Trước khi nạp, script đọc cấu hình JSON của file .h5 và từ chối nếu
có lớp ngoài danh sách cho phép (ví dụ Lambda hay lớp tuỳ biến, có thể chứa mã chạy được).

Chạy (TensorFlow 2.15, không cần mạng):
  docker run --rm --network none -v <work>:/work -v <repo>/4_paper/analyses:/scripts:ro \
      tensorflow/tensorflow:2.15.0 python /scripts/supp_deepbeat_cnn_run.py --work /work
Đầu ra: <work>/cnn_predictions.npz, xác suất AF (đầu ra nhịp, cột 1) cho từng cửa sổ.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ALLOWED = {
    "InputLayer", "Conv1D", "BatchNormalization", "Activation", "LeakyReLU", "MaxPooling1D", "Dropout",
    "Flatten", "Dense", "Concatenate", "Add", "GlobalAveragePooling1D", "AveragePooling1D", "ZeroPadding1D",
    "Reshape", "ReLU", "Softmax", "GlobalMaxPooling1D", "Functional", "Model", "Sequential",
    # initializer và regularizer của các lớp chuẩn: chỉ là cấu hình, không chứa mã
    "Zeros", "Ones", "Constant", "VarianceScaling", "GlorotUniform", "GlorotNormal", "HeNormal", "HeUniform",
    "RandomNormal", "RandomUniform", "TruncatedNormal", "Orthogonal", "L1L2", "L1", "L2",
}


def layer_classes(node, found=None) -> set:
    found = set() if found is None else found
    if isinstance(node, dict):
        if "class_name" in node and isinstance(node.get("config"), dict):
            found.add(node["class_name"])
        for v in node.values():
            layer_classes(v, found)
    elif isinstance(node, list):
        for v in node:
            layer_classes(v, found)
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["deepbeat.h5", "deepbeat_singletask_pretrained.h5"])
    args = ap.parse_args()

    import tensorflow as tf

    X = np.load(args.work / "deepbeat_devtest_X.npy")
    results = {}
    for name in args.models:
        with h5py.File(args.work / name, "r") as f:
            raw = f.attrs["model_config"]
            cfg = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        blocked = sorted(layer_classes(cfg) - ALLOWED)
        if blocked:
            raise SystemExit(f"{name}: lớp không nằm trong danh sách cho phép: {blocked}")
        model = tf.keras.models.load_model(args.work / name, compile=False)
        out = model.predict(X, batch_size=256, verbose=0)
        rhythm = out[1] if isinstance(out, list) else out  # deepbeat.h5: [chất lượng, nhịp]
        results[name.removesuffix(".h5")] = rhythm[:, 1].astype(np.float32)
        print(name, model.input_shape, [tuple(o.shape) for o in model.outputs])
    np.savez(args.work / "cnn_predictions.npz", **results)


if __name__ == "__main__":
    main()
