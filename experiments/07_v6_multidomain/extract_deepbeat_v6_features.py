import sys
sys.path.insert(0, "src")

from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd

from healthsense_ml.signal_processing import (
    extract_nn_series,
    bandpass_filter,
)

from healthsense_ml.hrv_features import (
    compute_hrv_features,
)


FS = 32.0
MIN_NN = 10

FEATURES = [
    "HR_mean",
    "Mean_NN",
    "SDNN",
    "RMSSD",
    "NN50",
    "pNN50",
    "CV",
    "HF",
    "Total_Power",
    "HF_norm",
    "SD1",
    "SD2",
    "SampEn",
    "PPG_AC",
]


CONFIG = {
    "train": {
        "manifest":
            "experiments/07_v6_multidomain/"
            "artifacts/"
            "deepbeat_train_v6_classifier.csv",

        "signal":
            "data/cache/deepbeat_memmap/"
            "train/signal.npy",

        "output":
            "data/features/v6/"
            "deepbeat_train_v6_14f.csv",
    },

    "validate": {
        "manifest":
            "experiments/07_v6_multidomain/"
            "artifacts/"
            "deepbeat_validate_v6_classifier.csv",

        "signal":
            "data/cache/deepbeat_memmap/"
            "validate/signal.npy",

        "output":
            "data/features/v6/"
            "deepbeat_validate_v6_14f.csv",
    },

    "test_independent": {
        "manifest":
            "experiments/07_v6_multidomain/"
            "artifacts/"
            "deepbeat_test_independent_v6_classifier.csv",

        "signal":
            "data/cache/deepbeat_memmap/"
            "test/signal.npy",

        "output":
            "data/features/v6/"
            "deepbeat_test_independent_v6_14f.csv",
    },
}


def autocorrelation_feature(
    x,
    fs,
):
    # EXACT frozen-v5 definition
    x = np.asarray(
        x,
        dtype=float,
    )

    x = bandpass_filter(
        x,
        fs=fs,
    )

    x = x - np.mean(x)

    n = len(x)

    if n < 10:
        return np.nan

    nfft = 1 << (
        (2 * n - 1).bit_length()
    )

    spectrum = np.fft.rfft(
        x,
        n=nfft,
    )

    ac = np.fft.irfft(
        spectrum
        * np.conj(spectrum),
        n=nfft,
    )[:n]

    ac = ac / n

    if (
        not np.isfinite(ac[0])
        or abs(ac[0]) < 1e-12
    ):
        return np.nan

    ac = ac / ac[0]

    return float(
        np.mean(
            np.abs(
                ac[1:]
            )
        )
    )


def extract_one(
    x,
):
    nn_ms, nn_times = (
        extract_nn_series(
            x,
            fs=FS,
        )
    )

    n_nn = len(nn_ms)

    if n_nn < MIN_NN:
        return (
            False,
            "TOO_FEW_NN",
            n_nn,
            {}
        )

    feats = compute_hrv_features(
        nn_ms,
        nn_times,
    )

    feats["PPG_AC"] = (
        autocorrelation_feature(
            x,
            FS,
        )
    )

    finite = all(
        np.isfinite(
            feats[f]
        )
        for f in FEATURES
    )

    if not finite:
        return (
            False,
            "NONFINITE_FEATURE",
            n_nn,
            feats
        )

    return (
        True,
        "",
        n_nn,
        feats
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=CONFIG.keys(),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing output",
    )

    parser.add_argument(
        "--checkpoint",
        type=int,
        default=500,
    )

    args = parser.parse_args()

    cfg = CONFIG[
        args.split
    ]

    manifest_path = Path(
        cfg["manifest"]
    )

    signal_path = Path(
        cfg["signal"]
    )

    output_path = Path(
        args.output
        if args.output
        else cfg["output"]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest = pd.read_csv(
        manifest_path
    )

    if args.limit is not None:
        manifest = (
            manifest
            .head(args.limit)
            .copy()
        )

    signal = np.load(
        signal_path,
        mmap_mode="r",
    )

    print("=" * 100)
    print("DEEPBEAT V6 FEATURE EXTRACTION")
    print("=" * 100)

    print("Split    :", args.split)
    print("Manifest :", manifest_path)
    print("Signal   :", signal_path)
    print("Output   :", output_path)
    print("Rows     :", len(manifest))
    print("Signal   :", signal.shape)
    print("dtype    :", signal.dtype)
    print("FS       :", FS)
    print("MIN_NN   :", MIN_NN)
    print("Features :", FEATURES)

    required_manifest = [
        "global_index",
        "subject_id",
        "timestamp",
        "source",
        "label",
        "rhythm",
        "quality",
        "quality_name",
    ]

    missing = [
        c for c in required_manifest
        if c not in manifest.columns
    ]

    if missing:
        raise RuntimeError(
            f"Manifest missing columns: "
            f"{missing}"
        )

    # ----------------------------------------
    # RESUME
    # ----------------------------------------

    done = set()
    existing = None

    if (
        output_path.exists()
        and not args.fresh
    ):
        existing = pd.read_csv(
            output_path
        )

        if (
            "global_index"
            in existing.columns
        ):
            done = set(
                existing[
                    "global_index"
                ]
                .astype(int)
                .tolist()
            )

        print(
            "Resume existing rows:",
            len(done)
        )

    todo = manifest[
        ~manifest.global_index
        .astype(int)
        .isin(done)
    ].copy()

    print("Remaining:", len(todo))

    rows = []
    started = time.time()

    def flush():
        nonlocal rows

        if not rows:
            return

        chunk = pd.DataFrame(
            rows
        )

        header = (
            not output_path.exists()
            or output_path.stat().st_size == 0
            or args.fresh
        )

        mode = (
            "w"
            if args.fresh
            and header
            else "a"
        )

        chunk.to_csv(
            output_path,
            index=False,
            mode=mode,
            header=header,
        )

        args.fresh = False

        rows = []

    for k, r in enumerate(
        todo.itertuples(
            index=False
        ),
        start=1,
    ):
        gi = int(
            r.global_index
        )

        base = {
            "global_index":
                gi,

            "subject_id":
                int(
                    r.subject_id
                ),

            "timestamp":
                r.timestamp,

            "source":
                r.source,

            "label":
                int(
                    r.label
                ),

            "rhythm":
                r.rhythm,

            "quality":
                int(
                    r.quality
                ),

            "quality_name":
                r.quality_name,
        }

        try:
            x = np.asarray(
                signal[
                    gi,
                    :,
                    0
                ],
                dtype=float,
            )

            (
                feature_ok,
                error,
                n_nn,
                feats,
            ) = extract_one(x)

            row = {
                **base,

                "feature_ok":
                    bool(
                        feature_ok
                    ),

                "error":
                    error,

                "n_nn":
                    int(
                        n_nn
                    ),

                **{
                    f:
                        feats.get(
                            f,
                            np.nan
                        )
                    for f in FEATURES
                },
            }

        except Exception as e:
            row = {
                **base,

                "feature_ok":
                    False,

                "error":
                    repr(e),

                "n_nn":
                    np.nan,

                **{
                    f: np.nan
                    for f in FEATURES
                },
            }

        rows.append(row)

        if (
            k % args.checkpoint == 0
            or k == len(todo)
        ):
            flush()

            elapsed = (
                time.time()
                - started
            )

            rate = (
                k / elapsed
                if elapsed > 0
                else np.nan
            )

            print(
                f"[{k:,}/{len(todo):,}] "
                f"{rate:.2f} window/s"
            )

    # ----------------------------------------
    # FINAL AUDIT
    # ----------------------------------------

    final = pd.read_csv(
        output_path
    )

    # Keep only manifest rows in case output
    # was produced using --limit previously.
    wanted = set(
        manifest.global_index
        .astype(int)
        .tolist()
    )

    final = final[
        final.global_index
        .astype(int)
        .isin(wanted)
    ].copy()

    final = (
        final
        .drop_duplicates(
            subset=["global_index"],
            keep="last",
        )
        .sort_values(
            "global_index"
        )
        .reset_index(drop=True)
    )

    final.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 100)
    print("FINAL AUDIT")
    print("=" * 100)

    print(
        "Manifest rows:",
        len(manifest)
    )

    print(
        "Extracted rows:",
        len(final)
    )

    print(
        "Feature OK:",
        int(
            final.feature_ok.sum()
        )
    )

    print(
        "Feature fail:",
        int(
            (~final.feature_ok).sum()
        )
    )

    print(
        "Coverage:",
        f"{final.feature_ok.mean():.4%}"
    )

    print("\nErrors:")
    print(
        final.error
        .fillna("")
        .value_counts()
        .head(20)
        .to_string()
    )

    print("\nBy rhythm:")
    print(
        pd.crosstab(
            final.rhythm,
            final.feature_ok,
        ).to_string()
    )

    print("\nBy quality:")
    print(
        pd.crosstab(
            final.quality_name,
            final.feature_ok,
        ).to_string()
    )

    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
