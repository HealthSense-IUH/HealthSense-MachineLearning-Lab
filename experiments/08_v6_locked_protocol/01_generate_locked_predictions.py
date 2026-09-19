from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd


ROOT = Path("/home/phuc/Documents/HealthSense_rungtamnhi")

OUT = ROOT / "experiments/08_v6_locked_protocol/artifacts"

MODEL = ROOT / "models/multidomain/healthsense_af_v6c_stacking.pkl"


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


DATASETS = {

    "deepbeat_dev": {
        "dataset": "deepbeat",
        "split": "dev",
        "path":
            ROOT /
            "data/features/v6/deepbeat_validate_v6_14f.csv",
        "subject_candidates": [
            "subject_id",
            "subject",
            "uid"
        ],
        "label_mode": "binary",
        "label_candidates": [
            "label",
            "binary_label",
            "binary_label_clean"
        ],
    },

    "deepbeat_test": {
        "dataset": "deepbeat",
        "split": "test_reused",
        "path":
            ROOT /
            "data/features/v6/deepbeat_test_independent_v6_14f.csv",
        "subject_candidates": [
            "subject_id",
            "subject",
            "uid"
        ],
        "label_mode": "binary",
        "label_candidates": [
            "label",
            "binary_label",
            "binary_label_clean"
        ],
    },

    "pulsewatch_dev": {
        "dataset": "pulsewatch",
        "split": "dev",
        "path":
            ROOT /
            "experiments/07_v6_multidomain/artifacts/"
            "hard_domain_splits/pulsewatch_dev.csv",
        "subject_candidates": [
            "uid",
            "subject_id",
            "subject"
        ],
        "label_mode": "pulsewatch",
        "label_candidates": [
            "label",
            "binary_label",
            "class_name"
        ],
    },

    "pulsewatch_test": {
        "dataset": "pulsewatch",
        "split": "test_reused",
        "path":
            ROOT /
            "experiments/07_v6_multidomain/artifacts/"
            "hard_domain_splits/pulsewatch_test.csv",
        "subject_candidates": [
            "uid",
            "subject_id",
            "subject"
        ],
        "label_mode": "pulsewatch",
        "label_candidates": [
            "label",
            "binary_label",
            "class_name"
        ],
    },

    "liu_dev": {
        "dataset": "liu",
        "split": "dev",
        "path":
            ROOT /
            "experiments/07_v6_multidomain/artifacts/"
            "hard_domain_splits/liu_dev.csv",
        "subject_candidates": [
            "subject_id",
            "subject",
            "uid"
        ],
        "label_mode": "liu",
        "label_candidates": [
            "binary_label_clean",
            "binary_label",
            "label"
        ],
    },

    "liu_test": {
        "dataset": "liu",
        "split": "test_reused",
        "path":
            ROOT /
            "experiments/07_v6_multidomain/artifacts/"
            "hard_domain_splits/liu_test.csv",
        "subject_candidates": [
            "subject_id",
            "subject",
            "uid"
        ],
        "label_mode": "liu",
        "label_candidates": [
            "binary_label_clean",
            "binary_label",
            "label"
        ],
    },
}


KEEP_METADATA = [
    # general
    "record_id",
    "source",
    "global_index",
    "window_index",
    "timestamp",
    "t_start",

    # PulseWatch
    "segment",
    "trial",
    "class_name",

    # Liu
    "rhythm",
    "label_original",
    "idx_0",
    "idx_1",
    "idx_2",

    # quality / extraction
    "feature_ok",
    "timing_method",
    "duration_ms",
    "n_beats",
    "n_nn",
]


def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def make_binary_label(df, cfg):

    mode = cfg["label_mode"]

    if mode == "pulsewatch":

        if "class_name" in df.columns:
            return (
                df["class_name"]
                .astype(str)
                .str.upper()
                .eq("AF")
                .astype(int)
            )

        if "label" in df.columns:
            return (
                pd.to_numeric(
                    df["label"],
                    errors="coerce"
                )
                .eq(1)
                .astype(int)
            )

    if mode == "liu":

        if "binary_label_clean" in df.columns:
            return (
                pd.to_numeric(
                    df["binary_label_clean"],
                    errors="coerce"
                )
                .fillna(0)
                .astype(int)
            )

        if "binary_label" in df.columns:
            return (
                pd.to_numeric(
                    df["binary_label"],
                    errors="coerce"
                )
                .fillna(0)
                .astype(int)
            )

        if "rhythm" in df.columns:
            return (
                df["rhythm"]
                .astype(str)
                .str.upper()
                .eq("AF")
                .astype(int)
            )

    label_col = first_existing(
        df,
        cfg["label_candidates"]
    )

    if label_col is None:
        raise RuntimeError(
            f"No label column found. "
            f"columns={list(df.columns)}"
        )

    return (
        pd.to_numeric(
            df[label_col],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )


def get_order_description(dataset, df):

    if dataset == "deepbeat":

        if (
            "source" in df.columns
            and "timestamp" in df.columns
        ):
            return {
                "session_key": "source",
                "order_key": "timestamp",
                "temporal_quality": "verified_candidate"
            }

        if "timestamp" in df.columns:
            return {
                "session_key": None,
                "order_key": "timestamp",
                "temporal_quality": "partial_no_source"
            }

        if "t_start" in df.columns:
            return {
                "session_key": "record_id"
                if "record_id" in df.columns
                else None,
                "order_key": "t_start",
                "temporal_quality": "partial"
            }

    if dataset == "pulsewatch":

        if "segment" in df.columns:

            return {
                "session_key": "trial"
                if "trial" in df.columns
                else None,
                "order_key": "segment",
                "temporal_quality":
                    "candidate_requires_audit"
            }

    if dataset == "liu":

        if "idx_0" in df.columns:

            return {
                "session_key": None,
                "order_key": "idx_0",
                "temporal_quality": "verified_candidate"
            }

    return {
        "session_key": None,
        "order_key": "_row_order",
        "temporal_quality": "unverified_row_order"
    }


print("=" * 100)
print("LOAD MODEL")
print(MODEL)

bundle = joblib.load(MODEL)

print("bundle type:", type(bundle))

if not isinstance(bundle, dict):
    raise RuntimeError(
        "Expected stacking model bundle to be dict"
    )

print("bundle keys:", bundle.keys())


base_models = bundle["base_models"]
meta = bundle["meta"]

model_features = bundle.get(
    "features",
    FEATURES
)

model_features = list(model_features)

print("features:")
print(model_features)

missing_expected = [
    x for x in FEATURES
    if x not in model_features
]

if missing_expected:
    warnings.warn(
        f"Bundle feature schema differs: "
        f"{missing_expected}"
    )


audit_rows = []


for key, cfg in DATASETS.items():

    path = cfg["path"]

    print("\n" + "=" * 100)
    print(key)
    print(path)

    if not path.exists():
        print("MISSING FILE")
        audit_rows.append({
            "key": key,
            "dataset": cfg["dataset"],
            "split": cfg["split"],
            "exists": False
        })
        continue


    df = pd.read_csv(path)

    df["_row_order"] = np.arange(len(df))


    subject_col = first_existing(
        df,
        cfg["subject_candidates"]
    )

    if subject_col is None:
        raise RuntimeError(
            f"{key}: subject column not found.\n"
            f"columns={list(df.columns)}"
        )


    missing_features = [
        c for c in model_features
        if c not in df.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"{key}: missing features "
            f"{missing_features}"
        )


    binary_label = make_binary_label(
        df,
        cfg
    )


    valid = (
        df[model_features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .notna()
        .all(axis=1)
    )

    if "feature_ok" in df.columns:
        try:
            valid &= (
                df["feature_ok"]
                .astype(str)
                .str.lower()
                .isin([
                    "1",
                    "true",
                    "yes"
                ])
            )
        except Exception:
            pass


    work = df.loc[valid].copy()

    y = binary_label.loc[valid].astype(int)


    X = work[model_features].copy()


    base_probs = {}

    for name, model in base_models.items():

        p = model.predict_proba(X)[:, 1]

        base_probs[name] = p


    base_df = pd.DataFrame(
        base_probs,
        index=work.index
    )


    ensemble_mean = (
        base_df.mean(axis=1)
    )

    ensemble_std = (
        base_df.std(
            axis=1,
            ddof=0
        )
    )


    try:

        meta_prob = (
            meta.predict_proba(
                base_df
            )[:, 1]
        )

    except Exception:

        meta_prob = (
            meta.predict_proba(
                base_df.to_numpy()
            )[:, 1]
        )


    out = pd.DataFrame({
        "dataset":
            cfg["dataset"],

        "split":
            cfg["split"],

        "subject_id":
            work[subject_col]
            .astype(str)
            .values,

        "label":
            y.values,

        "_row_order":
            work["_row_order"]
            .values,
    })


    for c in KEEP_METADATA:

        if c in work.columns:

            out[c] = (
                work[c]
                .values
            )


    for name in base_df.columns:

        out[f"{name}_prob"] = (
            base_df[name]
            .values
        )


    out["ensemble_mean"] = (
        ensemble_mean.values
    )

    out["ensemble_std"] = (
        ensemble_std.values
    )

    out["meta_probability"] = (
        meta_prob
    )


    order_info = get_order_description(
        cfg["dataset"],
        out
    )


    out["temporal_session_key"] = (
        str(order_info["session_key"])
    )

    out["temporal_order_key"] = (
        str(order_info["order_key"])
    )

    out["temporal_quality"] = (
        order_info["temporal_quality"]
    )


    outfile = (
        OUT /
        f"{key}_locked_predictions.csv"
    )

    out.to_csv(
        outfile,
        index=False
    )


    print("rows input:", len(df))
    print("rows valid:", len(out))
    print("subjects:", out.subject_id.nunique())
    print(
        "labels:",
        out.label.value_counts().to_dict()
    )

    print(
        "meta_probability:",
        out.meta_probability.describe().to_dict()
    )

    print(
        "uncertainty:",
        out.ensemble_std.describe().to_dict()
    )

    print("order:", order_info)
    print("saved:", outfile)


    audit_rows.append({
        "key":
            key,

        "dataset":
            cfg["dataset"],

        "split":
            cfg["split"],

        "exists":
            True,

        "rows_input":
            len(df),

        "rows_valid":
            len(out),

        "subjects":
            out.subject_id.nunique(),

        "AF_windows":
            int(out.label.sum()),

        "nonAF_windows":
            int((out.label == 0).sum()),

        "session_key":
            order_info["session_key"],

        "order_key":
            order_info["order_key"],

        "temporal_quality":
            order_info["temporal_quality"],
    })


audit = pd.DataFrame(audit_rows)

audit.to_csv(
    OUT / "prediction_generation_audit.csv",
    index=False
)


with open(
    OUT / "protocol_metadata.json",
    "w"
) as f:

    json.dump(
        {
            "model":
                str(MODEL),

            "model_features":
                model_features,

            "principle":
                (
                    "Tune all decision rules on DEV only. "
                    "Do not select operating points using TEST. "
                    "TEST is explicitly marked reused where "
                    "it has already been inspected."
                ),

            "mimic_source":
                "MIMIC_PERform_AF",

            "mimic_iii_ext_ppg_status":
                "NOT_USED_PENDING_ACCESS",

            "triggersaf_status":
                "NOT_USED_PENDING_ACCESS",
        },
        f,
        indent=2
    )


print("\n" + "=" * 100)
print("AUDIT")
print(audit.to_string(index=False))

print("\nDONE")
