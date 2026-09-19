from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path("/home/phuc/Documents/HealthSense_rungtamnhi")
ART = ROOT / "experiments/08_v6_locked_protocol/artifacts"


def summarize(df, dataset, split):
    lengths = (
        df.groupby("stream_id")
        .size()
        .rename("stream_length")
    )

    row_lengths = df["stream_id"].map(lengths)

    return {
        "dataset": dataset,
        "split": split,
        "rows": len(df),
        "subjects": df["subject_id"].nunique(),
        "streams": df["stream_id"].nunique(),
        "median_stream_windows": float(lengths.median()),
        "p95_stream_windows": float(lengths.quantile(.95)),
        "max_stream_windows": int(lengths.max()),
        "rows_in_stream_ge_2":
            float((row_lengths >= 2).mean()),
        "rows_in_stream_ge_3":
            float((row_lengths >= 3).mean()),
        "rows_in_stream_ge_5":
            float((row_lengths >= 5).mean()),
    }


def build_deepbeat(df):
    df = df.copy()

    df["timestamp_dt"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
        utc=True
    )

    keys = [
        "subject_id",
        "source"
    ]

    df = df.sort_values(
        keys + ["timestamp_dt", "_row_order"]
    ).reset_index(drop=True)

    df["gap_sec"] = (
        df.groupby(keys)["timestamp_dt"]
        .diff()
        .dt.total_seconds()
    )

    # DeepBeat thin25:
    # expected interval ≈ 25 s.
    # Break if one or more windows are missing.
    df["new_stream"] = (
        df["gap_sec"].isna()
        |
        (df["gap_sec"] <= 0)
        |
        (df["gap_sec"] > 37.5)
    )

    df["stream_run"] = (
        df.groupby(keys)["new_stream"]
        .cumsum()
        .astype(int)
    )

    df["stream_id"] = (
        "deepbeat:"
        + df["subject_id"].astype(str)
        + ":"
        + df["source"].astype(str)
        + ":"
        + df["stream_run"].astype(str)
    )

    df["order_in_stream"] = (
        df.groupby("stream_id")
        .cumcount()
    )

    return df


def build_pulsewatch(df):
    df = df.copy()

    parsed = df["segment"].astype(str).str.extract(
        r"^(?P<recording_id>.+)_ppg_(?P<segment_seq>\d+)$"
    )

    if parsed.isna().any().any():
        bad = df.loc[
            parsed.isna().any(axis=1),
            "segment"
        ].head(20)

        raise RuntimeError(
            "Cannot parse PulseWatch segment names:\n"
            + bad.to_string(index=False)
        )

    df["recording_id"] = parsed[
        "recording_id"
    ]

    df["segment_seq"] = pd.to_numeric(
        parsed["segment_seq"],
        errors="raise"
    )

    keys = [
        "subject_id",
        "recording_id"
    ]

    df = df.sort_values(
        keys + ["segment_seq", "_row_order"]
    ).reset_index(drop=True)

    df["seq_gap"] = (
        df.groupby(keys)["segment_seq"]
        .diff()
    )

    # Break if segment numbers are not exactly consecutive.
    df["new_stream"] = (
        df["seq_gap"].isna()
        |
        (df["seq_gap"] != 1)
    )

    df["stream_run"] = (
        df.groupby(keys)["new_stream"]
        .cumsum()
        .astype(int)
    )

    df["stream_id"] = (
        "pulsewatch:"
        + df["subject_id"].astype(str)
        + ":"
        + df["recording_id"].astype(str)
        + ":"
        + df["stream_run"].astype(str)
    )

    df["order_in_stream"] = (
        df.groupby("stream_id")
        .cumcount()
    )

    return df


def build_liu(df):
    df = df.copy()

    for c in [
        "idx_0",
        "idx_1",
        "idx_2"
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="raise"
        )

    keys = ["subject_id"]

    df = df.sort_values(
        ["subject_id", "idx_0", "_row_order"]
    ).reset_index(drop=True)

    df["previous_idx_2"] = (
        df.groupby(keys)["idx_2"]
        .shift(1)
    )

    # Each pseudo30 row covers idx_0..idx_2.
    # Only concatenate if next interval starts exactly
    # after the previous interval.
    df["new_stream"] = (
        df["previous_idx_2"].isna()
        |
        (
            df["idx_0"]
            !=
            df["previous_idx_2"] + 1
        )
    )

    df["stream_run"] = (
        df.groupby(keys)["new_stream"]
        .cumsum()
        .astype(int)
    )

    df["stream_id"] = (
        "liu:"
        + df["subject_id"].astype(str)
        + ":"
        + df["stream_run"].astype(str)
    )

    df["order_in_stream"] = (
        df.groupby("stream_id")
        .cumcount()
    )

    return df


builders = {
    "deepbeat": build_deepbeat,
    "pulsewatch": build_pulsewatch,
    "liu": build_liu,
}


summary = []


for dataset in builders:

    for split in [
        "dev",
        "test"
    ]:

        src = (
            ART /
            f"{dataset}_{split}_locked_predictions.csv"
        )

        df = pd.read_csv(
            src,
            low_memory=False
        )

        df["subject_id"] = (
            df["subject_id"]
            .astype(str)
        )

        out = builders[dataset](df)

        dst = (
            ART /
            f"{dataset}_{split}_verified_streams.csv"
        )

        out.to_csv(
            dst,
            index=False
        )

        row = summarize(
            out,
            dataset,
            split
        )

        summary.append(row)

        print("\n" + "=" * 100)
        print(dataset, split)
        print(pd.Series(row))
        print("\nTop stream lengths:")

        print(
            out.groupby("stream_id")
            .size()
            .sort_values(ascending=False)
            .head(15)
            .to_string()
        )


summary = pd.DataFrame(summary)

summary.to_csv(
    ART / "verified_stream_summary.csv",
    index=False
)

print("\n" + "=" * 100)
print("FINAL SUMMARY")
print(summary.to_string(index=False))
