from pathlib import Path
import pandas as pd

ROOT = Path("experiments/06_external_validation")

RUNS = ROOT / "pulsewatch_v5_alert_runs.csv"

OUT_PAC = ROOT / "pulsewatch_v5_pac_false_alert_shortlist.csv"
OUT_AF = ROOT / "pulsewatch_v5_af_miss_shortlist.csv"
OUT_SUBJ = ROOT / "pulsewatch_v5_failure_by_subject.csv"

runs = pd.read_csv(
    RUNS,
    dtype={"uid": str},
)

runs["uid"] = (
    runs["uid"]
    .astype(str)
    .str.zfill(3)
)


# ============================================================
# PAC/PVC FALSE ALERTS
# ============================================================

pac = runs[
    (runs["class_name"] == "PAC_PVC")
    &
    (runs["alerted"])
].copy()

pac = pac.sort_values(
    [
        "max_positive_streak",
        "positive_segments",
        "segments",
    ],
    ascending=False,
)

pac.to_csv(
    OUT_PAC,
    index=False,
)


# ============================================================
# ELIGIBLE AF RUNS MISSED
# ============================================================

af = runs[
    (runs["class_name"] == "AF")
    &
    (runs["eligible_3_segments"])
    &
    (~runs["alerted"])
].copy()

af = af.sort_values(
    [
        "segments",
        "positive_segments",
    ],
    ascending=[
        False,
        True,
    ],
)

af.to_csv(
    OUT_AF,
    index=False,
)


# ============================================================
# SUBJECT CONCENTRATION
# ============================================================

pac_subject = (
    pac.groupby("uid")
    .size()
    .rename("pac_false_alert_runs")
)

af_subject = (
    af.groupby("uid")
    .size()
    .rename("af_missed_runs")
)

subject = pd.concat(
    [
        pac_subject,
        af_subject,
    ],
    axis=1,
).fillna(0)

subject[
    [
        "pac_false_alert_runs",
        "af_missed_runs",
    ]
] = subject[
    [
        "pac_false_alert_runs",
        "af_missed_runs",
    ]
].astype(int)

subject = (
    subject.reset_index()
    .sort_values(
        [
            "pac_false_alert_runs",
            "af_missed_runs",
        ],
        ascending=False,
    )
)

subject.to_csv(
    OUT_SUBJ,
    index=False,
)


print("=" * 90)
print("06A.4.6 FAILURE SHORTLIST")
print("=" * 90)

print()
print("PAC/PVC false-alert runs:", len(pac))

print(
    pac.groupby("uid")
    .size()
    .sort_values(
        ascending=False
    )
    .head(15)
    .to_string()
)

print()
print("AF eligible missed runs:", len(af))

print(
    af.groupby("uid")
    .size()
    .sort_values(
        ascending=False
    )
    .head(15)
    .to_string()
)

print()
print("TOP PAC/PVC RUNS")
print(
    pac.head(15)[
        [
            "uid",
            "session",
            "start_index",
            "end_index",
            "segments",
            "positive_segments",
            "max_positive_streak",
        ]
    ].to_string(index=False)
)

print()
print("TOP AF MISSES")
print(
    af.head(15)[
        [
            "uid",
            "session",
            "start_index",
            "end_index",
            "segments",
            "positive_segments",
            "max_positive_streak",
        ]
    ].to_string(index=False)
)

print()
print("Saved:")
print(" ", OUT_PAC)
print(" ", OUT_AF)
print(" ", OUT_SUBJ)
