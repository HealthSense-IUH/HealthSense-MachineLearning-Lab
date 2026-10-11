from pathlib import Path
import re
import time

import numpy as np
import pandas as pd


ROOT = Path("experiments/06_external_validation")

FEATURES = ROOT / "pulsewatch_frozen_v5_features.csv"
PRED = ROOT / "pulsewatch_frozen_v5_predictions.csv"

OUT_SEG = ROOT / "pulsewatch_v5_alert_segment_timeline.csv"
OUT_RUN = ROOT / "pulsewatch_v5_alert_runs.csv"
OUT_SUM = ROOT / "pulsewatch_v5_alert_summary.csv"
OUT_SUBJ = ROOT / "pulsewatch_v5_alert_subject_summary.csv"

THRESHOLD = 0.90
N_CONSECUTIVE = 3

PROGRESS_EVERY = 20000


# ============================================================
# LOAD
# ============================================================

feat = pd.read_csv(
    FEATURES,
    dtype={"uid": str},
    low_memory=False,
)

pred = pd.read_csv(
    PRED,
    dtype={"uid": str},
    low_memory=False,
)

feat["uid"] = feat["uid"].astype(str).str.zfill(3)
pred["uid"] = pred["uid"].astype(str).str.zfill(3)


pred_small = pred[
    [
        "segment",
        "af_probability",
    ]
].copy()


df = feat.merge(
    pred_small,
    on="segment",
    how="left",
    validate="one_to_one",
)


# ============================================================
# PARSE SEGMENT NAME
#
# Example:
# 075_2020_..._ppg_0009
#
# session = everything before "_ppg_XXXX"
# ============================================================

pattern = re.compile(
    r"^(?P<session>.+)_ppg_(?P<idx>\d+)$"
)


sessions = []
indices = []
parse_ok = []

for s in df["segment"].astype(str):

    m = pattern.match(s)

    if m is None:

        sessions.append(None)
        indices.append(np.nan)
        parse_ok.append(False)

    else:

        sessions.append(
            m.group("session")
        )

        indices.append(
            int(m.group("idx"))
        )

        parse_ok.append(True)


df["session"] = sessions
df["segment_index"] = indices
df["parse_ok"] = parse_ok


print("=" * 90)
print("06A.4.5 PULSEWATCH 3-OF-3 ALERT PERSISTENCE AUDIT")
print("=" * 90)

print("Segments       :", f"{len(df):,}")
print("Subjects       :", df["uid"].nunique())
print("Parsed names   :", f"{int(df.parse_ok.sum()):,}")
print("Parse failures :", f"{int((~df.parse_ok).sum()):,}")
print("Threshold      :", THRESHOLD)
print("Policy         :", f"{N_CONSECUTIVE}-of-{N_CONSECUTIVE}")

print()


# ============================================================
# DECISION
#
# feature failure / missing probability = no decision
# no decision resets the consecutive-positive streak.
# ============================================================

df["decision_available"] = (
    df["feature_ok"].fillna(False)
    &
    df["af_probability"].notna()
)

df["positive"] = (
    df["decision_available"]
    &
    (
        df["af_probability"]
        >= THRESHOLD
    )
)


# ============================================================
# ORDER
# ============================================================

df = df[
    df["parse_ok"]
].copy()

df = df.sort_values(
    [
        "uid",
        "session",
        "segment_index",
    ]
).reset_index(drop=True)


# ============================================================
# BUILD SAME-RHYTHM CONTIGUOUS RUNS
#
# New run if:
# - subject/session changes
# - segment index is not consecutive
# - rhythm label changes
# ============================================================

run_ids = []

run_id = -1

prev_uid = None
prev_session = None
prev_idx = None
prev_class = None

for i, r in df.iterrows():

    new_run = False

    if prev_uid is None:
        new_run = True

    elif r["uid"] != prev_uid:
        new_run = True

    elif r["session"] != prev_session:
        new_run = True

    elif int(r["segment_index"]) != int(prev_idx) + 1:
        new_run = True

    elif r["class_name"] != prev_class:
        new_run = True

    if new_run:
        run_id += 1

    run_ids.append(run_id)

    prev_uid = r["uid"]
    prev_session = r["session"]
    prev_idx = r["segment_index"]
    prev_class = r["class_name"]

    n = i + 1

    if (
        n % PROGRESS_EVERY == 0
        or n == len(df)
    ):

        print(
            f"Indexed "
            f"{n:,}/{len(df):,} "
            f"({n/len(df)*100:6.2f}%)"
            f" | runs so far {run_id+1:,}",
            flush=True,
        )


df["run_id"] = run_ids


# ============================================================
# APPLY 3-CONSECUTIVE POLICY
# ============================================================

alert_flags = np.zeros(
    len(df),
    dtype=bool,
)

streak_values = np.zeros(
    len(df),
    dtype=int,
)


processed = 0
t0 = time.time()

for rid, idxs in df.groupby(
    "run_id",
    sort=True,
).groups.items():

    streak = 0

    for idx in idxs:

        # A no-decision explicitly resets streak.
        if not bool(
            df.at[
                idx,
                "decision_available",
            ]
        ):

            streak = 0

        elif bool(
            df.at[
                idx,
                "positive",
            ]
        ):

            streak += 1

        else:
            streak = 0

        streak_values[idx] = streak

        if streak >= N_CONSECUTIVE:
            alert_flags[idx] = True

        processed += 1


df["positive_streak"] = streak_values
df["alert_3of3"] = alert_flags


# ============================================================
# RUN SUMMARY
# ============================================================

run_rows = []

for rid, g in df.groupby(
    "run_id",
    sort=True,
):

    g = g.sort_values(
        "segment_index"
    )

    alerts = g[
        g["alert_3of3"]
    ]

    first_alert_position = np.nan
    first_alert_index = np.nan

    if len(alerts):

        first_alert_idx = alerts.index[0]

        first_alert_position = int(
            np.where(
                g.index.to_numpy()
                == first_alert_idx
            )[0][0]
        ) + 1

        first_alert_index = int(
            alerts.iloc[0][
                "segment_index"
            ]
        )

    run_rows.append(
        {
            "run_id":
                rid,

            "uid":
                g.iloc[0]["uid"],

            "session":
                g.iloc[0]["session"],

            "class_name":
                g.iloc[0]["class_name"],

            "format":
                g.iloc[0]["format"],

            "trial":
                g.iloc[0]["trial"],

            "start_index":
                int(
                    g["segment_index"].min()
                ),

            "end_index":
                int(
                    g["segment_index"].max()
                ),

            "segments":
                len(g),

            "decisions":
                int(
                    g[
                        "decision_available"
                    ].sum()
                ),

            "no_decision":
                int(
                    (
                        ~g[
                            "decision_available"
                        ]
                    ).sum()
                ),

            "positive_segments":
                int(
                    g["positive"].sum()
                ),

            "max_positive_streak":
                int(
                    g[
                        "positive_streak"
                    ].max()
                ),

            "alerted":
                bool(
                    g[
                        "alert_3of3"
                    ].any()
                ),

            "alert_events":
                int(
                    g[
                        "alert_3of3"
                    ].sum()
                ),

            "first_alert_position":
                first_alert_position,

            "first_alert_segment_index":
                first_alert_index,
        }
    )


runs = pd.DataFrame(
    run_rows
)

runs["eligible_3_segments"] = (
    runs["segments"]
    >= N_CONSECUTIVE
)


# ============================================================
# CLASS SUMMARY
# ============================================================

summary_rows = []

for cls, g in runs.groupby(
    "class_name"
):

    eligible = g[
        g["eligible_3_segments"]
    ]

    summary_rows.append(
        {
            "class_name":
                cls,

            "runs":
                len(g),

            "eligible_runs":
                len(eligible),

            "alerted_runs":
                int(
                    g["alerted"].sum()
                ),

            "alerted_eligible_runs":
                int(
                    eligible[
                        "alerted"
                    ].sum()
                ),

            "alert_rate_all_runs":
                float(
                    g["alerted"].mean()
                )
                if len(g)
                else np.nan,

            "alert_rate_eligible_runs":
                float(
                    eligible[
                        "alerted"
                    ].mean()
                )
                if len(eligible)
                else np.nan,

            "median_run_segments":
                float(
                    g["segments"].median()
                ),

            "median_max_streak":
                float(
                    g[
                        "max_positive_streak"
                    ].median()
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# SUBJECT SUMMARY
# ============================================================

subject_rows = []

for (uid, cls), g in runs.groupby(
    [
        "uid",
        "class_name",
    ]
):

    eligible = g[
        g["eligible_3_segments"]
    ]

    subject_rows.append(
        {
            "uid":
                uid,

            "class_name":
                cls,

            "runs":
                len(g),

            "eligible_runs":
                len(eligible),

            "any_alert":
                bool(
                    g["alerted"].any()
                ),

            "eligible_any_alert":
                bool(
                    eligible["alerted"].any()
                )
                if len(eligible)
                else False,

            "alerted_runs":
                int(
                    g["alerted"].sum()
                ),

            "max_streak":
                int(
                    g[
                        "max_positive_streak"
                    ].max()
                ),

            "segments":
                int(
                    g["segments"].sum()
                ),

            "decisions":
                int(
                    g["decisions"].sum()
                ),
        }
    )


subject = pd.DataFrame(
    subject_rows
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUT_SEG,
    index=False,
)

runs.to_csv(
    OUT_RUN,
    index=False,
)

summary.to_csv(
    OUT_SUM,
    index=False,
)

subject.to_csv(
    OUT_SUBJ,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 90)
print("RUN-LEVEL 3-OF-3 RESULT")
print("=" * 90)

print(
    summary.to_string(
        index=False
    )
)


print()
print("=" * 90)
print("SUBJECTS WITH >=1 ALERT")
print("=" * 90)

for cls in [
    "AF",
    "NSR",
    "PAC_PVC",
]:

    g = subject[
        subject["class_name"]
        == cls
    ]

    if len(g) == 0:
        continue

    print()

    print(
        f"{cls:8s}: "
        f"{int(g.any_alert.sum())}"
        f"/{len(g)} subjects "
        f"({g.any_alert.mean()*100:.2f}%)"
    )


print()
print("=" * 90)
print("PAC/PVC RUNS WITH FALSE ALERT")
print("=" * 90)

pac = runs[
    (runs["class_name"] == "PAC_PVC")
    &
    (runs["alerted"])
].copy()

if len(pac):

    print(
        pac.sort_values(
            [
                "max_positive_streak",
                "segments",
            ],
            ascending=False,
        )
        .head(30)
        .to_string(
            index=False
        )
    )

else:

    print("None")


print()
print("=" * 90)
print("AF RUNS MISSED DESPITE >=3 SEGMENTS")
print("=" * 90)

missed_af = runs[
    (runs["class_name"] == "AF")
    &
    (runs["eligible_3_segments"])
    &
    (~runs["alerted"])
].copy()

print(
    "Missed eligible AF runs:",
    len(missed_af),
)

if len(missed_af):

    print(
        missed_af.sort_values(
            "segments",
            ascending=False,
        )
        .head(30)
        .to_string(
            index=False
        )
    )


print()
print("Saved:")
print(" ", OUT_SEG)
print(" ", OUT_RUN)
print(" ", OUT_SUM)
print(" ", OUT_SUBJ)
