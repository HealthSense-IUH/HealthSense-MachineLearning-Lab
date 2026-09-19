from pathlib import Path
import re
import numpy as np
import pandas as pd


ROOT = Path("/home/phuc/Documents/HealthSense_rungtamnhi")

ART = (
    ROOT /
    "experiments/08_v6_locked_protocol/artifacts"
)


# ============================================================
# Helpers
# ============================================================

def norm_subject(x):
    return x.astype(str).str.strip()


def trailing_number(x):

    s = x.astype(str)

    z = s.str.extract(
        r"(-?\d+(?:\.\d+)?)\s*$",
        expand=False
    )

    return pd.to_numeric(
        z,
        errors="coerce"
    )


def parse_order(series, mode):

    if mode == "segment":
        numeric = pd.to_numeric(
            series,
            errors="coerce"
        )

        if numeric.notna().mean() >= 0.95:
            return numeric, "numeric"

        trailing = trailing_number(series)

        if trailing.notna().mean() >= 0.95:
            return trailing, "trailing_number"

        return numeric, "unresolved"


    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    if numeric.notna().mean() >= 0.95:
        return numeric, "numeric"


    dt = pd.to_datetime(
        series,
        errors="coerce",
        utc=True
    )

    if dt.notna().mean() >= 0.95:

        sec = pd.Series(
            dt.astype("int64") / 1e9,
            index=series.index
        )

        sec[dt.isna()] = np.nan

        return sec, "datetime"


    return numeric, "unresolved"


def temporal_audit(
    file,
    dataset,
    group_cols,
    order_col,
    mode="generic"
):

    print("\n" + "=" * 100)
    print("TEMPORAL AUDIT:", dataset)
    print(file)

    df = pd.read_csv(file)

    for c in group_cols + [order_col]:
        if c not in df.columns:
            raise RuntimeError(
                f"{dataset}: missing {c}"
            )


    order, parser = parse_order(
        df[order_col],
        mode
    )

    df["_order_value"] = order


    parse_rate = (
        df["_order_value"]
        .notna()
        .mean()
    )


    group_stats = []


    for keys, g in df.groupby(
        group_cols,
        dropna=False,
        sort=False
    ):

        g = g.copy()

        original = (
            g.sort_values("_row_order")
            if "_row_order" in g.columns
            else g
        )

        v = (
            original["_order_value"]
            .dropna()
            .to_numpy()
        )


        if len(v) >= 2:

            diffs_original = np.diff(v)

            monotonic_original = bool(
                np.all(
                    diffs_original >= 0
                )
            )

        else:

            monotonic_original = True


        sorted_values = np.sort(
            g["_order_value"]
            .dropna()
            .to_numpy()
        )


        if len(sorted_values) >= 2:

            gaps = np.diff(
                sorted_values
            )

            positive_gaps = gaps[
                gaps > 0
            ]

        else:

            gaps = np.array([])
            positive_gaps = np.array([])


        duplicate_count = int(
            g["_order_value"]
            .dropna()
            .duplicated()
            .sum()
        )


        row = {
            "dataset":
                dataset,

            "group":
                str(keys),

            "rows":
                len(g),

            "valid_order":
                int(
                    g["_order_value"]
                    .notna()
                    .sum()
                ),

            "duplicates":
                duplicate_count,

            "monotonic_original":
                monotonic_original,

            "gap_median":
                (
                    float(
                        np.median(
                            positive_gaps
                        )
                    )
                    if len(positive_gaps)
                    else np.nan
                ),

            "gap_p95":
                (
                    float(
                        np.quantile(
                            positive_gaps,
                            0.95
                        )
                    )
                    if len(positive_gaps)
                    else np.nan
                ),

            "gap_max":
                (
                    float(
                        positive_gaps.max()
                    )
                    if len(positive_gaps)
                    else np.nan
                ),
        }


        if (
            mode == "segment"
            and len(positive_gaps)
        ):

            row["gap_eq_1_fraction"] = (
                float(
                    np.mean(
                        np.isclose(
                            positive_gaps,
                            1
                        )
                    )
                )
            )

        group_stats.append(row)


    stat = pd.DataFrame(
        group_stats
    )


    print("parser:", parser)
    print(
        "parse coverage:",
        round(parse_rate, 6)
    )

    print(
        "sessions:",
        len(stat)
    )

    print(
        "subjects:",
        df["subject_id"].nunique()
        if "subject_id" in df.columns
        else "N/A"
    )

    print(
        "sessions monotonic in original row order:",
        int(
            stat[
                "monotonic_original"
            ].sum()
        ),
        "/",
        len(stat)
    )

    print(
        "total duplicate order values:",
        int(
            stat["duplicates"].sum()
        )
    )


    if "gap_eq_1_fraction" in stat.columns:

        print(
            "median fraction gap==1:",
            stat[
                "gap_eq_1_fraction"
            ].median()
        )


    print("\nOrder examples:")

    cols = (
        group_cols +
        [order_col, "_row_order"]
    )

    cols = [
        c for c in cols
        if c in df.columns
    ]

    print(
        df[cols]
        .head(25)
        .to_string(index=False)
    )


    outfile = (
        ART /
        f"{dataset}_temporal_order_audit.csv"
    )

    stat.to_csv(
        outfile,
        index=False
    )

    print(
        "\nsaved:",
        outfile
    )


    return {
        "dataset":
            dataset,

        "parser":
            parser,

        "parse_coverage":
            parse_rate,

        "sessions":
            len(stat),

        "monotonic_sessions":
            int(
                stat[
                    "monotonic_original"
                ].sum()
            ),

        "duplicate_orders":
            int(
                stat[
                    "duplicates"
                ].sum()
            ),

        "median_gap":
            stat[
                "gap_median"
            ].median(),
    }


# ============================================================
# Temporal audit
# DEV + TEST may be inspected for structural metadata only.
# We are NOT selecting a decision rule here.
# ============================================================

temporal_summary = []


for split in [
    "dev",
    "test"
]:

    temporal_summary.append(
        temporal_audit(
            ART /
            f"deepbeat_{split}_locked_predictions.csv",
            f"deepbeat_{split}",
            [
                "subject_id",
                "source"
            ],
            "timestamp"
        )
    )


    temporal_summary.append(
        temporal_audit(
            ART /
            f"pulsewatch_{split}_locked_predictions.csv",
            f"pulsewatch_{split}",
            [
                "subject_id",
                "trial"
            ],
            "segment",
            mode="segment"
        )
    )


    temporal_summary.append(
        temporal_audit(
            ART /
            f"liu_{split}_locked_predictions.csv",
            f"liu_{split}",
            [
                "subject_id"
            ],
            "idx_0"
        )
    )


pd.DataFrame(
    temporal_summary
).to_csv(
    ART /
    "temporal_order_summary.csv",
    index=False
)


# ============================================================
# Liu interval sanity
# ============================================================

print(
    "\n" +
    "=" * 100
)
print("LIU INTERVAL SANITY")


liu_rows = []


for split in [
    "dev",
    "test"
]:

    f = (
        ART /
        f"liu_{split}_locked_predictions.csv"
    )

    df = pd.read_csv(f)


    for c in [
        "idx_0",
        "idx_1",
        "idx_2"
    ]:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )


    ordered_01 = (
        df["idx_0"]
        <=
        df["idx_1"]
    )

    ordered_12 = (
        df["idx_1"]
        <=
        df["idx_2"]
    )


    print("\n", split)

    print(
        "idx_0 <= idx_1:",
        ordered_01.mean()
    )

    print(
        "idx_1 <= idx_2:",
        ordered_12.mean()
    )

    print(
        df[
            [
                "subject_id",
                "idx_0",
                "idx_1",
                "idx_2",
                "label"
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


    liu_rows.append({
        "split":
            split,

        "rows":
            len(df),

        "idx0_le_idx1":
            ordered_01.mean(),

        "idx1_le_idx2":
            ordered_12.mean(),

        "idx0_unique_fraction":
            df[
                [
                    "subject_id",
                    "idx_0"
                ]
            ]
            .drop_duplicates()
            .shape[0]
            /
            len(df)
    })


pd.DataFrame(
    liu_rows
).to_csv(
    ART /
    "liu_interval_audit.csv",
    index=False
)


# ============================================================
# Subject split overlap
# ============================================================

print(
    "\n" +
    "=" * 100
)
print("SUBJECT SPLIT OVERLAP")


split_files = {

    "deepbeat": {
        "train":
            ROOT /
            "data/features/v6/deepbeat_train_v6_14f.csv",

        "dev":
            ROOT /
            "data/features/v6/deepbeat_validate_v6_14f.csv",

        "test":
            ROOT /
            "data/features/v6/deepbeat_test_independent_v6_14f.csv",

        "subject":
            "subject_id"
    },

    "pulsewatch": {
        "train":
            ROOT /
            "experiments/07_v6_multidomain/artifacts/"
            "hard_domain_splits/pulsewatch_train.csv",

        "dev":
            ROOT /
            "experiments/07_v6_multidomain/artifacts/"
            "hard_domain_splits/pulsewatch_dev.csv",

        "test":
            ROOT /
            "experiments/07_v6_multidomain/artifacts/"
            "hard_domain_splits/pulsewatch_test.csv",

        "subject":
            "uid"
    },

    "liu": {
        "train":
            ROOT /
            "experiments/07_v6_multidomain/artifacts/"
            "hard_domain_splits/liu_train.csv",

        "dev":
            ROOT /
            "experiments/07_v6_multidomain/artifacts/"
            "hard_domain_splits/liu_dev.csv",

        "test":
            ROOT /
            "experiments/07_v6_multidomain/artifacts/"
            "hard_domain_splits/liu_test.csv",

        "subject":
            "subject_id"
    },
}


overlap_rows = []


for dataset, cfg in split_files.items():

    sets = {}


    print(
        "\n---",
        dataset,
        "---"
    )


    for split in [
        "train",
        "dev",
        "test"
    ]:

        f = cfg[split]

        if not f.exists():

            print(
                split,
                "MISSING:",
                f
            )

            sets[split] = set()

            continue


        df = pd.read_csv(f)

        col = cfg["subject"]

        if col not in df.columns:

            raise RuntimeError(
                f"{dataset}/{split}: "
                f"missing {col}"
            )


        s = set(
            norm_subject(
                df[col]
            )
            .dropna()
        )

        sets[split] = s


        print(
            split,
            "subjects=",
            len(s)
        )


    for a, b in [
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test")
    ]:

        overlap = sorted(
            sets[a]
            &
            sets[b]
        )

        print(
            f"{a} vs {b}:",
            len(overlap),
            overlap[:20]
        )


        overlap_rows.append({
            "dataset":
                dataset,

            "split_a":
                a,

            "split_b":
                b,

            "n_overlap":
                len(overlap),

            "overlap_subjects":
                ";".join(
                    overlap
                )
        })


pd.DataFrame(
    overlap_rows
).to_csv(
    ART /
    "subject_split_overlap_audit.csv",
    index=False
)


# ============================================================
# V6C training manifest audit
# ============================================================

print(
    "\n" +
    "=" * 100
)
print("V6C TRAIN MANIFEST")


manifest = (
    ROOT /
    "data/features/v7/"
    "healthsense_af_v6c_pacaware_train.csv"
)


if manifest.exists():

    m = pd.read_csv(
        manifest
    )

    print(
        "rows:",
        len(m)
    )

    print(
        "columns:",
        list(
            m.columns
        )
    )


    if "domain" in m.columns:

        print(
            "\ndomain counts:"
        )

        print(
            m["domain"]
            .value_counts(
                dropna=False
            )
        )


    if (
        "domain" in m.columns
        and
        "subject_id" in m.columns
    ):

        m["group_id"] = (
            m["domain"]
            .astype(str)
            +
            ":"
            +
            m["subject_id"]
            .astype(str)
        )


        print(
            "\nraw subject IDs:",
            m[
                "subject_id"
            ].nunique()
        )

        print(
            "domain-prefixed group IDs:",
            m[
                "group_id"
            ].nunique()
        )


        collisions = (
            m.groupby(
                "subject_id"
            )["domain"]
            .nunique()
        )

        collisions = (
            collisions[
                collisions > 1
            ]
        )


        print(
            "subject IDs reused across domains:",
            len(
                collisions
            )
        )

        if len(
            collisions
        ):

            print(
                collisions
                .head(30)
            )


print(
    "\nAUDIT COMPLETE"
)
