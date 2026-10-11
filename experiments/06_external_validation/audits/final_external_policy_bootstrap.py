from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("experiments/06_external_validation")

DEEP = OUT / "deepbeat_frozen_v5_subject_summary.csv"
LIU  = OUT / "liu_pseudo30_frozen_v5_subject_rhythm.csv"
PEAK = OUT / "peakdet_pac_pvc_frozen_v5_subject_summary.csv"

OUT_CSV = OUT / "final_external_policy_bootstrap.csv"

SEED = 20260916
B = 50000

rng = np.random.default_rng(SEED)

rows = []


def ci_boot_mean(values, B=B):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan, np.nan

    idx = rng.integers(
        0,
        len(x),
        size=(B, len(x)),
    )

    boot = x[idx].mean(axis=1)

    return (
        float(x.mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def add_result(
    dataset,
    subset,
    policy,
    metric,
    values,
    note="",
):
    mean, lo, hi = ci_boot_mean(values)

    rows.append({
        "dataset": dataset,
        "subset": subset,
        "policy": policy,
        "metric": metric,
        "n_subjects": len(values),
        "estimate": mean,
        "ci95_low": lo,
        "ci95_high": hi,
        "note": note,
    })


# ============================================================
# DEEPBEAT
# ============================================================

deep = pd.read_csv(DEEP)

deep_policies = {
    "baseline_090":
        "fixed_090_positive_rate",

    "candidate_A_085":
        "fixed_085_positive_rate",

    "candidate_B_adaptive":
        "hr_lt60_080_else090_positive_rate",
}

print("=" * 100)
print("DEEPBEAT SUBJECT BOOTSTRAP")
print("=" * 100)

for label, subset in [
    (1, "AF"),
    (0, "NON_AF"),
]:
    d = deep[
        deep["label"] == label
    ].copy()

    print(
        f"\n{subset}: "
        f"{len(d)} subjects"
    )

    base = d[
        deep_policies[
            "baseline_090"
        ]
    ].to_numpy()

    for policy, col in deep_policies.items():

        values = d[col].to_numpy()

        metric = (
            "sensitivity"
            if label == 1
            else "false_positive_rate"
        )

        add_result(
            "DeepBeat",
            subset,
            policy,
            metric,
            values,
            note="25-s independent stress test",
        )

        m, lo, hi = ci_boot_mean(values)

        print(
            f"{policy:24s} "
            f"{m*100:7.3f}% "
            f"[{lo*100:7.3f}, "
            f"{hi*100:7.3f}]"
        )

        if policy != "baseline_090":

            delta = values - base

            add_result(
                "DeepBeat",
                subset,
                policy,
                "delta_vs_baseline",
                delta,
                note=(
                    "positive = more AF detection"
                    if label == 1
                    else "positive = more false positives"
                ),
            )

            dm, dlo, dhi = (
                ci_boot_mean(delta)
            )

            print(
                f"  Δ vs baseline: "
                f"{dm*100:+7.3f} pp "
                f"[{dlo*100:+7.3f}, "
                f"{dhi*100:+7.3f}]"
            )


# ============================================================
# LIU
# ============================================================

liu = pd.read_csv(LIU)

liu_policies = {
    "baseline_090":
        "fixed_090_positive_rate",

    "candidate_A_085":
        "fixed_085_positive_rate",

    "candidate_B_adaptive":
        "adaptive_positive_rate",
}

print()
print("=" * 100)
print("LIU SUBJECT-RHYTHM BOOTSTRAP")
print("=" * 100)

for rhythm in [
    "AF",
    "SR",
    "PAC",
    "PVC",
    "SVT",
    "VT",
]:

    d = liu[
        liu["rhythm"] == rhythm
    ].copy()

    if len(d) == 0:
        continue

    print(
        f"\n{rhythm}: "
        f"{len(d)} subjects"
    )

    base = d[
        liu_policies[
            "baseline_090"
        ]
    ].to_numpy()

    for policy, col in liu_policies.items():

        values = d[col].to_numpy()

        metric = (
            "sensitivity"
            if rhythm == "AF"
            else "false_positive_rate"
        )

        add_result(
            "Liu_pseudo30",
            rhythm,
            policy,
            metric,
            values,
            note=(
                "pseudo-30s sensitivity analysis; "
                "not confirmed continuous 30-s"
            ),
        )

        m, lo, hi = ci_boot_mean(values)

        print(
            f"{policy:24s} "
            f"{m*100:7.3f}% "
            f"[{lo*100:7.3f}, "
            f"{hi*100:7.3f}]"
        )

        if policy != "baseline_090":

            delta = values - base

            add_result(
                "Liu_pseudo30",
                rhythm,
                policy,
                "delta_vs_baseline",
                delta,
                note=(
                    "positive = more AF detection"
                    if rhythm == "AF"
                    else "positive = more false positives"
                ),
            )

            dm, dlo, dhi = (
                ci_boot_mean(delta)
            )

            print(
                f"  Δ vs baseline: "
                f"{dm*100:+7.3f} pp "
                f"[{dlo*100:+7.3f}, "
                f"{dhi*100:+7.3f}]"
            )


# ============================================================
# PEAKDET — DESCRIPTIVE ONLY, ONLY 3 SUBJECTS
# ============================================================

peak = pd.read_csv(PEAK)

peak_policies = {
    "baseline_090":
        "fixed_090_positive_rate",

    "candidate_A_085":
        "fixed_085_positive_rate",

    "candidate_B_adaptive":
        "hr_lt60_080_else090_positive_rate",
}

print()
print("=" * 100)
print("PEAKDET PAC/PVC — DESCRIPTIVE")
print("=" * 100)

print(
    "Subjects:",
    len(peak)
)

base = peak[
    peak_policies[
        "baseline_090"
    ]
].to_numpy()

for policy, col in peak_policies.items():

    values = peak[col].to_numpy()

    # Keep CI for transparency but mark as
    # descriptive because n=3.
    add_result(
        "PeakDet",
        "PAC_PVC",
        policy,
        "false_positive_rate",
        values,
        note=(
            "only 3 independent subjects; "
            "descriptive supporting evidence"
        ),
    )

    m, lo, hi = ci_boot_mean(values)

    print(
        f"{policy:24s} "
        f"{m*100:7.3f}% "
        f"[{lo*100:7.3f}, "
        f"{hi*100:7.3f}]"
    )

    if policy != "baseline_090":

        delta = values - base

        add_result(
            "PeakDet",
            "PAC_PVC",
            policy,
            "delta_vs_baseline",
            delta,
            note=(
                "positive = more false positives; "
                "n=3 subjects"
            ),
        )


# ============================================================
# SAVE
# ============================================================

result = pd.DataFrame(rows)

result.to_csv(
    OUT_CSV,
    index=False,
)

print()
print("=" * 100)
print("POLICY DIRECTION SUMMARY")
print("=" * 100)

for dataset in [
    "DeepBeat",
    "Liu_pseudo30",
    "PeakDet",
]:

    d = result[
        (result.dataset == dataset)
        &
        (result.metric == "delta_vs_baseline")
    ]

    print(
        "\n",
        dataset,
        sep="",
    )

    if len(d):
        print(
            d[
                [
                    "subset",
                    "policy",
                    "estimate",
                    "ci95_low",
                    "ci95_high",
                    "note",
                ]
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.5f}",
            )
        )


print()
print("=" * 100)
print("INTERPRETATION RULE")
print("=" * 100)

print(
    "Do not select thresholds from these "
    "bootstrap results."
)

print(
    "Policies were frozen before these "
    "external supporting analyses."
)

print(
    "Pulsewatch candidate-policy results "
    "remain exploratory because Pulsewatch "
    "was used during policy discovery."
)

print(
    "DeepBeat is a 25-s duration-shift "
    "stress test."
)

print(
    "Liu pseudo-30s is not verified "
    "continuous 30-s data."
)

print(
    "PeakDet PAC/PVC has only 3 subjects."
)

print()
print("SAVED:")
print(OUT_CSV)
