"""Run every experiment, write CSV results to results/ and figures to figures/.

Usage:
    python experiments/run_experiments.py            # full run (≈ 5–8 min)
    python experiments/run_experiments.py --quick    # 2 seeds, smaller sweeps

Experiments
-----------
E1  Peer-review groups: clustering methods compared on cluster fairness, topical
    cost, and the bias of the resulting peer grades (raw and calibrated).
E2  Bias regimes: which mechanism (leniency culture vs in-group favouritism)
    each method mitigates.
E3  Fairness / cost trade-off: sweep the representation slack of constrained
    fair k-means.
E4  Reviewer assignment: affinity-maximising ILP vs fair ILP (panel
    representation, anti-homogeneity).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fairpeer import (ReviewModel, ReviewerAssigner, affinity_matrix,  # noqa: E402
                      calibrated_grades, make_cohort, make_method, mean_grades,
                      pairs_from_clusters)
from fairpeer import metrics as M  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

# Validated categorical palette (fixed order, never cycled).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHODS = ["kmeans", "size_only", "random", "fairlet", "constrained_gender", "constrained"]
LABEL = {"kmeans": "K-means", "size_only": "Size-constrained", "random": "Random groups",
         "fairlet": "Fairlet k-means", "constrained_gender": "Constrained (gender only)",
         "constrained": "Constrained (gender + region)"}
COLOR = {m: SLOTS[i] for i, m in enumerate(METHODS)}
SHORT = {"kmeans": "K-means", "size_only": "Size-\nconstrained", "random": "Random\ngroups",
         "fairlet": "Fairlet\nk-means", "constrained_gender": "Constrained\n(gender)",
         "constrained": "Constrained\n(gen.+region)"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 130,
    "savefig.bbox": "tight", "legend.frameon": False,
})

N_CLUSTERS = 30   # 240 students -> review groups of 8


def evaluate_groups(cohort, labels, review):
    g, reg = cohort.sensitive["gender"], cohort.sensitive["region"]
    pairs = pairs_from_clusters(labels)
    s = review.score(cohort, pairs)
    raw = M.outcome_metrics(mean_grades(cohort.n, pairs, s), cohort.quality, g)
    cal = M.outcome_metrics(calibrated_grades(cohort.n, pairs, s, g), cohort.quality, g)
    lo, hi = M.size_spread(labels)
    row = {
        "balance_gender": M.balance(labels, g), "balance_region": M.balance(labels, reg),
        "propviol_gender": M.proportional_violation(labels, g),
        "propviol_region": M.proportional_violation(labels, reg),
        "homogeneity_gender": M.homogeneity(labels, g),
        "entropy_gender": M.mean_entropy(labels, g),
        "cost": M.kmeans_cost(cohort.X, labels), "size_min": lo, "size_max": hi,
        "ingroup_minority": M.ingroup_review_share(pairs, g)[1],
        "ingroup_majority": M.ingroup_review_share(pairs, g)[0],
    }
    row.update({f"raw_{k}": v for k, v in raw.items()})
    row.update({f"cal_{k}": v for k, v in cal.items()})
    return row


# --------------------------------------------------------------------------- #
def e1_clustering(seeds):
    rows, examples = [], {}
    for seed in seeds:
        cohort = make_cohort(seed=seed)
        review = ReviewModel(seed=seed)
        base_cost = None
        for m in METHODS:
            t = time.time()
            labels = make_method(m, N_CLUSTERS, seed=seed).fit_predict(cohort.X, cohort.sensitive)
            row = {"seed": seed, "method": m, "runtime_s": time.time() - t,
                   **evaluate_groups(cohort, labels, review)}
            if m == "kmeans":
                base_cost = row["cost"]
            row["cost_ratio"] = row["cost"] / base_cost
            rows.append(row)
            if seed == seeds[0]:
                examples[m] = (labels, cohort.sensitive["gender"])
    df = pd.DataFrame(rows)
    df.to_csv(RES / "e1_clustering_raw.csv", index=False)
    return df, examples


def e2_regimes(seeds):
    regimes = {
        "Leniency culture only": dict(leniency_strength=3.0, delta=0.0, delta_region=0.0),
        "In-group favouritism only": dict(leniency_strength=0.0, delta=4.0, delta_region=1.5),
        "Both mechanisms": dict(leniency_strength=3.0, delta=4.0, delta_region=1.5),
    }
    rows = []
    for name, p in regimes.items():
        for seed in seeds:
            cohort = make_cohort(seed=seed, leniency_strength=p["leniency_strength"])
            review = ReviewModel(delta=p["delta"], delta_region=p["delta_region"], seed=seed)
            for m in METHODS:
                labels = make_method(m, N_CLUSTERS, seed=seed).fit_predict(cohort.X, cohort.sensitive)
                r = evaluate_groups(cohort, labels, review)
                rows.append({"regime": name, "seed": seed, "method": m,
                             "raw_gap": r["raw_gap"], "cal_gap": r["cal_gap"],
                             "raw_spearman": r["raw_spearman"], "cal_spearman": r["cal_spearman"]})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "e2_regimes_raw.csv", index=False)
    return df


def e3_tradeoff(seeds, slacks):
    rows = []
    for seed in seeds:
        cohort = make_cohort(seed=seed)
        review = ReviewModel(seed=seed)
        km = make_method("kmeans", N_CLUSTERS, seed=seed).fit_predict(cohort.X, cohort.sensitive)
        base = M.kmeans_cost(cohort.X, km)
        for sl in slacks:
            model = make_method("constrained", N_CLUSTERS, seed=seed, count_slack=sl)
            labels = model.fit_predict(cohort.X, cohort.sensitive)
            r = evaluate_groups(cohort, labels, review)
            rows.append({"seed": seed, "count_slack": sl,
                         "slack_used": model.count_slack_used_,
                         "cost_ratio": r["cost"] / base, **{k: r[k] for k in
                         ("balance_gender", "balance_region", "propviol_gender",
                          "raw_gap", "cal_gap", "cal_spearman")}})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "e3_tradeoff_raw.csv", index=False)
    return df


def e4_assignment(seeds, r=4):
    configs = {
        "Affinity-max ILP": dict(fair=False),
        "Fair ILP: panel representation": dict(fair=True),
        "Fair ILP: panel + anti-homogeneity": dict(fair=True, max_same_cluster=1),
    }
    rows = []
    for seed in seeds:
        cohort = make_cohort(seed=seed)
        g, reg = cohort.sensitive["gender"], cohort.sensitive["region"]
        S = affinity_matrix(cohort.X)
        clusters = make_method("kmeans", N_CLUSTERS, seed=seed).fit_predict(cohort.X, cohort.sensitive)
        base_aff = None
        for name, kw in configs.items():
            t = time.time()
            a = ReviewerAssigner(reviews_per_submission=r, **kw)
            pairs = a.assign(S, cohort.sensitive, clusters)
            rt = time.time() - t
            if base_aff is None:
                base_aff = a.total_affinity_
            s = ReviewModel(seed=seed).score(cohort, pairs)
            raw = M.outcome_metrics(mean_grades(cohort.n, pairs, s), cohort.quality, g)
            cal = M.outcome_metrics(calibrated_grades(cohort.n, pairs, s, g), cohort.quality, g)
            ing = M.ingroup_review_share(pairs, g)
            loads = np.bincount(pairs[:, 0], minlength=cohort.n)
            rows.append({
                "seed": seed, "config": name, "runtime_s": rt,
                "affinity_retained": a.total_affinity_ / base_aff,
                "panel_violation_gender": M.panel_representation_violation(pairs, g, r),
                "panel_violation_region": M.panel_representation_violation(pairs, reg, r),
                "same_cluster_share": M.same_cluster_review_share(pairs, clusters),
                "ingroup_majority": ing[0], "ingroup_minority": ing[1],
                "load_minority_mean": loads[g == 1].mean(), "load_majority_mean": loads[g == 0].mean(),
                **{f"raw_{k}": v for k, v in raw.items()},
                **{f"cal_{k}": v for k, v in cal.items()},
            })
            print(f"  E4 seed={seed} {name:38s} {rt:5.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "e4_assignment_raw.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
def summarise(df, by, cols):
    agg = df.groupby(by, sort=False)[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


def fmt_table(df, by, cols, names):
    """Markdown table of mean ± std."""
    head = "| " + " | ".join([by] + [names[c] for c in cols]) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [head, sep]
    for key, grp in df.groupby(by, sort=False):
        cells = [LABEL.get(key, key)]
        for c in cols:
            cells.append(f"{grp[c].mean():.2f} ± {grp[c].std(ddof=0):.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fig_composition(examples):
    fig, axes = plt.subplots(1, len(examples), figsize=(14, 2.8), sharey=True)
    for ax, (m, (labels, g)) in zip(axes, examples.items()):
        k = labels.max() + 1
        share = np.array([g[labels == c].mean() if np.any(labels == c) else np.nan for c in range(k)])
        order = np.argsort(share)
        ax.bar(np.arange(k), share[order], width=0.8, color=COLOR[m], linewidth=0)
        ax.axhline(g.mean(), color=INK, lw=1, ls="--")
        ax.set_title(LABEL[m] if len(LABEL[m]) < 22 else LABEL[m].replace(" (", "\n("), fontsize=9, color=INK)
        ax.set_xticks([]); ax.set_xlabel("review groups (sorted)", color=MUTED)
        ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("minority share in group")
    axes[-1].annotate("cohort share", xy=(k - 1, g.mean()), xytext=(k - 1, g.mean() + 0.12),
                      ha="right", color=INK, fontsize=9)
    fig.suptitle("Demographic composition of peer-review groups (one cohort, 30 groups of ~8)",
                 x=0.01, ha="left", color=INK, y=1.08)
    fig.savefig(FIG / "fig1_group_composition.png"); plt.close(fig)


def fig_fairness_cost(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, title in ((axes[0], "balance_gender", "Balance (gender)"),
                           (axes[1], "balance_region", "Balance (region)")):
        for m in METHODS:
            d = df[df.method == m]
            ax.scatter(d.cost_ratio, d[col], s=40, color=COLOR[m], edgecolor="white",
                       linewidth=1.5, label=LABEL[m], zorder=3)
        ax.set_xlabel("cost ratio vs k-means (lower = more topical)")
        ax.set_ylabel(title); ax.set_ylim(-0.05, 1.05)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Fairness vs topical coherence of review groups (each dot = one seed)",
                 x=0.01, ha="left", color=INK)
    fig.savefig(FIG / "fig2_fairness_vs_cost.png"); plt.close(fig)


def fig_gaps(df):
    s = df.groupby("method", sort=False)[["raw_gap", "cal_gap"]].agg(["mean", "std"])
    x = np.arange(len(METHODS)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for off, kind, label, alpha in ((-w / 2, "raw_gap", "Mean of peer scores", 1.0),
                                    (w / 2, "cal_gap", "Calibrated grades", 0.45)):
        means = [s.loc[m, (kind, "mean")] for m in METHODS]
        stds = [s.loc[m, (kind, "std")] for m in METHODS]
        ax.bar(x + off, means, w - 0.03, yerr=stds, color=[COLOR[m] for m in METHODS],
               alpha=alpha, capsize=3, error_kw={"ecolor": MUTED, "lw": 1}, label=label)
    ax.set_xticks(x, [SHORT[m] for m in METHODS], fontsize=9)
    ax.set_ylabel("grade gap between groups\n(points on a 100 scale)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=a) for a in (1, .45)],
              labels=["Mean of peer scores", "Calibrated grades"], loc="upper right")
    ax.set_title("Evaluation bias: group gap in (grade − true quality); true quality is "
                 "identical across groups", loc="left", fontsize=10, color=INK)
    fig.savefig(FIG / "fig3_grade_gap.png"); plt.close(fig)


def fig_regimes(df):
    regimes = list(dict.fromkeys(df.regime))
    fig, axes = plt.subplots(1, 3, figsize=(17, 3.8), sharey=True)
    for ax, reg in zip(axes, regimes):
        d = df[df.regime == reg].groupby("method", sort=False)[["raw_gap", "cal_gap"]].mean()
        x = np.arange(len(METHODS)); w = 0.38
        ax.bar(x - w / 2, [d.loc[m, "raw_gap"] for m in METHODS], w - 0.03,
               color=[COLOR[m] for m in METHODS])
        ax.bar(x + w / 2, [d.loc[m, "cal_gap"] for m in METHODS], w - 0.03,
               color=[COLOR[m] for m in METHODS], alpha=0.45)
        ax.set_xticks(x, [SHORT[m] for m in METHODS], fontsize=7.5)
        ax.set_title(reg, fontsize=10, color=INK, loc="left")
    axes[0].set_ylabel("group grade gap")
    axes[-1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=a) for a in (1, .45)],
                    labels=["Mean of peer scores", "Calibrated"], loc="upper right")
    fig.savefig(FIG / "fig4_bias_regimes.png"); plt.close(fig)


def fig_tradeoff(df):
    s = df.groupby("count_slack").mean(numeric_only=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, col, title, slot in ((axes[0], "cost_ratio", "cost ratio vs k-means", 0),
                                 (axes[1], "propviol_gender", "max proportional violation (gender)", 1),
                                 (axes[2], "cal_gap", "calibrated grade gap", 2)):
        ax.plot(s.index, s[col], color=SLOTS[slot], lw=2, marker="o", ms=7,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.set_xlabel("representation slack (students per group)"); ax.set_title(title, fontsize=10,
                                                                                loc="left", color=INK)
        ax.set_xticks(s.index)
    fig.suptitle("Constrained fair k-means: loosening the fairness constraint",
                 x=0.01, ha="left", color=INK, y=1.03)
    fig.savefig(FIG / "fig5_tradeoff.png"); plt.close(fig)


def fig_assignment(df):
    configs = list(dict.fromkeys(df.config))
    s = df.groupby("config", sort=False).mean(numeric_only=True)
    panels = [("affinity_retained", "expertise affinity retained"),
              ("same_cluster_share", "share of reviews from author's own cluster"),
              ("ingroup_minority", "minority submissions: in-group review share"),
              ("cal_gap", "calibrated grade gap")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, (col, title) in zip(axes, panels):
        ax.barh(np.arange(len(configs)), [s.loc[c, col] for c in configs],
                color=SLOTS[:len(configs)], height=0.6)
        ax.set_yticks(np.arange(len(configs)), configs if ax is axes[0] else [""] * len(configs))
        ax.invert_yaxis(); ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
        for i, c in enumerate(configs):
            ax.text(s.loc[c, col], i, f" {s.loc[c, col]:.2f}", va="center", fontsize=9, color=INK)
    fig.suptitle("Reviewer assignment: affinity-maximising vs fairness-constrained ILP",
                 x=0.01, ha="left", color=INK, y=1.04)
    fig.savefig(FIG / "fig6_reviewer_assignment.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = [0, 1] if args.quick else list(range(10))
    a_seeds = [0, 1] if args.quick else list(range(5))
    slacks = [0, 1, 2] if args.quick else [0, 1, 2, 3, 4]

    t0 = time.time()
    print("E1 clustering ..."); e1, examples = e1_clustering(seeds)
    print("E2 bias regimes ..."); e2 = e2_regimes(seeds)
    print("E3 trade-off ..."); e3 = e3_tradeoff(seeds, slacks)
    print("E4 reviewer assignment ..."); e4 = e4_assignment(a_seeds)

    fig_composition(examples); fig_fairness_cost(e1); fig_gaps(e1)
    fig_regimes(e2); fig_tradeoff(e3); fig_assignment(e4)

    names = {"balance_gender": "Balance (gender) ↑", "balance_region": "Balance (region) ↑",
             "homogeneity_gender": "Homogeneity ↓", "cost_ratio": "Cost ratio ↓",
             "raw_gap": "Raw gap ↓", "cal_gap": "Calibrated gap ↓",
             "cal_spearman": "Calibrated ρ ↑", "raw_top_rate_ratio": "Top-20% DI ratio ↑",
             "cal_top_rate_ratio": "Top-20% DI ratio (cal.) ↑"}
    t1 = fmt_table(e1, "method", ["balance_gender", "balance_region", "homogeneity_gender",
                                  "cost_ratio", "raw_gap", "cal_gap", "cal_spearman",
                                  "cal_top_rate_ratio"], names)
    t2 = e2.pivot_table(index="method", columns="regime", values=["raw_gap", "cal_gap"],
                        sort=False).round(2)
    names4 = {"affinity_retained": "Affinity retained", "panel_violation_gender": "Panel viol. (gender) ↓",
              "same_cluster_share": "Same-cluster share ↓", "ingroup_minority": "In-group share (minority)",
              "ingroup_majority": "In-group share (majority)", "raw_gap": "Raw gap ↓",
              "cal_gap": "Calibrated gap ↓", "cal_spearman": "Calibrated ρ ↑"}
    t4 = fmt_table(e4, "config", list(names4), names4)
    t3 = e3.groupby("count_slack")[["cost_ratio", "balance_gender", "propviol_gender",
                                    "raw_gap", "cal_gap"]].mean().round(3)

    with open(RES / "summary.md", "w") as f:
        f.write("# Results summary\n\n")
        f.write(f"Seeds: clustering {len(seeds)}, assignment {len(a_seeds)}. "
                f"Cohort n=240, 30 groups, r=4 reviews per submission.\n\n")
        f.write("## E1 Peer-review groups\n\n" + t1 + "\n\n")
        f.write("## E2 Bias regimes (mean group gap)\n\n" + t2.to_markdown() + "\n\n")
        f.write("## E3 Fairness/cost trade-off\n\n" + t3.to_markdown() + "\n\n")
        f.write("## E4 Reviewer assignment\n\n" + t4 + "\n")
    summarise(e1, "method", [c for c in e1.columns if c not in ("seed", "method")]).to_csv(
        RES / "e1_clustering_summary.csv", index=False)
    summarise(e4, "config", [c for c in e4.columns if c not in ("seed", "config")]).to_csv(
        RES / "e4_assignment_summary.csv", index=False)
    print(open(RES / "summary.md").read())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
