"""Fairness, quality and outcome metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


# ----------------------------- cluster fairness ----------------------------- #
def balance(labels: np.ndarray, groups: np.ndarray) -> float:
    """Chierichetti et al. balance, generalised to many groups.

    For each cluster, the ratio min_g |C∩g| / max_g |C∩g| over all groups present
    in the population; the score is the minimum over clusters (1 = perfectly
    mixed, 0 = some cluster lacks a group entirely).
    """
    vals = np.unique(groups)
    worst = 1.0
    for c in np.unique(labels):
        counts = np.array([np.sum((labels == c) & (groups == v)) for v in vals])
        worst = min(worst, counts.min() / counts.max())
    return float(worst)


def proportional_violation(labels: np.ndarray, groups: np.ndarray) -> float:
    """Largest |share of g in cluster − share of g in population| over clusters, groups."""
    vals = np.unique(groups)
    pop = np.array([np.mean(groups == v) for v in vals])
    worst = 0.0
    for c in np.unique(labels):
        mask = labels == c
        share = np.array([np.mean(groups[mask] == v) for v in vals])
        worst = max(worst, float(np.abs(share - pop).max()))
    return worst


def mean_entropy(labels: np.ndarray, groups: np.ndarray) -> float:
    """Size-weighted group entropy of clusters, normalised by population entropy."""
    vals = np.unique(groups)

    def H(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    pop = H(np.array([np.mean(groups == v) for v in vals]))
    tot = 0.0
    for c in np.unique(labels):
        mask = labels == c
        tot += mask.mean() * H(np.array([np.mean(groups[mask] == v) for v in vals]))
    return tot / pop if pop > 0 else 1.0


def homogeneity(labels: np.ndarray, groups: np.ndarray) -> float:
    """Size-weighted average share of each cluster's dominant group."""
    tot = 0.0
    for c in np.unique(labels):
        mask = labels == c
        tot += mask.mean() * np.bincount(groups[mask]).max() / mask.sum()
    return float(tot)


def kmeans_cost(X: np.ndarray, labels: np.ndarray) -> float:
    """Within-cluster sum of squared distances to cluster means."""
    return float(sum(((X[labels == c] - X[labels == c].mean(axis=0)) ** 2).sum()
                     for c in np.unique(labels)))


def size_spread(labels: np.ndarray) -> tuple[int, int]:
    s = np.bincount(labels)
    s = s[s > 0]
    return int(s.min()), int(s.max())


# ---------------------------- assignment metrics ---------------------------- #
def ingroup_review_share(pairs: np.ndarray, groups: np.ndarray) -> dict[int, float]:
    """For submissions of each group, share of their reviews written by same-group reviewers."""
    i, j = pairs[:, 0], pairs[:, 1]
    same = groups[i] == groups[j]
    return {int(v): float(same[groups[j] == v].mean()) for v in np.unique(groups)}


def panel_representation_violation(pairs: np.ndarray, groups: np.ndarray, r: int) -> float:
    """Mean over submissions of max_g |share of g in panel − population share|."""
    vals = np.unique(groups)
    pop = np.array([np.mean(groups == v) for v in vals])
    out = []
    for j in np.unique(pairs[:, 1]):
        rev = groups[pairs[pairs[:, 1] == j, 0]]
        share = np.array([np.mean(rev == v) for v in vals])
        out.append(np.abs(share - pop).max())
    return float(np.mean(out))


def same_cluster_review_share(pairs: np.ndarray, clusters: np.ndarray) -> float:
    return float(np.mean(clusters[pairs[:, 0]] == clusters[pairs[:, 1]]))


# ------------------------------ outcome metrics ----------------------------- #
def outcome_metrics(grades: np.ndarray, quality: np.ndarray, groups: np.ndarray,
                    top_frac: float = 0.2) -> dict[str, float]:
    """Bias of evaluation outcomes with respect to the true quality.

    * ``gap`` – max − min over groups of mean(grade − quality). Because quality is
      group-independent, a non-zero gap is pure evaluation bias.
    * ``err_std`` – std of (grade − quality); level-free measure of grading noise.
    * ``spearman`` – rank agreement between grades and true quality.
    * ``top_rate_ratio`` – disparate-impact ratio of being selected in the top
      ``top_frac`` (min group rate / max group rate; 1 = parity).
    * ``top_precision`` – share of selected students that are truly in the top.
    """
    err = grades - quality
    vals = np.unique(groups)
    means = [err[groups == v].mean() for v in vals]
    k = max(1, int(round(top_frac * len(grades))))
    sel = np.zeros(len(grades), bool); sel[np.argsort(-grades)[:k]] = True
    true_top = np.zeros(len(grades), bool); true_top[np.argsort(-quality)[:k]] = True
    rates = [sel[groups == v].mean() for v in vals]
    return {
        "gap": float(max(means) - min(means)),
        "err_std": float(np.std(err)),
        "spearman": float(spearmanr(grades, quality).statistic),
        "top_rate_ratio": float(min(rates) / max(rates)) if max(rates) > 0 else 1.0,
        "top_precision": float((sel & true_top).sum() / k),
    }
