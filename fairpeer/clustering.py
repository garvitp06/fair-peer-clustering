"""Clustering algorithms for forming peer-review groups.

All estimators share one interface::

    labels = model.fit_predict(X, sensitive)

where ``sensitive`` maps attribute name -> integer group labels. Methods:

``StandardKMeans``
    Plain Lloyd k-means (fairness-unaware baseline). Groups similar students
    together, which on group-correlated features yields homogeneous clusters.

``RandomBalanced``
    Random partition into equal-size groups. Fair in expectation but ignores
    topical coherence entirely; a reference point for the cost axis.

``FairletKMeans``
    Fair k-means via *fairlet decomposition* (Chierichetti et al., NeurIPS 2017)
    for a binary attribute. Points are first partitioned into small, balanced
    "fairlets" (one minority point plus t or t+1 majority points) with a
    minimum-cost assignment; fairlets are then clustered by weighted k-means.
    Every cluster is a union of fairlets, so its balance is guaranteed to be at
    least 1 / (t + 1).

``ConstrainedFairKMeans``
    Constraint-based fair k-means in the style of Bera et al. (NeurIPS 2019):
    Lloyd iterations whose assignment step is an integer program with
    (a) proportional-representation (integer count) bounds for *every* value of *every*
    sensitive attribute in every cluster (so several attributes are handled
    simultaneously), and (b) cluster-size bounds, so review groups have
    near-equal workload. Each assignment step is solved with HiGHS (LP relaxation plus an exact
    integer repair of the few fractional points, see ``_solver``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint, linear_sum_assignment

from ._solver import solve_binary
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return cdist(X, C, metric="sqeuclidean")


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
@dataclass
class StandardKMeans:
    n_clusters: int
    seed: int | None = 0
    n_init: int = 10

    def fit_predict(self, X, sensitive=None):
        km = KMeans(self.n_clusters, n_init=self.n_init, random_state=self.seed)
        labels = km.fit_predict(X)
        self.cluster_centers_ = km.cluster_centers_
        return labels


@dataclass
class RandomBalanced:
    n_clusters: int
    seed: int | None = 0

    def fit_predict(self, X, sensitive=None):
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        labels = np.arange(n) % self.n_clusters
        rng.shuffle(labels)
        self.cluster_centers_ = np.vstack(
            [X[labels == c].mean(axis=0) for c in range(self.n_clusters)]
        )
        return labels


# --------------------------------------------------------------------------- #
# Fairlet decomposition (Chierichetti et al. 2017)
# --------------------------------------------------------------------------- #
@dataclass
class FairletKMeans:
    """Fair k-means through fairlet decomposition for one binary attribute.

    Parameters
    ----------
    n_clusters : number of clusters.
    attribute : name of the binary sensitive attribute to balance. If ``None``
        the first attribute in ``sensitive`` is used.
    seed : random state for the k-means step on fairlet centers.
    """

    n_clusters: int
    attribute: str | None = None
    seed: int | None = 0
    n_init: int = 10

    def decompose(self, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """Return a fairlet id for every point.

        Each fairlet contains exactly one point of the smaller group ("anchor")
        and either t or t+1 points of the larger group, where
        t = floor(n_large / n_small). The assignment of large-group points to
        anchors minimises total squared distance, solved exactly with the
        Hungarian algorithm on an expanded slot matrix: each anchor owns t
        mandatory slots (given a large negative bonus so they are always filled)
        plus one optional slot.
        """
        values = np.unique(groups)
        if len(values) != 2:
            raise ValueError("FairletKMeans supports exactly two groups; "
                             "use ConstrainedFairKMeans for multi-valued attributes.")
        counts = {v: int(np.sum(groups == v)) for v in values}
        small = min(counts, key=counts.get)
        anchors = np.flatnonzero(groups == small)
        others = np.flatnonzero(groups != small)
        m, M = len(anchors), len(others)
        t = M // m
        if t < 1:
            raise ValueError("degenerate group sizes")

        D = _sq_dists(X[others], X[anchors])           # (M, m)
        big = D.max() * (M + 1) + 1.0
        slots_per_anchor = t + 1
        # columns: anchor a, slot s -> column a * (t+1) + s
        cost = np.repeat(D, slots_per_anchor, axis=1)
        mandatory = np.tile(np.arange(slots_per_anchor) < t, m)
        cost[:, mandatory] -= big
        rows, cols = linear_sum_assignment(cost)

        fairlet = np.empty(X.shape[0], dtype=int)
        fairlet[anchors] = np.arange(m)
        fairlet[others[rows]] = cols // slots_per_anchor
        self.t_ = t
        return fairlet

    def fit_predict(self, X, sensitive):
        attr = self.attribute or next(iter(sensitive))
        groups = np.asarray(sensitive[attr])
        fairlet = self.decompose(X, groups)
        n_f = fairlet.max() + 1
        centers = np.vstack([X[fairlet == f].mean(axis=0) for f in range(n_f)])
        sizes = np.bincount(fairlet, minlength=n_f)
        km = KMeans(self.n_clusters, n_init=self.n_init, random_state=self.seed)
        f_labels = km.fit_predict(centers, sample_weight=sizes)
        labels = f_labels[fairlet]
        self.fairlets_ = fairlet
        self.cluster_centers_ = np.vstack(
            [X[labels == c].mean(axis=0) if np.any(labels == c) else km.cluster_centers_[c]
             for c in range(self.n_clusters)]
        )
        return labels


# --------------------------------------------------------------------------- #
# Constraint-based fair k-means (Bera et al. 2019 style, exact MILP assignment)
# --------------------------------------------------------------------------- #
@dataclass
class ConstrainedFairKMeans:
    """Lloyd-style fair k-means with an exact constrained assignment step.

    Parameters
    ----------
    n_clusters : number of clusters (review groups).
    fair_attributes : attributes whose representation is constrained. ``None``
        means all attributes in ``sensitive``; an empty list disables fairness
        constraints (leaving only size constraints, a useful ablation).
    count_slack : representation slack, in students. A group with n_v members
        has a fair share of n_v / k per cluster; every cluster must contain
        between floor(n_v/k) - count_slack and ceil(n_v/k) + count_slack of
        them. ``count_slack = 0`` is "as balanced as integrality allows".
        Integer count bounds are used instead of fractional share bounds
        because review groups are small (6–10 students), where fractional
        bounds are either vacuous or infeasible.
    size_slack : cluster sizes must lie in [floor(n/k) - size_slack,
        ceil(n/k) + size_slack].
    max_relax : if the program is infeasible (possible when several attributes
        interact), ``count_slack`` is increased by one up to this many times.
        The slack actually used is stored in ``count_slack_used_``.
    """

    n_clusters: int
    fair_attributes: list[str] | None = None
    count_slack: int = 0
    size_slack: int = 0
    max_iter: int = 20
    seed: int | None = 0
    max_relax: int = 5
    time_limit: float = 30.0
    history_: list = field(default_factory=list, init=False, repr=False)

    def _build_constraints(self, n, sensitive, slack):
        k = self.n_clusters
        nv = n * k                                   # x[i, c] -> index i * k + c
        cols = np.arange(nv)
        point_of = np.repeat(np.arange(n), k)
        cluster_of = np.tile(np.arange(k), n)
        rows_A, rows_lo, rows_hi = [], [], []

        def add(row_ids, n_rows, coef, lb, ub):
            rows_A.append(sparse.csr_matrix((coef, (row_ids, cols)), shape=(n_rows, nv)))
            rows_lo.append(np.full(n_rows, lb, float)); rows_hi.append(np.full(n_rows, ub, float))

        ones = np.ones(nv)
        add(point_of, n, ones, 1, 1)                                     # one cluster each
        add(cluster_of, k, ones, max(1, n // k - self.size_slack),       # balanced sizes
            -(-n // k) + self.size_slack)

        attrs = self.fair_attributes if self.fair_attributes is not None else list(sensitive)
        for a in attrs:                                                  # fair representation
            g = np.asarray(sensitive[a])
            for v in np.unique(g):
                n_v = int(np.sum(g == v))
                lo = max(0, n_v // k - slack)
                hi = -(-n_v // k) + slack
                add(cluster_of, k, (g[point_of] == v).astype(float), lo, hi)
        return LinearConstraint(sparse.vstack(rows_A).tocsr(),
                                np.concatenate(rows_lo), np.concatenate(rows_hi))

    def _assign(self, D, sensitive, slack):
        n, k = D.shape
        cons = self._build_constraints(n, sensitive, slack)
        x = solve_binary(D.ravel(), cons, self.time_limit, owner=np.repeat(np.arange(n), k))
        if x is None:
            return None
        return x.reshape(n, k).argmax(axis=1)

    def fit_predict(self, X, sensitive):
        k = self.n_clusters
        centers = KMeans(k, n_init=10, random_state=self.seed).fit(X).cluster_centers_

        slack = self.count_slack
        labels = None
        self.history_ = []
        for it in range(self.max_iter):
            D = _sq_dists(X, centers)
            new = self._assign(D, sensitive, slack)
            while new is None:
                slack += 1
                if slack > self.count_slack + self.max_relax:
                    raise RuntimeError("no feasible fair assignment found")
                new = self._assign(D, sensitive, slack)
            cost = float(D[np.arange(len(new)), new].sum())
            self.history_.append(cost)
            if labels is not None and np.array_equal(new, labels):
                break
            labels = new
            centers = np.vstack([X[labels == c].mean(axis=0) if np.any(labels == c)
                                 else centers[c] for c in range(k)])
        self.count_slack_used_ = slack
        self.n_iter_ = it + 1
        self.cluster_centers_ = centers
        return labels


def make_method(name: str, n_clusters: int, seed: int = 0, **kw):
    """Factory used by experiments."""
    name = name.lower()
    if name == "kmeans":
        return StandardKMeans(n_clusters, seed=seed)
    if name == "random":
        return RandomBalanced(n_clusters, seed=seed)
    if name == "fairlet":
        return FairletKMeans(n_clusters, attribute=kw.get("attribute"), seed=seed)
    if name == "size_only":
        return ConstrainedFairKMeans(n_clusters, fair_attributes=[], seed=seed,
                                     size_slack=kw.get("size_slack", 0))
    if name == "constrained_gender":
        return ConstrainedFairKMeans(n_clusters, fair_attributes=[kw.get("attribute", "gender")],
                                     count_slack=kw.get("count_slack", 0), seed=seed,
                                     size_slack=kw.get("size_slack", 0))
    if name == "constrained":
        return ConstrainedFairKMeans(n_clusters, fair_attributes=kw.get("fair_attributes"),
                                     count_slack=kw.get("count_slack", 0), seed=seed,
                                     size_slack=kw.get("size_slack", 0))
    raise ValueError(f"unknown method {name}")
