"""Reviewer assignment as an integer program.

Each of n students submits one piece of work and reviews ``r`` others. Variable
x[i, j] = 1 means student i reviews student j's submission.

Baseline (``fair=False``) — the common practice of matching by expertise:
    maximise   sum_ij  S[i, j] * x[i, j]
    subject to each submission gets exactly r reviewers,
               each reviewer has load in [r - load_slack, r + load_slack],
               no self-review.

Fair version (``fair=True``) adds, for every submission j:

* **Panel representation**: for each sensitive attribute and value v,
  floor(r * p_v) <= #reviewers of j from v <= ceil(r * p_v), so every panel
  mirrors the cohort's composition.
* **Anti-homogeneity**: at most ``max_same_cluster`` reviewers may share j's
  similarity cluster, which breaks up "echo chambers" with a shared grading
  culture.
* Optionally **min_outgroup**: at least that many reviewers from a different
  group than the author (for the primary attribute).

The constraint matrix is a bipartite b-matching plus side constraints; HiGHS
solves instances with a few hundred students in seconds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint

from ._solver import solve_binary


def affinity_matrix(X: np.ndarray) -> np.ndarray:
    """Cosine similarity of feature vectors (expertise / topic match)."""
    Z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Z @ Z.T


@dataclass
class ReviewerAssigner:
    reviews_per_submission: int = 4
    load_slack: int = 0
    fair: bool = False
    fair_attributes: list[str] | None = None
    max_same_cluster: int | None = None
    min_outgroup: int = 0
    time_limit: float = 60.0

    def assign(self, S: np.ndarray, sensitive: dict[str, np.ndarray],
               clusters: np.ndarray | None = None) -> np.ndarray:
        """Return an (m, 2) array of (reviewer, submission) pairs."""
        n = S.shape[0]
        r = self.reviews_per_submission
        # variables: all ordered pairs i != j
        I, J = np.where(~np.eye(n, dtype=bool))
        nv = len(I)
        var = np.arange(nv)
        A_rows, lo, hi = [], [], []

        def add(row_ids, n_rows, coefs, lb, ub):
            A_rows.append(sparse.csr_matrix((coefs, (row_ids, var)), shape=(n_rows, nv)))
            lo.append(np.broadcast_to(lb, n_rows).astype(float))
            hi.append(np.broadcast_to(ub, n_rows).astype(float))

        ones = np.ones(nv)
        add(J, n, ones, r, r)                                                # coverage
        add(I, n, ones, max(0, r - self.load_slack), r + self.load_slack)    # load

        if self.fair:
            attrs = self.fair_attributes if self.fair_attributes is not None else list(sensitive)
            for a in attrs:
                g = np.asarray(sensitive[a])
                for v in np.unique(g):
                    p = np.mean(g == v)
                    coef = (g[I] == v).astype(float)
                    add(J, n, coef, np.floor(r * p), np.ceil(r * p))
            if self.max_same_cluster is not None and clusters is not None:
                coef = (clusters[I] == clusters[J]).astype(float)
                add(J, n, coef, 0, self.max_same_cluster)
            if self.min_outgroup:
                g = np.asarray(next(iter(sensitive.values())))
                coef = (g[I] != g[J]).astype(float)
                add(J, n, coef, self.min_outgroup, r)

        cons = LinearConstraint(sparse.vstack(A_rows).tocsr(),
                                np.concatenate(lo), np.concatenate(hi))
        x = solve_binary(-S[I, J], cons, self.time_limit, owner=J)
        if x is None:
            raise RuntimeError("reviewer assignment infeasible under the given constraints")
        chosen = x > 0.5
        self.total_affinity_ = float(S[I[chosen], J[chosen]].sum())
        return np.column_stack([I[chosen], J[chosen]])


def pairs_from_clusters(labels: np.ndarray) -> np.ndarray:
    """All ordered (reviewer, submission) pairs inside each review group."""
    out = []
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        for i in members:
            for j in members:
                if i != j:
                    out.append((i, j))
    return np.asarray(out, dtype=int)
