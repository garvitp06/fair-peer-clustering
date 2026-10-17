"""Peer-review simulator and grade aggregation.

Score model for reviewer i grading submission j::

    s_ij = q_j + leniency_i + delta * [g_i == g_j] + delta_region * [r_i == r_j] + eps

* ``q_j`` – true quality (independent of group).
* ``leniency_i`` – reviewer harshness/generosity. It varies smoothly in feature
  space, so reviewers in the same similarity cluster share it: a homogeneous
  group inflates or deflates all its members' grades together.
* ``delta`` – in-group favouritism on the primary attribute.

Two aggregators are provided:

``mean_grades``
    The usual average of received scores.
``calibrated_grades``
    Least-squares fit of the additive model s_ij = q_j + b_i + delta*same_ij
    (ridge on b, the reviewer effects), returning the estimated q_j. Leniency and
    favouritism are only *identifiable* when panels mix groups and clusters; under
    homogeneous assignment they are confounded with quality. Fair assignment is
    therefore what makes this calibration work.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from .data import Cohort


@dataclass
class ReviewModel:
    delta: float = 4.0
    delta_region: float = 1.5
    noise: float = 4.0
    seed: int | None = 0

    def score(self, cohort: Cohort, pairs: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        i, j = pairs[:, 0], pairs[:, 1]
        g = cohort.sensitive.get("gender", cohort.primary())
        s = cohort.quality[j] + cohort.leniency[i] + self.delta * (g[i] == g[j])
        if "region" in cohort.sensitive:
            reg = cohort.sensitive["region"]
            s = s + self.delta_region * (reg[i] == reg[j])
        return s + rng.normal(0.0, self.noise, size=len(pairs))


def mean_grades(n: int, pairs: np.ndarray, scores: np.ndarray) -> np.ndarray:
    tot = np.bincount(pairs[:, 1], weights=scores, minlength=n)
    cnt = np.bincount(pairs[:, 1], minlength=n)
    return tot / np.maximum(cnt, 1)


def calibrated_grades(n: int, pairs: np.ndarray, scores: np.ndarray,
                      groups: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """Estimate submission quality while removing reviewer and in-group effects."""
    m = len(pairs)
    i, j = pairs[:, 0], pairs[:, 1]
    same = (groups[i] == groups[j]).astype(float)
    rows = np.arange(m)
    # columns: [q_0..q_{n-1}, b_0..b_{n-1}, delta]
    A = sparse.hstack([
        sparse.csr_matrix((np.ones(m), (rows, j)), shape=(m, n)),
        sparse.csr_matrix((np.ones(m), (rows, i)), shape=(m, n)),
        sparse.csr_matrix(same.reshape(-1, 1)),
    ])
    # ridge rows on reviewer effects b (centres them on zero)
    R = sparse.hstack([sparse.csr_matrix((n, n)),
                       np.sqrt(ridge) * sparse.identity(n),
                       sparse.csr_matrix((n, 1))])
    A = sparse.vstack([A, R]).tocsr()
    y = np.concatenate([scores, np.zeros(n)])
    sol = lsqr(A, y, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    q_hat = sol[:n]
    # fix the overall level (not identifiable against the in-group constant)
    return q_hat - q_hat.mean() + scores.mean() - sol[-1] * same.mean()
