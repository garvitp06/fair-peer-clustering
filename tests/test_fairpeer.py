import numpy as np
import pytest

from fairpeer import (ConstrainedFairKMeans, FairletKMeans, ReviewModel, ReviewerAssigner,
                      StandardKMeans, affinity_matrix, calibrated_grades, make_cohort,
                      mean_grades, pairs_from_clusters)
from fairpeer import metrics as M


@pytest.fixture(scope="module")
def cohort():
    return make_cohort(n=120, seed=3)


# ------------------------------- data -------------------------------------- #
def test_quality_independent_of_group(cohort):
    big = make_cohort(n=4000, seed=1)
    g = big.sensitive["gender"]
    assert abs(big.quality[g == 1].mean() - big.quality[g == 0].mean()) < 1.0


def test_features_are_group_correlated(cohort):
    labels = StandardKMeans(15, seed=0).fit_predict(cohort.X)
    assert M.homogeneity(labels, cohort.sensitive["gender"]) > 0.78


# ------------------------------ metrics ------------------------------------ #
def test_balance_known_values():
    g = np.array([0, 0, 1, 1, 0, 0])
    assert M.balance(np.array([0, 0, 0, 1, 1, 1]), g) == pytest.approx(0.5)
    assert M.balance(np.array([0, 0, 1, 1, 0, 0]), g) == 0.0


def test_proportional_violation_zero_when_mirrored():
    g = np.array([0, 1, 0, 1])
    assert M.proportional_violation(np.array([0, 0, 1, 1]), g) == pytest.approx(0.0)


# ---------------------------- clustering ----------------------------------- #
def test_fairlet_balance_guarantee(cohort):
    g = cohort.sensitive["gender"]
    model = FairletKMeans(12, seed=0)
    labels = model.fit_predict(cohort.X, cohort.sensitive)
    # each fairlet: exactly one minority point plus t or t+1 majority points
    for f in np.unique(model.fairlets_):
        members = g[model.fairlets_ == f]
        minority = 1 if g.mean() < 0.5 else 0
        assert (members == minority).sum() == 1
        assert model.t_ <= (members != minority).sum() <= model.t_ + 1
    assert M.balance(labels, g) >= 1 / (model.t_ + 1) - 1e-9


def test_fairlet_rejects_multivalued(cohort):
    with pytest.raises(ValueError):
        FairletKMeans(5, attribute="region").fit_predict(cohort.X, cohort.sensitive)


def test_constrained_satisfies_bounds(cohort):
    k = 15
    model = ConstrainedFairKMeans(k, count_slack=0, size_slack=0, seed=0)
    labels = model.fit_predict(cohort.X, cohort.sensitive)
    sizes = np.bincount(labels, minlength=k)
    assert sizes.min() >= cohort.n // k and sizes.max() <= -(-cohort.n // k)
    slack = model.count_slack_used_
    for g in cohort.sensitive.values():
        for v in np.unique(g):
            n_v = (g == v).sum()
            counts = np.bincount(labels[g == v], minlength=k)
            assert counts.min() >= n_v // k - slack
            assert counts.max() <= -(-n_v // k) + slack


def test_constrained_beats_kmeans_on_fairness(cohort):
    g = cohort.sensitive["gender"]
    km = StandardKMeans(15, seed=0).fit_predict(cohort.X)
    fair = ConstrainedFairKMeans(15, seed=0).fit_predict(cohort.X, cohort.sensitive)
    assert M.balance(fair, g) > M.balance(km, g)
    assert M.kmeans_cost(cohort.X, fair) >= M.kmeans_cost(cohort.X, km)


def test_constrained_without_fairness_is_size_only(cohort):
    labels = ConstrainedFairKMeans(15, fair_attributes=[], seed=0).fit_predict(
        cohort.X, cohort.sensitive)
    assert set(np.bincount(labels)) == {8}


# ---------------------------- assignment ----------------------------------- #
@pytest.mark.parametrize("fair", [False, True])
def test_assignment_constraints(cohort, fair):
    r = 3
    S = affinity_matrix(cohort.X)
    clusters = StandardKMeans(15, seed=0).fit_predict(cohort.X)
    a = ReviewerAssigner(r, fair=fair, max_same_cluster=1 if fair else None)
    pairs = a.assign(S, cohort.sensitive, clusters)
    assert not np.any(pairs[:, 0] == pairs[:, 1])                     # no self review
    assert set(np.bincount(pairs[:, 1], minlength=cohort.n)) == {r}   # coverage
    assert set(np.bincount(pairs[:, 0], minlength=cohort.n)) == {r}   # equal load
    assert len({tuple(p) for p in pairs}) == len(pairs)               # no duplicates
    if fair:
        for g in cohort.sensitive.values():
            for v in np.unique(g):
                p = np.mean(g == v)
                cnt = np.bincount(pairs[g[pairs[:, 0]] == v, 1], minlength=cohort.n)
                assert cnt.min() >= np.floor(r * p) and cnt.max() <= np.ceil(r * p)
        same = clusters[pairs[:, 0]] == clusters[pairs[:, 1]]
        assert np.bincount(pairs[same, 1], minlength=cohort.n).max() <= 1


def test_fair_assignment_costs_some_affinity(cohort):
    S = affinity_matrix(cohort.X)
    base = ReviewerAssigner(3); base.assign(S, cohort.sensitive)
    fair = ReviewerAssigner(3, fair=True); fair.assign(S, cohort.sensitive)
    assert fair.total_affinity_ <= base.total_affinity_ + 1e-6


# ---------------------------- simulation ----------------------------------- #
def test_calibration_recovers_quality_without_noise():
    c = make_cohort(n=90, seed=5)
    labels = ConstrainedFairKMeans(10, seed=0).fit_predict(c.X, c.sensitive)
    pairs = pairs_from_clusters(labels)
    s = ReviewModel(noise=0.0, delta_region=0.0, seed=0).score(c, pairs)
    g = c.sensitive["gender"]
    q_hat = calibrated_grades(c.n, pairs, s, g, ridge=1e-6)
    raw = mean_grades(c.n, pairs, s)
    assert np.corrcoef(q_hat, c.quality)[0, 1] > np.corrcoef(raw, c.quality)[0, 1]
    assert M.outcome_metrics(q_hat, c.quality, g)["gap"] < 0.5


def test_outcome_metrics_zero_gap_for_perfect_grades():
    q = np.linspace(0, 100, 50); g = np.arange(50) % 2
    out = M.outcome_metrics(q.copy(), q, g)
    assert out["gap"] == pytest.approx(0.0) and out["spearman"] == pytest.approx(1.0)
