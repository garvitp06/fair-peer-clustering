"""Synthetic student cohorts and a CSV loader.

The generator produces a cohort in which

* each student has a feature vector ``X`` (topic interests / skill profile),
* each student belongs to one or more demographic groups (sensitive attributes),
* features are *correlated* with group membership, so similarity-based grouping
  naturally produces demographically homogeneous clusters, and
* each student has a latent true quality ``q`` that is **independent** of group.

Because true quality is identical in distribution across groups, any group-level
difference in peer grades produced later is bias introduced by the evaluation
process, not a real difference in work quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Cohort:
    """A cohort of students.

    Attributes
    ----------
    X : (n, d) array of non-sensitive features used for clustering / affinity.
    sensitive : mapping attribute name -> (n,) integer-coded group labels.
    quality : (n,) latent true quality of each student's submission.
    leniency : (n,) each student's grading leniency when acting as a reviewer.
    group_names : mapping attribute name -> list of readable group labels.
    """

    X: np.ndarray
    sensitive: dict[str, np.ndarray]
    quality: np.ndarray
    leniency: np.ndarray
    group_names: dict[str, list[str]] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def primary(self) -> np.ndarray:
        """Labels of the first (primary) sensitive attribute."""
        return next(iter(self.sensitive.values()))

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.X, columns=[f"f{i}" for i in range(self.X.shape[1])])
        for name, labels in self.sensitive.items():
            df[name] = labels
        df["quality"] = self.quality
        df["leniency"] = self.leniency
        return df


def make_cohort(
    n: int = 240,
    d: int = 6,
    minority_frac: float = 0.3,
    n_regions: int = 3,
    group_shift: float = 1.6,
    leniency_strength: float = 3.0,
    leniency_noise: float = 1.0,
    seed: int | None = 0,
) -> Cohort:
    """Generate a synthetic cohort with group-correlated features.

    Parameters
    ----------
    n : number of students.
    d : feature dimension (>= 3).
    minority_frac : share of the minority group for the binary attribute ``gender``.
    n_regions : number of values of the second attribute ``region``.
    group_shift : separation (in feature std units) between group means. Larger
        values make similarity clustering more segregated.
    leniency_strength : how strongly a reviewer's grading leniency depends on
        their position in feature space. This models "local grading cultures":
        students in the same similarity cluster share a systematic harshness or
        generosity, which is the homogeneity-driven bias the project targets.
    leniency_noise : idiosyncratic per-reviewer leniency noise.
    seed : RNG seed.
    """
    if d < 3:
        raise ValueError("d must be >= 3")
    rng = np.random.default_rng(seed)

    gender = (rng.random(n) < minority_frac).astype(int)  # 1 = minority
    region = rng.integers(0, n_regions, size=n)

    X = rng.normal(size=(n, d))
    # Group-correlated structure: gender shifts dims 0-1, region shifts dim 2.
    X[:, 0] += group_shift * gender
    X[:, 1] -= 0.75 * group_shift * gender
    region_offsets = np.linspace(-group_shift, group_shift, n_regions)
    X[:, 2] += region_offsets[region]

    # Latent quality is independent of every sensitive attribute.
    quality = rng.normal(70.0, 10.0, size=n)

    # Leniency depends on the feature-space position (shared within clusters).
    u = np.zeros(d)
    u[0], u[1], u[2] = -0.8, 0.5, 0.3
    u /= np.linalg.norm(u)
    leniency = leniency_strength * np.tanh((X @ u) / 1.5)
    leniency += rng.normal(0.0, leniency_noise, size=n)

    return Cohort(
        X=X,
        sensitive={"gender": gender, "region": region},
        quality=quality,
        leniency=leniency,
        group_names={
            "gender": ["majority", "minority"],
            "region": [f"region_{r}" for r in range(n_regions)],
        },
    )


def load_cohort_csv(
    path: str,
    feature_cols: list[str],
    sensitive_cols: list[str],
    quality_col: str | None = None,
    leniency_col: str | None = None,
    standardize: bool = True,
) -> Cohort:
    """Load a real cohort from a CSV file.

    Sensitive columns may hold any categorical values; they are integer-coded.
    If ``quality_col`` is missing (real data rarely has ground truth) quality is
    set to NaN and outcome metrics that need it should be skipped. If
    ``leniency_col`` is missing, leniency is set to 0 (only used by the simulator).
    """
    df = pd.read_csv(path)
    X = df[feature_cols].to_numpy(dtype=float)
    if standardize:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    sensitive, names = {}, {}
    for col in sensitive_cols:
        codes, uniques = pd.factorize(df[col])
        sensitive[col] = codes.astype(int)
        names[col] = [str(u) for u in uniques]
    quality = df[quality_col].to_numpy(float) if quality_col else np.full(len(df), np.nan)
    leniency = df[leniency_col].to_numpy(float) if leniency_col else np.zeros(len(df))
    return Cohort(X=X, sensitive=sensitive, quality=quality, leniency=leniency, group_names=names)
