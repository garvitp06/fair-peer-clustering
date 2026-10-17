"""Minimal end-to-end example: form fair review groups and grade a cohort."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fairpeer import (ConstrainedFairKMeans, ReviewModel, StandardKMeans, calibrated_grades,
                      make_cohort, mean_grades, pairs_from_clusters)
from fairpeer import metrics as M

cohort = make_cohort(n=240, seed=42)
gender = cohort.sensitive["gender"]

for name, model in [("k-means", StandardKMeans(30, seed=0)),
                    ("constrained fair k-means", ConstrainedFairKMeans(30, seed=0))]:
    labels = model.fit_predict(cohort.X, cohort.sensitive)
    pairs = pairs_from_clusters(labels)                    # everyone reviews their group
    scores = ReviewModel(seed=0).score(cohort, pairs)      # simulated biased peer scores
    raw = M.outcome_metrics(mean_grades(cohort.n, pairs, scores), cohort.quality, gender)
    cal = M.outcome_metrics(calibrated_grades(cohort.n, pairs, scores, gender),
                            cohort.quality, gender)
    print(f"{name:26s} balance={M.balance(labels, gender):.2f}  "
          f"cost={M.kmeans_cost(cohort.X, labels):6.0f}  "
          f"gap raw={raw['gap']:.2f} calibrated={cal['gap']:.2f}")
