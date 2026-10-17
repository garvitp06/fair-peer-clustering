"""fairpeer: fairness-aware clustering and reviewer assignment for peer evaluation."""
from .data import Cohort, make_cohort, load_cohort_csv
from .clustering import (StandardKMeans, RandomBalanced, FairletKMeans,
                         ConstrainedFairKMeans, make_method)
from .assignment import ReviewerAssigner, affinity_matrix, pairs_from_clusters
from .simulation import ReviewModel, mean_grades, calibrated_grades
from . import metrics

__version__ = "1.0.0"
