# Results summary

Seeds: clustering 10, assignment 5. Cohort n=240, 30 groups, r=4 reviews per submission.

## E1 Peer-review groups

| method | Balance (gender) ↑ | Balance (region) ↑ | Homogeneity ↓ | Cost ratio ↓ | Raw gap ↓ | Calibrated gap ↓ | Calibrated ρ ↑ | Top-20% DI ratio (cal.) ↑ |
|---|---|---|---|---|---|---|---|---|
| K-means | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.84 ± 0.02 | 1.00 ± 0.00 | 2.88 ± 0.34 | 1.86 ± 0.41 | 0.97 ± 0.01 | 0.72 ± 0.15 |
| Size-constrained | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.84 ± 0.03 | 1.05 ± 0.01 | 2.87 ± 0.26 | 1.88 ± 0.26 | 0.97 ± 0.01 | 0.68 ± 0.18 |
| Random groups | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.73 ± 0.03 | 3.31 ± 0.09 | 1.66 ± 0.45 | 0.30 ± 0.19 | 0.98 ± 0.00 | 0.83 ± 0.14 |
| Fairlet k-means | 0.34 ± 0.06 | 0.00 ± 0.00 | 0.71 ± 0.03 | 1.39 ± 0.05 | 1.68 ± 0.41 | 0.20 ± 0.17 | 0.98 ± 0.00 | 0.80 ± 0.11 |
| Constrained (gender only) | 0.31 ± 0.06 | 0.00 ± 0.00 | 0.72 ± 0.03 | 1.32 ± 0.05 | 1.81 ± 0.34 | 0.26 ± 0.20 | 0.98 ± 0.00 | 0.76 ± 0.16 |
| Constrained (gender + region) | 0.31 ± 0.06 | 0.65 ± 0.05 | 0.72 ± 0.03 | 1.97 ± 0.09 | 1.80 ± 0.40 | 0.21 ± 0.08 | 0.98 ± 0.00 | 0.79 ± 0.10 |

## E2 Bias regimes (mean group gap)

| method             |   ('raw_gap', 'Leniency culture only') |   ('raw_gap', 'In-group favouritism only') |   ('raw_gap', 'Both mechanisms') |   ('cal_gap', 'Leniency culture only') |   ('cal_gap', 'In-group favouritism only') |   ('cal_gap', 'Both mechanisms') |
|:-------------------|---------------------------------------:|-------------------------------------------:|---------------------------------:|---------------------------------------:|-------------------------------------------:|---------------------------------:|
| kmeans             |                                   1.83 |                                       1.16 |                             2.88 |                                   1.86 |                                       0.37 |                             1.86 |
| size_only          |                                   1.79 |                                       1.14 |                             2.87 |                                   1.85 |                                       0.19 |                             1.88 |
| random             |                                   0.26 |                                       1.68 |                             1.66 |                                   0.28 |                                       0.31 |                             0.3  |
| fairlet            |                                   0.27 |                                       1.88 |                             1.68 |                                   0.2  |                                       0.22 |                             0.2  |
| constrained_gender |                                   0.22 |                                       1.89 |                             1.81 |                                   0.24 |                                       0.22 |                             0.26 |
| constrained        |                                   0.16 |                                       1.89 |                             1.8  |                                   0.23 |                                       0.19 |                             0.21 |

## E3 Fairness/cost trade-off

|   count_slack |   cost_ratio |   balance_gender |   propviol_gender |   raw_gap |   cal_gap |
|--------------:|-------------:|-----------------:|------------------:|----------:|----------:|
|             0 |        1.975 |            0.314 |             0.079 |     1.797 |     0.209 |
|             1 |        1.535 |            0.129 |             0.204 |     2.524 |     0.863 |
|             2 |        1.234 |            0     |             0.32  |     2.84  |     1.521 |
|             3 |        1.111 |            0     |             0.44  |     2.847 |     1.705 |
|             4 |        1.062 |            0     |             0.565 |     2.839 |     1.794 |

## E4 Reviewer assignment

| config | Affinity retained | Panel viol. (gender) ↓ | Same-cluster share ↓ | In-group share (minority) | In-group share (majority) | Raw gap ↓ | Calibrated gap ↓ | Calibrated ρ ↑ |
|---|---|---|---|---|---|---|---|---|
| Affinity-max ILP | 1.00 ± 0.00 | 0.26 ± 0.02 | 0.52 ± 0.02 | 0.57 ± 0.04 | 0.83 ± 0.02 | 3.04 ± 0.18 | 1.31 ± 0.27 | 0.97 ± 0.01 |
| Fair ILP: panel representation | 0.86 ± 0.01 | 0.06 ± 0.03 | 0.33 ± 0.02 | 0.34 ± 0.06 | 0.74 ± 0.02 | 3.21 ± 0.44 | 0.79 ± 0.36 | 0.97 ± 0.00 |
| Fair ILP: panel + anti-homogeneity | 0.85 ± 0.01 | 0.06 ± 0.03 | 0.21 ± 0.01 | 0.34 ± 0.06 | 0.74 ± 0.02 | 3.05 ± 0.57 | 0.68 ± 0.50 | 0.97 ± 0.01 |
