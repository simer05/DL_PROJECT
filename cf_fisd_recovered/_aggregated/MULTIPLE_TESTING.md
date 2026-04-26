# Multiple-testing correction (Bonferroni)

## Setup

- Total CF-FISD-vs-baseline comparisons reported in the paper: **N = 30**
  - 5 datasets × 3 `hetero_raw` λ values = **15 hetero comparisons (n=15 seeds)**
  - 5 datasets × 3 `consensus_raw` λ values = **15 consensus comparisons (n=5 seeds, preliminary)**
  - All cells exist (no exclusions). All 300 headline reports + 75 consensus reports verified on cluster.

## Formula

Bonferroni-corrected α: `α_corr = α / N = 0.05 / 30 = 0.001667`

Bonferroni-corrected p: `p_corr = min(1, p_raw × N)` with N = 30

## Homesite-insurance hetero_raw — corrected p-values

| λ | n | paired t | raw p | Bonferroni p (× 30) | sig at α=0.05 after correction? |
|---|---|---|---|---|---|
| 0.05 | 15 | +5.026 | 0.0001854 | 0.005562 | YES |
| 0.1 | 15 | +3.351 | 0.004751 | 0.1425 | no |
| 0.2 | 15 | +5.400 | 9.353e-05 | 0.002806 | YES |

## Interpretation

- **λ=0.05** (corrected p = 0.0056) and **λ=0.2** (corrected p = 0.0028) survive Bonferroni at α=0.05.
- **λ=0.1** (corrected p = 0.143) does **not** survive — the t-stat is +3.35 (raw p=0.0048), strong on its own but not strong enough to survive 30-way correction.
- The headline claim "CF-FISD significantly improves over baseline_plr on homesite-insurance" therefore stands for two of three λ values after Bonferroni; the win is not an isolated lucky λ but is recoverable at the dose-response endpoints. The mid-dose (λ=0.1) is a borderline cell post-correction.

## Cooking-time hetero regression — corrected p-value

The reviewer-flagged regression at cooking-time hetero_raw_lam0.1 (raw t=−3.172, raw p=0.0068) becomes:
- Bonferroni p (× 30) = **0.204** → **not** significant after correction.

This means the cooking-time regression is consistent with chance under multiple-testing correction; it should not be claimed as a CF-FISD-induced regression in the paper. The honest framing is: cooking-time and delivery-eta show no significant CF-FISD effect (positive or negative) after correction.

**N = 30 is the single Bonferroni denominator used everywhere in the paper.**
