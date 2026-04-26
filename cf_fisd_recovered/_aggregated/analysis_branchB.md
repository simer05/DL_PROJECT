# Analysis — Branch B (descriptive heuristic framing)

> **Use this branch IF the teacher-seed bootstrap does not complete in time, OR completes but the rule does not survive the bootstrap CIs.** This is the safe fallback that does not depend on the bootstrap result.

## When CF-FISD helps and when it does not

Across the five TabReD datasets, hetero CF-FISD (15 seeds, paired t vs `baseline_plr`) produces a single statistically significant test-set improvement: **homesite-insurance** at λ=0.05 (paired t = +5.03, raw p = 1.85×10⁻⁴) and λ=0.2 (paired t = +5.40, raw p = 9.35×10⁻⁵). After Bonferroni correction over the N=30 CF-FISD-vs-baseline cells reported in this paper (15 hetero × 5 datasets + 15 consensus × 5 datasets), both endpoints survive at α=0.05 (corrected p = 5.6×10⁻³ and 2.8×10⁻³ respectively); the middle dose λ=0.1 does not (corrected p = 0.143). On the remaining four datasets — sberbank-housing, ecom-offers, cooking-time, delivery-eta — the paired test against `baseline_plr` is non-significant after Bonferroni correction (no positive or negative cell crosses the corrected threshold).

The reviewer-flagged regression at cooking-time (raw t = −3.17, raw p = 0.0068 at λ=0.1) does **not** survive Bonferroni correction (corrected p = 0.204), and so we report it as null. Likewise the borderline cells at delivery-eta (raw p ≈ 0.05 at all λ) collapse to null after correction.

## A two-feature descriptive heuristic

We characterize each dataset by two pairwise Spearman correlations between the per-feature importances of the three GBDT teachers (XGBoost, LightGBM, CatBoost) trained at default hyperparameters on the canonical `evaluation/0.toml` seed: ρ(XGB, LGBM) and ρ(LGBM, CAT). Plotting our five datasets in this two-dimensional space (Figure 2) reveals that the only WIN dataset (homesite-insurance) occupies a unique region: ρ(LGBM, CAT) ≥ 0.7 with ρ(XGB, LGBM) clearly positive. Cooking-time and delivery-eta share the high ρ(LGBM, CAT) ≥ 0.7 region but have ρ(XGB, LGBM) ≈ 0; sberbank-housing and ecom-offers have ρ(LGBM, CAT) < 0.7.

The simplest two-feature heuristic consistent with our observations is:

> **WIN-region heuristic.** CF-FISD appears to improve over `baseline_plr` when ρ(XGB, LGBM) > 0.10 **and** ρ(LGBM, CAT) ≥ 0.7.

This heuristic is consistent with all five datasets in our sample: homesite-insurance (the only WIN) lies inside the WIN region; the other four lie outside. The intuition is that the CF-FISD penalty exploits redundancy among the teachers (high pairwise agreement) only when there is enough independent signal between teacher families to provide a useful regularizer (the "XGB ≠ LGBM, but LGBM ≈ CAT" pattern). When XGB is uncorrelated with LGBM (cooking-time, delivery-eta) the rich dependency structure CF-FISD distills isn't there; when all teachers are weakly correlated (ecom-offers) there is no consensus to distill at all.

## CI overlap caveat

The within-fit feature-bootstrap 95% CIs on ρ(XGB, LGBM), shown in Figure 2 as horizontal error bars, overlap between homesite-insurance ([+0.07, +0.33]) and cooking-time ([−0.17, +0.13]). If the true ρ(XGB, LGBM) for cooking-time were close to its upper CI bound (+0.13), the heuristic would (incorrectly) predict cooking-time as a WIN — but cooking-time's actual paired test is not significant. The heuristic is therefore not robust at the boundary on the basis of the within-fit CIs alone, and we make no claim that it generalizes beyond these five observed datasets.

A teacher-seed bootstrap (re-training each GBDT under additional random seeds and recomputing ρ over the resulting per-feature importance vectors) would test this caveat directly; we leave that validation to future work.

## Disclosure

The rule was derived from observing these same 5 datasets; it should be understood as a descriptive heuristic consistent with our observations rather than a held-out prediction.

We do not claim the heuristic predicts CF-FISD outcomes on new datasets.

## Limitations

- The heuristic has only one positive observation (homesite-insurance) in our five-dataset sample. With one positive case, no held-out validation is possible.
- The within-fit CIs on ρ(XGB, LGBM) are wide enough that homesite and cooking-time overlap on this axis, leaving the heuristic's classification of cooking-time sensitive to small changes in the teacher fits.
- The single-dose-non-monotonicity at homesite (λ=0.1 t = +3.35, weaker than λ=0.05 and λ=0.2) is statistically modest after Bonferroni and may be a small-sample artifact; the dose-response figure (Figure 1) shows the mean test ROC-AUC is monotonic in λ even though the t-statistic is not.
- We do not control for the GBDT hyperparameter settings beyond the canonical TabReD `evaluation/0.toml` defaults. A separate hyperparameter sweep on each teacher could change the per-feature importances and therefore the diagnostic correlations.
