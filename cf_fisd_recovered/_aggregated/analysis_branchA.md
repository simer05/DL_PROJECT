# Analysis — Branch A (validated predictor framing)

> **Use this branch IF the teacher-seed bootstrap (5 datasets × 4 seeds = 20 jobs) completes and the diagnostic rule survives the bootstrap CIs.** This branch upgrades the rule from "descriptive heuristic" to "validated predictor of CF-FISD outcome."

## When CF-FISD helps and when it does not

Across the five TabReD datasets, hetero CF-FISD (15 seeds, paired t vs `baseline_plr`) produces a single statistically significant test-set improvement: **homesite-insurance** at λ=0.05 (paired t = +5.03, raw p = 1.85×10⁻⁴) and λ=0.2 (paired t = +5.40, raw p = 9.35×10⁻⁵). After Bonferroni correction over the N=30 CF-FISD-vs-baseline cells reported in this paper (15 hetero × 5 datasets + 15 consensus × 5 datasets), both endpoints survive at α=0.05 (corrected p = 5.6×10⁻³ and 2.8×10⁻³ respectively); the middle dose λ=0.1 does not (corrected p = 0.143). On the remaining four datasets — sberbank-housing, ecom-offers, cooking-time, delivery-eta — the paired test against `baseline_plr` is non-significant after Bonferroni correction (no positive or negative cell crosses the corrected threshold).

The reviewer-flagged regression at cooking-time (raw t = −3.17, raw p = 0.0068 at λ=0.1) does **not** survive Bonferroni correction (corrected p = 0.204), and so we report it as null. Likewise the borderline cells at delivery-eta (raw p ≈ 0.05 at all λ) collapse to null after correction.

## A two-feature diagnostic rule for CF-FISD outcome

We characterize each dataset by two pairwise Spearman correlations between the per-feature importances of the three GBDT teachers (XGBoost, LightGBM, CatBoost) trained at default hyperparameters on the canonical `evaluation/0.toml` seed: ρ(XGB, LGBM) and ρ(LGBM, CAT). Plotting our five datasets in this two-dimensional space (Figure 2) and overlaying the three regions of the heuristic rule reveals that the only WIN dataset (homesite-insurance) occupies a unique region: ρ(LGBM, CAT) ≥ 0.7 with ρ(XGB, LGBM) clearly positive. Cooking-time and delivery-eta share the high ρ(LGBM, CAT) ≥ 0.7 region but have ρ(XGB, LGBM) ≈ 0; sberbank-housing and ecom-offers have ρ(LGBM, CAT) < 0.7. The simplest two-feature rule consistent with our observations is:

> **WIN-region rule.** CF-FISD is predicted to improve over `baseline_plr` if and only if ρ(XGB, LGBM) > 0.10 **and** ρ(LGBM, CAT) ≥ 0.7.

This rule classifies all five TabReD datasets correctly (homesite WIN; the other four match) — but only when the teacher-importance correlations are taken at face value. Because the rule was derived from observing these same five datasets, we test its robustness directly.

## Teacher-seed bootstrap

We re-train each GBDT teacher (XGB, LGBM, CAT) under four additional random seeds (`seed=1, 2, 3, 4`), recompute the pairwise Spearman correlations on the resulting per-feature importances, and recheck the rule on each draw. The empirical 2.5–97.5 percentile interval over the four bootstrap teacher fits (i.e., the teacher-seed bootstrap CI) gives a far more honest measure of the rule's robustness than the within-fit feature-bootstrap CI shown in Figure 2.

**Bootstrap result (under Branch A's assumption that the bootstrap completes and the rule survives):**

| Dataset | ρ(XGB, LGBM) — teacher-seed CI | ρ(LGBM, CAT) — teacher-seed CI | Rule prediction | Actual outcome |
|---|---|---|---|---|
| homesite-insurance | [{HOM_XL_LO}, {HOM_XL_HI}] | [{HOM_LC_LO}, {HOM_LC_HI}] | WIN | WIN |
| cooking-time       | [{COO_XL_LO}, {COO_XL_HI}] | [{COO_LC_LO}, {COO_LC_HI}] | match | match |
| delivery-eta       | [{DEL_XL_LO}, {DEL_XL_HI}] | [{DEL_LC_LO}, {DEL_LC_HI}] | match | match |
| sberbank-housing   | [{SBE_XL_LO}, {SBE_XL_HI}] | [{SBE_LC_LO}, {SBE_LC_HI}] | match | match |
| ecom-offers        | [{ECO_XL_LO}, {ECO_XL_HI}] | [{ECO_LC_LO}, {ECO_LC_HI}] | match | match |

_Bootstrap CIs to be filled in once the 20 teacher-seed jobs complete (replace `{...}` placeholders with values from `BOOTSTRAP_STATUS.md`)._

**Claim (Branch A):** Under the teacher-seed bootstrap, the WIN-region rule classifies all five datasets correctly with bootstrap CIs that do not cross the rule boundaries. The classification is therefore not an artifact of the specific seed used to fit the teachers; it reflects a structural property of the (XGB, LGBM, CAT) feature-importance geometry on each dataset.

## Disclosure

The rule was derived from observing these same 5 datasets; it should be understood as a descriptive heuristic consistent with our observations rather than a held-out prediction.

The teacher-seed bootstrap evaluates how stable the rule is under the variation introduced by re-fitting each GBDT under different seeds — it is **not** a held-out test of generalization to new datasets. Validating the rule on a sixth, held-out TabReD dataset (or an out-of-distribution tabular benchmark) is left as future work.

## Limitations

- The rule has only one positive observation (homesite-insurance) in our five-dataset sample. The bootstrap addresses sampling noise from teacher fitting but does not address dataset-sampling variability.
- The single-dose-non-monotonicity at homesite (λ=0.1 t = +3.35, weaker than λ=0.05 and λ=0.2) is statistically modest after Bonferroni and may be a small-sample artifact; the dose-response figure (Figure 1) shows the mean test ROC-AUC is monotonic in λ even though the t-statistic is not.
- We do not control for the GBDT hyperparameter settings beyond the canonical TabReD `evaluation/0.toml` defaults. A separate hyperparameter sweep on each teacher could change the per-feature importances and therefore the diagnostic correlations.
