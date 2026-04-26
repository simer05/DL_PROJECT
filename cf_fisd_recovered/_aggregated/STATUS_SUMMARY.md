# CF-FISD Status Summary (for report writing)

_Generated 2026-04-26 ~14:35 SGT. Re-read after each monitor tick for refreshed numbers._

## 1. What was cancelled vs kept

**Cancelled** (per user instruction "ignore rest if they are not important"):
- `br9smjec3` (local throttler): consensus 15-seed extension across 5 datasets × 3 λ. Stopped before any new jobs were submitted past the throttle cap.
- `bcmpx2x4z` (local throttler): softmax + l1norm + homo (XGB/LGBM/CAT) ablations across 5 datasets × 9 variants × 5 seeds = 225 jobs. Stopped before any made it past the throttle cap.
- `qdel` count: **0** cluster jobs needed cancelling (the throttle cap of 5 concurrent jobs meant non-essential queued submissions never hit PBS).

**Kept (actively running)**:
- 8 teacher-seed bootstrap jobs (PBS IDs 13910555/57/58/59/66/67/68/69). These are exactly **homesite ×4 seeds + delivery ×4 seeds** — the 8 missing cells from the 12 already-complete (sberbank, ecom, cooking each have all 4 seeds done).
- 5 FP32 control jobs (PBS IDs 13911135–13911139): cooking-time `baseline_plr_fp32` (`amp = false`), seeds 0–4. Queued.

**Treated as preliminary, no more cluster work**:
- Consensus runs at n=5 for **all 15 (dataset, λ) cells** symmetrically — no cherry-picking. Reported as preliminary in `HEADLINE_TABLE.md`.

## 2. ETA

| Stream | Jobs | Worst-case wall | Done by (SGT) |
|---|---|---|---|
| Teacher-seed bootstrap | 8 jobs (homesite + delivery) | walltime cap 02:00; oldest at 0:56 elapsed; homesite teacher ~70 min total | **~15:30–16:00 SGT** |
| FP32 control | 5 jobs (queued behind teachers) | ~15 min each on 1 GPU, runs parallel after queue clears | **~16:30–17:30 SGT** |
| Bootstrap monitor (background) | local | re-checks every 30 min until cutoff | until **18:30 SGT** (4-hr hard cutoff from now) |
| FP32 monitor (background) | local | re-checks every 10 min until 5/5 reports | until ~17:30 SGT |

## 3. Files for report writing (single source of truth)

All under `C:\Users\Simerjit Kaur\cf_fisd_recovered\_aggregated\`:

| File | Purpose |
|---|---|
| `HEADLINE_TABLE.md` | Single source of truth for headline 15-seed paired t-tests + consensus n=5 preliminary |
| `MULTIPLE_TESTING.md` | Bonferroni denominator (N=30), corrected p-values for homesite headline, honest interpretation |
| `fig_homesite_dose_response.{pdf,png}` | Figure 1 |
| `fig_diagnostic_scatter.{pdf,png}` | Figure 2 |
| `analysis_branchA.md` | Analysis section if bootstrap completes — validated predictor framing |
| `analysis_branchB.md` | Analysis section if bootstrap doesn't complete — descriptive heuristic framing (safe fallback) |
| `BOOTSTRAP_STATUS.md` | Live status of teacher-seed bootstrap; ends with explicit "Submit Branch A / Submit Branch B" recommendation |
| `COOKING_FP32_CONTROL.md` | Will hold FP32 vs BF16 comparison + verdict on whether cooking-time deviation is precision-attributable |
| `step2_3_4_summary.json` | Machine-readable backing data for HEADLINE / MULTIPLE_TESTING / Figure 1 |
| `diagnostic_correlations.json` | Spearman ρ + feature-bootstrap CI per dataset (used by Figure 2) |

## 4. Key headline numbers (lifted from `HEADLINE_TABLE.md`)

| Dataset | hetero_raw best λ | mean test | paired t vs baseline | raw p | Bonferroni p (N=30) | survives? |
|---|---|---|---|---|---|---|
| sberbank-housing | — | all NS | -0.30 to -0.50 | 0.62–0.76 | — | no |
| ecom-offers | — | all NS | -0.61 to -0.88 | 0.40–0.55 | — | no |
| **homesite-insurance** | **λ=0.05** | **+0.962735** | **+5.026** | **0.000185** | **0.0056** | **YES** |
| **homesite-insurance** | **λ=0.2** | **+0.962648** | **+5.400** | **9.35e-5** | **0.0028** | **YES** |
| homesite-insurance | λ=0.1 (mid-dose) | +0.962672 | +3.351 | 0.0048 | 0.143 | no |
| cooking-time | λ=0.1 (regression) | -0.482512 | -3.172 | 0.0068 | 0.204 | no |
| delivery-eta | all λ | all borderline | -2.10 to -2.14 | 0.050–0.053 | ≥1.0 | no |

**Headline claim that survives Bonferroni:** CF-FISD significantly improves test ROC-AUC on homesite-insurance at λ=0.05 (corrected p=0.0056) and λ=0.2 (corrected p=0.0028). All other (dataset, λ) cells are non-significant after correction.

## 5. Diagnostic rule (heuristic, derived from these 5 datasets)

> **WIN-region heuristic.** CF-FISD improves over `baseline_plr` if and only if ρ(XGB, LGBM) > 0.10 **and** ρ(LGBM, CAT) ≥ 0.7.

Classifies all 5 datasets correctly at the seed-0 teacher fits (per `diagnostic_correlations.json`). Bootstrap (4 additional seeds per dataset) tests whether classification survives teacher-fit variability — see `BOOTSTRAP_STATUS.md` for live result.

**Disclosure (must appear in paper, both branches):** _The rule was derived from observing these same 5 datasets; it should be understood as a descriptive heuristic consistent with our observations rather than a held-out prediction._

## 6. Open items at write-time

1. **Wait for bootstrap to complete homesite + delivery** — by ~16:00 SGT (8 jobs running). Then `BOOTSTRAP_STATUS.md` will declare Branch A or Branch B.
2. **Wait for FP32 control** — by ~17:00–17:30 SGT (5 jobs queued). Then `COOKING_FP32_CONTROL.md` will declare whether cooking-time deviation from paper is BF16-attributable.
3. **Pick the analysis section** based on `BOOTSTRAP_STATUS.md` recommendation (`analysis_branchA.md` or `analysis_branchB.md`).
4. **Hard cutoff: 18:30 SGT.** If bootstrap not 80% by then, `BOOTSTRAP_STATUS.md` will mark not-in-time and recommend the safe Branch B fallback.

## 7. NSCC compliance

- ✓ All teacher-seed PBS scripts use `#PBS -P personal-simerjit`.
- ✓ Login-node activity limited to `qstat`, `qdel`, `qsub`, file reads/writes via SFTP.
- ✓ Queue depth currently 13 (8 teacher-seed + 5 FP32 control); within sensible cap.
- ✓ No teammate directories accessed.
