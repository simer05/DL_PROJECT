# SAFE ESAM Restart Plan

## Known Issues to Fix
1. Forced share_training_batches override in run_experiment_esam.py.
2. Weak compact budget leaking into final runs.
3. Selection/report parser mismatch with nested output layout.
4. Simplistic preprocessing split and leakage risk (especially ecom-offers).
5. Unstratified train-fraction subsampling.
6. Feature NaN validation bug in paper/lib/data.py.
7. delivery-eta metric/data degeneracy.
8. ecom-offers leakage-risk and trivial baseline behavior.

## Dataset Status Target
- Primary valid targets: homesite-insurance, cooking-time, sberbank-housing.
- Conditional: ecom-offers (only if audit passes), delivery-eta (only if audit passes).

## Planned Phases
- Phase 0: archive/reset/progress files.
- Phase 1: implementation fixes and parser/report fixes.
- Phase 2: preprocessing + dataset readiness + audits.
- Phase 3: baseline equivalence smoke runs.
- Phase 4: define Safe-ESAM++ variants + validation margin selection.
- Phase 5: full-data seed42 corrected screen.
- Phase 6: low-data seed42 corrected screen.
- Phase 7: confirmation seeds (only where ESAM selected by validation).
- Phase 8: robustness evaluation (if time permits and checkpoints available).
- Phase 9: final reports/CSV/patch.

## Initial Run Estimate (sequential one-GPU baseline plan)
- Full-data screen: up to 5 datasets x 5 variants = 25 runs.
- Low-data screen: up to 5 datasets x 5 variants = 25 runs.
- Confirmation: variable (only selected pairs), estimated 8-20 runs.
- Total expected: 58-70 runs plus audits/smokes.

## ETA (initial)
- Fixes + audits: 3-5 hours.
- Corrected full/low screens + confirmation: 14-24 hours (depends on delivery-eta).
- Reporting: 1-2 hours.
