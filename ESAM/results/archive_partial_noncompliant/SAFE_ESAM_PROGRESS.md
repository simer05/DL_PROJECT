# SAFE ESAM Progress Log

## 2026-04-26T12:00:26
- Phase: 0 (Archive + Reset)
- Completed: git state capture, archive completed, clean directories recreated, restart plan written.
- Running: moving to Phase 1 code fixes.
- Runs completed/planned: 0 / 58-70 (initial estimate)
- ETA current phase: 1-2 hours
- ETA remaining pipeline: 18-31 hours (single-GPU compliant plan)
- Latest result path: /mnt/ssd/users/prithvi/deepLearning/results/esam

## 2026-04-26T12:26:41
- Phase: 1-2 (Fixes + Preprocessing)
- Completed: run_experiment_esam.py fixed (no forced share_training_batches, stratified low-data subsampling, new Safe-ESAM flags); selection/report scripts fixed; data.py NaN bug fixed; new seed42 5-variant parallel launcher prepared.
- Running: preprocess_tabred_to_tabm.py rebuilding all 5 datasets with corrected split policies.
- Runs completed/planned: 0 / 25 (seed42 5-dataset x 5 variants full-data screen)
- Current dataset block: preprocessing delivery-eta tail
- ETA current phase: 20-60 min
- ETA remaining pipeline: 10-18 hours (depends on delivery-eta and confirmation scope)
- Latest result path: /mnt/ssd/users/prithvi/deepLearning/results/esam

## 2026-04-26T12:45:42
- Phase: 2 (Preprocessing tail)
- Completed: code fixes and launch scripts are ready; 4/5 datasets regenerated (homesite, cooking-time, sberbank-housing, ecom-offers).
- Running: delivery-eta preprocessing tail (PID 528866).
- Runs completed/planned: 0 / 25 (seed42 full-data 5 variants x up to 5 datasets).
- ETA current phase: 20-60 min (depends on delivery finalization).
- ETA from training start: 4-6 hours (2-GPU parallel on GPU0/1 only).
- ETA total from now: 5-7 hours (including readiness + audits + smoke + selection/report; confirmation only if time remains).
- Risk factors: delivery-eta audit/readiness may fail or run slow; GPU0/1 currently occupied by external process.
- Latest result path: /mnt/ssd/users/prithvi/deepLearning/results/esam

## 2026-04-26T13:15:40
- Phase: Seed42 full-data screen (continue, no restart)
- Completed: homesite-insurance 5/5 variants finished; cooking-time baseline running; ecom + delivery audits finished.
- Reused runs: 6 valid runs reused/kept (no restart, no deletion).
- Running: cooking-time variant loop on GPU1.
- Remaining: 19 runs (cooking 4, sberbank 5, ecom 5, delivery 5).
- Audit status:
  - ecom-offers: audit_passed=false, included_in_scientific_claims=false, reason=leakage-risk (trivial logistic 1.0/1.0).
  - delivery-eta: audit_passed=partial, included_in_scientific_claims=conditional, reason=target pipeline numerically valid; training-metric behavior still to verify in run rows.
- ETA ecom+delivery block: ~2.5 to 5.5 hours (depends heavily on delivery runtime).
- ETA total remaining: ~4.5 to 8.5 hours single-GPU; faster if GPU0 becomes free and second worker starts.
- Latest result path: /mnt/ssd/users/prithvi/deepLearning/results/esam

## 2026-04-26T13:18:18
- Continuation queued: ecom-offers + delivery-eta 5-variant block will auto-start immediately after current core block (homesite/cooking/sber) completes.
- Watcher log: /mnt/ssd/users/prithvi/deepLearning/logs/esam/continue_ecom_delivery_waiter.log
- Continuation launch log: /mnt/ssd/users/prithvi/deepLearning/logs/esam/safe_esam_launch_ecom_delivery.log

## 2026-04-26T13:34:17
- Phase: Implementation validity check
- Completed: SAFE_ESAM_IMPLEMENTATION_VALIDITY_CHECK.md/json generated; selection dryrun CSV generated.
- Decision: B. CONTINUE BUT MARK SOME DATASETS DIAGNOSTIC-ONLY
- Completed runs: 6, Active runs: 3, Failed heuristic: 0, Skipped: 1
- ecom validity: audit_passed=False, include_in_claims=False
- delivery validity: audit_passed=True, include_in_claims=True (conditional until delivery run rows complete)
