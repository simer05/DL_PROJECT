# ECOM Root Cause and Fix Plan

## Root Cause
- Raw columns: id, chain, offer, market, repeattrips, repeater, offerdate.
- With current configured features [chain, market, repeattrips], label is deterministic by feature-group (all groups pure).
- This is a leakage-risk pattern for the experiment objective.

## Proposed Fix
- Create `ecom-offers-v2` by removing `repeattrips` in addition to id/offer/offerdate.
- Use exact-feature-key group split (no hash-collision risk).
- Keep train-only encoding/fitting.
- Re-run majority/logistic sanity checks before any GPU training rows are claimed scientific.

## Evidence
- Deterministic groups with old feature set: yes
- ecom-v2 duplicate overlap: {'train_val': 0, 'train_test': 0, 'val_test': 0}
- ecom-v2 majority val/test: 0.714394/0.723298
- ecom-v2 logistic val/test: 0.714394/0.723298

## Decision
- ecom-offers-v2 is suitable for a scientific row candidate (no trivial perfect baseline, no split overlap).