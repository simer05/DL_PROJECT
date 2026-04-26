# Cooking-time FP32 control

_Last updated: 2026-04-26T06:39:42.584808+00:00 UTC_

## Setup

A single config: `cooking-time / baseline_plr_fp32-evaluation`, identical to the `baseline_plr` config except `amp = false` (= FP32 forward+backward), 5 seeds (0–4). Submitted as PBS jobs 13911135–13911139.

## Reference

- TabReD paper TabM-PLR cooking-time: mean = **-0.480366** (RMSE on negative), std = **0.000155** (n=15 in paper).

## Our 15-seed BF16 baseline (default amp=true)

- mean = **-0.482375**, std = **0.000131**, n=15
- deviation from paper: |Δ| = 0.002009 = **13.0σ** of paper std
- OUTSIDE ±2σ ✗ of paper TabM-PLR

## Our 5-seed FP32 control

- mean = **-0.485955**, std = **0.000463**, n=3
- deviation from paper: |Δ| = 0.005589 = **36.1σ** of paper std
- OUTSIDE ±2σ ✗ of paper TabM-PLR

## Verdict

Both FP32 and BF16 are outside ±2σ of paper. The cooking-time deviation is NOT precision-attributable. Other factors (data version, hyperparameter selection, or paper σ being unusually tight at 0.000155) may explain.

## Per-seed values

| seed | FP32 test | amp_dtype |
|---|---|---|
| 0 | -0.485583 | fp32 |
| 1 | -0.486473 | fp32 |
| 2 | -0.485809 | fp32 |