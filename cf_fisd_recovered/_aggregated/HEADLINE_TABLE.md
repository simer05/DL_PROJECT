# CF-FISD Headline Table — 15-seed paired tests vs baseline_plr

Single source of truth for headline numbers. Pulled from cluster `paper/exp/cf_fisd/tabred/<ds>/<variant>-evaluation/<seed>/report.json`.

_Higher is better for ROC-AUC datasets (homesite, ecom); higher (less negative) is better for RMSE-on-negative datasets (sberbank, cooking, delivery)._

| Dataset | Variant | n | mean (test) | std (test) | paired t vs baseline | paired p | seeds present |
|---|---|---|---|---|---|---|---|
| sberbank-housing | baseline_plr | 15 | -0.232882 | 0.002196 | (ref) | (ref) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| sberbank-housing | hetero_raw_lam0.05 | 15 | -0.233056 | 0.002605 | -0.306 | 0.7642 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| sberbank-housing | hetero_raw_lam0.1 | 15 | -0.233095 | 0.002394 | -0.338 | 0.7404 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| sberbank-housing | hetero_raw_lam0.2 | 15 | -0.233137 | 0.002452 | -0.501 | 0.6239 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| ecom-offers | baseline_plr | 15 | +0.590172 | 0.001508 | (ref) | (ref) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| ecom-offers | hetero_raw_lam0.05 | 15 | +0.590073 | 0.001487 | -0.714 | 0.4868 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| ecom-offers | hetero_raw_lam0.1 | 15 | +0.590089 | 0.001481 | -0.611 | 0.551 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| ecom-offers | hetero_raw_lam0.2 | 15 | +0.590046 | 0.001484 | -0.877 | 0.3951 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| homesite-insurance | baseline_plr | 15 | +0.962372 | 0.000298 | (ref) | (ref) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| homesite-insurance | hetero_raw_lam0.05 | 15 | +0.962735 | 0.000274 | +5.026 | 0.0001854 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| homesite-insurance | hetero_raw_lam0.1 | 15 | +0.962672 | 0.000280 | +3.351 | 0.004751 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| homesite-insurance | hetero_raw_lam0.2 | 15 | +0.962648 | 0.000265 | +5.400 | 9.353e-05 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| cooking-time | baseline_plr | 15 | -0.482375 | 0.000131 | (ref) | (ref) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| cooking-time | hetero_raw_lam0.05 | 15 | -0.482431 | 0.000106 | -1.554 | 0.1425 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| cooking-time | hetero_raw_lam0.1 | 15 | -0.482512 | 0.000116 | -3.172 | 0.006789 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| cooking-time | hetero_raw_lam0.2 | 15 | -0.482469 | 0.000111 | -2.421 | 0.02965 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| delivery-eta | baseline_plr | 15 | -0.551563 | 0.002073 | (ref) | (ref) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| delivery-eta | hetero_raw_lam0.05 | 15 | -0.553538 | 0.002290 | -2.114 | 0.05293 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| delivery-eta | hetero_raw_lam0.1 | 15 | -0.553378 | 0.003021 | -2.144 | 0.0501 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |
| delivery-eta | hetero_raw_lam0.2 | 15 | -0.553596 | 0.002553 | -2.108 | 0.05349 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] |

## Consensus (preliminary, n=5 across all 15 cells)

| Dataset | Variant | n | mean (test) | std (test) | paired t vs baseline | paired p | seeds present |
|---|---|---|---|---|---|---|---|
| sberbank-housing | consensus_raw_lam0.05 | 5 | -0.233064 | 0.001552 | +1.159 | 0.3109 | [0, 1, 2, 3, 4] |
| sberbank-housing | consensus_raw_lam0.1 | 5 | -0.234097 | 0.002589 | +0.238 | 0.8236 | [0, 1, 2, 3, 4] |
| sberbank-housing | consensus_raw_lam0.2 | 5 | -0.232447 | 0.000668 | +1.828 | 0.1416 | [0, 1, 2, 3, 4] |
| ecom-offers | consensus_raw_lam0.05 | 5 | +0.589881 | 0.000992 | -2.208 | 0.0918 | [0, 1, 2, 3, 4] |
| ecom-offers | consensus_raw_lam0.1 | 5 | +0.589825 | 0.001073 | -4.703 | 0.009286 | [0, 1, 2, 3, 4] |
| ecom-offers | consensus_raw_lam0.2 | 5 | +0.589696 | 0.001060 | -2.648 | 0.05707 | [0, 1, 2, 3, 4] |
| homesite-insurance | consensus_raw_lam0.05 | 5 | +0.962805 | 0.000333 | +2.365 | 0.0772 | [0, 1, 2, 3, 4] |
| homesite-insurance | consensus_raw_lam0.1 | 5 | +0.962790 | 0.000284 | +2.787 | 0.04948 | [0, 1, 2, 3, 4] |
| homesite-insurance | consensus_raw_lam0.2 | 5 | +0.962742 | 0.000300 | +3.423 | 0.02671 | [0, 1, 2, 3, 4] |
| cooking-time | consensus_raw_lam0.05 | 5 | -0.482891 | 0.000164 | -6.328 | 0.003192 | [0, 1, 2, 3, 4] |
| cooking-time | consensus_raw_lam0.1 | 5 | -0.482903 | 0.000162 | -6.107 | 0.003638 | [0, 1, 2, 3, 4] |
| cooking-time | consensus_raw_lam0.2 | 5 | -0.482869 | 0.000157 | -6.394 | 0.003071 | [0, 1, 2, 3, 4] |
| delivery-eta | consensus_raw_lam0.05 | 5 | -0.550972 | 0.002713 | +0.764 | 0.4872 | [0, 1, 2, 3, 4] |
| delivery-eta | consensus_raw_lam0.1 | 5 | -0.553753 | 0.002878 | -0.855 | 0.4409 | [0, 1, 2, 3, 4] |
| delivery-eta | consensus_raw_lam0.2 | 5 | -0.551203 | 0.002840 | +0.501 | 0.6425 | [0, 1, 2, 3, 4] |

_Generated by `step2_3_4_headline.py` from cluster reports. Baseline-paired t-tests use same-seed pairing._