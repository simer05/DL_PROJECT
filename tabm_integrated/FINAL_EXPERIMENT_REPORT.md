# Integrated TabM final experiment report

Official TabM PLR/PiecewiseLinearEmbeddings baseline configs are preserved exactly; variants add only module flags for RLA, ESAM, MFB, and CF-FISD.

## Final 3-seed results

| dataset | variant | source config | metric | inference | mean ± std | delta | status |
|---|---|---|---|---|---:|---:|---|
| sberbank-housing | baseline_plr | baseline_plr | RMSE | mean | 0.234418 ± 0.00354822 | 0 | baseline |
| sberbank-housing | best_rla_only | rla_rank2_noise0.001 | RMSE | greedy-heads | 0.232368 ± 0.00228551 | 0.0020499 | weak_win |
| sberbank-housing | best_esam_only | esam_rho0.0025 | RMSE | greedy-heads | 0.234406 ± 0.00401172 | 1.17846e-05 | weak_win |
| sberbank-housing | best_mfb_only | mfb_keep0.8 | RMSE | greedy-heads | 0.2351 ± 0.00395661 | -0.000681772 | loss |
| sberbank-housing | best_cf_fisd_only | cf_fisd_only | RMSE | greedy-heads | 0.233093 ± 0.000585082 | 0.0013248 | weak_win |
| sberbank-housing | best_combined | mfb_cf_fisd | RMSE | greedy-heads | 0.234979 ± 0.00337063 | -0.000560913 | loss |
| ecom-offers | baseline_plr | baseline_plr | AUROC | mean | 0.590058 ± 0.000970364 | 0 | baseline |
| ecom-offers | best_rla_only | rla_rank2_noise0.0001 | AUROC | best-head | 0.598447 ± 0.00279735 | 0.00838888 | clear_win |
| ecom-offers | best_esam_only | esam_rho0.001 | AUROC | greedy-heads | 0.60042 ± 0.00349335 | 0.0103616 | clear_win |
| ecom-offers | best_mfb_only | mfb_keep0.7 | AUROC | best-head | 0.609516 ± 0.00208948 | 0.0194584 | clear_win |
| ecom-offers | best_cf_fisd_only | cf_fisd_lambda0.05 | AUROC | greedy-heads | 0.600205 ± 0.00345825 | 0.0101468 | clear_win |
| ecom-offers | best_combined | mfb_cf_fisd | AUROC | greedy-heads | 0.605211 ± 0.00322248 | 0.0151529 | clear_win |
| homesite-insurance | baseline_plr | baseline_plr | AUROC | mean | 0.962494 ± 0.000178079 | 0 | baseline |
| homesite-insurance | best_rla_only | rla_rank4_noise1e-05 | AUROC | greedy-heads | 0.962564 ± 0.000608529 | 6.98613e-05 | weak_win |
| homesite-insurance | best_esam_only | esam_only | AUROC | greedy-heads | 0.962732 ± 0.00044923 | 0.000237685 | clear_win |
| homesite-insurance | best_mfb_only | mfb_keep0.95 | AUROC | greedy-heads | 0.962382 ± 0.00045772 | -0.000111855 | loss |
| homesite-insurance | best_cf_fisd_only | cf_fisd_only | AUROC | greedy-heads | 0.962908 ± 0.000474027 | 0.000413458 | clear_win |
| homesite-insurance | best_combined | mfb_cf_fisd | AUROC | greedy-heads | 0.962849 ± 0.000437968 | 0.000354836 | clear_win |
| cooking-time | baseline_plr | baseline_plr | RMSE | mean | 0.480381 ± 0.000123385 | 0 | baseline |
| cooking-time | best_rla_only | rla_rank4_noise0.001 | RMSE | greedy-heads | 0.480146 ± 0.00016089 | 0.000234391 | clear_win |
| cooking-time | best_esam_only | esam_rho0.0025 | RMSE | greedy-heads | 0.480065 ± 8.02121e-05 | 0.000315778 | clear_win |
| cooking-time | best_mfb_only | mfb_keep0.8 | RMSE | greedy-heads | 0.479973 ± 0.000194552 | 0.000407806 | clear_win |
| cooking-time | best_cf_fisd_only | cf_fisd_lambda0.05 | RMSE | greedy-heads | 0.480337 ± 4.64953e-05 | 4.32134e-05 | weak_win |
| cooking-time | best_combined | rla_esam | RMSE | greedy-heads | 0.480026 ± 9.6554e-05 | 0.000354592 | clear_win |
| delivery-eta | baseline_plr | baseline_plr | RMSE | mean | 0.550226 ± 0.000582292 | 0 | baseline |
| delivery-eta | best_rla_only | rla_rank2_noise0.001 | RMSE | greedy-heads | 0.551407 ± 0.00109716 | -0.00118071 | loss |
| delivery-eta | best_esam_only | esam_only | RMSE | greedy-heads | 0.551148 ± 0.000504719 | -0.000922076 | loss |
| delivery-eta | best_mfb_only | mfb_keep0.7 | RMSE | greedy-heads | 0.551366 ± 0.00171387 | -0.00114032 | loss |
| delivery-eta | best_cf_fisd_only | cf_fisd_lambda0.05 | RMSE | greedy-heads | 0.552434 ± 0.00139117 | -0.00220824 | loss |
| delivery-eta | best_combined | rla_esam | RMSE | greedy-heads | 0.552336 ± 0.000198974 | -0.00211043 | loss |

## Validation-selected configs

| dataset | final variant | selected sweep variant | inference | validation metric |
|---|---|---|---|---:|
| sberbank-housing | baseline_plr | baseline_plr | mean |  |
| sberbank-housing | best_rla_only | rla_rank2_noise0.001 | greedy-heads | 0.2328613038082034 |
| sberbank-housing | best_esam_only | esam_rho0.0025 | greedy-heads | 0.23295802298295287 |
| sberbank-housing | best_mfb_only | mfb_keep0.8 | greedy-heads | 0.23237596587359466 |
| sberbank-housing | best_cf_fisd_only | cf_fisd_only | greedy-heads | 0.23327114512588593 |
| sberbank-housing | best_combined | mfb_cf_fisd | greedy-heads | 0.23320900639668657 |
| ecom-offers | baseline_plr | baseline_plr | mean |  |
| ecom-offers | best_rla_only | rla_rank2_noise0.0001 | best-head | 0.6396541901648656 |
| ecom-offers | best_esam_only | esam_rho0.001 | greedy-heads | 0.641845068859757 |
| ecom-offers | best_mfb_only | mfb_keep0.7 | best-head | 0.6547032702847417 |
| ecom-offers | best_cf_fisd_only | cf_fisd_lambda0.05 | greedy-heads | 0.6419149054445072 |
| ecom-offers | best_combined | mfb_cf_fisd | greedy-heads | 0.6506783465179156 |
| homesite-insurance | baseline_plr | baseline_plr | mean |  |
| homesite-insurance | best_rla_only | rla_rank4_noise1e-05 | greedy-heads | 0.9591329763748824 |
| homesite-insurance | best_esam_only | esam_only | greedy-heads | 0.958991133281215 |
| homesite-insurance | best_mfb_only | mfb_keep0.95 | greedy-heads | 0.9592957095194116 |
| homesite-insurance | best_cf_fisd_only | cf_fisd_only | greedy-heads | 0.9594844321379927 |
| homesite-insurance | best_combined | mfb_cf_fisd | greedy-heads | 0.9597570226838343 |
| cooking-time | baseline_plr | baseline_plr | mean |  |
| cooking-time | best_rla_only | rla_rank4_noise0.001 | greedy-heads | 0.4624107384246987 |
| cooking-time | best_esam_only | esam_rho0.0025 | greedy-heads | 0.46246732196836127 |
| cooking-time | best_mfb_only | mfb_keep0.8 | greedy-heads | 0.4624822561623469 |
| cooking-time | best_cf_fisd_only | cf_fisd_lambda0.05 | greedy-heads | 0.4628482268025946 |
| cooking-time | best_combined | rla_esam | greedy-heads | 0.4624234187671886 |
| delivery-eta | baseline_plr | baseline_plr | mean |  |
| delivery-eta | best_rla_only | rla_rank2_noise0.001 | greedy-heads | 0.5552720495549479 |
| delivery-eta | best_esam_only | esam_only | greedy-heads | 0.5551513290724005 |
| delivery-eta | best_mfb_only | mfb_keep0.7 | greedy-heads | 0.5551008108444078 |
| delivery-eta | best_cf_fisd_only | cf_fisd_lambda0.05 | greedy-heads | 0.5556691040453957 |
| delivery-eta | best_combined | rla_esam | greedy-heads | 0.5552988041877751 |

## Module wins vs baseline

- `best_rla_only`: sberbank-housing, ecom-offers, homesite-insurance, cooking-time
- `best_esam_only`: sberbank-housing, ecom-offers, homesite-insurance, cooking-time
- `best_mfb_only`: ecom-offers, cooking-time
- `best_cf_fisd_only`: sberbank-housing, ecom-offers, homesite-insurance, cooking-time
- `best_combined`: ecom-offers, homesite-insurance, cooking-time
