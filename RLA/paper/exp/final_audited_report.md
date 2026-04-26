# Final audited results

RLA improves sberbank-housing, ecom-offers, homesite-insurance, cooking-time, and delivery-eta under validation-selected matched inference modes.
LEO improves ecom-offers and cooking-time under default mean inference.
All comparisons are matched by dataset, precision, k, data path, seed count, GPU, and inference mode.
Do not claim that RLA improves default mean inference on all datasets; the winning inference mode is listed per dataset.

## LEO / IA-TabM

| dataset | matched baseline | selected config | metric | test delta | percent delta | n_seeds | claim status |
|---|---|---|---|---:|---:|---:|---|
| sberbank-housing | baseline_plr:mean | leo_plr_lam005_start10:mean | rmse | 0.00168626673 | 0.7181% | 3 | loss |
| ecom-offers | baseline_plr:mean | leo_plr_lam010_start00:mean | roc-auc | 0.000433808269 | 0.0735% | 3 | win |
| cooking-time | baseline_plr:mean | leo_plr_lam002_start25:mean | rmse | 4.63360111e-06 | 0.0010% | 3 | win |
| delivery-eta | baseline_plr:mean | leo_plr_lam010_start00:mean | rmse | -0.000847715267 | -0.1543% | 3 | loss |

## RLA

| dataset | matched baseline | selected config | metric | test delta | percent delta | n_seeds | claim status |
|---|---|---|---|---:|---:|---:|---|
| sberbank-housing | baseline_plr_fp32:best-head | rla_plr_first_r4_basepreserve_fp32:best-head | rmse | 0.000577948789 | 0.2405% | 3 | win |
| ecom-offers | baseline_plr:best-head | rla_ecom_sweep_k32_first_r2_bf16_n1e4_lr150:best-head | roc-auc | 0.00453755897 | 0.7582% | 3 | win |
| homesite-insurance | baseline_plr_fp32:best-head | rla_plr_uniform_r8_basepreserve_fp32:best-head | roc-auc | 0.00124365872 | 0.1296% | 3 | win |
| cooking-time | baseline_plr_fp32:best-head | rla_plr_uniform_r8_basepreserve_fp32:best-head | rmse | 0.000395051122 | 0.0818% | 3 | win |
| delivery-eta | baseline_plr:greedy-heads | rla_plr_first_r2_basepreserve:greedy-heads | rmse | 0.000373672107 | 0.0678% | 3 | win |
