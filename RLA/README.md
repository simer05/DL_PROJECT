# RLA Submission Artifacts

This folder contains the RLA module code and audited result artifacts copied from the TabM/RLA working repository.

RLA is implemented through Rank-r Low-Rank BatchEnsemble adapters. The main implementation files are:

- `paper/lib/deep.py`
- `paper/bin/model.py`

The final audited claim is that RLA improves all 5 TabReD datasets under validation-selected matched inference mode. The five datasets are:

- `sberbank-housing`
- `ecom-offers`
- `homesite-insurance`
- `cooking-time`
- `delivery-eta`

Use `paper/exp/rla/_aggregated/rla_claim_table_final.csv` and `paper/exp/rla/_aggregated/rla_claim_audit_final.csv` as the source-backed audited tables. The preserved experiment directories contain only final-claim evidence files (`report.json`, `DONE`, and config TOMLs), not full checkpoints or raw datasets.

Do not claim that default mean inference wins all datasets unless the audited table says so. The winning inference mode is listed per dataset in the final audited table.

LEO/IA-TabM is included only as secondary/optional ablation evidence, not the RLA headline.
