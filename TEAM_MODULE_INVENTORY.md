# Team Module Inventory

- `RLA/`: Rank-low-rank adapter experiments and audited final evidence. Integrated flag: `model.rla_enabled`.
- `ESAM/`: Efficient/adaptive SAM training-loop variant. Integrated flag: `model.esam_enabled`.
- `MFB/`: member/feature bagging notebook implementation. Integrated flag: `model.mfb_enabled`.
- `cf_fisd_recovered/`: CF-FISD feature-importance diagnostics and model hooks. Integrated flag: `model.cf_fisd_enabled`.

The integrated runner keeps the official TabM PLR baseline when all four flags are off and exposes module-only plus combined variants through `tabm_integrated/tools/generate_integrated_configs.py`.
