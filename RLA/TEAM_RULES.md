# Team Manifesto — Rules for Every Module on TabM/TabReD

These are the invariants. Any deviation breaks apples-to-apples comparison
with the paper or with each other and disqualifies cross-module composition.
Read every item before you push code.

## References

- TabM paper (ICLR 2025): [arXiv:2410.24210](https://arxiv.org/abs/2410.24210)
- TabM repo: [github.com/yandex-research/tabm](https://github.com/yandex-research/tabm)
- TabReD paper (ICLR 2025 Spotlight): [arXiv:2406.19380](https://arxiv.org/abs/2406.19380)
- TabReD repo: [github.com/yandex-research/tabred](https://github.com/yandex-research/tabred)
- TabReD datasheet: [tabred/datasheet.md](https://github.com/yandex-research/tabred/blob/main/datasheet.md)
- TabReD preprocessing: [tabred/preprocessing](https://github.com/yandex-research/tabred/tree/main/preprocessing)
- BatchEnsemble (the paper TabM extends): [arXiv:2002.06715](https://arxiv.org/abs/2002.06715)
- PiecewiseLinearEmbeddings (PLR): [rtdl-num-embeddings](https://yura52.github.io/rtdl-num-embeddings/stable/index.html)
- delu (used for seeding, timer, early-stopping): [yura52.github.io/delu](https://yura52.github.io/delu/stable/index.html)
- AdamW: [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- LoRA (closest prior art for adapter-rank discussions): [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Negative Correlation Learning: [arXiv:2011.02952](https://arxiv.org/abs/2011.02952)

---

## Repository, branch, version

1. Fork from the same Yandex Research TabM repo at a pinned commit (currently `28e47ae30` or whatever the team settles on). Pin this commit in every report you produce.

2. Each module lives on its own feature branch named `<module>-<short-name>` (for example `rla-rank-r`, `sls-density-gate`). Do not push module work to `main`.

3. Never modify files under `paper/` that are not your module's surface area. Specifically: do not touch `paper/lib/data.py`'s data loading code path (only the manifest/sha256 helper additions are allowed), do not touch `paper/bin/evaluate.py`'s seed loop, do not touch `paper/lib/metrics.py` scoring functions.

4. The new module files live in clearly namespaced locations: `paper/lib/<module>.py` for new layer/loss code, `paper/bin/model.py` only for config-flag wiring, `paper/exp/<module>/<dataset>/<variant>-evaluation/0.toml` for config templates. Do not scatter your code across the repo.

5. Tag the git commit you submitted to the cluster alongside every results dump. If the commit changes mid-experiment, all in-flight runs are stale and must be re-launched.

---

## Data — TabReD canonical, byte-for-byte

6. The canonical preparation comes from the official TabReD `preprocessing/` scripts at [github.com/yandex-research/tabred/tree/main/preprocessing](https://github.com/yandex-research/tabred/tree/main/preprocessing). Use the team's pinned preprocessed `.npy` artifacts (verified by sha256 in `manifest.json`) and the canonical `split-default/*_idx.npy` indices. Do not re-derive `.npy` files from raw Kaggle CSVs with a different pipeline; you will lose comparability with paper numbers and with teammate results.

7. Use only the `split-default/*_idx.npy` chronological train/val/test split. No random splits. No rolling-window cross-validation. No different time-window cuts. The paper trains on this split and so does TabM's reference code.

8. Do not invent a "raw Kaggle" pipeline. The TabReD preprocessing is the canonical preparation. Our `.npy` files are bit-identical (sha256) to the team's pinned preprocessed artifacts for all five datasets; if and when the upstream TabReD project publishes reference checksums, we will additionally verify against those.

9. Verify dataset integrity before every full sweep using the manifest's sha256. The check is: every file under `paper/data/<dataset>/` whose name appears in `manifest.json` must hash to the manifest's recorded sha256. A single hash mismatch invalidates every result on that dataset.

10. The numbers everyone must use for the *training* row counts are the canonical `split-default` totals from the team's pinned local snapshot under `paper/data/<dataset>/`. The current canonical totals are: sberbank-housing 28321, ecom-offers 160057, homesite-insurance 260753, cooking-time 319986, **delivery-eta 350516** (train 279415 + val 34174 + test 36927). Four of these (sberbank, ecom, homesite, cooking) match the TabReD datasheet's "Instances Used" column exactly. **Delivery-eta has known discrepancies that must be disclosed in any writeup**:
    - **vs TabReD datasheet (416,451)**: a 65,935-row gap (~16% smaller). Likely attributable to TabReD's preprocessing filter (`drop delivery-time < 1 minute`) and the 56-day train + 1-week val + ~5-week test time-window split boundaries.
    - **vs other team snapshots (e.g. 350,253 on the team cluster)**: a small ~263-row gap (~0.075%). Caused by Polars-version drift in TabReD's `data.sample(fraction=0.025, seed=0)` call — different polars versions produce different RNG sequences even with the same seed. Each snapshot is internally reproducible; the metric-level impact of a 0.075% shift is well below seed noise. Teammates working from a different snapshot should disclose the count their own snapshot produces.

    Do not silently substitute 416,451 (or any teammate's slightly different number) for our actual training-set size in our results tables. Cite the exact count of the snapshot used.

11. Use only the five datasets that are present on the cluster: sberbank-housing, ecom-offers, homesite-insurance, cooking-time, delivery-eta. If a teammate later adds homecredit-default, maps-routing, or weather, every other module's evaluation must be extended to those datasets too.

12. Do not subsample, filter, augment, or re-balance the training data. Use it as-is.

13. Persist a `manifest.json` per dataset directory with sha256 of every `.npy` file, source pointer (`tabred_preprocessed/<dataset> (split-default/*_idx.npy)`), and row counts. Anyone who wants to verify can rehash and compare in seconds.

---

## Runtime preprocessing — match paper exactly

14. **Inherit `num_policy` and `cat_policy` exactly from the paper's per-dataset template** at `paper/exp/tabm-piecewiselinear/tabred/<dataset>/0-evaluation/0.toml`. Do not hardcode either policy across all datasets. The paper's PLR-tuned settings vary:
    - `num_policy = "noisy-quantile"`: sberbank-housing, ecom-offers, homesite-insurance
    - `num_policy` omitted (uses no transform): cooking-time, delivery-eta
    - `cat_policy = "ordinal"`: sberbank-housing, homesite-insurance, cooking-time, delivery-eta
    - `cat_policy` omitted: ecom-offers (no categorical features)

15. The paper handles unknown-at-test-time categorical values with the all-zeros encoding inside `lib.deep.OneHotEncoding0d`; do not add custom unknown-handling on top.

16. Regression targets are standardised at runtime (`lib.data.standardize_labels`); do not pre-standardise inside the dataset files.

17. Binary features are converted to categorical inside `bin/model.py` (binary → cat with degree-2 cardinality). Do not re-encode them upstream.

18. Bin computation for `PiecewiseLinearEmbeddings` uses `compute_bins` with the dataset's training-set Y for tree-based bins. Use the paper's `n_bins` per dataset; do not change it.

---

## Hyperparameters — paper-tuned, no re-tune unless documented

19. Inherit hyperparameters from `paper/exp/tabm-piecewiselinear/tabred/<dataset>/0-evaluation/0.toml` for every module. The relevant fields are `batch_size`, `patience`, `gradient_clipping_norm`, `lr`, `weight_decay`, `n_blocks`, `d_block`, `dropout`, `d_embedding`, `n_bins`. These are paper-Optuna-tuned per dataset; do not silently change them.

20. Keep `arch_type = "tabm"` for any module that builds on TabM's BatchEnsemble. If your module operates inside `tabm-mini` or `tabm-packed`, label it explicitly and accept that those families have separate hyperparameters.

21. Keep `k = 32` ensemble members. Changing k changes the parameter count and breaks comparisons.

22. Keep `share_training_batches = false` (the TabM-PLR paper default). Switching to `true` is a separate, paper-known variant; do not silently switch.

23. Keep `gradient_clipping_norm = 1.0` unless your module has a documented stability reason. If you must change it, label the variant explicitly.

24. If you re-tune for your specific module, run the re-tune as a separate variant labelled `<module>_retuned` with all 100 Optuna trials documented. Do not replace the paper-tuned baseline silently.

25. Re-tuning must use validation metric only (val score) for trial selection. Never use test metric.

---

## Architecture — preserve paper baseline at off-state

26. Your module must reduce to the paper baseline exactly when its flag is off (rank=1, lambda=0, alpha=0, whatever your knob is). This is a hard requirement, not a best-effort.

27. Add a unit test that constructs the paper baseline `Model(...)` and your module-enabled `Model(..., your_flag=off_value)` with the same seed and verifies bit-identical forward output (max abs diff under 1e-6).

28. Add a full-model unit test, not just a layer-level test. A layer-level test cannot catch RNG-cascade bugs where extra random draws inside your custom layer shift the global RNG and silently change downstream layer initialisation. We hit this exact bug on the `base_preserving` init for RLA; do not repeat it.

29. Isolate any extra RNG draws inside your module using `torch.random.get_rng_state()` / `set_rng_state()` or a dedicated `torch.Generator` with a deterministic seed derived from the parameter shape. Any random draw not present in the paper baseline must not advance the global RNG state.

30. The shared backbone weight `W` of every linear layer must be initialised by the paper's `init_rsqrt_uniform_` from `paper/lib/deep.py`. Do not switch to Kaiming, Xavier, orthogonal, or any other scheme.

31. Bias vectors are initialised via `init_rsqrt_uniform_` against the input feature count, just like the paper. Do not zero-init biases unless your module specifically requires it (and even then, document).

32. Do not add or remove `nn.LayerNorm` or `nn.BatchNorm` layers. The paper's MLP block is `Linear → ReLU → Dropout`; keep that exact shape.

33. The output head is `lib.deep.NLinear(k, d_block, d_out)` for ensembled models. Do not replace it with a single `nn.Linear` unless your module specifically modifies the head.

---

## Training loop — paper-faithful

34. Use `delu.random.seed(config['seed'])` at the start of every run. Never use `torch.manual_seed` directly. The paper's seed function also seeds numpy and Python random.

35. Do not modify the optimizer choice (`AdamW`) or the parameter-grouping logic (`lib.deep.make_parameter_groups`).

36. Use the paper's `loss_fn` wrapper that broadcasts ensemble outputs against the labels; do not bypass it. If your module needs an extra penalty term, add it as `loss_total = paper_loss + lambda * your_penalty`, never replace.

37. The early-stopping signal is `metrics['val']['score']` with patience 16. Do not change this. Score is automatically signed (positive = better) by `dataset.calculate_metrics`.

38. Save `report.json` with the standard fields: `n_parameters`, `prediction_type`, `epoch_size`, `metrics` (per-part with `score` + raw metric), `time` (formatted timer string). Append your module-specific diagnostics as additional top-level keys, never overwrite paper fields.

39. Add a fail-fast `assert torch.isfinite(loss)` inside the training loop after the loss computation. NaN losses produce zero gradients and the run looks like a clean exit despite producing garbage. We hit this on RLA-PLR-r4 with BF16; catch it at source.

40. Add a `report['failure'] = {...}` block before raising the assertion error, so the failed run's `report.json` records why it died (which step, which `amp_dtype`).

---

## Precision policy — BF16 by default, FP32 fallback, never silently swap

41. AMP in this repo means BF16 (`torch.bfloat16`), not FP16. Verify with `paper/bin/model.py` line 649. Never describe AMP as FP16 in a writeup.

42. Persist `amp_enabled` (bool) and `amp_dtype` (`bfloat16` / `float16` / `fp32`) in every `report.json`. The aggregator must filter on `report.config.amp == toml.config.amp` parity; mismatched dirs are stale and must be wiped.

43. If your module crashes under BF16 (NaN, inf) at high rank or high lambda, fall back to `amp = false` (FP32) for both your variant AND its matching baseline. Reporting your-FP32 vs paper-baseline-BF16 conflates architecture and precision effects, which reviewers will catch.

44. Make `amp` an explicit per-variant override in your config-generator script, not a manual edit on the produced `.toml`. Manual edits silently revert when configs are regenerated.

45. Never enable `compile = true` (`torch.compile`) without a separate stability test. The paper's reference comment warns that `compile` is a younger feature and can hurt small models.

---

## Inference modes — three per variant, every time

46. Always report all three inference modes for every variant: mean ensemble (default), best-head selection, greedy-heads selection. The paper's `bin/evaluate.py` produces these three sub-evaluation directories automatically (`<variant>-best-head-evaluation`, `<variant>-greedy-heads-evaluation`); do not strip them from your aggregator.

47. Pick your headline number based on validation-set best inference mode, not test-set best. Picking variants on test score is data leakage and is the most common mistake reviewers will catch.

48. Do not invent new aggregation schemes silently. If you add validation-learned non-negative head weights, learnable softmax weights, distillation-based aggregation, or anything else, label it as a new variant column, do not replace mean-ensemble.

49. The aggregator must use the normalised `score` field (positive = better) for ranking variants and choosing val-best, but report the raw metric (RMSE for regression, AUROC for binclass) in the table cells.

---

## Seeds and statistical reporting

50. Three seeds is the minimum for any reported number. Five seeds for any headline claim. Fifteen seeds for paper-style headline rows where you want tight confidence. The paper itself uses 15 seeds for headline numbers.

51. Always report mean ± std across seeds, never a single seed's number. Single-seed values can disagree with the mean by more than the apparent module gain.

52. Use deterministic seeds 0..N-1, not random ones. Re-runs on the same seed must produce identical results.

53. Log per-seed val and test metrics in the long-format CSV (one row per seed × inference mode); do not throw away the per-seed numbers when computing the mean.

54. If your module's gain is within one standard deviation of the baseline, report it as "matches baseline within seed noise" and do not call it a win. The standard deviation is the noise reference, not eyeballed visual difference.

---

## NSCC cluster compliance — non-negotiable

References:
- NSCC user portal: [help.nscc.sg](https://help.nscc.sg/)
- PBS Pro reference: [altair.com/pbs-professional](https://altair.com/pbs-professional/)

55. Every PBS script must include `#PBS -P personal-abhipray` (or whatever your personal allocation is — coordinate with the team). Never use `aspire2a` or `43001002` (different projects' allocations).

56. Use `-q normal` only. The `gpu`, `gpu1`, and `gpu2` queues are disabled.

57. Never run `python`, `python3`, `pip`, `pytest`, `Rscript`, `julia`, or any inline `python -c '...'` on the login node. Even tiny tasks must go through PBS.

58. Never run `make`, `cmake`, `gcc`, `g++`, `nvcc` (compilation) on the login node. Submit a job.

59. Never run `wget` or `curl` for files larger than 10 MB on the login node. Never run `tar` or `unzip` on archives larger than 50 MB on the login node.

60. Never use `nohup`, `screen`, or `tmux` to keep compute alive on the login node.

61. Login-node allowed commands: `ls`, `mkdir`, `cat`, `head`, `tail`, `wc`, `grep`, `find`, `du`, `df`, `cp`, `mv`, `rm`, `chmod`, `chown`, `scp`, `rsync`, `qsub`, `qstat`, `qdel`, `module load/list/avail`, `echo`, `env`, `which`, `whoami`, `hostname`, `vi`, `nano`, `sed`. Anything else: PBS job.

62. Activation inside a PBS job is `module load python/3.11.5-gcc12 cuda/11.8.0 && source <your venv path>/bin/activate`. Each teammate uses their own home directory's venv.

---

## Code structure — same shape across modules

63. Configs go in `paper/exp/<module>/<dataset>/<variant>-evaluation/0.toml`. Do not put them anywhere else. The aggregator and submit scripts assume this layout.

64. Config templates are generated by `tools/generate_<module>_configs.py` from the paper's tuned templates. Do not hand-edit the produced `.toml` files (they get overwritten on regen).

65. Per-variant overrides (RLA: `rla_rank`, `rla_first_only`, `rla_init`, `rla_additive`; SLS: equivalent) are explicit kwargs to the generator, not silently injected via post-processing.

66. Any non-standard runtime behaviour (AMP off, larger gradient clipping, etc.) is also an explicit per-variant override in the generator. Manual `.toml` edits drift on regen.

67. Aggregator scripts read every `report.json` under `paper/exp/<module>/...` and produce three artifacts: long-format CSV (one row per seed × inference mode), wide summary CSV (mean ± std per dataset/variant/inference mode), markdown report. Anyone can re-run the aggregator and get the same numbers.

68. Aggregator must filter on config-report parity (`report.json.config.amp == 0.toml.amp`). Stale dirs (mismatch) are wiped, not averaged in.

69. Plot scripts read the wide summary CSV and produce per-figure PNG + PDF. The paper specifies five figures (test vs rank, time vs rank, member-correlation vs rank, first-vs-uniform, params vs rank). At minimum produce the matplotlib equivalents for your module's analogous diagnostics.

---

## Reporting and writeup

70. Every results table specifies: dataset, variant name, n_seeds, mean ± std for the primary metric, inference mode (mean / best-head / greedy-heads). No table is allowed to mix inference modes inside a single column.

71. Always present results against the matched-family baseline. PLR-RLA must compare against `baseline_plr` (PLR rank-1 = TabM-PLR), not against `baseline` (plain TabM). Mixing baselines inflates apparent gains.

72. Distinguish "matches baseline within seed noise" (legitimate negative result, paper-worthy) from "loses meaningfully" (must be flagged, paper-worthy as a limitation). Use seed std as the noise reference.

73. Document every deviation from the paper's setting in the Methods or Limitations section. Even one-line sentences ("AMP disabled for high-rank variants due to BF16 NaN") are mandatory.

74. Report parameter counts, mean train wall-clock, and seed variance alongside test metrics. A method that achieves equal mean with half the variance is a genuine win and must be claimed as such.

75. The team report's Methods section must include the math statement of each module, the paper's baseline equation, and the modification equation, in that order.

76. The Related Work section must explicitly distinguish your module from the closest prior work (LoRA for RLA, MaxUp for SLS, NCL for NCL itself, etc.). One paragraph each.

77. The Results section must include the rank-1-equals-baseline sanity check sentence ("we verify the implementation produces bit-identical forward outputs to the paper baseline at the off-state, max abs diff under 1e-6").

---

## Module composition (final integration)

78. The composition row (`TabM-PLR + SLS + NCL + IERC + RLA`) must inherit each module's tuned best-rank, best-init, best-AMP setting from the standalone sweep results. Do not re-tune at composition time.

79. Composition must be tested at three seeds first; headline composition numbers reported at five or more seeds.

80. If two modules' designs conflict (e.g., NCL changes the loss in a way that interacts with SLS's gradient assumption, or RLA's rank-r adapter changes the input-distribution that IERC's perturbation regularizer expects), report the conflict openly, do not silently disable one to make composition work.

81. The composition must reduce to baseline TabM-PLR exactly when all module flags are off. Verify with the same rank-1-equals-baseline test pattern.

---

## Reproducibility

82. Every run dir under `paper/exp/<module>/<dataset>/<variant>-evaluation/<seed>/` must contain `0.toml` (config), `report.json` (metrics + standard paper fields), and `DONE` marker. No `DONE` means the run did not complete and must not be aggregated.

83. **Forward-looking requirement** (not yet present in older `report.json` files): pin the git commit hash inside `report.json` via `report['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()`, and pin `report['amp_dtype']` per the precision-policy section. Older runs predating this rule are grandfathered in but must be re-run with the field if cited in a final report. New runs after this rule is adopted must include both fields.

84. Any teammate change to shared code (`paper/lib/deep.py`, `paper/bin/model.py`, `paper/lib/data.py`) requires a coordination message to the rest of the team. In-flight runs must be re-launched on the new commit.

85. Aggregator output (CSV + markdown + figures) lives under `paper/exp/<module>/_aggregated/` and is the single source of truth. Anyone reading the report can re-derive the numbers by running the aggregator on the run dirs.

86. The team's shared results directory (e.g., `~/tabmpp/team_results/`) is read-only on the cluster. To update, run the aggregator locally, push to the team dir, do not edit by hand.

---

## Data-integrity quick checks (do before every full sweep)

87. `find paper/data/<ds> -name "*.npy" | wc -l` returns 12 per dataset (sberbank/homesite/cooking/delivery) or 9 (ecom-offers, no X_cat). Anything else means files are missing or extra.

88. `python3 -c "import json,hashlib; m=json.load(open('paper/data/<ds>/manifest.json')); print(all(hashlib.sha256(open('paper/data/<ds>/'+f,'rb').read()).hexdigest()==h for f,h in m['sha256'].items()))"` returns `True`. False means data drift.

89. Row counts in `manifest.json` exactly match the canonical numbers (sberbank 28321, ecom 160057, homesite 260753, cooking 319986, delivery 350516). If they differ, you have the wrong split or a corrupted file.

---

## Audit trail (never skip)

90. Each report file's `config.amp` must equal the `0.toml`'s `amp` field. Run `tools/wipe_stale_amp_results.py --apply` before every aggregation to remove dirs where they disagree.

91. Each report file's `seed` must equal the seed-dir name. If `report.json[seed]` is 0 inside seed dir `1/`, the run is corrupt.

92. Each report file's `n_parameters` must match the paper's expected count for the variant (rank-1 equivalent of your module = baseline TabM count, exactly). Drift here means the architecture is wrong.

---

These rules ensure every module is a paper-faithful extension, every comparison is apples-to-apples, every result is reproducible, and the final composition row is honest. Violations are not "minor" — any one of them invalidates the comparison and the module's contribution claim.
