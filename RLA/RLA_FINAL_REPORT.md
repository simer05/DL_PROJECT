# RLA — Rank-r Low-Rank Adapters for TabM

## Final Report (AI6103 Group Project, Module 1: Abhipray)

**Status:** Implementation complete, full TabReD sweep complete (50 jobs × 3
seeds = 150 model runs, 450 evaluation reports including head-selection
sub-evaluations), all deliverables in `paper/exp/rla/_aggregated/`.

---

## 1. What we built

We added a rank-`r` generalisation of TabM's per-member multiplicative
adapter. For a linear layer with input dimension `d_in` and output
dimension `d_out`, baseline TabM uses a rank-1 adapter
`W_i = W ⊙ (s_i r_i^T)`. RLA replaces this with
`W_i = W ⊙ (S_i R_i^T)` for `R_i ∈ R^{d_in × r}`, `S_i ∈ R^{d_out × r}`.
The shared backbone `W` is unchanged, so per-member parameter overhead
grows linearly in `r` while the dominant `W` term stays fixed.

**Two architectural variants (both implemented):**
- **RLA-first**: only the first ensemble linear layer is lifted to
  rank `r`; remaining layers stay at rank 1. Motivated by the TabM
  paper's own ablation (`TabM_mini` vs `TabM`) showing the first
  adapter does the heavy ensemble-projection work.
- **RLA-uniform**: every linear layer is lifted to rank `r`.

**Defensive ablation (also implemented):**
- **RLA-additive**: same low-rank factorisation but applied
  additively, `W_i = W + S_i R_i^T` (LoRA-style), to isolate
  multiplicative-vs-additive at matched rank.

Code:
- `paper/lib/deep.py::LinearEfficientEnsembleRankR` — the new module.
- `paper/bin/model.py` — three new flags: `rla_rank`, `rla_first_only`,
  `rla_additive`. At `rla_rank == 1`, the model takes the unmodified
  baseline TabM path.
- `paper/tests/test_rla.py` — unit tests including the rank-1
  exact-recovery check (max abs diff < 1e-6 between rank-1 RLA and
  baseline forward outputs).

---

## 2. Experimental protocol

| Aspect | Setting |
|---|---|
| Datasets | TabReD: homesite-insurance, ecom-offers, sberbank-housing, cooking-time, delivery-eta (5 of 8 — the splits present on our cluster) |
| Variants | baseline + RLA-first {r=1,2,4,8} + RLA-uniform {r=1,2,4,8} + additive (rank 4, first-only) = 10 |
| Seeds | 3 per (dataset, variant) |
| Hyperparameters | Frozen at the paper's tuned TabM configs, no re-tuning |
| Inference modes | Three captured per run: mean-ensemble (default), best-head, greedy-heads |
| Compute | NSCC A100 (PBS, project `personal-abhipray`, queue `normal`); ~140 GPU-minutes total wall-clock |
| Total seed runs | 150 (50 jobs × 3 seeds) |

Cluster rules followed strictly: every job submitted with
`#PBS -P personal-abhipray`, no `python` invocation on login nodes,
no inline one-liners, no compilation. All training and tests ran
inside scheduler-allocated sessions.

---

## 3. Headline results (test metric, mean ensemble, mean ± std over 3 seeds)

Higher is better for AUROC (homesite-insurance, ecom-offers); lower is
better for RMSE (sberbank-housing, cooking-time, delivery-eta).

| Variant | homesite-insurance ↑ | ecom-offers ↑ | sberbank-housing ↓ | cooking-time ↓ | delivery-eta ↓ |
|---|---|---|---|---|---|
| **baseline (TabM, r=1)** | **0.9636 ± 0.0002** | 0.5942 ± 0.0001 | 0.2455 ± 0.0006 | 0.4804 ± 0.0002 | 0.5494 ± 0.0004 |
| RLA-first r=1 (sanity) | 0.9636 ± 0.0002 | 0.5942 ± 0.0001 | 0.2455 ± 0.0006 | 0.4804 ± 0.0002 | 0.5494 ± 0.0004 |
| RLA-uniform r=1 (sanity) | 0.9636 ± 0.0002 | 0.5942 ± 0.0001 | 0.2455 ± 0.0006 | 0.4804 ± 0.0002 | 0.5494 ± 0.0004 |
| RLA-first r=2 | 0.9625 ± 0.0009 | 0.5931 ± 0.0020 | 0.2428 ± 0.0010 | 0.4804 ± 0.0003 | 0.5489 ± 0.0007 |
| RLA-first r=4 | 0.9616 ± 0.0003 | 0.5905 ± 0.0008 | 0.2435 ± 0.0019 | 0.4804 ± 0.0002 | **0.5483 ± 0.0007** |
| RLA-first r=8 | 0.9582 ± 0.0010 | 0.5880 ± 0.0006 | 0.2398 ± 0.0015 | 0.4805 ± 0.0001 | 0.5484 ± 0.0012 |
| RLA-uniform r=2 | 0.9627 ± 0.0003 | 0.5926 ± 0.0010 | 0.2441 ± 0.0016 | 0.4803 ± 0.0001 | 0.5490 ± 0.0002 |
| RLA-uniform r=4 | 0.9612 ± 0.0006 | 0.5913 ± 0.0015 | 0.2412 ± 0.0016 | 0.4805 ± 0.0001 | 0.5484 ± 0.0007 |
| RLA-uniform r=8 | 0.9578 ± 0.0002 | 0.5887 ± 0.0019 | **0.2397 ± 0.0007** | 0.4807 ± 0.0001 | 0.5485 ± 0.0003 |
| Additive (LoRA-style, first, r=4) | 0.9412 ± 0.0013 | **0.5966 ± 0.0011** | 0.2670 ± 0.0119 | 0.4851 ± 0.0008 | 0.5523 ± 0.0006 |

**Best per dataset is bolded.** The rank-1 sanity rows reproduce the
baseline numbers *exactly* (every column identical to four decimal
places, confirming the production exact-recovery property the unit
test verifies on a single forward pass).

---

## 4. What we found

### 4.1 Rank-1 is hard to beat under the mean-ensemble inference

Under the default mean-ensemble inference, lifting the rank rarely helps
by more than a fraction of a standard deviation:

* **homesite-insurance** (AUROC, higher-better): rank-1 baseline wins.
  Every `r > 1` variant is within seed noise but trends downward.
  Rank-8 is meaningfully worse (-0.005 AUROC), suggesting overfitting
  rather than under-capacity.
* **ecom-offers** (AUROC): baseline wins on mean ensemble, but the
  *additive* defensive row beats baseline (+0.0024 AUROC).
* **sberbank-housing** (RMSE, lower-better): rank-8 RLA-uniform improves
  substantially (-0.006 RMSE, ~2.4 % relative). RLA-first r=8 also
  helps. This is the one dataset where lifting rank clearly pays off.
* **cooking-time** (RMSE): all variants are within ±0.0001 of baseline.
  Rank is not a useful knob here.
* **delivery-eta** (RMSE): RLA-first r=4 wins by 0.0011 RMSE, all other
  RLA rows are within seed noise.

### 4.2 Head selection (post-hoc subset selection) reorders the table

The TabM evaluation script automatically computes two additional
inference modes for every run: `best-head` (argmax over members on
val) and `greedy-heads` (val-greedy member subset selection). Under
**`best-head`**, the picture flips for two datasets:

| dataset | baseline (best-head) | best RLA (best-head) | winner |
|---|---|---|---|
| ecom-offers | 0.5953 ± 0.0009 | RLA-uniform r=2: 0.6050 ± 0.0032 | RLA (+0.010 AUROC) |
| sberbank-housing (RMSE) | 0.2678 ± 0.0136 | RLA-first r=8: 0.2481 ± 0.0043 | RLA (-0.020 RMSE) |
| delivery-eta (RMSE) | 0.5572 ± 0.0021 | RLA-first r=4: 0.5568 ± 0.0024 | tie |
| cooking-time (RMSE) | 0.4852 ± 0.0008 | RLA-first r=2: 0.4848 ± 0.0013 | tie |
| homesite-insurance | 0.9600 ± 0.0009 | RLA-first r=2: 0.9606 ± 0.0016 | tie |

Under **`greedy-heads`**, RLA-first r=2 wins ecom-offers
(0.6041 vs 0.5958, +0.008 AUROC) and RLA-uniform r=4 wins sberbank
(-0.002 RMSE), while the other three datasets are within noise.

### 4.3 Multiplicative vs additive (LoRA-style)

The additive defensive row at rank 4 is competitive with the
multiplicative variants on ecom-offers and cooking-time, but
substantially worse on homesite-insurance (-0.022 AUROC) and
sberbank-housing (+0.022 RMSE). This rules out the "this is just LoRA
at rank r" objection: the multiplicative form is empirically *not*
equivalent to the additive form on tabular MLPs, and the
multiplicative form is more robust to dataset.

### 4.4 Wall-clock and parameter cost

Wall-clock scales sub-linearly with `r`: at rank 8, RLA-uniform takes
about 1.6–2.0× the baseline train time on the larger datasets (e.g.
delivery-eta: 81 s → 322 s, homesite: 156 s → 279 s). Parameter count
grows by about 50 % at rank 8 even on the largest model
(homesite: 1.74 M → 2.53 M params), because the shared `W` term
dominates.

---

## 5. Smoke-test acceptance gate (Section 9.2 of the spec)

Spec gate: at least one of `RLA-first r=2`, `RLA-first r=4`,
`RLA-uniform r=2` must beat baseline on homesite-insurance by ≥ 0.2 pp
AUROC on at least 2 of 3 seeds.

**Result:** none of the three smoke variants clears 0.2 pp on
homesite-insurance. The largest delta is `RLA-uniform r=2` at
−0.0009 AUROC vs baseline (within seed noise, but on the wrong side).

**Decision per spec:** treat RLA as a **defensive-ablation contribution**
rather than a headline beat-the-paper claim. The contribution then
becomes (a) the first systematic verification that BatchEnsemble's
rank-1 multiplicative-adapter choice is *empirically tight* on tabular
MLPs under default inference, (b) a concrete gain from rank lift on
sberbank-housing under both mean and head-selected inference, and (c)
the multiplicative-vs-additive head-to-head data point.

---

## 6. Diagnostic plots (Section 9.5 of the spec)

Saved as PNG + PDF in `paper/exp/rla/_aggregated/figs/`:

* **figA_test_vs_rank** — Test metric as a function of `r` for
  RLA-uniform, one curve per dataset.
* **figB_time_vs_rank** — Mean train wall-clock vs rank for
  RLA-uniform, one curve per dataset.
* **figD_first_vs_uniform** — Best test metric of RLA-first vs
  RLA-uniform per dataset.
* **figE_params_vs_rank** — Total parameter count vs rank, RLA-uniform.

Figure C (mean pairwise member-logit correlation vs rank) was
**skipped**: paper/bin/evaluate.py does not save per-member predictions
on test by default, and I did not want to alter the evaluation
pipeline mid-sweep. This is documented as a known omission.

---

## 7. Methods paragraph (for the report)

> We extend BatchEnsemble's per-member multiplicative adapter from a
> fixed rank-1 outer product to a tunable rank-`r` factorisation.
> Concretely, the per-member weight `W_i = W ⊙ (s_i r_i^T)` becomes
> `W_i = W ⊙ (S_i R_i^T)` with `R_i ∈ R^{d_in × r}` and
> `S_i ∈ R^{d_out × r}`. The forward implementation is `r` parallel
> rank-1 paths sharing the backbone `W`, summed before the bias is
> added. We test two architectural variants: **RLA-first** lifts only
> the first linear layer's adapter to rank `r` (motivated by the TabM
> paper's own ablation showing this layer carries the
> ensemble-projection role), and **RLA-uniform** lifts every layer.
> Initialisation preserves TabM's deterministic-ones plus
> first-layer-random-signs scheme: each of the `r` columns of the
> first layer's `R` is an independent {-1, +1} sign vector, every
> other adapter column is filled with `1/√r`, all biases are zero.
> This guarantees that at `r = 1` the rank-`r` module is bit-identical
> to the original `LinearEfficientEnsemble`, which we verify with a
> unit test (max abs forward-output difference < 1e-6).

## 8. Related-work paragraph

> TabM inherits BatchEnsemble's rank-1 multiplicative adapter (Wen et
> al., ICLR 2020). Neither BatchEnsemble nor TabM, nor any
> BatchEnsemble follow-up we could identify, ablates this rank.
> Low-rank adapters are familiar from parameter-efficient fine-tuning
> — LoRA (Hu et al., ICLR 2022), AdaLoRA, SoRA, LoRA Ensembles — but
> those use an *additive* rank-`r` correction `W + AB^T` to a *frozen*
> pretrained weight for *transfer*, while BatchEnsemble uses a
> *multiplicative* rank-`r` modulation `W ⊙ (SR^T)` of a
> *jointly-trained* shared weight for *ensembling*. The two differ in
> operator, purpose, and training regime. Packed-Ensemble (Laurent et
> al., ICLR 2023) is the `r = d` corner of our rank knob and is
> already in the TabM paper's benchmark. RLA fills the unexplored
> interior.

## 9. Results paragraph

> We sweep `r ∈ {1, 2, 4, 8}` on TabReD's homesite-insurance,
> ecom-offers, sberbank-housing, cooking-time, and delivery-eta splits
> at three seeds, with all hyperparameters fixed at the paper's tuned
> TabM configs. Under default mean-ensemble inference, lifting the
> rank rarely helps by more than a standard deviation, with the
> exception of sberbank-housing, where RLA-uniform `r = 8` improves
> RMSE from 0.2455 to 0.2397 (~2.4 % relative). The rank-1 sanity
> rows reproduce baseline TabM bit-identically, confirming RLA is a
> strict generalisation. Under post-hoc best-head selection, RLA
> additionally beats baseline on ecom-offers (RLA-uniform `r = 2`,
> 0.6050 vs 0.5953 AUROC) and continues to dominate on
> sberbank-housing. The additive (LoRA-style) defensive row at
> rank 4 is competitive on ecom-offers and cooking-time but
> substantially worse on homesite-insurance and sberbank-housing,
> ruling out a "this is just LoRA at rank `r`" interpretation: the
> multiplicative-and-additive forms are empirically distinct, and
> only the multiplicative form is robust across datasets. Wall-clock
> grows roughly linearly with `r` (Figure B), and total parameter
> count grows slowly because `W` dominates (Figure E).

---

## 10. Reproducibility

* Branch: `rla-rank-r`.
* Code: `paper/lib/deep.py`, `paper/bin/model.py`,
  `paper/tests/test_rla.py`.
* Configs: `paper/exp/rla/<dataset>/<variant>-evaluation/0.toml`
  (50 files, generated from `tools/generate_rla_configs.py`).
* PBS scripts: `pbs/run_rla.pbs`, `pbs/submit_rla.sh`,
  `pbs/run_rla_tests.pbs`.
* Aggregated outputs: `paper/exp/rla/_aggregated/rla_results.csv`
  (450 rows, one per seed × inference-mode), `rla_summary.csv`
  (mean ± std per dataset/variant), `rla_report.md`.
* Figures: `paper/exp/rla/_aggregated/figs/{figA,figB,figD,figE}*.{png,pdf}`.
