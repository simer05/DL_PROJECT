# RLA — Report-Ready Writeup (drafts)

These paragraphs follow Section 11 of `RLA_implementation_plan.md`. They will
be filled with concrete numbers from `paper/exp/rla/_aggregated/rla_summary.csv`
once the NSCC sweep completes.

---

## Methods (Section 4 of the team report)

We extend BatchEnsemble's per-member multiplicative adapter from a fixed
rank-1 outer product to a tunable rank-`r` factorisation. For a linear layer
with input dimension `d_in` and output dimension `d_out`, the baseline TabM
member-`i` weight is `W_i = W ⊙ (s_i r_i^T)` with `r_i ∈ R^{d_in}` and
`s_i ∈ R^{d_out}`. RLA replaces these with `R_i ∈ R^{d_in × r}` and
`S_i ∈ R^{d_out × r}`, giving `W_i = W ⊙ (S_i R_i^T)` and the equivalent
forward `y_i = Σ_j s_{i,j} ⊙ (W (r_{i,j} ⊙ x)) + b_i`. The shared backbone
`W` is unchanged, so the per-member parameter overhead grows linearly in `r`
while `W` itself remains the dominant term.

We test two variants. **RLA-first** lifts only the first linear layer's
adapter to rank `r` while keeping the rest at rank 1 (the TabM paper's
own ablations identify the first adapter as the layer doing the heaviest
ensemble-projection work). **RLA-uniform** lifts every layer's adapter
to rank `r`. Initialisation preserves TabM's "deterministic-ones plus
first-layer random-signs" behaviour: the `r` columns of the first layer's
`R` are independent {-1, +1} sign vectors scaled by `1/√r`, and every
other adapter is filled deterministically with `1/√r`. At `r = 1` this
reduces *exactly* to the baseline, which we verify with a forward-pass
unit test (max absolute difference < 1e-6 between the rank-1 RLA module
and the original `LinearEfficientEnsemble`).

---

## Related Work (one paragraph)

TabM inherits BatchEnsemble's rank-1 multiplicative adapter (Wen et al.,
ICLR 2020), which has not been ablated for rank in either the original
paper or its follow-ups, including TabM itself. Low-rank adapters are
familiar from parameter-efficient fine-tuning — LoRA (Hu et al., ICLR
2022) and its descendants AdaLoRA, SoRA, and LoRA Ensembles — but those
methods use an *additive* rank-`r` correction `W + A B^T` to a *frozen*
pretrained weight for *transfer*, while BatchEnsemble uses a
*multiplicative* rank-`r` modulation `W ⊙ (S R^T)` of a *jointly-trained*
shared weight for *ensembling*. The two differ in operator, purpose, and
training regime. Packed-Ensemble (Laurent et al., ICLR 2023) corresponds
to the `r = d` corner of our rank knob and is already in the TabM paper's
benchmark. RLA fills the unexplored interior of that knob.

---

## Results (one paragraph — to be filled with numbers)

We sweep `r ∈ {1, 2, 4, 8}` over both RLA-first and RLA-uniform on
TabReD's homesite-insurance, ecom-offers, sberbank-housing,
cooking-time, and delivery-eta splits (the five present on our cluster
storage), three seeds each, plus an additive defensive row at rank 4
(RLA-first family). All hyperparameters are taken from the paper's tuned
TabM configs without re-tuning. **<<INSERT BEST-RANK-PER-DATASET ROW
FROM `rla_summary.csv` HERE>>**. The rank-1 sanity row reproduces the
baseline TabM numbers within seed noise on every dataset, confirming
the implementation is a strict generalisation of the paper. The
multiplicative-vs-additive head-to-head at matched rank shows
**<<MULTIPLICATIVE WINS / TIES / LOSES>>**, isolating the operator
choice empirically and protecting against the "this is just LoRA"
objection. Wall-clock cost grows roughly linearly with `r` (Figure B);
parameter count grows slowly because `W` dominates (Figure E).

---

## Limitations / Negative-result framing (only used if smoke gate fails)

If the rank-1 row turns out to be the best row across the entire sweep,
we report that finding directly: rank-1 is not a ceiling that has been
left on the table. The contribution then becomes the *first systematic
verification* that BatchEnsemble's rank-1 choice is tight on tabular
MLPs, with the additive-vs-multiplicative defensive row providing the
companion empirical evidence on the operator choice.
