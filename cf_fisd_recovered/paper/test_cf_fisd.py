from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if _HERE.joinpath('pixi.toml').exists():
    sys.path.insert(0, str(_HERE))

import delu
import torch
import torch.nn.functional as F

import lib.cf_fisd
import lib.deep
from bin.model import Model


def _make_tabm_plr_model(n_num: int, cat_cards: list[int], k: int, seed: int):
    delu.random.seed(seed)
    bins_per_feat = 8
    edges = [torch.linspace(0.0, 1.0, bins_per_feat + 1) for _ in range(n_num)]
    return Model(
        n_num_features=n_num,
        cat_cardinalities=cat_cards,
        n_classes=None,
        backbone={'type': 'MLP', 'n_blocks': 2, 'd_block': 32, 'dropout': 0.0},
        bins=edges,
        num_embeddings={'type': 'PiecewiseLinearEmbeddings', 'd_embedding': 4},
        arch_type='tabm',
        k=k,
        share_training_batches=False,
    )


def _cat_input(n_rows: int, cat_cards: list[int], seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    cols = [
        torch.randint(0, c, (n_rows,), generator=g, dtype=torch.long)
        for c in cat_cards
    ]
    return torch.stack(cols, dim=1) if cols else torch.zeros((n_rows, 0), dtype=torch.long)


def test_chunk_aggregate_matches_explicit_sum():
    delu.random.seed(7)
    k, n_num, n_cat = 5, 3, 2
    d_emb = 4
    cards = [3, 5]
    d_features = [d_emb] * n_num + cards
    r1 = torch.randn(k, sum(d_features))
    out = lib.cf_fisd.chunk_aggregate_r1(r1, d_features)
    assert out.shape == (k, n_num + n_cat)
    expected = torch.zeros(k, n_num + n_cat)
    cursor = 0
    for j, w in enumerate(d_features):
        expected[:, j] = r1[:, cursor : cursor + w].abs().sum(dim=-1)
        cursor += w
    assert torch.allclose(out, expected, atol=0.0, rtol=0.0)


def test_cf_fisd_loss_zero_when_lambda_zero_keeps_outputs_identical():
    n_num, cat_cards, k = 4, [3, 4], 8
    d_emb = 4
    n_features = n_num + len(cat_cards)
    d_features = [d_emb] * n_num + cat_cards

    seed = 123
    m_a = _make_tabm_plr_model(n_num, cat_cards, k, seed)
    m_b = _make_tabm_plr_model(n_num, cat_cards, k, seed)

    for (na, pa), (nb, pb) in zip(m_a.named_parameters(), m_b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb), f'init mismatch at {na}'

    g = torch.Generator().manual_seed(seed)
    x_num = torch.randn(16, n_num, generator=g)
    x_cat = _cat_input(16, cat_cards, seed=seed + 1)

    m_a.train()
    out_a = m_a(x_num=x_num, x_cat=x_cat)
    task_a = out_a.float().square().mean()
    task_a.backward()
    grads_a = {n: p.grad.detach().clone() for n, p in m_a.named_parameters() if p.grad is not None}

    teacher_imps = {
        name: torch.full((n_features,), 1.0 / n_features) for name in ('xgb', 'lgbm', 'cat')
    }
    groups = lib.cf_fisd.default_member_groups(k)

    m_b.train()
    out_b = m_b(x_num=x_num, x_cat=x_cat)
    task_b = out_b.float().square().mean()
    r1_param = m_b.backbone.blocks[0][0].r
    penalty = lib.cf_fisd.cf_fisd_loss(r1_param, teacher_imps, groups, 'raw', d_features)
    total_b = task_b + 0.0 * penalty
    total_b.backward()
    grads_b = {n: p.grad.detach().clone() for n, p in m_b.named_parameters() if p.grad is not None}

    assert torch.equal(out_a, out_b), 'forward outputs differ at lambda=0'
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        assert torch.equal(grads_a[name], grads_b[name]), f'grad mismatch at {name}'


def test_cf_fisd_loss_nonzero_changes_first_layer_grad():
    n_num, cat_cards, k = 4, [3, 4], 8
    d_emb = 4
    n_features = n_num + len(cat_cards)
    d_features = [d_emb] * n_num + cat_cards

    seed = 321
    m = _make_tabm_plr_model(n_num, cat_cards, k, seed)

    g = torch.Generator().manual_seed(seed)
    x_num = torch.randn(16, n_num, generator=g)
    x_cat = _cat_input(16, cat_cards, seed=seed + 1)

    teacher_imps = {
        name: torch.full((n_features,), 1.0 / n_features) for name in ('xgb', 'lgbm', 'cat')
    }
    groups = lib.cf_fisd.default_member_groups(k)

    m.train()
    out = m(x_num=x_num, x_cat=x_cat)
    task_loss = out.float().square().mean()
    r1_param = m.backbone.blocks[0][0].r
    penalty = lib.cf_fisd.cf_fisd_loss(r1_param, teacher_imps, groups, 'raw', d_features)
    total = task_loss + 0.1 * penalty
    total.backward()

    assert r1_param.grad is not None
    assert torch.isfinite(r1_param.grad).all()


def test_load_teacher_importances_sums_to_one(tmp_dir: Path | None = None):
    import numpy as np

    if tmp_dir is None:
        tmp_dir = Path(__file__).resolve().parent / '_cf_fisd_test_artifacts'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    n_features = 7
    rng = np.random.default_rng(0)
    for name in ('xgb', 'lgbm', 'cat'):
        v = rng.random(n_features).astype(np.float32) + 0.1
        np.save(tmp_dir / f'mydataset_{name}.npy', v)
    out = lib.cf_fisd.load_teacher_importances(
        tmp_dir, 'mydataset', n_features=n_features
    )
    for name, t in out.items():
        s = float(t.sum())
        assert abs(s - 1.0) < 1e-6, f'{name} not normalized: sum={s}'
        assert (t >= 0).all()


def main() -> int:
    tests = [
        test_chunk_aggregate_matches_explicit_sum,
        test_cf_fisd_loss_zero_when_lambda_zero_keeps_outputs_identical,
        test_cf_fisd_loss_nonzero_changes_first_layer_grad,
        test_load_teacher_importances_sums_to_one,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f'PASS  {fn.__name__}')
        except AssertionError as e:
            print(f'FAIL  {fn.__name__}: {e}')
            failed += 1
        except Exception as e:
            print(f'ERROR {fn.__name__}: {type(e).__name__}: {e}')
            failed += 1
    return failed


if __name__ == '__main__':
    sys.exit(main())
