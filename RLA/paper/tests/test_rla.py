"""Unit tests for RLA (rank-r low-rank adapters) — Section 8.2 of the spec.

Covers:
    (a) forward-pass equality between rank-1 RLA and the baseline TabM
        LinearEfficientEnsemble on random inputs (exact-recovery test);
    (b) parameter count per layer matches Section 2.3 of the spec;
    (c) every RLA parameter receives a non-zero gradient after one backward
        pass;
    (d) rla_first_only restricts the rank-r class to the first layer.

The tests assume PYTHONPATH includes the paper/ directory so that
``import lib.deep`` and ``import bin.model`` resolve.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# Ensure paper/ is on sys.path when running from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
if _PAPER not in sys.path:
    sys.path.insert(0, _PAPER)

import lib.deep  # noqa: E402
from bin.model import Model  # noqa: E402


# --------------------------------------------------------------------------
# (a) Exact-recovery test for rank-1 RLA vs baseline
# --------------------------------------------------------------------------


def _make_baseline(seed: int = 0):
    torch.manual_seed(seed)
    layer = lib.deep.LinearEfficientEnsemble(
        in_features=8,
        out_features=12,
        bias=True,
        k=4,
        ensemble_scaling_in=True,
        ensemble_scaling_out=True,
        ensemble_bias=True,
        scaling_init='ones',
    )
    return layer


def _make_rankr(seed: int = 0, rank: int = 1):
    torch.manual_seed(seed)
    layer = lib.deep.LinearEfficientEnsembleRankR(
        in_features=8,
        out_features=12,
        bias=True,
        k=4,
        rank=rank,
        scaling_init='ones',
        additive=False,
    )
    return layer


def test_rank1_rla_layer_matches_baseline():
    base = _make_baseline(seed=0)
    rla = _make_rankr(seed=0, rank=1)

    # Both classes init weight then ones-fill scaling then bias from rsqrt-uniform.
    # Since the seed is identical and the RNG draws happen in the same order,
    # the weight and bias tensors must be element-wise identical.
    assert torch.equal(base.weight, rla.weight), 'shared backbone weight differs'
    assert torch.equal(base.bias, rla.bias), 'bias differs'

    x = torch.randn(5, 4, 8)  # (B, K, D_in)
    y_base = base(x)
    y_rla = rla(x)
    diff = (y_base - y_rla).abs().max().item()
    assert diff < 1e-6, f'rank-1 RLA forward differs from baseline by {diff}'


def test_rank1_rla_model_matches_baseline():
    """Full Model rank-1 RLA forward must match baseline TabM bit-identically."""
    common_kwargs = dict(
        n_num_features=6,
        cat_cardinalities=[],
        n_classes=None,
        backbone=dict(type='MLP', n_blocks=2, d_block=16, dropout=0.0),
        bins=None,
        num_embeddings=None,
        arch_type='tabm',
        k=4,
        share_training_batches=False,
    )

    torch.manual_seed(123)
    m_base = Model(**common_kwargs)
    torch.manual_seed(123)
    m_rla = Model(**common_kwargs, rla_rank=1, rla_first_only=False)

    m_base.eval()
    m_rla.eval()

    x_num = torch.randn(7, 6)
    with torch.no_grad():
        y_base = m_base(x_num=x_num)
        y_rla = m_rla(x_num=x_num)
    diff = (y_base - y_rla).abs().max().item()
    assert diff < 1e-5, f'rank-1 RLA model diverges from baseline by {diff}'


# --------------------------------------------------------------------------
# (b) Parameter count formula
# --------------------------------------------------------------------------


@pytest.mark.parametrize('rank', [1, 2, 4, 8])
def test_param_count_formula(rank: int):
    d_in, d_out, k = 16, 24, 8
    layer = lib.deep.LinearEfficientEnsembleRankR(
        in_features=d_in,
        out_features=d_out,
        bias=True,
        k=k,
        rank=rank,
        scaling_init='ones',
    )
    expected = (
        d_in * d_out  # shared backbone W
        + k * d_in * rank  # R adapter
        + k * d_out * rank  # S adapter
        + k * d_out  # per-member bias
    )
    actual = sum(p.numel() for p in layer.parameters())
    assert actual == expected, f'rank={rank}: expected {expected} params, got {actual}'


# --------------------------------------------------------------------------
# (c) All RLA parameters receive non-zero gradients
# --------------------------------------------------------------------------


@pytest.mark.parametrize('rank', [2, 4])
@pytest.mark.parametrize('additive', [False, True])
def test_all_params_receive_grad(rank: int, additive: bool):
    layer = lib.deep.LinearEfficientEnsembleRankR(
        in_features=8,
        out_features=12,
        bias=True,
        k=4,
        rank=rank,
        scaling_init='ones',
        additive=additive,
    )
    x = torch.randn(5, 4, 8, requires_grad=False)
    # Take TWO forward/backward steps with a tiny update between them.
    # In additive (LoRA-style) mode, R has zero grad at the very first step
    # because S is zero-initialised and annihilates the chain rule. After
    # one optimiser step S becomes non-zero, and from step 2 onward every
    # parameter receives a non-zero gradient.
    for step in range(2):
        if step > 0:
            with torch.no_grad():
                for p in layer.parameters():
                    if p.grad is not None:
                        p.add_(-1e-3 * p.grad)
                        p.grad.zero_()
        y = layer(x)
        y.sum().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f'{name} has no grad'
        assert p.grad.abs().max().item() > 0, f'{name} grad is exactly zero'


# --------------------------------------------------------------------------
# (d) rla_first_only wires only the first layer to the rank-r class
# --------------------------------------------------------------------------


def test_first_only_wiring():
    m = Model(
        n_num_features=6,
        cat_cardinalities=[],
        n_classes=None,
        backbone=dict(type='MLP', n_blocks=3, d_block=16, dropout=0.0),
        bins=None,
        num_embeddings=None,
        arch_type='tabm',
        k=4,
        share_training_batches=False,
        rla_rank=4,
        rla_first_only=True,
    )
    blocks = m.backbone.blocks  # type: ignore[attr-defined]
    # Block 0 first linear: rank-r class.
    assert isinstance(blocks[0][0], lib.deep.LinearEfficientEnsembleRankR)
    # Subsequent blocks' first linear: baseline class.
    assert isinstance(blocks[1][0], lib.deep.LinearEfficientEnsemble)
    assert isinstance(blocks[2][0], lib.deep.LinearEfficientEnsemble)


def test_uniform_wiring():
    m = Model(
        n_num_features=6,
        cat_cardinalities=[],
        n_classes=None,
        backbone=dict(type='MLP', n_blocks=3, d_block=16, dropout=0.0),
        bins=None,
        num_embeddings=None,
        arch_type='tabm',
        k=4,
        share_training_batches=False,
        rla_rank=2,
        rla_first_only=False,
    )
    blocks = m.backbone.blocks  # type: ignore[attr-defined]
    for i in range(3):
        assert isinstance(blocks[i][0], lib.deep.LinearEfficientEnsembleRankR)


@pytest.mark.parametrize('rank', [2, 4, 8])
def test_base_preserving_layer_matches_baseline_at_init(rank: int):
    """At init, base-preserving rank-r RLA must equal the rank-1 baseline forward."""
    base = _make_baseline(seed=42)

    torch.manual_seed(42)
    rla = lib.deep.LinearEfficientEnsembleRankR(
        in_features=8,
        out_features=12,
        bias=True,
        k=4,
        rank=rank,
        scaling_init='ones',
        additive=False,
        init_mode='base_preserving',
    )

    # Shared weight & bias must be element-identical because both classes
    # consume the same RNG sequence (weight rsqrt-uniform → ones-fill of
    # adapter scales → bias rsqrt-uniform).
    assert torch.equal(base.weight, rla.weight)
    assert torch.equal(base.bias, rla.bias)

    x = torch.randn(5, 4, 8)
    diff = (base(x) - rla(x)).abs().max().item()
    assert diff < 1e-6, f'base_preserving rank-{rank} differs from baseline by {diff}'


@pytest.mark.parametrize('rank', [2, 4])
def test_base_preserving_extra_paths_get_grad(rank: int):
    """Extra rank paths should pick up gradient within ~2 optimiser steps."""
    layer = lib.deep.LinearEfficientEnsembleRankR(
        in_features=8,
        out_features=12,
        bias=True,
        k=4,
        rank=rank,
        scaling_init='ones',
        additive=False,
        init_mode='base_preserving',
        base_preserve_noise=1e-3,
    )
    x = torch.randn(5, 4, 8)
    for _ in range(2):
        for p in layer.parameters():
            if p.grad is not None:
                p.grad.zero_()
        layer(x).sum().backward()
        with torch.no_grad():
            for p in layer.parameters():
                if p.grad is not None:
                    p.add_(-1e-3 * p.grad)

    # Path 0 must always have non-zero grad.
    assert layer.S.grad[:, :, 0].abs().max().item() > 0
    assert layer.R.grad[:, :, 0].abs().max().item() > 0
    # Extra paths (j>=1) must also have non-zero grad after step 2.
    for j in range(1, rank):
        assert layer.S.grad[:, :, j].abs().max().item() > 0, f'S path {j} grad zero'
        assert layer.R.grad[:, :, j].abs().max().item() > 0, f'R path {j} grad zero'


@pytest.mark.parametrize('rank', [2, 4, 8])
@pytest.mark.parametrize('first_only', [True, False])
def test_base_preserving_FULL_MODEL_matches_baseline_at_init(
    rank: int, first_only: bool
):
    """Full-model rank>1 base-preserving forward must equal baseline forward.

    Regression test for the RNG-isolation bug: extra-path noise inside the
    rank-r layer must NOT advance the global RNG, otherwise downstream
    layers (which still consume RNG for weight + bias init) will see a
    different RNG state than the baseline TabM construction would have,
    and the full model becomes silently divergent.
    """
    common_kwargs = dict(
        n_num_features=6,
        cat_cardinalities=[],
        n_classes=None,
        backbone=dict(type='MLP', n_blocks=3, d_block=32, dropout=0.0),
        bins=None,
        num_embeddings=None,
        arch_type='tabm',
        k=4,
        share_training_batches=False,
    )

    torch.manual_seed(7)
    m_base = Model(**common_kwargs)
    torch.manual_seed(7)
    m_rla = Model(
        **common_kwargs,
        rla_rank=rank,
        rla_first_only=first_only,
        rla_init='base_preserving',
    )
    m_base.eval()
    m_rla.eval()

    x_num = torch.randn(11, 6)
    with torch.no_grad():
        y_base = m_base(x_num=x_num)
        y_rla = m_rla(x_num=x_num)
    diff = (y_base - y_rla).abs().max().item()
    assert diff < 1e-5, (
        f'rank={rank} first_only={first_only}: '
        f'full-model diff {diff:.6g} (should be < 1e-5)'
    )


def test_additive_layer_starts_at_baseline():
    """Additive RLA at init equals plain shared-backbone forward (S=0)."""
    layer = lib.deep.LinearEfficientEnsembleRankR(
        in_features=8,
        out_features=12,
        bias=True,
        k=4,
        rank=4,
        scaling_init='ones',
        additive=True,
    )
    x = torch.randn(5, 4, 8)
    y = layer(x)
    expected = x @ layer.weight.T + layer.bias
    assert torch.allclose(y, expected, atol=1e-6)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
