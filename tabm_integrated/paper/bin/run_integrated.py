import math
import json
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Literal

import delu
import numpy as np
import rtdl_num_embeddings
import scipy
import torch
import torch.nn as nn
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from typing_extensions import NotRequired, TypedDict

if __name__ == '__main__':
    _cwd = Path.cwd()
    assert _cwd.joinpath(
        'pixi.toml'
    ).exists(), 'The script must be run from the `paper/` directory'
    sys.path.append(str(_cwd))
    del _cwd

import lib
import lib.data
import lib.deep
import lib.env
import lib.cf_fisd
from lib import KWArgs, PartKey


def _get_first_ensemble_layer(
    backbone: lib.deep.MLP,
) -> lib.deep.LinearEfficientEnsemble:
    if isinstance(backbone, lib.deep.MLP):
        return backbone.blocks[0][0]  # type: ignore[code]
    else:
        raise RuntimeError(f'Unsupported backbone: {backbone}')


@torch.inference_mode()
def _init_first_adapter(
    weight: Tensor,
    distribution: Literal['normal', 'random-signs'],
    init_sections: list[int],
) -> None:
    """Initialize the first adapter.

    NOTE
    The `init_sections` argument is a historical artifact that accidentally leaked
    from irrelevant experiments to the final models. Perhaps, the code related
    to `init_sections` can be simply removed, but this was not tested.
    """
    assert weight.ndim == 2
    assert weight.shape[1] == sum(init_sections)

    if distribution == 'normal':
        init_fn_ = nn.init.normal_
    elif distribution == 'random-signs':
        init_fn_ = lib.deep.init_random_signs_
    else:
        raise ValueError(f'Unknown distribution: {distribution}')

    section_bounds = [0, *torch.tensor(init_sections).cumsum(0).tolist()]
    for i in range(len(init_sections)):
        # NOTE
        # As noted above, this section-based initialization is an arbitrary historical
        # artifact. Consider the first adapter of one ensemble member.
        # This adapter vector is implicitly split into "sections",
        # where one section corresponds to one feature. The code below ensures that
        # the adapter weights in one section are initialized with the same random value
        # from the given distribution.
        w = torch.empty((len(weight), 1), dtype=weight.dtype, device=weight.device)
        init_fn_(w)
        weight[:, section_bounds[i] : section_bounds[i + 1]] = w


def _get_first_rankr_layer(
    backbone: lib.deep.MLP,
) -> lib.deep.LinearEfficientEnsembleRankR:
    if isinstance(backbone, lib.deep.MLP):
        return backbone.blocks[0][0]  # type: ignore[code]
    raise RuntimeError(f'Unsupported backbone: {backbone}')


@torch.inference_mode()
def _init_first_adapter_rankr(
    R: Tensor,
    distribution: Literal['normal', 'random-signs'],
    init_sections: list[int],
) -> None:
    """Initialise the first-layer R parameter (k, d_in, rank).

    Each rank column receives an independent section-wise random sign
    (or normal) draw, then the entire R tensor is scaled by 1/sqrt(rank)
    so the sum over the rank axis preserves the variance of the rank-1
    baseline at initialisation.
    """
    assert R.ndim == 3
    k, d_in, rank = R.shape
    assert d_in == sum(init_sections)

    if distribution == 'normal':
        init_fn_ = nn.init.normal_
    elif distribution == 'random-signs':
        init_fn_ = lib.deep.init_random_signs_
    else:
        raise ValueError(f'Unknown distribution: {distribution}')

    section_bounds = [0, *torch.tensor(init_sections).cumsum(0).tolist()]
    for j in range(rank):
        for i in range(len(init_sections)):
            w = torch.empty((k, 1), dtype=R.dtype, device=R.device)
            init_fn_(w)
            R[:, section_bounds[i] : section_bounds[i + 1], j] = w
    R.mul_(rank**-0.5)


def _replace_first_with_rankr(
    backbone: lib.deep.MLP,
    *,
    k: int,
    rank: int,
    additive: bool,
    init_mode: str = 'variance_preserving',
    base_preserve_noise: float = 1e-3,
) -> None:
    """Replace the first ensemble linear layer with the rank-r variant.

    The remaining layers stay as the baseline rank-1 class. This implements
    the RLA-first variant from Section 3.1 of the spec.

    NOTE on RNG isolation: ``LinearEfficientEnsembleRankR.__init__`` calls
    ``reset_parameters`` which consumes global RNG for its (about-to-be
    overwritten) weight and bias initialisation. Without isolation, that
    extra draw drifts the RNG state and ``_init_first_adapter_basepreserve``
    below would then sample a different sequence of random signs than the
    baseline construction did, breaking full-model bit-equivalence at
    init. Save+restore the global RNG state around the construction.
    """
    first = _get_first_ensemble_layer(backbone)
    rng_state = torch.random.get_rng_state()
    try:
        new = lib.deep.LinearEfficientEnsembleRankR(
            in_features=first.in_features,
            out_features=first.out_features,
            bias=first.bias is not None,
            k=k,
            rank=rank,
            scaling_init='ones',
            additive=additive,
            init_mode=init_mode,
            base_preserve_noise=base_preserve_noise,
        )
    finally:
        torch.random.set_rng_state(rng_state)
    # Re-use the same backbone weights for stability of the comparison.
    with torch.inference_mode():
        new.weight.copy_(first.weight)
        if first.bias is not None and new.bias is not None:
            new.bias.copy_(first.bias)
    # Splice into the backbone (paper/lib MLP block 0 sequential[0]).
    backbone.blocks[0][0] = new  # type: ignore[code]


@torch.inference_mode()
def _init_first_adapter_basepreserve(
    R: Tensor,
    distribution: Literal['normal', 'random-signs'],
    init_sections: list[int],
) -> None:
    """Re-initialise *only path 0* of a rank-r R parameter using the
    section-based scheme. Extra paths (j >= 1) are left untouched,
    preserving the base-preserving zero/tiny-noise init.
    """
    assert R.ndim == 3
    k, d_in, rank = R.shape
    assert d_in == sum(init_sections)

    if distribution == 'normal':
        init_fn_ = nn.init.normal_
    elif distribution == 'random-signs':
        init_fn_ = lib.deep.init_random_signs_
    else:
        raise ValueError(f'Unknown distribution: {distribution}')

    section_bounds = [0, *torch.tensor(init_sections).cumsum(0).tolist()]
    for i in range(len(init_sections)):
        w = torch.empty((k, 1), dtype=R.dtype, device=R.device)
        init_fn_(w)
        R[:, section_bounds[i] : section_bounds[i + 1], 0] = w
    # Note: do NOT scale by 1/sqrt(r) here — only path 0 contributes at init,
    # so it must carry the full baseline magnitude.




def _mean_pairwise_jaccard(mask: np.ndarray) -> float:
    if mask.shape[0] < 2:
        return 1.0
    values: list[float] = []
    for i in range(mask.shape[0]):
        for j in range(i + 1, mask.shape[0]):
            union = np.logical_or(mask[i], mask[j]).sum()
            values.append(1.0 if union == 0 else float(np.logical_and(mask[i], mask[j]).sum() / union))
    return float(np.mean(values)) if values else 1.0


def _mfb_mask_coverage_stats(mask: np.ndarray) -> dict[str, float]:
    features_per_member = mask.sum(axis=1)
    members_per_feature = mask.sum(axis=0)
    return {
        'keep_rate_actual': float(mask.mean()),
        'min_features_per_member': float(features_per_member.min()),
        'mean_features_per_member': float(features_per_member.mean()),
        'max_features_per_member': float(features_per_member.max()),
        'min_members_per_feature': float(members_per_feature.min()),
        'mean_members_per_feature': float(members_per_feature.mean()),
        'max_members_per_feature': float(members_per_feature.max()),
        'mean_pairwise_jaccard': _mean_pairwise_jaccard(mask),
    }


def _make_mfb_feature_group_mask(
    *,
    k: int,
    feature_widths: list[int],
    keep_rate: float,
    seed: int,
    anchor_fraction: float = 0.0,
    protected_feature_ids: None | list[int] = None,
    ensure_each_feature_seen: bool = True,
    ensure_each_member_nonempty: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    feature_index: list[int] = []
    for feature_id, width in enumerate(feature_widths):
        feature_index.extend([feature_id] * int(width))
    n_features = len(feature_widths)
    gen = np.random.RandomState(seed)
    feature_mask = (gen.rand(k, n_features) < keep_rate).astype(np.float32)
    protected = np.array([] if protected_feature_ids is None else protected_feature_ids, dtype=np.int64)
    n_anchor = max(0, min(k, int(round(k * anchor_fraction))))
    if protected.size > 0:
        feature_mask[:, protected] = 1.0
    if n_anchor > 0:
        feature_mask[:n_anchor, :] = 1.0
    if ensure_each_member_nonempty:
        for row in range(k):
            if feature_mask[row].sum() == 0:
                feature_mask[row, gen.randint(0, n_features)] = 1.0
    if ensure_each_feature_seen:
        for col in range(n_features):
            if feature_mask[:, col].sum() == 0:
                feature_mask[gen.randint(0, k), col] = 1.0
    dim_mask = feature_mask[:, np.asarray(feature_index, dtype=np.int64)]
    stats = _mfb_mask_coverage_stats(feature_mask)
    stats.update({
        'anchor_fraction_configured': float(anchor_fraction),
        'n_anchor_members': float(n_anchor),
        'core_fraction_configured': float(len(protected) / max(1, n_features)),
        'n_core_features': float(len(protected)),
    })
    return feature_mask.astype(np.float32), dim_mask.astype(np.float32), stats


def _get_first_adapter_for_cf_fisd(backbone: lib.deep.MLP) -> Tensor:
    first = backbone.blocks[0][0]  # type: ignore[code]
    if isinstance(first, lib.deep.LinearEfficientEnsembleRankR):
        return first.R[:, :, 0]
    if isinstance(first, lib.deep.LinearEfficientEnsemble):
        if first.r is None:
            raise RuntimeError('CF-FISD requires first-layer input scaling.')
        return first.r
    raise RuntimeError(f'Unsupported first ensemble layer for CF-FISD: {type(first)!r}')

DEFAULT_SHARE_TRAINING_BATCHES = True


class Model(nn.Module):
    """MLP & TabM."""

    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: None | int,
        backbone: dict,
        bins: None | list[Tensor],  # For piecewise-linear encoding/embeddings.
        num_embeddings: None | dict = None,
        arch_type: Literal[
            # Plain feed-forward network without any kind of ensembling.
            'plain',
            #
            # TabM
            'tabm',
            #
            # TabM-mini
            'tabm-mini',
            #
            # TabM-packed
            'tabm-packed',
            #
            # TabM. The first adapter is initialized from the normal distribution.
            # This variant was not used in the paper, but it may be useful in practice.
            'tabm-normal',
            #
            # TabM-mini. The adapter is initialized from the normal distribution.
            # This variant was not used in the paper.
            'tabm-mini-normal',
        ],
        k: None | int = None,
        share_training_batches: bool = DEFAULT_SHARE_TRAINING_BATCHES,
        rla_rank: int = 1,
        rla_first_only: bool = False,
        rla_additive: bool = False,
        rla_init: Literal['variance_preserving', 'base_preserving'] = 'variance_preserving',
        rla_base_preserve_noise: float = 1e-3,
        mfb: None | dict[str, Any] = None,
    ) -> None:
        # >>> Validate arguments.
        assert n_num_features >= 0
        assert n_num_features or cat_cardinalities
        if arch_type == 'plain':
            assert k is None
            assert (
                share_training_batches
            ), 'If `arch_type` is set to "plain", then `simple` must remain True'
        else:
            assert k is not None
            assert k > 0

        super().__init__()

        # >>> Continuous (numerical) features
        first_adapter_sections = []  # See the comment in `_init_first_adapter`.

        if n_num_features == 0:
            assert bins is None
            self.num_module = None
            d_num = 0

        elif num_embeddings is None:
            assert bins is None
            self.num_module = None
            d_num = n_num_features
            first_adapter_sections.extend(1 for _ in range(n_num_features))

        else:
            if bins is None:
                self.num_module = lib.deep.make_module(
                    **num_embeddings, n_features=n_num_features
                )
            else:
                assert num_embeddings['type'].startswith('PiecewiseLinearEmbeddings')
                self.num_module = lib.deep.make_module(**num_embeddings, bins=bins)
            d_num = n_num_features * num_embeddings['d_embedding']
            first_adapter_sections.extend(
                num_embeddings['d_embedding'] for _ in range(n_num_features)
            )

        # >>> Categorical features
        self.cat_module = (
            lib.deep.OneHotEncoding0d(cat_cardinalities) if cat_cardinalities else None
        )
        first_adapter_sections.extend(cat_cardinalities)
        d_cat = sum(cat_cardinalities)

        # >>> Backbone
        d_flat = d_num + d_cat
        self.minimal_ensemble_adapter = None
        backbone = dict(backbone)
        backbone_type = backbone.pop('type', 'MLP')
        self.backbone = lib.deep.make_module(backbone_type, d_in=d_flat, **backbone)

        if arch_type != 'plain':
            assert k is not None
            first_adapter_init = (
                None
                if arch_type == 'tabm-packed'
                else 'normal'
                if arch_type in ('tabm-mini-normal', 'tabm-normal')
                # For other arch_types, the initialization depends
                # on the presense of num_embeddings.
                else 'random-signs'
                if num_embeddings is None
                else 'normal'
            )

            if arch_type in ('tabm', 'tabm-normal'):
                # Like BatchEnsemble, but all multiplicative adapters,
                # except for the very first one, are initialized with ones.
                assert first_adapter_init is not None

                use_rla = rla_rank > 1 or rla_additive
                if not use_rla:
                    # Baseline path: rank-1 multiplicative adapter, identical
                    # to upstream TabM. This branch is kept exact so the rank-1
                    # RLA exact-recovery test compares against the original.
                    lib.deep.make_efficient_ensemble(
                        self.backbone,
                        lib.deep.LinearEfficientEnsemble,
                        k=k,
                        ensemble_scaling_in=True,
                        ensemble_scaling_out=True,
                        ensemble_bias=True,
                        scaling_init='ones',
                    )
                    _init_first_adapter(
                        _get_first_ensemble_layer(self.backbone).r,  # type: ignore[code]
                        first_adapter_init,
                        first_adapter_sections,
                    )
                else:
                    # RLA path: rank-r (or additive) adapters.
                    if rla_first_only:
                        # All layers rank-1 baseline; replace only the first
                        # with the rank-r class.
                        lib.deep.make_efficient_ensemble(
                            self.backbone,
                            lib.deep.LinearEfficientEnsemble,
                            k=k,
                            ensemble_scaling_in=True,
                            ensemble_scaling_out=True,
                            ensemble_bias=True,
                            scaling_init='ones',
                        )
                        _replace_first_with_rankr(
                            self.backbone,
                            k=k,
                            rank=rla_rank,
                            additive=rla_additive,
                            init_mode=rla_init, base_preserve_noise=rla_base_preserve_noise,
                        )
                    else:
                        # RLA-uniform: every linear layer is rank-r / additive.
                        lib.deep.make_efficient_ensemble(
                            self.backbone,
                            lib.deep.LinearEfficientEnsembleRankR,
                            k=k,
                            rank=rla_rank,
                            scaling_init='ones',
                            additive=rla_additive,
                            init_mode=rla_init, base_preserve_noise=rla_base_preserve_noise,
                        )
                    # First-layer R column initialisation:
                    #   - variance_preserving: each rank column gets section-based
                    #     random signs / normal, then scaled by 1/sqrt(rank).
                    #   - base_preserving: path 0 already holds the proper
                    #     section-based init from the layer constructor; extra
                    #     paths stay zero/tiny so we *do not* overwrite them.
                    if not rla_additive and rla_init != 'base_preserving':
                        _init_first_adapter_rankr(
                            _get_first_rankr_layer(self.backbone).R,  # type: ignore[code]
                            first_adapter_init,
                            first_adapter_sections,
                        )
                    elif not rla_additive and rla_init == 'base_preserving':
                        # Apply the section-based init only to path 0 of R.
                        _init_first_adapter_basepreserve(
                            _get_first_rankr_layer(self.backbone).R,  # type: ignore[code]
                            first_adapter_init,
                            first_adapter_sections,
                        )

            elif arch_type in ('tabm-mini', 'tabm-mini-normal'):
                # MiniEnsemble
                assert first_adapter_init is not None
                self.minimal_ensemble_adapter = lib.deep.ScaleEnsemble(
                    k,
                    d_flat,
                    init='random-signs' if num_embeddings is None else 'normal',
                )
                _init_first_adapter(
                    self.minimal_ensemble_adapter.weight,  # type: ignore[code]
                    first_adapter_init,
                    first_adapter_sections,
                )

            elif arch_type == 'tabm-packed':
                # Packed ensemble.
                # In terms of the Packed Ensembles paper by Laurent et al.,
                # TabM-packed is PackedEnsemble(alpha=k, M=k, gamma=1).
                assert first_adapter_init is None
                lib.deep.make_efficient_ensemble(self.backbone, lib.deep.NLinear, n=k)

            else:
                raise ValueError(f'Unknown arch_type: {arch_type}')

        # >>> Output
        d_block = backbone['d_block']
        d_out = 1 if n_classes is None else n_classes
        self.output = (
            nn.Linear(d_block, d_out)
            if arch_type == 'plain'
            else lib.deep.NLinear(k, d_block, d_out)  # type: ignore[code]
        )

        # >>>
        self.arch_type = arch_type
        self.k = k
        self.share_training_batches = share_training_batches
        self.mfb_cfg = {} if mfb is None else dict(mfb)
        self.mfb_enabled = bool(self.mfb_cfg.get('enabled', False))
        self.mfb_mask_mode = str(self.mfb_cfg.get('mask_mode', 'member_fixed'))
        self.mfb_mask_granularity = str(self.mfb_cfg.get('mask_granularity', 'feature_group'))
        self.mfb_keep_rate = float(self.mfb_cfg.get('keep_rate', 1.0))
        self.mfb_inverted_scaling = bool(self.mfb_cfg.get('inverted_scaling', True))
        self.mfb_use_soft_mask = bool(self.mfb_cfg.get('use_soft_mask', False))
        self.mfb_mask_strength = float(self.mfb_cfg.get('mask_strength', 1.0))
        self.mfb_warmup_epochs = int(self.mfb_cfg.get('warmup_epochs', 0))
        self.mfb_start_epoch = int(self.mfb_cfg.get('start_epoch', 0))
        self.mfb_group_mode = str(self.mfb_cfg.get('group_mode', 'feature_group'))
        self.mfb_categorical_handling = str(self.mfb_cfg.get('categorical_handling', 'drop_allowed'))
        self.mfb_epoch = 0
        self.mfb_feature_widths = [int(x) for x in first_adapter_sections]
        if self.mfb_enabled:
            assert self.k is not None
            if self.mfb_mask_granularity != 'feature_group':
                raise ValueError(f'Unsupported MFB mask_granularity={self.mfb_mask_granularity!r}')
            if self.mfb_group_mode not in {'feature_group', 'numerical_only', 'per_member'}:
                raise ValueError(f'Unsupported MFB group_mode={self.mfb_group_mode!r}')
            protected_feature_ids = list(self.mfb_cfg.get('protected_feature_ids') or [])
            if (
                self.mfb_group_mode == 'numerical_only'
                or self.mfb_categorical_handling in {'no_cat_drop', 'num_only'}
            ):
                protected_feature_ids.extend(range(n_num_features, n_num_features + len(cat_cardinalities)))
            protected_feature_ids = sorted(set(int(x) for x in protected_feature_ids))
            feature_mask, dim_mask, mask_stats = _make_mfb_feature_group_mask(
                k=self.k,
                feature_widths=self.mfb_feature_widths,
                keep_rate=self.mfb_keep_rate,
                seed=int(self.mfb_cfg.get('mask_seed', 0)),
                anchor_fraction=float(self.mfb_cfg.get('anchor_fraction', 0.0)),
                protected_feature_ids=protected_feature_ids,
            )
            self.register_buffer('mfb_fixed_feature_mask', torch.from_numpy(feature_mask), persistent=True)
            self.register_buffer('mfb_fixed_dim_mask', torch.from_numpy(dim_mask), persistent=True)
            self.mfb_mask_stats = mask_stats
        else:
            self.register_buffer('mfb_fixed_feature_mask', torch.empty((0, 0), dtype=torch.float32), persistent=True)
            self.register_buffer('mfb_fixed_dim_mask', torch.empty((0, 0), dtype=torch.float32), persistent=True)
            self.mfb_mask_stats = {'keep_rate_actual': 1.0}

    def set_epoch(self, epoch: int) -> None:
        self.mfb_epoch = int(epoch)

    def _current_mfb_mask_strength(self) -> float:
        if not self.mfb_use_soft_mask:
            return 1.0
        if self.mfb_warmup_epochs <= 0:
            return self.mfb_mask_strength
        effective_epoch = max(0, self.mfb_epoch - self.mfb_start_epoch)
        return self.mfb_mask_strength * min(1.0, max(0.0, float(effective_epoch) / float(self.mfb_warmup_epochs)))

    def _sample_mfb_mask(self, device_: torch.device, dtype: torch.dtype) -> Tensor:
        assert self.k is not None
        mask = (torch.rand((self.k, len(self.mfb_feature_widths)), device=device_) < self.mfb_keep_rate).to(dtype)
        zero_rows = mask.sum(dim=1) == 0
        if bool(zero_rows.any()):
            zero_indices = torch.where(zero_rows)[0]
            random_cols = torch.randint(0, len(self.mfb_feature_widths), size=(len(zero_indices),), device=device_)
            mask[zero_rows] = 0.0
            mask[zero_indices, random_cols] = 1.0
        feature_index = torch.tensor(
            [feature_id for feature_id, width in enumerate(self.mfb_feature_widths) for _ in range(width)],
            device=device_,
            dtype=torch.long,
        )
        return mask[:, feature_index]

    def _apply_mfb_mask(self, x: Tensor) -> Tensor:
        if not self.mfb_enabled or self.mfb_mask_mode == 'none':
            return x
        if self.mfb_epoch < self.mfb_start_epoch:
            return x
        if self.mfb_mask_mode == 'member_fixed':
            raw_mask = self.mfb_fixed_dim_mask.to(device=x.device, dtype=x.dtype)
        elif self.mfb_mask_mode == 'stochastic':
            if not self.training:
                return x
            raw_mask = self._sample_mfb_mask(x.device, x.dtype)
        else:
            raise ValueError(f'Unknown MFB mask_mode={self.mfb_mask_mode!r}')
        if self.mfb_use_soft_mask and self.mfb_mask_mode == 'member_fixed':
            alpha = self._current_mfb_mask_strength()
            effective_mask = (1.0 - alpha) + alpha * raw_mask
            return x * effective_mask.unsqueeze(0)
        x = x * raw_mask.unsqueeze(0)
        if self.mfb_inverted_scaling and self.mfb_keep_rate < 1.0:
            x = x / max(self.mfb_keep_rate, 1e-6)
        return x

    def forward(
        self, x_num: None | Tensor = None, x_cat: None | Tensor = None
    ) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num if self.num_module is None else self.num_module(x_num))
        if x_cat is None:
            assert self.cat_module is None
        else:
            assert self.cat_module is not None
            x.append(self.cat_module(x_cat).float())
        x = torch.column_stack([x_.flatten(1, -1) for x_ in x])

        if self.k is not None:
            if self.share_training_batches or not self.training:
                # (B, D) -> (B, K, D)
                x = x[:, None].expand(-1, self.k, -1)
            else:
                # (B * K, D) -> (B, K, D)
                x = x.reshape(len(x) // self.k, self.k, *x.shape[1:])
            if self.minimal_ensemble_adapter is not None:
                x = self.minimal_ensemble_adapter(x)
            x = self._apply_mfb_mask(x)
        else:
            assert self.minimal_ensemble_adapter is None

        x = self.backbone(x)
        x = self.output(x)
        if self.k is None:
            # Adjust the output shape for plain networks to make them compatible
            # with the rest of the script (loss, metrics, predictions, ...).
            # (B, D_OUT) -> (B, 1, D_OUT)
            x = x[:, None]
        return x

    def get_member_logits(
        self, x_num: None | Tensor = None, x_cat: None | Tensor = None
    ) -> Tensor:
        """Return per-member logits/predictions with shape (B, K, D_OUT)."""
        return self.forward(x_num, x_cat)

    def get_first_batchensemble_r(self) -> None | Tensor:
        """Expose the first BatchEnsemble `r` parameter when present."""
        if self.arch_type in ('tabm', 'tabm-normal'):
            return _get_first_ensemble_layer(self.backbone).r
        return None


class Config(TypedDict):
    seed: int
    data: KWArgs
    bins: NotRequired[KWArgs]
    model: KWArgs
    head_selection: NotRequired[bool]
    optimizer: KWArgs
    n_lr_warmup_epochs: NotRequired[int]
    batch_size: int
    eval_batch_size: NotRequired[int]
    patience: int
    n_epochs: int
    gradient_clipping_norm: NotRequired[float]
    parameter_statistics: NotRequired[bool]
    # NOTE
    # Please, read these notes before using AMP and/or `torch.compile`.
    #
    # The usage of the following efficiency-related settings depends on the model.
    # To learn if a given model can run with AMP and torch.compile on a given task,
    # try activating these settings and check if the task metrics are satisfactory.
    # The following notes can be helpful.
    #
    # - For simple architectures, such as MLP or TabM, these settings often
    #   make models significantly faster without any negative side-effects.
    #   For a real world task, it is worth to doublecheck that by comparing runs
    #   with and without AMP and/or torch.compile.
    #
    # - For more complex architectures, these settings should be used
    #   with extra caution. For example, some baselines used in this project showed
    #   worse performance when trained with AMP. For some models, AMP with BF16 hurts
    #   the performance, but AMP with FP16 works fine. Sometimes, it is the opposite.
    #   Sometimes, it depends on a dataset. Because of that, all baselines were run
    #   without AMP and torch.compile to ensure that results are representative.
    #
    # - AMP usually provides significantly larger speedups than `torch.compile`.
    #   So, if there are any issues with `torch.compile`, using only AMP will still
    #   lead to substantially faster models.
    #
    # - If a training run is already fast (e.g. on small datasets),
    #   `torch.compile` can make it *slower*, because the compilation itself
    #   takes some time (in particular, at the beginning of the first epoch,
    #   and at the beginning of the first evaluation).
    #
    # - Generally, compared to AMP, `torch.compile` is a younger technology, and a
    #   model must meet certain requirements to be compatible with `torch.compile`.
    #   In case of any issues, try updating PyTorch.
    amp: NotRequired[bool]  # torch.autocast
    compile: NotRequired[bool]  # torch.compile
    use_ncl: NotRequired[bool]
    lambda_ncl: NotRequired[float]
    ncl_warmup_epochs: NotRequired[int]
    ncl_space: NotRequired[Literal['logits', 'probs', 'hybrid']]
    use_esam: NotRequired[bool]
    esam_rho: NotRequired[float]
    esam_eps: NotRequired[float]
    esam_adapter_only: NotRequired[bool]
    esam_memberwise: NotRequired[bool]
    esam_warmup_epochs: NotRequired[int]
    esam_start_epoch: NotRequired[int]
    esam_end_epoch: NotRequired[int]
    esam_log_diagnostics: NotRequired[bool]
    esam_diagnostics_every: NotRequired[int]
    rla_adapter_lr_multiplier: NotRequired[float]
    rla_extra_paths_freeze_fraction: NotRequired[float]
    cf_fisd: NotRequired[KWArgs]


def main(
    config: Config | str | Path,
    output: None | str | Path = None,
    *,
    force: bool = False,
) -> None | lib.JSONDict:
    # >>> Start
    config, output = lib.check(config, output, config_type=Config)
    if not lib.start(output, force=force):
        return None

    lib.print_config(config)  # type: ignore[code]
    delu.random.seed(config['seed'])
    device = lib.get_device()
    report = lib.create_report(main, config)

    # >>> Data
    dataset = lib.data.build_dataset(**config['data'])
    if dataset.task.is_regression:
        dataset.data['y'], regression_label_stats = lib.data.standardize_labels(
            dataset.data['y']
        )
    else:
        regression_label_stats = None

    # Convert binary features to categorical features.
    if dataset.n_bin_features > 0:
        x_bin = dataset.data.pop('x_bin')
        # Remove binary features with just one unique value in the training set.
        # This must be done, otherwise, the script will fail on one specific dataset
        # from the "why" benchmark.
        n_bin_features = x_bin['train'].shape[1]
        good_bin_idx = [
            i for i in range(n_bin_features) if len(np.unique(x_bin['train'][:, i])) > 1
        ]
        if len(good_bin_idx) < n_bin_features:
            x_bin = {k: v[:, good_bin_idx] for k, v in x_bin.items()}

        if dataset.n_cat_features == 0:
            dataset.data['x_cat'] = {
                part: np.zeros((dataset.size(part), 0), dtype=np.int64)
                for part in x_bin
            }
        for part in x_bin:
            dataset.data['x_cat'][part] = np.column_stack(
                [dataset.data['x_cat'][part], x_bin[part].astype(np.int64)]
            )
        del x_bin
    dataset = dataset.to_torch(device)
    Y_train = dataset.data['y']['train'].to(
        torch.long if dataset.task.is_classification else torch.float
    )

    # >>> Model
    if 'bins' in config:
        # Compute the bins for PiecewiseLinearEncoding and PiecewiseLinearEmbeddings.
        compute_bins_kwargs = (
            {
                'y': Y_train.to(
                    torch.long if dataset.task.is_classification else torch.float
                ),
                'regression': dataset.task.is_regression,
                'verbose': True,
            }
            if 'tree_kwargs' in config['bins']
            else {}
        )
        bin_edges = rtdl_num_embeddings.compute_bins(
            dataset.data['x_num']['train'], **config['bins'], **compute_bins_kwargs
        )
        logger.info(f'Bin counts: {[len(x) - 1 for x in bin_edges]}')
    else:
        bin_edges = None
    model = Model(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),
        n_classes=dataset.task.try_compute_n_classes(),
        **config['model'],
        bins=bin_edges,
    )
    report['n_parameters'] = lib.deep.get_n_parameters(model)
    logger.info(f'n_parameters = {report["n_parameters"]}')
    report['prediction_type'] = 'labels' if dataset.task.is_regression else 'probs'
    model.to(device)
    if lib.is_dataparallel_available():
        model = nn.DataParallel(model)

    root_model = model.module if isinstance(model, nn.DataParallel) else model
    report['mfb'] = {
        'enabled': bool(getattr(root_model, 'mfb_enabled', False)),
        'config': dict(getattr(root_model, 'mfb_cfg', {})),
        'mask_stats': dict(getattr(root_model, 'mfb_mask_stats', {})),
    }

    cf_fisd_cfg = config.get('cf_fisd')
    if cf_fisd_cfg:
        cf_fisd_cat_cards = list(dataset.compute_cat_cardinalities())
        if 'num_embeddings' in config['model']:
            cf_fisd_d_emb = int(config['model']['num_embeddings']['d_embedding'])
            cf_fisd_d_features = [cf_fisd_d_emb] * dataset.n_num_features + cf_fisd_cat_cards
        else:
            cf_fisd_d_features = [1] * dataset.n_num_features + cf_fisd_cat_cards
        cf_fisd_n_features = dataset.n_num_features + len(cf_fisd_cat_cards)
        cf_fisd_teacher_names = tuple(cf_fisd_cfg.get('teacher_names', lib.cf_fisd.DEFAULT_TEACHER_NAMES))
        cf_fisd_teachers = lib.cf_fisd.load_teacher_importances(
            cf_fisd_cfg['teacher_dir'],
            cf_fisd_cfg['dataset_name'],
            n_features=cf_fisd_n_features,
            teacher_names=cf_fisd_teacher_names,
        )
        cf_fisd_member_groups = (
            lib.cf_fisd.default_member_groups(int(config['model']['k']), cf_fisd_teacher_names)
            if cf_fisd_cfg.get('member_groups') is None
            else {k: list(v) for k, v in cf_fisd_cfg['member_groups'].items()}
        )
        cf_fisd_lambda = float(cf_fisd_cfg.get('lambda', 0.0))
        cf_fisd_variant = str(cf_fisd_cfg.get('variant', 'raw'))
        cf_fisd_start_epoch = int(cf_fisd_cfg.get('start_epoch', 0))
        cf_fisd_r1_param = _get_first_adapter_for_cf_fisd(root_model.backbone)
        report['cf_fisd'] = {
            'lambda': cf_fisd_lambda,
            'variant': cf_fisd_variant,
            'teacher_names': list(cf_fisd_teacher_names),
            'member_groups': {k: list(v) for k, v in cf_fisd_member_groups.items()},
            'd_features': list(cf_fisd_d_features),
            'n_features': cf_fisd_n_features,
            'teacher_dir': str(cf_fisd_cfg['teacher_dir']),
            'start_epoch': cf_fisd_start_epoch,
            'mode': cf_fisd_cfg.get('mode', cf_fisd_variant),
        }
    else:
        cf_fisd_lambda = 0.0
        cf_fisd_variant = 'raw'
        cf_fisd_d_features = []
        cf_fisd_teachers = {}
        cf_fisd_member_groups = {}
        cf_fisd_start_epoch = 0
        cf_fisd_r1_param = None
        report['cf_fisd'] = {'lambda': 0.0}

    def compute_cf_fisd_penalty() -> Tensor:
        if cf_fisd_r1_param is None or cf_fisd_lambda <= 0.0:
            return Y_train.new_zeros((), dtype=torch.float32)
        if step // epoch_size < cf_fisd_start_epoch:
            return Y_train.new_zeros((), dtype=torch.float32)
        return lib.cf_fisd.cf_fisd_loss(
            cf_fisd_r1_param,
            cf_fisd_teachers,
            cf_fisd_member_groups,
            cf_fisd_variant,
            cf_fisd_d_features,
        )

    # >>> Training
    step = 0
    batch_size = config['batch_size']
    report['epoch_size'] = epoch_size = math.ceil(dataset.size('train') / batch_size)
    eval_batch_size = config.get(
        'eval_batch_size',
        # With torch.compile,
        # the largest possible evaluation batch size is noticeably smaller.
        2048 if config.get('compile', False) else 32768,
    )
    chunk_size = None  # Currently, not used.
    share_training_batches = config['model'].get(
        'share_training_batches', DEFAULT_SHARE_TRAINING_BATCHES
    )
    use_ncl = config.get('use_ncl', False)
    lambda_ncl = float(config.get('lambda_ncl', 0.0))
    ncl_warmup_epochs = int(config.get('ncl_warmup_epochs', 0))
    ncl_space = config.get('ncl_space', 'logits')
    if ncl_space not in ('logits', 'probs', 'hybrid'):
        raise ValueError(f'Unknown ncl_space: {ncl_space}')
    report['ncl'] = {
        'use_ncl': bool(use_ncl),
        'lambda_ncl': lambda_ncl,
        'ncl_warmup_epochs': ncl_warmup_epochs,
        'ncl_space': ncl_space,
    }
    use_esam = bool(config.get('use_esam', False))
    esam_rho = float(config.get('esam_rho', 0.01))
    esam_eps = float(config.get('esam_eps', 1e-12))
    esam_adapter_only = bool(config.get('esam_adapter_only', True))
    esam_memberwise = bool(config.get('esam_memberwise', True))
    esam_warmup_epochs = int(config.get('esam_warmup_epochs', 0))
    esam_start_epoch = int(config.get('esam_start_epoch', 0))
    esam_end_epoch = int(config.get('esam_end_epoch', -1))
    esam_log_diagnostics = bool(config.get('esam_log_diagnostics', True))
    esam_diagnostics_every = int(config.get('esam_diagnostics_every', 100))
    report['esam'] = {
        'use_esam': use_esam,
        'esam_rho': esam_rho,
        'esam_eps': esam_eps,
        'esam_adapter_only': esam_adapter_only,
        'esam_memberwise': esam_memberwise,
        'esam_warmup_epochs': esam_warmup_epochs,
        'esam_start_epoch': esam_start_epoch,
        'esam_end_epoch': esam_end_epoch,
        'esam_log_diagnostics': esam_log_diagnostics,
        'esam_diagnostics_every': esam_diagnostics_every,
    }

    rla_adapter_lr_multiplier = float(config.get('rla_adapter_lr_multiplier', 1.0))
    rla_custom_groups = []
    if rla_adapter_lr_multiplier != 1.0:
        rla_adapter_params = [
            p
            for module in model.modules()
            if isinstance(module, lib.deep.LinearEfficientEnsembleRankR)
            for p in (module.R, module.S)
        ]
        if rla_adapter_params:
            rla_custom_groups.append({
                'params': rla_adapter_params,
                'lr': config['optimizer']['lr'] * rla_adapter_lr_multiplier,
            })
    optimizer_cfg = dict(config['optimizer'])
    optimizer_type = optimizer_cfg.pop('type', 'AdamW')
    optimizer = lib.deep.make_optimizer(
        optimizer_type,
        **optimizer_cfg,
        params=lib.deep.make_parameter_groups(model, custom_groups=rla_custom_groups),
    )
    rla_extra_paths_freeze_fraction = float(config.get('rla_extra_paths_freeze_fraction', 0.0))
    assert 0.0 <= rla_extra_paths_freeze_fraction <= 1.0
    rla_extra_paths_freeze_until_epoch = (
        math.ceil(config['n_epochs'] * rla_extra_paths_freeze_fraction)
        if config['n_epochs'] > 0
        else math.ceil(2 * config['patience'] * rla_extra_paths_freeze_fraction)
    )
    report['rla_training'] = {
        'adapter_lr_multiplier': rla_adapter_lr_multiplier,
        'extra_paths_freeze_fraction': rla_extra_paths_freeze_fraction,
        'extra_paths_freeze_until_epoch': rla_extra_paths_freeze_until_epoch,
    }

    def zero_rla_extra_path_grads() -> None:
        if rla_extra_paths_freeze_fraction == 0.0:
            return
        if step // epoch_size >= rla_extra_paths_freeze_until_epoch:
            return
        for module in model.modules():
            if isinstance(module, lib.deep.LinearEfficientEnsembleRankR) and module.rank > 1:
                if module.R.grad is not None:
                    module.R.grad[:, :, 1:].zero_()
                if module.S.grad is not None:
                    module.S.grad[:, :, 1:].zero_()
    gradient_clipping_norm = config.get('gradient_clipping_norm')
    _loss_fn = (
        nn.functional.mse_loss
        if dataset.task.is_regression
        else nn.functional.cross_entropy
    )

    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        return _loss_fn(
            y_pred.flatten(0, 1),
            (
                y_true.repeat_interleave(y_pred.shape[1])
                if share_training_batches
                else y_true
            ),
        )

    def ncl_penalty(y_pred: Tensor) -> Tensor:
        if y_pred.shape[1] <= 1:
            return y_pred.new_zeros(())

        def corr_penalty(member_outputs: Tensor) -> Tensor:
            if member_outputs.ndim == 2:
                member_outputs = member_outputs.unsqueeze(-1)
            centered = member_outputs - member_outputs.mean(dim=1, keepdim=True)
            norms = centered.norm(dim=-1)
            denom = torch.einsum('bi,bj->bij', norms, norms).clamp_min(1e-8)
            corr = torch.einsum('bid,bjd->bij', centered, centered) / denom
            k_members = centered.shape[1]
            off_diag = ~torch.eye(
                k_members, dtype=torch.bool, device=centered.device
            ).unsqueeze(0)
            return corr.masked_select(off_diag).mean()

        if not dataset.task.is_classification:
            return corr_penalty(y_pred)

        if ncl_space == 'logits':
            return corr_penalty(y_pred)
        if ncl_space == 'probs':
            return corr_penalty(y_pred.softmax(-1))
        # Hybrid penalty: equal weighting between logits-space and probability-space.
        return 0.5 * (corr_penalty(y_pred) + corr_penalty(y_pred.softmax(-1)))

    base_model = model.module if isinstance(model, nn.DataParallel) else model

    def get_esam_adapter_params() -> list[tuple[str, nn.Parameter]]:
        params: list[tuple[str, nn.Parameter]] = []
        if not esam_adapter_only:
            params.extend(
                [
                    (name, param)
                    for name, param in base_model.named_parameters()
                    if param.requires_grad
                ]
            )
            return params

        for module_name, module in base_model.named_modules():
            if isinstance(module, lib.deep.LinearEfficientEnsembleRankR):
                for pname in ['R', 'S', 'bias']:
                    p = getattr(module, pname)
                    if p is None or not p.requires_grad:
                        continue
                    params.append((f'{module_name}.{pname}', p))
            elif isinstance(module, lib.deep.LinearEfficientEnsemble):
                for pname in ['r', 's', 'bias']:
                    p = getattr(module, pname)
                    if p is None or not p.requires_grad:
                        continue
                    if pname == 'bias' and p.ndim != 2:
                        # Shared bias is not member-specific.
                        continue
                    params.append((f'{module_name}.{pname}', p))
            elif isinstance(module, lib.deep.ScaleEnsemble):
                if module.weight.requires_grad:
                    params.append((f'{module_name}.weight', module.weight))

        if not params:
            logger.warning('ESAM adapter-only selection found no adapter parameters.')
        return params

    esam_adapter_named_params = get_esam_adapter_params()
    esam_adapter_params = [p for _, p in esam_adapter_named_params]
    report['esam']['adapter_parameter_names'] = [
        {'name': name, 'shape': list(param.shape)}
        for name, param in esam_adapter_named_params
    ]
    logger.info(
        'ESAM adapter params: {}',
        [f'{name}:{tuple(param.shape)}' for name, param in esam_adapter_named_params],
    )

    def compute_esam_perturbations(
        grads: list[None | Tensor],
        rho: float,
        eps: float,
        memberwise: bool,
    ) -> tuple[list[Tensor], float, float, bool]:
        if not grads:
            return [], 0.0, 0.0, True

        grad_norm_sq = 0.0
        fallback_used = False
        for param, grad in zip(esam_adapter_params, grads):
            if grad is None:
                continue
            if memberwise and param.ndim >= 2 and param.shape[0] > 1:
                member_norms = grad.reshape(param.shape[0], -1).norm(dim=1)
                grad_norm_sq += float((member_norms**2).sum().item())
            else:
                if memberwise:
                    fallback_used = True
                grad_norm_sq += float((grad.norm() ** 2).item())
        grad_norm = math.sqrt(grad_norm_sq)

        perturbations: list[Tensor] = []
        perturb_norm_sq = 0.0
        for param, grad in zip(esam_adapter_params, grads):
            if grad is None:
                perturb = torch.zeros_like(param)
            elif memberwise and param.ndim >= 2 and param.shape[0] > 1:
                member_norms = grad.reshape(param.shape[0], -1).norm(dim=1)
                view_shape = (param.shape[0],) + (1,) * (param.ndim - 1)
                scale = rho / (member_norms.view(view_shape) + eps)
                perturb = grad * scale
            else:
                perturb = grad * (rho / (grad_norm + eps))
            perturbations.append(perturb)
            perturb_norm_sq += float((perturb.norm() ** 2).item())
        perturb_norm = math.sqrt(perturb_norm_sq)
        return perturbations, grad_norm, perturb_norm, fallback_used

    def apply_esam_perturbation(perturbations: list[Tensor]) -> None:
        with torch.no_grad():
            for param, perturb in zip(esam_adapter_params, perturbations):
                param.add_(perturb)

    def restore_esam_perturbation(perturbations: list[Tensor]) -> float:
        max_abs_restore = 0.0
        with torch.no_grad():
            for param, perturb in zip(esam_adapter_params, perturbations):
                param.sub_(perturb)
                if perturb.numel():
                    max_abs_restore = max(max_abs_restore, float(perturb.abs().max().item()))
        return max_abs_restore

    # The following generator is used only for creating training batches,
    # so the random seed fully determines the sequence of training objects.
    batch_generator = torch.Generator(device).manual_seed(config['seed'])
    timer = delu.tools.Timer()
    early_stopping = delu.tools.EarlyStopping(config['patience'], mode='max')
    parameter_statistics = config.get('parameter_statistics', config['seed'] == 1)
    training_log = []
    esam_diag_records: list[dict[str, Any]] = []
    writer = torch.utils.tensorboard.SummaryWriter(output)  # type: ignore[code]
    writer_failed = False
    run_id = f'{output.parent.name}__{output.name}'
    esam_diag_path = (output.parent / '_esam_diagnostics')
    esam_diag_file = esam_diag_path / f'{run_id}.jsonl'
    if use_esam and esam_log_diagnostics:
        esam_diag_path.mkdir(parents=True, exist_ok=True)
        esam_diag_file.write_text('')

    def safe_add_scalars(tag: str, values: dict[str, Any], step_: int, walltime: float) -> None:
        nonlocal writer, writer_failed
        if writer is None or writer_failed:
            return
        try:
            writer.add_scalars(tag, values, step_, walltime)
        except OSError as err:
            writer_failed = True
            logger.warning(f'TensorBoard write disabled due to OS error: {err}')
            try:
                writer.close()
            except Exception:
                pass
            writer = None

    # Only bfloat16 was tested as amp_dtype.
    # However, float16 is supported as a fallback.
    # To enable float16, uncomment the two lines below.
    amp_dtype = (
        torch.bfloat16
        if config.get('amp', False)
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        # else torch.float16
        # if config.get('amp', False) and and torch.cuda.is_available()
        else None
    )
    amp_enabled = amp_dtype is not None
    # For FP16, the gradient scaler must be used.
    grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore[code]
    logger.info(f'AMP enabled: {amp_enabled}')

    if config.get('compile', False):
        # NOTE
        # `torch.compile` is intentionally called without the `mode` argument,
        # because it caused issues with training.
        model = torch.compile(model)
        evaluation_mode = torch.no_grad
    else:
        evaluation_mode = torch.inference_mode

    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
    def apply_model(part: PartKey, idx: Tensor) -> Tensor:
        return (
            model(
                dataset.data['x_num'][part][idx] if 'x_num' in dataset.data else None,
                dataset.data['x_cat'][part][idx] if 'x_cat' in dataset.data else None,
            )
            .squeeze(-1)  # Remove the last dimension for regression predictions.
            .float()
        )

    @evaluation_mode()
    def evaluate(
        parts: list[PartKey], eval_batch_size: int
    ) -> tuple[
        dict[PartKey, Any],
        dict[PartKey, np.ndarray],
        dict[PartKey, np.ndarray],
        dict[PartKey, Any],
        int,
    ]:
        def compute_diversity(head_preds: np.ndarray) -> dict[str, float]:
            if head_preds.ndim == 2:
                members = head_preds[..., None]
            else:
                members = head_preds
            members = members.reshape(members.shape[0], members.shape[1], -1)
            if members.shape[1] <= 1:
                return {
                    'mean_centered_corr': 0.0,
                    'mean_pairwise_disagreement': 0.0,
                    'member_std': 0.0,
                }

            centered = members - members.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(centered, axis=-1)
            denom = np.einsum('bi,bj->bij', norms, norms) + 1e-8
            corr = np.einsum('bid,bjd->bij', centered, centered) / denom
            off_diag = ~np.eye(members.shape[1], dtype=bool)
            mean_centered_corr = float(corr[:, off_diag].mean())

            if dataset.task.is_regression:
                pairwise_disagreement = float('nan')
            elif dataset.task.is_binclass:
                labels = (head_preds > 0.5).astype(np.int64)
                pairwise_disagreement = float(
                    (labels[:, :, None] != labels[:, None, :])[:, off_diag].mean()
                )
            else:
                labels = head_preds.argmax(-1)
                pairwise_disagreement = float(
                    (labels[:, :, None] != labels[:, None, :])[:, off_diag].mean()
                )

            member_std = float(members.std(axis=1).mean())
            return {
                'mean_centered_corr': mean_centered_corr,
                'mean_pairwise_disagreement': pairwise_disagreement,
                'member_std': member_std,
            }

        model.eval()
        head_predictions: dict[PartKey, np.ndarray] = {}
        for part in parts:
            while eval_batch_size:
                try:
                    head_predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx)
                                for idx in torch.arange(
                                    dataset.size(part), device=device
                                ).split(eval_batch_size)
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    logger.warning(f'eval_batch_size = {eval_batch_size}')
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')
        if dataset.task.is_regression:
            assert regression_label_stats is not None
            head_predictions = {
                k: v * regression_label_stats.std + regression_label_stats.mean
                for k, v in head_predictions.items()
            }
        else:
            head_predictions = {
                k: scipy.special.softmax(v, axis=-1)
                for k, v in head_predictions.items()
            }
            if dataset.task.is_binclass:
                head_predictions = {k: v[..., 1] for k, v in head_predictions.items()}

        predictions = {k: v.mean(1) for k, v in head_predictions.items()}
        metrics = (
            dataset.task.calculate_metrics(predictions, report['prediction_type'])
            if lib.are_valid_predictions(predictions)
            else {x: {'score': lib.WORST_SCORE} for x in predictions}
        )
        diversity = {part: compute_diversity(head_predictions[part]) for part in parts}
        return metrics, predictions, head_predictions, diversity, eval_batch_size

    def save_checkpoint() -> None:
        lib.dump_checkpoint(
            output,
            {
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'batch_generator': batch_generator.get_state(),
                'random_state': delu.random.get_state(),
                'early_stopping': early_stopping,
                'report': report,
                'timer': timer,
                'training_log': training_log,
            }
            | (
                {} if grad_scaler is None else {'grad_scaler': grad_scaler.state_dict()}
            ),
        )
        lib.dump_report(output, report)
        lib.backup_output(output)

    print()
    timer.run()
    while config['n_epochs'] == -1 or step // epoch_size < config['n_epochs']:
        print(f'[...] {lib.try_get_relative_path(output)} | {timer}')

        model.train()
        if hasattr(model, 'set_epoch'):
            model.set_epoch(step // epoch_size + 1)
        epoch_total_losses = []
        epoch_task_losses = []
        epoch_ncl_losses = []
        epoch_esam_sharpness = []
        epoch_esam_grad_norm = []
        epoch_esam_perturb_norm = []
        batches = (
            torch.randperm(
                dataset.size('train'),
                generator=batch_generator,
                device=device,
            ).split(batch_size)
            if share_training_batches
            else [
                x.transpose(0, 1).flatten()
                for x in torch.rand(
                    (config['model']['k'], dataset.size('train')),
                    generator=batch_generator,
                    device=device,
                )
                .argsort(dim=1)
                .split(batch_size, dim=1)
            ]
        )
        for batch_idx in tqdm(batches, desc=f'Epoch {step // epoch_size} Step {step}'):
            optimizer.zero_grad()
            y_pred = apply_model('train', batch_idx)
            task_loss = loss_fn(y_pred, Y_train[batch_idx])
            ncl_raw = (
                ncl_penalty(y_pred)
                if use_ncl and lambda_ncl > 0.0 and y_pred.shape[1] > 1
                else task_loss.new_zeros(())
            )
            if use_ncl and lambda_ncl > 0.0 and ncl_warmup_epochs > 0:
                warmup_factor = min(
                    1.0, float(step // epoch_size + 1) / float(ncl_warmup_epochs)
                )
            else:
                warmup_factor = 1.0
            lambda_effective = lambda_ncl * warmup_factor if use_ncl else 0.0
            cf_fisd_penalty = compute_cf_fisd_penalty()
            clean_loss = task_loss + lambda_effective * ncl_raw + cf_fisd_lambda * cf_fisd_penalty
            if not torch.isfinite(clean_loss):
                report['failure'] = {'reason': 'non_finite_loss', 'step': int(step), 'stage': 'clean'}
                lib.dump_report(output, report)
                raise RuntimeError(f'Non-finite clean loss at step {step}')

            epoch = step // epoch_size
            esam_enabled_now = (
                use_esam
                and esam_rho > 0.0
                and bool(esam_adapter_params)
                and epoch >= esam_start_epoch
                and (esam_end_epoch < 0 or epoch <= esam_end_epoch)
            )
            if esam_enabled_now and esam_warmup_epochs > 0:
                esam_rho_effective = esam_rho * min(
                    1.0, float(epoch - esam_start_epoch + 1) / float(esam_warmup_epochs)
                )
            else:
                esam_rho_effective = esam_rho

            esam_clean = float(clean_loss.detach().item())
            esam_perturbed = esam_clean
            esam_sharpness = 0.0
            esam_grad_norm = 0.0
            esam_perturb_norm = 0.0
            esam_fallback_used = False
            esam_restore_max_abs = 0.0
            esam_pending_perturbations = None
            loss = clean_loss

            if esam_enabled_now and esam_rho_effective > 0.0:
                grads = torch.autograd.grad(
                    clean_loss,
                    esam_adapter_params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                perturbations, esam_grad_norm, esam_perturb_norm, esam_fallback_used = (
                    compute_esam_perturbations(
                        list(grads),
                        esam_rho_effective,
                        esam_eps,
                        esam_memberwise,
                    )
                )
                apply_esam_perturbation(perturbations)
                esam_pending_perturbations = perturbations
                try:
                    y_pred_perturbed = apply_model('train', batch_idx)
                    pert_task_loss = loss_fn(y_pred_perturbed, Y_train[batch_idx])
                    pert_ncl_raw = (
                        ncl_penalty(y_pred_perturbed)
                        if use_ncl and lambda_ncl > 0.0 and y_pred_perturbed.shape[1] > 1
                        else pert_task_loss.new_zeros(())
                    )
                    pert_cf_fisd_penalty = compute_cf_fisd_penalty()
                    pert_loss = pert_task_loss + lambda_effective * pert_ncl_raw + cf_fisd_lambda * pert_cf_fisd_penalty
                    if not torch.isfinite(pert_loss):
                        report['failure'] = {'reason': 'non_finite_loss', 'step': int(step), 'stage': 'perturbed'}
                        lib.dump_report(output, report)
                        raise RuntimeError(f'Non-finite perturbed loss at step {step}')
                except Exception:
                    esam_restore_max_abs = restore_esam_perturbation(perturbations)
                    raise
                loss = pert_loss
                esam_perturbed = float(pert_loss.detach().item())
                esam_sharpness = esam_perturbed - esam_clean

            optimizer.zero_grad()
            if grad_scaler is None:
                loss.backward()
            else:
                grad_scaler.scale(loss).backward()

            if esam_pending_perturbations is not None:
                esam_restore_max_abs = restore_esam_perturbation(esam_pending_perturbations)

            zero_rla_extra_path_grads()

            if parameter_statistics and (
                step % epoch_size == 0  # The first batch of the epoch.
                or step // epoch_size == 0  # The first epoch.
            ):
                for k, v in lib.deep.compute_parameter_stats(model).items():
                    safe_add_scalars(k, v, step, timer.elapsed())
                    del k, v

            if gradient_clipping_norm is not None:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), gradient_clipping_norm
                )
            if grad_scaler is None:
                optimizer.step()
            else:
                grad_scaler.step(optimizer)
                grad_scaler.update()

            step += 1
            epoch_total_losses.append(loss.detach())
            epoch_task_losses.append(task_loss.detach())
            epoch_ncl_losses.append(ncl_raw.detach())
            if esam_enabled_now:
                epoch_esam_sharpness.append(esam_sharpness)
                epoch_esam_grad_norm.append(esam_grad_norm)
                epoch_esam_perturb_norm.append(esam_perturb_norm)

            if use_esam and esam_log_diagnostics and (step % max(1, esam_diagnostics_every) == 0):
                rec = {
                    'step': step,
                    'epoch': epoch,
                    'dataset': str(config['data']['path']),
                    'esam_enabled_now': esam_enabled_now,
                    'esam_fallback_mode': bool(esam_fallback_used),
                    'esam_rho_effective': float(esam_rho_effective),
                    'clean_loss': esam_clean,
                    'perturbed_loss': esam_perturbed,
                    'sharpness_proxy': esam_sharpness,
                    'adapter_grad_norm': esam_grad_norm,
                    'adapter_perturbation_norm': esam_perturb_norm,
                    'restore_max_abs_perturb': esam_restore_max_abs,
                    'member_loss_available': bool(y_pred.shape[1] > 1),
                    # Only populated in a follow-up pass where labels are aligned per member.
                    'train_member_loss_mean': None,
                }
                esam_diag_records.append(rec)
                with esam_diag_file.open('a') as f:
                    f.write(json.dumps(rec) + '\n')

        epoch_total_losses = torch.stack(epoch_total_losses).tolist()
        epoch_task_losses = torch.stack(epoch_task_losses).tolist()
        epoch_ncl_losses = torch.stack(epoch_ncl_losses).tolist()
        mean_esam_sharpness = (
            statistics.mean(epoch_esam_sharpness) if epoch_esam_sharpness else 0.0
        )
        mean_esam_grad_norm = (
            statistics.mean(epoch_esam_grad_norm) if epoch_esam_grad_norm else 0.0
        )
        mean_esam_perturb_norm = (
            statistics.mean(epoch_esam_perturb_norm) if epoch_esam_perturb_norm else 0.0
        )
        mean_loss = statistics.mean(epoch_total_losses)
        mean_task_loss = statistics.mean(epoch_task_losses)
        mean_ncl_loss = statistics.mean(epoch_ncl_losses)
        metrics, predictions, _, diversity, eval_batch_size = evaluate(
            ['val', 'test'], eval_batch_size
        )

        training_log.append(
            {
                'epoch-losses': epoch_total_losses,
                'epoch-task-losses': epoch_task_losses,
                'epoch-ncl-losses': epoch_ncl_losses,
                'lambda_ncl_effective': lambda_effective,
                'esam_mean_sharpness': mean_esam_sharpness,
                'esam_mean_grad_norm': mean_esam_grad_norm,
                'esam_mean_perturb_norm': mean_esam_perturb_norm,
                'metrics': metrics,
                'diversity': diversity,
                'time': timer.elapsed(),
            }
        )
        lib.print_metrics(mean_loss, metrics)
        safe_add_scalars(
            'loss',
            {
                'train_total': mean_loss,
                'train_task': mean_task_loss,
                'train_ncl': mean_ncl_loss,
                'train_esam_sharpness': mean_esam_sharpness,
                'train_esam_grad_norm': mean_esam_grad_norm,
                'train_esam_perturb_norm': mean_esam_perturb_norm,
            },
            step,
            timer.elapsed(),
        )
        for part in metrics:
            safe_add_scalars(
                'score', {part: metrics[part]['score']}, step, timer.elapsed()
            )
        for part in diversity:
            safe_add_scalars(
                'diversity/mean_centered_corr',
                {part: diversity[part]['mean_centered_corr']},
                step,
                timer.elapsed(),
            )
            safe_add_scalars(
                'diversity/member_std',
                {part: diversity[part]['member_std']},
                step,
                timer.elapsed(),
            )
            if not math.isnan(diversity[part]['mean_pairwise_disagreement']):
                safe_add_scalars(
                    'diversity/mean_pairwise_disagreement',
                    {part: diversity[part]['mean_pairwise_disagreement']},
                    step,
                    timer.elapsed(),
                )

        if (
            'metrics' not in report
            or metrics['val']['score'] > report['metrics']['val']['score']
        ):
            print('🌸 New best epoch! 🌸')
            report['best_step'] = step
            report['metrics'] = metrics
            save_checkpoint()
            lib.dump_predictions(output, predictions)

        early_stopping.update(metrics['val']['score'])
        if early_stopping.should_stop() or not lib.are_valid_predictions(predictions):
            break

        print()
    report['time'] = str(timer)

    # >>>
    if lib.get_checkpoint_path(output).exists():
        model.load_state_dict(lib.load_checkpoint(output)['model'])
    report['metrics'], predictions, head_predictions, report['diversity'], _ = evaluate(
        ['train', 'val', 'test'], eval_batch_size
    )
    report['chunk_size'] = chunk_size
    report['eval_batch_size'] = eval_batch_size
    report['esam']['diagnostics_path'] = str(esam_diag_file) if use_esam else None
    report['esam']['n_diagnostics_records'] = len(esam_diag_records)
    lib.dump_predictions(output, predictions)
    lib.dump_summary(output, lib.summarize(report))
    save_checkpoint()

    # >>> Submodel selection (TabM[B] & TabM[G]).
    if (
        config.get('head_selection', True)
        and head_predictions['train'].shape[1] > 1
        # The following conditions is a hack preventing the head selection during
        # the hyperparameter tuning, because bin/tune.py runs training
        # outside of the project directory.
        and lib.env.get_project_dir() in output.parents
        and output.parent.name != 'trials'
    ):
        if output.parent.name.endswith('-evaluation'):
            best_head_output = (
                output.parent.with_name(
                    output.parent.name.removesuffix('-evaluation')
                    + '-best-head-evaluation'
                )
                / output.name
            )
            greedy_heads_output = (
                output.parent.with_name(
                    output.parent.name.removesuffix('-evaluation')
                    + '-greedy-heads-evaluation'
                )
                / output.name
            )
        else:
            best_head_output = output.with_name(output.name + '-best-head')
            greedy_heads_output = output.with_name(output.name + '-greedy-heads')
        for dir_ in [best_head_output, greedy_heads_output]:
            if dir_.exists():
                logger.warning(f'Removing the existing output: {dir_}')
                shutil.rmtree(dir_)

        prediction_type = (
            lib.PredictionType.PROBS
            if dataset.task.is_classification
            else lib.PredictionType.LABELS
        )
        head_selection_timer = delu.tools.Timer()
        head_selection_timer.run()

        # >>> TabM[B]: select the Best submodel.
        n_heads = head_predictions['val'].shape[1]
        head_val_scores = np.array(
            [
                dataset.task.calculate_metrics(
                    {'val': head_predictions['val'][:, i]}, prediction_type
                )['val']['score']
                for i in range(n_heads)
            ]
        )
        best_head_idx = int(np.argmax(head_val_scores))
        best_head_output.mkdir(parents=True)
        lib.finish(
            best_head_output,
            report
            | {
                'heads': [best_head_idx],
                'head_selection_time': str(head_selection_timer),
                'metrics': dataset.task.calculate_metrics(
                    {k: v[:, best_head_idx] for k, v in head_predictions.items()},
                    prediction_type,
                ),
            },
        )

        # >>> TabM[G]: Greedily select a powerful subset of submodels.

        # Start with the best head.
        greedy_idx = [best_head_idx]
        greedy_score = head_val_scores[best_head_idx]

        greedy_mask = [False] * n_heads
        greedy_mask[best_head_idx] = True

        while len(greedy_idx) < n_heads:
            new_idx = None
            new_score = None

            # Iterating through all heads.
            for head_idx in range(n_heads):
                # If the head is already in greedy_idx, skip it.
                if greedy_mask[head_idx]:
                    continue

                candidate_idx = [*greedy_idx, head_idx]
                candidate_score = dataset.task.calculate_metrics(
                    {'val': head_predictions['val'][:, candidate_idx].mean(1)},
                    prediction_type,
                )['val']['score']
                if candidate_score > greedy_score and (
                    new_score is None or candidate_score > new_score
                ):
                    new_idx = candidate_idx
                    new_score = candidate_score

            # If no head improves the current greedy score,
            # the head selection process is stopped.
            if new_idx is None:
                break
            else:
                assert new_score is not None
                greedy_score = new_score
                greedy_idx = new_idx

        greedy_heads_output.mkdir(parents=True)
        lib.finish(
            greedy_heads_output,
            report
            | {
                'heads': greedy_idx,
                'head_selection_time': str(head_selection_timer),
                'metrics': dataset.task.calculate_metrics(
                    {k: v[:, greedy_idx].mean(1) for k, v in head_predictions.items()},
                    prediction_type,
                ),
            },
        )

    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run(main)
