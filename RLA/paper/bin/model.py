import math
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
        # >>> RLA (rank-r low-rank adapters) options.
        # rla_rank == 1 reduces exactly to baseline TabM. rla_rank > 1
        # generalises the rank-1 multiplicative adapter to a rank-r
        # factorisation. rla_first_only restricts the lift to the first
        # ensemble layer (RLA-first variant); otherwise every layer is
        # lifted (RLA-uniform). rla_additive switches multiplicative
        # adapters to additive LoRA-style ones (defensive ablation).
        rla_rank: int = 1,
        rla_first_only: bool = False,
        rla_additive: bool = False,
        rla_init: Literal['variance_preserving', 'base_preserving'] = (
            'variance_preserving'
        ),
        rla_base_preserve_noise: float = 1e-3,
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
        self.backbone = lib.deep.make_module(d_in=d_flat, **backbone)

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
    late_ensemble_loss_weight: NotRequired[float]
    late_ensemble_loss_start_fraction: NotRequired[float]
    rla_adapter_lr_multiplier: NotRequired[float]
    rla_extra_paths_freeze_fraction: NotRequired[float]
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
            rla_custom_groups.append(
                {
                    'params': rla_adapter_params,
                    'lr': config['optimizer']['lr'] * rla_adapter_lr_multiplier,
                }
            )
    optimizer = lib.deep.make_optimizer(
        **config['optimizer'],
        params=lib.deep.make_parameter_groups(model, custom_groups=rla_custom_groups),
    )
    rla_extra_paths_freeze_fraction = float(
        config.get('rla_extra_paths_freeze_fraction', 0.0)
    )
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
            if (
                isinstance(module, lib.deep.LinearEfficientEnsembleRankR)
                and module.rank > 1
            ):
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

    late_ensemble_loss_weight = float(config.get('late_ensemble_loss_weight', 0.0))
    late_ensemble_loss_start_fraction = float(
        config.get('late_ensemble_loss_start_fraction', 0.5)
    )
    if late_ensemble_loss_weight:
        assert 0.0 <= late_ensemble_loss_start_fraction <= 1.0
    late_ensemble_start_epoch = (
        math.ceil(config['n_epochs'] * late_ensemble_loss_start_fraction)
        if config['n_epochs'] > 0
        else math.ceil(2 * config['patience'] * late_ensemble_loss_start_fraction)
    )
    report['late_ensemble_loss'] = {
        'weight': late_ensemble_loss_weight,
        'start_fraction': late_ensemble_loss_start_fraction,
        'start_epoch': late_ensemble_start_epoch,
    }

    def ensemble_loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        if dataset.task.is_regression:
            return nn.functional.mse_loss(y_pred.mean(1), y_true)
        p_ens = y_pred.softmax(-1).mean(1)
        return nn.functional.nll_loss((p_ens + 1e-12).log(), y_true)

    def _set_model_share_training_batches(value: bool) -> bool:
        module = model.module if isinstance(model, nn.DataParallel) else model
        old_value = bool(module.share_training_batches)
        module.share_training_batches = value
        return old_value

    def training_loss_fn(
        y_pred: Tensor, y_true: Tensor, batch_idx: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, float]:
        individual_loss = loss_fn(y_pred, y_true)
        epoch = step // epoch_size
        lambda_t = (
            late_ensemble_loss_weight
            if late_ensemble_loss_weight and epoch >= late_ensemble_start_epoch
            else 0.0
        )
        if not lambda_t:
            zero = individual_loss.detach().new_tensor(0.0)
            return individual_loss, individual_loss.detach(), zero, 0.0

        if share_training_batches:
            ensemble_pred = y_pred
            ensemble_y = y_true
        else:
            ensemble_idx = batch_idx.reshape(-1, config['model']['k'])[:, 0]
            old_value = _set_model_share_training_batches(True)
            try:
                ensemble_pred = apply_model('train', ensemble_idx)
            finally:
                _set_model_share_training_batches(old_value)
            ensemble_y = Y_train[ensemble_idx]

        ensemble_loss = ensemble_loss_fn(ensemble_pred, ensemble_y)
        return (
            individual_loss + lambda_t * ensemble_loss,
            individual_loss.detach(),
            ensemble_loss.detach(),
            lambda_t,
        )

    # The following generator is used only for creating training batches,
    # so the random seed fully determines the sequence of training objects.
    batch_generator = torch.Generator(device).manual_seed(config['seed'])
    timer = delu.tools.Timer()
    early_stopping = delu.tools.EarlyStopping(config['patience'], mode='max')
    parameter_statistics = config.get('parameter_statistics', config['seed'] == 1)
    training_log = []
    writer = torch.utils.tensorboard.SummaryWriter(output)  # type: ignore[code]

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
    # Persist precision policy + reproducibility metadata for auditing.
    report['amp_enabled'] = amp_enabled
    report['amp_dtype'] = (
        'bfloat16' if amp_dtype is torch.bfloat16
        else 'float16' if amp_dtype is torch.float16
        else 'fp32'
    )
    report['seed'] = config['seed']
    # Try git first; fall back to baked-in GIT_COMMIT.txt for environments
    # without a `.git` dir (e.g. rsynced rented boxes).
    git_commit = 'unknown'
    try:
        import subprocess
        git_commit = (
            subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=Path(__file__).resolve().parent.parent.parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        # Fallback: look for GIT_COMMIT.txt at the repo root.
        for candidate in (
            Path(__file__).resolve().parent.parent.parent / 'GIT_COMMIT.txt',
            Path('/workspace/tabm/GIT_COMMIT.txt'),
        ):
            if candidate.exists():
                git_commit = candidate.read_text().strip()
                break
    report['git_commit'] = git_commit
    try:
        report['gpu_name'] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
        )
    except Exception:
        report['gpu_name'] = 'unknown'

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
        dict[PartKey, Any], dict[PartKey, np.ndarray], dict[PartKey, np.ndarray], int
    ]:
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
        return metrics, predictions, head_predictions, eval_batch_size

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
        epoch_losses = []
        epoch_individual_losses = []
        epoch_ensemble_losses = []
        epoch_late_ensemble_weights = []
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
            loss, individual_loss, ensemble_loss, lambda_t = training_loss_fn(
                apply_model('train', batch_idx), Y_train[batch_idx], batch_idx
            )
            # Fail-fast on non-finite loss. Without this guard, NaN losses
            # silently zero-out gradients and the run looks like a clean
            # exit despite producing garbage. Common cause: BF16 autocast
            # at high RLA rank with extra-path noise.
            if not torch.isfinite(loss):
                report['failure'] = {
                    'reason': 'non_finite_loss',
                    'step': step,
                    'amp_dtype': report.get('amp_dtype'),
                }
                lib.dump_report(output, report)
                raise RuntimeError(
                    f'Non-finite loss at step {step} '
                    f'(amp_dtype={report.get("amp_dtype")}). '
                    f'Aborting run; see report.json.failure.'
                )
            if grad_scaler is None:
                loss.backward()
            else:
                grad_scaler.scale(loss).backward()
            zero_rla_extra_path_grads()

            if parameter_statistics and (
                step % epoch_size == 0  # The first batch of the epoch.
                or step // epoch_size == 0  # The first epoch.
            ):
                for k, v in lib.deep.compute_parameter_stats(model).items():
                    writer.add_scalars(k, v, step, timer.elapsed())
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
            epoch_losses.append(loss.detach())
            epoch_individual_losses.append(individual_loss)
            epoch_ensemble_losses.append(ensemble_loss)
            epoch_late_ensemble_weights.append(lambda_t)

        epoch_losses = torch.stack(epoch_losses).tolist()
        mean_loss = statistics.mean(epoch_losses)
        mean_individual_loss = statistics.mean(
            torch.stack(epoch_individual_losses).tolist()
        )
        mean_ensemble_loss = statistics.mean(torch.stack(epoch_ensemble_losses).tolist())
        mean_late_ensemble_weight = statistics.mean(epoch_late_ensemble_weights)
        metrics, predictions, _, eval_batch_size = evaluate(
            ['val', 'test'], eval_batch_size
        )

        training_log.append(
            {
                'epoch-losses': epoch_losses,
                'epoch-individual-loss': mean_individual_loss,
                'epoch-ensemble-loss': mean_ensemble_loss,
                'late-ensemble-loss-weight': mean_late_ensemble_weight,
                'metrics': metrics,
                'time': timer.elapsed(),
            }
        )
        lib.print_metrics(mean_loss, metrics)
        writer.add_scalars(
            'loss',
            {
                'train': mean_loss,
                'individual': mean_individual_loss,
                'ensemble': mean_ensemble_loss,
            },
            step,
            timer.elapsed(),
        )
        for part in metrics:
            writer.add_scalars(
                'score', {part: metrics[part]['score']}, step, timer.elapsed()
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
    report['metrics'], predictions, head_predictions, _ = evaluate(
        ['train', 'val', 'test'], eval_batch_size
    )
    report['chunk_size'] = chunk_size
    report['eval_batch_size'] = eval_batch_size
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
