import itertools
from typing import Any, Literal

import rtdl_num_embeddings
import rtdl_revisiting_models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter


# ======================================================================================
# Initialization
# ======================================================================================
def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor:
    assert d > 0
    d_rsqrt = d**-0.5
    return nn.init.uniform_(x, -d_rsqrt, d_rsqrt)


@torch.inference_mode()
def init_random_signs_(x: Tensor) -> Tensor:
    return x.bernoulli_(0.5).mul_(2).add_(-1)


# ======================================================================================
# Modules
# ======================================================================================
class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class NLinear(nn.Module):
    """A stack of N linear layers. Each layer is applied to its own part of the input.

    **Shape**

    - Input: ``(B, N, in_features)``
    - Output: ``(B, N, out_features)``

    The i-th linear layer is applied to the i-th matrix of the shape (B, in_features).

    Technically, this is a simplified version of delu.nn.NLinear:
    https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html.
    The difference is that this layer supports only 3D inputs
    with exactly one batch dimension. By contrast, delu.nn.NLinear supports
    any number of batch dimensions.
    """

    def __init__(
        self, n: int, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.bias = Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        d = self.weight.shape[-2]
        init_rsqrt_uniform_(self.weight, d)
        if self.bias is not None:
            init_rsqrt_uniform_(self.bias, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]

        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class PiecewiseLinearEmbeddings(rtdl_num_embeddings.PiecewiseLinearEmbeddings):
    """
    This class simply adds the default values for `activation` and `version`.
    """

    def __init__(
        self,
        *args,
        activation: bool = False,
        version: None | Literal['A', 'B'] = 'B',
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs, activation=activation, version=version)


class OneHotEncoding0d(nn.Module):
    # Input:  (*, n_cat_features=len(cardinalities))
    # Output: (*, sum(cardinalities))

    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self._cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1
        assert x.shape[-1] == len(self._cardinalities)

        return torch.cat(
            [
                # NOTE
                # This is a quick hack to support out-of-vocabulary categories.
                #
                # Recall that lib.data.transform_cat encodes categorical features
                # as follows:
                # - In-vocabulary values receive indices from `range(cardinality)`.
                # - All out-of-vocabulary values (i.e. new categories in validation
                #   and test data that are not presented in the training data)
                #   receive the index `cardinality`.
                #
                # As such, the line below will produce the standard one-hot encoding for
                # known categories, and the all-zeros encoding for unknown categories.
                # This may not be the best approach to deal with unknown values,
                # but should be enough for our purposes.
                F.one_hot(x[..., i], cardinality + 1)[..., :-1]
                for i, cardinality in enumerate(self._cardinalities)
            ],
            -1,
        )


class ScaleEnsemble(nn.Module):
    def __init__(
        self,
        k: int,
        d: int,
        *,
        init: Literal['ones', 'normal', 'random-signs'],
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, d))
        self._weight_init = init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self._weight_init == 'ones':
            nn.init.ones_(self.weight)
        elif self._weight_init == 'normal':
            nn.init.normal_(self.weight)
        elif self._weight_init == 'random-signs':
            init_random_signs_(self.weight)
        else:
            raise ValueError(f'Unknown weight_init: {self._weight_init}')

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 2
        return x * self.weight


class LinearEfficientEnsemble(nn.Module):
    """
    This layer is a more configurable version of the "BatchEnsemble" layer
    from the paper
    "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning"
    (link: https://arxiv.org/abs/2002.06715).

    First, this layer allows to select only some of the "ensembled" parts:
    - the input scaling  (r_i in the BatchEnsemble paper)
    - the output scaling (s_i in the BatchEnsemble paper)
    - the output bias    (not mentioned in the BatchEnsemble paper,
                          but is presented in public implementations)

    Second, the initialization of the scaling weights is configurable
    through the `scaling_init` argument.

    NOTE
    The term "adapter" is used in the TabM paper only to tell the story.
    The original BatchEnsemble paper does NOT use this term. So this class also
    avoids the term "adapter".
    """

    r: None | Tensor
    s: None | Tensor
    bias: None | Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        k: int,
        ensemble_scaling_in: bool,
        ensemble_scaling_out: bool,
        ensemble_bias: bool,
        scaling_init: Literal['ones', 'random-signs'],
    ):
        assert k > 0
        if ensemble_bias:
            assert bias
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_parameter(
            'r',
            (
                nn.Parameter(torch.empty(k, in_features))
                if ensemble_scaling_in
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            's',
            (
                nn.Parameter(torch.empty(k, out_features))
                if ensemble_scaling_out
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            'bias',
            (
                nn.Parameter(torch.empty(out_features))  # type: ignore[code]
                if bias and not ensemble_bias
                else nn.Parameter(torch.empty(k, out_features))
                if ensemble_bias
                else None
            ),
        )

        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.scaling_init = scaling_init

        self.reset_parameters()

    def reset_parameters(self):
        init_rsqrt_uniform_(self.weight, self.in_features)
        scaling_init_fn = {'ones': nn.init.ones_, 'random-signs': init_random_signs_}[
            self.scaling_init
        ]
        if self.r is not None:
            scaling_init_fn(self.r)
        if self.s is not None:
            scaling_init_fn(self.s)
        if self.bias is not None:
            bias_init = torch.empty(
                # NOTE: the shape of bias_init is (out_features,) not (k, out_features).
                # It means that all biases have the same initialization.
                # This is similar to having one shared bias plus
                # k zero-initialized non-shared biases.
                self.out_features,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            bias_init = init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, K, D)
        assert x.ndim == 3

        # >>> The equation (5) from the BatchEnsemble paper (arXiv v2).
        if self.r is not None:
            x = x * self.r
        x = x @ self.weight.T
        if self.s is not None:
            x = x * self.s
        # <<<

        if self.bias is not None:
            x = x + self.bias
        return x


class LinearEfficientEnsembleRankR(nn.Module):
    """Rank-r generalization of `LinearEfficientEnsemble`.

    Replaces the per-member rank-1 outer-product adapter `s_i r_i^T`
    with a rank-r factorization `S_i R_i^T`, where
    `R_i in R^{d_in x rank}` and `S_i in R^{d_out x rank}`.

    The per-member weight is `W_i = W ⊙ (S_i R_i^T)`. Equivalently,
    the rank-r forward is `r` parallel rank-1 paths sharing `W`,
    summed before the bias is added:

        y_i = sum_{j=1..rank} s_{i,j} ⊙ (W (r_{i,j} ⊙ x)) + b_i

    At ``rank == 1`` and ``additive == False`` this reduces exactly to
    `LinearEfficientEnsemble` with both ``ensemble_scaling_in`` and
    ``ensemble_scaling_out`` set to True.

    The ``additive`` flag switches the construction to an *additive*
    LoRA-style correction `W_i = W + S_i R_i^T` for the defensive ablation
    in Section 6.3 of the RLA spec. In this mode the per-member weight
    starts equal to ``W`` because R/S are zero-initialized.
    """

    bias: None | Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        k: int,
        rank: int = 1,
        scaling_init: Literal['ones', 'random-signs'] = 'ones',
        additive: bool = False,
        init_mode: Literal['variance_preserving', 'base_preserving'] = 'variance_preserving',
        base_preserve_noise: float = 1e-3,
    ) -> None:
        """
        ``init_mode`` selects how the rank-r adapter is initialised:

        * ``variance_preserving`` (default, original behaviour): every
          column of R and S is filled with ``1/sqrt(r)`` so the sum over
          rank paths preserves the variance of the rank-1 baseline.
          For first-layer ``random-signs`` scaling, each R column gets
          independent random signs scaled by ``1/sqrt(r)``. The forward
          output is *not* bit-identical to the rank-1 baseline at rank>1
          (the per-path random signs interact differently).

        * ``base_preserving``: rank path 0 is initialised exactly as the
          rank-1 baseline (R[:,:,0]=1 or random-signs, S[:,:,0]=1).
          Extra rank paths j>=1 are initialised with S[:,:,j]=0 and
          R[:,:,j]=tiny noise (std=``base_preserve_noise``). At init the
          extra paths contribute zero to the forward output, so a
          base-preserving rank-r RLA layer produces *bit-identical*
          forward outputs to the baseline (rank-1) layer. After one
          optimiser step S becomes non-zero and R picks up gradient.
        """
        assert k > 0
        assert rank >= 1
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.rank = rank
        self.scaling_init = scaling_init
        self.additive = additive
        self.init_mode = init_mode
        self.base_preserve_noise = base_preserve_noise

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        # R: (k, d_in, rank), S: (k, d_out, rank)
        self.R = nn.Parameter(torch.empty(k, in_features, rank))
        self.S = nn.Parameter(torch.empty(k, out_features, rank))
        self.register_parameter(
            'bias',
            nn.Parameter(torch.empty(k, out_features)) if bias else None,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_rsqrt_uniform_(self.weight, self.in_features)

        if self.additive:
            # Additive LoRA-style start at the baseline weight: zero S, normal R.
            nn.init.zeros_(self.S)
            init_rsqrt_uniform_(self.R, self.in_features)
        elif self.init_mode == 'base_preserving':
            # Path 0 reproduces the rank-1 baseline exactly. Extra paths
            # are zero-gated on the S side (so they contribute nothing to
            # the forward output at init) and receive tiny noise on the
            # R side (so the next step gives S non-zero gradient).
            #
            # NOTE on RNG ordering: to keep the shared weight and bias
            # element-identical to the baseline class under the same seed,
            # the only RNG-consuming step inside the adapter init must
            # match what the baseline does (ones-fill = no RNG). The
            # extra-path noise *would* break that match, so we defer it
            # until after the bias has been drawn (see below).
            with torch.inference_mode():
                # Path 0 of S: deterministic 1.0 (matches baseline 's=ones').
                self.S.zero_()
                self.S[:, :, 0].fill_(1.0)
                # Path 0 of R: depends on scaling_init.
                self.R.zero_()
                if self.scaling_init == 'random-signs':
                    # Note: this draws RNG, just like the baseline class
                    # does for its own r=random-signs init, so equivalent
                    # baseline + base_preserving construction stays in sync.
                    sign_path = torch.empty_like(self.R[:, :, 0])
                    init_random_signs_(sign_path)
                    self.R[:, :, 0].copy_(sign_path)
                elif self.scaling_init == 'ones':
                    self.R[:, :, 0].fill_(1.0)
                else:
                    raise ValueError(f'Unknown scaling_init: {self.scaling_init}')
        elif self.init_mode == 'variance_preserving':
            # Variance-preserving (original) form. Match the rank-1
            # baseline exactly only at rank=1.
            scale = self.rank**-0.5
            if self.scaling_init == 'random-signs':
                init_random_signs_(self.R)
                self.R.data.mul_(scale)
                with torch.inference_mode():
                    self.S.fill_(scale)
            elif self.scaling_init == 'ones':
                with torch.inference_mode():
                    self.R.fill_(scale)
                    self.S.fill_(scale)
            else:
                raise ValueError(f'Unknown scaling_init: {self.scaling_init}')
        else:
            raise ValueError(f'Unknown init_mode: {self.init_mode}')

        if self.bias is not None:
            bias_init = torch.empty(
                self.out_features,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            bias_init = init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

        # Deferred extra-path noise for base_preserving.
        #
        # CRITICAL: this draw must NOT advance the global RNG, otherwise
        # downstream layers in the same model will be initialised at a
        # different RNG state than the baseline class would have used,
        # breaking full-model bit-equivalence at rank 1 / rank > 1.
        # We isolate the noise draw by save+restore of the global RNG
        # state and use a separate Generator seeded deterministically
        # from a hash of the parameter shape (so the noise pattern is
        # reproducible across runs without consuming the main RNG).
        if (
            self.init_mode == 'base_preserving'
            and not self.additive
            and self.rank > 1
            and self.base_preserve_noise > 0
        ):
            # Deterministic per-layer seed: stable cross-process hash of
            # the shape tuple via hashlib (Python's built-in hash() is
            # process-salted by PYTHONHASHSEED, so identical configs would
            # otherwise initialise the extra-path noise differently across
            # processes; we want bitwise reproducibility).
            import hashlib
            shape_key = (
                f'rla_base_preserve_noise|'
                f'{self.in_features}|{self.out_features}|{self.k}|{self.rank}'
            ).encode('utf-8')
            seed = (
                int.from_bytes(hashlib.sha256(shape_key).digest()[:4], 'big')
                & 0x7FFFFFFF
            )
            gen = torch.Generator(device=self.R.device)
            gen.manual_seed(seed)
            noise = torch.empty_like(self.R[:, :, 1:])
            noise.normal_(mean=0.0, std=self.base_preserve_noise, generator=gen)
            with torch.inference_mode():
                self.R[:, :, 1:].copy_(noise)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, K, D_in)
        assert x.ndim == 3
        # Backbone projection (shared across rank paths and members).
        Wx = x @ self.weight.T  # (B, K, D_out)

        if self.additive:
            # W_i x = W x + S_i (R_i^T x). Sum the rank-r correction.
            # x: (B,K,Din)  R: (K,Din,r)  -> z: (B,K,r) per member
            z = torch.einsum('bki,kir->bkr', x, self.R)
            # S: (K, Dout, r) -> correction: (B,K,Dout)
            corr = torch.einsum('bkr,kor->bko', z, self.S)
            y = Wx + corr
        else:
            # Multiplicative: y = sum_j S[:,:,j] ⊙ (W (R[:,:,j] ⊙ x)).
            # Loop over rank (rank ≤ 8, GEMM-bound; loop overhead negligible).
            y = x.new_zeros(x.shape[0], x.shape[1], self.out_features)
            for j in range(self.rank):
                # x_scaled: (B,K,Din) — element-wise scale by R[:,:,j] (K,Din)
                x_scaled = x * self.R[:, :, j]
                proj = x_scaled @ self.weight.T  # (B,K,Dout)
                y = y + proj * self.S[:, :, j]   # broadcast (K,Dout) -> (B,K,Dout)

        if self.bias is not None:
            y = y + self.bias
        return y


def make_efficient_ensemble(module: nn.Module, EnsembleLayer, **kwargs) -> None:
    """Replace linear layers with efficient ensembles of linear layers.

    NOTE
    In the paper, there are no experiments with networks with normalization layers.
    Perhaps, their trainable weights (the affine transformations) also need
    "ensemblification" as in the paper about "FiLM-Ensemble".
    Additional experiments are required to make conclusions.
    """
    for name, submodule in list(module.named_children()):
        if isinstance(submodule, nn.Linear):
            module.add_module(
                name,
                EnsembleLayer(
                    in_features=submodule.in_features,
                    out_features=submodule.out_features,
                    bias=submodule.bias is not None,
                    **kwargs,
                ),
            )
        else:
            make_efficient_ensemble(submodule, EnsembleLayer, **kwargs)


class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: None | int = None,
        d_out: None | int = None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str = 'ReLU',
    ) -> None:
        super().__init__()

        d_first = d_block if d_in is None else d_in
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_first if i == 0 else d_block, d_block),
                    getattr(nn, activation)(),
                    nn.Dropout(dropout),
                )
                for i in range(n_blocks)
            ]
        )
        self.output = None if d_out is None else nn.Linear(d_block, d_out)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        if self.output is not None:
            x = self.output(x)
        return x


_CUSTOM_MODULES = {
    # https://docs.python.org/3/library/stdtypes.html#definition.__name__
    CustomModule.__name__: CustomModule
    for CustomModule in [
        rtdl_num_embeddings.LinearEmbeddings,
        rtdl_num_embeddings.LinearReLUEmbeddings,
        rtdl_num_embeddings.PeriodicEmbeddings,
        PiecewiseLinearEmbeddings,
        MLP,
    ]
}


def make_module(type: str, *args, **kwargs) -> nn.Module:
    Module = getattr(nn, type, None)
    if Module is None:
        Module = _CUSTOM_MODULES[type]
    return Module(*args, **kwargs)


def get_n_parameters(m: nn.Module):
    return sum(x.numel() for x in m.parameters() if x.requires_grad)


@torch.inference_mode()
def compute_parameter_stats(module: nn.Module) -> dict[str, dict[str, float]]:
    stats = {'norm': {}, 'gradnorm': {}, 'gradratio': {}}
    for name, parameter in module.named_parameters():
        stats['norm'][name] = parameter.norm().item()
        if parameter.grad is not None:
            stats['gradnorm'][name] = parameter.grad.norm().item()
            # Avoid computing statistics for zero-initialized parameters.
            if (parameter.abs() > 1e-6).any():
                stats['gradratio'][name] = (
                    (parameter.grad.abs() / parameter.abs().clamp_min_(1e-6))
                    .mean()
                    .item()
                )
    stats['norm']['model'] = (
        torch.cat([x.flatten() for x in module.parameters()]).norm().item()
    )
    stats['gradnorm']['model'] = (
        torch.cat([x.grad.flatten() for x in module.parameters() if x.grad is not None])
        .norm()
        .item()
    )
    return stats


# ======================================================================================
# Optimization
# ======================================================================================
def default_zero_weight_decay_condition(
    module_name: str, module: nn.Module, parameter_name: str, parameter: Parameter
):
    from rtdl_num_embeddings import _Periodic

    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        nn.BatchNorm1d
        | nn.LayerNorm
        | nn.InstanceNorm1d
        | rtdl_revisiting_models.LinearEmbeddings
        | rtdl_num_embeddings.LinearEmbeddings
        | rtdl_num_embeddings.LinearReLUEmbeddings
        | _Periodic,
    )


def make_parameter_groups(
    module: nn.Module,
    zero_weight_decay_condition=default_zero_weight_decay_condition,
    custom_groups: None | list[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    if custom_groups is None:
        custom_groups = []
    custom_params = frozenset(
        itertools.chain.from_iterable(group['params'] for group in custom_groups)
    )
    assert len(custom_params) == sum(
        len(group['params']) for group in custom_groups
    ), 'Parameters in custom_groups must not intersect'
    zero_wd_params = frozenset(
        p
        for mn, m in module.named_modules()
        for pn, p in m.named_parameters()
        if p not in custom_params and zero_weight_decay_condition(mn, m, pn, p)
    )
    default_group = {
        'params': [
            p
            for p in module.parameters()
            if p not in custom_params and p not in zero_wd_params
        ]
    }
    return [
        default_group,
        {'params': list(zero_wd_params), 'weight_decay': 0.0},
        *custom_groups,
    ]


def make_optimizer(type: str, **kwargs) -> torch.optim.Optimizer:
    Optimizer = getattr(torch.optim, type)
    return Optimizer(**kwargs)
