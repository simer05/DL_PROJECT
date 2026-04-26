from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

VARIANTS = ('softmax', 'l1norm', 'raw')
DEFAULT_TEACHER_NAMES: tuple[str, ...] = ('xgb', 'lgbm', 'cat')


def chunk_aggregate_r1(r1: Tensor, d_features: Sequence[int]) -> Tensor:
    if r1.dim() != 2:
        raise ValueError(f'r1 must be 2D, got shape {tuple(r1.shape)}')
    expected = int(sum(d_features))
    if r1.shape[-1] != expected:
        raise ValueError(
            f'r1 last dim {r1.shape[-1]} != sum(d_features)={expected}'
        )
    abs_r = r1.abs()
    chunks = abs_r.split(list(d_features), dim=-1)
    return torch.stack([c.sum(dim=-1) for c in chunks], dim=-1)


def cf_fisd_loss(
    r1: Tensor,
    teacher_importances: Mapping[str, Tensor],
    member_groups: Mapping[str, Sequence[int]],
    variant: str,
    d_features: Sequence[int],
) -> Tensor:
    if variant not in VARIANTS:
        raise ValueError(f'unknown variant: {variant!r}; expected one of {VARIANTS}')
    if not member_groups:
        return r1.new_zeros(())

    r1_mag = chunk_aggregate_r1(r1, d_features)
    losses: list[Tensor] = []
    for teacher_name, member_ids in member_groups.items():
        if not member_ids:
            continue
        if teacher_name not in teacher_importances:
            raise KeyError(f'teacher_importances missing {teacher_name!r}')
        t_imp = teacher_importances[teacher_name].to(r1.device, dtype=r1_mag.dtype)
        for m in member_ids:
            r_m = r1_mag[int(m)]
            if variant == 'softmax':
                pred = F.softmax(r_m, dim=-1)
                tgt = F.softmax(t_imp, dim=-1)
            elif variant == 'l1norm':
                pred = r_m / (r_m.sum() + 1e-8)
                tgt = t_imp / (t_imp.sum() + 1e-8)
            else:
                tgt = t_imp / (t_imp.sum() + 1e-8) * r_m.sum().detach()
                pred = r_m
            losses.append(F.mse_loss(pred, tgt, reduction='mean'))
    if not losses:
        return r1.new_zeros(())
    return torch.stack(losses).mean()


def alignment_cosine(
    r1: Tensor,
    teacher_imp: Tensor,
    member_ids: Sequence[int],
    d_features: Sequence[int],
) -> Tensor:
    r1_soft = F.softmax(chunk_aggregate_r1(r1, d_features), dim=-1)
    t_soft = F.softmax(teacher_imp.to(r1.device, dtype=r1_soft.dtype), dim=-1)
    ids = list(member_ids)
    return F.cosine_similarity(
        r1_soft[ids], t_soft.unsqueeze(0).expand(len(ids), -1), dim=-1
    )


def load_teacher_importances(
    teacher_dir: str | Path,
    dataset_name: str,
    n_features: int,
    teacher_names: Sequence[str] = DEFAULT_TEACHER_NAMES,
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    root = Path(teacher_dir)
    for name in teacher_names:
        f = root / f'{name}.npy'
        if not f.exists():
            f_alt = root / f'{dataset_name}_{name}.npy'
            if f_alt.exists():
                f = f_alt
            else:
                raise FileNotFoundError(f)
        v = np.load(f).astype(np.float32)
        if v.shape != (n_features,):
            raise ValueError(
                f'{f}: expected shape ({n_features},), got {v.shape}'
            )
        v = np.clip(v, 0.0, None)
        s = float(v.sum())
        if s > 0.0:
            v = v / s
        else:
            v = np.full_like(v, 1.0 / n_features)
        out[name] = torch.from_numpy(v)
    return out


def default_member_groups(
    k: int,
    teacher_names: Sequence[str] = DEFAULT_TEACHER_NAMES,
) -> dict[str, list[int]]:
    n = len(teacher_names)
    if n == 0:
        raise ValueError('teacher_names must not be empty')
    base = k // n
    extra = k - base * n
    groups: dict[str, list[int]] = {}
    cursor = 0
    for i, name in enumerate(teacher_names):
        size = base + (1 if i < extra else 0)
        groups[name] = list(range(cursor, cursor + size))
        cursor += size
    return groups
