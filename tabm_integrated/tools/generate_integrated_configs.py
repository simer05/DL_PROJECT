
from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path
from typing import Any

import tomli_w

PAPER = Path(__file__).resolve().parents[1] / 'paper'
EXP_ROOT = PAPER / 'exp' / 'integrated'
TEACHER_ROOT = Path('../../cf_fisd_recovered/paper/exp/cf_fisd/_teachers/tabred')

DATASET_ORDER = [
    'sberbank-housing',
    'ecom-offers',
    'homesite-insurance',
    'cooking-time',
    'delivery-eta',
]

OFFICIAL_BASE: dict[str, dict[str, Any]] = {
    'sberbank-housing': {
        'batch_size': 256,
        'patience': 16,
        'n_epochs': -1,
        'gradient_clipping_norm': 1.0,
        'amp': True,
        'data': {'cache': True, 'path': 'data/sberbank-housing', 'num_policy': 'noisy-quantile', 'cat_policy': 'ordinal'},
        'optimizer': {'lr': 0.0008922700423431547, 'weight_decay': 0.00021333759467820313},
        'model': {
            'arch_type': 'tabm', 'k': 32, 'share_training_batches': False,
            'backbone': {'n_blocks': 3, 'd_block': 256, 'dropout': 0.3128700072370906},
            'num_embeddings': {'type': 'PiecewiseLinearEmbeddings', 'd_embedding': 28},
        },
        'bins': {'n_bins': 84},
    },
    'ecom-offers': {
        'batch_size': 1024,
        'patience': 16,
        'n_epochs': -1,
        'gradient_clipping_norm': 1.0,
        'amp': True,
        'data': {'cache': True, 'path': 'data/ecom-offers', 'num_policy': 'noisy-quantile'},
        'optimizer': {'lr': 0.00024262819114537424, 'weight_decay': 0.0001501852317298042},
        'model': {
            'arch_type': 'tabm', 'k': 32, 'share_training_batches': False,
            'backbone': {'n_blocks': 1, 'd_block': 960, 'dropout': 0.0},
            'num_embeddings': {'type': 'PiecewiseLinearEmbeddings', 'd_embedding': 16},
        },
        'bins': {'n_bins': 47},
    },
    'homesite-insurance': {
        'batch_size': 1024,
        'patience': 16,
        'n_epochs': -1,
        'gradient_clipping_norm': 1.0,
        'amp': True,
        'data': {'cache': True, 'path': 'data/homesite-insurance', 'num_policy': 'noisy-quantile', 'cat_policy': 'ordinal'},
        'optimizer': {'lr': 0.0018580623030886075, 'weight_decay': 0.0001614529849348179},
        'model': {
            'arch_type': 'tabm', 'k': 32, 'share_training_batches': False,
            'backbone': {'n_blocks': 3, 'd_block': 704, 'dropout': 0.0},
            'num_embeddings': {'type': 'PiecewiseLinearEmbeddings', 'd_embedding': 8},
        },
        'bins': {'n_bins': 15},
    },
    'cooking-time': {
        'batch_size': 1024,
        'patience': 16,
        'n_epochs': -1,
        'gradient_clipping_norm': 1.0,
        'amp': True,
        'data': {'cache': True, 'path': 'data/cooking-time', 'cat_policy': 'ordinal'},
        'optimizer': {'lr': 0.00012065020494450812, 'weight_decay': 0.07000081679295954},
        'model': {
            'arch_type': 'tabm', 'k': 32, 'share_training_batches': False,
            'backbone': {'n_blocks': 1, 'd_block': 416, 'dropout': 0.0},
            'num_embeddings': {'type': 'PiecewiseLinearEmbeddings', 'd_embedding': 16},
        },
        'bins': {'n_bins': 8},
    },
    'delivery-eta': {
        'batch_size': 1024,
        'patience': 16,
        'n_epochs': -1,
        'gradient_clipping_norm': 1.0,
        'amp': True,
        'data': {'cache': True, 'path': 'data/delivery-eta', 'cat_policy': 'ordinal'},
        'optimizer': {'lr': 0.0025639267063470926, 'weight_decay': 0.0507074044872309},
        'model': {
            'arch_type': 'tabm', 'k': 32, 'share_training_batches': False,
            'backbone': {'n_blocks': 2, 'd_block': 752, 'dropout': 0.18437573041970334},
            'num_embeddings': {'type': 'PiecewiseLinearEmbeddings', 'd_embedding': 32},
        },
        'bins': {'n_bins': 81},
    },
}

MODULE_DEFAULTS = {
    'rla': {'rank': 4, 'noise': 1e-4, 'inference': 'mean'},
    'esam': {'rho': 0.005},
    'mfb': {'keep': 0.90},
    'cf_fisd': {'lambda': 0.10},
}


def base_config(dataset: str, seed: int) -> dict[str, Any]:
    cfg = copy.deepcopy(OFFICIAL_BASE[dataset])
    cfg['seed'] = int(seed)
    cfg['head_selection'] = True
    return cfg


def apply_rla(cfg: dict[str, Any], *, rank: int = 4, noise: float = 1e-4) -> None:
    cfg['model']['rla_rank'] = int(rank)
    cfg['model']['rla_first_only'] = False
    cfg['model']['rla_additive'] = False
    cfg['model']['rla_init'] = 'base_preserving'
    cfg['model']['rla_base_preserve_noise'] = float(noise)
    cfg['rla_adapter_lr_multiplier'] = 1.0
    cfg['rla_extra_paths_freeze_fraction'] = 0.0


def apply_esam(cfg: dict[str, Any], *, rho: float = 0.005) -> None:
    cfg['use_esam'] = True
    cfg['esam_rho'] = float(rho)
    cfg['esam_eps'] = 1e-12
    cfg['esam_adapter_only'] = True
    cfg['esam_memberwise'] = True
    cfg['esam_warmup_epochs'] = 0
    cfg['esam_start_epoch'] = 0
    cfg['esam_end_epoch'] = -1
    cfg['esam_log_diagnostics'] = False
    cfg['esam_diagnostics_every'] = 100


def apply_mfb(cfg: dict[str, Any], *, keep: float = 0.90) -> None:
    cfg['model']['mfb'] = {
        'enabled': True,
        'mask_mode': 'member_fixed',
        'mask_granularity': 'feature_group',
        'keep_rate': float(keep),
        'training_only': False,
        'inverted_scaling': True,
        'use_soft_mask': False,
        'mask_strength': 1.0,
        'anchor_fraction': 0.0,
        'warmup_epochs': 0,
        'mask_seed': int(cfg['seed']),
    }


def apply_cf_fisd(cfg: dict[str, Any], dataset: str, *, lam: float = 0.10) -> None:
    cfg['cf_fisd'] = {
        'lambda': float(lam),
        'variant': 'raw',
        'dataset_name': dataset,
        'teacher_dir': str(TEACHER_ROOT / dataset),
        'teacher_names': ['xgb', 'lgbm', 'cat'],
    }


def make_variant_config(dataset: str, seed: int, variant: str) -> dict[str, Any]:
    cfg = base_config(dataset, seed)
    if variant == 'baseline_plr':
        return cfg
    if variant.startswith('rla_rank'):
        parts = variant.split('_')
        apply_rla(cfg, rank=int(parts[1].removeprefix('rank')), noise=float(parts[2].removeprefix('noise')))
        return cfg
    if variant.startswith('esam_rho'):
        apply_esam(cfg, rho=float(variant.removeprefix('esam_rho')))
        return cfg
    if variant.startswith('mfb_keep'):
        apply_mfb(cfg, keep=float(variant.removeprefix('mfb_keep')))
        return cfg
    if variant.startswith('cf_fisd_lambda'):
        apply_cf_fisd(cfg, dataset, lam=float(variant.removeprefix('cf_fisd_lambda')))
        return cfg

    modules: set[str]
    if variant == 'rla_only':
        modules = {'rla'}
    elif variant == 'esam_only':
        modules = {'esam'}
    elif variant == 'mfb_only':
        modules = {'mfb'}
    elif variant == 'cf_fisd_only':
        modules = {'cf_fisd'}
    elif variant == 'all_four_combined':
        modules = {'rla', 'esam', 'mfb', 'cf_fisd'}
    elif variant == 'rla_esam':
        modules = {'rla', 'esam'}
    elif variant == 'rla_mfb':
        modules = {'rla', 'mfb'}
    elif variant == 'rla_cf_fisd':
        modules = {'rla', 'cf_fisd'}
    elif variant == 'esam_mfb':
        modules = {'esam', 'mfb'}
    elif variant == 'esam_cf_fisd':
        modules = {'esam', 'cf_fisd'}
    elif variant == 'mfb_cf_fisd':
        modules = {'mfb', 'cf_fisd'}
    elif variant == 'all_minus_rla':
        modules = {'esam', 'mfb', 'cf_fisd'}
    elif variant == 'all_minus_esam':
        modules = {'rla', 'mfb', 'cf_fisd'}
    elif variant == 'all_minus_mfb':
        modules = {'rla', 'esam', 'cf_fisd'}
    elif variant == 'all_minus_cf_fisd':
        modules = {'rla', 'esam', 'mfb'}
    else:
        raise KeyError(variant)
    if 'rla' in modules:
        apply_rla(cfg, rank=MODULE_DEFAULTS['rla']['rank'], noise=MODULE_DEFAULTS['rla']['noise'])
    if 'esam' in modules:
        apply_esam(cfg, rho=MODULE_DEFAULTS['esam']['rho'])
    if 'mfb' in modules:
        apply_mfb(cfg, keep=MODULE_DEFAULTS['mfb']['keep'])
    if 'cf_fisd' in modules:
        apply_cf_fisd(cfg, dataset, lam=MODULE_DEFAULTS['cf_fisd']['lambda'])
    return cfg


def write_config(wave: str, dataset: str, variant: str, seed: int, *, force: bool) -> Path:
    cfg = make_variant_config(dataset, seed, variant)
    path = EXP_ROOT / wave / dataset / f'{variant}-evaluation' / f'{seed}.toml'
    path.parent.mkdir(parents=True, exist_ok=True)
    if force or not path.exists():
        path.write_text(tomli_w.dumps(cfg))
    return path


def write_manifest(name: str, paths: list[Path]) -> Path:
    manifest = EXP_ROOT / f'manifest_{name}.txt'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('\n'.join(str(p.relative_to(PAPER)) for p in paths) + ('\n' if paths else ''))
    return manifest


def configs_for_stage(stage: str) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    if stage in {'initial', 'baseline_fidelity'}:
        out['baseline_fidelity'] = [write_config('baseline_fidelity', d, 'baseline_plr', 0, force=True) for d in DATASET_ORDER]
    if stage in {'initial', 'smoke'}:
        variants = ['baseline_plr', 'rla_only', 'esam_only', 'mfb_only', 'cf_fisd_only', 'all_four_combined']
        out['smoke'] = [write_config('smoke', d, v, 0, force=True) for d in DATASET_ORDER for v in variants]
    if stage in {'initial', 'sweeps'}:
        variants: list[str] = []
        for rank in [2, 4, 8]:
            for noise in [1e-3, 1e-4, 1e-5]:
                variants.append(f'rla_rank{rank}_noise{noise:g}')
        variants += [f'esam_rho{rho:g}' for rho in [0.001, 0.0025, 0.005, 0.01]]
        variants += [f'mfb_keep{keep:g}' for keep in [0.70, 0.80, 0.90, 0.95]]
        variants += [f'cf_fisd_lambda{lam:g}' for lam in [0.05, 0.10, 0.20]]
        variants += ['rla_esam', 'rla_mfb', 'rla_cf_fisd', 'esam_mfb', 'esam_cf_fisd', 'mfb_cf_fisd', 'all_four_combined', 'all_minus_rla', 'all_minus_esam', 'all_minus_mfb', 'all_minus_cf_fisd']
        out['sweeps'] = [write_config('sweeps', d, v, 0, force=True) for d in DATASET_ORDER for v in variants]
    return out


def verify_baseline_configs(paths: list[Path]) -> None:
    import tomllib
    for path in paths:
        dataset = path.parts[-3]
        cfg = tomllib.loads(path.read_text())
        expected = copy.deepcopy(OFFICIAL_BASE[dataset])
        for key, value in expected.items():
            if cfg.get(key) != value:
                raise AssertionError(f'{path}: {key} differs from official template')
        if cfg['seed'] != 0 or cfg.get('head_selection') is not True:
            raise AssertionError(f'{path}: seed/head_selection wrapper invalid')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['initial', 'baseline_fidelity', 'smoke', 'sweeps'], default='initial')
    parser.add_argument('--clean', action='store_true')
    args = parser.parse_args()
    if args.clean and EXP_ROOT.exists():
        shutil.rmtree(EXP_ROOT)
    generated = configs_for_stage(args.stage)
    for name, paths in generated.items():
        manifest = write_manifest(name, paths)
        print(f'{name}: {len(paths)} configs -> {manifest.relative_to(PAPER)}')
    if 'baseline_fidelity' in generated:
        verify_baseline_configs(generated['baseline_fidelity'])
        print('baseline_fidelity config check: OK')


if __name__ == '__main__':
    main()
