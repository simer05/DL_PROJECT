#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tomli
import toml


DEFAULT_FRACTION_CACHE_ROOT = Path('/mnt/ssd/users/prithvi/deepLearning/data/tabred_preprocessed_fractions')


def _fmt_fraction(x: float) -> str:
    return f'{x:.4f}'.rstrip('0').rstrip('.').replace('.', 'p')


def _rho_tag(x: float) -> str:
    return f"rho_{str(x).replace('-', 'm').replace('.', 'p')}"


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _sample_by_groups(groups: np.ndarray, n_keep: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_total = len(groups)
    if n_keep >= n_total:
        return np.arange(n_total, dtype=np.int64)

    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    target = counts.astype(np.float64) * (n_keep / float(n_total))
    alloc = np.floor(target).astype(np.int64)
    alloc = np.minimum(alloc, counts)

    remainder = int(n_keep - alloc.sum())
    if remainder > 0:
        frac = target - alloc
        order = np.argsort(-frac)
        for i in order:
            if remainder == 0:
                break
            if alloc[i] < counts[i]:
                alloc[i] += 1
                remainder -= 1
    elif remainder < 0:
        order = np.argsort(-(alloc - target))
        for i in order:
            if remainder == 0:
                break
            if alloc[i] > 0:
                alloc[i] -= 1
                remainder += 1

    selected: list[np.ndarray] = []
    for gi in range(len(uniq)):
        k = int(alloc[gi])
        if k <= 0:
            continue
        idx = np.where(inv == gi)[0]
        selected.append(rng.choice(idx, size=k, replace=False))

    out = np.sort(np.concatenate(selected).astype(np.int64)) if selected else np.array([], dtype=np.int64)
    if len(out) < n_keep:
        remaining = np.setdiff1d(np.arange(n_total, dtype=np.int64), out, assume_unique=False)
        extra = rng.choice(remaining, size=n_keep - len(out), replace=False)
        out = np.sort(np.concatenate([out, extra]).astype(np.int64))
    elif len(out) > n_keep:
        out = np.sort(rng.choice(out, size=n_keep, replace=False).astype(np.int64))
    return out


def _prepare_fraction_dataset(
    dataset_dir: Path,
    dataset: str,
    train_fraction: float,
    subset_seed: int,
    cache_root: Path,
    task_type: str | None,
) -> tuple[Path, dict]:
    if not (0.0 < train_fraction <= 1.0):
        raise ValueError(f'train_fraction must be in (0,1], got {train_fraction}')

    frac_tag = _fmt_fraction(train_fraction)
    subset_dir = cache_root / dataset / f'frac_{frac_tag}_seed_{subset_seed}'
    manifest_path = subset_dir / 'fraction_manifest.json'
    indices_path = subset_dir / 'selected_train_indices.npy'

    if subset_dir.exists() and manifest_path.exists() and indices_path.exists():
        manifest = json.loads(manifest_path.read_text())
        return subset_dir, manifest

    y_train = np.load(dataset_dir / 'Y_train.npy', mmap_mode='r')
    n_total = int(len(y_train))
    n_keep = max(1, int(round(n_total * train_fraction)))

    strategy = 'random'
    if task_type in {'binclass', 'multiclass'}:
        strategy = 'stratified_class'
        idx = _sample_by_groups(np.asarray(y_train), n_keep=n_keep, seed=subset_seed)
    elif task_type == 'regression':
        y_arr = np.asarray(y_train, dtype=np.float64)
        q = np.unique(np.quantile(y_arr, np.linspace(0.0, 1.0, 11)))
        if len(q) >= 3:
            strategy = 'stratified_quantile'
            bins = np.digitize(y_arr, q[1:-1], right=False)
            idx = _sample_by_groups(bins, n_keep=n_keep, seed=subset_seed)
        else:
            strategy = 'random_regression_fallback'
            rng = np.random.default_rng(subset_seed)
            idx = np.sort(rng.choice(n_total, size=n_keep, replace=False).astype(np.int64))
    else:
        rng = np.random.default_rng(subset_seed)
        idx = np.sort(rng.choice(n_total, size=n_keep, replace=False).astype(np.int64))

    subset_dir.mkdir(parents=True, exist_ok=True)
    np.save(indices_path, idx)

    all_npy = sorted(dataset_dir.glob('*.npy'))
    for p in all_npy:
        name = p.name
        if name.endswith('_train.npy'):
            arr = np.load(p)
            np.save(subset_dir / name, arr[idx])
        else:
            _link_or_copy(p, subset_dir / name)

    info_src = dataset_dir / 'info.json'
    if info_src.exists():
        info = json.loads(info_src.read_text())
    else:
        info = {}
    split_sizes = dict(info.get('split_sizes', {}))
    split_sizes['train'] = int(n_keep)
    info['split_sizes'] = split_sizes
    info['train_fraction'] = float(train_fraction)
    info['train_subsample_seed'] = int(subset_seed)
    info['selected_train_indices_file'] = str(indices_path)
    info['source_preprocessed_dir'] = str(dataset_dir)
    info['train_subset_strategy'] = strategy
    (subset_dir / 'info.json').write_text(json.dumps(info, indent=2))

    manifest = {
        'dataset': dataset,
        'source_dir': str(dataset_dir),
        'subset_dir': str(subset_dir),
        'train_fraction': float(train_fraction),
        'train_subsample_seed': int(subset_seed),
        'n_train_total': int(n_total),
        'n_train_selected': int(n_keep),
        'selected_train_indices_file': str(indices_path),
        'sampling_strategy': strategy,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return subset_dir, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--base-config', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--rho', type=float, required=True)
    parser.add_argument('--n-epochs', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--batch-size-override', type=int, default=0)
    parser.add_argument('--max-retries', type=int, default=1)
    parser.add_argument('--train-fraction', type=float, default=1.0)
    parser.add_argument('--train-subsample-seed', type=int, default=42)
    parser.add_argument('--fraction-cache-root', type=Path, default=DEFAULT_FRACTION_CACHE_ROOT)
    parser.add_argument('--esam-start-epoch', type=int, default=0)
    parser.add_argument('--esam-end-epoch', type=int, default=-1)
    parser.add_argument('--esam-start-fraction', type=float, default=None)
    parser.add_argument('--esam-end-fraction', type=float, default=None)
    parser.add_argument('--esam-rho-schedule', type=str, default='none')
    parser.add_argument('--esam-param-scope', type=str, default='adapter_only')
    parser.add_argument('--esam-mode', type=str, default='standard')
    parser.add_argument('--esam-selection-margin', type=float, default=0.0002)
    parser.add_argument('--safe-esam-policy', type=str, default='val_margin')
    parser.add_argument('--compact-screen', action='store_true')
    parser.add_argument('--run-tag', type=str, default='')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    cfg_path = Path(args.base_config)
    cfg = tomli.loads(cfg_path.read_text())

    data_root = Path(args.data_root).resolve()
    dataset_dir = data_root / args.dataset
    if not dataset_dir.exists():
        raise SystemExit(f'dataset dir missing: {dataset_dir}')
    info_path = dataset_dir / 'info.json'
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    task_type = info.get('task_type')

    effective_data_dir = dataset_dir
    subset_manifest = None
    if args.train_fraction < 1.0:
        effective_data_dir, subset_manifest = _prepare_fraction_dataset(
            dataset_dir,
            args.dataset,
            args.train_fraction,
            args.train_subsample_seed,
            args.fraction_cache_root.resolve(),
            task_type=task_type,
        )

    cfg['seed'] = args.seed
    if args.n_epochs is not None:
        cfg['n_epochs'] = args.n_epochs
    if args.patience is not None:
        cfg['patience'] = args.patience
    if args.batch_size_override > 0:
        cfg['batch_size'] = int(args.batch_size_override)

    cfg['data']['path'] = str(effective_data_dir)

    cfg['use_ncl'] = False
    cfg['lambda_ncl'] = 0.0
    cfg['ncl_space'] = 'logits'
    cfg['ncl_warmup_epochs'] = 0

    cfg['use_esam'] = args.rho > 0.0
    cfg['esam_rho'] = float(args.rho)
    cfg['esam_eps'] = 1e-12
    cfg['esam_adapter_only'] = True
    cfg['esam_memberwise'] = True
    cfg['esam_warmup_epochs'] = 0

    total_epochs = int(cfg.get('n_epochs', args.n_epochs or 0))
    if args.esam_start_fraction is not None:
        if not (0.0 <= args.esam_start_fraction <= 1.0):
            raise SystemExit('esam_start_fraction must be in [0,1]')
        start_epoch = int(round(total_epochs * args.esam_start_fraction))
    else:
        start_epoch = int(args.esam_start_epoch)
    cfg['esam_start_epoch'] = start_epoch

    if args.esam_end_fraction is not None:
        if not (0.0 <= args.esam_end_fraction <= 1.0):
            raise SystemExit('esam_end_fraction must be in [0,1]')
        end_epoch = int(round(total_epochs * args.esam_end_fraction)) - 1
    elif args.esam_end_epoch >= 0:
        end_epoch = int(args.esam_end_epoch)
    else:
        end_epoch = -1
    cfg['esam_end_epoch'] = end_epoch

    cfg['esam_log_diagnostics'] = True
    cfg['esam_diagnostics_every'] = 100

    frac_tag = _fmt_fraction(args.train_fraction)
    rho_tag = _rho_tag(args.rho)
    warm_tag = f'start_{start_epoch}'
    run_tag = args.run_tag.strip().replace(' ', '_') if args.run_tag else ''

    out_dir = Path(args.output_root) / args.dataset / f'trainfrac_{frac_tag}' / f'seed{args.seed}' / warm_tag / rho_tag
    if run_tag:
        out_dir = out_dir / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_gen = out_dir / 'config.generated.toml'
    conf_gen.write_text(toml.dumps(cfg))

    run_meta = {
        'dataset': args.dataset,
        'seed': int(args.seed),
        'rho': float(args.rho),
        'train_fraction': float(args.train_fraction),
        'train_subsample_seed': int(args.train_subsample_seed),
        'source_data_root': str(data_root),
        'source_dataset_dir': str(dataset_dir),
        'effective_data_dir': str(effective_data_dir),
        'esam_start_epoch': int(start_epoch),
        'esam_end_epoch': int(end_epoch),
        'run_tag': run_tag,
        'subset_manifest': subset_manifest,
        'share_training_batches': (cfg.get('model') or {}).get('share_training_batches'),
        'n_epochs_effective': int(cfg.get('n_epochs', 0)),
        'patience_effective': int(cfg.get('patience', 0)),
        'batch_size_effective': int(cfg.get('batch_size', 0)),
        'compact_screen': bool(args.compact_screen),
    }
    (out_dir / 'run_meta.json').write_text(json.dumps(run_meta, indent=2))

    cmd = ['python', 'bin/model.py', str(conf_gen), '--output', str(out_dir), '--force']
    print('[run]', ' '.join(cmd))
    if args.dry_run:
        return

    report = out_dir / 'report.json'
    if report.exists():
        print(f'[skip] exists: {report}')
        return

    rc = 1
    for _ in range(args.max_retries + 1):
        p = subprocess.run(cmd)
        rc = p.returncode
        if rc == 0 and report.exists():
            print(f'[ok] {out_dir}')
            return
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
