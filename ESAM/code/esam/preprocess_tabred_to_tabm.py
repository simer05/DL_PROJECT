#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover
    train_test_split = None


def _load_table(raw_root: Path, spec: dict[str, Any]) -> pd.DataFrame:
    src = raw_root / spec['train_file']
    fmt = spec['format']
    if fmt == 'zip_csv':
        with zipfile.ZipFile(src) as zf:
            names = [n for n in zf.namelist() if n.endswith('.csv')]
            if not names:
                raise RuntimeError(f'No csv found inside {src}')
            with zf.open(names[0]) as f:
                return pd.read_csv(f)
    if fmt == 'csv_gz':
        with gzip.open(src, mode='rt', newline='') as f:
            return pd.read_csv(f)
    if fmt == 'parquet':
        return pd.read_parquet(src)
    raise ValueError(f'Unsupported format: {fmt}')


def _write_schema_debug(path: Path, dataset: str, columns: list[str], dtypes: list[str], note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'dataset': dataset, 'note': note, 'columns': columns, 'dtypes': dtypes}
    path.write_text(json.dumps(payload, indent=2))


def _normalize_target(y: pd.Series, task_type: str) -> np.ndarray:
    if task_type == 'regression':
        return y.astype('float32').to_numpy()

    if task_type in {'binclass', 'multiclass'}:
        if y.dtype == object:
            vals = y.astype(str).str.lower()
            mapping = {
                'false': 0,
                'true': 1,
                'f': 0,
                't': 1,
                '0': 0,
                '1': 1,
                'no': 0,
                'yes': 1,
                'n': 0,
                'y': 1,
            }
            if vals.isin(mapping.keys()).all():
                return vals.map(mapping).astype('int64').to_numpy()
        cats = pd.Categorical(y)
        return cats.codes.astype('int64')
    raise ValueError(f'Unsupported task_type: {task_type}')


def _class_or_quantile_bins(y: np.ndarray, task_type: str) -> np.ndarray | None:
    if task_type in {'binclass', 'multiclass'}:
        return y.astype('int64')
    if task_type == 'regression':
        yv = y.astype(np.float64)
        edges = np.unique(np.quantile(yv, np.linspace(0.0, 1.0, 11)))
        if len(edges) >= 3:
            return np.digitize(yv, edges[1:-1], right=False).astype('int64')
    return None


def _feature_hash_groups(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    cols = [c for c in feat_cols if c in df.columns]
    if not cols:
        return np.arange(len(df), dtype=np.int64)
    s = df[cols].astype(str).fillna('<NA>').agg('|'.join, axis=1)
    return s.map(lambda x: int(hashlib.md5(x.encode('utf-8')).hexdigest()[:12], 16)).to_numpy(dtype=np.int64)


def _split_indices(
    n: int,
    y: np.ndarray,
    task_type: str,
    seed: int,
    split_cfg: dict[str, float],
    split_policy: str,
    groups: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    p_train = float(split_cfg.get('train', 0.7))
    p_val = float(split_cfg.get('val', 0.15))
    p_test = float(split_cfg.get('test', 0.15))
    if abs((p_train + p_val + p_test) - 1.0) > 1e-6:
        raise ValueError('default_split must sum to 1.0')

    idx = np.arange(n, dtype=np.int64)
    strata = _class_or_quantile_bins(y, task_type)

    if split_policy == 'group_hash' and groups is not None:
        rng = np.random.default_rng(seed)
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        g_map = {g: i for i, g in enumerate(uniq)}
        order = np.argsort(np.vectorize(g_map.get)(groups))
        idx_ord = idx[order]
        n_train = int(round(n * p_train))
        n_val = int(round(n * p_val))
        i_train = idx_ord[:n_train]
        i_val = idx_ord[n_train : n_train + n_val]
        i_test = idx_ord[n_train + n_val :]
    else:
        if train_test_split is None:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
            n_train = int(round(n * p_train))
            n_val = int(round(n * p_val))
            i_train = idx[:n_train]
            i_val = idx[n_train : n_train + n_val]
            i_test = idx[n_train + n_val :]
        else:
            strat = strata if split_policy.startswith('stratified') and strata is not None else None
            i_train, i_tmp, y_train, y_tmp = train_test_split(
                idx,
                y,
                test_size=(1.0 - p_train),
                random_state=seed,
                stratify=strat,
            )
            p_val_rel = p_val / (p_val + p_test)
            strat_tmp = _class_or_quantile_bins(y_tmp, task_type) if strat is not None else None
            i_val, i_test = train_test_split(
                i_tmp,
                test_size=(1.0 - p_val_rel),
                random_state=seed,
                stratify=strat_tmp,
            )

    return {
        'train': np.sort(np.asarray(i_train, dtype=np.int64)),
        'val': np.sort(np.asarray(i_val, dtype=np.int64)),
        'test': np.sort(np.asarray(i_test, dtype=np.int64)),
    }


def _detect_feature_types(x: pd.DataFrame, idx_train: np.ndarray) -> tuple[list[str], list[str], list[str]]:
    cat_cols = [
        c
        for c in x.columns
        if pd.api.types.is_object_dtype(x[c]) or pd.api.types.is_categorical_dtype(x[c]) or pd.api.types.is_bool_dtype(x[c])
    ]
    num_cols = [c for c in x.columns if c not in cat_cols]

    bin_cols: list[str] = []
    keep_num: list[str] = []
    for c in num_cols:
        vals = pd.to_numeric(x.iloc[idx_train][c], errors='coerce').dropna().unique()
        if len(vals) <= 2 and set(np.asarray(vals).astype(float).tolist()).issubset({0.0, 1.0}):
            bin_cols.append(c)
        else:
            keep_num.append(c)
    return keep_num, cat_cols, bin_cols


def _build_features(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
    splits: dict[str, np.ndarray],
) -> dict[str, Any]:
    use_cols = [c for c in df.columns if c != target_col and c not in set(drop_cols)]
    x = df[use_cols].copy()
    idx_train = splits['train']

    num_cols, cat_cols, bin_cols = _detect_feature_types(x, idx_train)
    out: dict[str, Any] = {'splits': splits, 'num_cols': num_cols, 'cat_cols': cat_cols, 'bin_cols': bin_cols}

    if num_cols:
        xn = x[num_cols].copy()
        med = xn.iloc[idx_train].median(numeric_only=True)
        xn = xn.fillna(med)
        out['x_num'] = {p: xn.iloc[splits[p]].to_numpy(dtype='float32', copy=True) for p in ['train', 'val', 'test']}
        out['num_missing_filled'] = med.to_dict()

    if bin_cols:
        xb = x[bin_cols].copy().fillna(0.0)
        xb = xb.astype('float32')
        out['x_bin'] = {p: xb.iloc[splits[p]].to_numpy(dtype='float32', copy=True) for p in ['train', 'val', 'test']}

    if cat_cols:
        xc = x[cat_cols].copy().fillna('__MISSING__').astype(str)
        cat_maps: dict[str, dict[str, int]] = {}
        for c in cat_cols:
            values = xc.iloc[idx_train][c].astype(str).unique().tolist()
            cat_maps[c] = {v: i for i, v in enumerate(values)}

        def encode_cat(part_idx: np.ndarray) -> np.ndarray:
            frame = xc.iloc[part_idx]
            cols = []
            for col in cat_cols:
                mp = cat_maps[col]
                vals = frame[col].astype(str).map(lambda z: mp.get(z, -1)).to_numpy(dtype='int64', copy=False)
                cols.append(vals)
            return np.column_stack(cols).astype('int64', copy=False)

        out['x_cat'] = {p: encode_cat(splits[p]) for p in ['train', 'val', 'test']}
        out['cat_categories'] = {c: list(m.keys()) for c, m in cat_maps.items()}

    return out


def _duplicate_overlap(groups: np.ndarray, splits: dict[str, np.ndarray]) -> dict[str, int]:
    g_train = set(groups[splits['train']].tolist())
    g_val = set(groups[splits['val']].tolist())
    g_test = set(groups[splits['test']].tolist())
    return {
        'train_val': int(len(g_train & g_val)),
        'train_test': int(len(g_train & g_test)),
        'val_test': int(len(g_val & g_test)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('esam/configs/esam_dataset_config.json'))
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    raw_root = Path(cfg['raw_data_root']).expanduser().resolve()
    pre_root = Path(cfg['preprocessed_root']).expanduser().resolve()
    art_root = Path(cfg['artifacts_root']).expanduser().resolve()
    split_cfg = cfg.get('default_split', {'train': 0.7, 'val': 0.15, 'test': 0.15})
    pre_root.mkdir(parents=True, exist_ok=True)
    art_root.mkdir(parents=True, exist_ok=True)

    selected = [x.strip() for x in args.datasets.split(',') if x.strip()]
    datasets = cfg['datasets']
    if selected:
        unknown = sorted(set(selected) - set(datasets))
        if unknown:
            raise SystemExit(f'Unknown datasets: {unknown}')
        keys = selected
    else:
        keys = [k for k, v in datasets.items() if v.get('enabled', True)]

    seed = int(args.seed if args.seed is not None else cfg.get('default_seed', 42))
    manifest: dict[str, Any] = {
        'seed': seed,
        'raw_data_root': str(raw_root),
        'preprocessed_root': str(pre_root),
        'datasets': {},
    }

    for name in keys:
        spec = datasets[name]
        out_dir = pre_root / name
        if out_dir.exists() and not args.force:
            required = [out_dir / f'Y_{p}.npy' for p in ['train', 'val', 'test']]
            if all(p.exists() for p in required):
                manifest['datasets'][name] = {'status': 'skipped_existing', 'output_dir': str(out_dir)}
                print(f'[skip] {name}: already exists')
                continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'[build] {name}')

        df = _load_table(raw_root, spec)
        target = spec.get('target_column')
        if target is None or target not in df.columns:
            debug_path = art_root / f'schema_debug_{name}.json'
            _write_schema_debug(
                debug_path,
                name,
                [str(c) for c in df.columns],
                [str(t) for t in df.dtypes],
                f"target_column '{target}' missing",
            )
            raise SystemExit(f'Target column not found for {name}. See {debug_path}')

        task_type = spec.get('task_type')
        if task_type not in {'binclass', 'multiclass', 'regression'}:
            raise SystemExit(f'Invalid task_type for {name}: {task_type}')

        drop_cols = list(spec.get('drop_columns', []))
        split_policy = str(spec.get('split_policy', 'stratified' if task_type != 'regression' else 'stratified_quantile'))

        y_all = _normalize_target(df[target], task_type)
        feat_cols = [c for c in df.columns if c != target and c not in set(drop_cols)]
        groups = _feature_hash_groups(df, feat_cols)
        if name == 'ecom-offers':
            split_policy = 'group_hash'

        splits = _split_indices(
            n=len(df),
            y=y_all,
            task_type=task_type,
            seed=seed,
            split_cfg=split_cfg,
            split_policy=split_policy,
            groups=groups,
        )

        feat = _build_features(df, target, drop_cols, splits)
        y_split = {p: y_all[splits[p]] for p in ['train', 'val', 'test']}

        for p in ['train', 'val', 'test']:
            np.save(out_dir / f'Y_{p}.npy', y_split[p])

        if 'x_num' in feat:
            for p in ['train', 'val', 'test']:
                np.save(out_dir / f'X_num_{p}.npy', feat['x_num'][p])

        if 'x_cat' in feat:
            for p in ['train', 'val', 'test']:
                np.save(out_dir / f'X_cat_{p}.npy', feat['x_cat'][p])

        if 'x_bin' in feat:
            for p in ['train', 'val', 'test']:
                np.save(out_dir / f'X_bin_{p}.npy', feat['x_bin'][p])

        dups = _duplicate_overlap(groups, splits)
        suspicious_target_like = [c for c in feat_cols if c.lower() in {'target', 'label', target.lower()}]

        info = {
            'name': name,
            'task_type': task_type,
            'score': 'accuracy' if task_type in {'binclass', 'multiclass'} else 'rmse',
            'metric_direction': 'higher_is_better_score',
            'n_num_features': len(feat['num_cols']),
            'n_cat_features': len(feat['cat_cols']),
            'n_bin_features': len(feat['bin_cols']),
            'seed': seed,
            'target_column': target,
            'source_train_file': spec['train_file'],
            'split_sizes': {p: int(len(splits[p])) for p in ['train', 'val', 'test']},
            'split_policy': split_policy,
            'preprocessing': {
                'missing_numeric': 'median(train)',
                'missing_categorical': '__MISSING__',
                'categorical_encoding': 'train_fit_ordinal_unknown_-1',
                'fit_scope': 'train_only',
            },
            'leakage_checks': {
                'duplicate_feature_hash_overlap': dups,
                'target_like_columns': suspicious_target_like,
                'id_like_columns': [c for c in feat_cols if 'id' in c.lower()],
            },
        }
        (out_dir / 'info.json').write_text(json.dumps(info, indent=2))

        manifest['datasets'][name] = {
            'status': 'ok',
            'output_dir': str(out_dir),
            'rows': int(len(df)),
            'num_features': len(feat['num_cols']),
            'cat_features': len(feat['cat_cols']),
            'bin_features': len(feat['bin_cols']),
            'task_type': task_type,
            'target_column': target,
            'split_sizes': info['split_sizes'],
            'split_policy': split_policy,
            'source_file': spec['train_file'],
            'drop_columns': drop_cols,
            'duplicate_feature_hash_overlap': dups,
        }

    manifest_path = art_root / 'preprocessing_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'[done] manifest: {manifest_path}')


if __name__ == '__main__':
    main()
