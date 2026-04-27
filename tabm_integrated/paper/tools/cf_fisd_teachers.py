from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    _cwd = Path.cwd()
    assert _cwd.joinpath(
        'pixi.toml'
    ).exists(), 'The script must be run from the `paper/` directory'
    sys.path.append(str(_cwd))
    del _cwd

import delu
import numpy as np
import pandas as pd
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]
from loguru import logger

import lib
import lib.data


SUPPORTED_DATASETS = (
    'sberbank-housing',
    'ecom-offers',
    'homesite-insurance',
    'cooking-time',
    'delivery-eta',
)
TEACHERS = ('xgb', 'lgbm', 'cat')


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open('rb') as f:
        return tomllib.load(f)


def _build_combined_features(dataset) -> tuple[dict[str, pd.DataFrame], list[int]]:
    parts = list(dataset.parts())
    n_num = dataset.n_num_features
    n_bin = dataset.n_bin_features
    n_cat = dataset.n_cat_features

    frames: dict[str, pd.DataFrame] = {part: pd.DataFrame() for part in parts}
    for part in parts:
        if 'x_num' in dataset:
            frames[part] = pd.concat(
                [frames[part], pd.DataFrame(dataset['x_num'][part])], axis=1
            )
        if 'x_cat' in dataset:
            frames[part] = pd.concat(
                [frames[part], pd.DataFrame(dataset['x_cat'][part].astype('int64'))],
                axis=1,
            )
        if 'x_bin' in dataset:
            frames[part] = pd.concat(
                [frames[part], pd.DataFrame(dataset['x_bin'][part].astype('int64'))],
                axis=1,
            )
        frames[part].columns = list(range(frames[part].shape[1]))

    cat_indices_in_combined: list[int] = []
    if 'x_cat' in dataset:
        cat_indices_in_combined.extend(range(n_num, n_num + n_cat))
    if 'x_bin' in dataset:
        cat_indices_in_combined.extend(range(n_num + n_cat, n_num + n_cat + n_bin))
    return frames, cat_indices_in_combined


def _train_xgb(
    cfg: dict[str, Any],
    X: dict[str, pd.DataFrame],
    y: dict[str, np.ndarray],
    cat_idx: list[int],
    is_regression: bool,
    is_binclass: bool,
    seed: int,
) -> np.ndarray:
    from xgboost import XGBClassifier, XGBRegressor

    df_X = {part: frame.astype(np.float32) for part, frame in X.items()}

    model_kwargs = dict(cfg['model'])
    _force_cpu_in_model_cfg('xgb', model_kwargs)
    extra = {'random_state': seed}
    if is_regression:
        model = XGBRegressor(**model_kwargs, **extra)
    else:
        eval_metric = 'auc' if is_binclass else 'merror'
        model = XGBClassifier(
            **model_kwargs,
            **extra,
            disable_default_eval_metric=True,
            eval_metric=eval_metric,
        )
    fit_kwargs = dict(cfg.get('fit', {}))
    model.fit(
        df_X['train'],
        y['train'],
        eval_set=[(df_X['val'], y['val'])],
        **fit_kwargs,
    )
    return np.asarray(model.feature_importances_, dtype=np.float64)


def _train_lgbm(
    cfg: dict[str, Any],
    X: dict[str, pd.DataFrame],
    y: dict[str, np.ndarray],
    cat_idx: list[int],
    is_regression: bool,
    is_binclass: bool,
    seed: int,
) -> np.ndarray:
    import lightgbm
    from lightgbm import LGBMClassifier, LGBMRegressor

    model_kwargs = dict(cfg['model'])
    _force_cpu_in_model_cfg('lgbm', model_kwargs)
    stopping_rounds = model_kwargs.pop('stopping_rounds', None)
    extra = {'random_state': seed}
    if is_regression:
        model = LGBMRegressor(**model_kwargs, **extra)
        fit_extra = {'eval_metric': 'rmse'}
    elif is_binclass:
        model = LGBMClassifier(**model_kwargs, **extra)
        fit_extra = {'eval_metric': 'auc'}
    else:
        model = LGBMClassifier(**model_kwargs, **extra)
        fit_extra = {'eval_metric': 'multi_error'}
    callbacks = []
    if stopping_rounds is not None:
        callbacks.append(lightgbm.early_stopping(stopping_rounds=int(stopping_rounds)))
    fit_kwargs = dict(cfg.get('fit', {}))
    if cat_idx:
        fit_extra['categorical_feature'] = list(cat_idx)
    model.fit(
        X['train'],
        y['train'],
        eval_set=[(X['val'], y['val'])],
        callbacks=callbacks,
        **fit_kwargs,
        **fit_extra,
    )
    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    if importances.sum() > 0.0:
        return importances
    return np.asarray(
        model.booster_.feature_importance(importance_type='gain'), dtype=np.float64
    )


def _train_cat(
    cfg: dict[str, Any],
    X: dict[str, pd.DataFrame],
    y: dict[str, np.ndarray],
    cat_idx: list[int],
    is_regression: bool,
    is_binclass: bool,
    seed: int,
) -> np.ndarray:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    model_kwargs = dict(cfg['model'])
    _force_cpu_in_model_cfg('cat', model_kwargs)
    model_kwargs.setdefault('random_seed', seed)
    model_kwargs.setdefault('verbose', False)
    model_kwargs.setdefault('allow_writing_files', False)
    if is_regression:
        model = CatBoostRegressor(**model_kwargs)
    else:
        model = CatBoostClassifier(**model_kwargs)
    train_pool = Pool(X['train'], y['train'], cat_features=cat_idx if cat_idx else None)
    val_pool = Pool(X['val'], y['val'], cat_features=cat_idx if cat_idx else None)
    fit_kwargs = dict(cfg.get('fit', {}))
    model.fit(train_pool, eval_set=val_pool, **fit_kwargs)
    return np.asarray(model.get_feature_importance(type='PredictionValuesChange'), dtype=np.float64)


def _normalize_to_simplex(v: np.ndarray) -> np.ndarray:
    v = np.clip(v.astype(np.float64), 0.0, None)
    s = float(v.sum())
    if s > 0.0:
        return (v / s).astype(np.float32)
    return np.full_like(v, 1.0 / len(v), dtype=np.float32)


def _resolve_tuned_toml(tabred_root: Path, teacher: str, dataset: str) -> Path:
    name_map = {'xgb': 'xgboost_', 'lgbm': 'lightgbm_', 'cat': 'catboost_'}
    base = tabred_root / 'exp' / name_map[teacher] / dataset
    candidates = [base / 'evaluation' / '0.toml', base / '0.toml']
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _force_cpu_in_model_cfg(teacher: str, model_cfg: dict[str, Any]) -> None:
    if teacher == 'xgb':
        model_cfg['device'] = 'cpu'
        model_cfg['tree_method'] = model_cfg.get('tree_method', 'hist')
        model_cfg.pop('gpu_id', None)
    elif teacher == 'lgbm':
        model_cfg['device_type'] = 'cpu'
        model_cfg.pop('gpu_use_dp', None)
        model_cfg.pop('gpu_platform_id', None)
        model_cfg.pop('gpu_device_id', None)
    elif teacher == 'cat':
        model_cfg['task_type'] = 'CPU'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=SUPPORTED_DATASETS)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument(
        '--tabred-root',
        type=Path,
        default=Path(os.environ.get('TABRED_REPO_ROOT', 'D:/TabM_PROJ/tabred_fork')),
    )
    ap.add_argument(
        '--out-root',
        type=Path,
        default=Path('exp/cf_fisd/_teachers/tabred'),
    )
    ap.add_argument('--teachers', nargs='+', default=list(TEACHERS), choices=list(TEACHERS))
    ap.add_argument('--data-path', default=None,
                    help='Override dataset path; defaults to data/<dataset>.')
    args = ap.parse_args()

    delu.random.seed(args.seed)
    data_path = args.data_path or f'data/{args.dataset}'
    dataset = lib.data.build_dataset(path=data_path, cache=True, cat_policy='ordinal')
    if dataset.task.is_regression:
        dataset.data['y'], _ = lib.data.standardize_labels(dataset.data['y'])

    X, cat_idx = _build_combined_features(dataset)
    y = {part: np.asarray(dataset['y'][part]) for part in dataset.parts()}

    is_reg = dataset.task.is_regression
    is_bin = bool(getattr(dataset.task, 'is_binclass', False))

    out_dir = args.out_root / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    n_num = dataset.n_num_features
    n_cat = dataset.n_cat_features
    n_bin = dataset.n_bin_features
    n_features = n_num + n_cat + n_bin
    logger.info(
        f'{args.dataset}: n_num={n_num}  n_cat={n_cat}  n_bin={n_bin}  n_features={n_features}'
    )

    trainers = {'xgb': _train_xgb, 'lgbm': _train_lgbm, 'cat': _train_cat}
    for t in args.teachers:
        toml_path = _resolve_tuned_toml(args.tabred_root, t, args.dataset)
        if not toml_path.exists():
            raise FileNotFoundError(toml_path)
        cfg = _load_toml(toml_path)
        logger.info(f'training {t} for {args.dataset}: hp from {toml_path}')
        importances = trainers[t](
            cfg=cfg,
            X=X,
            y=y,
            cat_idx=cat_idx,
            is_regression=is_reg,
            is_binclass=is_bin,
            seed=args.seed,
        )
        if importances.shape != (n_features,):
            raise ValueError(
                f'importance shape {importances.shape} != ({n_features},) for {t}/{args.dataset}'
            )
        importances = _normalize_to_simplex(importances)
        out_path = out_dir / f'{t}.npy'
        np.save(out_path, importances)
        topk = np.argsort(importances)[::-1][:5].tolist()
        entropy = float(-(importances * np.log(importances + 1e-12)).sum())
        logger.info(
            f'  saved {out_path}  top5_idx={topk}  entropy={entropy:.3f}'
        )

    meta = {
        'dataset': args.dataset,
        'task_type': dataset.task.type_.value,
        'n_num': n_num,
        'n_cat': n_cat,
        'n_bin': n_bin,
        'n_features': n_features,
        'cat_indices_in_combined': cat_idx,
        'feature_order': 'num,cat,bin',
        'seed': args.seed,
        'tabred_repo_root': str(args.tabred_root),
        'teachers': list(args.teachers),
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    logger.info(f'wrote {out_dir / "meta.json"}')


if __name__ == '__main__':
    lib.configure_libraries()
    main()
