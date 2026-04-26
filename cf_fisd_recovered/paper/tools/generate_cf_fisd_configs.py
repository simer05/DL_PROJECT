from __future__ import annotations

import argparse
from pathlib import Path

DATASETS = (
    'sberbank-housing',
    'ecom-offers',
    'homesite-insurance',
    'cooking-time',
    'delivery-eta',
)
N_SEEDS_DEFAULT = 5
TEACHER_NAMES = ('xgb', 'lgbm', 'cat')


def _replace_seed(toml_text: str, new_seed: int) -> str:
    out_lines = []
    seen = False
    for line in toml_text.splitlines():
        if not seen and line.startswith('seed = '):
            out_lines.append(f'seed = {new_seed}')
            seen = True
        else:
            out_lines.append(line)
    if not seen:
        out_lines.insert(0, f'seed = {new_seed}')
    return '\n'.join(out_lines).rstrip() + '\n'


def _format_cf_fisd_block(
    *,
    lam: float,
    variant: str,
    teacher_dir: str,
    dataset_name: str,
    teacher_names: tuple[str, ...],
) -> str:
    teacher_list = ', '.join(f'"{n}"' for n in teacher_names)
    return (
        '\n'
        '[cf_fisd]\n'
        f'lambda = {lam}\n'
        f'variant = "{variant}"\n'
        f'teacher_dir = "{teacher_dir}"\n'
        f'dataset_name = "{dataset_name}"\n'
        f'teacher_names = [{teacher_list}]\n'
    )


def _write_variant_configs(
    *,
    paper_root: Path,
    out_root: Path,
    dataset: str,
    variant_name: str,
    n_seeds: int,
    cf_fisd_block: str | None,
) -> int:
    src = (
        paper_root
        / 'exp'
        / 'tabm-piecewiselinear'
        / 'tabred'
        / dataset
        / '0-evaluation'
        / '0.toml'
    )
    if not src.exists():
        raise FileNotFoundError(src)
    base = src.read_text()
    target_dir = out_root / dataset / f'{variant_name}-evaluation'
    target_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for s in range(n_seeds):
        text = _replace_seed(base, s)
        if cf_fisd_block is not None:
            text = text.rstrip() + '\n' + cf_fisd_block
        (target_dir / f'{s}.toml').write_text(text)
        written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper-root', type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--n-seeds', type=int, default=N_SEEDS_DEFAULT)
    ap.add_argument('--lambdas', type=float, nargs='+', default=[0.05, 0.1, 0.2])
    ap.add_argument('--variant', default='raw', choices=['softmax', 'l1norm', 'raw'])
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    args = ap.parse_args()

    paper_root = args.paper_root.resolve()
    out_root = paper_root / 'exp' / 'cf_fisd' / 'tabred'
    teacher_root_rel = 'exp/cf_fisd/_teachers/tabred'

    total_files = 0
    for dataset in args.datasets:
        n = _write_variant_configs(
            paper_root=paper_root,
            out_root=out_root,
            dataset=dataset,
            variant_name='baseline_plr',
            n_seeds=args.n_seeds,
            cf_fisd_block=None,
        )
        total_files += n
        print(f'{dataset:<22} baseline_plr  -> {n} configs')
        for lam in args.lambdas:
            tag = f'hetero_{args.variant}_lam{lam}'
            block = _format_cf_fisd_block(
                lam=lam,
                variant=args.variant,
                teacher_dir=f'{teacher_root_rel}/{dataset}',
                dataset_name=dataset,
                teacher_names=TEACHER_NAMES,
            )
            n = _write_variant_configs(
                paper_root=paper_root,
                out_root=out_root,
                dataset=dataset,
                variant_name=tag,
                n_seeds=args.n_seeds,
                cf_fisd_block=block,
            )
            total_files += n
            print(f'{dataset:<22} {tag:<28} -> {n} configs')
    print(f'\nwrote {total_files} configs under {out_root}')


if __name__ == '__main__':
    main()
