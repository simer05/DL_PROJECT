import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


PARTS = ('train', 'val', 'test')


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _file_manifest(path: Path, *, relative_to: Path) -> dict:
    out = {
        'path': str(path.relative_to(relative_to)),
        'bytes': path.stat().st_size,
        'sha256': _sha256(path),
    }
    if path.suffix == '.npy':
        arr = np.load(path, allow_pickle=True)
        out |= {
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
        }
    return out


def _load_split_indices(src: Path) -> dict[str, np.ndarray]:
    split_dir = src / 'split-default'
    if not split_dir.exists():
        raise FileNotFoundError(f'Missing split directory: {split_dir}')
    return {part: np.load(split_dir / f'{part}_idx.npy') for part in PARTS}


def _validate_split_indices(
    *, dataset_name: str, idx: dict[str, np.ndarray], n_rows: int
) -> None:
    seen: set[int] = set()
    total = 0
    for part, values in idx.items():
        if values.ndim != 1:
            raise ValueError(f'{dataset_name}: {part}_idx.npy must be 1D')
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f'{dataset_name}: {part}_idx.npy must contain integers')
        if len(values) == 0:
            raise ValueError(f'{dataset_name}: {part} split is empty')
        if values.min() < 0 or values.max() >= n_rows:
            raise ValueError(
                f'{dataset_name}: {part} split has indices outside [0, {n_rows})'
            )
        unique = set(map(int, values.tolist()))
        if len(unique) != len(values):
            raise ValueError(f'{dataset_name}: duplicate row indices inside {part}')
        overlap = seen & unique
        if overlap:
            sample = sorted(overlap)[:5]
            raise ValueError(
                f'{dataset_name}: split-default overlaps across parts, sample={sample}'
            )
        seen |= unique
        total += len(values)

    if len(seen) != total:
        raise ValueError(f'{dataset_name}: split-default contains duplicate rows')
    if len(seen) != n_rows:
        raise ValueError(
            f'{dataset_name}: split-default covers {len(seen)} rows, but Y.npy has '
            f'{n_rows} rows'
        )


def _validate_converted_dataset(
    *, dst: Path, expected_files: set[str], expected_rows: dict[str, int]
) -> None:
    actual_files = {p.name for p in dst.glob('*.npy')}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise ValueError(
            f'{dst.name}: converted file set mismatch; missing={missing}, extra={extra}'
        )

    for part in PARTS:
        y_path = dst / f'Y_{part}.npy'
        if not y_path.exists():
            raise FileNotFoundError(f'Missing converted labels: {y_path}')
        y_rows = np.load(y_path, allow_pickle=True).shape[0]
        if y_rows != expected_rows[part]:
            raise ValueError(
                f'{dst.name}: Y_{part}.npy has {y_rows} rows, expected '
                f'{expected_rows[part]}'
            )
        for path in sorted(dst.glob(f'X_*_{part}.npy')):
            x_rows = np.load(path, allow_pickle=True).shape[0]
            if x_rows != y_rows:
                raise ValueError(
                    f'{dst.name}: {path.name} has {x_rows} rows, but '
                    f'Y_{part}.npy has {y_rows}'
                )


def prepare_tabred(
    tabred_src: Path, tabred_dst: Path, print_only: bool = False, exist_ok: bool = False
) -> None:
    assert tabred_src.exists()
    if not print_only:
        tabred_dst.mkdir(exist_ok=True)

    for src in sorted(x for x in tabred_src.iterdir() if x.is_dir()):
        print('>>>', src.name)

        dst = tabred_dst / src.name
        if not print_only and dst.exists() and exist_ok:
            # `--force` means a clean rebuild, not "overwrite some files and
            # keep stale files that no longer belong to this dataset".
            shutil.rmtree(dst)
        if not print_only:
            dst.mkdir(exist_ok=exist_ok)
            shutil.copyfile(src / 'info.json', dst / 'info.json')

        idx = (
            _load_split_indices(src)
            if not print_only
            else {}
        )

        paths = [*sorted(src.glob('X_*.npy')), src / 'Y.npy']
        paths = [path for path in paths if path.name != 'X_meta.npy']
        if not (src / 'Y.npy').exists():
            raise FileNotFoundError(f'Missing labels: {src / "Y.npy"}')
        if not any(path.name.startswith('X_') for path in paths):
            raise FileNotFoundError(f'No X_*.npy feature files found under {src}')

        if not print_only:
            y = np.load(src / 'Y.npy', allow_pickle=True)
            n_rows = y.shape[0]
            _validate_split_indices(dataset_name=src.name, idx=idx, n_rows=n_rows)
            expected_rows = {part: len(idx[part]) for part in PARTS}
            expected_files: set[str] = {
                f'{path.stem}_{part}.npy' for path in paths for part in PARTS
            }

        for path in paths:
            print(path.name)
            if print_only:
                continue
            x = np.load(path, allow_pickle=True)
            if x.shape[0] != n_rows:
                raise ValueError(
                    f'{src.name}: {path.name} has {x.shape[0]} rows, but Y.npy '
                    f'has {n_rows} rows'
                )
            for part in PARTS:
                np.save(dst / f'{path.stem}_{part}.npy', x[idx[part]])

        if not print_only:
            _validate_converted_dataset(
                dst=dst, expected_files=expected_files, expected_rows=expected_rows
            )
            manifest = {
                'dataset': src.name,
                # Keep the manifest path-independent so teammates can compare
                # checksums even when their home directories differ.
                'source_dataset': src.name,
                'row_counts': expected_rows,
                'source_files': [
                    _file_manifest(src / 'info.json', relative_to=src),
                    *[
                        _file_manifest(
                            src / 'split-default' / f'{part}_idx.npy',
                            relative_to=src,
                        )
                        for part in PARTS
                    ],
                    *[_file_manifest(path, relative_to=src) for path in paths],
                ],
                'output_files': [
                    _file_manifest(path, relative_to=dst)
                    for path in sorted(dst.glob('*.npy'))
                ],
            }
            (dst / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=Path, help='Source folder of TabReD datasets.')
    parser.add_argument('dst', type=Path, help='Where to save TabReD datasets.')
    parser.add_argument(
        '--print_only',
        action='store_true',
        help='Only print files from source, without loading and saving.',
    )
    parser.add_argument(
        '--force', action='store_true', help='Overwrite existing dst folders.'
    )

    args = parser.parse_args()
    try:
        prepare_tabred(
            args.src, args.dst, print_only=args.print_only, exist_ok=args.force
        )
    except FileExistsError as e:
        raise FileExistsError(
            f'File exists: {e.filename}. Use --force if you want to overwrite.'
        )


if __name__ == '__main__':
    main()
