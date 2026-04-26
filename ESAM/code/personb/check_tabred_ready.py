#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check TabReD dataset readiness for Person B pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("personb/tabred_dataset_config.json"),
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any dataset is missing/incomplete")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    data_root = Path(cfg["data_root"]).expanduser().resolve()
    required_files = cfg.get("required_files", [])
    feature_any = cfg.get("required_feature_train_files_any_of", [])
    datasets = cfg["datasets"]

    print(f"[info] config: {args.config}")
    print(f"[info] data_root: {data_root}")

    n_ready = 0
    missing_rows: list[str] = []
    for key, meta in datasets.items():
        dataset_dir = meta["dataset_dir"]
        base_config = Path(meta["base_config"])
        path = (data_root / dataset_dir).resolve()

        issues = []
        if not base_config.exists():
            issues.append(f"missing base config: {base_config}")
        if not path.exists():
            issues.append(f"missing dataset dir: {path}")
        else:
            for rf in required_files:
                if not (path / rf).exists():
                    issues.append(f"missing file: {path / rf}")
            if feature_any and not any((path / f).exists() for f in feature_any):
                issues.append(
                    "missing feature train file: expected one of "
                    + ", ".join(str(path / f) for f in feature_any)
                )

        if issues:
            print(f"[MISSING] {key} ({dataset_dir})")
            for i in issues:
                print(f"  - {i}")
            missing_rows.append(key)
        else:
            print(f"[READY]   {key} ({dataset_dir})")
            n_ready += 1

    total = len(datasets)
    print(f"\n[summary] READY={n_ready}/{total} MISSING={total - n_ready}/{total}")
    if args.strict and missing_rows:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
