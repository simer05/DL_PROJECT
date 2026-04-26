#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import tomli
import tomli_w

BASE_CONFIGS = {
    "adult": "exp/tabm/adult/0-evaluation/0.toml",
    "california": "exp/tabm/california/0-evaluation/0.toml",
    "covtype2": "exp/tabm/covtype2/0-evaluation/0.toml",
}


def _resolve_dataset_and_config(dataset: str | None, base_config: str | None) -> tuple[str, Path]:
    if base_config is not None:
        cfg_path = Path(base_config)
        if dataset is None:
            dataset = cfg_path.parent.parent.name
        return dataset, cfg_path

    if dataset is None:
        raise SystemExit("Either --dataset or --base-config must be provided")
    if dataset not in BASE_CONFIGS:
        raise SystemExit(
            f"Unknown dataset {dataset!r}. Use one of {sorted(BASE_CONFIGS)} or pass --base-config"
        )
    return dataset, Path(BASE_CONFIGS[dataset])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--base-config", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="outputs/personb_ncl_rigorous")

    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--lambda-ncl", required=True, type=float)
    parser.add_argument("--ncl-warmup-epochs", default=10, type=int)
    parser.add_argument(
        "--ncl-space",
        default="logits",
        choices=["logits", "probs", "hybrid"],
    )
    parser.add_argument("--n-epochs", default=80, type=int)
    parser.add_argument("--patience", default=12, type=int)
    parser.add_argument("--max-retries", default=2, type=int)
    parser.add_argument(
        "--head-selection",
        action="store_true",
        help="Enable TabM head selection artifacts",
    )
    parser.add_argument("--tag", default="", help="Optional suffix tag for output dir")

    parser.add_argument(
        "--share-training-batches",
        dest="share_training_batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Team convention for final runs is true; can be overridden for debug/dev.",
    )

    args = parser.parse_args()

    dataset, base_path = _resolve_dataset_and_config(args.dataset, args.base_config)

    with base_path.open("rb") as f:
        config = tomli.load(f)

    config["seed"] = args.seed
    config["n_epochs"] = args.n_epochs
    config["patience"] = args.patience
    config["amp"] = True

    config.setdefault("model", {})
    config["model"]["share_training_batches"] = bool(args.share_training_batches)

    if args.dataset_path is not None:
        config["data"]["path"] = args.dataset_path
    elif args.data_root is not None:
        dataset_dir = args.dataset_dir or dataset
        config["data"]["path"] = str(Path(args.data_root) / dataset_dir)

    use_ncl = args.lambda_ncl > 0.0
    config["use_ncl"] = use_ncl
    config["lambda_ncl"] = float(args.lambda_ncl)
    config["ncl_warmup_epochs"] = int(args.ncl_warmup_epochs)
    config["ncl_space"] = args.ncl_space
    config["head_selection"] = bool(args.head_selection)

    lam_tag = str(args.lambda_ncl).replace("-", "m").replace(".", "p")
    out_name = f"lambda_{lam_tag}" + (f"_{args.tag}" if args.tag else "")
    output_dir = Path(args.output_root) / dataset / f"seed{args.seed}" / out_name
    report_path = output_dir / "report.json"
    if report_path.exists():
        print(f"[skip] exists: {report_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.generated.toml"

    cmd = ["python", "bin/model.py", str(config_path), "--output", str(output_dir), "--force"]
    print("[run]", " ".join(cmd))

    last_code = 1
    for attempt in range(1, args.max_retries + 2):
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(tomli_w.dumps(config))
        proc = subprocess.run(cmd)
        last_code = proc.returncode
        if last_code == 0 and report_path.exists():
            print(f"[ok] {output_dir}")
            return
        print(f"[warn] failed attempt {attempt} for {output_dir} (code={last_code})")

    raise SystemExit(last_code)


if __name__ == "__main__":
    main()
