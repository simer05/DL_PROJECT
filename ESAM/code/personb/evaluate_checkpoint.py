#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy
import torch
import torch.nn as nn
import tomli
import rtdl_num_embeddings


def load_model_module(paper_dir: Path):
    spec = importlib.util.spec_from_file_location("tabm_model_script", paper_dir / "bin" / "model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load model.py module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def to_builtin(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: to_builtin(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_builtin(v) for v in x]
    return x


def compute_diversity(head_preds: np.ndarray, is_regression: bool, is_binclass: bool) -> dict[str, float]:
    if head_preds.ndim == 2:
        members = head_preds[..., None]
    else:
        members = head_preds
    members = members.reshape(members.shape[0], members.shape[1], -1)
    if members.shape[1] <= 1:
        return {
            "mean_centered_corr": 0.0,
            "mean_pairwise_disagreement": 0.0,
            "member_std": 0.0,
        }

    centered = members - members.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=-1)
    denom = np.einsum("bi,bj->bij", norms, norms) + 1e-8
    corr = np.einsum("bid,bjd->bij", centered, centered) / denom
    off_diag = ~np.eye(members.shape[1], dtype=bool)
    mean_centered_corr = float(corr[:, off_diag].mean())

    if is_regression:
        pairwise_disagreement = float("nan")
    elif is_binclass:
        labels = (head_preds > 0.5).astype(np.int64)
        pairwise_disagreement = float((labels[:, :, None] != labels[:, None, :])[:, off_diag].mean())
    else:
        labels = head_preds.argmax(-1)
        pairwise_disagreement = float((labels[:, :, None] != labels[:, None, :])[:, off_diag].mean())

    member_std = float(members.std(axis=1).mean())
    return {
        "mean_centered_corr": mean_centered_corr,
        "mean_pairwise_disagreement": pairwise_disagreement,
        "member_std": member_std,
    }


def ece_score(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    if probs.ndim == 1:
        pred = (probs >= 0.5).astype(np.int64)
        conf = np.maximum(probs, 1.0 - probs)
    else:
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)
    acc = (pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        ece += (mask.mean()) * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def calibration_metrics(probs: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    if probs.ndim == 1:
        p = np.clip(probs, eps, 1.0 - eps)
        nll = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        brier = np.mean((p - y_true) ** 2)
    else:
        p = np.clip(probs, eps, 1.0)
        nll = -np.mean(np.log(p[np.arange(len(y_true)), y_true]))
        one_hot = np.eye(p.shape[1])[y_true]
        brier = np.mean(np.sum((p - one_hot) ** 2, axis=1))
    ece = ece_score(probs, y_true)
    return {"nll": float(nll), "brier": float(brier), "ece": float(ece)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, default=Path("."))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--part", default="test", choices=["train", "val", "test"])
    parser.add_argument("--corruption", default="none", choices=["none", "mask", "noise"])
    parser.add_argument("--severity", default="mild", choices=["mild", "moderate"])
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    paper_dir = args.paper_dir.resolve()
    run_dir = args.run_dir.resolve()

    if not paper_dir.joinpath("pixi.toml").exists():
        raise RuntimeError("paper-dir must be TabM/paper directory")
    sys.path.append(str(paper_dir))

    import lib
    import lib.data

    mod = load_model_module(paper_dir)
    config_path = run_dir / "config.generated.toml"
    if config_path.exists():
        config = tomli.loads(config_path.read_text())
    else:
        report_path = run_dir / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing both {config_path} and {report_path}")
        config = json.loads(report_path.read_text())["config"]

    delu = __import__("delu")
    delu.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = lib.data.build_dataset(**config["data"])
    if dataset.task.is_regression:
        dataset.data["y"], regression_label_stats = lib.data.standardize_labels(dataset.data["y"])
    else:
        regression_label_stats = None

    if dataset.n_bin_features > 0:
        x_bin = dataset.data.pop("x_bin")
        n_bin_features = x_bin["train"].shape[1]
        good_bin_idx = [i for i in range(n_bin_features) if len(np.unique(x_bin["train"][:, i])) > 1]
        if len(good_bin_idx) < n_bin_features:
            x_bin = {k: v[:, good_bin_idx] for k, v in x_bin.items()}
        if dataset.n_cat_features == 0:
            dataset.data["x_cat"] = {part: np.zeros((dataset.size(part), 0), dtype=np.int64) for part in x_bin}
        for part in x_bin:
            dataset.data["x_cat"][part] = np.column_stack([dataset.data["x_cat"][part], x_bin[part].astype(np.int64)])

    dataset = dataset.to_torch(device)

    if "bins" in config:
        y_train = dataset.data["y"]["train"].to(torch.long if dataset.task.is_classification else torch.float)
        compute_bins_kwargs = (
            {
                "y": y_train,
                "regression": dataset.task.is_regression,
                "verbose": False,
            }
            if "tree_kwargs" in config["bins"]
            else {}
        )
        bin_edges = rtdl_num_embeddings.compute_bins(dataset.data["x_num"]["train"], **config["bins"], **compute_bins_kwargs)
    else:
        bin_edges = None

    model = mod.Model(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),
        n_classes=dataset.task.try_compute_n_classes(),
        **config["model"],
        bins=bin_edges,
    ).to(device)

    state = lib.load_checkpoint(run_dir)["model"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing keys in checkpoint: {missing}")
    model.eval()

    x_num = dataset.data["x_num"][args.part] if "x_num" in dataset.data else None
    x_cat = dataset.data["x_cat"][args.part] if "x_cat" in dataset.data else None

    if args.corruption != "none":
        sev = 0.05 if args.severity == "mild" else 0.15
        if x_num is not None:
            x_num = x_num.clone()
            if args.corruption == "mask":
                mask = torch.rand_like(x_num) < sev
                x_num[mask] = 0.0
            elif args.corruption == "noise":
                train_std = dataset.data["x_num"]["train"].std(dim=0, keepdim=True).clamp_min(1e-8)
                noise = torch.randn_like(x_num) * train_std * sev
                x_num = x_num + noise
        if x_cat is not None and args.corruption == "mask":
            x_cat = x_cat.clone()
            mask = torch.rand(x_cat.shape, device=x_cat.device) < sev
            x_cat[mask] = 0

    @torch.inference_mode()
    def apply(idx: torch.Tensor) -> torch.Tensor:
        return model(
            x_num[idx] if x_num is not None else None,
            x_cat[idx] if x_cat is not None else None,
        ).squeeze(-1).float()

    parts = torch.arange(dataset.size(args.part), device=device).split(args.eval_batch_size)
    head_pred = torch.cat([apply(i) for i in parts]).cpu().numpy()

    if dataset.task.is_regression:
        assert regression_label_stats is not None
        head_pred = head_pred * regression_label_stats.std + regression_label_stats.mean
        ensemble_pred = head_pred.mean(axis=1)
        prediction_type = "labels"
    else:
        head_pred = scipy.special.softmax(head_pred, axis=-1)
        if dataset.task.is_binclass:
            head_pred = head_pred[..., 1]
        ensemble_pred = head_pred.mean(axis=1)
        prediction_type = "probs"

    metrics = dataset.task.calculate_metrics({args.part: ensemble_pred}, prediction_type)[args.part]

    member_scores = []
    n_members = head_pred.shape[1] if head_pred.ndim >= 2 else 1
    for i in range(n_members):
        m_pred = head_pred[:, i] if head_pred.ndim >= 2 else head_pred
        m_score = dataset.task.calculate_metrics({args.part: m_pred}, prediction_type)[args.part]["score"]
        member_scores.append(float(m_score))

    ensemble_gain = float(metrics["score"] - float(np.mean(member_scores))) if member_scores else float("nan")

    diversity = compute_diversity(head_pred, dataset.task.is_regression, dataset.task.is_binclass)

    y_true = dataset.data["y"][args.part].detach().cpu().numpy()
    if dataset.task.is_classification:
        y_true = y_true.astype(np.int64)
        calib = calibration_metrics(ensemble_pred, y_true)
    else:
        calib = {"nll": float("nan"), "brier": float("nan"), "ece": float("nan")}

    out = {
        "run_dir": str(run_dir),
        "dataset": config["data"]["path"].split("/")[-1],
        "seed": int(config["seed"]),
        "lambda_ncl": float(config.get("lambda_ncl", 0.0)),
        "use_ncl": bool(config.get("use_ncl", False)),
        "ncl_space": config.get("ncl_space", "logits"),
        "ncl_warmup_epochs": int(config.get("ncl_warmup_epochs", 0)),
        "part": args.part,
        "corruption": args.corruption,
        "severity": args.severity,
        "metrics": to_builtin(metrics),
        "diversity": to_builtin(diversity),
        "member_score_mean": float(np.mean(member_scores)) if member_scores else float("nan"),
        "member_score_std": float(np.std(member_scores)) if member_scores else float("nan"),
        "ensemble_gain": ensemble_gain,
        "calibration": calib,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
