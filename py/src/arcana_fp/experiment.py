"""Experiment runner for the entropy rounds.

Test seeds are touched exactly once, at the end, for scoring. Model selection
uses the validation split only.

A run names a feature set (`r1`, `gf2`, `all`). Several feature sets share one
extraction pass but are fitted and scored independently, so an ablation varies
nothing except which columns the model is shown.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from arcana_fp.dataset import FeatureSet, load
from arcana_fp.features import FEATURE_SETS, feature_names
from arcana_fp.stats import contains_chance, wilson_interval

RANDOM_STATE = 20260929
MAX_ITER_GRID = (120, 300)
N_CURVE = (8, 32, 128)


def _git_sha(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def _fit(x: np.ndarray, y: np.ndarray, max_iter: int) -> HistGradientBoostingClassifier:
    # balanced weights, because the CSPRNG class is deliberately over-sampled 3x
    # for a tighter control interval and would otherwise own the prior.
    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.1,
        early_stopping=False,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(x, y)
    return model


def _select_and_score(fs: FeatureSet, y: np.ndarray, name: str) -> dict:
    train = fs.mask("train")
    val = fs.mask("val")
    test = fs.mask("test")

    best = None
    for max_iter in MAX_ITER_GRID:
        model = _fit(fs.x[train], y[train], max_iter)
        score = balanced_accuracy_score(y[val], model.predict(fs.x[val]))
        if best is None or score > best[0]:
            best = (score, max_iter, model)
    assert best is not None
    val_score, max_iter, model = best

    pred = model.predict(fs.x[test])
    truth = y[test]
    balanced = float(balanced_accuracy_score(truth, pred))
    labels = sorted(set(truth.tolist()) | set(pred.tolist()))
    cm = confusion_matrix(truth, pred, labels=labels)
    per_class = {}
    for i, label in enumerate(labels):
        total = int(cm[i].sum())
        hits = int(cm[i, i])
        if total == 0:
            continue
        low, high = wilson_interval(hits, total)
        per_class[str(label)] = {
            "recall": hits / total,
            "n": total,
            "ci95": [low, high],
        }
    hits = int((pred == truth).sum())
    low, high = wilson_interval(hits, int(truth.size))
    return {
        "head": name,
        "chosen_max_iter": max_iter,
        "val_balanced_accuracy": float(val_score),
        "test_balanced_accuracy": balanced,
        "test_accuracy": hits / int(truth.size),
        "test_accuracy_ci95": [low, high],
        "test_n": int(truth.size),
        "chance": 1.0 / len(labels),
        "per_class": per_class,
        "labels": [str(label) for label in labels],
        "confusion": cm.tolist(),
    }


def _csprng_control(fs: FeatureSet) -> dict:
    """Leakage canary. CSPRNG samples only, split into two arbitrary groups by
    seed parity. Anything above chance here means the pipeline leaks."""
    csprng = fs.is_csprng
    y = (fs.seed % 2 == 0).astype(int).astype(str)
    train = csprng & fs.mask("train")
    test = csprng & fs.mask("test")
    model = _fit(fs.x[train], y[train], MAX_ITER_GRID[0])
    pred = model.predict(fs.x[test])
    truth = y[test]
    hits = int((pred == truth).sum())
    total = int(truth.size)
    low, high = wilson_interval(hits, total)
    return {
        "head": "csprng_control_seed_parity",
        "accuracy": hits / total,
        "ci95": [low, high],
        "n": total,
        "chance": 0.5,
        "at_chance": contains_chance((low, high), 0.5),
    }


def _subset(fs: FeatureSet, mask: np.ndarray) -> FeatureSet:
    return FeatureSet(
        n_draws=fs.n_draws,
        x=fs.x[mask],
        generator=fs.generator[mask],
        shuffle=fs.shuffle[mask],
        is_csprng=fs.is_csprng[mask],
        split=fs.split[mask],
        seed=fs.seed[mask],
    )


def _weak_generator_head(fs: FeatureSet) -> dict:
    """Generator identification with the CSPRNG removed: are the six weak
    generators even mutually separable, or only separable from ChaCha20?"""
    sub = _subset(fs, ~fs.is_csprng)
    return _select_and_score(sub, sub.generator, "generator_weak_only")


def _csprng_shuffle_head(fs: FeatureSet) -> dict:
    """Shuffle identification restricted to CSPRNG samples: isolates shuffle-shape
    signal from generator-quality signal."""
    sub = _subset(fs, fs.is_csprng)
    return _select_and_score(sub, sub.shuffle, "shuffle_csprng_only")


def _columns(fs: FeatureSet, feature_set: str) -> FeatureSet:
    return FeatureSet(
        n_draws=fs.n_draws,
        x=fs.x[:, FEATURE_SETS[feature_set]],
        generator=fs.generator,
        shuffle=fs.shuffle,
        is_csprng=fs.is_csprng,
        split=fs.split,
        seed=fs.seed,
    )


def _score_all(
    sets: dict[int, FeatureSet], n_curve: tuple[int, ...], feature_set: str
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for n in n_curve:
        fs = _columns(sets[n], feature_set)
        block = {
            "generator": _select_and_score(fs, fs.generator, "generator"),
            "shuffle": _select_and_score(fs, fs.shuffle, "shuffle"),
            "is_csprng": _select_and_score(
                fs, np.where(fs.is_csprng, "csprng", "weak"), "is_csprng"
            ),
            "csprng_control": _csprng_control(fs),
            "generator_weak_only": _weak_generator_head(fs),
        }
        if n == n_curve[-1]:
            block["shuffle_csprng_only"] = _csprng_shuffle_head(fs)
        results[str(n)] = block
    return results


def run(
    dataset: Path,
    run_id: str,
    out_dir: Path,
    workers: int,
    repo: Path,
    feature_sets: tuple[str, ...] = ("all",),
    n_curve: tuple[int, ...] = N_CURVE,
) -> list[dict]:
    started = time.time()
    sets = load(dataset, n_curve, workers=workers)
    load_seconds = time.time() - started

    names = feature_names()
    payloads: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for feature_set in feature_sets:
        arm_started = time.time()
        results = _score_all(sets, n_curve, feature_set)
        columns = names[FEATURE_SETS[feature_set]]
        payload = {
            "id": run_id if len(feature_sets) == 1 else f"{run_id}-{feature_set}",
            "gitSha": _git_sha(repo),
            "randomState": RANDOM_STATE,
            "config": {
                "dataset": str(dataset),
                "n_curve": list(n_curve),
                "feature_set": feature_set,
                "model": "HistGradientBoostingClassifier",
                "max_iter_grid": list(MAX_ITER_GRID),
                "learning_rate": 0.1,
                "class_weight": "balanced",
                "n_features": len(columns),
                "features": columns,
                "n_samples": int(sets[n_curve[0]].x.shape[0]),
                "split_sizes": {
                    split: int((sets[n_curve[0]].split == split).sum())
                    for split in ("train", "val", "test")
                },
            },
            "metrics": results,
            "durationMs": int((time.time() - arm_started) * 1000),
            "featureLoadMs": int(load_seconds * 1000),
        }
        (out_dir / f"{payload['id']}.json").write_text(json.dumps(payload, indent=2) + "\n")
        payloads.append(payload)
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(prog="arcana-fp")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", default="r1-baseline")
    parser.add_argument("--out", default="lab/runs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--feature-sets", default="all")
    parser.add_argument("--n-curve", default=",".join(str(n) for n in N_CURVE))
    args = parser.parse_args()
    feature_sets = tuple(name.strip() for name in args.feature_sets.split(",") if name.strip())
    unknown = set(feature_sets) - set(FEATURE_SETS)
    if unknown:
        parser.error(f"unknown feature sets {sorted(unknown)}")
    n_curve = tuple(int(n) for n in args.n_curve.split(","))
    payloads = run(
        Path(args.dataset),
        args.run_id,
        Path(args.out),
        args.workers,
        Path(args.repo),
        feature_sets,
        n_curve,
    )
    summary = {
        payload["id"]: {
            n: {
                "generator": round(block["generator"]["test_balanced_accuracy"], 4),
                "shuffle": round(block["shuffle"]["test_balanced_accuracy"], 4),
                "is_csprng": round(block["is_csprng"]["test_balanced_accuracy"], 4),
                "generator_weak_only": round(
                    block["generator_weak_only"]["test_balanced_accuracy"], 4
                ),
                "control": [round(v, 4) for v in block["csprng_control"]["ci95"]],
            }
            for n, block in payload["metrics"].items()
        }
        for payload in payloads
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
