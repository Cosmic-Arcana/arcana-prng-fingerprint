"""Round 6: does any statistical signal separate pcg32 and xorshift128+ from ChaCha20
as the draw count grows. Detection only: it consumes the round-2 feature columns and
never reads generator words, so it cannot recover state or predict output.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from arcana_fp.dataset import FeatureSet, load
from arcana_fp.experiment import MAX_ITER_GRID, RANDOM_STATE, _csprng_control, _fit, _git_sha, _subset

BOOTSTRAP_SEED = 20261031
BOOTSTRAP_RESAMPLES = 2000
N_CURVE = (128, 256, 512)
CONTROL_CLASS = "chacha20"
SELECTED = ("pcg32", "xorshift128+", CONTROL_CLASS)
DECIDED_HEADS = (("pcg32", CONTROL_CLASS), ("xorshift128+", CONTROL_CLASS))
REFERENCE_HEADS = (("pcg32", "xorshift128+"), SELECTED)


def head_name(classes: tuple[str, ...]) -> str:
    return "-vs-".join(classes) if len(classes) == 2 else "three-class"


def bootstrap_balanced_accuracy(
    truth: np.ndarray,
    pred: np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, tuple[float, float]]:
    """Point estimate and 95% interval, resampling inside each true class so every
    class stays represented however the draw falls."""
    rng = np.random.default_rng(seed)
    hits = pred == truth
    per_class = [hits[truth == label] for label in np.unique(truth)]
    point = float(np.mean([h.mean() for h in per_class]))
    samples = np.mean(
        [h[rng.integers(0, h.size, size=(resamples, h.size))].mean(axis=1) for h in per_class],
        axis=0,
    )
    low, high = np.percentile(samples, [2.5, 97.5])
    return point, (float(low), float(high))


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def score_head(fs: FeatureSet, classes: tuple[str, ...]) -> dict:
    sub = _subset(fs, np.isin(fs.generator, classes))
    train, val, test = (sub.mask(name) for name in ("train", "val", "test"))
    best = max(
        (
            (balanced_accuracy_score(sub.generator[val], _fit(sub.x[train], sub.generator[train], it).predict(sub.x[val])), -it, it)
            for it in MAX_ITER_GRID
        ),
    )
    val_score, chosen = best[0], best[2]
    model = _fit(sub.x[train], sub.generator[train], chosen)
    point, interval = bootstrap_balanced_accuracy(sub.generator[test], model.predict(sub.x[test]))
    chance = 1 / len(classes)
    return {
        "head": head_name(classes),
        "classes": list(classes),
        "n_draws": fs.n_draws,
        "chance": chance,
        "max_iter": chosen,
        "val_balanced_accuracy": float(val_score),
        "test_balanced_accuracy": point,
        "ci95": list(interval),
        "n_test": int(test.sum()),
        "signal": bool(interval[0] > chance),
        "at_chance": bool(interval[0] <= chance <= interval[1]),
    }


def verdicts(results: dict[int, dict]) -> dict[str, bool]:
    """H14-H17 exactly as registered in lab/reports/round-6.md."""
    first, last = min(results), max(results)
    heads = {n: {h["head"]: h for h in block["heads"]} for n, block in results.items()}
    return {
        "H14": all(h["at_chance"] for h in results[last]["heads"]),
        "H15": all(
            intervals_overlap(tuple(heads[first][name]["ci95"]), tuple(heads[last][name]["ci95"]))
            for name in heads[last]
        ),
        "H16": all(block["control"]["at_chance"] for block in results.values()),
        "H17": any(h["signal"] for block in results.values() for h in block["heads"]),
    }


def run(dataset: Path, out_dir: Path, workers: int, repo: Path, n_curve: tuple[int, ...] = N_CURVE) -> dict:
    started = time.time()
    sets = load(dataset, n_curve, workers=workers)
    load_ms = int((time.time() - started) * 1000)
    results: dict[int, dict] = {}
    for n in n_curve:
        selected = _subset(sets[n], np.isin(sets[n].generator, SELECTED))
        results[n] = {
            "heads": [score_head(selected, classes) for classes in DECIDED_HEADS],
            "reference": [score_head(selected, classes) for classes in REFERENCE_HEADS],
            "control": _csprng_control(selected),
        }
    anchor = _subset(sets[n_curve[0]], np.isin(sets[n_curve[0]].generator, SELECTED))
    payload = {
        "id": "r6-weak-ceiling",
        "gitSha": _git_sha(repo),
        "randomState": RANDOM_STATE,
        "config": {
            "dataset": str(dataset),
            "n_curve": list(n_curve),
            "classes": list(SELECTED),
            "feature_set": "all",
            "model": "HistGradientBoostingClassifier",
            "max_iter_grid": list(MAX_ITER_GRID),
            "class_weight": "balanced",
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
            "split_sizes": {
                label: {s: int(((anchor.generator == label) & (anchor.split == s)).sum()) for s in ("train", "val", "test")}
                for label in SELECTED
            },
        },
        "metrics": {str(n): block for n, block in results.items()},
        "verdicts": verdicts(results),
        "durationMs": int((time.time() - started) * 1000),
        "featureLoadMs": load_ms,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "r6-weak-ceiling.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(prog="arcana-fp-r6")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="lab/runs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    payload = run(Path(args.dataset), Path(args.out), args.workers, Path(args.repo))
    summary = {
        n: {
            "heads": {h["head"]: [round(h["test_balanced_accuracy"], 3), [round(v, 3) for v in h["ci95"]]] for h in block["heads"] + block["reference"]},
            "control": [round(v, 3) for v in block["control"]["ci95"]],
        }
        for n, block in payload["metrics"].items()
    }
    print(json.dumps({"verdicts": payload["verdicts"], "metrics": summary, "durationMs": payload["durationMs"]}, indent=2))


if __name__ == "__main__":
    main()
