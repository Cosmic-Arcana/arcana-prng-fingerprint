import numpy as np

from arcana_fp.dataset import FeatureSet
from arcana_fp.weak_ceiling import bootstrap_balanced_accuracy, intervals_overlap, score_head, verdicts

SPLIT_SIZES = {"train": 60, "val": 20, "test": 40}


def _feature_set(shift: float) -> FeatureSet:
    rng = np.random.default_rng(7)
    generators, splits = [], []
    for generator in ("pcg32", "xorshift128+", "chacha20"):
        for split, count in SPLIT_SIZES.items():
            generators += [generator] * count
            splits += [split] * count
    generator = np.array(generators)
    x = rng.normal(size=(generator.size, 4))
    x[generator == "pcg32", 0] += shift
    return FeatureSet(
        n_draws=512,
        x=x,
        generator=generator,
        shuffle=np.full(generator.size, "fisher-yates"),
        is_csprng=generator == "chacha20",
        split=np.array(splits),
        seed=np.arange(generator.size, dtype=np.int64),
    )


def test_bootstrap_interval_is_tight_for_a_perfect_detector() -> None:
    truth = np.array(["a"] * 50 + ["b"] * 50)
    point, (low, high) = bootstrap_balanced_accuracy(truth, truth.copy())
    assert point == 1.0
    assert low == 1.0 and high == 1.0


def test_bootstrap_interval_contains_chance_for_coin_flips() -> None:
    rng = np.random.default_rng(3)
    truth = np.array(["a"] * 100 + ["b"] * 100)
    pred = rng.choice(["a", "b"], size=200)
    point, (low, high) = bootstrap_balanced_accuracy(truth, pred)
    assert low <= point <= high
    assert low <= 0.5 <= high


def test_bootstrap_is_deterministic() -> None:
    truth = np.array(["a"] * 30 + ["b"] * 30)
    pred = np.array(["a"] * 20 + ["b"] * 10 + ["b"] * 25 + ["a"] * 5)
    assert bootstrap_balanced_accuracy(truth, pred) == bootstrap_balanced_accuracy(truth, pred)


def test_score_head_sees_a_planted_shift() -> None:
    result = score_head(_feature_set(shift=3.0), ("pcg32", "chacha20"))
    assert result["signal"] is True
    assert result["test_balanced_accuracy"] > 0.8


def test_score_head_stays_at_chance_without_a_shift() -> None:
    result = score_head(_feature_set(shift=0.0), ("pcg32", "chacha20"))
    assert result["signal"] is False
    assert result["at_chance"] is True


def test_score_head_ignores_classes_outside_the_head() -> None:
    result = score_head(_feature_set(shift=3.0), ("xorshift128+", "chacha20"))
    assert result["n_test"] == 2 * SPLIT_SIZES["test"]
    assert result["at_chance"] is True


def test_three_class_chance_is_one_third() -> None:
    result = score_head(_feature_set(shift=0.0), ("pcg32", "xorshift128+", "chacha20"))
    assert abs(result["chance"] - 1 / 3) < 1e-12


def test_intervals_overlap() -> None:
    assert intervals_overlap((0.4, 0.6), (0.55, 0.7))
    assert not intervals_overlap((0.4, 0.5), (0.6, 0.7))


def _head(name: str, low: float, high: float) -> dict:
    return {"head": name, "ci95": [low, high], "at_chance": low <= 0.5 <= high, "signal": low > 0.5}


def _round(low: float, high: float, control_ok: bool = True) -> dict:
    return {
        "heads": [_head("pcg32-vs-chacha20", low, high), _head("xorshift128+-vs-chacha20", 0.4, 0.6)],
        "control": {"at_chance": control_ok},
    }


def test_verdicts_hold_when_everything_is_at_chance() -> None:
    results = {128: _round(0.42, 0.58), 512: _round(0.44, 0.60)}
    assert verdicts(results) == {"H14": True, "H15": True, "H16": True, "H17": False}


def test_verdicts_report_a_signal_and_a_broken_control() -> None:
    results = {128: _round(0.42, 0.58, control_ok=False), 512: _round(0.62, 0.78)}
    out = verdicts(results)
    assert out["H14"] is False
    assert out["H15"] is False
    assert out["H16"] is False
    assert out["H17"] is True
