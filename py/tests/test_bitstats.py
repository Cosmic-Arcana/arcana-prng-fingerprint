import numpy as np

from arcana_fp import bitstats


def test_monobit_is_zero_for_a_balanced_stream() -> None:
    bits = np.array([0, 1] * 50, dtype=np.uint8)
    assert bitstats.monobit(bits) == 0.0


def test_monobit_grows_with_imbalance() -> None:
    biased = np.ones(100, dtype=np.uint8)
    assert bitstats.monobit(biased) == 10.0


def test_runs_statistic_flags_a_perfectly_alternating_stream() -> None:
    alternating = np.array([0, 1] * 50, dtype=np.uint8)
    blocky = np.array([0] * 50 + [1] * 50, dtype=np.uint8)
    assert bitstats.runs(alternating) > bitstats.runs(blocky)


def test_degenerate_streams_return_finite_zeros() -> None:
    empty = np.zeros(0, dtype=np.uint8)
    constant = np.zeros(64, dtype=np.uint8)
    for value in (
        bitstats.monobit(empty),
        bitstats.runs(empty),
        bitstats.runs(constant),
        bitstats.serial(empty, 2),
        bitstats.approximate_entropy(empty, 2),
    ):
        assert value == 0.0


def test_approximate_entropy_is_lower_for_a_predictable_stream() -> None:
    rng = np.random.default_rng(7)
    random_bits = rng.integers(0, 2, size=4096).astype(np.uint8)
    predictable = np.tile(np.array([0, 0, 1, 1], dtype=np.uint8), 1024)
    assert bitstats.approximate_entropy(predictable, 2) < bitstats.approximate_entropy(
        random_bits, 2
    )


def test_serial_statistic_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=512).astype(np.uint8)
    assert bitstats.serial(bits, 2) == bitstats.serial(bits, 2)
