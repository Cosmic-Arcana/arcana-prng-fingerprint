import pytest

from arcana_fp.stats import contains_chance, wilson_interval


def test_wilson_interval_is_symmetric_at_a_half() -> None:
    low, high = wilson_interval(50, 100)
    assert abs((low + high) / 2 - 0.5) < 1e-12
    assert 0.39 < low < 0.41
    assert 0.59 < high < 0.61


def test_wilson_interval_narrows_as_n_grows() -> None:
    small = wilson_interval(50, 100)
    large = wilson_interval(5000, 10000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_interval_handles_the_edges() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(10, 10)
    assert high == pytest.approx(1.0)
    assert low > 0.5


def test_contains_chance() -> None:
    assert contains_chance((0.4, 0.6), 0.5)
    assert not contains_chance((0.6, 0.7), 0.5)
