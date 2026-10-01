import numpy as np
import pytest

from arcana_fp import mt_consistency as mc


class _Mt19937:
    def __init__(self, seed: int) -> None:
        self.mt = [0] * 624
        self.index = 624
        self.mt[0] = seed & 0xFFFFFFFF
        for i in range(1, 624):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self.initial = list(self.mt)

    def next_u32(self) -> int:
        if self.index >= 624:
            for i in range(624):
                x = (self.mt[i] & 0x80000000) + (self.mt[(i + 1) % 624] & 0x7FFFFFFF)
                xa = x >> 1
                if x & 1:
                    xa ^= mc.MATRIX_A
                self.mt[i] = self.mt[(i + 397) % 624] ^ xa
            self.index = 0
        y = self.mt[self.index]
        self.index += 1
        return mc.temper(y)


def _state_vector(words: list[int]) -> np.ndarray:
    bits = np.array([(w >> b) & 1 for w in words for b in range(32)], dtype=np.uint8)
    return mc.pack_bits(bits)


def _rank_reference(rows: list[int]) -> int:
    basis: dict[int, int] = {}
    for row in rows:
        while row:
            top = row.bit_length() - 1
            if top not in basis:
                basis[top] = row
                break
            row ^= basis[top]
    return len(basis)


def test_reference_mt_reproduces_the_published_vector() -> None:
    rng = _Mt19937(5489)
    assert [rng.next_u32() for _ in range(3)] == [3499211612, 581869302, 3890346734]


def test_low_bit_taps_are_the_tempered_low_bit() -> None:
    taps = mc.low_bit_taps()
    for y in (0, 1, 0xDEADBEEF, 0x12345678, 0xFFFFFFFF, 0x80000001):
        assert mc.temper(y) & 1 == sum((y >> j) & 1 for j in taps) & 1


@pytest.mark.parametrize("budget", [77, 78])
def test_symbolic_rows_predict_every_observed_bit_of_the_real_generator(budget: int) -> None:
    n_draws = 30
    rows = mc.observation_matrix(budget, n_draws)
    for seed in (1_000_000, 3_000_017):
        rng = _Mt19937(seed)
        state = _state_vector(rng.initial)
        stream = [rng.next_u32() & 1 for _ in range(n_draws * (budget + mc.DECK))]
        observed = [stream[p] for p in mc.observation_positions(budget, n_draws)]
        predicted = (np.bitwise_count(rows & state).sum(axis=1) & 1).tolist()
        assert predicted == observed


def test_bit_packing_round_trips() -> None:
    bits = np.random.default_rng(0).integers(0, 2, size=(5, 131), dtype=np.uint8)
    assert np.array_equal(mc.unpack_bits(mc.pack_bits(bits), 131), bits)


@pytest.mark.parametrize(("n_rows", "n_columns"), [(20, 50), (70, 200), (150, 140), (64, 64)])
def test_parity_checks_span_the_null_space(n_rows: int, n_columns: int) -> None:
    rng = np.random.default_rng(n_rows * 1000 + n_columns)
    a = rng.integers(0, 2, size=(n_columns, n_rows), dtype=np.uint8)
    a[:, 3] = a[:, 1] ^ a[:, 2]
    checks = mc.parity_checks(mc.pack_bits(a.T), n_columns)

    rank = _rank_reference([int("".join(map(str, col[::-1])), 2) for col in a.T])
    assert checks.rank == rank
    assert checks.free_columns.size == n_columns - rank

    check_bits = mc.unpack_bits(checks.checks, n_columns)
    assert not ((check_bits.astype(np.int64) @ a.astype(np.int64)) & 1).any()
    assert _rank_reference(
        [int("".join(map(str, row[::-1])), 2) for row in check_bits]
    ) == checks.free_columns.size
    for k, free in enumerate(checks.free_columns):
        assert check_bits[k, free] == 1
        assert not check_bits[k, free + 1 :].any()


def test_consistent_observations_pass_and_random_ones_fail() -> None:
    rng = np.random.default_rng(7)
    n_rows, n_columns = 40, 120
    a = rng.integers(0, 2, size=(n_columns, n_rows), dtype=np.uint8)
    checks = mc.parity_checks(mc.pack_bits(a.T), n_columns)
    secret = rng.integers(0, 2, size=n_rows, dtype=np.uint8)
    consistent = mc.pack_bits((a.astype(np.int64) @ secret) & 1)
    assert checks.satisfied(consistent, n_columns) is True
    noise = mc.pack_bits(rng.integers(0, 2, size=n_columns, dtype=np.uint8))
    assert checks.satisfied(noise, n_columns) is False


def test_no_checks_inside_a_prefix_means_abstain() -> None:
    a = np.eye(16, dtype=np.uint8)
    checks = mc.parity_checks(mc.pack_bits(a.T), 16)
    assert checks.satisfied(mc.pack_bits(np.zeros(16, dtype=np.uint8)), 16) is None
