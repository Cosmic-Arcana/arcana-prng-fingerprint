import numpy as np

from arcana_fp import gf2


def _lfsr(taps: tuple[int, ...], degree: int, count: int) -> list[int]:
    state = [1] + [0] * (degree - 1)
    out = []
    for _ in range(count):
        out.append(state[0])
        feedback = 0
        for tap in taps:
            feedback ^= state[tap]
        state = state[1:] + [feedback]
    return out


def test_complexity_of_a_known_lfsr_is_its_degree() -> None:
    bits = _lfsr((0, 3), 5, 120)
    assert gf2.berlekamp_massey(bits[:60]) == 5


def test_complexity_of_a_constant_stream_is_one() -> None:
    assert gf2.berlekamp_massey([1] * 64) == 1
    assert gf2.berlekamp_massey([0] * 64) == 0


def test_complexity_of_an_alternating_stream_is_two() -> None:
    assert gf2.berlekamp_massey([0, 1] * 32) == 2


def test_random_complexity_sits_near_the_expected_curve() -> None:
    rng = np.random.default_rng(7)
    lengths = [gf2.berlekamp_massey(rng.integers(0, 2, 78).tolist()) for _ in range(200)]
    assert abs(float(np.mean(lengths)) - gf2.expected_complexity(78)) < 1.5


def test_profile_agrees_with_the_endpoint_and_counts_jumps() -> None:
    bits = _lfsr((0, 3), 5, 120)[:60]
    length, jumps, height = gf2.berlekamp_massey_profile(bits)
    assert length == gf2.berlekamp_massey(bits)
    assert jumps >= 1
    assert height == length


def test_expected_complexity_matches_the_nist_formula() -> None:
    assert abs(gf2.expected_complexity(78) - 39.2222222) < 1e-5
    assert abs(gf2.expected_complexity(624) - 312.2222222) < 1e-5


def test_rank_of_an_identity_block_is_full_and_a_zero_block_is_empty() -> None:
    identity = np.eye(32, dtype=np.uint8).reshape(-1)
    assert gf2.rank32_batch(gf2.pack_rows32(identity)).tolist() == [32]
    assert gf2.rank32_batch(gf2.pack_rows32(np.zeros(1024, dtype=np.uint8))).tolist() == [0]


def test_rank_sees_a_repeated_row_as_a_deficiency() -> None:
    rng = np.random.default_rng(3)
    rows = rng.integers(0, 2, (32, 32)).astype(np.uint8)
    rows[5] = rows[9]
    ranks = gf2.rank32_batch(gf2.pack_rows32(rows.reshape(-1)))
    assert int(ranks[0]) <= 31


def test_random_ranks_follow_the_nist_distribution() -> None:
    rng = np.random.default_rng(11)
    mean_rank, full, minus1, lower, _ = gf2.rank_stats(
        rng.integers(0, 2, 1024 * 400).astype(np.uint8)
    )
    assert abs(full - gf2.RANK_FULL_P) < 0.06
    assert abs(minus1 - gf2.RANK_MINUS1_P) < 0.06
    assert abs(lower - gf2.RANK_LOWER_P) < 0.06
    assert 30.5 < mean_rank < 32.0


def test_degenerate_streams_return_finite_zeros() -> None:
    empty = np.zeros(0, dtype=np.uint8)
    assert gf2.complexities(empty, 78).size == 0
    assert gf2.rank_stats(empty) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert gf2.rank_stats(np.zeros(100, dtype=np.uint8)) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert gf2.expected_complexity(0) == 0.0
    assert gf2.complexity_buckets(np.zeros(0), 78) == 0.0
