import numpy as np
import pytest

from arcana_fp import mt_consistency as mc
from arcana_fp import mt_derived as md


class Mt19937Ref:
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


def test_bit31_taps_are_the_tempered_high_bit() -> None:
    taps = md.bit31_taps()
    for y in (0, 1, 0xDEADBEEF, 0x12345678, 0xFFFFFFFF, 0x80000001):
        assert (mc.temper(y) >> 31) & 1 == sum((y >> j) & 1 for j in taps) & 1


def test_derived_positions_drop_the_redundant_card() -> None:
    positions = md.derived_positions(3)
    assert positions.size == 3 * 77
    # Each draw spans exactly the 77 trace words, contiguous, no gap for orientation.
    assert positions[:77].tolist() == list(range(77))
    assert positions[77:154].tolist() == list(range(77, 154))


def test_symbolic_rows_predict_the_real_derived_bits() -> None:
    n_draws = 40
    rows = mc.observation_matrix(
        md.FY_TRACE_WORDS, n_draws, positions=md.derived_positions(n_draws), taps=md.bit31_taps()
    )
    for seed in (1_000_000, 3_000_011):
        rng = Mt19937Ref(seed)
        state = _state_vector(rng.initial)
        # Replay the derived draw: 77 trace words per draw, reversed[c] = bit31(trace[c % 77]).
        observed = []
        for _ in range(n_draws):
            trace = [rng.next_u32() for _ in range(md.FY_TRACE_WORDS)]
            observed.extend((trace[c] >> 31) & 1 for c in range(md.FY_TRACE_WORDS))
        predicted = (np.bitwise_count(rows & state).sum(axis=1) & 1).tolist()
        assert predicted == observed


def test_card_77_duplicates_card_0_in_the_dataset_rev_string() -> None:
    # The invariant the observation model relies on: with 78 cards and a 77-word
    # Fisher-Yates trace, reversed[77] = bit31(trace[77 % 77]) = bit31(trace[0]) = reversed[0].
    # Verified against the derived dataset so the check binds to real emitted data.
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    path = Path(__file__).resolve().parents[2] / "data" / "r3-derived-state.jsonl"
    if not path.exists():
        pytest.skip("derived dataset not generated")
    with path.open() as handle:
        for _ in range(50):
            sample = json.loads(handle.readline())
            if sample["labels"]["shuffle"] != "fisher-yates":
                continue
            for draw in sample["draws"][:5]:
                assert draw["rev"][77] == draw["rev"][0]


@pytest.mark.parametrize("n_draws", [128, 259])
def test_check_threshold_matches_independent_observations(n_draws: int) -> None:
    checks, stats = md.build_checks(n_draws)
    predicted = max(0, n_draws * md.FY_TRACE_WORDS - mc.EFFECTIVE_STATE_BITS)
    assert stats["checks"] == predicted
    assert checks.rank == min(n_draws * md.FY_TRACE_WORDS, mc.EFFECTIVE_STATE_BITS)
