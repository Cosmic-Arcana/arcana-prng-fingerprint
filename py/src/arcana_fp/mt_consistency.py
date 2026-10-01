"""MT19937 state consistency over GF(2).

Round 2 showed that no statistic over per-draw blocks can see MT19937: its
output low bit is a linear form of a 19937-bit state. This module asks the
algebraic question instead — does any MT19937 state produce exactly these
orientation bits?

Twist and tempering are GF(2)-linear, and with a fixed shuffle word budget every
observed orientation bit sits at a known position of the generator stream. Each
observation is therefore a known linear functional of the unknown initial state,
and every vector in the left null space of the observation matrix is a parity
check: every MT19937 sample satisfies it, any other stream fails it with
probability 1/2.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arcana_fp.experiment import _git_sha

DECK = 78
MT_N = 624
MT_M = 397
MATRIX_A = 0x9908B0DF
STATE_BITS = MT_N * 32
STATE_WORDS = STATE_BITS // 64
EFFECTIVE_STATE_BITS = 19937
WORD_BUDGETS = {
    "fisher-yates": 77,
    "modulo-biased": 77,
    "naive-swap-any": 78,
    "random-comparator-sort": 78,
}
OBSERVED_DRAWS = 260
N_CURVE = (128, 255, 256, 257, 260)

_MATRIX_A_BITS = tuple(k for k in range(32) if (MATRIX_A >> k) & 1)
# The twist updates mt[] in place, so later words read already-twisted ones.
# These four slices are the largest ranges whose reads are all of one age.
_TWIST_CHUNKS = (
    np.arange(0, MT_N - MT_M),
    np.arange(MT_N - MT_M, 2 * (MT_N - MT_M)),
    np.arange(2 * (MT_N - MT_M), MT_N - 1),
    np.array([MT_N - 1]),
)


def temper(y: int) -> int:
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= y >> 18
    return y & 0xFFFFFFFF


def low_bit_taps() -> tuple[int, ...]:
    """Raw state-word bits whose XOR is bit 0 of the tempered output."""
    return tuple(j for j in range(32) if temper(1 << j) & 1)


def observation_positions(budget: int, n_draws: int) -> np.ndarray:
    draw = np.arange(n_draws, dtype=np.int64)[:, None]
    card = np.arange(DECK, dtype=np.int64)[None, :]
    return (draw * (budget + DECK) + budget + card).ravel()


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """Bit j of the last axis lands in word j // 64, bit j % 64."""
    n_words = -(-bits.shape[-1] // 64)
    padded = np.zeros(bits.shape[:-1] + (n_words * 64,), dtype=np.uint8)
    padded[..., : bits.shape[-1]] = bits
    return np.packbits(padded, axis=-1, bitorder="little").view("<u8")


def unpack_bits(words: np.ndarray, n_bits: int) -> np.ndarray:
    raw = np.ascontiguousarray(words, dtype="<u8").view(np.uint8)
    return np.unpackbits(raw, axis=-1, count=n_bits, bitorder="little")


def _symbolic_state() -> np.ndarray:
    state = np.zeros((STATE_BITS, STATE_WORDS), dtype=np.uint64)
    bit = np.arange(STATE_BITS)
    state[bit, bit // 64] = np.left_shift(np.uint64(1), (bit % 64).astype(np.uint64))
    return state.reshape(MT_N, 32, STATE_WORDS)


def _twist(state: np.ndarray) -> None:
    for rows in _TWIST_CHUNKS:
        x = np.concatenate(
            [state[(rows + 1) % MT_N, :31], state[rows, 31:32]], axis=1
        )
        xa = np.zeros_like(x)
        xa[:, :31] = x[:, 1:]
        for k in _MATRIX_A_BITS:
            xa[:, k] ^= x[:, 0]
        state[rows] = state[(rows + MT_M) % MT_N] ^ xa


def observation_matrix(
    budget: int,
    n_draws: int,
    positions: np.ndarray | None = None,
    taps: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Row i: the initial-state bits whose XOR is observed orientation bit i.

    Defaults give the independent-bit model (bit 0 of consecutive words). A caller
    may pass its own stream positions and tempering taps for another orientation.
    """
    if positions is None:
        positions = observation_positions(budget, n_draws)
    taps = list(low_bit_taps() if taps is None else taps)
    state = _symbolic_state()
    out = np.empty((positions.size, STATE_WORDS), dtype=np.uint64)
    generation = positions // MT_N
    for g in range(int(generation[-1]) + 1):
        _twist(state)
        selected = np.nonzero(generation == g)[0]
        if selected.size:
            words = positions[selected] % MT_N
            out[selected] = np.bitwise_xor.reduce(state[words][:, taps], axis=1)
    return out


@dataclass(frozen=True)
class ParityChecks:
    n_columns: int
    rank: int
    free_columns: np.ndarray
    checks: np.ndarray

    def available(self, n_columns: int) -> int:
        return int(np.searchsorted(self.free_columns, n_columns))

    def satisfied(self, observed: np.ndarray, n_columns: int) -> bool | None:
        """None when no check fits inside the first `n_columns` observations."""
        k = self.available(n_columns)
        if k == 0:
            return None
        syndrome = np.bitwise_count(self.checks[:k] & observed).sum(axis=1) & 1
        return not syndrome.any()


def _pivot_word_table(t: np.ndarray, pivots: list[tuple[int, int]], word: int) -> np.ndarray:
    table = np.zeros(1 << len(pivots), dtype=np.uint64)
    for j, (_, row) in enumerate(pivots):
        table[1 << j : 2 << j] = table[: 1 << j] ^ t[row, word]
    return table


def _pivot_index(values: np.ndarray, pivots: list[tuple[int, int]], shift: int) -> np.ndarray:
    index = np.zeros(values.shape, dtype=np.intp)
    for j, (bit, _) in enumerate(pivots):
        index |= (((values >> np.uint64(shift + bit)) & np.uint64(1)).astype(np.intp)) << j
    return index


def parity_checks(transposed: np.ndarray, n_columns: int) -> ParityChecks:
    """Null space of a packed GF(2) matrix, one check per free column.

    Gauss-Jordan in column order, so a check whose free column is `f` only
    touches columns <= f, and the checks for any prefix of the columns are a
    prefix of this list. Columns go eight at a time: the block's pivots are kept
    reduced against each other, then every other row clears all of them with one
    lookup into a table of pivot-row combinations (Four Russians). Pivot rows
    are zero left of their block, so only the words from the block onward move.
    """
    t = transposed.copy()
    n_rows = t.shape[0]
    rank = 0
    pivot_columns: list[int] = []
    for start in range(0, n_columns, 8):
        if rank == n_rows:
            break
        word, shift = divmod(start, 64)
        pivots: list[tuple[int, int]] = []
        for bit in range(min(8, n_columns - start)):
            first = rank + len(pivots)
            candidates = t[first:, word]
            reduced = candidates ^ _pivot_word_table(t, pivots, word)[
                _pivot_index(candidates, pivots, shift)
            ]
            hits = np.nonzero((reduced >> np.uint64(shift + bit)) & np.uint64(1))[0]
            if hits.size == 0:
                continue
            chosen = first + int(hits[0])
            if chosen != first:
                t[[first, chosen]] = t[[chosen, first]]
            for pivot_bit, row in pivots:
                if (int(t[first, word]) >> (shift + pivot_bit)) & 1:
                    t[first, word:] ^= t[row, word:]
            for _, row in pivots:
                if (int(t[row, word]) >> (shift + bit)) & 1:
                    t[row, word:] ^= t[first, word:]
            pivots.append((bit, first))
        if not pivots:
            continue
        table = np.zeros((1 << len(pivots), t.shape[1] - word), dtype=np.uint64)
        for j, (_, row) in enumerate(pivots):
            table[1 << j : 2 << j] = table[: 1 << j] ^ t[row, word:]
        index = _pivot_index(t[:, word], pivots, shift)
        index[rank : rank + len(pivots)] = 0
        rows = np.nonzero(index)[0]
        t[rows, word:] ^= table[index[rows]]
        pivot_columns.extend(start + bit for bit, _ in pivots)
        rank += len(pivots)

    pivot_columns_arr = np.asarray(pivot_columns, dtype=np.int64)
    free = np.setdiff1d(np.arange(n_columns, dtype=np.int64), pivot_columns_arr)
    bits = np.zeros((free.size, n_columns), dtype=np.uint8)
    if free.size:
        pivot_rows = t[:rank][:, free // 64]
        coefficients = (pivot_rows >> (free % 64).astype(np.uint64)) & np.uint64(1)
        bits[:, pivot_columns_arr] = coefficients.T.astype(np.uint8)
        bits[np.arange(free.size), free] = 1
    return ParityChecks(
        n_columns=n_columns, rank=rank, free_columns=free, checks=pack_bits(bits)
    )


def build_checks(budget: int, n_draws: int = OBSERVED_DRAWS) -> tuple[ParityChecks, dict]:
    started = time.perf_counter()
    rows = observation_matrix(budget, n_draws)
    built = time.perf_counter()
    n_columns = rows.shape[0]
    checks = parity_checks(pack_bits(unpack_bits(rows, STATE_BITS).T), n_columns)
    done = time.perf_counter()
    return checks, {
        "budget": budget,
        "observations": n_columns,
        "rank": checks.rank,
        "checks": int(checks.free_columns.size),
        "symbolic_seconds": round(built - started, 1),
        "elimination_seconds": round(done - built, 1),
    }


def observed_bits(sample: dict, n_draws: int) -> np.ndarray:
    rev = "".join(draw["rev"] for draw in sample["draws"][:n_draws])
    return pack_bits(np.frombuffer(rev.encode(), dtype=np.uint8) - ord("0"))


def _score(records: list[dict], n_curve: tuple[int, ...]) -> dict:
    out: dict = {}
    for split in ("train", "val", "test"):
        rows = [r for r in records if r["split"] == split]
        per_n = {}
        for n in n_curve:
            mt = [r for r in rows if r["generator"] == "mt19937"]
            others = [r for r in rows if r["generator"] != "mt19937"]
            false_flags: dict[str, int] = {}
            for r in others:
                if r["flagged"][str(n)]:
                    false_flags[r["generator"]] = false_flags.get(r["generator"], 0) + 1
            per_n[str(n)] = {
                "mt19937_recall": round(
                    sum(r["flagged"][str(n)] for r in mt) / max(len(mt), 1), 4
                ),
                "mt19937_samples": len(mt),
                "non_mt_samples": len(others),
                "non_mt_flagged": sum(false_flags.values()),
                "false_flags_by_generator": false_flags,
            }
        out[split] = per_n
    return out


def _budget_agreement(records: list[dict], n: int) -> dict:
    agreement = {"matched_true_budget_only": 0, "matched_wrong_budget": 0, "unflagged": 0}
    for r in records:
        if r["generator"] != "mt19937":
            continue
        matched = r["matched_budgets"][str(n)]
        if not matched:
            agreement["unflagged"] += 1
        elif matched == [WORD_BUDGETS[r["shuffle"]]]:
            agreement["matched_true_budget_only"] += 1
        else:
            agreement["matched_wrong_budget"] += 1
    return agreement


def run(dataset: Path, run_id: str, out_dir: Path, repo: Path, n_curve: tuple[int, ...]) -> dict:
    started = time.perf_counter()
    n_draws = max(n_curve)
    budgets = sorted(set(WORD_BUDGETS.values()))
    with ProcessPoolExecutor(max_workers=len(budgets)) as pool:
        built = list(pool.map(build_checks, budgets, [n_draws] * len(budgets)))
    checks = {budget: result[0] for budget, result in zip(budgets, built, strict=True)}
    elimination = []
    for budget, (parity, stats) in zip(budgets, built, strict=True):
        stats["checks_at_n"] = {str(n): parity.available(n * DECK) for n in n_curve}
        stats["predicted_checks_at_n"] = {
            str(n): max(0, n * DECK - EFFECTIVE_STATE_BITS) for n in n_curve
        }
        elimination.append(stats)

    records = []
    with dataset.open() as handle:
        for line in handle:
            sample = json.loads(line)
            observed = observed_bits(sample, n_draws)
            flagged, matched = {}, {}
            for n in n_curve:
                hits = [b for b in budgets if checks[b].satisfied(observed, n * DECK)]
                matched[str(n)] = hits
                flagged[str(n)] = bool(hits)
            records.append(
                {
                    "split": sample["split"],
                    "generator": sample["labels"]["generator"],
                    "shuffle": sample["labels"]["shuffle"],
                    "flagged": flagged,
                    "matched_budgets": matched,
                }
            )

    headline_n = str(n_draws)
    scores = _score(records, n_curve)
    test = scores["test"][headline_n]
    payload = {
        "id": run_id,
        "gitSha": _git_sha(repo),
        "config": {
            "dataset": str(dataset),
            "n_samples": len(records),
            "observed_draws": n_draws,
            "n_curve": list(n_curve),
            "word_budgets": WORD_BUDGETS,
            "effective_state_bits": EFFECTIVE_STATE_BITS,
            "low_bit_taps": list(low_bit_taps()),
            "decision": "flag mt19937 iff every available check holds under some budget",
        },
        "elimination": elimination,
        "scores": scores,
        "budget_agreement_test": _budget_agreement(
            [r for r in records if r["split"] == "test"], n_draws
        ),
        "headline": {
            "n": n_draws,
            "test_mt19937_recall": test["mt19937_recall"],
            "test_non_mt_flagged": test["non_mt_flagged"],
            "test_chacha20_flagged": test["false_flags_by_generator"].get("chacha20", 0),
            "h8_confirmed": test["mt19937_recall"] == 1.0 and test["non_mt_flagged"] == 0,
        },
        "wallSeconds": round(time.perf_counter() - started, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_id}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(prog="arcana-fp-mt")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", default="r4-mt-consistency")
    parser.add_argument("--out", default="lab/runs")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--n-curve", default=",".join(str(n) for n in N_CURVE))
    args = parser.parse_args()
    payload = run(
        Path(args.dataset),
        args.run_id,
        Path(args.out),
        Path(args.repo),
        tuple(int(n) for n in args.n_curve.split(",")),
    )
    print(json.dumps({k: payload[k] for k in ("headline", "elimination", "wallSeconds")}, indent=2))


if __name__ == "__main__":
    main()
