"""MT19937 state consistency for `derived` orientation.

R4 (`mt_consistency`) covers independent-bit orientation, where each reversed flag
is the low bit of a fresh word. Derived orientation reuses the shuffle's own words:
with Fisher-Yates on 78 cards the shuffle draws 77 words (the trace), and
`reversed[i] = bit31(trace[i % 77])`. So no extra words are consumed, the observed
bit is tempered bit 31 rather than bit 0, and card 77 reuses trace[0], making its
flag identical to card 0's.

This is the same detection/measurement question as R4 for a different orientation.
It tests state consistency and labels samples; it does not recover state.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from arcana_fp.experiment import _git_sha
from arcana_fp.mt_consistency import (
    DECK,
    EFFECTIVE_STATE_BITS,
    MT_N,
    STATE_BITS,
    ParityChecks,
    observation_matrix,
    pack_bits,
    parity_checks,
    temper,
    unpack_bits,
)

# Fisher-Yates draws one word per swap for i = 77..1, so the trace holds 77 words.
FY_TRACE_WORDS = DECK - 1
OBSERVED_DRAWS = 320
N_CURVE = (128, 258, 259, 260, 320)


def bit31_taps() -> tuple[int, ...]:
    """Raw state-word bits whose XOR is bit 31 of the tempered output."""
    return tuple(j for j in range(32) if (temper(1 << j) >> 31) & 1)


def derived_positions(n_draws: int) -> np.ndarray:
    """Stream offset of each observed orientation bit, with the redundant 78th
    card of every draw dropped.

    A draw is 77 Fisher-Yates words. Card c reads trace[c % 77], and the trace's
    word t (t = 0..76) is generator word `draw * 77 + t`. Card 77 duplicates card
    0, so only cards 0..76 are kept: 77 independent observations per draw.
    """
    draw = np.arange(n_draws, dtype=np.int64)[:, None]
    card = np.arange(FY_TRACE_WORDS, dtype=np.int64)[None, :]
    return (draw * FY_TRACE_WORDS + card).ravel()


def build_checks(n_draws: int = OBSERVED_DRAWS) -> tuple[ParityChecks, dict]:
    started = time.perf_counter()
    positions = derived_positions(n_draws)
    rows = observation_matrix(FY_TRACE_WORDS, n_draws, positions=positions, taps=bit31_taps())
    built = time.perf_counter()
    n_columns = rows.shape[0]
    checks = parity_checks(pack_bits(unpack_bits(rows, STATE_BITS).T), n_columns)
    done = time.perf_counter()
    return checks, {
        "orientation": "derived",
        "shuffle": "fisher-yates",
        "observations": n_columns,
        "independent_per_draw": FY_TRACE_WORDS,
        "rank": checks.rank,
        "checks": int(checks.free_columns.size),
        "symbolic_seconds": round(built - started, 1),
        "elimination_seconds": round(done - built, 1),
    }


def observed_bits(sample: dict, n_draws: int) -> np.ndarray:
    # The read model stores all 78 flags; keep the 77 independent ones per draw.
    kept = "".join(draw["rev"][:FY_TRACE_WORDS] for draw in sample["draws"][:n_draws])
    return pack_bits(np.frombuffer(kept.encode(), dtype=np.uint8) - ord("0"))


def _score(records: list[dict], n_curve: tuple[int, ...]) -> dict:
    out: dict = {}
    for split in ("train", "val", "test"):
        rows = [r for r in records if r["split"] == split]
        mt = [r for r in rows if r["generator"] == "mt19937"]
        others = [r for r in rows if r["generator"] != "mt19937"]
        per_n = {}
        for n in n_curve:
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


def run(dataset: Path, run_id: str, out_dir: Path, repo: Path, n_curve: tuple[int, ...]) -> dict:
    started = time.perf_counter()
    n_draws = max(n_curve)
    with ProcessPoolExecutor(max_workers=1) as pool:
        checks, stats = list(pool.map(build_checks, [n_draws]))[0]
    stats["checks_at_n"] = {str(n): checks.available(n * FY_TRACE_WORDS) for n in n_curve}
    stats["predicted_checks_at_n"] = {
        str(n): max(0, n * FY_TRACE_WORDS - EFFECTIVE_STATE_BITS) for n in n_curve
    }

    records = []
    with dataset.open() as handle:
        for line in handle:
            sample = json.loads(line)
            if sample["labels"]["shuffle"] != "fisher-yates":
                continue
            observed = observed_bits(sample, n_draws)
            flagged = {
                str(n): checks.satisfied(observed, n * FY_TRACE_WORDS) is True for n in n_curve
            }
            records.append(
                {
                    "split": sample["split"],
                    "generator": sample["labels"]["generator"],
                    "flagged": flagged,
                }
            )

    headline_n = str(n_draws)
    scores = _score(records, n_curve)
    test = scores["test"][headline_n]
    first_check_n = next((n for n in n_curve if stats["checks_at_n"][str(n)] > 0), None)
    payload = {
        "id": run_id,
        "gitSha": _git_sha(repo),
        "config": {
            "dataset": str(dataset),
            "orientation": "derived",
            "shuffle": "fisher-yates",
            "n_samples": len(records),
            "observed_draws": n_draws,
            "n_curve": list(n_curve),
            "independent_bits_per_draw": FY_TRACE_WORDS,
            "effective_state_bits": EFFECTIVE_STATE_BITS,
            "bit31_taps": list(bit31_taps()),
            "decision": "flag mt19937 iff every available derived check holds",
        },
        "elimination": stats,
        "scores": scores,
        "headline": {
            "n": n_draws,
            "test_mt19937_recall": test["mt19937_recall"],
            "test_non_mt_flagged": test["non_mt_flagged"],
            "test_chacha20_flagged": test["false_flags_by_generator"].get("chacha20", 0),
            "first_check_at_n": first_check_n,
            "predicted_threshold_draws": -(-EFFECTIVE_STATE_BITS // FY_TRACE_WORDS),
            "h12_confirmed": test["mt19937_recall"] == 1.0 and test["non_mt_flagged"] == 0,
        },
        "wallSeconds": round(time.perf_counter() - started, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_id}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(prog="arcana-fp-mt-derived")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", default="r3-derived-state")
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
