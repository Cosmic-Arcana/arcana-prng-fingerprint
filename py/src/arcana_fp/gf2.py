"""GF(2) linear-structure statistics: binary matrix rank and linear complexity.

Round 1 showed that frequency-shaped tests (monobit, runs, serial, approximate
entropy) cannot see a generator whose output bits are a linear form over GF(2)
of a short state: those bits are perfectly balanced and perfectly unpredictable
to any test that only counts patterns. The two tests here are the standard ones
that do see it, and neither needs a dependency beyond numpy.

As in `bitstats`, every helper returns a raw statistic rather than a pass/fail
verdict, and a degenerate input returns 0.0 so a feature vector stays finite.
"""

from __future__ import annotations

import numpy as np

# NIST SP 800-22 2.5: rank distribution of a random 32x32 GF(2) matrix.
RANK_FULL_P = 0.2888
RANK_MINUS1_P = 0.5776
RANK_LOWER_P = 0.1336

# NIST SP 800-22 2.10: reference probabilities for the seven T buckets.
LC_BUCKET_P = (0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833)
LC_BUCKET_EDGES = (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)


def berlekamp_massey(bits: list[int]) -> int:
    """Linear complexity of a GF(2) sequence: the length of the shortest LFSR
    that generates it.

    Polynomials are carried as Python ints and the discrepancy is a popcount
    parity, which keeps the inner loop in CPython's bignum C code instead of in
    a per-bit Python loop.
    """
    connection = 1
    previous = 1
    length = 0
    gap = 1
    window = 0
    for index, bit in enumerate(bits):
        window = (window << 1) | bit
        if (connection & window).bit_count() & 1:
            saved = connection
            connection ^= previous << gap
            if 2 * length <= index:
                length = index + 1 - length
                previous = saved
                gap = 1
            else:
                gap += 1
        else:
            gap += 1
    return length


def berlekamp_massey_profile(bits: list[int]) -> tuple[int, int, int]:
    """(final complexity, number of jumps, total jump height) in one pass.

    The profile is more informative than the endpoint: a truly random sequence
    climbs in many small steps, a short LFSR climbs to its degree and then stops.
    """
    connection = 1
    previous = 1
    length = 0
    gap = 1
    window = 0
    jumps = 0
    height = 0
    for index, bit in enumerate(bits):
        window = (window << 1) | bit
        if (connection & window).bit_count() & 1:
            saved = connection
            connection ^= previous << gap
            if 2 * length <= index:
                grown = index + 1 - length
                jumps += 1
                height += grown - length
                length = grown
                previous = saved
                gap = 1
            else:
                gap += 1
        else:
            gap += 1
    return length, jumps, height


def expected_complexity(n: int) -> float:
    """NIST mu_n: the mean linear complexity of a random n-bit sequence."""
    if n <= 0:
        return 0.0
    sign = 1.0 if (n + 1) % 2 == 0 else -1.0
    return n / 2.0 + (9.0 + sign) / 36.0 - (n / 3.0 + 2.0 / 9.0) / (2.0**n)


def complexity_statistic(length: int, n: int) -> float:
    """NIST T_i = (-1)^n (L_i - mu_n) + 2/9."""
    sign = 1.0 if n % 2 == 0 else -1.0
    return sign * (length - expected_complexity(n)) + 2.0 / 9.0


def _blocks(bits: np.ndarray, size: int) -> np.ndarray:
    count = bits.size // size
    if count == 0:
        return np.zeros((0, size), dtype=np.uint8)
    return bits[: count * size].reshape(count, size)


def complexities(bits: np.ndarray, block: int) -> np.ndarray:
    rows = _blocks(bits, block)
    if rows.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return np.array([berlekamp_massey(row.tolist()) for row in rows], dtype=np.float64)


def complexity_buckets(lengths: np.ndarray, n: int) -> np.ndarray:
    """Chi-square of the T histogram against the NIST reference probabilities."""
    if lengths.size == 0:
        return 0.0
    t = np.array([complexity_statistic(int(length), n) for length in lengths])
    counts = np.histogram(t, bins=[-np.inf, *LC_BUCKET_EDGES, np.inf])[0].astype(np.float64)
    expected = np.array(LC_BUCKET_P) * lengths.size
    return float(((counts - expected) ** 2 / expected).sum())


def pack_rows32(bits: np.ndarray) -> np.ndarray:
    """Whole 1024-bit blocks of a stream as (n_blocks, 32) uint32 rows."""
    rows = _blocks(bits, 1024)
    if rows.shape[0] == 0:
        return np.zeros((0, 32), dtype=np.uint32)
    packed = np.packbits(rows.reshape(-1, 32, 32), axis=-1, bitorder="big")
    packed = packed.astype(np.uint32)
    return (
        (packed[..., 0] << np.uint32(24))
        | (packed[..., 1] << np.uint32(16))
        | (packed[..., 2] << np.uint32(8))
        | packed[..., 3]
    )


def rank32_batch(rows: np.ndarray) -> np.ndarray:
    """GF(2) rank of many 32x32 matrices at once.

    Gaussian elimination is driven bit-column by bit-column across every matrix
    simultaneously, so the Python loop runs 32 times in total rather than 32
    times per block.
    """
    if rows.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    work = rows.astype(np.uint32, copy=True)
    count = work.shape[0]
    index = np.arange(work.shape[1])
    pivot = np.zeros(count, dtype=np.int64)
    rank = np.zeros(count, dtype=np.int64)
    for bit in range(31, -1, -1):
        has_bit = ((work >> np.uint32(bit)) & np.uint32(1)).astype(bool)
        eligible = has_bit & (index[None, :] >= pivot[:, None])
        selected = np.nonzero(eligible.any(axis=1))[0]
        if selected.size == 0:
            continue
        first = np.argmax(eligible[selected], axis=1)
        at = pivot[selected]
        swapped = work[selected, at].copy()
        work[selected, at] = work[selected, first]
        work[selected, first] = swapped
        pivots = work[selected, at]
        block = work[selected]
        below = (index[None, :] > at[:, None]) & (
            ((block >> np.uint32(bit)) & np.uint32(1)) != 0
        )
        work[selected] = np.where(below, block ^ pivots[:, None], block)
        pivot[selected] = at + 1
        rank[selected] += 1
    return rank


def rank_stats(bits: np.ndarray) -> tuple[float, float, float, float, float]:
    """(mean rank, frac rank 32, frac rank 31, frac rank <= 30, chi-square)."""
    ranks = rank32_batch(pack_rows32(bits))
    total = ranks.size
    if total == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    full = int((ranks == 32).sum())
    minus1 = int((ranks == 31).sum())
    lower = total - full - minus1
    observed = np.array([full, minus1, lower], dtype=np.float64)
    expected = np.array([RANK_FULL_P, RANK_MINUS1_P, RANK_LOWER_P]) * total
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    return (
        float(ranks.mean()),
        full / total,
        minus1 / total,
        lower / total,
        chi2,
    )
