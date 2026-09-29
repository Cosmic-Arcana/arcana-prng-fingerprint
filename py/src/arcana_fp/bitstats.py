"""NIST SP 800-22 subset, as raw statistics rather than pass/fail verdicts.

The classifier wants a continuous signal, so each helper returns the test
statistic itself. Empty or degenerate streams return 0.0 so a feature vector is
always finite.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def monobit(bits: np.ndarray) -> float:
    n = bits.size
    if n == 0:
        return 0.0
    return float(abs((2.0 * bits - 1.0).sum()) / np.sqrt(n))


def runs(bits: np.ndarray) -> float:
    n = bits.size
    if n < 2:
        return 0.0
    pi = float(bits.mean())
    if pi <= 0.0 or pi >= 1.0:
        return 0.0
    v_obs = 1.0 + float((bits[1:] != bits[:-1]).sum())
    denom = 2.0 * np.sqrt(2.0 * n) * pi * (1.0 - pi)
    return float((v_obs - 2.0 * n * pi * (1.0 - pi)) / (denom + EPS))


def _block_counts(bits: np.ndarray, m: int) -> np.ndarray:
    n = bits.size
    if m <= 0 or n == 0:
        return np.zeros(1)
    extended = np.concatenate([bits, bits[: m - 1]]) if m > 1 else bits
    idx = np.zeros(n, dtype=np.int64)
    for offset in range(m):
        idx = (idx << 1) | extended[offset : offset + n].astype(np.int64)
    return np.bincount(idx, minlength=1 << m).astype(np.float64)


def _psi_squared(bits: np.ndarray, m: int) -> float:
    n = bits.size
    if m <= 0 or n == 0:
        return 0.0
    counts = _block_counts(bits, m)
    return float((1 << m) / n * (counts**2).sum() - n)


def serial(bits: np.ndarray, m: int) -> float:
    """First-order serial statistic delta-psi^2_m."""
    if bits.size < m:
        return 0.0
    return _psi_squared(bits, m) - _psi_squared(bits, m - 1)


def _phi(bits: np.ndarray, m: int) -> float:
    n = bits.size
    if n == 0:
        return 0.0
    counts = _block_counts(bits, m)
    p = counts / n
    nz = p > 0
    return float((p[nz] * np.log(p[nz])).sum())


def approximate_entropy(bits: np.ndarray, m: int) -> float:
    if bits.size < m + 1:
        return 0.0
    return _phi(bits, m) - _phi(bits, m + 1)
