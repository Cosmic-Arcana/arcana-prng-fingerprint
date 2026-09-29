"""Hand-built statistics for one sample (N consecutive draws from one stream).

Nothing here is fitted on the dataset: every statistic is computed from a single
sample in isolation, so train/val/test splits cannot leak through the features.
"""

from __future__ import annotations

import numpy as np

from arcana_fp import bitstats

DECK = 78
POSITION_BUCKETS = 6
_FIRST_POSITIONS = 6


def _perm_matrix(sample: dict, n_draws: int) -> tuple[np.ndarray, np.ndarray]:
    draws = sample["draws"][:n_draws]
    order = np.array([draw["order"] for draw in draws], dtype=np.int16)
    reversed_bits = np.array(
        [np.frombuffer(draw["rev"].encode("ascii"), dtype=np.uint8) - ord("0") for draw in draws],
        dtype=np.uint8,
    )
    return order, reversed_bits


def _inversions(order: np.ndarray) -> np.ndarray:
    left = order[:, :, None]
    right = order[:, None, :]
    upper = np.triu(np.ones((order.shape[1], order.shape[1]), dtype=bool), k=1)
    return (((left > right) & upper).sum(axis=(1, 2))).astype(np.float64)


def _cycle_stats(position_of_card: np.ndarray) -> np.ndarray:
    """(n_cycles, max_cycle, fixed_points) per draw. Cycle walking stays a Python
    loop; it is O(deck) per draw and the caller parallelises across samples."""
    out = np.zeros((position_of_card.shape[0], 3), dtype=np.float64)
    deck = position_of_card.shape[1]
    for row in range(position_of_card.shape[0]):
        perm = position_of_card[row]
        seen = np.zeros(deck, dtype=bool)
        n_cycles = 0
        longest = 0
        fixed = 0
        for start in range(deck):
            if seen[start]:
                continue
            n_cycles += 1
            length = 0
            node = start
            while not seen[node]:
                seen[node] = True
                node = int(perm[node])
                length += 1
            longest = max(longest, length)
            if length == 1:
                fixed += 1
        out[row] = (n_cycles, longest, fixed)
    return out


def _lag_autocorr(series: np.ndarray, lag: int) -> np.ndarray:
    """Lag-k autocorrelation along axis 0, per column, nan-safe."""
    if series.shape[0] <= lag + 1:
        return np.zeros(series.shape[1] if series.ndim > 1 else 1)
    a = series[:-lag]
    b = series[lag:]
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a * b).sum(axis=0) / denom
    return np.nan_to_num(out)


def _chi2(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    expected = total / counts.size
    return float(((counts - expected) ** 2 / expected).sum())


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.std())


def feature_names() -> list[str]:
    names: list[str] = []
    for stat in (
        "inversions",
        "fixed_points",
        "n_cycles",
        "max_cycle",
        "rising_sequences",
        "adjacent_consecutive",
        "mean_displacement",
        "max_displacement",
        "first_card",
    ):
        names += [f"{stat}_mean", f"{stat}_std"]
    names += ["chi2_card_position", "chi2_position_bucket", "chi2_first_position", "diag_frac"]
    names += [
        "autocorr1_pos_mean",
        "autocorr1_pos_std",
        "autocorr2_pos_mean",
        "autocorr1_first_card",
        "autocorr1_inversions",
    ]
    names += [
        "rev_mean",
        "rev_monobit",
        "rev_runs",
        "rev_serial2",
        "rev_serial3",
        "rev_apen2",
        "rev_apen3",
        "rev_corr_position",
        "rev_corr_card_id",
    ]
    names += ["lsb_monobit", "lsb_runs", "lsb_serial2", "lsb_apen2"]
    names += ["up_monobit", "up_runs"]
    for i in range(_FIRST_POSITIONS):
        names += [f"pos{i}_mean", f"pos{i}_std"]
    return names


N_FEATURES = len(feature_names())


def extract(sample: dict, n_draws: int) -> np.ndarray:
    order, rev = _perm_matrix(sample, n_draws)
    n, deck = order.shape

    position_of_card = np.empty_like(order, dtype=np.int16)
    rows = np.arange(n)[:, None]
    position_of_card[rows, order] = np.arange(deck, dtype=np.int16)[None, :]

    per_draw: list[np.ndarray] = []
    per_draw.append(_inversions(order) / (deck * (deck - 1) / 2))

    cycles = _cycle_stats(position_of_card.astype(np.int64))
    per_draw.append(cycles[:, 2])
    per_draw.append(cycles[:, 0])
    per_draw.append(cycles[:, 1])

    diffs = np.diff(order.astype(np.int32), axis=1)
    per_draw.append(1.0 + (diffs < 0).sum(axis=1))
    per_draw.append((np.abs(diffs) == 1).mean(axis=1))

    displacement = np.abs(position_of_card.astype(np.int32) - np.arange(deck)[None, :])
    per_draw.append(displacement.mean(axis=1))
    per_draw.append(displacement.max(axis=1))
    per_draw.append(order[:, 0].astype(np.float64))

    values: list[float] = []
    for column in per_draw:
        mean, std = _mean_std(column.astype(np.float64))
        values += [mean, std]

    card_position = np.zeros((deck, deck), dtype=np.float64)
    np.add.at(card_position, (order.ravel(), np.tile(np.arange(deck), n)), 1.0)
    values.append(_chi2(card_position.ravel()))
    bucket = card_position.reshape(deck, POSITION_BUCKETS, deck // POSITION_BUCKETS).sum(axis=2)
    values.append(_chi2(bucket.ravel()))
    values.append(_chi2(np.bincount(order[:, 0], minlength=deck).astype(np.float64)))
    values.append(float(np.trace(card_position) / max(n, 1) / deck))

    positions = position_of_card.astype(np.float64)
    ac1 = _lag_autocorr(positions, 1)
    values += [float(ac1.mean()), float(ac1.std())]
    values.append(float(_lag_autocorr(positions, 2).mean()))
    values.append(float(_lag_autocorr(order[:, 0].astype(np.float64)[:, None], 1)[0]))
    values.append(float(_lag_autocorr(per_draw[0][:, None], 1)[0]))

    rev_flat = rev.ravel().astype(np.uint8)
    values.append(float(rev_flat.mean()))
    values.append(bitstats.monobit(rev_flat))
    values.append(bitstats.runs(rev_flat))
    values.append(bitstats.serial(rev_flat, 2))
    values.append(bitstats.serial(rev_flat, 3))
    values.append(bitstats.approximate_entropy(rev_flat, 2))
    values.append(bitstats.approximate_entropy(rev_flat, 3))
    pos_index = np.tile(np.arange(deck, dtype=np.float64), n)
    values.append(_safe_corr(rev_flat.astype(np.float64), pos_index))
    values.append(_safe_corr(rev_flat.astype(np.float64), order.ravel().astype(np.float64)))

    lsb = (order.ravel() & 1).astype(np.uint8)
    values.append(bitstats.monobit(lsb))
    values.append(bitstats.runs(lsb))
    values.append(bitstats.serial(lsb, 2))
    values.append(bitstats.approximate_entropy(lsb, 2))

    if n > 1:
        moved_up = (np.diff(positions, axis=0) > 0).astype(np.uint8).ravel()
    else:
        moved_up = np.zeros(0, dtype=np.uint8)
    values.append(bitstats.monobit(moved_up))
    values.append(bitstats.runs(moved_up))

    for i in range(_FIRST_POSITIONS):
        mean, std = _mean_std(order[:, i].astype(np.float64))
        values += [mean, std]

    return np.nan_to_num(np.asarray(values, dtype=np.float64), posinf=0.0, neginf=0.0)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom <= 0:
        return 0.0
    return float((a * b).sum() / denom)
