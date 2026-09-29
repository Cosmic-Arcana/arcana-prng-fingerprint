from __future__ import annotations

import math

Z95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval. Preferred over the normal approximation here because
    the CSPRNG control is expected to sit exactly at p=0.5 with a modest n."""
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def contains_chance(interval: tuple[float, float], chance: float) -> bool:
    return interval[0] <= chance <= interval[1]
