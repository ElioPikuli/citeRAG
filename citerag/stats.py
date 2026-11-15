"""Small statistics helpers shared by answer evaluation and request tracing."""

from __future__ import annotations

import math


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: the ceil(p/100*n)-th smallest value. None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]
