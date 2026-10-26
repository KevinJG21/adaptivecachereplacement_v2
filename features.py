from __future__ import annotations

from collections import Counter
from math import sqrt
from statistics import mean
from typing import Iterable


FEATURE_NAMES = [
    "unique_ratio",
    "sequentiality",
    "reuse_distance",
    "frequency_concentration",
    "inter_arrival_time",
    "working_set_size",
]


def _reuse_distances(window: list[int]) -> list[float]:
    last_seen: dict[int, int] = {}
    distances = []
    for i, block in enumerate(window):
        if block in last_seen:
            # Number of accesses since the previous reference.
            distances.append(i - last_seen[block])
        last_seen[block] = i
    return distances


def extract_features(window: Iterable[int]) -> dict[str, float]:
    x = list(window)
    n = len(x)
    if n == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    counts = Counter(x)
    unique = len(counts)

    # Adjacent/nearby block accesses are a simple spatial-locality signal.
    adjacent = sum(
        1 for a, b in zip(x, x[1:]) if abs(a - b) <= 1
    )
    sequentiality = adjacent / max(1, n - 1)

    reuse = _reuse_distances(x)
    reuse_distance = mean(reuse) if reuse else float(n)

    # Concentration: fraction of requests accounted for by the hottest 20%.
    k = max(1, int(len(counts) * 0.20))
    hot_requests = sum(v for _, v in counts.most_common(k))
    frequency_concentration = hot_requests / n

    inter_arrival = mean(reuse) if reuse else float(n)

    return {
        "unique_ratio": unique / n,
        "sequentiality": sequentiality,
        "reuse_distance": reuse_distance,
        "frequency_concentration": frequency_concentration,
        "inter_arrival_time": inter_arrival,
        "working_set_size": float(unique),
    }


def window_feature_matrix(trace: list[int], window_size: int):
    rows = []
    starts = range(0, len(trace), window_size)
    for start in starts:
        window = trace[start:start + window_size]
        if len(window) < max(10, window_size // 4):
            continue
        row = extract_features(window)
        row["window_start"] = start
        row["window_end"] = start + len(window)
        rows.append(row)

    if not rows:
        raise ValueError("Trace is too short to create observation windows.")
    return rows


def standardize(rows: list[dict[str, float]]):
    means = {}
    stds = {}
    for name in FEATURE_NAMES:
        values = [float(r[name]) for r in rows]
        mu = mean(values)
        variance = sum((v - mu) ** 2 for v in values) / len(values)
        sd = sqrt(variance) or 1.0
        means[name] = mu
        stds[name] = sd

    matrix = [
        [(float(r[name]) - means[name]) / stds[name] for name in FEATURE_NAMES]
        for r in rows
    ]
    return matrix, means, stds
