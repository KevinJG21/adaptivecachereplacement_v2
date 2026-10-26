from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Callable

from adaptive_cache import AdaptiveCache
from cache_policies import make_policy


@dataclass
class SimulationResult:
    policy: str
    cache_capacity: int
    requests: int
    hits: int
    misses: int
    hit_ratio: float
    miss_ratio: float
    average_access_cost: float
    policy_switch_count: int = 0
    clustering_time_ms: float = 0.0
    adaptation_time_ms: float = 0.0

    def to_dict(self):
        return asdict(self)


def simulate_fixed(trace: list[int], policy_name: str, capacity: int,
                   hit_cost: float = 1.0, miss_cost: float = 10.0) -> SimulationResult:
    cache = make_policy(policy_name, capacity)
    hits = 0
    total_cost = 0.0

    for block in trace:
        if cache.access(block):
            hits += 1
            total_cost += hit_cost
        else:
            total_cost += miss_cost

    n = len(trace)
    return SimulationResult(
        policy=policy_name,
        cache_capacity=capacity,
        requests=n,
        hits=hits,
        misses=n - hits,
        hit_ratio=hits / n,
        miss_ratio=(n - hits) / n,
        average_access_cost=total_cost / n,
    )


def simulate_adaptive(
    trace: list[int],
    capacity: int,
    window_size: int,
    window_rows: list[dict],
    cluster_names: dict[int, str],
    policy_map: dict[str, str],
    labels: list[int],
    hit_cost: float = 1.0,
    miss_cost: float = 10.0,
) -> SimulationResult:
    cache = AdaptiveCache(capacity, "LRU")
    hits = 0
    total_cost = 0.0
    switch_count_before = 0

    start_time = perf_counter()

    row_index = 0
    for start in range(0, len(trace), window_size):
        end = min(len(trace), start + window_size)

        if row_index < len(labels):
            cluster_id = labels[row_index]
            semantic_name = cluster_names[cluster_id]
            desired_policy = policy_map.get(semantic_name, "LRU")

            cache.switch_policy(
                desired_policy,
                position=start,
                reason=semantic_name,
            )

        for block in trace[start:end]:
            if cache.access(block):
                hits += 1
                total_cost += hit_cost
            else:
                total_cost += miss_cost

        row_index += 1

    elapsed_ms = (perf_counter() - start_time) * 1000.0
    n = len(trace)

    return SimulationResult(
        policy="ADAPTIVE",
        cache_capacity=capacity,
        requests=n,
        hits=hits,
        misses=n - hits,
        hit_ratio=hits / n,
        miss_ratio=(n - hits) / n,
        average_access_cost=total_cost / n,
        policy_switch_count=cache.switch_count,
        adaptation_time_ms=elapsed_ms,
    )
