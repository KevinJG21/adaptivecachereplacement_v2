from __future__ import annotations

import csv
import os
from time import perf_counter

from clustering import KMeans, classify_clusters
from features import FEATURE_NAMES, standardize, window_feature_matrix
from simulator import simulate_fixed, simulate_adaptive


BASELINES = ["LRU", "LFU", "MRU", "ARC"]


def cache_capacity_from_fraction(trace: list[int], fraction: float) -> int:
    working_set = len(set(trace))
    return max(1, int(round(working_set * fraction)))


def prepare_clustering(trace, window_size, n_clusters, max_iter, n_init, seed):
    rows = window_feature_matrix(trace, window_size)
    X, means, stds = standardize(rows)

    timer = perf_counter()
    model = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        n_init=n_init,
        seed=seed,
    ).fit(X)
    clustering_ms = (perf_counter() - timer) * 1000.0

    names, profiles = classify_clusters(rows, model.labels_)

    assignments = []
    for row, label in zip(rows, model.labels_):
        assignments.append({
            **row,
            "cluster_id": label,
            "cluster_name": names[label],
        })

    return {
        "rows": rows,
        "X": X,
        "means": means,
        "stds": stds,
        "model": model,
        "cluster_names": names,
        "profiles": profiles,
        "assignments": assignments,
        "clustering_ms": clustering_ms,
    }


def run_experiment(
    trace: list[int],
    window_size: int,
    n_clusters: int,
    max_iter: int,
    n_init: int,
    seed: int,
    cache_percentages: tuple[float, ...],
    policy_map: dict[str, str],
    hit_cost: float,
    miss_cost: float,
):
    clustering = prepare_clustering(
        trace, window_size, n_clusters, max_iter, n_init, seed
    )

    results = []
    for fraction in cache_percentages:
        capacity = cache_capacity_from_fraction(trace, fraction)

        for policy in BASELINES:
            r = simulate_fixed(
                trace, policy, capacity, hit_cost, miss_cost
            )
            d = r.to_dict()
            d["cache_fraction"] = fraction
            d["clustering_time_ms"] = clustering["clustering_ms"]
            results.append(d)

        adaptive = simulate_adaptive(
            trace=trace,
            capacity=capacity,
            window_size=window_size,
            window_rows=clustering["rows"],
            cluster_names=clustering["cluster_names"],
            policy_map=policy_map,
            labels=clustering["model"].labels_,
            hit_cost=hit_cost,
            miss_cost=miss_cost,
        )
        d = adaptive.to_dict()
        d["cache_fraction"] = fraction
        d["clustering_time_ms"] = clustering["clustering_ms"]
        results.append(d)

    return results, clustering


def write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
