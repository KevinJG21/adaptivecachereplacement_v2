from __future__ import annotations

import os
from collections import defaultdict

import matplotlib.pyplot as plt


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_hit_ratio(results, path):
    grouped = defaultdict(list)
    for r in results:
        grouped[r["policy"]].append(r)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    for policy, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda x: x["cache_fraction"])
        ax.plot(
            [r["cache_fraction"] * 100 for r in rows],
            [r["hit_ratio"] * 100 for r in rows],
            marker="o",
            label=policy,
        )

    ax.set_xlabel("Cache size (% of workload working set)")
    ax.set_ylabel("Hit ratio (%)")
    ax.set_title("Cache Hit Ratio vs Cache Size")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_access_cost(results, path):
    grouped = defaultdict(list)
    for r in results:
        grouped[r["policy"]].append(r)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    for policy, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda x: x["cache_fraction"])
        ax.plot(
            [r["cache_fraction"] * 100 for r in rows],
            [r["average_access_cost"] for r in rows],
            marker="o",
            label=policy,
        )

    ax.set_xlabel("Cache size (% of workload working set)")
    ax.set_ylabel("Average access cost")
    ax.set_title("Average Access Cost vs Cache Size")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_cluster_timeline(assignments, path):
    if not assignments:
        return

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    name_to_y = {"sequential": 0, "looping": 1, "random": 2, "mixed": 3}
    xs = [a["window_start"] for a in assignments]
    ys = [name_to_y.get(a["cluster_name"], 4) for a in assignments]

    ax.step(xs, ys, where="post")
    ax.set_yticks(list(name_to_y.values()))
    ax.set_yticklabels(list(name_to_y.keys()))
    ax.set_xlabel("Trace request index")
    ax.set_ylabel("Detected workload class")
    ax.set_title("Detected Access-Pattern Clusters")
    ax.grid(True, alpha=0.25)
    _save(fig, path)


def plot_policy_switches(assignments, policy_map, path):
    if not assignments:
        return

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    names = [policy_map.get(a["cluster_name"], "LRU") for a in assignments]
    policy_ids = {p: i for i, p in enumerate(sorted(set(names)))}
    ys = [policy_ids[p] for p in names]
    xs = [a["window_start"] for a in assignments]

    ax.step(xs, ys, where="post")
    ax.set_yticks(list(policy_ids.values()))
    ax.set_yticklabels(list(policy_ids.keys()))
    ax.set_xlabel("Trace request index")
    ax.set_ylabel("Selected cache policy")
    ax.set_title("Adaptive Policy Selection")
    ax.grid(True, alpha=0.25)
    _save(fig, path)
