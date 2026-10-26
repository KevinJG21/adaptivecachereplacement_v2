from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass


def _sqdist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


@dataclass
class KMeans:
    n_clusters: int = 4
    max_iter: int = 100
    n_init: int = 8
    seed: int = 42

    def fit(self, X: list[list[float]]):
        if not X:
            raise ValueError("Cannot cluster empty data.")
        k = min(self.n_clusters, len(X))
        rng = random.Random(self.seed)

        best_labels = None
        best_centroids = None
        best_inertia = float("inf")

        for _ in range(self.n_init):
            centroids = [list(p) for p in rng.sample(X, k)]

            for _ in range(self.max_iter):
                labels = [
                    min(range(k), key=lambda c: _sqdist(x, centroids[c]))
                    for x in X
                ]

                new_centroids = []
                for c in range(k):
                    members = [x for x, label in zip(X, labels) if label == c]
                    if not members:
                        new_centroids.append(list(rng.choice(X)))
                    else:
                        new_centroids.append([
                            sum(row[j] for row in members) / len(members)
                            for j in range(len(X[0]))
                        ])

                if all(
                    _sqdist(a, b) < 1e-12
                    for a, b in zip(centroids, new_centroids)
                ):
                    centroids = new_centroids
                    break
                centroids = new_centroids

            inertia = sum(_sqdist(x, centroids[label])
                          for x, label in zip(X, labels))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_clusters_ = k
        return self

    def predict(self, X: list[list[float]]) -> list[int]:
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("KMeans must be fitted before predict().")
        return [
            min(range(self.n_clusters_),
                key=lambda c: _sqdist(x, self.cluster_centers_[c]))
            for x in X
        ]


def classify_clusters(rows, labels):
    """
    Convert anonymous K-Means clusters into semantic workload labels.

    This is a heuristic naming step:
    - sequential: high sequentiality + relatively low reuse/frequency signals
    - looping: high frequency concentration + low reuse distance
    - random: high unique ratio/reuse distance + low sequentiality
    - mixed: remaining/ambiguous cluster

    The final policy mapping remains a hypothesis to be tested.
    """
    profiles = {}
    for cluster in sorted(set(labels)):
        members = [r for r, y in zip(rows, labels) if y == cluster]
        profiles[cluster] = {
            key: sum(r[key] for r in members) / len(members)
            for key in [
                "unique_ratio", "sequentiality", "reuse_distance",
                "frequency_concentration", "inter_arrival_time",
                "working_set_size"
            ]
        }

    names = {}
    unused = set(profiles)

    if unused:
        seq = max(unused, key=lambda c: profiles[c]["sequentiality"])
        names[seq] = "sequential"
        unused.remove(seq)

    if unused:
        loop = max(
            unused,
            key=lambda c: (
                profiles[c]["frequency_concentration"]
                - 0.25 * profiles[c]["reuse_distance"]
            )
        )
        names[loop] = "looping"
        unused.remove(loop)

    if unused:
        rnd = max(
            unused,
            key=lambda c: (
                profiles[c]["unique_ratio"]
                + profiles[c]["reuse_distance"]
                - profiles[c]["sequentiality"]
            )
        )
        names[rnd] = "random"
        unused.remove(rnd)

    for c in unused:
        names[c] = "mixed"

    return names, profiles
