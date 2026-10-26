from dataclasses import dataclass, field

#here we are just giving values for the parameteres of the experiment. 
#we have the number of requests
#number of blocks
#window size


@dataclass
class ExperimentConfig:
    # Trace
    num_requests: int = 50_000
    num_blocks: int = 5_000
    seed: int = 42

    # Observation/clustering
    window_size: int = 1_000
    n_clusters: int = 4
    kmeans_iterations: int = 100
    kmeans_restarts: int = 8

    # Cache sizes are percentages of the workload working set / block universe.
    cache_percentages: tuple[float, ...] = (0.05, 0.10, 0.20)

    # Cost model. A miss is intentionally much more expensive than a hit.
    hit_cost: float = 1.0
    miss_cost: float = 10.0

    # Adaptive policy mapping. Cluster names are assigned after inspecting
    # cluster centroids against feature profiles.
    policy_map: dict[str, str] = field(default_factory=lambda: {
        "sequential": "MRU",
        "looping": "LRU",
        "random": "LFU",
        "mixed": "ARC",
    })

    output_dir: str = "results"
