from __future__ import annotations

import os

from config import ExperimentConfig
from evaluation import run_experiment, write_csv
from plotting import plot_hit_ratio, plot_access_cost
from trace_generator import generate_trace


def main():
    cfg = ExperimentConfig()
    workloads = ["sequential", "looping", "random", "mixed"]
    all_results = []

    for workload in workloads:
        print(f"Running: {workload}")
        trace = generate_trace(
            workload,
            cfg.num_requests,
            cfg.num_blocks,
            cfg.seed,
        )

        results, clustering = run_experiment(
            trace=trace,
            window_size=cfg.window_size,
            n_clusters=cfg.n_clusters,
            max_iter=cfg.kmeans_iterations,
            n_init=cfg.kmeans_restarts,
            seed=cfg.seed,
            cache_percentages=cfg.cache_percentages,
            policy_map=cfg.policy_map,
            hit_cost=cfg.hit_cost,
            miss_cost=cfg.miss_cost,
        )

        for row in results:
            row["workload"] = workload
        all_results.extend(results)

        workload_dir = os.path.join(cfg.output_dir, workload)
        os.makedirs(workload_dir, exist_ok=True)
        write_csv(os.path.join(workload_dir, "summary.csv"), results)
        write_csv(
            os.path.join(workload_dir, "window_assignments.csv"),
            clustering["assignments"],
        )
        plot_hit_ratio(results, os.path.join(workload_dir, "hit_ratio.png"))
        plot_access_cost(
            results,
            os.path.join(workload_dir, "average_access_cost.png"),
        )

    write_csv(os.path.join(cfg.output_dir, "all_workloads.csv"), all_results)
    print(f"\nComplete. Results are in {os.path.abspath(cfg.output_dir)}/")


if __name__ == "__main__":
    main()
