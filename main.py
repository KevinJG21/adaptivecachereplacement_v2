from __future__ import annotations

import argparse
import os

from config import ExperimentConfig
from evaluation import run_experiment, write_csv
from plotting import (
    plot_access_cost,
    plot_cluster_timeline,
    plot_hit_ratio,
    plot_policy_switches,
)
from trace_generator import generate_trace, read_trace


def parse_args():
    p = argparse.ArgumentParser(
        description="Adaptive file cache management using access-pattern clustering."
    )
    p.add_argument(
        "--workload",
        choices=["sequential", "looping", "random", "mixed"],
        default="mixed",
    )
    p.add_argument("--trace", default=None, help="Path to a text/CSV-like trace with one block ID per line")
    p.add_argument("--custom-trace", default=None, help="Comma/space-separated block IDs, e.g. 1,2,3,1,2,4")
    p.add_argument("--interactive", action="store_true", help="Enter a custom access trace in the terminal")
    p.add_argument("--export-trace", action="store_true", help="Save the input trace as results/input_trace.csv")
    p.add_argument("--requests", type=int, default=None)
    p.add_argument("--blocks", type=int, default=None)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--clusters", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def print_results(results, clustering, output_dir):
    print("\n=== Adaptive File Cache Management ===")
    print(f"Clustering time: {clustering['clustering_ms']:.3f} ms")
    print(f"K-Means inertia: {clustering['model'].inertia_:.4f}")
    print("\nCluster interpretation:")
    for cluster_id, name in sorted(clustering["cluster_names"].items()):
        print(f"  Cluster {cluster_id}: {name}")

    print("\nResults:")
    header = (
        f"{'Policy':<10} {'Cache':>8} {'Hit %':>10} "
        f"{'Miss %':>10} {'Avg Cost':>12} {'Switches':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['policy']:<10} "
            f"{r['cache_fraction'] * 100:>7.0f}% "
            f"{r['hit_ratio'] * 100:>9.2f} "
            f"{r['miss_ratio'] * 100:>9.2f} "
            f"{r['average_access_cost']:>12.3f} "
            f"{r['policy_switch_count']:>10}"
        )

    print(f"\nFiles written to: {output_dir}/")


def main():
    args = parse_args()
    cfg = ExperimentConfig()

    if args.requests is not None:
        cfg.num_requests = args.requests
    if args.blocks is not None:
        cfg.num_blocks = args.blocks
    if args.window is not None:
        cfg.window_size = args.window
    if args.clusters is not None:
        cfg.n_clusters = args.clusters
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output is not None:
        cfg.output_dir = args.output

    input_modes = sum(bool(x) for x in (args.trace, args.custom_trace, args.interactive))
    if input_modes > 1:
        raise SystemExit("Choose only one of --trace, --custom-trace, or --interactive.")

    if args.trace:
        trace = read_trace(args.trace)
        workload_name = "custom_file"
    elif args.custom_trace:
        import re
        parts = [part for part in re.split(r"[,\s]+", args.custom_trace.strip()) if part]
        try:
            trace = [int(part) for part in parts]
        except ValueError as exc:
            raise SystemExit("Custom trace must contain only integer block IDs.") from exc
        if not trace:
            raise SystemExit("Custom trace cannot be empty.")
        if any(block < 0 for block in trace):
            raise SystemExit("Block IDs must be non-negative integers.")
        workload_name = "custom_manual"
    elif args.interactive:
        print("Enter block IDs separated by spaces or commas (example: 1,2,3,1,2,4).")
        raw = input("Access trace: ").strip()
        import re
        try:
            trace = [int(part) for part in re.split(r"[,\s]+", raw) if part]
        except ValueError as exc:
            raise SystemExit("Custom trace must contain only integer block IDs.") from exc
        if not trace:
            raise SystemExit("Custom trace cannot be empty.")
        if any(block < 0 for block in trace):
            raise SystemExit("Block IDs must be non-negative integers.")
        workload_name = "custom_manual"
    else:
        trace = generate_trace(
            args.workload,
            cfg.num_requests,
            cfg.num_blocks,
            cfg.seed,
        )
        workload_name = args.workload

    if len(trace) < cfg.window_size:
        raise SystemExit(
            f"Trace has {len(trace)} requests but window size is "
            f"{cfg.window_size}. Use --window with a smaller value."
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

    out = os.path.abspath(cfg.output_dir)
    os.makedirs(out, exist_ok=True)

    # Add workload name to result rows.
    for row in results:
        row["workload"] = workload_name

    write_csv(os.path.join(out, "summary.csv"), results)

    # Save a transparent, reusable dataset of every request processed.
    if args.export_trace or workload_name.startswith("custom"):
        write_csv(
            os.path.join(out, "input_trace.csv"),
            [{"request_index": i, "block_id": block} for i, block in enumerate(trace, start=1)],
        )

    write_csv(
        os.path.join(out, "window_features.csv"),
        clustering["rows"],
    )
    write_csv(
        os.path.join(out, "window_assignments.csv"),
        clustering["assignments"],
    )

    plot_hit_ratio(results, os.path.join(out, "hit_ratio.png"))
    plot_access_cost(results, os.path.join(out, "average_access_cost.png"))
    plot_cluster_timeline(
        clustering["assignments"],
        os.path.join(out, "cluster_timeline.png"),
    )
    plot_policy_switches(
        clustering["assignments"],
        cfg.policy_map,
        os.path.join(out, "policy_selection.png"),
    )

    print_results(results, clustering, out)


if __name__ == "__main__":
    main()
