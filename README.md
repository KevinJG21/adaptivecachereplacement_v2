# Adaptive File Cache Management Using Access Pattern Clustering

An OS project prototype based on the Review 1 design:

**observe → window → extract features → cluster → select policy → cache → measure**

The simulator compares:
- LRU
- LFU
- MRU
- ARC
- Adaptive clustering-based policy selection

It generates synthetic workloads containing:
- sequential access
- looping/high-reuse access
- random access
- mixed/changing phases

## Requirements

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Quick start

Generate traces and run the full experiment:

```bash
python main.py
```

Outputs are written to `results/`:
- `summary.csv`
- `window_features.csv`
- `window_assignments.csv`
- `metrics_by_cache_size.csv`
- PNG charts

Run a smaller/faster experiment:

```bash
python main.py --requests 20000 --blocks 2000 --window 500
```

Run only one trace:

```bash
python main.py --workload mixed --requests 20000 --blocks 2000
```

Use a custom trace:

```bash
python main.py --trace path/to/trace.txt --workload custom
```

The custom trace format is one integer block ID per line, e.g.:

```text
12
15
16
17
12
80
3
...
```

## Project structure

```text
adaptive_file_cache_project/
├── main.py
├── config.py
├── trace_generator.py
├── features.py
├── clustering.py
├── cache_policies.py
├── adaptive_cache.py
├── simulator.py
├── evaluation.py
├── plotting.py
├── requirements.txt
└── README.md
```

## Important experimental design

The adaptive policy is deliberately treated as a hypothesis rather than a guaranteed improvement.

The experiment:
1. Creates access traces with controlled locality.
2. Splits the trace into observation windows.
3. Calculates window-level locality features.
4. Standardizes features and clusters the windows.
5. Gives each discovered cluster a policy using a configurable mapping.
6. Replays the same trace with fixed policies and the adaptive policy.
7. Reports hit ratio, miss ratio, average access cost, policy switches, and clustering time.
8. Repeats the comparison for several cache sizes.

### Why the adaptive result is meaningful

The baseline policies see exactly the same requests as the adaptive cache. The adaptive cache is only allowed to use information from the current/previous observation window when deciding which policy to use.

For a clean academic experiment, do not claim that the adaptive method "always wins". The useful result is showing **when** pattern-aware policy selection helps and what overhead it introduces.

## Default policy mapping

The code uses the project Review 1 hypothesis:

| Detected cluster | Policy |
|---|---|
| Sequential-like | MRU |
| Looping/high-reuse | LRU |
| Random-like | LFU |
| Mixed/changing | ARC |

The mapping is configurable in `config.py` and is intentionally evaluated rather than assumed correct.


## Dataset and custom input

This project uses **access traces** as its dataset. Each row/request identifies a memory block accessed by the workload. It does not require a labeled training dataset: the project generates synthetic traces for controlled experiments and derives window-level features from those traces.

### Run a generated synthetic dataset

```bash
python main.py --workload mixed --requests 20000 --blocks 2000 --window 500 --export-trace
```

This writes `results/input_trace.csv` with `request_index` and `block_id`, alongside the policy comparison, window features, cluster assignments, and plots. Use `--workload sequential`, `looping`, or `random` to test the other generated workload types.

### Enter a custom trace directly

```bash
python main.py --custom-trace "1,2,3,1,2,4,1,2,3,4" --window 10 --clusters 2 --output custom_results
```

Or run interactively:

```bash
python main.py --interactive --window 10 --clusters 2 --output custom_results
```

### Load a trace from a file

Create a text file with one non-negative integer block ID per line, then run:

```bash
python main.py --trace my_trace.txt --window 100 --clusters 3 --output custom_results
```

The trace must be long enough to form at least one observation window. For a small custom trace, lower `--window` and choose a sensible number of clusters.

### What to say during the review

- The synthetic trace is the experimental dataset, not a claim of collected real-world data.
- Each replacement policy receives the same request sequence for a fair comparison.
- The adaptive method extracts features from windows, clusters the windows, maps detected workload patterns to a policy, and measures hits, misses, average access cost, and policy switches.
- The custom input is a way to test the complete pipeline on a new trace; it is not a separate replacement policy.
- Synthetic experiments validate behavior under controlled patterns. Real production traces would be a useful future extension.

**Research limitation:** the current implementation clusters the windows of the supplied trace before replaying it. It demonstrates trace-level adaptive policy selection, but it is not yet a strictly online learner that trains only on past windows. Be transparent about this if asked; an online clustering/prediction design would be a further improvement.
