from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable
#this figures out what kind of workload we are going to generate.
#it can identify if it is sequential, looping, random or mixed. 
#accordingly it will generate a trace of requests that follow that pattern.

@dataclass
class Phase:
    kind: str
    length: int
    working_set: int | None = None


def _sequential(rng: random.Random, start: int, size: int) -> list[int]:
    if size <= 0:
        return []
    direction = 1 if rng.random() < 0.8 else -1
    if direction == 1:
        return [(start + i) for i in range(size)]
    return [(start + size - 1 - i) for i in range(size)]


def _looping(rng: random.Random, start: int, size: int) -> list[int]:
    if size <= 0:
        return []
    order = list(range(start, start + size))
    # Mostly regular cyclic reuse with occasional short jumps.
    out = []
    pos = 0
    for _ in range(size):
        if rng.random() < 0.10:
            pos = rng.randrange(size)
        out.append(order[pos])
        pos = (pos + 1) % size
    return out


def _random_access(rng: random.Random, start: int, size: int) -> list[int]:
    if size <= 0:
        return []
    return [rng.randrange(start, start + size) for _ in range(size)]


def _mixed(rng: random.Random, start: int, size: int) -> list[int]:
    if size <= 0:
        return []
    parts = max(1, size // 4)
    return (
        _sequential(rng, start, parts)
        + _looping(rng, start, parts)
        + _random_access(rng, start, parts)
        + _looping(rng, start, size - 3 * parts)
    )


def generate_phase(kind: str, length: int, num_blocks: int, rng: random.Random,
                   working_set: int | None = None) -> list[int]:
    ws = working_set or max(10, min(num_blocks, max(10, num_blocks // 10)))
    ws = min(ws, num_blocks)

    start = rng.randrange(0, max(1, num_blocks - ws + 1))

    if kind == "sequential":
        return _sequential(rng, start, length) if length <= ws else [
            start + (i % ws) for i in range(length)
        ]
    if kind == "looping":
        base = _looping(rng, start, ws)
        return [base[i % len(base)] for i in range(length)]
    if kind == "random":
        # Draw each request independently so the trace always has `length` items.
        return [rng.randrange(start, start + ws) for _ in range(length)]
    if kind == "mixed":
        return _mixed(rng, start, length)
    raise ValueError(f"Unknown workload kind: {kind}")


def generate_trace(workload: str, num_requests: int, num_blocks: int,
                   seed: int = 42) -> list[int]:
    rng = random.Random(seed)

    if workload == "sequential":
        return generate_phase("sequential", num_requests, num_blocks, rng,
                              max(10, min(num_blocks, num_requests // 2)))
    if workload == "looping":
        return generate_phase("looping", num_requests, num_blocks, rng,
                              max(10, min(num_blocks, max(20, num_requests // 20))))
    if workload == "random":
        return generate_phase("random", num_requests, num_blocks, rng,
                              max(10, min(num_blocks, num_blocks // 2)))
    if workload == "mixed":
        # Controlled workload changes: sequential -> looping -> random -> looping.
        lengths = [
            num_requests // 4,
            num_requests // 4,
            num_requests // 4,
            num_requests - 3 * (num_requests // 4),
        ]
        kinds = ["sequential", "looping", "random", "looping"]
        trace = []
        for kind, length in zip(kinds, lengths):
            trace.extend(generate_phase(
                kind, length, num_blocks, rng,
                {
                    "sequential": max(10, min(num_blocks, num_requests // 10)),
                    "looping": max(10, min(num_blocks, num_requests // 30)),
                    "random": max(10, min(num_blocks, num_blocks // 2)),
                }.get(kind)
            ))
        return trace

    raise ValueError("workload must be sequential, looping, random, or mixed")


def read_trace(path: str) -> list[int]:
    trace = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                trace.append(int(line.split()[0]))
            except ValueError as exc:
                raise ValueError(f"Invalid block ID on line {line_no}: {line}") from exc
    if not trace:
        raise ValueError("Trace file contains no block IDs.")
    return trace
