#!/usr/bin/env python3
"""Compare the `[BATCH-HASH]` lines of two DrewsChessMachine session logs.

Each training run logs, every 100 trainer steps,

    [BATCH-HASH] trainerStep=N batchHash=<16 hex> batchChain=<16 hex | partial>

`batchHash` is the SHA-256 of that step's batch (boards, moves, outcomes as
trained); `batchChain` chains every batch of the step's 1,000-step window up
to it (`partial` when the run didn't see the whole window, e.g. it resumed
mid-window). Two corpus-replay runs trained on the same batches exactly when
their lines agree.

Usage:
    compare_batch_hashes.py <log A> <log B>

Prints every common step that differs and a verdict:
- "identical at every common step (N steps, through trainer step S)";
- the first common step where the batch hashes differ;
- steps where one run's chain is partial are compared by batch hash only.

A trainer step that appears more than once in one log (a GUI promotion rewinds
the trainer's clock) is reported; the last occurrence is used.

Exit status: 0 identical, 1 different, 2 nothing to compare or bad input.
"""
from __future__ import annotations

import re
import sys

LINE = re.compile(r"\[BATCH-HASH\] trainerStep=(\d+) batchHash=([0-9a-f]{16}) batchChain=([0-9a-f]{16}|partial)\b")


def read(path: str) -> tuple[dict[int, tuple[str, str]], list[int]]:
    """Map trainer step -> (batch hash, chain); also the steps seen twice."""
    entries: dict[int, tuple[str, str]] = {}
    repeated: list[int] = []
    with open(path, errors="replace") as handle:
        for line in handle:
            match = LINE.search(line)
            if not match:
                continue
            step = int(match.group(1))
            if step in entries:
                repeated.append(step)
            entries[step] = (match.group(2), match.group(3))
    return entries, repeated


def compare(a: dict[int, tuple[str, str]], b: dict[int, tuple[str, str]]) -> tuple[list[str], int | None, int]:
    """Lines describing differences, the first differing step, and the common-step count."""
    common = sorted(set(a) & set(b))
    lines: list[str] = []
    first_difference: int | None = None
    for step in common:
        hash_a, chain_a = a[step]
        hash_b, chain_b = b[step]
        if hash_a != hash_b:
            lines.append(f"step {step}: batch hash differs ({hash_a} vs {hash_b})")
            first_difference = step if first_difference is None else first_difference
        elif "partial" not in (chain_a, chain_b) and chain_a != chain_b:
            lines.append(f"step {step}: same batch, chain differs ({chain_a} vs {chain_b}) — "
                         "an earlier batch in this window differed")
            first_difference = step if first_difference is None else first_difference
    return lines, first_difference, len(common)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[0])
        print("usage: compare_batch_hashes.py <log A> <log B>")
        return 2
    a, repeated_a = read(argv[1])
    b, repeated_b = read(argv[2])
    for name, repeated in ((argv[1], repeated_a), (argv[2], repeated_b)):
        if repeated:
            print(f"note: {name} logs trainer steps {sorted(set(repeated))} more than once (clock rewound); "
                  "the last occurrence is compared")
    lines, first_difference, count = compare(a, b)
    if count == 0:
        print("no common trainer steps with [BATCH-HASH] lines")
        return 2
    for line in lines:
        print(line)
    if first_difference is None:
        last = max(set(a) & set(b))
        print(f"identical at every common step ({count} steps, through trainer step {last})")
        return 0
    print(f"first difference at trainer step {first_difference}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
