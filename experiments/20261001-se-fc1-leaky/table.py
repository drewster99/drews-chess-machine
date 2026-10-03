#!/usr/bin/env python3
"""Per-1000-step table: the three SE-experiment seed-1 arms vs leaky-FC1.

pElo / NLL for the SE arms come from the dashboard CSVs (their enumerated
checkpoints probed with `--probe-set wide`); the leaky-FC1 arm's come from
probes.jsonl (written by probe_loop.sh, same probe set), read through
`probe_record.load_probe_points` with the run's model_id. A cell is blank where
a run never reached that step.

Buffer plies/game is the average game length in the 500k-position replay
buffer at that step (`table_common.buffer_plies_per_game`), from the ReLU
scale+bias run's [REPLAY] lines (complete to 33k). All arms replay the same
corpus in the same order with the same feed per step, so it is the same for
every arm; the script stops if the leaky run's log disagrees where both exist.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import probe_record  # noqa: E402
from table_common import (BASELINE_LOG, buffer_plies_per_game, csv_points, csv_probe_builds,  # noqa: E402
                          print_probe_builds)

LEAKY_LOG = "dcm_log_20261001-151822.txt"
LEAKY_MODEL_ID = "20261001-43-NbWz"

SE_ARMS = [("ReLU scale+bias", "se_sb"), ("ReLU attenuate-only", "se_att"), ("ReLU no SE", "se_none")]


def leaky_points():
    return probe_record.load_probe_points(os.path.join(HERE, "probes.jsonl"), LEAKY_MODEL_ID)


def main():
    # The leaky-FC1 arm sits right after its direct comparator, ReLU scale+bias.
    se = [(label, csv_points(run)) for label, run in SE_ARMS]
    arms = [se[0], ("leaky FC1 scale+bias", leaky_points())] + se[1:]
    plies = buffer_plies_per_game(BASELINE_LOG)
    leaky_plies = buffer_plies_per_game(LEAKY_LOG)
    for step, value in leaky_plies.items():
        if step in plies and abs(plies[step] - value) > 0.05:
            raise SystemExit(f"buffer plies/game differs at {step}: ReLU {plies[step]:.2f} vs leaky {value:.2f}")
    last = max(max(points) for _, points in arms if points)
    header = ["step", "buffer plies/game"] + [f"pElo {label}" for label, _ in arms] + [f"NLL {label}" for label, _ in arms]
    print("| " + " | ".join(header) + " |")
    print("|" + "---:|" * len(header))
    for step in range(1000, last + 1, 1000):
        row = [f"{step:,}", f"{plies[step]:.1f}" if step in plies else ""]
        row += [probe_record.pelo_cell(points, step) for _, points in arms]
        row += [probe_record.nll_cell(points, step) for _, points in arms]
        print("| " + " | ".join(row) + " |")
    builds = [(label, csv_probe_builds(run)) for label, run in SE_ARMS]
    print_probe_builds([builds[0], ("leaky FC1 scale+bias", probe_record.probe_builds(
        os.path.join(HERE, "probes.jsonl")))] + builds[1:])


if __name__ == "__main__":
    main()
