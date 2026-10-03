#!/usr/bin/env python3
"""Per-1000-step table: no-SE with ReZero (SE experiment seeds 1 and 2) vs no-SE
without ReZero (this run), plus zero-init ReZero once it runs. pElo / NLL from
`--probe-set wide` probes; blank where a run never reached the step.

Probe files are read through `probe_record.load_probe_points` with each run's
model_id, so a file holding another run's records is refused rather than tabulated.

Buffer plies/game is the average game length in the 500k-position replay buffer
at that step, from the SE experiment's ReLU scale+bias run's [REPLAY] lines
(complete to 33k), computed by `table_common.buffer_plies_per_game`. Every
arm replays the same corpus in the same
order with the same feed per step, so it is the same for every arm; the script
stops if this run's log disagrees where both exist."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import probe_record  # noqa: E402
from table_common import (BASELINE_LOG, LOGS, buffer_plies_per_game, csv_points, csv_probe_builds,  # noqa: E402
                          print_probe_builds)

NO_REZERO_LOG = "dcm_log_20261002-011124.txt"

# (label, probes file, model_id); NOT_STARTED until the run's first checkpoint exists.
PROBE_ARMS = [
    ("no SE, no ReZero s1", "probes.jsonl", "20261002-2-5tKN"),
    ("no SE, no ReZero s2", "probes-seed2.jsonl", "20261002-4-T79u"),
    ("zero-init ReZero", os.path.join("..", "20261002-rezero-zero-init", "probes.jsonl"), "20261003-1-NKTv"),
]


def probe_points(file_name, model_id, label):
    return probe_record.arm_points(os.path.join(HERE, file_name), model_id, label)


def main():
    probed = {label: probe_points(f, m, label) for label, f, m in PROBE_ARMS}
    files = {label: os.path.join(HERE, f) for label, f, _ in PROBE_ARMS}
    arms = [("no SE + ReZero s1", csv_points("se_none"))]
    arms += [(label, probed[label]) for label, _, _ in PROBE_ARMS[:2]]
    arms += [("no SE + ReZero s2", csv_points("se_none2"))]
    if probed[PROBE_ARMS[2][0]] is not None:
        arms.append((PROBE_ARMS[2][0], probed[PROBE_ARMS[2][0]]))
    builds = [("no SE + ReZero s1", csv_probe_builds("se_none"))]
    builds += [(label, probe_record.probe_builds(files[label])) for label, points in probed.items() if points is not None]
    builds += [("no SE + ReZero s2", csv_probe_builds("se_none2"))]
    not_started = [label for label, points in probed.items() if points is None]
    plies = buffer_plies_per_game(BASELINE_LOG)
    for step, value in buffer_plies_per_game(NO_REZERO_LOG).items():
        if step in plies and abs(plies[step] - value) > 0.05:
            raise SystemExit(f"buffer plies/game differs at {step}: baseline {plies[step]:.2f} vs no-ReZero {value:.2f}")
    last = max((max(p) for _, p in arms if p), default=0)
    header = ["step", "buffer plies/game"] + [f"pElo {l}" for l, _ in arms] + [f"NLL {l}" for l, _ in arms]
    print("| " + " | ".join(header) + " |")
    print("|" + "---:|" * len(header))
    for step in range(1000, last + 1, 1000):
        row = [f"{step:,}", f"{plies[step]:.1f}" if step in plies else ""]
        row += [probe_record.pelo_cell(p, step) for _, p in arms]
        row += [f"{p[step][1]:.4f}" if step in p else "" for _, p in arms]
        print("| " + " | ".join(row) + " |")
    print_probe_builds(builds)
    if not_started:
        print(f"\nNot started: {', '.join(not_started)}.")


if __name__ == "__main__":
    main()
