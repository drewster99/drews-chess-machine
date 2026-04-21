#!/usr/bin/env python3
"""
validate_params.py — sanity-bound check for an autotrain parameters.json.

Usage:
    python3 validate_params.py <path-to-parameters.json>

Exits 0 if every declared parameter is within wide-but-finite bounds.
Exits 1 and prints violations on stderr otherwise. The point is not to
gatekeep parameter tuning — it's to catch agent hallucinations (e.g.
`learning_rate: 5.0`, `replay_buffer_capacity: 1e9`) before we waste a
training run on them. Bounds are intentionally generous.

Unknown keys are accepted silently (new params can be added without
editing this script). Missing known keys are accepted (if the app
doesn't need them, that's the app's problem).
"""

import json
import sys
from pathlib import Path


# (min, max, must_be_integer) — bounds are wide on purpose.
BOUNDS = {
    "entropy_bonus":                              (0.0,        1.0,       False),
    "grad_clip_max_norm":                         (1e-3,       1e5,       False),
    "weight_decay":                               (0.0,        0.5,       False),
    "K":                                          (1e-3,       1e4,       False),
    "learning_rate":                              (1e-7,       1.0,       False),
    "draw_penalty":                               (0.0,        5.0,       False),
    "self_play_start_tau":                        (1e-3,       100.0,     False),
    "self_play_target_tau":                       (1e-3,       100.0,     False),
    "self_play_tau_decay_per_ply":                (0.0,        10.0,      False),
    "arena_start_tau":                            (1e-3,       100.0,     False),
    "arena_target_tau":                           (1e-3,       100.0,     False),
    "arena_tau_decay_per_ply":                    (0.0,        10.0,      False),
    "replay_ratio_target":                        (1e-3,       1e4,       False),
    "replay_ratio_auto_adjust":                   (None,       None,      False),  # boolean, handled specially
    "self_play_workers":                          (1,          256,       True),
    "training_step_delay_ms":                     (0,          600000,    True),
    "training_batch_size":                        (1,          524288,    True),
    "replay_buffer_capacity":                     (100,        100000000, True),
    "replay_buffer_min_positions_before_training":(0,          100000000, True),
    "arena_promote_threshold":                    (0.0,        1.0,       False),
    "arena_games_per_tournament":                 (2,          100000,    True),
    "arena_auto_interval_sec":                    (1,          604800,    True),  # 1s to 1 week
    "candidate_probe_interval_sec":               (1,          86400,     True),  # 1s to 1 day
}


def validate(params):
    violations = []

    if not isinstance(params, dict):
        return ["top-level value is not a JSON object"]

    if params.get("replay_ratio_auto_adjust") is not None and \
       not isinstance(params["replay_ratio_auto_adjust"], bool):
        violations.append(
            f"replay_ratio_auto_adjust: must be boolean, got {type(params['replay_ratio_auto_adjust']).__name__}"
        )

    for key, (lo, hi, must_int) in BOUNDS.items():
        if key not in params:
            continue
        if lo is None and hi is None:
            continue  # boolean or free field
        value = params[key]
        if value is None:
            violations.append(f"{key}: null is not allowed")
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            violations.append(f"{key}: must be numeric, got {type(value).__name__} ({value!r})")
            continue
        if must_int and not isinstance(value, int):
            violations.append(f"{key}: must be an integer, got float ({value})")
            continue
        if value < lo or value > hi:
            violations.append(f"{key}: out of bounds [{lo}, {hi}], got {value}")

    # Cross-field sanity.
    cap = params.get("replay_buffer_capacity")
    floor = params.get("replay_buffer_min_positions_before_training")
    if isinstance(cap, int) and isinstance(floor, int) and floor > cap:
        violations.append(
            f"replay_buffer_min_positions_before_training ({floor}) > replay_buffer_capacity ({cap})"
        )

    return violations


def main(argv):
    if len(argv) != 2:
        print("usage: validate_params.py <path-to-parameters.json>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    try:
        params = json.loads(path.read_text())
    except FileNotFoundError:
        print(f"validate_params.py: not found: {path}", file=sys.stderr)
        return 3
    except json.JSONDecodeError as e:
        print(f"validate_params.py: invalid JSON at {path}: {e}", file=sys.stderr)
        return 4

    violations = validate(params)
    if violations:
        print("validate_params.py: out-of-bounds proposal:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
