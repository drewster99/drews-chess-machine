#!/usr/bin/env python3
"""
validate_params.py — sanity-bound check for an autotrain parameters.json.

Usage:
    python3 validate_params.py <path-to-parameters.json> [--latest-result <path>]

Exits 0 if every declared parameter is within wide-but-finite bounds.
Exits 1 and prints violations on stderr otherwise. The point is not to
gatekeep parameter tuning — it's to catch agent hallucinations (e.g.
`learning_rate: 5.0`, `replay_buffer_capacity: 1e9`) before we waste a
training run on them. Bounds are intentionally generous.

When `--latest-result <path>` is passed, cross-checks parameters against
the latest run's realized budget (e.g. `lr_warmup_steps` must be at most
half of `training_steps` from the latest run, or the warmup consumes the
entire training window and the post-warmup lr is never exercised).

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
    "sqrt_batch_scaling_lr":                      (None,       None,      False),  # boolean, handled specially
    "lr_warmup_steps":                            (0,          1000000,   True),
    "self_play_workers":                          (1,          256,       True),
    "training_step_delay_ms":                     (0,          600000,    True),
    "training_batch_size":                        (1,          524288,    True),
    "replay_buffer_capacity":                     (100,        100000000, True),
    "replay_buffer_min_positions_before_training":(0,          100000000, True),
    "arena_promote_threshold":                    (0.0,        1.0,       False),
    "arena_games_per_tournament":                 (2,          100000,    True),
    "arena_auto_interval_sec":                    (1,          604800,    True),  # 1s to 1 week
    "candidate_probe_interval_sec":               (1,          86400,     True),  # 1s to 1 day
    "legal_mass_collapse_threshold":               (0.5,        0.999999,  False),
    "legal_mass_collapse_grace_seconds":           (0.0,        86400.0,   False),  # 0s to 1 day
    "legal_mass_collapse_no_improvement_probes":   (1,          1000,      True),
}


def validate(params, latest_result=None):
    violations = []

    if not isinstance(params, dict):
        return ["top-level value is not a JSON object"]

    for bool_key in (
        "replay_ratio_auto_adjust",
        "sqrt_batch_scaling_lr",
    ):
        if params.get(bool_key) is not None and not isinstance(params[bool_key], bool):
            violations.append(
                f"{bool_key}: must be boolean, got {type(params[bool_key]).__name__}"
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

    # Budget-aware cross-check against the latest run's realized step count.
    # `lr_warmup_steps` larger than half the window's training_steps means
    # the ramp-up never finishes, so the configured learning_rate is never
    # actually exercised. Reject anything more aggressive than that.
    if latest_result is not None:
        latest_steps = latest_result.get("training_steps")
        warmup = params.get("lr_warmup_steps")
        if (
            isinstance(latest_steps, int) and latest_steps > 0
            and isinstance(warmup, int) and warmup > 0
            and warmup > latest_steps // 2
        ):
            violations.append(
                f"lr_warmup_steps ({warmup}) exceeds 50% of latest run's "
                f"training_steps ({latest_steps}); the lr ramp would never finish "
                f"within a run of that size. Use at most {latest_steps // 3} "
                f"(advisory target; hard cap is {latest_steps // 2})."
            )

    return violations


def main(argv):
    # Parse args: first positional is params path, optional --latest-result <path>.
    params_path = None
    latest_result_path = None
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--latest-result":
            if i + 1 >= len(argv):
                print("--latest-result requires a path argument", file=sys.stderr)
                return 2
            latest_result_path = argv[i + 1]
            i += 2
        elif params_path is None and not arg.startswith("-"):
            params_path = arg
            i += 1
        else:
            print(f"unknown argument: {arg}", file=sys.stderr)
            return 2
    if params_path is None:
        print(
            "usage: validate_params.py <path-to-parameters.json> "
            "[--latest-result <path>]",
            file=sys.stderr,
        )
        return 2

    path = Path(params_path)
    try:
        params = json.loads(path.read_text())
    except FileNotFoundError:
        print(f"validate_params.py: not found: {path}", file=sys.stderr)
        return 3
    except json.JSONDecodeError as e:
        print(f"validate_params.py: invalid JSON at {path}: {e}", file=sys.stderr)
        return 4

    latest_result = None
    if latest_result_path is not None:
        try:
            latest_result = json.loads(Path(latest_result_path).read_text())
        except FileNotFoundError:
            # Missing latest-result is non-fatal — skip the cross-check,
            # proposer shouldn't be blocked on a first/seed run.
            latest_result = None
        except json.JSONDecodeError:
            # Malformed result file is also non-fatal for validation.
            latest_result = None

    violations = validate(params, latest_result=latest_result)
    if violations:
        print("validate_params.py: out-of-bounds proposal:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
