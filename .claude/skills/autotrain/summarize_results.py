#!/usr/bin/env python3
"""
summarize_results.py — compact digest of a DCM results.json file.

Usage:
    python3 summarize_results.py <path-to-results.json>

Emits a JSON digest to stdout. The autotrain skill passes this digest to
its proposer / analyzer subagents instead of the raw file (which is often
~1 MB ≈ 300k tokens). Agents that need detail not in the digest run
`jq` or `python3 -c '...'` via Bash against the original file whose path
is passed alongside the summary.

Intentionally defensive about missing keys — the chess app's output
schema evolves; we'd rather emit partial info than crash on a new field.
"""

import json
import sys
from pathlib import Path


TRACKED_STAT_KEYS = [
    "policy_entropy",
    "policy_logit_abs_max",
    "top1_legal_fraction",
    "legal_mass",
    "grad_global_norm",
    "policy_loss",
    "value_loss",
    "value_abs_mean",
    "value_mean",
    "played_move_prob",
    "played_move_prob_pos_adv",
    "played_move_prob_neg_adv",
    "adv_mean",
    "adv_std",
    "ratio_current",
]


def trajectory(series):
    nums = [float(x) for x in series if isinstance(x, (int, float))]
    if not nums:
        return None
    ordered = sorted(nums)
    return {
        "first": round(nums[0], 6),
        "min": round(ordered[0], 6),
        "median": round(ordered[len(ordered) // 2], 6),
        "mean": round(sum(nums) / len(nums), 6),
        "max": round(ordered[-1], 6),
        "final": round(nums[-1], 6),
    }


def collapse_signals(stats, probes):
    """Surface every binary collapse signal the analyzer keys off of.

    Hard-reject criteria (any one is enough to classify a run as regressed)
    and soft-reject criteria (multiple must trip) are documented in
    SKILL.md step 7. This function exposes the raw booleans / extrema the
    analyzer evaluates them against; it deliberately does NOT decide
    classification — it only computes the inputs.
    """
    if not stats:
        return {}
    def series(k):
        return [s.get(k) for s in stats if isinstance(s.get(k), (int, float))]
    pEnt = series("policy_entropy")
    plogit = series("policy_logit_abs_max")
    top1legal = series("top1_legal_fraction")
    legalmass = series("legal_mass")
    gnorm = series("grad_global_norm")
    vabs = series("value_abs_mean")
    ploss = series("policy_loss")

    # Late-window candidate-probe assessment: if the last few probes (the
    # final ~5 minutes of training) all show max_prob ≥ 0.99, the model
    # has crystallized on a single (illegal) move at end of run. This is
    # the "Forward Pass shows 100% on illegals" signal, derived without
    # needing the UI demo button.
    late_probe_collapsed = False
    late_probe_window = []
    if probes:
        tail = probes[-min(15, len(probes)):]  # ~last 5min at 20s probe interval
        for p in tail:
            ph = (p.get("policy_head") or {}).get("policy_stats") or {}
            mp = ph.get("max")
            lm = ph.get("legal_mass_sum")
            if isinstance(mp, (int, float)) and isinstance(lm, (int, float)):
                late_probe_window.append((mp, lm))
        if late_probe_window:
            # All probes in window show degenerate softmax + ~zero legal mass
            late_probe_collapsed = all(mp >= 0.99 and lm < 0.01 for mp, lm in late_probe_window)

    return {
        # pEnt: log(4864) ≈ 8.49 at uniform; alarm floor at 5.0
        "min_policy_entropy": round(min(pEnt), 4) if pEnt else None,
        "final_policy_entropy": round(pEnt[-1], 4) if pEnt else None,
        "pEnt_ever_below_5": bool(pEnt and min(pEnt) < 5.0),
        "pEnt_final_below_5": bool(pEnt and pEnt[-1] < 5.0),

        # Single-logit blowup — softmax becomes a delta function above ~30
        "max_policy_logit_abs_max": round(max(plogit), 3) if plogit else None,
        "final_policy_logit_abs_max": round(plogit[-1], 3) if plogit else None,
        "policy_logit_blowup": bool(plogit and max(plogit) > 30.0),
        "policy_logit_severe_blowup": bool(plogit and max(plogit) > 50.0),

        # Top-1 legal: did the network EVER put nontrivial mass on a legal move?
        "final_top1_legal_fraction": round(top1legal[-1], 4) if top1legal else None,
        "top1_legal_ever_positive": bool(top1legal and max(top1legal) > 0.0),

        # Legal mass concentration over the run
        "min_legal_mass": round(min(legalmass), 6) if legalmass else None,
        "final_legal_mass": round(legalmass[-1], 6) if legalmass else None,
        "legal_mass_below_pct1_at_end": bool(legalmass and legalmass[-1] < 0.01),

        # Gradient norm — sustained explosion is bad even if no early-bail
        "max_grad_global_norm": round(max(gnorm), 3) if gnorm else None,
        "final_grad_global_norm": round(gnorm[-1], 3) if gnorm else None,
        "grad_norm_ever_exceeded_100": bool(gnorm and max(gnorm) > 100.0),
        "grad_norm_ever_exceeded_200": bool(gnorm and max(gnorm) > 200.0),

        # Value head saturation (tanh dead zone)
        "final_value_abs_mean": round(vabs[-1], 4) if vabs else None,
        "value_head_saturated": bool(vabs and vabs[-1] >= 0.95),

        # Reward-hacking watch: large negative pLoss without arena promotions
        # is the network exploiting entropy bonus / advantage scaling rather
        # than learning legal play
        "final_policy_loss": round(ploss[-1], 3) if ploss else None,
        "policy_loss_extreme_negative": bool(ploss and ploss[-1] < -10.0),

        # Late-window candidate-probe collapse — equivalent to your UI
        # Forward Pass showing 100% on illegals, computed from the last
        # ~5min of probes
        "late_probe_collapsed": late_probe_collapsed,
        "late_probe_window_size": len(late_probe_window),
    }


def arena_summary(arenas):
    if not arenas:
        return {"count": 0, "promoted": 0, "mean_score": None, "rounds": []}
    rounds = []
    for a in arenas:
        rounds.append({
            "index": a.get("index"),
            "promoted": a.get("promoted"),
            "score": round(a.get("score", 0.0), 4) if isinstance(a.get("score"), (int, float)) else None,
            "score_ci": [
                round(a.get("score_lo", 0.0), 4) if isinstance(a.get("score_lo"), (int, float)) else None,
                round(a.get("score_hi", 0.0), 4) if isinstance(a.get("score_hi"), (int, float)) else None,
            ],
            "elo": round(a.get("elo", 0.0), 2) if isinstance(a.get("elo"), (int, float)) else None,
            "games": a.get("games_played"),
            "candidate_W_L_D": [
                a.get("candidate_wins"),
                a.get("champion_wins"),
                a.get("draws"),
            ],
            "score_white": round(a.get("candidate_score_as_white", 0.0), 3) if isinstance(a.get("candidate_score_as_white"), (int, float)) else None,
            "score_black": round(a.get("candidate_score_as_black", 0.0), 3) if isinstance(a.get("candidate_score_as_black"), (int, float)) else None,
            "candidate_id": a.get("candidate_id"),
            "champion_id": a.get("champion_id"),
            "duration_sec": round(a.get("duration_sec", 0.0), 1) if isinstance(a.get("duration_sec"), (int, float)) else None,
        })
    scores = [r["score"] for r in rounds if isinstance(r["score"], (int, float))]
    return {
        "count": len(arenas),
        "promoted": sum(1 for r in rounds if r["promoted"]),
        "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
        "rounds": rounds,
    }


def probe_digest(p):
    ph = (p.get("policy_head") or {}).get("policy_stats") or {}
    vh = p.get("value_head") or {}
    return {
        "elapsed_sec": round(p.get("elapsed_sec", 0.0), 1) if isinstance(p.get("elapsed_sec"), (int, float)) else None,
        "max_prob": round(ph.get("max", 0.0), 5) if isinstance(ph.get("max"), (int, float)) else None,
        "above_uniform_count": ph.get("above_uniform_count"),
        "legal_mass_sum": round(ph.get("legal_mass_sum", 0.0), 5) if isinstance(ph.get("legal_mass_sum"), (int, float)) else None,
        "top100_sum": round(ph.get("top100_sum", 0.0), 5) if isinstance(ph.get("top100_sum"), (int, float)) else None,
        "value_output": round(vh.get("output", 0.0), 5) if isinstance(vh.get("output"), (int, float)) else None,
    }


def probe_summary(probes):
    if not probes:
        return {"count": 0}
    return {
        "count": len(probes),
        "first": probe_digest(probes[0]),
        "mid": probe_digest(probes[len(probes) // 2]),
        "last": probe_digest(probes[-1]),
    }


def summarize(data):
    stats = data.get("stats") or []
    arenas = data.get("arena_results") or []
    probes = data.get("candidate_tests") or []
    last_stat = stats[-1] if stats else {}

    trajectories = {}
    for key in TRACKED_STAT_KEYS:
        t = trajectory([s.get(key) for s in stats if s.get(key) is not None])
        if t is not None:
            trajectories[key] = t

    # Build-number surfacing — if the app code was rebuilt between runs,
    # comparisons are apples-to-oranges. Callers (analyzer) want an explicit
    # version stamp.
    build_numbers = [s.get("build_number") for s in stats if s.get("build_number") is not None]
    if build_numbers:
        unique_builds = sorted(set(build_numbers))
        build_info = {
            "first": build_numbers[0],
            "last": build_numbers[-1],
            "changed_mid_run": len(unique_builds) > 1,
            "unique": unique_builds if len(unique_builds) > 1 else None,
        }
    else:
        build_info = None

    def pick(src, *keys):
        return {k: src.get(k) for k in keys if k in src} if src else {}

    # Derived budget — hints for the proposer about per-iteration scale
    # so it doesn't pick values incompatible with the 10-min training
    # window (e.g. lr_warmup_steps=500 when only ~330 steps fit).
    # Bounds are advisory; enforced in validate_params.py.
    training_steps = data.get("training_steps")
    total_secs = data.get("total_training_seconds")
    if isinstance(training_steps, int) and training_steps > 0:
        # Warmup longer than ~half the window means the configured lr is
        # never reached. Cap the proposer's recommended warmup at 1/3 of
        # steps so the post-warmup plateau is visibly exercised.
        derived_budget = {
            "training_steps_per_window": training_steps,
            "recommended_lr_warmup_max": max(1, training_steps // 3),
            "steps_per_sec": round(training_steps / total_secs, 3)
                if isinstance(total_secs, (int, float)) and total_secs > 0 else None,
        }
    else:
        derived_budget = None

    return {
        "session_id": data.get("session_id"),
        "build_number": build_info,
        "termination_reason": data.get("termination_reason"),
        "total_training_seconds": data.get("total_training_seconds"),
        "training_steps": data.get("training_steps"),
        "derived_budget": derived_budget,
        "positions_trained": data.get("positions_trained"),
        "self_play_games_final": last_stat.get("self_play_games"),
        "buffer_count_final": last_stat.get("buffer_count"),
        "avg_game_length_final": last_stat.get("avg_len"),
        "trainer_id_final": last_stat.get("trainer_id"),
        "champion_id_final": last_stat.get("champion_id"),
        "diversity_final": pick(
            last_stat, "diversity_unique_percent", "diversity_avg_divergence_ply",
            "diversity_games_in_window", "diversity_unique_games",
        ) or None,
        "draws_by_type_final": pick(
            last_stat, "fifty_move_draws", "insufficient_material_draws",
            "threefold_repetition_draws", "stalemates",
            "white_checkmates", "black_checkmates",
        ) or None,
        "training_trajectory": trajectories,
        "collapse_signals": collapse_signals(stats, probes),
        "arenas": arena_summary(arenas),
        "candidate_probes": probe_summary(probes),
    }


def main(argv):
    if len(argv) != 2:
        print("usage: summarize_results.py <path-to-results.json>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        print(f"summarize_results.py: not found: {path}", file=sys.stderr)
        return 3
    except json.JSONDecodeError as e:
        print(f"summarize_results.py: invalid JSON at {path}: {e}", file=sys.stderr)
        return 4
    json.dump(summarize(data), sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
