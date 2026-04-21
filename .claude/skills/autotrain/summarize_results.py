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


def collapse_signals(stats):
    if not stats:
        return {}
    def series(k):
        return [s.get(k) for s in stats if isinstance(s.get(k), (int, float))]
    pEnt = series("policy_entropy")
    top1legal = series("top1_legal_fraction")
    gnorm = series("grad_global_norm")
    vabs = series("value_abs_mean")
    return {
        "min_policy_entropy": round(min(pEnt), 4) if pEnt else None,
        "pEnt_ever_below_5": bool(pEnt and min(pEnt) < 5.0),
        "final_top1_legal_fraction": round(top1legal[-1], 4) if top1legal else None,
        "top1_legal_ever_positive": bool(top1legal and max(top1legal) > 0.0),
        "max_grad_global_norm": round(max(gnorm), 3) if gnorm else None,
        "grad_norm_ever_exceeded_100": bool(gnorm and max(gnorm) > 100.0),
        "final_value_abs_mean": round(vabs[-1], 4) if vabs else None,
        "value_head_saturated": bool(vabs and vabs[-1] >= 0.95),
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

    return {
        "session_id": data.get("session_id"),
        "build_number": build_info,
        "total_training_seconds": data.get("total_training_seconds"),
        "training_steps": data.get("training_steps"),
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
        "collapse_signals": collapse_signals(stats),
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
