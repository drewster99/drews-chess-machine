#!/usr/bin/env python3
"""Render the markdown tables for REPORT.md from results/ (every row).

Writes results/tables/*.md. Each file is a complete table (or a set of
complete tables); REPORT.md embeds the main ones verbatim.
"""
import csv
import gzip
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

import policy_head_lib as lib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "..", "results")
TABLES = os.path.join(RESULTS, "tables")
DCM_ROOT = os.path.expanduser("~/Library/Application Support/DrewsChessMachine")
SURVEY = os.path.join(SCRIPT_DIR, "..", "..", "bf16-head-offset", "results", "all_out.jsonl")
SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def fmt(value, spec=".3g"):
    if value is None or value == "":
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return format(value, spec)


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def ckpt(result):
    step = result["training_step"]
    return f"{result['model_id']} @ {step if step is not None else 'fresh'}"


def short_role(result):
    return {"trainer": "T", "champion": "C"}.get(result["role"], "")


def main():
    os.makedirs(TABLES, exist_ok=True)
    detailed = [json.loads(line) for line in gzip.open(os.path.join(RESULTS, "detailed.jsonl.gz"), "rt")]
    lineages = json.load(open(os.path.join(RESULTS, "lineages.json")))
    by_lineage = defaultdict(list)
    for result in detailed:
        by_lineage[result["lineage"]].append(result)
    lineage_order = [entry["lineage"] for entry in lineages if entry["qualifies"]]

    # ------------------------------------------------ inventory
    rows = []
    for name in lineage_order:
        for r in by_lineage[name]:
            rows.append([name, ckpt(r), short_role(r) or "M", fmt(r["cum_step"], "d") if r["cum_step"] is not None else "",
                         r["creator"], "`" + os.path.relpath(r["path"], DCM_ROOT) + "`", r["file_sha256"][:12],
                         "yes" if r["has_velocity"] else "no",
                         fmt(r["tensors"]["policy.conv.weight"]["bf16_exact_fraction"], ".3f")])
    with open(os.path.join(TABLES, "inventory.md"), "w") as handle:
        handle.write(table(["lineage", "model_id @ step", "role", "cum step", "creator", "file (under DrewsChessMachine/)",
                            "sha256[:12]", "velocity", "bf16-exact fraction (policy.conv.weight)"], rows))

    # ------------------------------------------------ lineage list incl. excluded
    rows = []
    for entry in lineages:
        rows.append([entry["lineage"], entry["group"], "yes" if entry["qualifies"] else "no", entry["max_segment_step"],
                     entry["checkpoints"], f"{entry['reference_model_id']} @ {entry['reference_step'] if entry['reference_step'] >= 0 else 'fresh'}",
                     ", ".join(s.split(":")[0][-4:] + (":" + s.split(":")[1].split(":")[-1] if "selfplay" in s else "") for s in entry["segments"])])
    with open(os.path.join(TABLES, "lineages.md"), "w") as handle:
        handle.write(table(["lineage", "group", "analyzed in detail", "max segment step", "unique checkpoints",
                            "reference (init comparison)", "segments (model_id suffix)"], rows))

    # ------------------------------------------------ segments not in any lineage (trajectory only)
    trajectory_rows = list(csv.DictReader(open(os.path.join(RESULTS, "trajectory.csv"))))
    unassigned = defaultdict(list)
    for row in trajectory_rows:
        if row["lineage"].startswith("(unassigned)"):
            unassigned[row["lineage"]].append(row)
    rows = []
    for name, members in sorted(unassigned.items()):
        steps = [int(m["step"]) for m in members if m["step"] not in ("", "-1")]
        rows.append([name.replace("(unassigned) ", ""), len(members), max(steps) if steps else "fresh only",
                     members[0]["style"], members[0]["K"], members[-1]["path"]])
    with open(os.path.join(TABLES, "excluded_segments.md"), "w") as handle:
        handle.write(table(["segment (model_id base : kind)", "unique checkpoints", "max training_step",
                            "policy style", "K", "latest file"], rows))

    # ------------------------------------------------ cross-lineage (E)
    findings = list(csv.DictReader(open(os.path.join(RESULTS, "findings_aggregated.csv"))))
    rows = []
    for name in lineage_order:
        for r in by_lineage[name]:
            if not r["is_latest"]:
                continue
            p = r.get("pre_summary")
            c = r["conv_summary"]
            arch = r["arch_summary"]
            notes = sorted({f["rule"] for f in findings
                            if f["lineage"] == name and f["is_latest"] == "True" and f["checkpoint"].startswith(r["model_id"] + "@")
                            and f["severity"] in ("HIGH", "MEDIUM")})
            rows.append([name, ckpt(r) + (f" [{r['role']}]" if r["role"] != "model" else ""),
                         fmt(r["cum_step"], "d") if r["cum_step"] is not None else "",
                         arch, r["policy_style"], r["K"] if p else "—", r["compute_dtype"],
                         f"{p['dead']}/{p['mostly_off']}/{p['always_on']}" if p else "n/a",
                         f"{fmt(p['hottest_ratio'], '.1f')} / {fmt(p['rv_max_over_median'], '.1f')}" if p else "n/a",
                         fmt(p["mean_over_std"]["max"], ".1f") if p else "n/a",
                         fmt(c["mean_row_ratio"], ".2f"),
                         fmt(c.get("static_shared_level"), "+.1f") if p else "n/a",
                         fmt(c["bias_mean"], "+.2f"),
                         f"{c['weakest_family']} ({c['weakest_family_vs_queen_median']:.2f})",
                         f"{fmt(c['underpromo_vs_queen_median'], '.2f')} / "
                         f"{fmt(r['families']['underpromo all']['residual_norm_mean'] / float(np.median(np.array(r['conv']['residual_norm'])[:56])), '.2f')}",
                         fmt(r["tensors"]["policy.conv.weight"]["max_abs"], ".2f"),
                         "; ".join(notes)])
    with open(os.path.join(TABLES, "cross_lineage.md"), "w") as handle:
        handle.write(table(["lineage", "checkpoint", "cum step", "tower", "policy style", "K", "compute",
                            "dead/mostly-off/always-on", "rv max/mean / max/median", "max abs(mu)/sigma", "mean-row ratio",
                            "static shared logit level", "bias mean", "weakest family (vs Q median)",
                            "underpromo vs Q median: row / residual", "max abs W (final)", "MEDIUM+ findings"], rows))

    # ------------------------------------------------ per-lineage evolution: pre-block (A)
    out = []
    for name in lineage_order:
        rs = by_lineage[name]
        if "pre_summary" not in rs[0]:
            continue
        rows = []
        for r in rs:
            p = r["pre_summary"]
            vel = ""
            if r["has_velocity"]:
                vel = f"{len(p.get('zero_velocity_gamma_beta_channels', []))}/{len(p.get('zero_velocity_pre_rows', []))}"
            rows.append([ckpt(r) + (f" [{r['role']}]" if r["role"] != "model" else ""),
                         fmt(r["cum_step"], "d") if r["cum_step"] is not None else "",
                         p["dead"], p["mostly_off"], p["always_on"], p["flat"], p["negative_gamma"],
                         fmt(p["abs_gamma"]["median"]), f"{fmt(p['beta']['min'], '+.2f')}..{fmt(p['beta']['max'], '+.2f')}",
                         fmt(p["on_ratio"]["min"], "+.2f"), fmt(p["p_on"]["median"], ".2f"),
                         f"{fmt(p['running_var']['min'])} / {fmt(p['running_var']['median'])} / {fmt(p['running_var']['max'])}",
                         fmt(p["rv_max_over_median"], ".1f"), fmt(p["mean_over_std"]["max"], ".1f"),
                         f"{fmt(p['pre_row_norm']['min'])} / {fmt(p['pre_row_norm']['median'])} / {fmt(p['pre_row_norm']['max'])}",
                         fmt(r["tensors"]["policy.pre_conv.weight"]["max_abs"]),
                         len(p["pre_col_weak"]), vel])
        out.append(f"#### {name}\n\n" + table(
            ["checkpoint", "cum", "dead", "mostly-off", "always-on", "flat", "gamma<0", "median abs(gamma)",
             "beta range", "min beta/abs(gamma)", "median P(on)", "running var min / median / max", "rv max/median",
             "max abs(mu)/sigma", "pre_conv row norm min / median / max", "max abs pre_conv", "weak tower cols",
             "zero-velocity gamma+beta / pre rows"], rows))
    with open(os.path.join(TABLES, "evolution_pre.md"), "w") as handle:
        handle.write("\n".join(out))

    # ------------------------------------------------ per-lineage evolution: final conv (B)
    out = []
    for name in lineage_order:
        rows = []
        for r in by_lineage[name]:
            c = r["conv_summary"]
            vel = ""
            if r["has_velocity"]:
                vel = f"{fmt(c['row_velocity_norm']['min'])} / {fmt(c['row_velocity_norm']['median'])}"
            rows.append([ckpt(r) + (f" [{r['role']}]" if r["role"] != "model" else ""),
                         fmt(r["cum_step"], "d") if r["cum_step"] is not None else "",
                         f"{fmt(c['row_norm']['min'])} / {fmt(c['row_norm']['median'])} / {fmt(c['row_norm']['max'])}",
                         fmt(c["mean_row_norm"]), fmt(c["mean_row_ratio"], ".2f"), fmt(c["residual_norm_median"]),
                         f"{fmt(c['bias_mean'], '+.3f')} / {fmt(c['bias_std'], '.3f')}",
                         f"{fmt(c['bias']['min'], '+.2f')}..{fmt(c['bias']['max'], '+.2f')}",
                         fmt(c.get("static_shared_level"), "+.1f"), fmt(c.get("shared_row_rounding_noise"), ".4f"),
                         fmt(c["underpromo_vs_queen_median"], ".2f"), fmt(c["knight_vs_queen_median"], ".2f"),
                         fmt(c["queen_promo_vs_queen_median"], ".2f"),
                         f"{fmt(c.get('row_rel_change', {}).get('min'), '.3f')} / {fmt(c.get('row_rel_change', {}).get('median'), '.3f')}",
                         fmt(r["tensors"]["policy.conv.weight"]["max_abs"], ".2f"), len(c["final_col_weak"]), vel])
        out.append(f"#### {name}\n\n" + table(
            ["checkpoint", "cum", "row norm min / median / max", "mean-row norm", "mean-row ratio",
             "residual norm median", "bias mean / std", "bias range", "static shared level",
             "shared-row rounding noise (nats)", "underpromo/Q", "knight/Q", "Q-promo/Q",
             "row rel. change vs reference min / median", "max abs W", "weak final cols",
             "row velocity norm min / median"], rows))
    with open(os.path.join(TABLES, "evolution_final.md"), "w") as handle:
        handle.write("\n".join(out))

    # ------------------------------------------------ families per lineage-latest (B)
    family_names = [f"queen-style dist {d}" for d in range(1, 8)] + ["knight", "underpromo knight", "underpromo rook",
                                                                      "underpromo bishop", "underpromo dir fwd",
                                                                      "underpromo dir capL", "underpromo dir capR",
                                                                      "queen-promo"]
    rows_norm, rows_bias, rows_change, rows_residual = [], [], [], []
    for name in lineage_order:
        for r in by_lineage[name]:
            if not r["is_latest"]:
                continue
            fam = r["families"]
            label = [name, ckpt(r) + (f" [{r['role']}]" if r["role"] != "model" else "")]
            rows_norm.append(label + [fmt(fam[f]["row_norm_vs_queen_median"], ".2f") for f in family_names])
            rows_bias.append(label + [fmt(fam[f]["bias_mean"], "+.2f") for f in family_names])
            rows_change.append(label + [f"{r['reference_model_id']} @ {r['reference_step']}" if r["reference_model_id"] else ""]
                               + [fmt(fam[f].get("row_rel_change_mean"), ".2f") for f in family_names])
            residual = np.array(r["conv"]["residual_norm"])
            queen_residual_median = float(np.median(residual[:56]))
            rows_residual.append(label + [fmt(fam[f]["residual_norm_mean"] / queen_residual_median, ".2f") for f in family_names])
    short = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Kn", "UP-N", "UP-R", "UP-B", "UP-fwd", "UP-capL", "UP-capR", "QP"]
    with open(os.path.join(TABLES, "families_latest.md"), "w") as handle:
        handle.write("**Row norm, family mean / median queen-style row norm**\n\n")
        handle.write(table(["lineage", "checkpoint"] + short, rows_norm))
        handle.write("\n**Bias, family mean (nats; init 0)**\n\n")
        handle.write(table(["lineage", "checkpoint"] + short, rows_bias))
        handle.write("\n**Residual row norm (row minus the shared mean row), family mean / median queen-style residual** — "
                     "the family comparison that is not masked by the shared row\n\n")
        handle.write(table(["lineage", "checkpoint"] + short, rows_residual))
        handle.write("\n**Row relative change vs lineage reference, family mean (‖W−W_ref‖/‖W_ref‖)** — meaningful only where "
                     "the reference is the lineage's fresh net or seed; for self-play rows the reference is simply the "
                     "earliest surviving file of that run\n\n")
        handle.write(table(["lineage", "checkpoint", "reference"] + short, rows_change))

    # ------------------------------------------------ per-direction (queen-style) for latest
    rows = []
    for name in lineage_order:
        for r in by_lineage[name]:
            if not r["is_latest"]:
                continue
            fam = r["families"]
            rows.append([name, ckpt(r) + (f" [{r['role']}]" if r["role"] != "model" else "")]
                        + [f"{fam['queen-style dir ' + d]['row_norm_vs_queen_median']:.2f} / {fam['queen-style dir ' + d]['bias_mean']:+.2f}"
                           for d in lib.QUEEN_DIRECTIONS])
    with open(os.path.join(TABLES, "directions_latest.md"), "w") as handle:
        handle.write(table(["lineage", "checkpoint"] + [f"{d} norm/Q-med / bias" for d in lib.QUEEN_DIRECTIONS], rows))

    # ------------------------------------------------ input usage (C)
    rows = []
    for name in lineage_order:
        for r in by_lineage[name]:
            if not r["is_latest"]:
                continue
            c = r["conv_summary"]
            p = r.get("pre_summary")
            col = c["final_col_norm"]
            rows.append([name, ckpt(r) + (f" [{r['role']}]" if r["role"] != "model" else ""), r["K"],
                         f"{fmt(col['min'])} / {fmt(col['median'])} / {fmt(col['max'])}",
                         fmt(col["min"] / col["median"], ".2f"), len(c["final_col_weak"]),
                         c.get("weak_col_and_dead_or_off", "n/a"),
                         f"{fmt(p['effective_contribution']['min'])} / {fmt(p['effective_contribution']['median'])} / {fmt(p['effective_contribution']['max'])}" if p else "n/a",
                         f"{len(p['pre_col_weak'])} of {r['input_width']}" if p else "n/a",
                         f"{fmt(p['pre_col_norm']['min'] / p['pre_col_norm']['median'], '.2f')}" if p else "n/a"])
    with open(os.path.join(TABLES, "input_usage.md"), "w") as handle:
        handle.write(table(["lineage", "checkpoint", "K", "final-conv column norm min / median / max",
                            "min/median", "cols < 10% median", "weak AND dead/mostly-off",
                            "effective contribution (col norm × act std) min / median / max",
                            "tower channels barely read (pre_conv col < 10% median)", "pre_conv col min/median"], rows))

    # ------------------------------------------------ offset carriers
    rows = []
    for name in lineage_order:
        for r in by_lineage[name]:
            if not r["is_latest"] or "pre_summary" not in r:
                continue
            p = r["pre_summary"]
            c = r["conv_summary"]
            tops = "; ".join(f"k{t['k']}: {t['contribution']:+.2f} (β {t['beta']:+.2f}, γ {t['gamma']:+.2f}, rv {t['running_var']:.3g}, col {t['final_col_norm']:.2f}, shared frac {t['column_shared_fraction']:.2f})"
                             for t in p["top_shared_contributors"])
            rows.append([name, ckpt(r) + (f" [{r['role']}]" if r["role"] != "model" else ""),
                         fmt(c["static_shared_level"], "+.2f"), fmt(c["bias_mean"], "+.3f"),
                         fmt(c["shared_from_mean_row"], "+.2f"), p["always_on"],
                         fmt(p["shared_level_from_always_on"], "+.2f"), tops])
    with open(os.path.join(TABLES, "offset_carriers.md"), "w") as handle:
        handle.write(table(["lineage", "checkpoint", "static shared level", "from bias mean", "from mean row · E[a]",
                            "always-on channels", "from always-on", "top 5 contributors (k: m_k·E[a_k])"], rows))

    # ------------------------------------------------ validation vs bf16 survey
    survey = {}
    if os.path.exists(SURVEY):
        for line in open(SURVEY):
            record = json.loads(line)
            if "policy_mag" in record and record.get("model_id"):
                survey[(record["model_id"], str(record["step"]))] = record["policy_mag"]
    trajectory = list(csv.DictReader(open(os.path.join(RESULTS, "trajectory.csv"))))
    rows = []
    seen = set()
    for row in trajectory:
        key = (row["model_id"], row["step"] if row["step"] not in ("", "-1") else "None")
        if key in survey and key not in seen and row.get("static_shared_level"):
            seen.add(key)
            rows.append([key[0], key[1], fmt(float(row["static_shared_level"]), "+.2f"),
                         fmt(survey[key]["glob_mean_med"], "+.2f"), fmt(survey[key]["legal_mean_med"], "+.2f")])
    with open(os.path.join(TABLES, "validation_static_level.md"), "w") as handle:
        handle.write(table(["model_id", "step", "static shared level (this study, weights only)",
                            "measured all-logit mean, median over positions (bf16-head-offset survey)",
                            "measured legal-logit mean (survey)"], rows))

    # ------------------------------------------------ leaky-FC1 vs ReLU (D)
    leaky = {r["training_step"]: r for r in by_lineage["leaky-FC1 (SE scale+bias, FC1 leaky)"]}
    relu = {r["training_step"]: r for r in by_lineage["SE scale+bias s1"]}
    relu_seed2 = {r["training_step"]: r for r in by_lineage["SE scale+bias s2"]}
    common = sorted(s for s in leaky if s in relu and s is not None)
    fresh_leaky = leaky[None]
    rows = []
    diff_rows = []
    names = ["policy.pre_conv.weight", "policy.pre_bn.weight", "policy.pre_bn.bias", "policy.pre_bn.running_mean",
             "policy.pre_bn.running_var", "policy.conv.weight", "policy.conv.bias"]
    _, fresh_tensors, _ = lib.read_policy(fresh_leaky["path"])
    for step in [None] + common:
        for label, r in (("leaky-FC1", leaky[step]), ("ReLU s1", relu[step])):
            p, c = r["pre_summary"], r["conv_summary"]
            rows.append([fmt(step, "d") if step else "fresh", label, ckpt(r), p["dead"], p["mostly_off"], p["always_on"],
                         fmt(p["abs_gamma"]["median"], ".3f"), f"{p['beta']['min']:+.3f}..{p['beta']['max']:+.3f}",
                         f"{fmt(p['running_var']['min'])} / {fmt(p['running_var']['median'])} / {fmt(p['running_var']['max'])}",
                         fmt(p["pre_row_norm"]["median"], ".3f"), fmt(c["row_norm"]["median"], ".3f"),
                         fmt(c["mean_row_ratio"], ".3f"), fmt(c["bias_mean"], "+.4f"), fmt(c["bias_std"], ".3f"),
                         fmt(c["static_shared_level"], "+.3f"), fmt(c["underpromo_vs_queen_median"], ".3f"),
                         fmt(c.get("row_rel_change", {}).get("median"), ".3f")])
        if step is None:
            continue
        _, a, _ = lib.read_policy(leaky[step]["path"])
        _, b, _ = lib.read_policy(relu[step]["path"])
        cells = []
        for name in names:
            moved_a = np.linalg.norm(a[name] - fresh_tensors[name])
            moved_b = np.linalg.norm(b[name] - fresh_tensors[name])
            between = np.linalg.norm(a[name] - b[name])
            cells.append(f"{between / max(moved_b, 1e-30):.2f}")
        Wa, Wb = a["policy.conv.weight"].reshape(76, -1), b["policy.conv.weight"].reshape(76, -1)
        row_cos = (Wa * Wb).sum(1) / (np.linalg.norm(Wa, axis=1) * np.linalg.norm(Wb, axis=1))
        diff_rows.append([step] + cells + [f"{row_cos.min():.3f} / {np.median(row_cos):.3f}",
                                           f"{np.corrcoef(a['policy.conv.bias'].ravel(), b['policy.conv.bias'].ravel())[0, 1]:.4f}"])
    with open(os.path.join(TABLES, "leaky_vs_relu.md"), "w") as handle:
        handle.write(table(["step", "run", "checkpoint", "dead", "mostly-off", "always-on", "median abs(gamma)",
                            "beta range", "running var min / median / max", "pre_conv row norm median",
                            "final row norm median", "mean-row ratio", "bias mean", "bias std",
                            "static shared level", "underpromo/Q", "row rel. change vs fresh (median)"], rows))
        handle.write("\n**Distance between the two runs relative to how far the ReLU run moved from the shared fresh net** "
                     "(‖leaky − ReLU‖ / ‖ReLU − fresh‖ per tensor; 0 = identical, 1 = as different as the training "
                     "displacement itself), plus per-row cosine of the final conv and bias correlation:\n\n")
        handle.write(table(["step"] + names + ["final-conv row cosine min / median", "bias corr"], diff_rows))
        # context: another single-change run from the same fresh weights (SE zero-beta s1, derived from JZOe)
        zero_beta = {r["training_step"]: r for r in by_lineage["SE zero-beta s1"]}
        context_rows = []
        for step in sorted(s for s in zero_beta if s is not None and s in relu):
            _, a, _ = lib.read_policy(zero_beta[step]["path"])
            _, b, _ = lib.read_policy(relu[step]["path"])
            cells = []
            for name in names:
                cells.append(f"{np.linalg.norm(a[name] - b[name]) / max(np.linalg.norm(b[name] - fresh_tensors[name]), 1e-30):.2f}")
            context_rows.append([step] + cells)
        handle.write("\n**Context: the same distance for SE zero-beta s1 (fresh net derived from the same JZOe weights, SE β init "
                     "changed) vs ReLU s1** — how far a different single change moves the policy head:\n\n")
        handle.write(table(["step"] + names, context_rows))
        # velocity comparator: leaky vs ReLU seed 2 (the ReLU seed 1 checkpoints carry no velocity)
        vrows = []
        for step in sorted(s for s in leaky if s is not None):
            for label, source in (("leaky-FC1", leaky), ("ReLU s2", relu_seed2)):
                r = source.get(step)
                if r is None or not r["has_velocity"]:
                    continue
                p, c = r["pre_summary"], r["conv_summary"]
                vrows.append([step, label, ckpt(r), len(p["zero_velocity_gamma_beta_channels"]),
                              len(p["zero_velocity_pre_rows"]), len(c["zero_velocity_rows"]),
                              f"{fmt(p['pre_row_velocity_norm']['min'])} / {fmt(p['pre_row_velocity_norm']['median'])} / {fmt(p['pre_row_velocity_norm']['max'])}",
                              f"{fmt(p['gamma_velocity_abs']['median'])} / {fmt(p['beta_velocity_abs']['median'])}",
                              f"{fmt(c['row_velocity_norm']['min'])} / {fmt(c['row_velocity_norm']['median'])} / {fmt(c['row_velocity_norm']['max'])}",
                              f"{fmt(p['pre_row_velocity_cos_w']['min'], '+.4f')}..{fmt(p['pre_row_velocity_cos_w']['max'], '+.4f')}"])
        handle.write("\n**Optimizer velocity, leaky-FC1 vs the ReLU seed-2 run (the only ReLU scale+bias run whose checkpoints carry velocity)**\n\n")
        handle.write(table(["step", "run", "checkpoint", "zero-velocity gamma+beta channels", "zero-velocity pre rows",
                            "zero-velocity final rows", "pre row velocity norm min / median / max",
                            "median abs velocity gamma / beta", "final row velocity norm min / median / max",
                            "cos(velocity, weight) range, pre rows"], vrows))

    # ------------------------------------------------ SE arms (all seeds)
    rows = []
    for name in lineage_order:
        if not name.startswith("SE ") and not name.startswith("leaky"):
            continue
        for r in by_lineage[name]:
            p, c = r["pre_summary"], r["conv_summary"]
            rows.append([name, ckpt(r), p["dead"], p["mostly_off"], p["always_on"], p["flat"],
                         fmt(p["abs_gamma"]["median"], ".3f"), f"{p['beta']['min']:+.3f}..{p['beta']['max']:+.3f}",
                         f"{fmt(p['running_var']['min'])} / {fmt(p['running_var']['median'])} / {fmt(p['running_var']['max'])}",
                         fmt(p["mean_over_std"]["max"], ".2f"),
                         fmt(c["row_norm"]["median"], ".3f"), fmt(c["mean_row_ratio"], ".3f"),
                         fmt(c["bias_mean"], "+.4f"), fmt(c["static_shared_level"], "+.3f"),
                         fmt(c["underpromo_vs_queen_median"], ".3f"), fmt(c["knight_vs_queen_median"], ".3f"),
                         fmt(c.get("row_rel_change", {}).get("min"), ".3f"),
                         fmt(r["tensors"]["policy.conv.weight"]["max_abs"], ".3f")])
    with open(os.path.join(TABLES, "se_arms.md"), "w") as handle:
        handle.write(table(["run", "checkpoint", "dead", "mostly-off", "always-on", "flat", "median abs(gamma)",
                            "beta range", "running var min / median / max", "max abs(mu)/sigma",
                            "final row norm median", "mean-row ratio", "bias mean", "static shared level",
                            "underpromo/Q", "knight/Q", "min row rel. change vs fresh", "max abs W"], rows))

    # ------------------------------------------------ growth over lineage (trajectory first -> last)
    growth = json.load(open(os.path.join(RESULTS, "growth.json")))
    rows = []
    metrics = ["static_shared_level", "bias_mean", "mean_row_norm", "row_norm_median", "conv_w_max_abs",
               "pre_row_norm_median", "rv_max", "gamma_median", "shared_row_rounding_noise"]
    for entry in growth:
        if entry["lineage"] not in lineage_order:
            continue
        rows.append([entry["lineage"], entry["first"], entry["last"]]
                    + [f"{fmt(entry[m][0])} → {fmt(entry[m][1])}" if m in entry else "" for m in metrics])
    with open(os.path.join(TABLES, "growth.md"), "w") as handle:
        handle.write(table(["lineage", "first trained", "last"] + metrics, rows))

    # ------------------------------------------------ findings (ranked, collapsed)
    grouped = {}
    for f in findings:
        key = (f["lineage"], f["rule"], f["tensor"], f["index"])
        grouped.setdefault(key, []).append(f)
    rows = []
    for key, items in grouped.items():
        latest = items[-1]
        latest_flag = any(i["is_latest"] == "True" for i in items)
        chosen = next((i for i in items if i["is_latest"] == "True"), items[-1])
        others = sorted({i["checkpoint"] for i in items if i is not chosen})
        rows.append((SEVERITY_ORDER[chosen["severity"]], 0 if latest_flag else 1, key[0], key[1], key[2], key[3],
                     chosen, others))
    rows.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4], str(t[5])))
    out_rows = []
    for severity, latest_rank, lineage, rule, tensor, index, chosen, others in rows:
        value = chosen["value"]
        try:
            value = fmt(float(value), ".4g")
        except (TypeError, ValueError):
            pass
        out_rows.append([chosen["severity"], lineage, chosen["checkpoint"], rule, f"`{tensor}`", index, value,
                         chosen["why"], "yes" if latest_rank == 0 else "no", len(others)])
    with open(os.path.join(TABLES, "findings_ranked.md"), "w") as handle:
        handle.write(table(["severity", "lineage", "checkpoint", "rule", "tensor", "index", "value", "why",
                            "present at lineage-latest", "other detailed checkpoints with the same hit"], out_rows))

    # ------------------------------------------------ 76-row final-conv tables for every lineage-latest
    out = []
    for name in lineage_order:
        for r in by_lineage[name]:
            if not r["is_latest"]:
                continue
            c = r["conv"]
            rows = []
            for channel in range(76):
                rows.append([channel, lib.channel_label(channel), fmt(c["row_norm"][channel], ".3f"),
                             fmt(c["residual_norm"][channel], ".3f"), fmt(c["row_cos_mean_row"][channel], "+.3f"),
                             fmt(c["bias"][channel], "+.3f"),
                             fmt(c["static_logit_level"][channel], "+.2f") if "static_logit_level" in c else "",
                             fmt(c["residual_logit_std_estimate"][channel], ".3f") if "residual_logit_std_estimate" in c else "",
                             fmt(c["row_rel_change"][channel], ".3f") if "row_rel_change" in c else "",
                             fmt(c["bias_delta"][channel], "+.3f") if "bias_delta" in c else "",
                             fmt(c["row_velocity_norm"][channel], ".3g") if "row_velocity_norm" in c else "",
                             fmt(c["row_velocity_cos_w"][channel], "+.3f") if "row_velocity_cos_w" in c else ""])
            out.append(f"#### {name} — {ckpt(r)}" + (f" [{r['role']}]" if r["role"] != "model" else "")
                       + f" (reference {r['reference_model_id']} @ {r['reference_step']})\n\n" + table(
                ["ch", "move type", "row norm", "residual norm", "cos(row, mean row)", "bias", "static logit level",
                 "residual logit std (indep. est.)", "row rel. change vs ref", "bias change vs ref",
                 "row velocity norm", "cos(velocity, row)"], rows))
    with open(os.path.join(TABLES, "final_conv_channels_latest.md"), "w") as handle:
        handle.write("\n".join(out))
    print("tables written to", TABLES)


if __name__ == "__main__":
    main()
