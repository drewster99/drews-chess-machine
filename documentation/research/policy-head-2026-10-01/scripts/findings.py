#!/usr/bin/env python3
"""Health rules over the detailed analyses and the full trajectory.

Writes results/findings.csv (every hit, every checkpoint) and
results/tables/findings.md (ranked; identical (lineage, rule, tensor, index)
hits are collapsed onto their latest checkpoint, with the other steps listed).

Thresholds are named constants below; severities:
  CRITICAL  numerically broken (NaN/Inf, a trainer tensor with all-zero velocity)
  HIGH      a component is not working or a known failure mechanism is active
  MEDIUM    partly not working, or a mechanism that could become a problem
  LOW       worth knowing, no evidence of harm
"""
import csv
import gzip
import json
import math
import os
import sys
from collections import defaultdict

import policy_head_lib as lib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "..", "results")

DEAD_ON_RATIO = -3.0
MOSTLY_OFF_ON_RATIO = -2.0
SHARED_LEVEL_HIGH = 32.0          # bf16 spacing >= 0.25 at the shared logit level
SHARED_LEVEL_MEDIUM = 8.0         # bf16 spacing >= 0.0625
SHARED_ROUNDING_NOISE_HIGH = 0.05  # nats, per-square shared-row rounding noise std
SHARED_ROUNDING_NOISE_MEDIUM = 0.01
MEAN_ROW_RATIO_HIGH = 1.0
MEAN_ROW_RATIO_MEDIUM = 0.5
BN_MEAN_OVER_STD_HIGH = 20.0
BN_MEAN_OVER_STD_MEDIUM = 8.0
RV_HOT_RATIO = 10.0
RV_SPREAD_HIGH = 100.0            # max/median running_var
ALWAYS_ON_CONTRIBUTION = 1.0      # nats added to every logit by one always-on channel
ALWAYS_ON_SHARE = 0.25            # share of the weight-borne shared level carried by always-on channels
RV_COLD_RATIO = 0.05
MIRROR_BIAS_ASYMMETRY = 1.0
MIRROR_NORM_RATIO = 1.5
STUCK_ROW_REL_CHANGE = 0.02
STUCK_MIN_STEP = 5000
OUTLIER_ROW_FACTOR = 3.0
BIAS_MEAN_DRIFT = 0.5
TRAJECTORY_JUMP = 0.25
TRAJECTORY_MAX_GAP = 5000         # steps between consecutive saves for a within-segment jump
TRAJECTORY_MIN_STEP = 5000        # ignore warmup
CASTLING_PAIR = {(2 * 7 + 1, 6 * 7 + 1)}  # E2 / W2 carry O-O / O-O-O

AGGREGATED_RULES = {"hot BN channel", "cold BN channel", "BN input mean >> std", "always-on pre-block channel",
                    "final-conv row ~unchanged from reference", "tower channel barely read by the policy head",
                    "mostly-off pre-block channel", "dead pre-block channel", "flat pre-block channel",
                    "final conv barely reads pre-block channel", "final conv barely reads tower channel",
                    "near-zero pre_conv row", "mirror-pair bias asymmetry", "mirror-pair row-norm asymmetry",
                    "outlier final-conv row", "zero velocity final-conv row",
                    "zero gamma/beta velocity on a live-looking channel"}
AGGREGATE_LIST_LENGTH = 12
SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def checkpoint_label(result):
    step = result["training_step"]
    cum = result.get("cum_step")
    text = f"{result['model_id']}@{step if step is not None else 'fresh'}"
    if cum is not None and cum != step:
        text += f" (cum {cum})"
    if result.get("role") in ("trainer", "champion"):
        text += f" [{result['role']}]"
    return text


def rules_for(result):
    hits = []
    pre = result.get("pre")
    pre_summary = result.get("pre_summary", {})
    conv = result["conv"]
    conv_summary = result["conv_summary"]
    step = result["training_step"] or 0
    trained = step > 0

    def hit(severity, rule, tensor, index, value, why):
        hits.append(dict(severity=severity, rule=rule, tensor=tensor, index=index, value=value, why=why))

    for name, stats in result["tensors"].items():
        if stats["nonfinite"]:
            hit("CRITICAL", "nonfinite", name, "", stats["nonfinite"], "NaN/Inf in a stored tensor")
        if name.startswith("opt.") and stats["exact_zero_fraction"] == 1.0 and trained:
            hit("CRITICAL", "all-zero velocity", name, "", 1.0, "the whole tensor received no gradient")

    if pre is not None:
        for k in range(result["K"]):
            a = pre["on_ratio"][k]
            label = f"k={k}"
            detail = f"gamma={pre['gamma'][k]:+.4g} beta={pre['beta'][k]:+.4g} beta/|gamma|={a:+.3g} P(on)={pre['p_on'][k]:.2e}"
            if "gamma_velocity" in pre:
                zero = pre["gamma_velocity"][k] == 0 and pre["beta_velocity"][k] == 0
                detail += f" gamma/beta velocity {'ZERO' if zero else 'nonzero'}"
            if a < DEAD_ON_RATIO:
                hit("HIGH", "dead pre-block channel", "policy.pre_bn", label, a,
                    detail + "; output is ~always 0 after the activation, so the channel feeds the logits nothing")
            elif a < MOSTLY_OFF_ON_RATIO:
                hit("MEDIUM", "mostly-off pre-block channel", "policy.pre_bn", label, a,
                    detail + "; on for < 2.3% of inputs")
            if pre["flat"][k]:
                hit("MEDIUM", "flat pre-block channel", "policy.pre_bn.weight", label, pre["gamma"][k],
                    f"|gamma| < 5% of median |gamma| ({detail}); output nearly constant, carries no position information")
            if "gamma_velocity" in pre and trained and pre["gamma_velocity"][k] == 0 and pre["beta_velocity"][k] == 0 and a >= DEAD_ON_RATIO:
                hit("HIGH", "zero gamma/beta velocity on a live-looking channel", "opt.policy.pre_bn", label, 0.0, detail)
            mos = pre["mean_over_std"][k]
            if mos >= BN_MEAN_OVER_STD_HIGH:
                hit("HIGH", "BN input mean >> std", "policy.pre_bn.running_mean", label, mos,
                    f"|running_mean|/sqrt(running_var)={mos:.3g}: rounding the BN input to bf16 adds ~{pre['bn_input_rounding_noise'][k]:.3f} std of noise to the normalized value")
            elif mos >= BN_MEAN_OVER_STD_MEDIUM:
                hit("MEDIUM", "BN input mean >> std", "policy.pre_bn.running_mean", label, mos,
                    f"|running_mean|/sqrt(running_var)={mos:.3g}; bf16 input rounding noise ~{pre['bn_input_rounding_noise'][k]:.3f} std")
        for k in pre_summary["always_on_channels"]:
            contribution = pre["shared_contribution"][k]
            hit("MEDIUM" if abs(contribution) >= ALWAYS_ON_CONTRIBUTION else "LOW", "always-on pre-block channel",
                "policy.pre_bn", f"k={k}", pre["on_ratio"][k],
                f"beta/|gamma|={pre['on_ratio'][k]:+.3g}: the ReLU never clips it, so the channel is linear and its mean "
                f"E[a]={pre['act_mean'][k]:.3g} acts as a constant input; it adds {contribution:+.3g} to every logit through the "
                f"final conv's mean row (column shared fraction {pre['column_shared_fraction'][k]:.2f})")
        level = conv_summary["static_shared_level"]
        from_always_on = pre_summary["shared_level_from_always_on"]
        if abs(level) >= SHARED_LEVEL_MEDIUM and abs(from_always_on) >= ALWAYS_ON_SHARE * abs(level - conv_summary["bias_mean"]):
            hit("HIGH", "shared offset carried by always-on channels", "policy.pre_bn + policy.conv.weight",
                f"{len(pre_summary['always_on_channels'])} channels", from_always_on,
                f"{from_always_on:+.1f} of the {level - conv_summary['bias_mean']:+.1f} weight-borne shared logit level comes from "
                f"channels the activation never clips; their final-conv columns point along the mean row")
        if pre_summary["rv_max_over_median"] >= RV_SPREAD_HIGH:
            hit("MEDIUM", "BN running-var spread", "policy.pre_bn.running_var", f"k={pre_summary['hottest_channel']}",
                pre_summary["rv_max_over_median"],
                f"max/median running_var = {pre_summary['rv_max_over_median']:.0f} (span max/min {pre_summary['rv_span']:.3g}); "
                "a few pre-conv outputs are orders of magnitude larger than the rest")
        rv = pre["running_var"]
        median_rv = sorted(rv)[len(rv) // 2]
        for k in range(result["K"]):
            if rv[k] / median_rv >= RV_HOT_RATIO:
                hit("MEDIUM", "hot BN channel", "policy.pre_bn.running_var", f"k={k}", rv[k],
                    f"running_var {rv[k]:.4g} = {rv[k] / median_rv:.1f}x the channel median (beta/|gamma|={pre['on_ratio'][k]:+.2f}, pre_conv row norm {pre['pre_row_norm'][k]:.3g})")
            if rv[k] / median_rv <= RV_COLD_RATIO:
                hit("MEDIUM", "cold BN channel", "policy.pre_bn.running_var", f"k={k}", rv[k],
                    f"running_var {rv[k]:.4g} = {rv[k] / median_rv:.3f}x the channel median; BN divides by a tiny std, amplifying rounding noise of its input")
        if pre_summary.get("pre_row_near_zero"):
            for k in pre_summary["pre_row_near_zero"]:
                hit("MEDIUM", "near-zero pre_conv row", "policy.pre_conv.weight", f"row {k}", pre["pre_row_norm"][k],
                    "row norm < 5% of median; BN rescales it, so the channel is driven by rounding noise")
        for k in pre_summary.get("pre_col_weak", []):
            hit("LOW", "tower channel barely read by the policy head", "policy.pre_conv.weight", f"col {k}",
                None, "pre_conv input column norm < 10% of median")
        if abs(level) >= SHARED_LEVEL_HIGH:
            hit("HIGH", "large shared policy logit level", "policy.conv", "", level,
                f"static mean logit {level:+.1f}: bf16 spacing there is {lib.bf16_spacing(level):.3g}; any path that rounds logits to bf16 merges near-equal moves")
        elif abs(level) >= SHARED_LEVEL_MEDIUM:
            hit("MEDIUM", "elevated shared policy logit level", "policy.conv", "", level,
                f"static mean logit {level:+.1f}: bf16 spacing {lib.bf16_spacing(level):.3g}")
        noise = conv_summary["shared_row_rounding_noise"]
        if noise >= SHARED_ROUNDING_NOISE_HIGH:
            hit("HIGH", "shared-row feature-rounding noise", "policy.conv.weight", "mean row", noise,
                f"rounding the K policy features to bf16 adds ~{noise:.3f} nats of per-square noise common to all move types at that square (does not cancel in softmax)")
        elif noise >= SHARED_ROUNDING_NOISE_MEDIUM:
            hit("MEDIUM", "shared-row feature-rounding noise", "policy.conv.weight", "mean row", noise,
                f"~{noise:.3f} nats per-square noise from bf16 feature rounding")
        dead_or_off = set(pre_summary["dead_channels"]) | set(pre_summary["mostly_off_channels"])
        for k in conv_summary["final_col_weak"]:
            hit("MEDIUM", "final conv barely reads pre-block channel", "policy.conv.weight", f"col {k}",
                pre["final_col_norm"][k],
                f"column norm < 10% of median; {'channel is also dead/mostly-off' if k in dead_or_off else 'channel is live by BN params'}")
    else:
        for k in conv_summary["final_col_weak"]:
            hit("MEDIUM", "final conv barely reads tower channel", "policy.conv.weight", f"col {k}", None,
                "column norm < 10% of median")

    ratio = conv_summary["mean_row_ratio"]
    if ratio >= MEAN_ROW_RATIO_HIGH:
        hit("HIGH", "dominant shared row", "policy.conv.weight", "mean row", ratio,
            f"||mean row|| / mean ||row|| = {ratio:.2f}: the 76 move-type rows share one large common direction (the offset carrier)")
    elif ratio >= MEAN_ROW_RATIO_MEDIUM:
        hit("MEDIUM", "large shared row", "policy.conv.weight", "mean row", ratio,
            f"||mean row|| / mean ||row|| = {ratio:.2f}")
    if abs(conv_summary["bias_mean"]) >= BIAS_MEAN_DRIFT:
        hit("LOW", "bias mean drift", "policy.conv.bias", "", conv_summary["bias_mean"],
            "softmax-invisible shared part of the bias (init 0); no weight decay on biases")

    row_norm = conv["row_norm"]
    median_norm = sorted(row_norm)[len(row_norm) // 2]
    for channel in range(76):
        if row_norm[channel] > OUTLIER_ROW_FACTOR * median_norm:
            hit("MEDIUM", "outlier final-conv row", "policy.conv.weight", f"{channel} {lib.channel_label(channel)}",
                row_norm[channel], f"{row_norm[channel] / median_norm:.1f}x the median row norm")
        reference_step = result.get("reference_step")
        reference_step = 0 if reference_step in (None, "fresh", "") else int(reference_step)
        if (trained and step - reference_step >= STUCK_MIN_STEP and "row_rel_change" in conv
                and conv["row_rel_change"][channel] < STUCK_ROW_REL_CHANGE):
            hit("MEDIUM", "final-conv row ~unchanged from reference", "policy.conv.weight",
                f"{channel} {lib.channel_label(channel)}", conv["row_rel_change"][channel],
                f"relative change {conv['row_rel_change'][channel]:.4f} vs reference ({result['reference_model_id']}@{result['reference_step']}); that move type is barely trained")
        if trained and "row_velocity_norm" in conv and conv["row_velocity_norm"][channel] == 0:
            hit("HIGH", "zero velocity final-conv row", "opt.policy.conv.weight.velocity",
                f"{channel} {lib.channel_label(channel)}", 0.0, "no gradient reached this move type recently")
    for pair in result["mirror_pairs"]:
        if (pair["channel"], pair["mirror"]) in CASTLING_PAIR:
            continue
        if abs(pair["bias_diff"]) >= MIRROR_BIAS_ASYMMETRY:
            hit("LOW", "mirror-pair bias asymmetry", "policy.conv.bias", f"{pair['label']} vs {pair['mirror_label']}",
                pair["bias_diff"], "left-right mirrored move types should have similar priors")
        if max(pair["row_norm_ratio"], 1 / pair["row_norm_ratio"]) >= MIRROR_NORM_RATIO:
            hit("LOW", "mirror-pair row-norm asymmetry", "policy.conv.weight", f"{pair['label']} vs {pair['mirror_label']}",
                pair["row_norm_ratio"], "mirrored move types have very different weight norms")
    return hits


def trajectory_rules(rows):
    """Consecutive-checkpoint jumps and start->end growth per lineage."""
    hits = []
    by_lineage = defaultdict(list)
    for row in rows:
        if row["lineage"].startswith("(unassigned)"):
            continue
        by_lineage[row["lineage"]].append(row)
    metrics = ["conv_w_max_abs", "mean_row_norm", "row_norm_median", "pre_row_norm_median", "rv_max", "gamma_median"]
    growth = []
    for lineage, members in by_lineage.items():
        trained = [r for r in members if r["step"] not in ("", "-1") and int(r["step"]) > 0]
        # one row per checkpoint content (models first; trainer files of
        # self-play sessions hold fp32 masters and are compared separately)
        models = [r for r in trained if r["role"] != "trainer"]
        for previous, current in zip(models, models[1:]):
            boundary = previous["segment_index"] != current["segment_index"]
            gap = int(current["step"]) - int(previous["step"])
            # Within a segment, only compare saves at the normal cadence and
            # past warmup; a long gap between surviving files is not a jump.
            if not boundary and (gap > TRAJECTORY_MAX_GAP or int(previous["step"]) < TRAJECTORY_MIN_STEP):
                continue
            for metric in metrics:
                if previous.get(metric, "") in ("", None) or current.get(metric, "") in ("", None):
                    continue
                a, b = float(previous[metric]), float(current[metric])
                if a != 0 and abs(b - a) / abs(a) > TRAJECTORY_JUMP:
                    hits.append(dict(lineage=lineage, severity="MEDIUM" if not boundary else "LOW",
                                     rule="trajectory jump" + (" at segment boundary" if boundary else ""),
                                     checkpoint=f"{current['model_id']}@{current['step']}",
                                     tensor=metric, index=f"from {previous['model_id']}@{previous['step']}",
                                     value=b / a, why=f"{metric} {a:.4g} -> {b:.4g} between consecutive saves"))
        if len(models) >= 2:
            first, last = models[0], models[-1]
            entry = dict(lineage=lineage, first=f"{first['model_id']}@{first['step']}", last=f"{last['model_id']}@{last['step']}")
            for metric in metrics + ["static_shared_level", "bias_mean", "shared_row_rounding_noise"]:
                if first.get(metric, "") not in ("", None) and last.get(metric, "") not in ("", None):
                    entry[metric] = (float(first[metric]), float(last[metric]))
            growth.append(entry)
    return hits, growth


def main():
    detailed = [json.loads(line) for line in gzip.open(os.path.join(RESULTS, "detailed.jsonl.gz"), "rt")]
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "trajectory.csv"))))
    all_hits = []
    for result in detailed:
        for item in rules_for(result):
            item.update(lineage=result["lineage"], checkpoint=checkpoint_label(result),
                        step=result["training_step"] or 0, is_latest=result["is_latest"])
            all_hits.append(item)
    trajectory_hits, growth = trajectory_rules(rows)
    for item in trajectory_hits:
        item.update(step=0, is_latest=False)
    all_hits.extend(trajectory_hits)

    with open(os.path.join(RESULTS, "findings.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["severity", "lineage", "checkpoint", "rule", "tensor", "index", "value", "why", "is_latest"])
        writer.writeheader()
        for item in sorted(all_hits, key=lambda h: (SEVERITY_ORDER[h["severity"]], h["lineage"], h["rule"], str(h["index"]))):
            writer.writerow({k: item.get(k, "") for k in writer.fieldnames})
    # Per-channel rules collapsed to one row per (checkpoint, rule, severity)
    # for the ranked table; findings.csv keeps every channel.
    aggregated = {}
    for item in all_hits:
        if item["rule"] not in AGGREGATED_RULES:
            continue
        key = (item["lineage"], item["checkpoint"], item["rule"], item["severity"])
        aggregated.setdefault(key, []).append(item)
    collapsed = [item for item in all_hits if item["rule"] not in AGGREGATED_RULES]
    for key, items in aggregated.items():
        items.sort(key=lambda i: -abs(float(i["value"])) if i["value"] not in (None, "") else 0)
        shown = ", ".join(str(i["index"]) for i in items[:AGGREGATE_LIST_LENGTH])
        more = f" (+{len(items) - AGGREGATE_LIST_LENGTH} more)" if len(items) > AGGREGATE_LIST_LENGTH else ""
        extreme = items[0]
        collapsed.append(dict(extreme, index=f"{len(items)} ch: {shown}{more}",
                              why=f"most extreme: {extreme['index']}: {extreme['why']}"))
    with open(os.path.join(RESULTS, "findings_aggregated.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["severity", "lineage", "checkpoint", "rule", "tensor", "index", "value", "why", "is_latest"])
        writer.writeheader()
        for item in sorted(collapsed, key=lambda h: (SEVERITY_ORDER[h["severity"]], h["lineage"], h["rule"], str(h["index"]))):
            writer.writerow({k: item.get(k, "") for k in writer.fieldnames})
    with open(os.path.join(RESULTS, "growth.json"), "w") as handle:
        json.dump(growth, handle, indent=1)
    print(f"{len(all_hits)} findings")


if __name__ == "__main__":
    main()
