#!/usr/bin/env python3
"""Run the policy-head study end to end (CPU only, read-only on checkpoints).

1. Scan every checkpoint under the DCM application-support folder by header.
2. Assign each checkpoint to a segment (metadata model_id base; self-play
   runs split by session) and to a lineage (the explicit chains below, each
   verified from `parent_model_id` links / replay logs).
3. Trajectory: compact policy-head metrics for EVERY unique checkpoint
   (deduplicated by `content_sha256`), with the lineage root as reference.
4. Detailed analysis for the selected checkpoints (fresh / first / mid /
   segment ends / latest per lineage; matched steps for the SE and
   leaky-FC1 runs), with full per-channel tables.

Usage:
    python3 run_all.py [OUT_DIR]   (default: ../results next to this script)
"""
import csv
import datetime
import gzip
import json
import os
import re
import sys

import numpy as np

import analyze_policy_head as analysis
import policy_head_lib as lib

DCM_ROOT = os.path.expanduser("~/Library/Application Support/DrewsChessMachine")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "..", "results")
QUALIFYING_STEP = 75_000

# Ejp0 self-play run boundary: run 2 is a fresh fork started 2026-08-08 17:40
# local (CDT) — it is not a continuation of run 1.
EJP0_RUN2_START = datetime.datetime(2026, 8, 8, 17, 40).timestamp()

# Each lineage: ordered segments (model_id base, kind, cumstep_base or None).
# kind "replay" = non-suffixed model_id (fresh/replay/frozen/vs-UCI files);
# kind "selfplay:<run>" = generation-suffixed ids (plus session champions).
# cumstep_base comes from documentation/dashboards/registry.json where the
# lineage is registered; Qeu8e's bases are the sums of the earlier segments'
# final saved steps; None = no cumulative axis is defined.
LINEAGES = [
    ("SE scale+bias full-leaky fresh (never trained)", "supplement", [("20261001-23-Dmwe", "replay", 0)]),
    ("leaky-FC1 (SE scale+bias, FC1 leaky)", "experiment",
     [("20261001-42-2q0Q", "replay", 0), ("20261001-43-NbWz", "replay", 0)]),
    ("SE scale+bias s1", "experiment", [("20260929-12-JZOe", "replay", 0), ("20260929-22-bWdy", "replay", 0)]),
    ("SE scale+bias s2", "experiment", [("20260930-1-H1Oq", "replay", 0), ("20260930-4-k98x", "replay", 0)]),
    ("SE attenuate-only s1", "experiment", [("20260929-13-06yp", "replay", 0), ("20260929-23-L6Qm", "replay", 0)]),
    ("SE attenuate-only s2", "experiment", [("20260930-2-Gf9P", "replay", 0), ("20260930-5-5TXu", "replay", 0)]),
    ("SE none s1", "experiment", [("20260929-18-D9is", "replay", 0), ("20260929-24-834D", "replay", 0)]),
    ("SE none s2", "experiment", [("20260930-3-V9zk", "replay", 0), ("20260930-6-LkS6", "replay", 0)]),
    ("SE zero-beta s1", "experiment", [("20260930-7-crxN", "replay", 0), ("20260930-9-RrGx", "replay", 0)]),
    ("SE zero-beta s2", "experiment", [("20260930-8-8qyR", "replay", 0), ("20260930-10-H51a", "replay", 0)]),
    ("v5", "lineage", [("20260628-1-tWtk", "replay", 0), ("20260628-2-a5fc", "replay", 0),
                       ("20260628-9-OdUt", "replay", 45441), ("20260629-1-Uf4p", "replay", 60901),
                       ("20260703-1-Dg5v", "replay", 100320), ("20260714-1-h7vI", "replay", 368826),
                       ("20260729-1-VZ2j", "replay", 705436), ("20260802-2-Xuub", "replay", 811769),
                       ("20260805-1-0pTW", "replay", 857769)]),
    ("mini2b", "lineage", [("20260629-3-3MIV", "replay", 0), ("20260629-4-y5u7", "replay", 0),
                           ("20260630-5-BEKK", "replay", 13464), ("20260701-2-SvRu", "replay", 134159),
                           ("20260705-1-znR7", "replay", 142159)]),
    ("coxw", "lineage", [("20260629-5-Coxw", "replay", 0), ("20260629-6-yqMI", "replay", 0),
                         ("20260709-1-avoB", "replay", 55550)]),
    ("ykkk", "lineage", [("20260630-1-YkKk", "replay", 0), ("20260630-2-6y0s", "replay", 0),
                         ("20260701-1-0Iwe", "replay", 40677), ("20260710-1-amlg", "replay", 203007)]),
    ("nt8y", "lineage", [("20260701-3-nT8Y", "replay", 0), ("20260701-4-CIvL", "replay", 0),
                         ("20260701-5-bOYQ", "replay", 65883), ("20260706-2-3CZF", "replay", 136662),
                         ("20260707-1-cslu", "replay", 151662), ("20260708-4-kEiZ", "replay", 291662)]),
    ("qeu8 (replay main, ends Ejp0)", "lineage", [("20260702-7-Qeu8", "replay", 0), ("20260702-9-GLu5", "replay", 0),
                                                  ("20260703-1-Lnji", "replay", 41407), ("20260706-1-PVZp", "replay", 108915),
                                                  ("20260727-1-Ejp0", "replay", 175915)]),
    ("qeu8e (epoch branch)", "lineage", [("20260702-7-Qeu8", "replay", 0), ("20260704-1-X79T", "replay", 0),
                                         ("20260704-2-jSjr", "replay", 21224), ("20260704-3-h7Pp", "replay", 63731),
                                         ("20260708-5-0YQL", "replay", 106238), ("20260708-6-sFzi", "replay", 132730)]),
    ("qeu8-1blk128", "lineage", [("20260711-16-VRR4", "replay", 0), ("20260711-17-pycz", "replay", 0)]),
    ("qeu8init sf100sl100 vs-UCI", "lineage", [("20260702-7-Qeu8", "replay", None), ("20260712-6-lTiK", "replay", None),
                                               ("20260714-1-NYAZ", "replay", None), ("20260722-1-syxR", "replay", None)]),
    ("Ejp0 headfix-phase2 (from Ejp0 @681k)", "supplement", [("20261001-18-oeNy", "replay", None)]),
    ("Ejp0 self-play run 1", "lineage", [("20260727-1-Ejp0", "selfplay:run1", None)]),
    ("Ejp0 self-play run 2", "lineage", [("20260727-1-Ejp0", "selfplay:run2", None)]),
    ("bzw3 self-play", "lineage", [("20260601-11-bzw3", "selfplay:run1", None)]),
    ("KbHZ self-play (fp32)", "lineage", [("20260514-1-KbHZ", "selfplay:run1", None)]),
    ("sMe9 self-play (fp32)", "lineage", [("20260525-1-sMe9", "selfplay:run1", None)]),
    ("LWKa self-play (v4 12-block)", "lineage", [("20260531-9-LWKa", "selfplay:run1", None)]),
    ("LMGh self-play", "lineage", [("20260609-12-LMGh", "selfplay:run1", None)]),
    ("WjRY self-play", "lineage", [("20260609-14-WjRY", "selfplay:run1", None)]),
]

# Explicit reference (the weights a lineage started from) where the root of
# the chain is not one of its own segments. Matched by metadata.
EXPLICIT_REFERENCE = {
    "Ejp0 headfix-phase2 (from Ejp0 @681k)": ("20260727-1-Ejp0", 681000),  # [REPLAY] start-model in dcm_log_20261001-015706
    "Ejp0 self-play run 1": ("20260727-1-Ejp0", 1300000),  # qeu8 meta-1300000 seed
    "Ejp0 self-play run 2": ("20260727-1-Ejp0", 1300000),
}

SE_MATCHED_STEPS = [1000, 5000, 10000, 11000, 20000]


def scan():
    records = []
    for directory, subdirectories, names in os.walk(DCM_ROOT):
        subdirectories.sort()
        for name in sorted(names):
            path = os.path.join(directory, name)
            if name.endswith(".safetensors"):
                metadata, header = lib_read_header(path)
                records.append(dict(path=path, kind="safetensors", meta=metadata,
                                    has_policy_velocity=any(k.startswith("opt.policy") for k in header)))
            elif name.endswith(".dcmmodel"):
                metadata, tensors, _ = lib.read_dcmmodel_policy(path)
                records.append(dict(path=path, kind="dcmmodel", meta=metadata,
                                    has_policy_velocity=any(k.startswith("opt.") for k in tensors)))
    return records


def lib_read_header(path):
    import struct
    with open(path, "rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    return header.pop("__metadata__", {}), header


def classify(record):
    meta = record["meta"]
    match = re.match(r"^(\d{8}-\d+-\w{4})(-\d+)?$", meta["model_id"])
    if match is None:
        raise ValueError(f"unexpected model_id {meta['model_id']} in {record['path']}")
    base = match.group(1)
    relative = os.path.relpath(record["path"], DCM_ROOT)
    is_selfplay = match.group(2) is not None or not relative.startswith("Models/")
    if not is_selfplay:
        return base, "replay"
    if base == "20260727-1-Ejp0":
        return base, "selfplay:run2" if int(meta["created_at_unix"]) >= EJP0_RUN2_START else "selfplay:run1"
    return base, "selfplay:run1"


def step_of(record):
    value = record["meta"].get("training_step")
    return int(value) if value not in (None, "") else -1


def source_rank(record):
    relative = os.path.relpath(record["path"], DCM_ROOT)
    return (0 if relative.startswith("Models/") else 1 if relative.startswith("Sessions/") else 2, relative)


def dedupe(records):
    """One record per distinct content; prefer Models/, then Sessions/."""
    by_key = {}
    for record in records:
        # content_sha256 hashes tensor content only, so a derived net with
        # unchanged weights (e.g. a fresh net re-stamped with a new
        # architecture) shares it; identity includes model_id + step.
        key = (record["meta"]["model_id"], record["meta"].get("training_step", ""),
               record["meta"].get("content_sha256") or record["path"])
        if record["kind"] == "dcmmodel":
            key = ("dcm", lib.sha256_of_file(record["path"]))
        by_key.setdefault(key, []).append(record)
    unique = []
    for key, group in by_key.items():
        group.sort(key=source_rank)
        chosen = dict(group[0])
        chosen["duplicates"] = [os.path.relpath(r["path"], DCM_ROOT) for r in group[1:]]
        unique.append(chosen)
    return unique


def role_of(record):
    name = os.path.basename(record["path"])
    if "trainer" in name:
        return "trainer"
    if "champion" in name:
        return "champion"
    return "model"


def ordered_lineage(lineage_segments, unique):
    order = []
    for index, (base, kind, cumstep_base) in enumerate(lineage_segments):
        members = [r for r in unique if r["segment"] == (base, kind)]
        members.sort(key=lambda r: (step_of(r), int(r["meta"]["created_at_unix"]), role_of(r) != "trainer"))
        for record in members:
            order.append((index, cumstep_base, record))
    return order


def find_reference(lineage_name, order, unique):
    if lineage_name in EXPLICIT_REFERENCE:
        model_id, step = EXPLICIT_REFERENCE[lineage_name]
        candidates = [r for r in unique if r["meta"]["model_id"] == model_id and step_of(r) == step
                      and r["segment"][1] == "replay"]
        if not candidates:
            raise LookupError(f"reference {model_id}@{step} for {lineage_name} not found")
        return candidates[0]
    return order[0][2]


def compact_row(result, lineage_name, segment_index, cumstep_base, record):
    pre = result.get("pre_summary", {})
    conv = result["conv_summary"]
    step = result["training_step"]
    row = dict(lineage=lineage_name, segment_index=segment_index, model_id=result["model_id"],
               step=step if step is not None else "",
               cum_step=(cumstep_base + step) if (cumstep_base is not None and step not in (None, -1)) else "",
               creator=result["creator"], role=role_of(record),
               created=datetime.datetime.fromtimestamp(int(result["created_at_unix"])).strftime("%Y-%m-%d %H:%M"),
               path=os.path.relpath(result["path"], DCM_ROOT), style=result["policy_style"], K=result["K"],
               input_width=result["input_width"], compute=result["compute_dtype"], activation=result["activation"],
               has_velocity=int(result["has_velocity"]), nonfinite=result["nonfinite_total"],
               conv_w_max_abs=result["tensors"]["policy.conv.weight"]["max_abs"],
               conv_b_min=conv["bias"]["min"], conv_b_max=conv["bias"]["max"],
               bias_mean=conv["bias_mean"], bias_std=conv["bias_std"],
               row_norm_median=conv["row_norm"]["median"], row_norm_min=conv["row_norm"]["min"],
               row_norm_max=conv["row_norm"]["max"],
               mean_row_norm=conv["mean_row_norm"], mean_row_ratio=conv["mean_row_ratio"],
               residual_norm_median=conv["residual_norm_median"],
               underpromo_vs_queen=conv["underpromo_vs_queen_median"],
               knight_vs_queen=conv["knight_vs_queen_median"], qpromo_vs_queen=conv["queen_promo_vs_queen_median"],
               weakest_family=conv["weakest_family"], weak_cols=len(conv["final_col_weak"]),
               row_rel_change_median=conv.get("row_rel_change", {}).get("median", ""),
               row_rel_change_min=conv.get("row_rel_change", {}).get("min", ""))
    if pre:
        row.update(pre_w_max_abs=result["tensors"]["policy.pre_conv.weight"]["max_abs"],
                   dead=pre["dead"], mostly_off=pre["mostly_off"], flat=pre["flat"],
                   negative_gamma=pre["negative_gamma"], always_on=pre["always_on"],
                   shared_from_always_on=pre["shared_level_from_always_on"],
                   rv_max_over_median=pre["rv_max_over_median"],
                   gamma_min=pre["gamma"]["min"], gamma_median=pre["gamma"]["median"], gamma_max=pre["gamma"]["max"],
                   beta_min=pre["beta"]["min"], beta_median=pre["beta"]["median"], beta_max=pre["beta"]["max"],
                   on_ratio_min=pre["on_ratio"]["min"], p_on_median=pre["p_on"]["median"],
                   rv_min=pre["running_var"]["min"], rv_median=pre["running_var"]["median"],
                   rv_max=pre["running_var"]["max"], hottest_ratio=pre["hottest_ratio"],
                   mean_over_std_max=pre["mean_over_std"]["max"],
                   pre_row_norm_min=pre["pre_row_norm"]["min"], pre_row_norm_median=pre["pre_row_norm"]["median"],
                   pre_row_norm_max=pre["pre_row_norm"]["max"],
                   static_shared_level=conv["static_shared_level"], static_level_min=conv["static_level_min"],
                   static_level_max=conv["static_level_max"],
                   shared_row_rounding_noise=conv["shared_row_rounding_noise"],
                   residual_logit_std_median=conv["residual_logit_std_median"],
                   zero_vel_gamma_beta=len(pre.get("zero_velocity_gamma_beta_channels", [])) if result["has_velocity"] else "",
                   zero_vel_pre_rows=len(pre.get("zero_velocity_pre_rows", [])) if result["has_velocity"] else "")
    if result["has_velocity"]:
        row.update(zero_vel_conv_rows=len(conv.get("zero_velocity_rows", [])),
                   conv_row_velocity_min=conv["row_velocity_norm"]["min"],
                   conv_row_velocity_median=conv["row_velocity_norm"]["median"])
    return row


def select_detailed(lineage_name, group, order, extra_steps=()):
    """Indices into `order` to analyze in detail."""
    if not order:
        return []
    chosen = {0, len(order) - 1}
    if group == "experiment":
        if lineage_name.startswith("leaky"):
            chosen.update(range(len(order)))
        for index, (_, _, record) in enumerate(order):
            if step_of(record) in SE_MATCHED_STEPS or step_of(record) in extra_steps:
                chosen.add(index)
    else:
        trained = [i for i, (_, _, r) in enumerate(order) if step_of(r) > 0]
        if trained:
            chosen.add(trained[0])
            chosen.add(trained[len(trained) // 2])
        for index in range(len(order) - 1):
            if order[index][0] != order[index + 1][0] and step_of(order[index][2]) > 0:
                chosen.add(index)  # segment end
        # latest session: keep both champion and trainer
        last_step = step_of(order[-1][2])
        for index, (_, _, record) in enumerate(order):
            if step_of(record) == last_step and order[index][0] == order[-1][0]:
                chosen.add(index)
    return sorted(chosen)


def write_channel_csv(result, directory):
    os.makedirs(directory, exist_ok=True)
    stem = f"{result['model_id']}-step{result['training_step']}-{os.path.basename(result['path']).split('.')[0][-40:]}"
    stem = re.sub(r"[^A-Za-z0-9_.+-]", "_", stem)
    conv = result["conv"]
    with open(os.path.join(directory, stem + "-final.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        keys = list(conv.keys())
        writer.writerow(["channel", "label", "mirror"] + keys)
        for channel in range(lib.POLICY_CHANNELS):
            writer.writerow([channel, lib.channel_label(channel), lib.mirror_channel(channel)]
                            + [f"{conv[k][channel]:.6g}" for k in keys])
    if "pre" in result:
        pre = result["pre"]
        keys = list(pre.keys())
        with open(os.path.join(directory, stem + "-pre.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["k"] + keys)
            for k in range(result["K"]):
                writer.writerow([k] + [f"{pre[key][k]:.6g}" for key in keys])
    return stem


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = scan()
    for record in records:
        record["segment"] = classify(record)
    unique = dedupe(records)
    for record in unique:
        record["segment"] = classify(record)

    with open(os.path.join(OUT_DIR, "inventory.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "kind", "model_id", "parent_model_id", "training_step", "creator", "created",
                         "segment_base", "segment_kind", "content_sha256", "has_policy_velocity", "duplicates"])
        for record in sorted(unique, key=lambda r: (r["segment"], step_of(r), r["path"])):
            meta = record["meta"]
            writer.writerow([os.path.relpath(record["path"], DCM_ROOT), record["kind"], meta["model_id"],
                             meta.get("parent_model_id", ""), meta.get("training_step", ""), meta.get("creator", ""),
                             datetime.datetime.fromtimestamp(int(meta["created_at_unix"])).strftime("%Y-%m-%d %H:%M"),
                             record["segment"][0], record["segment"][1], meta.get("content_sha256", ""),
                             int(record["has_policy_velocity"]), ";".join(record["duplicates"])])

    # segment maxima for qualification
    segment_max = {}
    for record in unique:
        segment_max[record["segment"]] = max(segment_max.get(record["segment"], -1), step_of(record))

    leaky_base = "20261001-43-NbWz"
    leaky_latest_step = max(step_of(r) for r in unique if r["segment"] == (leaky_base, "replay"))
    lineage_rows = []
    assigned = set()
    trajectory_rows = []
    detailed = []
    detail_dir = os.path.join(OUT_DIR, "channels")
    for lineage_name, group, segments in LINEAGES:
        order = ordered_lineage(segments, unique)
        if not order:
            raise LookupError(f"lineage {lineage_name} has no checkpoints")
        max_segment_step = max(segment_max[(base, kind)] for base, kind, _ in segments if (base, kind) in segment_max)
        qualifies = group in ("experiment", "supplement") or max_segment_step > QUALIFYING_STEP
        reference = find_reference(lineage_name, order, unique)
        lineage_rows.append(dict(lineage=lineage_name, group=group, qualifies=qualifies,
                                 max_segment_step=max_segment_step, checkpoints=len(order),
                                 reference=os.path.relpath(reference["path"], DCM_ROOT),
                                 reference_model_id=reference["meta"]["model_id"],
                                 reference_step=step_of(reference),
                                 segments=[f"{b}:{k}" for b, k, _ in segments]))
        print(f"[{lineage_name}] {len(order)} checkpoints, max segment step {max_segment_step}, qualifies={qualifies}",
              flush=True)
        for segment_index, cumstep_base, record in order:
            assigned.add(record["path"])
            result = analysis.analyze(record["path"], reference["path"], compute_file_sha256=False)
            trajectory_rows.append(compact_row(result, lineage_name, segment_index, cumstep_base, record))
        if not qualifies:
            continue
        for index in select_detailed(lineage_name, group, order, extra_steps=(leaky_latest_step,)):
            segment_index, cumstep_base, record = order[index]
            result = analysis.analyze(record["path"], reference["path"], compute_file_sha256=True)
            result["lineage"] = lineage_name
            result["group"] = group
            result["segment_index"] = segment_index
            result["cum_step"] = (cumstep_base + result["training_step"]) if (
                cumstep_base is not None and result["training_step"] not in (None, -1)) else None
            result["role"] = role_of(record)
            result["position_in_lineage"] = index
            result["lineage_length"] = len(order)
            result["is_latest"] = index == len(order) - 1 or (
                step_of(record) == step_of(order[-1][2]) and segment_index == order[-1][0])
            result["channel_csv_stem"] = write_channel_csv(result, os.path.join(detail_dir, re.sub(r"[^A-Za-z0-9]+", "_", lineage_name)))
            detailed.append(result)

    # unassigned checkpoints: trajectory only, reference = earliest of segment
    by_segment = {}
    for record in unique:
        if record["path"] not in assigned:
            by_segment.setdefault(record["segment"], []).append(record)
    for segment, members in sorted(by_segment.items()):
        members.sort(key=lambda r: (step_of(r), int(r["meta"]["created_at_unix"])))
        for record in members:
            result = analysis.analyze(record["path"], members[0]["path"], compute_file_sha256=False)
            trajectory_rows.append(compact_row(result, f"(unassigned) {segment[0]}:{segment[1]}", 0, None, record))
        print(f"[unassigned {segment}] {len(members)} checkpoints, max step {segment_max[segment]}", flush=True)

    keys = []
    for row in trajectory_rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(os.path.join(OUT_DIR, "trajectory.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in trajectory_rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    with open(os.path.join(OUT_DIR, "lineages.json"), "w") as handle:
        json.dump(lineage_rows, handle, indent=1)
    with gzip.open(os.path.join(OUT_DIR, "detailed.jsonl.gz"), "wt") as handle:
        for result in detailed:
            handle.write(json.dumps(result) + "\n")
    print(f"trajectory rows {len(trajectory_rows)}, detailed {len(detailed)}")


if __name__ == "__main__":
    main()
