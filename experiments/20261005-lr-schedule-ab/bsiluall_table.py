"""B-siluall: per-checkpoint puzzle results read from each model file's own safetensors header.

Writes probes-Bsiluall.csv (one row per enumerated checkpoint) and prints the markdown tables used in README.md.
Identifies checkpoints by header metadata (training_step, model_id), never by file name.
Run with: python3 -I experiments/20261005-lr-schedule-ab/bsiluall_table.py
"""
import csv, glob, json, os, re, struct

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Models/")
STEM = "20261008-lrBsiluall-relcap-replay-step*.safetensors"
LINEAGE_RUN = "66006003-5799-4FA0-B7AE-F3C19E6FD3A5"


def header_metadata(path):
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length))["__metadata__"]


def wide_and_200(test_set_results):
    sets = {entry["id"]: entry for entry in test_set_results["sets"]}
    return sets["lichess-wide"], sets["lichess-200"]


def read_rows():
    rows = {}
    for path in sorted(glob.glob(MODELS + STEM)):
        metadata = header_metadata(path)
        if metadata["lineage_run_id"] != LINEAGE_RUN:
            raise SystemExit(f"{path}: lineage run {metadata['lineage_run_id']} is not {LINEAGE_RUN}")
        wide, small = wide_and_200(json.loads(metadata["dcm_test_set_results"]))
        step = int(metadata["training_step"])
        if step in rows:
            raise SystemExit(f"two files for training_step {step}")
        rows[step] = {
            "training_step": step, "lineage_segment_index": int(metadata["lineage_segment_index"]), "model_id": metadata["model_id"],
            "wide_pelo": wide["pelo"], "wide_nll": wide["nll"],
            "wide_top1": wide["top1_correct"], "wide_top5": wide["top5_correct"], "wide_positions": wide["positions"],
            "lichess200_pelo": small["pelo"], "lichess200_nll": small["nll"],
        }
    return rows


if __name__ == "__main__":
    rows = read_rows()
    out = os.path.join(HERE, "probes-Bsiluall.csv")
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(rows.values())).keys()))
        writer.writeheader()
        for step in sorted(rows):
            writer.writerow(rows[step])
    print(f"wrote {out} ({len(rows)} rows)")
