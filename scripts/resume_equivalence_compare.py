#!/usr/bin/env python3
"""Compare the file an uninterrupted corpus-replay run ended with against the
file the same run ended with after a save and a `--resume-exact`
(determinism plan C6 step 7; `scripts/resume_equivalence.sh` drives the runs).

Reads only the safetensors headers. Every field is compared exactly:

- `content_sha256` — the SHA-256 of the tensor data region (every master,
  batch-norm statistic and velocity tensor);
- the run's stream positions (`rng.streams.sampler_state`,
  `rng.streams.dropout_stream_state`) and `rng.dropout_philox_state`;
- the corpus feed position (`fed.corpus` epoch, next game, shard, buffer
  fill, feed phase) and the cumulative trainer step, games and positions;
- the resumed file's segment must record an exact resume with no gaps.

On a machine where a training step is not bit-reproducible on the GPU the
tensor hash can differ while every stream and feed field still matches; the
report says which fields differ so that case is visible rather than hidden.

Usage: resume_equivalence_compare.py <uninterrupted.safetensors> <resumed.safetensors>
Exit status: 0 when every field matches, 1 when any differs, 2 on a file that
cannot be read as a lineage-carrying model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dcm_lineage  # noqa: E402


def load(path):
    metadata = dcm_lineage.read_metadata(path)
    record = dcm_lineage.lineage_of(metadata, path)
    if not isinstance(record, dict):
        raise dcm_lineage.LineageError(f"{path}: written before lineage records; nothing to compare")
    if "content_sha256" not in metadata:
        raise dcm_lineage.LineageError(f"{path}: no content_sha256 in the header")
    return metadata, record


def field(record, path, source):
    node = record
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise dcm_lineage.LineageError(f"{source}: lineage has no {'.'.join(path)}")
        node = node[key]
    return node


COMPARED = [
    ("rng", "streams", "master_seed"),
    ("rng", "streams", "sampler_state"),
    ("rng", "streams", "dropout_stream_state"),
    ("rng", "dropout_philox_state"),
    ("fed", "corpus", "epoch"),
    ("fed", "corpus", "next_game_index"),
    ("fed", "corpus", "shard"),
    ("fed", "corpus", "populated_plies"),
    ("fed", "corpus", "feed_ahead_positions"),
    ("steps", "cum_trainer_step"),
    ("fed", "cum_games"),
    ("fed", "cum_positions"),
]


def compare(uninterrupted_path, resumed_path):
    """Return the list of (name, uninterrupted, resumed) that differ."""
    u_meta, u_record = load(uninterrupted_path)
    r_meta, r_record = load(resumed_path)
    differences = []
    if u_meta["content_sha256"] != r_meta["content_sha256"]:
        differences.append(("content_sha256", u_meta["content_sha256"], r_meta["content_sha256"]))
    for path in COMPARED:
        u = field(u_record, path, uninterrupted_path)
        r = field(r_record, path, resumed_path)
        if u != r:
            differences.append((".".join(path), u, r))
    exact = field(r_record, ("run", "exact_resume"), resumed_path)
    gaps = field(r_record, ("run", "not_exact_items"), resumed_path)
    if exact is not True or gaps != []:
        differences.append(("resumed run.exact_resume / not_exact_items", "true / []", f"{exact} / {gaps}"))
    return differences


def main(argv):
    if len(argv) != 3 or argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if len(argv) == 2 else 2
    try:
        differences = compare(argv[1], argv[2])
    except (dcm_lineage.LineageError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not differences:
        print("EQUIVALENT: tensors, streams, feed position and totals all match; the resume is recorded exact")
        return 0
    print("DIFFERENT:")
    for name, u, r in differences:
        print(f"  {name}: uninterrupted={u!r} resumed={r!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
