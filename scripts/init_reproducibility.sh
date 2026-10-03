#!/bin/zsh
# Cross-machine check of seeded weight initialization (determinism plan B1.1).
#
# Mints fresh models with --new-model --init-seed for a set of seeds and
# presets, then prints the SHA-256 of each file's TENSOR DATA only (every mint
# gets a new ModelID and timestamp, so whole-file hashes always differ). Run it
# on each machine with the same build and compare the outputs: every line must
# match. The trainable tensors are pure CPU arithmetic and must match exactly;
# the BN running statistics come from a GPU forward pass and are reported
# separately, because GPU kernels need not be bit-identical across chips.
#
# Usage: scripts/init_reproducibility.sh <DrewsChessMachine binary> <scratch folder>
# The scratch folder must not exist yet; it receives the minted files.
#
# pipefail: each hash is piped through sed, and a pipeline's status is otherwise
# sed's, so a file that could not be hashed would only drop its line and the script
# would still exit 0 — a listing with lines missing that reads as complete.
set -euo pipefail

BIN=${1:?usage: $0 <DrewsChessMachine binary> <scratch folder>}
OUT=${2:?usage: $0 <DrewsChessMachine binary> <scratch folder>}
[ -x "$BIN" ] || { echo "error: $BIN is not executable" >&2; exit 2; }
[ -e "$OUT" ] && { echo "error: $OUT already exists; give a new folder" >&2; exit 2; }
mkdir -p "$OUT"

SEEDS=(1 2 42 1234 99999 18446744073709551615 7777777 31337)
PRESETS=(v4_5block_7x7 nt8y_3x3stem)
HERE=${0:A:h}

for preset in $PRESETS; do
  for seed in $SEEDS; do
    file="$OUT/$preset-seed$seed.safetensors"
    "$BIN" --new-model --architecture "$preset" --init-seed "$seed" --out-model "$file" > /dev/null
    python3 "$HERE/safetensors_tensor_hash.py" "$file" | sed "s|^|$preset seed=$seed |"
  done
done
