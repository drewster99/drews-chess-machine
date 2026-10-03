#!/bin/zsh
# End-to-end exact-resume check through the shipped binary (determinism plan
# C6 step 7). Runs a corpus replay N+M steps straight through, then the same
# replay N steps, then `--resume-exact` from that save for M more, and compares
# the two final files with resume_equivalence_compare.py (tensor hash, stream
# positions, feed position, totals, and the resumed segment's exact flag).
#
# It trains on the GPU: run it when nothing else is training. Every run uses the
# given start model as a new branch, the given parameters file and seed, and
# writes only into the scratch folder, which must not exist yet.
#
# Usage:
#   scripts/resume_equivalence.sh [--dry-run] <DrewsChessMachine binary> <corpus id or folder> \
#       <start model .safetensors> <parameters.json> <N> <M> <seed> <scratch folder>
#
# --dry-run prints the three commands and the comparison without running them.
set -eu

SCRIPT=$0
usage() {
  sed -n '2,16p' "$SCRIPT" | sed 's/^# \{0,1\}//'
}

DRY_RUN=0
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then usage; exit 0; fi
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; shift; fi
if [ $# -ne 8 ]; then usage >&2; exit 2; fi

BIN=$1 CORPUS=$2 START=$3 PARAMS=$4 N=$5 M=$6 SEED=$7 OUT=$8
HERE=${0:A:h}

for value in "$N" "$M"; do
  [[ "$value" == <1-> ]] || { echo "error: step counts must be positive whole numbers, got '$value'" >&2; exit 2; }
done
[[ "$SEED" == <0-> ]] || { echo "error: the seed must be a whole number, got '$SEED'" >&2; exit 2; }
if [ $DRY_RUN -eq 0 ]; then
  [ -x "$BIN" ] || { echo "error: $BIN is not executable" >&2; exit 2; }
  [ -f "$START" ] || { echo "error: start model $START does not exist" >&2; exit 2; }
  [ -f "$PARAMS" ] || { echo "error: parameters file $PARAMS does not exist" >&2; exit 2; }
  [ -e "$OUT" ] && { echo "error: $OUT already exists; give a new folder" >&2; exit 2; }
  mkdir -p "$OUT"
fi

STRAIGHT="$OUT/straight.safetensors"
FIRST="$OUT/first.safetensors"
SECOND="$OUT/second.safetensors"
TOTAL=$(( N + M ))

COMMON=(--replay-corpus "$CORPUS" --parameters "$PARAMS" --seed "$SEED")
RUN_STRAIGHT=("$BIN" "${COMMON[@]}" --start-model "$START" --training-step-limit "$TOTAL" --out-model "$STRAIGHT")
RUN_FIRST=("$BIN" "${COMMON[@]}" --start-model "$START" --training-step-limit "$N" --out-model "$FIRST")
RUN_SECOND=("$BIN" "${COMMON[@]}" --start-model "$FIRST" --resume-exact --training-step-limit "$M" --out-model "$SECOND")
COMPARE=(python3 "$HERE/resume_equivalence_compare.py" "$STRAIGHT" "$SECOND")

if [ $DRY_RUN -eq 1 ]; then
  print -r -- "${(q)RUN_STRAIGHT[@]}"
  print -r -- "${(q)RUN_FIRST[@]}"
  print -r -- "${(q)RUN_SECOND[@]}"
  print -r -- "${(q)COMPARE[@]}"
  exit 0
fi

echo "== straight: $TOTAL steps"
"${RUN_STRAIGHT[@]}" > "$OUT/straight.log" 2>&1
echo "== first segment: $N steps"
"${RUN_FIRST[@]}" > "$OUT/first.log" 2>&1
echo "== resumed segment: $M steps (--resume-exact)"
"${RUN_SECOND[@]}" > "$OUT/second.log" 2>&1
grep -E '^\[RESUME\]' "$OUT/second.log" || true
"${COMPARE[@]}"
