#!/bin/bash
# Stream Lichess's October 2026 standard rated games through the elite filter;
# only the filtered PGN touches disk (the 29 GB download never does).
set -o pipefail
MONTH=${MONTH:-2026-10}
OUT=${OUT:-$HOME/Dropbox/corpora_sources/elite_heldout_$MONTH}
mkdir -p "$OUT"
curl -sSfL --retry 5 "https://database.lichess.org/standard/lichess_db_standard_rated_$MONTH.pgn.zst" \
  | zstdcat \
  | python3 "$(dirname "$0")/filter_elite.py" > "$OUT/lichess_elite_$MONTH.pgn.partial" 2> "$OUT/filter.log"
status=("${PIPESTATUS[@]}")
echo "pipe status: ${status[*]}" >> "$OUT/filter.log"
if [ "${status[*]}" = "0 0 0" ]; then mv "$OUT/lichess_elite_$MONTH.pgn.partial" "$OUT/lichess_elite_$MONTH.pgn"; else exit 1; fi
