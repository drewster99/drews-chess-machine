#!/bin/sh
# Reproduce the rating analysis for the 2026-07-08 38-engine arena.
#
# Requirements:
#   - ordo 1.2.6            (https://github.com/michiguel/Ordo; built at
#                            /Users/andrew/checkouts/Ordo/ordo on the original host)
#   - gzip, python3
#
# The games themselves are NOT reproducible bit-for-bit: several engines used
# live "-replay-latest" weights that the trainer was overwriting during the run
# (see README "Caveats"). This script reproduces the ANALYSIS from the archived
# PGN.
set -eu
DIR=$(cd "$(dirname "$0")" && pwd)
ORDO=${ORDO:-/Users/andrew/checkouts/Ordo/ordo}

# 1. Decompress the raw PGN (all 70,638 games, incl. the 100 warmup tests).
gzip -dkf "$DIR/games_full.pgn.gz"   # -> games_full.pgn (keeps the .gz)

# 2. Strip the 100 stray leading warmup games → real tournament only.
python3 "$DIR/slice_tests.py" "$DIR/games_full.pgn" "$DIR/games_tournament.pgn"

# 3. Ratings + head-to-head crosstable.
#    -a 1320 -A Stockfish : pin Stockfish to 1320 (its UCI_Elo) as the scale anchor
#    -V                   : error bars relative to the pool average (within-pool honest)
#    -s 100               : 100 simulations for the error columns
#    -J -j h2h.txt        : add CFS columns + write the head-to-head file
#    -G                   : proceed despite the "isolated group" note (Sloppy went 100%)
"$ORDO" -q -p "$DIR/games_tournament.pgn" \
        -a 1320 -A Stockfish -V -s 100 -J \
        -j "$DIR/h2h.txt" -o "$DIR/ratings.txt" -G

echo "wrote ratings.txt and h2h.txt"
