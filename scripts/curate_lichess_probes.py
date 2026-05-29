#!/usr/bin/env python3
"""
Curate 200 Lichess puzzles for the Drew's Chess Machine tactical-probe set.

Reads the Lichess puzzle CSV (one-time download from
https://database.lichess.org/lichess_db_puzzle.csv.zst), filters to a
rating band, samples 25 puzzles per theme across 8 themes, and emits a
deterministic JSON file. Each entry stores the FEN of the position the
solver actually sees (i.e. with the Lichess "setup move" already applied)
and the target UCI move the network is expected to pick.

Sampling is deterministic: we collect every eligible puzzle into its
theme bucket, sort by puzzle id, and take 25 evenly-spaced slices so
re-running on the same DB snapshot yields the same set.
"""

import csv
import json
import sys
from pathlib import Path

import chess

CSV_PATH = Path("/tmp/lichess_db_puzzle.csv")
OUT_PATH = Path("/tmp/lichess_probes_200.json")

# Priority order: a puzzle is assigned to the FIRST bucket it qualifies
# for. Tactical themes win over phase themes so we don't dilute the
# tactical buckets with phase-only positions.
THEMES_IN_PRIORITY = [
    "mateIn1",
    "hangingPiece",
    "fork",
    "pin",
    "skewer",
    "endgame",
    "middlegame",
    "opening",
]
PER_THEME = 25

RATING_MIN = 800
RATING_MAX = 1800
MIN_POPULARITY = 80      # Lichess "popularity" 0-100; vote-stable
MIN_NB_PLAYS = 200       # Played enough that the rating is meaningful
MAX_RATING_DEV = 90


def assign_bucket(themes_str: str) -> str | None:
    """Pick the highest-priority theme the puzzle qualifies for, or None."""
    if not themes_str:
        return None
    themes = set(themes_str.split())
    for theme in THEMES_IN_PRIORITY:
        if theme in themes:
            return theme
    return None


def main() -> int:
    if not CSV_PATH.exists():
        print(f"missing {CSV_PATH}", file=sys.stderr)
        return 1

    # Collect every eligible row, bucketed.
    buckets: dict[str, list[dict]] = {t: [] for t in THEMES_IN_PRIORITY}
    rows_seen = 0
    rows_eligible = 0

    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_seen += 1
            try:
                rating = int(row["Rating"])
                dev = int(row["RatingDeviation"])
                pop = int(row["Popularity"])
                plays = int(row["NbPlays"])
            except (KeyError, ValueError):
                continue

            if not (RATING_MIN <= rating <= RATING_MAX):
                continue
            if dev > MAX_RATING_DEV:
                continue
            if pop < MIN_POPULARITY:
                continue
            if plays < MIN_NB_PLAYS:
                continue

            moves = row["Moves"].split()
            if len(moves) < 2:
                continue

            bucket = assign_bucket(row["Themes"])
            if bucket is None:
                continue

            rows_eligible += 1
            buckets[bucket].append({
                "id": row["PuzzleId"],
                "fen_start": row["FEN"],
                "moves": moves,
                "rating": rating,
                "popularity": pop,
                "nb_plays": plays,
                "themes": row["Themes"],
            })

            if rows_seen % 500_000 == 0:
                print(
                    f"  scanned {rows_seen:>8,} rows  "
                    f"eligible {rows_eligible:>6,}  "
                    f"per-bucket "
                    + " ".join(
                        f"{t[:5]}={len(buckets[t])}"
                        for t in THEMES_IN_PRIORITY
                    ),
                    file=sys.stderr,
                )

    print(
        f"scan done: {rows_seen:,} rows, {rows_eligible:,} eligible",
        file=sys.stderr,
    )

    # Deterministic sample: sort each bucket by puzzle id and take 25
    # evenly-spaced slices. Even spacing is preferred over the first 25
    # so the sample doesn't cluster on a single Lichess-id epoch.
    selected: list[dict] = []
    for theme in THEMES_IN_PRIORITY:
        rows = sorted(buckets[theme], key=lambda r: r["id"])
        if len(rows) < PER_THEME:
            print(
                f"WARNING: bucket '{theme}' has only {len(rows)} candidates, "
                f"need {PER_THEME}",
                file=sys.stderr,
            )
            picked = rows
        else:
            step = len(rows) / PER_THEME
            picked = [rows[int(i * step)] for i in range(PER_THEME)]

        for r in picked:
            r["theme_bucket"] = theme

        selected.extend(picked)
        print(
            f"  {theme:<14} {len(rows):>6} eligible -> picked {len(picked)}",
            file=sys.stderr,
        )

    # For each pick, apply moves[0] to the FEN to get the position the
    # network actually sees, and check that moves[1] is legal there.
    # Drop any puzzle that fails — we should never see this for well-
    # formed Lichess data but the check is cheap.
    out_entries: list[dict] = []
    for r in selected:
        board = chess.Board(r["fen_start"])
        try:
            setup = chess.Move.from_uci(r["moves"][0])
            if setup not in board.legal_moves:
                print(f"setup illegal for {r['id']}", file=sys.stderr)
                continue
            board.push(setup)

            target = chess.Move.from_uci(r["moves"][1])
            if target not in board.legal_moves:
                print(f"target illegal for {r['id']}", file=sys.stderr)
                continue
        except Exception as e:
            print(f"parse error for {r['id']}: {e}", file=sys.stderr)
            continue

        out_entries.append({
            "id": r["id"],
            "theme": r["theme_bucket"],
            "rating": r["rating"],
            "popularity": r["popularity"],
            "nb_plays": r["nb_plays"],
            "fen": board.fen(),
            "best_move_uci": r["moves"][1],
            "all_themes": r["themes"],
        })

    out_entries.sort(key=lambda e: (e["theme"], e["id"]))

    output = {
        "metadata": {
            "source": "Lichess puzzle DB (CC0) — database.lichess.org/lichess_db_puzzle.csv.zst",
            "rating_range": [RATING_MIN, RATING_MAX],
            "filters": {
                "min_popularity": MIN_POPULARITY,
                "min_nb_plays": MIN_NB_PLAYS,
                "max_rating_deviation": MAX_RATING_DEV,
            },
            "themes": THEMES_IN_PRIORITY,
            "per_theme": PER_THEME,
            "total": len(out_entries),
            "sampling": "deterministic: sort eligible by puzzle id, take 25 evenly-spaced",
            "fen_convention": "position AFTER Lichess setup move; best_move_uci is the solver's first move",
        },
        "puzzles": out_entries,
    }

    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"wrote {OUT_PATH} ({len(out_entries)} puzzles)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
