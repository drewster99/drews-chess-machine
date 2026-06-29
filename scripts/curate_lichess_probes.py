#!/usr/bin/env python3
"""
Curate Lichess puzzles into a fixed probe set for Drew's Chess Machine.

Reads the Lichess puzzle CSV (one-time download from
https://database.lichess.org/lichess_db_puzzle.csv.zst), filters to quality
puzzles, samples deterministically per theme (optionally rating-stratified),
and emits a JSON file. Each entry stores the FEN of the position the solver
actually sees (i.e. with the Lichess "setup move" already applied) and the
target UCI move the network is expected to pick.

Two presets, selected with `--preset`:

  legacy  Reproduces the original bundled `lichess_probes_200.json` EXACTLY:
          8 themes, 25 each, a single 800-1800 rating band, sorted by puzzle
          id and taken evenly-spaced. This is the permanent legacy yardstick
          and MUST keep reproducing the same 200 puzzles on a given DB
          snapshot — do not change its config. Writes to lichess_probes_200.json.

  wide    The fixed wide yardstick (see GPU/measurement design notes): a broad
          theme set across a 400-3000 band, RATING-STRATIFIED within each theme
          (so per-theme rows are difficulty-comparable) with density shaped
          toward the middle and thin anchors in the tails. NLL is the intended
          spine, so saturated tail items still contribute graded signal. Writes
          to a SEPARATE file (lichess_probes_wide.json) — it never touches the
          legacy set; the two run in parallel so the 200-set history is never
          lost.

Sampling is deterministic in both: within each (theme[, rating-tier]) cell we
sort eligible puzzles by id and take N evenly-spaced, so re-running on the same
DB snapshot yields the same set. Cells with fewer than N eligible puzzles take
what's available and warn (expected for thin tail tiers / rare themes).
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# `chess` (python-chess) is imported lazily inside `validate_and_emit` — it's
# only needed when actually generating a set (which also needs the CSV), so
# `--help` and config introspection work without the dependency installed.

DEFAULT_CSV_PATH = Path("/tmp/lichess_db_puzzle.csv")


@dataclass(frozen=True)
class QualityFilters:
    """Per-puzzle gates so the rating is trustworthy and the puzzle stable."""
    min_popularity: int = 80      # Lichess "popularity" 0-100; vote-stable
    min_nb_plays: int = 200       # Played enough that the rating is meaningful
    max_rating_deviation: int = 90


@dataclass(frozen=True)
class Preset:
    name: str
    out_path: Path
    # Priority order: a puzzle is assigned to the FIRST bucket it qualifies
    # for, so tactical themes win over phase themes and a puzzle is never
    # double-counted across buckets.
    theme_priority: list[str]
    # Per-theme rating tiers as (rating_lo, rating_hi_inclusive, count). A
    # single full-band tier == "uniform random over the band" (the legacy
    # behavior). Multiple tiers == rating-stratified with per-tier density.
    rating_tiers: list[tuple[int, int, int]]
    filters: QualityFilters = field(default_factory=QualityFilters)
    # Optional per-theme tier overrides. A theme listed here uses its own tiers
    # instead of the shared `rating_tiers` — e.g. to sample mate themes more
    # densely where they're abundant (mates are plentiful below ~2000 but
    # essentially absent above ~2400, so a heavier quota only fills the low-mid
    # bands and naturally caps at the top).
    theme_tiers: dict[str, list[tuple[int, int, int]]] = field(default_factory=dict)

    def tiers_for(self, theme: str) -> list[tuple[int, int, int]]:
        return self.theme_tiers.get(theme, self.rating_tiers)

    @property
    def rating_min(self) -> int:
        allt = [self.rating_tiers] + list(self.theme_tiers.values())
        return min(lo for tiers in allt for lo, _, _ in tiers)

    @property
    def rating_max(self) -> int:
        allt = [self.rating_tiers] + list(self.theme_tiers.values())
        return max(hi for tiers in allt for _, hi, _ in tiers)


def uniform_tiers(lo: int, hi: int, width: int, count: int) -> list[tuple[int, int, int]]:
    """Contiguous [lo, hi] split into `width`-wide rating bins, `count` per
    theme each — a flat per-rating-point density (subject to DB availability)."""
    return [(b, min(b + width - 1, hi), count) for b in range(lo, hi + 1, width)]


# --- Legacy 200-set: DO NOT CHANGE. Reproduces the bundled set exactly. ---
LEGACY = Preset(
    name="legacy",
    out_path=Path("/tmp/lichess_probes_200.json"),
    theme_priority=[
        "mateIn1", "hangingPiece", "fork", "pin", "skewer",
        "endgame", "middlegame", "opening",
    ],
    rating_tiers=[(800, 1800, 25)],   # one band, 25/theme == the original
)

# --- Wide fixed yardstick: broad themes, ~uniform density across 550-2800,
# thin anchors in the tails. Equal count per 250-wide bin through 550-2800 gives
# flat puzzles-per-rating-point (the requested density), so the probe is equally
# sensitive across the whole expected strength trajectory; the 400-550 and
# 2800-3200 tails are thin (Bradley-Terry conditioning + competence-cliff
# markers). NLL is the spine, so even the tail items contribute graded signal.
# Tier counts are upper bounds — rare theme×bin cells (e.g. mateIn2 at the
# extremes) take fewer, so the realized total runs a bit under the bound.
WIDE = Preset(
    name="wide",
    out_path=Path("/tmp/lichess_probes_wide.json"),
    theme_priority=[
        "mateIn1", "mateIn2",
        "hangingPiece", "fork", "pin", "skewer",
        "discoveredAttack", "deflection", "sacrifice", "promotion",
        "endgame", "middlegame", "opening",
    ],
    # Uniform density per 100-rating bin across 550-2849 (13/theme each) so the
    # per-100 profile is flat — not just per-250 — with thin anchor tails outside.
    # Per theme: 23*13 + 2*8 = 315; * 13 themes = 4095 upper bound (≈4096 target,
    # realized under where sparse theme×bin cells under-fill — e.g. hard mates,
    # the rating extremes, and naturally thin sub-ranges).
    rating_tiers=(
        [(400, 549, 8)]                          # low anchor (thin)
        + uniform_tiers(550, 2849, 100, 13)      # dense band, flat per-100
        + [(2850, 3200, 8)]                      # high anchor (thin)
    ),
    # Mate themes sampled 2x denser (26/100-bin) — the net is weak at mates so a
    # richer battery measures that better. Mates are abundant below ~2000 and
    # essentially absent above ~2400, so this fills the low-mid bands and caps
    # itself at the top (no padding of nonexistent hard mates).
    theme_tiers={
        "mateIn1": (
            [(400, 549, 16)] + uniform_tiers(550, 2849, 100, 26) + [(2850, 3200, 16)]
        ),
        "mateIn2": (
            [(400, 549, 16)] + uniform_tiers(550, 2849, 100, 26) + [(2850, 3200, 16)]
        ),
    },
)

PRESETS = {p.name: p for p in (LEGACY, WIDE)}


def assign_bucket(themes_str: str, theme_priority: list[str]) -> str | None:
    """Pick the highest-priority theme the puzzle qualifies for, or None."""
    if not themes_str:
        return None
    themes = set(themes_str.split())
    for theme in theme_priority:
        if theme in themes:
            return theme
    return None


def collect(csv_path: Path, preset: Preset) -> dict[str, list[dict]]:
    """Scan the CSV once and bucket every eligible row by its theme."""
    buckets: dict[str, list[dict]] = {t: [] for t in preset.theme_priority}
    f = preset.filters
    rmin, rmax = preset.rating_min, preset.rating_max
    rows_seen = rows_eligible = 0

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows_seen += 1
            try:
                rating = int(row["Rating"])
                dev = int(row["RatingDeviation"])
                pop = int(row["Popularity"])
                plays = int(row["NbPlays"])
            except (KeyError, ValueError):
                continue

            if not (rmin <= rating <= rmax):
                continue
            if dev > f.max_rating_deviation:
                continue
            if pop < f.min_popularity:
                continue
            if plays < f.min_nb_plays:
                continue

            moves = row["Moves"].split()
            if len(moves) < 2:
                continue

            bucket = assign_bucket(row["Themes"], preset.theme_priority)
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
                    f"  scanned {rows_seen:>9,} rows  eligible {rows_eligible:>7,}",
                    file=sys.stderr,
                )

    print(f"scan done: {rows_seen:,} rows, {rows_eligible:,} eligible", file=sys.stderr)
    return buckets


def select(buckets: dict[str, list[dict]], preset: Preset) -> list[dict]:
    """Per theme, per rating-tier: sort by id, take `count` evenly-spaced."""
    selected: list[dict] = []
    for theme in preset.theme_priority:
        theme_rows = buckets[theme]
        theme_picked = 0
        for (lo, hi, count) in preset.tiers_for(theme):
            cell = sorted(
                (r for r in theme_rows if lo <= r["rating"] <= hi),
                key=lambda r: r["id"],
            )
            if len(cell) < count:
                print(
                    f"  WARNING: {theme} [{lo}-{hi}] has {len(cell)} eligible, "
                    f"wanted {count} — taking all",
                    file=sys.stderr,
                )
                picked = cell
            else:
                step = len(cell) / count
                picked = [cell[int(i * step)] for i in range(count)]
            for r in picked:
                r["theme_bucket"] = theme
            selected.extend(picked)
            theme_picked += len(picked)
        print(f"  {theme:<16} picked {theme_picked}", file=sys.stderr)
    return selected


def validate_and_emit(selected: list[dict], preset: Preset, db_snapshot: str) -> int:
    """Apply the setup move, verify legality, write the JSON file."""
    import chess  # lazy: only needed for generation

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

    f = preset.filters
    output = {
        "metadata": {
            "source": "Lichess puzzle DB (CC0) — database.lichess.org/lichess_db_puzzle.csv.zst",
            "db_snapshot": db_snapshot,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "preset": preset.name,
            "rating_range": [preset.rating_min, preset.rating_max],
            "rating_tiers": [list(t) for t in preset.rating_tiers],
            "theme_tiers": {
                theme: [list(t) for t in tiers]
                for theme, tiers in preset.theme_tiers.items()
            },
            "filters": {
                "min_popularity": f.min_popularity,
                "min_nb_plays": f.min_nb_plays,
                "max_rating_deviation": f.max_rating_deviation,
            },
            "themes": preset.theme_priority,
            "total": len(out_entries),
            "sampling": "deterministic per (theme, rating-tier): sort by puzzle id, take N evenly-spaced",
            "fen_convention": "position AFTER Lichess setup move; best_move_uci is the solver's first move",
        },
        "puzzles": out_entries,
    }

    preset.out_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {preset.out_path} ({len(out_entries)} puzzles)", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="legacy",
                    help="which set to build (default: legacy — the original 200)")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH,
                    help=f"path to lichess_db_puzzle.csv (default: {DEFAULT_CSV_PATH})")
    ap.add_argument("--out", type=Path, default=None,
                    help="override the preset's output path")
    ap.add_argument("--snapshot", type=str, default=None,
                    help="label for the Lichess DB snapshot this was built from "
                         "(e.g. '2026-06'); defaults to the CSV's modification date. "
                         "Recorded in the JSON metadata for provenance of the frozen set.")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    if args.out is not None:
        preset = Preset(
            name=preset.name, out_path=args.out, theme_priority=preset.theme_priority,
            rating_tiers=preset.rating_tiers, filters=preset.filters,
            theme_tiers=preset.theme_tiers,
        )

    if not args.csv.exists():
        print(f"missing {args.csv} — download lichess_db_puzzle.csv.zst and decompress it", file=sys.stderr)
        return 1

    db_snapshot = args.snapshot or datetime.fromtimestamp(
        args.csv.stat().st_mtime
    ).strftime("%Y-%m-%d (csv mtime)")

    print(f"preset={preset.name} band={preset.rating_min}-{preset.rating_max} "
          f"themes={len(preset.theme_priority)} tiers={len(preset.rating_tiers)} "
          f"snapshot={db_snapshot}", file=sys.stderr)
    buckets = collect(args.csv, preset)
    selected = select(buckets, preset)
    return validate_and_emit(selected, preset, db_snapshot)


if __name__ == "__main__":
    sys.exit(main())
