"""Shared CSV column schema for the DCM dashboard trackers.

Both replay.py (corpus-replay runs) and selfplay.py (self-play runs) write the
SAME per-1000-step CSV layout so master.py can render them on one set of charts.
Keeping the column order in exactly one place stops the two trackers from
drifting (a mismatched column would silently misalign every downstream reader).

This module has NO import-time side effects (no registry load, no numpy) so it
is safe to import from either tracker without pulling in the other's heavyweight
startup.
"""

FIELDS = ["cum_step", "meta_step", "segment", "elapsed_train_sec", "wallclock_iso",
          "ms_per_step", "pElo", "nll", "loss", "pLoss", "vLoss", "legalMass", "pIllM",
          "bn1Mean", "gNorm", "sae2", "eff_alpha", "pLogit_mean", "pLogit_peak",
          "frozen_file", "note",
          # --- appended 2026-08-11: the second and third chart axes ---------------
          # A run's history is read on three axes, and conflating them hides real
          # effects. `cum_step` is the work axis only while batch size and replay
          # ratio hold constant; `elapsed_train_sec` is device-specific and is
          # additionally sleep-clamped (see _clamped_timeline); so neither one alone
          # can answer "how much has this network actually learned from?"
          #
          #   wall_sec   raw cumulative wall-clock training seconds, UNCLAMPED.
          #              Sits beside elapsed_train_sec so the clamp's effect is
          #              (wall_sec - elapsed_train_sec) instead of being invisible.
          #   games_fed  cumulative corpus games consumed across the whole lineage.
          #              The device-independent compute axis. Measured from the
          #              `games=` field, never modeled -- it stays correct even if
          #              batch size or replay ratio change, which `cum_step` does not.
          #
          # Appended rather than inserted so existing CSVs keep parsing: write_csv
          # fills missing keys with "" and every reader uses csv.DictReader.
          "wall_sec", "games_fed"]
