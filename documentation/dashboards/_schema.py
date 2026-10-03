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
          "wall_sec", "games_fed",
          # --- appended 2026-10-02 ------------------------------------------------
          #   probe_build  the app build that measured pElo / nll on this row
          #                (scripts/dcm_probe_build.py: bundle name + executable hash).
          #                Two builds can score one checkpoint differently, so values
          #                from different builds are not directly comparable. Blank on
          #                rows written before it was recorded, and on rows without pElo.
          "probe_build",
          # --- appended 2026-10-02 (lineage records, architecture format v7) -----
          #   train_step_sec  cumulative MEASURED trainer-step seconds behind the
          #                   checkpoint, from its lineage record. Idle, pauses and
          #                   sleep are excluded by construction, so it needs no
          #                   clamp; elapsed_train_sec (log-derived, sleep-clamped)
          #                   stays beside it for rows that predate records. Blank on
          #                   files without a record and where the record holds null.
          "train_step_sec"]

# The `note` a row carries when its checkpoint was probed and the probe measured a
# non-finite pElo (the probe then omits the key, and the row's pElo cell is blank).
# It is what tells such a row apart from one that was never probed: the trackers
# write it, and the experiment tables read it to show "non-finite" instead of a gap.
NON_FINITE_PELO_NOTE = "probe: pElo non-finite"
