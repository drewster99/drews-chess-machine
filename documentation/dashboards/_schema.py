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
          "frozen_file", "note"]
