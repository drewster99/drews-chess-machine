find-tools (partial, truncated after #6):
1 MED-HIGH: dcm_lineage.py:428-431 / selfplay.py:395-400 derive_runs raises "files disagree on model_id" for every GUI v7 session (champion vs trainer modelIDs differ, same lineage record; CheckpointManager:1430-1450). Untested (SessionFolderTests only vsuci).
2 MED: tick.py:29,32 / replay.py:799,928: probe() now raises ProbeFailure; track() uncaught -> one unprobeable ckpt freezes dashboard every tick (re-probes up to 300s holding lock).
3 MED: replay.py:835-843 _backfill_one: internals_cells/lineage_cells exceptions escape before write_csv -> whole pass's probes lost; legacy -frozen segment guess wrong at boundaries raises.
4 MED-LOW: table_common.py:27 csv_points drops rows with pElo "" (non-finite) -> shown as never-reached, NLL lost; probe arms show "non-finite"; review.py:118-120 asymmetric.
5 MED-LOW: replay.py:586 _lineage_scan discards unrecorded list and errors -> v7 read failures skipped silently.
6 MED-LOW: init_reproducibility.sh:14,30 set -eu without pipefail -> can report match when hashing failed [truncated]
6 (full) MED-LOW: init_reproducibility.sh:14,30 pipeline exit is sed's -> failed hash lines silently missing; diff can falsely match. Fix pipefail.
7 LOW-MED: probe_loop.sh:59-63,68-77,82,106 TRAINER_PID mismatch -> loop exits after one pass while trainer runs; startup wait doesn't check it.
8 LOW: probe_loop.sh:91-93 ignores probe exit status (rc only logged); replay.py probe() rejects nonzero.
9 LOW: dcm_arch.read_metadata (:103-116) unbounded header read vs dcm_lineage MAX_HEADER_BYTES (:87-88); damaged prefix -> MemoryError escapes callers; duplicate reader.
10 LOW: dcm_lineage _REQUIRED["run"] (:118-119) lacks not_exact_items; sessions_summary.py:97 KeyError kills listing.
11 LOW: se-fc1-leaky/review.py:146-152 identifies checkpoints by filename only; hard-coded range(3); probe() raises on non-finite.
12 LOW: selfplay_probe_append.py:14-21,95-103 doesn't refuse/repair already-torn last line.
13 LOW: noSE-noReZero/review.py:121-126 ZeroDivisionError when no shared step.
