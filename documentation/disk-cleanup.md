# Reclaiming disk space

A maintenance runbook for freeing space consumed by DrewsChessMachine's saved state. All app data lives under `~/Library/Application Support/DrewsChessMachine/`; session logs live separately under `~/Library/Logs/DrewsChessMachine/`.

## Where the space goes

`Sessions/` dominates. Each `.dcmsession` folder is ~5–12 GB, and roughly 99% of that is `replay_buffer.bin`. The actual trained weights inside a session (`champion.safetensors` and `trainer.safetensors`) total only ~6–17 MB — deleting a session throws away its warm replay buffer *and* its weight snapshot, so only delete lineages you don't need to resume or inspect.

Rough footprint on a full disk (2026-07-02): Sessions ~210 GB across ~30 folders, Logs ~6 GB, Corpora ~3 GB, everything else (Models, Analyses, Performance, SessionIndex) under ~350 MB combined.

## Why sessions accumulate without bound

`CheckpointPaths.prunePeriodicAutosaves` enforces the `maxPeriodicAutosavesKept` retention cap, but it **only prunes `-periodic.dcmsession` folders.** `manual` and `promote` saves are never auto-deleted. Over a long training campaign the `-promote` and `-manual` folders pile up unbounded and become the bulk of the disk usage.

## Two traps that make naive cleanup fail

1. **Time Machine local APFS snapshots pin "deleted" space.** After deleting session folders, free space (`df -h /System/Volumes/Data`) will not increase — the freed blocks stay referenced by the hourly `com.apple.TimeMachine.*.local` snapshots taken while those folders still existed. List them with `tmutil listlocalsnapshots /System/Volumes/Data`. To actually reclaim the space, thin the snapshots:

   ```
   tmutil thinlocalsnapshots /System/Volumes/Data <bytes> 4
   ```

   The `<bytes>` argument is how much to try to free (e.g. `150000000000` for ~150 GB); `4` is the most aggressive urgency. This deletes local Time Machine restore points only — real Time Machine backups to an external destination are untouched. macOS auto-thins local snapshots after ~24 h, but not fast enough when the disk is already full, so do it explicitly. This step is the one most easily forgotten: without it, a 100 GB deletion frees 0 GB and looks like it did nothing.

2. **`SessionIndex/` orphans are not cleaned up.** The app keeps a tiny (~760 B) per-session JSON in `SessionIndex/` as a metadata cache. Deleting a `.dcmsession` folder does not remove its index entry, so orphaned entries accumulate. The app tolerates orphans (the session picker copes), so cleaning them is optional; if you want the index tidy, delete the matching `SessionIndex/*.json` alongside each removed folder.

## A safe keeper-selection policy

The policy the maintainer has endorsed: keep the **latest-by-timestamp** session of each lineage, plus any deliberately named milestone folders, plus the current resume target.

- **Lineage** = the 4-character tag in a session's ModelID (`yyyymmdd-N-XXXX`), e.g. `t9sX`. In practice the latest-by-timestamp folder in a lineage is also the one with the highest `trainingSteps`.
- **Named milestones** are folders a human renamed on purpose (e.g. `old-Ko63-try-to-resume-from-here`, `last-before-big-changeup-*`). Keep them regardless.
- **Resume target** is the `LastSessionPointer`. Decode it with:

  ```
  defaults export com.drewben.DrewsChessMachine -
  ```

  and read the JSON under key `DrewsChessMachine.LastSessionPointer.v1` (its `directoryPath` field). The app's defaults domain is `com.drewben.DrewsChessMachine`.

Per-session metadata for the decision comes from each folder's `manifest.json` (newer sessions) or, for older pre-manifest sessions, the `trainingSteps` key inside `session.json`.

Any script that deletes should **guard** before removing anything: assert that the resume target, every named milestone, and the latest-per-lineage folder are all in the keep set (not the delete set) before calling `rm -r` / `shutil.rmtree`. Never use `rm -rf`.

## Logs

`~/Library/Logs/DrewsChessMachine/` holds one plain-text `dcm_log_*.txt` per launch. These are safe to delete individually; keep the most recent ones for debugging context. They are independent of the `Sessions/` cleanup above.
