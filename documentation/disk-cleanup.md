# Reclaiming disk space

A maintenance runbook for freeing space consumed by DrewsChessMachine's saved state. All app data lives under `~/Library/Application Support/DrewsChessMachine/`; session logs live separately under `~/Library/Logs/DrewsChessMachine/`.

## Where the space goes

`Sessions/` dominates. Each `.dcmsession` folder is ~5–12 GB, and roughly 99% of that is `replay_buffer.bin`. The actual trained weights inside a session (`champion.safetensors` and `trainer.safetensors`) total only ~6–17 MB — deleting a session throws away its warm replay buffer *and* its weight snapshot, so only delete lineages you don't need to resume or inspect.

Rough footprint on a full disk (2026-07-02): Sessions ~210 GB across ~30 folders, Logs ~6 GB, Corpora ~3 GB, everything else (Models, Analyses, Performance, SessionIndex) under ~350 MB combined.

## What the app prunes on its own, and what accumulates

**Today the app prunes nothing.** Automatic-save pruning is off by default — the `automaticSavePruningEnabled` parameter (`automatic_save_pruning_enabled`, default `false`; the Sessions tab's "Prune old autosaves" toggle) gates it — and the current build also **forces it off** regardless of that setting: `CheckpointPaths.automaticSavePruningForcedOff` is `true` by owner decision 2026-10-01, pending D-8 (saves without the replay buffer by default) and more confidence in deleting saves automatically. Every periodic or promotion save logs one `[PRUNE] skipped after <folder>: off: <reason> (automatic_save_pruning_enabled=… max_periodic_autosaves_kept=…)` line, and each Play-and-Train start logs `[PRUNE] automatic-save pruning at Play-and-Train start: …`. Until a build lifts that switch, every `.dcmsession` folder accumulates and the manual cleanup below is the only way space comes back.

The retention rule below is what pruning does when it is allowed to run. Since 2026-10-01 (plan #8, phase P14) `CheckpointPaths.pruneAutomaticSaves` keeps **one global pool** of automatic saves capped by the `maxPeriodicAutosavesKept` parameter (`max_periodic_autosaves_kept`, default 3; 0 = unlimited, nothing pruned). The pool is every `-periodic.dcmsession` and `-promote.dcmsession` folder in `Sessions/`, from any session — arena post-promotion and Train ▸ Promote Trainee Now saves both carry the `promote` tag and are treated alike — ranked newest first by the folder name's leading UTC timestamp. The sweep runs after every successful periodic or promotion save and deletes every pool member beyond the newest `N`, except the save just written and the current resume-pointer target.

A folder is only counted (and only ever deleted) when its name is exactly `<YYYYMMDD-HHMMSS>-<sessionID>-(periodic|promote).dcmsession` with a minted session ID (`yyyymmdd-N-XXXX`), it is a real directory rather than a symbolic link, and its `session.json` names that same session ID. Renamed folders (named milestones) therefore never match, and a folder with a placeholder ID (`unknown`, `unknown-session`), a missing or mismatched `session.json`, or a symlink is kept and logged with the reason on every sweep. Every sweep logs a `[PRUNE] retention: cap=N periodic=a promote=b kept=… protected=… unverified=… deleting=…` line, and one `[PRUNE]` line per removal, in the session log.

What still accumulates without bound, and is what this runbook is for:

- **Everything, while pruning is off** — the default, and the only state of a build with the kill switch set.
- **`-manual` and `-sigusr2` saves** — deliberate "keep this" saves; the app never deletes them.
- **Unverified automatic-save folders** (placeholder IDs, damaged folders) — kept on purpose so a person can look at them first.
- **Renamed milestone folders.**
- Everything when the cap is `0`.

With pruning running, lowering the cap — or turning pruning on — deletes the excess at the next periodic or promotion save, across every session at once — check that nothing you want is among the older automatic saves before doing either, or rename the folders you want to keep (a renamed folder is never a pool member).

### Before 2026-10-01

`CheckpointPaths.prunePeriodicAutosaves` (the last committed version before the global pool) enforced the `maxPeriodicAutosavesKept` retention cap over **`-periodic.dcmsession` folders only — from every session, selected by name suffix alone**. After each successful periodic save (and only then), when the cap was above `0`, it listed `Sessions/`, took every non-hidden entry whose name ended in `-periodic.dcmsession` — whatever it was (folder, file or symbolic link), with no `session.json` check and no check of the session ID in the name — ranked them newest first by name, and deleted every one beyond the newest `N` by path (`FileManager.removeItem`, recursive for a folder), sparing only the save just written. The resume-pointer target was **not** protected: an older periodic save that the pointer still named could be deleted. `-manual`, `-promote` and `-sigusr2` saves were never auto-deleted. Over a long training campaign the `-promote` and `-manual` folders piled up unbounded and became the bulk of the disk usage; disks filled under that rule may still hold such a backlog of `-promote` folders (and, wherever the cap was `0`, of `-periodic` ones too), which the first sweep of a build with the global pool removes down to the cap once pruning is allowed to run (it is off by default and forced off in the current build — see above).

### Hidden staging leftovers

Several writers — Save as Preset, `--create-parameters-file`, a corpus's `corpus.json`, the corpus-replay / train-vs-UCI rolling and step-enumerated checkpoints, and the `--new-model` / `--derive-model` outputs — stage a file first under a hidden, per-write unique sibling name, `.<file name>.<UUID>.tmp` beside it, and rename it into place only when complete (`FileSafety.temporarySibling(of:)`). A process killed between those two steps leaves the hidden staging file behind; removing a folder (pruning, a failed save's cleanup) likewise first renames it to a hidden name of the same shape, so a process killed mid-removal can leave a partly emptied hidden folder. Nothing can tell such a leftover from another process's write in flight except its age, so:

- **`Models/` and `Sessions/`:** the launch sweep (`CheckpointPaths.cleanupOrphans`, at every GUI launch) removes hidden staging *files* of exactly that shape once they are older than `CheckpointPaths.orphanStagingMinimumAge`, identity-checked, logging `[CLEANUP] Removed orphan …`. A *folder* of that shape is kept and logged (`[CLEANUP] Kept …: is not a regular file …`): it is an interrupted removal of something the app had already decided to delete, and is safe to delete by hand once no app instance is running.
- **Corpus folders:** `--validate-corpus` reports any `.tmp` entry as `stray-temp-file` and does not remove it.
- **Everywhere else** (`Presets/`, a CLI's output folder such as a `--out-model` outside `Models/`, a `--create-parameters-file` target): leftovers stay until removed by hand. They are hidden, so list with `ls -a`.

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
