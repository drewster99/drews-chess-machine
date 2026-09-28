# Autosave retention + weights-only saves

Status: planned, not started. Supersedes/completes the "Autosave retention
pruning" entry in `ROADMAP.md` (that entry stays as design history; this file
is the actionable plan).

## Problem

For a training run left going for weeks unattended, disk usage is currently
unbounded in the case that matters most:

- `PeriodicAutosaveIntervalSec` (default 4h) and `MaxPeriodicAutosavesKept`
  (default 3) are both already live-tunable from the UI (Settings ▸ Sessions
  tab) — **this part already works today**, contrary to the informal
  assumption that prompted this plan.
- `CheckpointPaths.prunePeriodicAutosaves` (`Persistence/CheckpointManager.swift:189`)
  only ever touches folders whose name ends `-periodic.dcmsession`. Every
  `-promote.dcmsession` folder — written both by the arena's automatic
  post-promotion autosave (`SessionController+Arena.swift:530-651`) and by
  the user-invoked "Promote Trainee Now" autosave (`SessionController+ManualPromote.swift:204-211`,
  which reuses the `promote` disk tag) — is **never pruned**. Manual
  (`-manual`) and signal (`-sigusr2`) saves are also never pruned, by design.
- Every save of any trigger writes the *entire* live replay buffer
  (`replay_buffer.bin`) if one exists — there is no way to opt out today.
  Per `documentation/disk-cleanup.md`, that file is ~99% of a `.dcmsession`'s
  5–12 GB footprint; the two weight files together are only ~6–17 MB.
- On a run with frequent promotions (e.g. `arenaAutoSec=900` → up to ~96
  arena attempts/day, each promotion writing a full buffer), the unpruned
  `-promote` pool is the actual multi-hundred-GB/week risk, not periodic
  saves.

## Goals (from user conversation, 2026-09-27)

1. Bound total disk usage indefinitely for a weeks-long run, with no manual
   intervention required.
2. Never lose more than one save interval's progress on an interruption —
   already achievable today by lowering `PeriodicAutosaveIntervalSec`, and
   preserved by this plan (the most recent *full*, resumable save is always
   protected from pruning).
3. Keep a longer trail of **cheap, weights-only** snapshots (default: last 72
   hours) for "how does the model now compare to N hours ago" analysis,
   without paying the replay-buffer cost for all of them.
4. Keep the replay buffer only in the last few (default: 3) most recent
   automatic saves, so a resume-after-interruption always has a warm buffer
   within one save interval.
5. Manual saves (`File ▸ Save Session`) and the SIGUSR2 pre-shutdown safety
   save stay exempt from all pruning — an explicit save is a deliberate
   "keep this," matching the existing design intent already stated in
   `ROADMAP.md`.
6. Add a genuine user-facing "save without the replay buffer" option, not
   just internal plumbing.

## Design

### Phase 1 — Weights-only save capability

This is the foundation both the user-facing feature and Phase 2's
space-efficient retention build on.

**Key existing seam** (`Persistence/CheckpointManager.swift:666`):

```swift
let wantsReplayBuffer = replayBuffer != nil && state.hasReplayBuffer == true
```

`saveSession` already only writes `replay_buffer.bin` when *both* a live
buffer was passed in *and* `state.hasReplayBuffer` is true. So the write path
itself needs **no change** — a caller can already opt out by passing
`replayBuffer: nil`. The only gap is that `state.hasReplayBuffer` is
currently computed purely from "does a live buffer object exist"
(`SessionController+Checkpoint.swift:889`, `hasReplayBuffer: bufferSnap != nil`)
— independent of whether *this* save intends to write it. If a caller passed
`replayBuffer: nil` today without also correcting this field, `session.json`
would claim `hasReplayBuffer: true` while no `replay_buffer.bin` exists on
disk — a real correctness bug for any future reader of that field (the
resume loader, the disk-cleanup keeper-selection policy, a "Manage
Autosaves" UI).

**Changes:**

1. Add `includeReplayBuffer: Bool = true` parameter to:
   - `SessionController.saveSessionInternal(...)` (`SessionController+Checkpoint.swift:264`)
   - The arena's inline post-promotion save block (`SessionController+Arena.swift:530-651`)
2. In both, when `includeReplayBuffer == false`:
   - Pass `replayBuffer: nil` to `CheckpointManager.saveSession`.
   - Force `state.hasReplayBuffer = false` via a new builder method
     `SessionCheckpointState.withReplayBufferOmitted()`, matching the
     existing `withTrainingSegments(_:)` / chart-data builder pattern
     already in `SessionCheckpointFile.swift` (keeps the memberwise init
     lean, per the existing code comments explaining why that pattern was
     adopted).
   - Leave `replayBufferStoredCount` / `replayBufferCapacity` /
     `replayBufferTotalPositionsAdded` populated as informational metadata
     ("the buffer had this many positions at save time, even though this
     particular save didn't persist it") — only `hasReplayBuffer` changes
     meaning to "is the file actually present."
3. Add a new File menu item, "Save Session (Weights Only)", calling
   `saveSessionInternal(..., trigger: .manual, includeReplayBuffer: false)`.
   Filename keeps the existing `-manual.dcmsession` suffix (no new disk tag)
   — the save's near-instant duration and small folder size, plus
   `session.json`'s `hasReplayBuffer: false`, are the distinguishing signal.
   Status-bar / log text should say "(weights only)" explicitly so it's
   never confused with a full manual save at a glance.
4. Resume-loading a `hasReplayBuffer: false` session already works — it's
   the same code path pre-2026-06-24 sessions (saved before the replay
   buffer was ever persisted) already exercise: the loader starts with an
   empty buffer. No loader changes needed; this is existing back-compat
   behavior, not new.

### Phase 2 — Extend automatic retention to promotion saves + add time-based pruning

**New parameters** (both `@TrainingParameter`, category `"Sessions"`,
`liveTunable: true`, in `TrainingParameters.swift` next to the existing two):

- `AutosaveWeightsRetentionHours` — default **72.0**, range **0...8760** (0 =
  keep forever). How long, from a save's own filename timestamp, an
  automatic (`-periodic` or `-promote`) save folder survives before the
  *entire* folder is deleted.
- Reuse and **widen the scope of** `MaxPeriodicAutosavesKept` (rename its
  description, not its `id`/property — this is a scope change to an
  existing knob the user is asking to extend, not a rename, so no
  UserDefaults-key churn) to mean: how many of the newest automatic saves
  (across **both** `-periodic` and `-promote` pools, combined and sorted by
  timestamp) keep their replay buffer. Default stays 3.

  **Critical edge-case, must not get inverted:** today, `0` means
  "unlimited — no pruning at all" (`prunePeriodicAutosaves` returns
  immediately on `keep <= 0`). The widened parameter **must preserve that
  exact meaning** — `0` disables *all* of Phase 2's automatic-pool logic
  (no stripping, no deletion; every automatic save keeps its buffer
  forever, matching today's behavior when the knob was set to unlimited).
  `0` must **not** be reinterpreted as "keep zero full-buffer saves" (which
  would strip/delete a just-written save's buffer immediately and could
  break goal 2 the moment someone sets it, expecting the old "unlimited"
  meaning). `enforceAutosaveRetention` should short-circuit exactly like
  `prunePeriodicAutosaves` does today: `guard fullRetentionCount > 0 else
  { return }` before touching anything.

**New retention function** in `CheckpointManager.swift`, replacing
`prunePeriodicAutosaves`'s current call sites (keep the old function or fold
its logic in — implementer's call, but the *combined* pool behavior must
replace today's periodic-only behavior):

```
enforceAutosaveRetention(fullRetentionCount: Int, weightsRetentionHours: Double, protecting: URL?)
```

Algorithm, run lazily off the main actor after every successful **periodic**
or **promote**-tagged save (mirroring the existing
`Task.detached(priority: .utility)` call site pattern):

1. List `Sessions/`, filter to folders whose name ends `-periodic.dcmsession`
   or `-promote.dcmsession`. `-manual` and `-sigusr2` are excluded by
   construction, same as today.
2. Sort newest-first by filename (already chronological, per
   `CheckpointPaths.makeSessionDirectoryName`).
3. The first `fullRetentionCount` folders: **untouched** (buffer, if
   present, stays).
4. Remaining folders whose age (now − filename timestamp) is **within**
   `weightsRetentionHours`: if `replay_buffer.bin` exists, **delete just that
   file** and rewrite `session.json` with `hasReplayBuffer: false` (small,
   already-JSON file — cheap rewrite). This "demotes" an old full save to a
   weights-only one in place, satisfying goal 3 (cheap trail for
   comparison) without the goal-4 buffer retention window growing
   unboundedly. **Verified low-risk:** `CheckpointManager.loadSession`
   (`CheckpointManager.swift:1049`) already computes
   `bufferPresent = (state.hasReplayBuffer == true) &&
   FileManager.default.fileExists(atPath: bufferURL.path)` — it requires
   *both* the flag and the file, so even if the rewrite step were skipped
   or failed, a stripped folder could never crash the loader (it would just
   silently load as "no buffer," same as any pre-2026-06-24 session). The
   `session.json` rewrite is still worth doing for metadata accuracy (a
   future "Manage Autosaves" UI, or `disk-cleanup.md`'s keeper policy,
   should be able to trust the flag), not because correctness depends on
   it.
5. Remaining folders older than `weightsRetentionHours`: delete the entire
   directory.
6. `protecting` (the just-written save URL) is exempt from every step above,
   belt-and-suspenders, same as today's `protecting:` parameter. Also
   cross-check against the current `LastSessionPointer` target and exempt it
   too, even if it wasn't the URL this specific call just wrote (covers the
   case where `LastSessionPointer` was set by a save this process didn't
   just make — e.g. after a session was resumed and a later save from a
   *different* run pointer is being pruned).
7. Per-item failures log `[PRUNE-ERR]` and don't abort the sweep, same as
   today.

**Why "always write the buffer, then strip after" instead of "skip the
write up front" for automatic saves beyond the keep window:** the MVP writes
every automatic save's buffer in full (unchanged from today), then Phase 2's
retention pass deletes/strips it moments later if it falls outside the
window. This is simpler and lower-risk — the write path
(`CheckpointManager.saveSession`) needs zero changes for this phase, only
Phase 1's already-planned `includeReplayBuffer` plumbing is new code. The
tradeoff is wasted disk I/O for buffers written and then immediately
stripped on a high promotion-frequency run. **This is an accepted, explicit
MVP tradeoff, not an oversight** — if it proves to matter in practice
(measured, not assumed — same bar `ROADMAP.md` already sets for this kind of
work), a follow-up can have the automatic-save call sites decide
`includeReplayBuffer` *before* writing, using the same "how many
full-with-buffer saves already exist in the last N" count Phase 2's pruning
pass computes anyway.

### Phase 3 — UI

- `TrainingSettingsPopover.swift` Sessions tab (`SessionsTab` struct,
  currently ~line 999): add the "Weights-retention (hours)" field next to
  the existing "Interval (min)" and "Max autosaves kept" fields, same
  stepper-with-text-field pattern, backed by
  `TrainingSettingsPopoverModel.swift`'s existing commit-on-Save flow (not
  live-propagating per keystroke, matching the other two Sessions fields).
  Update "Max autosaves kept"'s help text to reflect its widened scope
  (periodic **and** promotion pools combined, not periodic-only).
- File menu: new "Save Session (Weights Only)" item near the existing "Save
  Session", per Phase 1.

### Full parameter checklist (per this project's CLAUDE.md — walk every item)

For `AutosaveWeightsRetentionHours` (new) and the widened
`MaxPeriodicAutosavesKept` (existing, scope change only):

1. Declare `@TrainingParameter` + add to `allKeys`.
2. Wire the singleton (stored property, `collectValues`/`applyOne`, snapshot
   accessor).
3. Confirm both appear in `--show-default-parameters` and
   `--create-parameters-file` output; no CLI-path hand-listing to update
   (grep confirmed no existing hand-listed params for the two current
   Sessions-tab keys, so treat as macro-covered).
4. Session save/load (`.dcmsession`): **applicable — corrected from an
   earlier draft of this plan.** Both existing Sessions params
   (`periodicAutosaveIntervalSec`, `maxPeriodicAutosavesKept`) already
   round-trip through `SessionCheckpointState` (`SessionCheckpointFile.swift:468,473`,
   written at `SessionController+Checkpoint.swift:846-847`) and get a
   `[RESUME-PARAM]` block on load (`SessionController+Training.swift:300-335`,
   range-validated against each param's own bounds, "kept current" on an
   out-of-range saved value, "(defaulted)" when the saved field is nil).
   `AutosaveWeightsRetentionHours` needs the identical treatment: an
   `Optional` field in `SessionCheckpointState`, populated in
   `buildCurrentSessionState`, and a matching `[RESUME-PARAM]` block. The
   widened `MaxPeriodicAutosavesKept` needs no change here — its existing
   field/block already cover it since the property itself isn't renamed.
5. `results.json` / `CliTrainingRecorder`: not applicable — doesn't
   influence sampling/training math.
6. Runtime log: log the resolved values once per retention sweep in the
   `[PRUNE]`/`[PRUNE-ERR]` lines (e.g. `[PRUNE] retention: full=3
   weightsHrs=72.0`) so a misconfiguration is visible in the session log
   during a long unattended run, not just in the UI.
7. UI position: Sessions tab, per Phase 3 above.
8. Live tunability: both `liveTunable: true`; the retention sweep reads
   `TrainingParameters.shared` fresh at sweep time (it already runs as a
   one-shot `Task.detached` per save, so there's no "periodic reconcile
   loop" to update — it's inherently always-current).
9. Renames: `AutosaveWeightsRetentionHours` is new (no migration
   concern). `MaxPeriodicAutosavesKept`'s `id`/property name and UserDefaults
   key are **unchanged** — only its description and enforced scope widen —
   so no reset-to-default surprise for existing users of that knob.

## Open decisions (need your call before implementation)

1. **`manualPromote` shares the `-promote` disk tag with automatic arena
   promotions** (`SessionSaveTrigger.swift:31`, by design, "so its filename
   is grep-identical"). Under this plan, a user-invoked "Promote Trainee
   Now" autosave would be swept into the prunable automatic pool along with
   real arena auto-promotions, **not** treated as protected like a manual
   save. Alternative: give `manualPromote` its own disk tag so it can be
   exempted — bigger change (new suffix, `disk-cleanup.md` update, filename
   parsers elsewhere that key off `-promote.dcmsession` would need
   checking). **Recommendation: accept the sweep-in (simpler); flag if you
   disagree.**
2. **`signalSave` (SIGUSR2)** is proposed **exempt** from pruning, same
   treatment as `manual` — it's a deliberate "checkpoint now, I'm about to
   deploy a new build" action. Confirm this matches your intent.
3. **Should the "Save Session (Weights Only)" action be available while
   Play-and-Train is stopped, too** (e.g., to snapshot a loaded/resumed
   session's weights without its buffer, for a quick share/comparison), or
   only during an active run? Proposed: same availability as the existing
   "Save Session" item (no new restriction).

## Validation plan

- **Unit tests** (new, in `DrewsChessMachineTests`): a pure-logic
  decision function extracted from `enforceAutosaveRetention` (mirroring
  `PeriodicSaveController`'s "pure-logic scheduler, no Timer, testable in
  isolation" pattern) — given a list of `(filename, ageHours)` tuples plus
  `fullRetentionCount`/`weightsRetentionHours`, assert the correct
  partition into {untouched, strip-buffer, delete-entirely}, including
  boundary cases (exactly `fullRetentionCount` folders present; a folder
  exactly at the `weightsRetentionHours` boundary; `weightsRetentionHours
  == 0` meaning keep forever; `fullRetentionCount == 0`).
- **Manual integration test**: via `--train --parameters <file>` with
  `autosave_weights_retention_hours` set small (e.g. `0.05` = 3 min) and
  `max_periodic_autosaves_kept: 1`, run long enough to accumulate several
  periodic + at least one promotion save; confirm via session log
  `[PRUNE]` lines and `ls Sessions/` that: the newest 1 automatic save
  keeps its buffer, older-but-within-window ones lose `replay_buffer.bin`
  but keep `champion.dcmmodel`/`trainer.dcmmodel`/`session.json` (with
  `hasReplayBuffer: false`), and ones past the window are gone entirely.
  Confirm `-manual` and `-sigusr2` folders created during the same run are
  untouched.
- **Weights-only save test**: trigger "Save Session (Weights Only)" from
  the File menu; confirm the resulting folder has no `replay_buffer.bin`,
  `session.json` reports `hasReplayBuffer: false`, and the save completes
  in roughly the time of a champion/trainer export (not the multi-second-
  to-minute time a multi-GB buffer write takes).
- **Resume test**: resume from a stripped (weights-only, formerly full)
  automatic save; confirm it loads via the existing pre-buffer back-compat
  path (empty buffer, no crash, no error dialog) — this exercises no new
  loader code, just confirms Phase 2 doesn't corrupt `session.json` in a
  way that breaks the existing path.
- **Regression / back-compat**: with `AutosaveWeightsRetentionHours = 0`
  (keep forever) and `MaxPeriodicAutosavesKept = 0` (unlimited), confirm
  behavior is bit-for-bit today's behavior — nothing pruned, matching the
  existing "0 = unlimited" convention.
- Full `DrewsChessMachineTests` suite must pass (per this project's
  standing rule) before considering the plan executed. No existing test
  may be modified or deleted to make it pass without your explicit
  permission.
- Update `documentation/disk-cleanup.md` to reflect the new automatic
  behavior (it currently documents this exact gap as a *manual* cleanup
  problem — once Phase 2 ships, the runbook should say what's now automatic
  vs. what still needs the manual keeper-selection policy, e.g. very old
  `-manual` folders).

## Out of scope for this plan

- Any UI to browse/manage existing accumulated Sessions/ folders ("Manage
  Autosaves" / "Trim to last N" button) — `ROADMAP.md`'s original entry
  mentions this as a nice-to-have; not required to meet the stated goals.
- Retroactively cleaning up the ~210 GB of already-accumulated sessions
  from before this feature ships — that stays a manual `disk-cleanup.md`
  exercise, orthogonal to preventing future unbounded growth.
