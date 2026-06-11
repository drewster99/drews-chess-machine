# Session Picker Plan

A custom "Load Session" picker that shows significant detail about every saved
`.dcmsession` so the right one can be chosen without memorizing folder names.

## Problem

`File > Load Session…` currently presents a bare `.fileImporter` open panel
(`CheckpointController.swift`, `loadSessionImporterPresented`). The only
information visible is the folder name (`<UTC-timestamp>-<modelID>-<trigger>`).
With 83+ sessions on disk across many runs/architectures, picking anything
other than the most recent autosave means cross-referencing logs,
ARCH_EXPERIMENTS.md, or memory. Concrete recent example: locating the one
pre-LR-cycling plateau checkpoint of the Experiment-1 run
(`20260606-002543-20260601-12-5K7Z-periodic`, step 382,625) required log
forensics rather than the UI.

## Goals

- One-glance answer to "which run is this, where in the run is it, and what
  was it configured as" for every saved session.
- Group by run lineage (the user's mental model is *runs*, not files).
- Selection loads through the exact same code path as today's importer.

## Non-goals

- No deletion/management of sessions from the picker (immutability stays).
- No editing of saved sessions.
- No change to the auto-resume sheet (`AutoResumeController`) — it already
  knows which session it's offering.
- No migration of old sessions on disk (read-only legacy support instead).

## Data source facts (measured 2026-06-11)

- `session.json` inside each `.dcmsession` already contains everything a
  picker needs: `architecture` (channels/blocks/inputPlanes/parameterCount/
  policySize/valueHeadClasses), `championID`, `trainingSteps`,
  `elapsedTrainingSec`, `buildNumber`/`buildGitHash`/`buildGitBranch`/
  `buildGitDirty`, `arenaHistory` count, outcome counters, `learningRate`,
  `batchSize`, `hasReplayBuffer`, and (newer saves) the full parameter set.
- BUT `session.json` is ~16 MB (it embeds chart-event and arena-history
  arrays). Decoding 83 × 16 MB synchronously at picker-open time is not
  acceptable; even a slim `Decodable` struct must still parse the full JSON
  text.
- Session folders are ~7+ GB each (replay buffer dominates) — on-disk size is
  itself a useful display column.

## Design

### 1. `SessionManifest` — one compact struct per saved session

A small `Codable` struct holding the picker-displayable subset, organized in
four categories (decision 2026-06-11: maximize information in each):

**A. Architecture**
- architecture summary string (same `architectureSummary` wording as the
  `[ARCH]` log line) for saves that embed the full runtime config
- legacy saves' `architecture` dict carries only 8 scalars
  (architectureVersion, channels, numBlocks, inputPlanes, policySize,
  valueHeadClasses, seReductionRatio, parameterCount) — notably **no kernel
  size** — so legacy rows display those scalars and the parameter count,
  formatted in the same style but without unknowable fields
- parameter count, input-plane count, value-head classes

**B. Run / progress / lineage**
- champion ModelID, trainer ModelID, lineage tag (parsed from folder name)
- save trigger (`manual`/`periodic`/`promote`) + save date (folder name,
  UTC → local), formatVersion
- training steps, elapsed training time (formatted d:hh:mm), games +
  positions emitted, buffer fill/capacity, `hasReplayBuffer`, `hasChartData`
- provenance: build number, git hash + dirty flag, branch

**C. Performance**
- promotion count: `lichessProbeHistory.latestPromotionCount` where present
  (verified present in legacy saves back to at least 2026-06-06); fallback =
  derive by scanning `arenaHistory` during legacy indexing (decision: yes,
  always produce a promotion count)
- arena count; last arena result: candidate vs champion IDs, score, W/D/D
  split (white/black splits available in `arenaHistory` entries)
- latest tactical-battery snapshot from `lichessProbeHistory`: model label,
  training step, positions trained; latest pElo where derivable from the
  stored probe history
- game-outcome mix (checkmates / stalemates / 50-move / threefold /
  insufficient-material counters)

**D. Hyperparameter sampling**
- learning rate (+ cyclic flag), batch size, weight decay, promote
  threshold, self-play + arena tau schedules, replay-ratio target, worker
  count, draw penalty, entropy coefficient — whichever of these the save's
  vintage carries (older saves lack newer parameters; absent → shown as "—")

**Disk**
- total folder size in bytes (display GB)

### 2. Manifest production — write at save, index for legacy

- **New saves:** `CheckpointManager` writes `manifest.json` (small, < 4 KB)
  into the `.dcmsession` folder alongside `session.json` at save time, from
  values it already has in hand. Additive file; nothing existing is
  overwritten, preserving the never-overwrite rule.
- **Legacy sessions (no `manifest.json`):** a background indexer parses each
  `session.json` once with a slim decoder and writes the result to an index
  cache **outside** the session folders:
  `~/Library/Application Support/DrewsChessMachine/SessionIndex/<folderName>.json`
  — old session folders are never touched. Cache invalidation key: folder
  name + `session.json` file size + mtime (sessions are immutable, so this
  effectively never invalidates).
- Indexing runs off the main actor. The 16 MB JSON parses are long-running
  synchronous work → dispatch to a utility `DispatchQueue` and bridge back
  with a continuation, per the project's GPU/IO pattern. Each completed
  manifest is delivered incrementally so rows populate progressively on
  first launch (~83 parses, expected tens of seconds cold, instant warm).
- Decode failures (corrupt/foreign folder) surface as a visible "unreadable"
  row with the error string — not silently dropped.

### 3. Picker UI

New file `App/UpperContentView/SessionPickerSheet.swift` (one View per file),
plus `SessionPickerModel` (`@Observable`, owns the manifest list + scan
state).

- Presented as a sheet from the File menu in place of the raw importer.
- **Grouped by run lineage**: sections keyed by the saved ModelID lineage
  tag (e.g. `5K7Z`, `oItC`, `cwkO`), newest run first, sessions within a run
  in step order. Section header carries the architecture summary + branch +
  date range, so a whole run reads as one unit.
- **Row (table layout, aligned columns, monospaced digits):** save date,
  trigger badge, step count, elapsed training time, arenas/promotions,
  buffer fill, build, disk size.
- **Detail pane** for the selected row: full manifest including champion +
  trainer ModelIDs, git hash/dirty, key params at save, outcome mix, and
  the absolute path with a "Reveal in Finder" button.
- Footer buttons: **Cancel**, **Browse…** (falls back to today's
  `.fileImporter` for sessions outside the default directory), **Load**.
- Light/dark supported; trigger badge colors get fixed semantic meanings
  (manual / periodic / promote).

### 4. Integration

- `AppCommandHub.loadSession()` → presents `SessionPickerSheet` instead of
  toggling `loadSessionImporterPresented`.
- "Load" calls the existing `CheckpointController` load path with the chosen
  URL — zero change to load/validation/resume logic, `[BUTTON]` logging
  preserved (now including the chosen folder name).
- The legacy importer code stays, reachable via **Browse…**.

## Concurrency invariants respected

- `SessionPickerModel` is `@MainActor @Observable`; manifest parsing happens
  on a background `DispatchQueue` bridged by continuations (no long sync
  work on cooperative threads).
- No new locks; the model owns its state on the main actor.

## Validation / completeness criteria

1. Cold open of the picker on the real 83-session directory populates every
   row without beachballing; warm open is instant (index cache hit).
2. `manifest.json` from a fresh save matches the values the slim decoder
   extracts from the same session's `session.json` (round-trip equality
   test).
3. Loading a session via the picker produces a byte-identical code path to
   the old importer (same `[CHECKPOINT] Loaded session:` log line).
4. A deliberately corrupted `session.json` fixture shows an "unreadable" row
   and does not block the rest of the list.
5. All existing tests pass; new XCTests pass without modification after
   implementation.

## New tests (pure-logic, XCTest)

- `SessionManifestDecodeTests`: slim decode of a real (fixture, truncated)
  `session.json` extracts the expected fields; unknown keys ignored; missing
  optional fields tolerated (oldest saves lack newer parameters).
- `SessionManifestFolderNameTests`: trigger + UTC date + lineage tag parsing
  from the three real naming patterns (`manual`/`periodic`/`promote`).
- `SessionPickerModelTests`: lineage grouping, section ordering (newest run
  first), within-run step ordering, unreadable-row retention.

## Open questions — RESOLVED 2026-06-11

1. Promotion count — **yes, always produce it.** Prefer
   `lichessProbeHistory.latestPromotionCount` (present even in legacy saves);
   fall back to a one-time `arenaHistory` scan during legacy indexing.
   Manifest categories expanded to carry as much information as available
   about (A) architecture, (B) run/progress/lineage, (C) performance,
   (D) a sampling of hyperparameters — see §Design 1.
2. `.dcmmodel` files — **out of scope.** Individually saved models are not
   session loads and will be dealt with separately.
3. Sheet size — implementer's choice; default to a comfortably large,
   resizable sheet.
