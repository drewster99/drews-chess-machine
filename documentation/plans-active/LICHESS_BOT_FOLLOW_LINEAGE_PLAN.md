# Lichess bot: follow the newest checkpoint of one lineage on disk

Status (2026-10-06): **IN PROGRESS.** Implemented: P-ready, P0. OD-19 and the §5.1 test edits were decided by the owner on 2026-10-06 (§10); implementation follows the phase order of §7.
- **Owner decisions recorded 2026-10-06** (§10). They amend OD-6, OD-8, OD-9 and OD-17 (keep playing + alarm instead of declining) and add a rule for every source: **the bot builds its model generation before it goes online** (§3.10, OD-18, OD-19). The design, tests, validation and phasing below follow them.
- Every `file:line` was checked against `main` at `f5be524b`.
- Paths are relative to `DrewsChessMachine/DrewsChessMachine/` unless they start with `DrewsChessMachineTests/` (= `DrewsChessMachine/DrewsChessMachineTests/`) or `documentation/`.
- Numbers about the Models folder and the live runs were measured on this Mac on 2026-10-06 from the files and session logs themselves (sizes base-2).
- This extends `documentation/plans-active/LICHESS_BOT_PLAN.md` §9 ("Model sources") and §9.1 ("Choosing a model file by lineage"). When it ships, §9's source table gains a row and a §9.2 points here.

**The request (owner, 2026-10-06).** The bot can play the GUI's champion, a trainer snapshot, the live trainer, or one fixed file. CLI training runs (`--replay-corpus`, `--train-vs-uci`) are separate processes, so the bot cannot follow them. Add a source that follows the newest checkpoint of one model lineage on disk, re-checked on a cadence, using the existing refresh machinery.

**What this plan does.**
- Adds a fifth model source, **Follow lineage** (`LichessBotModelSourceKind.followLineage`).
- Identifies the followed lineage by the `LineageRecord` every current model file carries: the run (`lineage_run_id`) plus the segment the operator chose (`segment_id`). It never uses a filename or a modification time to decide anything.
- Re-checks the Models folder on a cadence. Only new or changed files' headers are read; unchanged files are recognized by device, inode, size and modification time.
- Loads the newest file through the existing model-file loader, records its lineage position with every move, and lets games in progress switch to it (when mid-game refresh is on), exactly as the live-trainer source does.
- When the source cannot vouch for a newer file while online (no file of the lineage, folder unreadable, a fork after the chosen segment, a newest file that fails to load), the bot **keeps playing the last good generation** and raises an alarm (status line + log). It never declines a challenge for it, never switches to another source and never steps back to an older file. Before going online the same conditions make the first build fail, and the bot stays offline with the error.
- **Ready before play, for every source** (owner rule, 2026-10-06): going online builds the first model generation before the event stream opens, so the bot never accepts, declines-for-model or sends a challenge without a generation. A failed build leaves the bot offline with the error. A source change while online builds the new generation in the background and keeps playing the old one until it is ready. The "model not ready" decline and the "no model available" paths are removed by construction (§3.10).
- Fixes two latent bugs found while reading the code (P0, regression tests first): the file loader hashes a second read of the file, not the bytes it decoded; and any model-settings edit (even the refresh interval) stops `refreshIfDue` until the next game starts.
- Fixes a third latent bug the feature would trip (P2, regression test first): the settings store reports saved settings as unreadable when they hold a struct-typed optional whose default is nil — exactly what `followedLineage` is (§3.1).

Rules this plan follows (CLAUDE.md files and the owner's standing rules):
- checkpoints are identified by safetensors `__metadata__` (`model_id`, `training_step`, the `dcm_lineage` record), never by filename;
- the `dcm_lineage` JSON is the single source of truth; its flat mirror keys (`lineage_run_id`, `cum_trainer_step`, …) are never read back (`Persistence/LineageRecord.swift:876-890`);
- no silent defaults or fallbacks: every "can't" is a stated, logged state;
- no `try?`, no force unwraps; long file work runs on a `DispatchQueue` behind a continuation, never synchronously inside a `Task`;
- saved settings from today's build keep loading (owner's past pain: no forced "Reset to defaults");
- SwiftUI: one `View` per new file, no `some View` helper properties, no `AnyView`, `.shown(_:)` instead of `if`-gated content, `onChange` 0/2-arg only;
- tests are not modified or deleted; new tests go in new files; bug fixes get a failing regression test first. **The one exception:** the existing-test edits the ready-before-play rule forces, each listed in §5.1. The owner approved them under OD-18 (2026-10-06); the items the second review added, and every deletion, still need the owner's express approval (§5.1, "Exact list for approval").

---

## 1. How things work today (verified)

### 1.1 Model sources and refresh

- Sources: `champion`, `trainerSnapshot`, `liveTrainer`, `file` (`LichessBot/LichessBotSettings.swift:94-105`). The default is `file` with an absolute path (`:108-110`).
- `LichessBotModelSettings` holds `source`, `filePath`, `liveTrainerRefreshIntervalSeconds` (120), `midGameRefresh` (false) (`LichessBotSettings.swift:107-115`). Validation: interval ≥ `LichessBotLimits.minimumLiveTrainerRefreshSeconds` (30), and a file source needs a path (`:257-260`, `:295-297`).
- Champion / trainer weights come from the GUI's own `SessionController` (`LichessBot/App/LichessBotSessionModelProvider.swift:29-76`). Nothing there can see another process.
- `LichessBotModelSlots` (actor) owns the current generation (`LichessBot/Play/LichessBotModelSlots.swift:80-98`):
  - `ready(for:)` returns the current generation when `currentSettings == settings`, else rebuilds (`:102-107`).
  - `rebuild` shares one in-flight build between callers (`:110-122`).
  - `refreshIfDue(for:)` re-snapshots on a champion ModelID change or when the live-trainer interval has elapsed; it does nothing unless `currentSettings == settings` (`:127-142`).
  - `forceRefresh(for:)` exists (`:145-147`) but nothing calls it — there is no "Re-snapshot now" button today (grep of `DrewsChessMachine/` and `DrewsChessMachineTests/`).
  - `sourceAvailable(for:)`: the `file` source is "available" when a path is set (`:152-161`).
  - `build(for:)` loads the file via `loadFile`, builds a fresh inference network, publishes the generation and logs `[LICHESS-BOT] model generation N ready: …` (`:163-207`).
  - `loadFile` runs on `DispatchQueue.global` behind a continuation; it calls `CheckpointManager.loadModelFile(at:)` (which reads the file) and then **reads the file a second time** to hash it (`:213-239`, the two reads at `:219-220`).
- The poll loop calls `refreshIfDue` every `modelRefreshInterval` = 15 s, with exponential backoff (30 s → 900 s) and an alarm on failure; a model-settings change resets the backoff (`LichessBot/App/LichessBotController.swift:2874-2897`, `:2913-2916`). It publishes `slots.current?.info` as `controller.generation` (`:2843-2845`).
- The session manager declines a challenge when `sourceAvailable` is false (`LichessBot/Play/LichessBotSessionManager.swift:624`), builds before accepting (`:659`) and at `gameStart` (`:736`), and hands each game `latestMoveSource: { await slots.current }` (`:769`).
- Mid-game refresh is live-trainer-only: the playing generation switches to `latestMoveSource()` only when the settings source, the playing generation and the latest generation are all `.liveTrainer` (`LichessBot/Play/LichessBotGameSession.swift:602-616`).

### 1.2 What a generation records

- `LichessBotGenerationInfo`: generation ID, `sourceKind`, `modelID`, `trainingStep`, `snapshotAt`, architecture summary, `filePath`, `fileSHA256`, `valueHeadRecenteredOnLoad` (`LichessBot/Play/LichessBotGameInterfaces.swift:19-40`). The last field is `var … = nil` so records written before it still decode (`:33-39`).
- The `trainingStep` doc says "nil for … a file" (`:26-28`), but the file source does record the file's `training_step` (`LichessBotModelSlots.swift:226`). The comment is wrong; P3 corrects it.
- Every move event carries its generation (`LichessBotGameInterfaces.swift:68`); a game record keeps every generation that chose a move (`LichessBot/Data/LichessBotGameRecord.swift:148-149`); the PGN writes `DCMModelIDs` / `DCMSources` tags (`LichessBot/Data/LichessBotPGNWriter.swift:46-47`); the index keeps `sourceKinds` (`LichessBot/Data/LichessBotIndex.swift:51`).
- Chat `{source}` expands to the source's raw value, validated against a 20-character budget (`LichessBot/Play/LichessBotChat.swift:24`, `LichessBotGameSession.swift:1145`).

### 1.3 The line picker and the catalog

- `ModelFileCatalog` lists `Models/` (not recursive), keeps non-hidden `.safetensors` files only (`Persistence/ModelFileCatalog.swift:92-97`), reads each file's header only (`:181-217`), and groups files into lines by `model_id` (`:99-113`). It has **no cache**: every scan reads every header.
- `ModelFileEntry` holds `model_id`, `training_step`, `created_at_unix`, architecture label, modification date, `parent_model_id`, `creator` (`:7-21`). It carries no lineage record and no content hash.
- Within a line, "latest" is the highest `training_step`, ties by newest modification date (`:116-129`).
- `ModelLineageTree` builds seed → segment trees from `parent_model_id`; a "segment" row is one `model_id` (`Persistence/ModelLineageTree.swift:6-9`, `:69-138`).
- `LichessBotModelLinePicker` scans `Models/` and `Sessions/` champions and returns one URL (`LichessBot/UI/LichessBotModelLinePicker.swift:8-10`, `:79-102`). The settings section's "Latest by lineage" button stores that path once (`LichessBot/UI/LichessBotSettingsView.swift:406-425`).

### 1.4 What CLI runs write, and how

- Every corpus-replay / train-vs-UCI launch mints a new `model_id`, an exact resume included (`CLI/CorpusReplayRunner.swift:610-620`). So one CLI process = one `model_id` = one lineage segment.
- `LineageRecord.Run`: `lineage_run_id` (minted on fresh / branch / derive, inherited by every exact resume), `segment_index` (+1 per resume), `segment_id` (per process), `recorded_unix` (save time) (`Persistence/LineageRecord.swift:91-123`). `Steps`: `cum_trainer_step` (nullable), `segment_local_step` (the CLI files' `training_step`) (`:176-212`). `Invocation.pathKind`: `gui | replay | vsuci | derive | new_model` (`:417-435`).
- A resume record carries every earlier segment of its run, oldest first, with each segment's `segment_id`: `segments = record.segments + [SegmentSummary(of: record)]` (`Persistence/LineageTracker.swift:188-202`; `SegmentSummary` at `LineageRecord.swift:713-745`). A resume that is not exact (`--accept-inexact`) also continues the run, so two processes resumed from the same file are **sibling segments of one run** (`LineageTracker.swift:102`, `:188-202`).
- Header-only read of identity + lineage already exists: `SafetensorsModelIO.readParentFile(fromMetadata:source:)` → model ID, `content_sha256`, trainer clock, `LineageRecord.Presence` (`Persistence/SafetensorsModelIO.swift:389-409`, lineage decode at `:372-383`). Files before format v7 report `.unrecorded` (`Network/ArchitectureFormat.swift:109`).
- Writes are complete before they are visible under their final name:
  - corpus replay's rolling `--out-model` and step-enumerated files go through `FileSafety.replaceRegularFile` / `publishNewFile` (`CLI/CorpusReplayRunner.swift:544-559`, `:566-592`), which stage in a hidden sibling `.<name>.<UUID>.tmp` and `rename` into place (`Utils/FileSafety.swift:561-571`, `:597-627`, `:817-818`);
  - GUI `saveModel` stages `<final>.safetensors.tmp` and renames exclusively (`Persistence/CheckpointManager.swift:1270`, `:1320`);
  - neither staging name passes the catalog's filter (hidden, or extension `tmp`) (`ModelFileCatalog.swift:93-97`).
- The rolling file and the step file of one save are written from the same `encoded` bytes (`CorpusReplayRunner.swift:1606`, `:1645`). Measured: `20261005-lrBleaky-cyc1-replay-latest.safetensors` and `…-step31000.safetensors` have the same `content_sha256` (`1447ee85e98b…`) and the same record.
- Train-vs-UCI writes step files only with `--enumerate-checkpoints`, into `Models/` unless a stem or a start-model file elsewhere sets the folder (`CLI/TrainVsUciSession.swift:248-268`). Its session folders go to `Sessions/` and are not model files in `Models/`.
- A full decode verifies `content_sha256` against the data region (`Persistence/SafetensorsFile.swift:231-237`), so a truncated or half-copied file fails loudly at load.

### 1.5 Measured cost (2026-10-06, this Mac)

- `Models/`: 4,462 `.safetensors` files, 83 GB on disk. Of those, 185 carry a lineage record (180 `replay`, 5 `new_model`, no `vsuci`, no `gui`); four runs have two segments (re-measured by the review, same day).
- One current header: about 21.5 KB (22,002 bytes); all headers: 37.3 MB. Reading them all (no JSON parse): 0.69 s warm.
- Listing the folder and `lstat`-ing every `.safetensors` file: about 13 ms per pass (review re-measure: 9 ms warm, 21 ms cold).
- A live replay file is 39 MB. The three live replay runs save every 1,000 steps, about every 33–36 minutes (`[REPLAY] enumerated checkpoint ->` lines in `dcm_log_20261005-204108.txt`, `dcm_log_20261005-234434.txt`, `dcm_log_20261005-234437.txt`).

### 1.6 Going online and model readiness today

- `goOnline` (`LichessBot/App/LichessBotController.swift:774-830`) sets `.connecting`, loads player notes, then calls `startRuntime`; on success it sets `.online` (`:816`), on failure `.error(text)` with an `[ALARM]` (`:822-829`).
- `startRuntime` (`:2537-2557`, `:2559-2708`): reads the token, takes the instance lock, verifies token and account (`:2580-2591`), creates the model slots **without building anything** (`:2600-2602`), creates the manager and reconciler, then starts the event-stream handler, `manager.run()` (`:2682`), the reconciler and the poll loop (`:2688-2690`). `runtime` is assigned last (`:2703`).
- The first generation is built lazily: when the first challenge is accepted (`LichessBot/Play/LichessBotSessionManager.swift:656-665`) or the first `gameStart` arrives (`:733-744`). A build failure at accept declines the challenge (`later`, anomaly "model not ready, declining …", `:660-664`); at `gameStart` it leaves a running game with no model (anomaly "game … started but no model is available", `:737-743`).
- The challenge policy declines with `later` / rule "model not ready" when `context.modelReady` is false (`LichessBot/Play/LichessBotChallengePolicy.swift:18-19`, `:53-55`), fed by `slots.sourceAvailable` (`LichessBotSessionManager.swift:624-629`).
- A source change while online is also lazy: the next `ready(for:)` call (a challenge or a game) rebuilds "source changed", and a failure there declines that challenge (`LichessBotModelSlots.swift:102-107`).
- Sends are refused unless `connection == .online` (`LichessBotController.swift:1657`, `:1881`), so nothing is sent while connecting.
- `goOffline` returns doing nothing while connecting, because `runtime` is still nil (`:1020-1021`); the Go Offline buttons are disabled then (`isRunning` = `runtime != nil`, `:470-472`; `LichessBot/UI/LichessBotOverviewView.swift:84-87`, `LichessBot/UI/LichessBotStatusChip.swift:44-47`). Going online cannot be cancelled today.
- `autoConnectOnLaunch` (`LichessBotSettings.swift:121`) is read nowhere (grep), so nothing goes online without the operator.
- Mid-game refresh reports an anomaly when no generation is built (`LichessBotGameSession.swift:612-614`); `latestMoveSource` is optional (`:45`).

---

## 2. Design alternatives considered

### 2.1 What "one lineage" is

- **A — run + anchor segment (chosen).** Store `lineage_run_id` and the `segment_id` the operator picked. A file belongs to the followed lineage when its run matches and its segment chain (`segments[].segment_id` + its own `segment_id`) contains the anchor.
  - Follows a run through every exact resume (a new process, a new `model_id`) automatically.
  - Never picks up a branch (`--start-model` without `--resume-exact` starts a new run).
  - Sibling segments after the anchor are detected exactly and reported, and the operator can resolve them by anchoring on one sibling. So is a parent segment that kept training after a child resumed it (§3.3 step 2b) — the chains are prefix-ordered there, so prefix order alone would miss it.
  - Uses only the record's own chain, so a segment that wrote no file into `Models/` does not break the chain.
- **B — run only.** Simpler, but sibling segments of one run (two `--accept-inexact` resumes of one file — a real pattern for A/B experiments) would be ambiguous with no way for the operator to say which.
- **C — `model_id` plus `parent_model_id` descendants (the picker's tree).** Every branch is a descendant, so most interesting seeds (Qeu8 → GLu5 → …, many branches) are permanently ambiguous. Rejected.

### 2.2 How to notice new files

- **A — poll with a stat cache (chosen).** One directory listing + `lstat` per check (≈13 ms for 4.4k files); headers are read only for new or changed files. Robust to sleep, external volume changes and missed events.
- **B — directory events (`DispatchSource.makeFileSystemObjectSource` / FSEvents).** Lower latency, but a rename-into-place, a coalesced event or a sleep can be missed, so a periodic rescan is still needed for correctness. The poll alone is cheap enough; events add a second mechanism for no measurable gain at a 30-minute save cadence. Rejected for v1.

### 2.3 How to rank "newest"

- **A — chain depth, then `segment_local_step`, then `recorded_unix` (chosen).** Within a linear chain a deeper segment is always later work. Within one segment, the local step only moves forward on the CLI paths. `recorded_unix` breaks a tie between two saves at one step (a final save replacing the last autosave).
- **B — `cum_trainer_step` first.** Wrong when a segment is resumed from an earlier file of its parent after the parent stopped: the new segment's files start below the parent's latest cumulative step but are the live continuation. (If the parent kept training, neither is "the" continuation: a fork, §3.3 step 2b.) Also `cum_trainer_step` is nullable (`LineageRecord.swift:180`). Shown for display only.
- Never: filename, `training_step` across segments (segment-local), modification time.

---

## 3. Design

### 3.1 Settings (`LichessBot/LichessBotSettings.swift`)

- New case `LichessBotModelSourceKind.followLineage` ("A model lineage on disk: the newest file of one training run, from a chosen segment on, re-checked on a cadence"). Raw value `followLineage` (13 characters, inside the chat `{source}` budget at `LichessBotChat.swift:24`).
- New type:
  ```swift
  /// The lineage the follow-lineage source plays: one training run, from
  /// one of its segments on (plan LICHESS_BOT_FOLLOW_LINEAGE_PLAN §3).
  struct LichessBotFollowedLineage: Sendable, Equatable, Codable {
      /// `LineageRecord.Run.lineageRunID` of the run.
      let lineageRunID: String
      /// `LineageRecord.Run.segmentID` of the segment the operator chose.
      /// Files of this segment and of every later segment descending from it
      /// are candidates.
      let anchorSegmentID: String
  }
  ```
- `LichessBotModelSettings` gains:
  - `var followedLineage: LichessBotFollowedLineage? = nil` — Optional. A save without it decodes as nil; the store leaves an absent optional absent (`Data/LichessBotSettingsStore.swift:63-71`).
  - **A save *with* it would not load today (store bug, verified).** A key present in the save but absent from the default object (an optional whose default is nil) goes through `isKnownOptionalKey` (`LichessBotSettingsStore.swift:123-133`, `:178-189`), which decodes the defaults with `{"__probe__": true}` at that path and counts the key as known only on `typeMismatch` / `dataCorrupted`. A probe at a **struct-typed** optional throws `keyNotFound` (`lineageRunID`), which is not caught there, propagates out of `overlay`, and `loadReporting` reports the whole saved settings as **unreadable** — the operator would be forced to "Reset" on the next launch after choosing a lineage. (Checked with a standalone Swift script of the same shapes: `DecodingError.keyNotFound … Path: model.followedLineage`.) Today's nil-default optionals are all `String?` (`LichessBotAlertSettings.botChallengeSoundName` / `humanChallengeSoundName`, `LichessBotSettings.swift:198-200`), whose probe fails with `typeMismatch`, so nothing has hit it yet.
  - Fix (P2, regression test first): `isKnownOptionalKey` also returns true for `DecodingError.keyNotFound` whose `context.codingPath` equals `keyPath` — the decoder entered the probe object, so the field exists. Any other `keyNotFound` still propagates.
  - `var lineageCheckIntervalSeconds = 60` — non-optional. A save without it is filled from today's default and **reported** in the existing `[LICHESS-BOT] settings: saved settings predate … using today's defaults for: model.lineageCheckIntervalSeconds` line (`LichessBotController.swift:420-423`). (OD-2.)
- `midGameRefresh` keeps its Swift name and key; its doc becomes "live trainer and followed lineage". `liveTrainerRefreshIntervalSeconds` is untouched (OD-3).
- `LichessBotLimits`:
  - `static let modelRefreshPollSeconds = 15` — moved here from `LichessBotController.modelRefreshInterval` (`LichessBotController.swift:2913-2914`), which then reads it. One source of truth for the poll cadence.
  - `static let minimumLineageCheckSeconds = modelRefreshPollSeconds` — a check cannot run more often than the poll that triggers it.
- `validationProblems()` adds:
  - `lineageCheckIntervalSeconds >= LichessBotLimits.minimumLineageCheckSeconds`;
  - when `source == .followLineage`: `followedLineage != nil` ("Choose a lineage to follow"), and both IDs non-empty.
- Tabs: model fields already belong to the Play tab as a whole (`LichessBot/UI/LichessBotSettingsTab.swift:65`), so new problems show on Play with no tab change.
- Downgrade: settings saved with `source = followLineage` do not decode in an older build (unknown enum raw value). Upgrades are what the store protects; a downgrade showing the existing "can't be read" state is accepted (OD-15).

### 3.2 Header facts and an incremental scan (`Persistence/`)

**`ModelFileEntry` gains lineage facts** (`Persistence/ModelFileCatalog.swift:7-21`):
- `var contentSHA256: String? = nil` — the header's `content_sha256`.
- `var lineage: ModelFileLineageFacts? = nil` — nil only for entries built outside the catalog (existing tests build entries memberwise: `DrewsChessMachineTests/ModelFileCatalogTests.swift:53`, `ModelLineageTreeTests.swift:8`); the catalog always sets it.
- New types (same file):
  ```swift
  /// What a model file's header says about its lineage.
  enum ModelFileLineageFacts: Sendable, Equatable {
      case recorded(ModelFileLineagePosition)
      /// Written at a format before `ArchitectureFormat.lineageRequiredFromVersion`.
      case unrecorded(formatVersion: Int)
      /// The record is present but does not decode; the file still lists.
      case unreadable(reason: String)
  }
  /// Where a file sits in its run, from its `dcm_lineage` record.
  struct ModelFileLineagePosition: Sendable, Equatable {
      let lineageRunID: String
      let segmentID: String
      let segmentIndex: Int
      let segmentStartedUnix: Int64
      /// Earlier segments' IDs, oldest first, then this file's own.
      let segmentChain: [String]
      /// One per earlier segment, aligned with `segmentChain`: the segment
      /// that resumed it, the step it handed on at and when the resuming
      /// segment started (`SegmentSummary.segmentLocalStep` / the next
      /// link's `startedUnix`). Read by the fork check (§3.3 step 2).
      let handoffs: [LineageHandoff]
      let segmentLocalStep: Int
      let cumTrainerStep: Int?
      let recordedUnix: Int64
      let pathKind: LineageRecord.PathKind
  }
  /// Segment `fromSegmentID` was resumed by `toSegmentID`, which started at
  /// `toStartedUnix` from `fromSegmentID`'s file at `atLocalStep`.
  struct LineageHandoff: Sendable, Equatable {
      let fromSegmentID: String
      let atLocalStep: Int
      let toSegmentID: String
      let toStartedUnix: Int64
  }
  ```
- `ModelFileCatalog.entry(for:)` (`:131-177`) becomes `entry(for:metadata:)` over a header already read, plus the existing `entry(for:)` reading it. Lineage comes from `SafetensorsModelIO.readParentFile(fromMetadata:source:)` (`SafetensorsModelIO.swift:396-409`) — the one header-only lineage reader. Any error it throws (it also throws for a bad format version, trainer clock or derivation history, not only for the record) becomes `.unreadable(reason)`, not a thrown error, so the file picker lists exactly the files it lists today.
- `handoffs` are built from the record's `segments` (each `SegmentSummary.segmentLocalStep` is the step that segment's resumed file was at; the resuming segment's start is the next summary's `startedUnix`, or the record's own `run.segmentStartedUnix` for the last link).

**`Persistence/ModelFolderHeaderCache.swift` (new)** — incremental scan of one folder:
- `struct ModelFolderHeaderCache: Sendable` — per URL: `FileSafety.FileIdentity` (device + inode, `Utils/FileSafety.swift:98-101`), size, modification date, and the result (`ModelFileEntry` or `UnreadableModelFile`).
- `static func scanSynchronously(directory:previous:) throws -> ModelFolderScan` — lists the folder with the catalog's filter exactly as it is today (extension `safetensors`, `.skipsHiddenFiles`; today's catalog does **not** filter by item type — `ModelFileCatalog.swift:93-97`), `stat`s each one, and re-reads a header only when identity, size or modification date differs from `previous`. Removed files drop out. Returns the new cache, entries, unreadable files and counts (`listed`, `headersRead`, `reused`, elapsed ms).
  - The `stat` follows a symbolic link, because the header read and the load follow it too: the cache key must describe the bytes that are read. `FileSafety.existingItem(at:)` (`:183-192`) returns only kind and identity, with `lstat`, so `FileSafety` gains one read-only helper returning identity, kind, size and modification time of what a path resolves to (CLAUDE.md: extend `FileSafety`, don't add a second helper elsewhere).
  - An item that resolves to something other than a regular file (a folder named `*.safetensors`, a dangling link) is an `UnreadableModelFile` with that reason — listed as unreadable, as today's header read already makes it — never dropped silently. This keeps `ModelFileCatalog`'s result unchanged.
- A replaced file (rolling `--out-model`, renamed over) is a new inode, so it is always re-read even at the same size and second.
- An unreadable result is cached by identity/size/date, so a half-copied file (an outside `cp`) is re-read as soon as it grows, and not re-read every check while it sits unchanged.
- A folder that cannot be listed throws (`folderUnreadable`), never returns "empty".
- `ModelFileCatalog.scanSynchronously` (`:92-114`) is re-expressed over this scan with an empty `previous`, so there is one listing filter and one header parser. Its result is unchanged (pinned by `ModelFileCatalogTests` and `ModelFileCatalogActivityOrderTests`).

### 3.3 Choosing the newest file (`Persistence/ModelLineageTip.swift`, new, pure)

`enum ModelLineageTip { static func select(entries:followed:notBelow:) -> Selection }`:
1. **Candidates**: entries whose lineage is `.recorded`, whose run is `followed.lineageRunID`, whose `segmentChain` contains `followed.anchorSegmentID`, whose `pathKind` is `replay` or `vsuci` (OD-4), and which have a `contentSHA256`. Everything else is excluded; files of the run that are excluded are **counted by reason** (other path kind, no content hash, unreadable record) for the status line.
2. **Fork check**:
   - (a) Collect the distinct segments among candidates. They must be totally ordered by "is an ancestor of" (one segment's chain is a prefix of the other's). If two are not, the result is `.fork(segments:)` naming each tip segment's ID, `model_id` and newest step (OD-9).
   - (b) **A segment that kept training after it was resumed is a fork too.** For every handoff in a candidate's chain (`fromSegmentID` resumed by `toSegmentID` at `atLocalStep`), a candidate file of `fromSegmentID` with `segmentLocalStep > atLocalStep` and `recordedUnix >= toStartedUnix` means the parent went on training while the child ran — two live continuations of one point, though the chains are prefix-ordered. Result `.fork`, naming both. (Without this, a second process `--resume-exact`-ed from a live run's step file — an A/B pattern — would silently become "newest" by chain depth while the run the operator anchored on keeps going.) Anchoring on the child resolves it (the parent's files then fall outside the candidates). No anchor can select the parent's own continuation instead, because the child's chain contains the parent; to follow the parent, use the fixed-file source until the child stops — a v1 limitation, named in OD-17.
   - (c) Parent files beyond the handoff that were all recorded **before** the child started mean the child resumed the parent from an earlier file after the parent stopped (a redo); the child is followed, and the first selection logs it once (`segment <child> resumed <parent> at step h, below its newest step t`). OD-17.
3. **Rank**: deepest segment chain, then `segmentLocalStep`, then `recordedUnix`.
4. **Same weights**: among the top-ranked candidates, files with equal `contentSHA256` are one candidate (the rolling file and its step copy, §1.4). The URL kept is the first in path order — a deterministic choice between byte-identical files, not an identity decision. (Path order puts `…-replay-latest` before `…-replay-step<N>`, so the kept file is usually the rolling one, the one most likely to be replaced before the load; the load's verification (§3.4) covers that, at the cost of a "file changed since the check" note.)
5. **Distinct weights at one rank** (two different files claim the same segment and step and save time — should not happen): `.conflict(files:)`, reported like a fork.
6. **Never backward** (OD-7): `notBelow` is the playing generation's position (its `LichessBotGenerationLineage`, §3.6, which carries the same rank key: chain, `segmentLocalStep`, `recordedUnix`). If the best candidate ranks below it (its file was deleted), the result is `.keepPlaying(newestOnDisk:)`. If the best candidate's chain is not prefix-ordered with the playing generation's chain, that is a fork against what is playing: `.fork`.
7. **Nothing**: no candidate → `.noFiles(excludedCounts:)`.

### 3.4 The follower in `LichessBotModelSlots`

New state on the actor (`LichessBot/Play/LichessBotModelSlots.swift`):
- `lineageCache: ModelFolderHeaderCache`, `lineageStatus: LichessBotLineageFollowStatus?`, `lastLineageCheckAt: Duration?`, `pendingLineageCheck: Task<…>?` (joined by concurrent callers, like `pendingBuild` at `:92`), and `failedLineageFiles: Set<FileFingerprint>` (identity + size + date of files that failed to load).
- A scanner protocol so tests can script the folder: `protocol LichessBotModelFolderScanning: Sendable { func scan(previous: ModelFolderHeaderCache) async throws -> ModelFolderScan }`. Production: `LichessBotModelsFolderScanner(directory: CheckpointPaths.modelsDir)`, which runs `ModelFolderHeaderCache.scanSynchronously` on its own serial `DispatchQueue` (`.utility`) behind `withCheckedThrowingContinuation` — the `ModelFileCatalog.scan` pattern (`ModelFileCatalog.swift:84-90`).
- Construction: only through `LichessBotModelSlots.prepare(for:provider:time:folderScanner:log:)` (§3.10); the actor's `init` is private and `init(provider:time:log:)` no longer exists, so every test that builds slots directly changes (§5.1). For `.followLineage`, `prepare` hands the actor the scan cache, the follow status and `lastLineageCheckAt` of its forced check together with the first generation, so the first poll does not re-read every header and the status is shown from the moment the bot is online. The app gets the scanner from `LichessBotControllerServices` (§3.10).

`LichessBotLineageFollowStatus` (Sendable, Equatable) — what the UI and logs show:
- `followed`, `checkedAt: Date`, `outcome` = `.following(newest:)` | `.keepPlaying(newestOnDisk:)` | `.noFiles` | `.fork` | `.conflict` | `.folderUnreadable(reason)` | `.newestFailedToLoad(file:reason:)`, plus scan counts and excluded-by-reason counts.

Behavior:
- **`checkLineage(settings, force:)`** — scans (joining an in-flight scan), selects with `notBelow` = the current generation's position when it is of this followed lineage, updates the status, logs on change only (§3.7). Throttled by `lineageCheckIntervalSeconds` measured on the injected `time` source unless forced.
  - Actor reentrancy: `current` can change while the scan is awaited (a build finishing), so `notBelow` is read **after** the scan returns, never captured before it.
- **No availability check.** `sourceAvailable` (`:152-161`) is removed for every source (§3.10): while online a generation always exists, so nothing is declined for the model. The follow status is reported (status line, log, alarm), never turned into a decline (OD-6, OD-8, OD-9 as amended).
- **Before going online** (`LichessBotModelSlots.prepare`, §3.10) for `.followLineage`: one forced check, then a build of the selected file. Any outcome other than `.following` — `.noFiles`, `.fork`, `.conflict`, `.folderUnreadable`, `.newestFailedToLoad` — throws a `LichessBotLineageFollowError` naming it, so the bot stays offline with that error.
- **`build`** (`:163-207`) for `.followLineage`: takes the selected candidate (checking first if needed; throws a `LichessBotLineageFollowError` naming the outcome when there is none), loads it with the shared loader (§3.5), then verifies the **decoded** file (its `lineageParent.lineage`, `ModelCheckpointFile.swift:370-385`): same run, chain contains the anchor, path kind allowed, its chain equal to or extending the candidate's (the same segment or a descendant — a sibling segment of the same run that took over the same `--out-model` path must not pass), and ranks at or above the candidate. A rolling file replaced between scan and load passes (it moved forward within the same lineage) and the generation records what was actually loaded, with a log note; anything else throws `followedFileChangedDuringLoad` and the next check rescans. A **file** failure (read, decode, `content_sha256`, the verification above) records the file's fingerprint in `failedLineageFiles`, so it is not retried until it changes on disk; the selection treats a failed file as `.newestFailedToLoad`, never as absent (no stepping back past it). While online the playing generation keeps serving new and in-progress games (OD-6 as amended); the failed load throws once, so the controller raises one alarm for that file, and later checks report `.newestFailedToLoad` in the status without throwing again until the file changes. A failure after the file decoded (building the inference network) says nothing about the file and does not mark it: it throws, and the controller's backoff retries it.
- **`refreshIfDue`** (`:127-142`) for `.followLineage`: when the interval has elapsed, check; rebuild when the selection's `contentSHA256` differs from the playing generation's and ranks above it. "No newer file" is not an error. `.noFiles`, `.fork`, `.conflict` and `.folderUnreadable` throw on every check, so the controller's existing backoff and alarm apply (`LichessBotController.swift:2887-2897`) while the playing generation keeps playing (OD-8, OD-9 as amended); §4's "alarm" rows depend on `.noFiles` throwing too. When the outcome clears, the next check logs `available again` and the backoff resets on its success.
  - Today's guard returns early when there is no `current` generation (`:128`). That case no longer exists: `current` is non-optional (§3.10), and the first check runs inside `prepare`, before going online.
- **Operator "Check now"** (OD-12): `checkLineageNow(for:)` = `checkLineage(force: true)` + the same rebuild rule. Never bypasses any rule.
- **Architecture changes**: an exact resume cannot change architecture, so within one run it should not happen. The bot builds a network per generation from the file's own architecture (`:186`), so a change would still play correctly; it is logged in the ready line, not refused (OD-16).

**Settings comparison fix (P0, latent bug).** `ready` and `refreshIfDue` compare whole `LichessBotModelSettings` (`:103`, `:128`). Editing the interval or the mid-game toggle while online therefore makes `refreshIfDue` return early until the next game calls `ready`, which then rebuilds needlessly ("source changed"). For a follower that means following silently stops after an interval edit.
- **How the phases split it.** P-ready (§3.10, done first) moves source switching into `refreshIfDue` (the poll loop) and removes `ready`, so the **stall** ends there: a changed settings value is acted on at the next poll. Kept as-is, the whole-settings comparison would then turn every interval or toggle edit into an immediate **needless rebuild** — the half P0 fixes, with its regression tests written against the P-ready API (§5).
- Fix: compare a `generationSource` value for "is this the same generation source", and adopt the other fields without a rebuild.
- `generationSource` holds only what selects the weights for the settings' kind: the kind, plus `filePath` for `.file` and `followedLineage` for `.followLineage`. (A `filePath` left over in the settings while the source is the live trainer must not rebuild the live trainer.)
- Both comparisons use it: the switch test in `refreshIfDue` and the in-flight build join in `rebuild` (`:111`, `:117`) — otherwise two callers that differ only in the interval start two builds.
- On a match, `currentSettings` takes the caller's settings, so the interval and toggle in force are the newest.
- The controller's backoff reset on "settings changed" (`LichessBotController.swift:2875`) keeps comparing whole settings: an interval edit is still a new attempt.

### 3.5 Loading: one read, one hash (P0, latent bug)

- Today `loadFile` decodes one read and hashes another (`LichessBotModelSlots.swift:219-220`). With the rolling file replaced every save, the recorded `fileSHA256` can name bytes that were never played.
- Fix: `CheckpointManager` gets `loadModelFile(fromBytes:source:)` — the body of `loadModelFile(at:)` after its read (`Persistence/CheckpointManager.swift:1851-1862`: decode with `.recenterUnlessMarked`, log legacy resolutions and value-head centering) — and `loadModelFile(at:)` becomes "read, then that". `LichessBotModelSlots.loadFile` reads the bytes once, decodes them through `loadModelFile(fromBytes:source:)`, and hashes the same `Data`. One loader, as §9 requires.
- `Data(contentsOf:)` keeps reading the opened inode even if the path is renamed over mid-read, so the bytes are one consistent file; the decode's `content_sha256` check catches a half-copied file (`SafetensorsFile.swift:231-237`).

### 3.6 Generation attribution (`LichessBot/Play/LichessBotGameInterfaces.swift`)

- `LichessBotGenerationInfo` gains `var lineage: LichessBotGenerationLineage? = nil` (same pattern as `valueHeadRecenteredOnLoad`, `:33-39`; old journals and records decode; existing memberwise constructions in tests, e.g. `DrewsChessMachineTests/LichessBotGameSessionFaultTests.swift:123-133`, compile unchanged):
  ```swift
  struct LichessBotGenerationLineage: Sendable, Equatable, Codable {
      let lineageRunID: String
      let segmentID: String
      let segmentIndex: Int
      /// Earlier segments' IDs, oldest first, then `segmentID` — with
      /// `segmentLocalStep` and `recordedUnix`, the rank key §3.3 compares
      /// a candidate against (never-backward, fork against what plays).
      let segmentChain: [String]
      let segmentLocalStep: Int
      let recordedUnix: Int64
      let cumTrainerStep: Int?
      /// The header's `content_sha256` of the file played.
      let contentSHA256: String
      /// The followed lineage this generation was built for; nil for the
      /// fixed-file source.
      let followed: LichessBotFollowedLineage?
  }
  ```
- Filled for `followLineage` and also for `file` when the file carries a record (OD-11): same loader, free, and every Lichess game becomes traceable to a run and cumulative step.
- `trainingStep` stays the file's `training_step`; its stale doc comment (`:26-28`) is corrected.
- Records, PGN (`DCMSources` gains `followLineage`) and the index need no other change. No new PGN tag (non-goal).

### 3.7 Logging (`[LICHESS-BOT]`, through the slots' `log`)

- On the first check and on every **change** of outcome or newest file (not every 60 s):
  `[LICHESS-BOT] lineage follow: run=<id> anchor=<segment_id> newest=<file> model=<model_id> seg=<k> step=<local> cum=<n|null> sha=<12> files=<n> excluded=<reason:n,…> scan ms=<x> headers read=<a> reused=<b>`
- Problems, on transition: `[LICHESS-BOT] lineage follow problem: <outcome with names>; still playing generation N (<model_id> step s)` and, when it clears, `[LICHESS-BOT] lineage follow available again: …`. The alarm itself is the controller's existing `Model refresh failed: …` (`LichessBotController.swift:2896`), whose text is the outcome's description.
- Never-backward: `[LICHESS-BOT] lineage follow: newest on disk <file> (seg k step s) ranks below the playing generation N (seg k step s'); keeping generation N`.
- A failed load: `[LICHESS-BOT] lineage follow: <file> could not be loaded: <error>; not retried until it changes; still playing generation N`. (There is no "nothing playing" variant: while online a generation always exists, §3.10.)
- The existing ready line (`LichessBotModelSlots.swift:205`) adds, for file-backed sources, `file=<name> seg=<k> cum=<n|null> sha=<12> arch=<summary>` and, for a load that found a newer rolling file than the scan, `(file changed since the check: loaded step s)`.
- Game start already logs the source and model (`LichessBotController.swift:3076-3083`); it adds `step=` and `cum=` when the generation has lineage.

### 3.8 Mid-game refresh (`LichessBot/Play/LichessBotGameSession.swift:602-616`)

- Generalize the live-trainer-only condition: when `midGameRefresh` is on, the settings source is `liveTrainer` or `followLineage`, the playing generation's `sourceKind` equals it, and the latest built generation has the same `sourceKind`, the same `lineage?.followed` (nil for live trainer) and a higher generation ID, switch to it.
- So a game never switches to a different followed lineage after the operator re-points the source, and never to an older generation.
- Still only to an already-built generation; it never starts or waits for a build on our clock (comment at `:602-604` stays true).

### 3.9 UI

**Settings → Play → "Model — applies to the next game"** (`LichessBot/UI/LichessBotSettingsView.swift:392-442`; the section stays where it is):
- Source picker gains "Follow lineage" (`:398-404`).
- New row `LichessBotFollowedLineageRow` (new file `LichessBot/UI/LichessBotFollowedLineageRow.swift`): the chosen lineage — anchor `model_id`, run ID (first 8 characters, full ID in `.help`), newest file + segment + step + cum step — and a "Choose Lineage…" button. Disabled unless the source is `followLineage`, like the file row (`:406-420`).
  - Shown text comes from the controller's published follow status while the bot is online **and** the draft's `followedLineage` equals the one in force (`controller.settings.model.followedLineage`); the section edits a draft, so a lineage chosen but not yet applied must not show the applied lineage's status. Otherwise the row resolves the draft's choice once in `.task(id: settings.followedLineage)` ("Reading model headers…" while it runs), through an `async` wrapper that runs `ModelFolderHeaderCache.scanSynchronously` on a `DispatchQueue` behind a continuation (the `ModelFileCatalog.scan` pattern, `ModelFileCatalog.swift:84-90`) — never the synchronous scan on the main actor. A failed scan shows its error in the row, never "None". Nothing displayed is stored in settings (single source of truth).
- New row "Check for new files every [60] s" (`LichessBotIntegerField`), enabled only for `followLineage`.
- The mid-game toggle (`:429-430`) becomes "Games in progress switch to each new generation (live trainer, followed lineage)", enabled for either source. Its key is unchanged.

**Line picker, follow mode** (`LichessBot/UI/LichessBotModelLinePicker.swift`):
- New `let purpose: LichessBotModelLinePickerPurpose` (`.chooseFile` / `.chooseLineageToFollow`, new file) and `onChoose: (ModelFileEntry) -> Void` (the one call site, `LichessBotSettingsView.swift:421-425`, uses `entry.url.path`).
- In follow mode only segment rows are selectable, and only when the segment's latest file has a `.recorded` lineage with path kind `replay` or `vsuci`; other rows show why in `.help` ("written before lineage records", "GUI run — use Champion or Live trainer", "unreadable lineage: …"). The footer reads "Follows this segment's run from here on, through every exact resume"; the button reads "Follow This Lineage".
- Choosing stores `LichessBotFollowedLineage(lineageRunID:, anchorSegmentID:)` from the entry's lineage facts — no second header read.

**Overview → Model card** (`LichessBot/UI/LichessBotOverviewView.swift:262-283`): new `LichessBotLineageFollowStatusView` (new file), `.shown(source == .followLineage)`: "following run 783BF744 from Xxvl", "newest <file> · seg 0 · step 31000 · cum 31000", "checked 11:30:02", the outcome in orange (a problem: "still playing generation N") or secondary (keep playing after a deleted file), and a "Check Now" button (OD-12) calling `controller.checkFollowedLineageNow()`.
- New `LichessBotModelSwitchStatusView` (new file), `.shown` while the settings' generation source differs from the playing generation's (§3.10): "Switching to Follow lineage — still playing champion 20260928-1-TEST (generation 3)", plus the last switch failure and its retry time when there is one.
- The card's headline (`:269`) shows `controller.settings.model.source` today, above the playing generation's model ID. During a switch that would label one source's model with the other's name, so the headline shows the playing generation's `sourceKind` when a generation exists (the settings' source only while offline); the switch status view says what it is switching to.

**Going online** (§3.10): while `connection == .connecting`, the status chip and the Overview show the step (`controller.goingOnlineStep`: "Verifying token", "Preparing model: Follow lineage — reading headers / loading <file> / building network"), and **Go Offline** is enabled and cancels going online (today it is disabled then, §1.6).

**Controller** (`LichessBot/App/LichessBotController.swift`): publishes `lineageFollowStatus` next to `generation` in the poll loop (`:2843-2845`), and `modelRefreshFailure` (the last refresh or switch error and its retry time, cleared on success) from the catch at `:2887-2897`; `checkFollowedLineageNow()` runs on the runtime's slots and surfaces errors through the existing `raiseAlarm` path.

### 3.10 Ready before play, for every source (owner rule, 2026-10-06; OD-18, OD-19)

> Owner: "we need to make sure we're ready to play before putting out challenges or receiving."

**Invariant.** While the bot is online (a runtime exists), a model generation exists. It is built before the event stream opens, it is only ever replaced by a newer generation that has finished building, and it is never cleared until the runtime is torn down. Therefore no challenge is ever declined, and no game ever starts, for want of a model. The invariant is enforced by types, not by checks (OD-19).

**Slots can't exist without a generation** (`LichessBot/Play/LichessBotModelSlots.swift`):
- `static func prepare(for settings: LichessBotModelSettings, provider:time:folderScanner:log:) async throws -> LichessBotModelSlots` builds the first generation through the same build code every refresh uses, then returns slots holding it. The builder is split out of the actor as a `Sendable` `LichessBotGenerationBuilder` (provider, scanner, loader, `InferenceNetworkFactory`), so `prepare` and the actor share one build path. The actor's `init` becomes private, taking the built first generation.
- `current` becomes non-optional (`LichessBotModelGeneration`, today optional at `:85`). `lastSnapshotAt` and `currentSettings` become non-optional with it.
- `ready(for:)` (`:102-107`) and `sourceAvailable(for:)` (`:152-161`) are **removed**. The session manager reads `await slots.current` for a new game: never a build, never a wait, never a throw (E15 holds trivially).
- `refreshIfDue(for:)` (`:127-142`) also does **source switching**: when the settings select a different generation source than the playing generation's, it builds the new source's generation (reason `source changed to <kind>`) and publishes it on success. In P-ready "different" is today's whole-settings comparison (`currentSettings != settings`); P0 narrows it to `generationSource` (§3.4), so P0's regression tests fail before P0. While that build runs, and after it fails, `current` stays the old generation, so new games start on it (and record that, honestly, as their generation's source). Its errors go to the controller's existing backoff and alarm (`LichessBotController.swift:2887-2897`) and to `modelRefreshFailure` (§3.9), and it is retried at the backoff's cadence.
- The build runs inside the poll loop's `await slots.refreshIfDue` (`:2882`), as a live-trainer rebuild does today: games, challenge answers and the manager are unaffected (they read `slots.current`, which an actor suspended on the build still serves), but the poll loop's other work (`:2839-2903`: gate snapshot, published generation, challenge time-outs, the queue pump, matchmaking) waits for the build. That is "in the background" for play, not for the poll loop.
- `LichessBotModelSettings.generationSource` (the value type of §3.4) is introduced here in P-ready for the switch status view only (§3.9): the view compares the applied settings' `generationSource` with the playing generation's (kind, plus `filePath` for `.file`; P3 adds `lineage?.followed` for `.followLineage`).
- Any change of `settings.model` seen by the poll loop sets its `nextModelRefreshAt` to now — the existing reset at `:2874-2878`, generalized from "after a failure" to "always" (the loop keeps the last `settings.model` it saw) — so a switch starts at the next poll tick (the loop wakes every second, `LichessBotController.swift:2905-2909`) instead of up to 15 s later. `refreshIfDue` does nothing when no switch or refresh is due, so a non-switching edit costs nothing.
- Mid-game refresh (§3.8) never crosses sources: a game keeps its source's generations.
- The other sources follow the same "keep playing + alarm" rule once online. Champion: `refreshIfDue` throws `noChampion` when `championModelID()` is nil (today it does nothing, `:130-134`). Live trainer: the rebuild already throws `noTrainer`. Either way the last generation keeps playing. Trainer snapshot and file: pinned, unchanged.
- `LichessBotModelProvider.trainerAvailable()` (`:19`) had one caller, `sourceAvailable` (`:157`); it is removed from the protocol and from `LichessBotSessionModelProvider` (`LichessBot/App/LichessBotSessionModelProvider.swift:52-54`). The test fakes' own `trainerAvailable()` methods stay as unused members (no test edit).

**Going online** (`LichessBot/App/LichessBotController.swift`):
- In `startRuntime` (`:2559-2708`), right after the token and account checks (`:2580-2591`) and before anything that talks to Lichess's streams: `let slots = try await LichessBotModelSlots.prepare(for: settings.model, …)` replaces the plain construction at `:2600-2602`. The manager, reconciler, event-stream handler, `manager.run()` (`:2682`), the poll loop (matchmaking and the challenge queue run from it, `:2901-2903`) and launch recovery all start after it. Sends need `.online` (`:1657`, `:1881`), which is set only after `startRuntime` returns (`:816`), so the operator cannot send one during the build either.
- Order rationale: token and account first (fast, and a bad token should say so without a model build), then the model, then the streams.
- **Settings edited while preparing reach the runtime.** `startRuntime` captures `self.settings` at its start (`:2538`) and creates the runtime's `settingsBox` from that copy only after the account check (`:2597-2598`); `apply` writes `settingsBox?.value` (`:620`), which is nil until then. Today the window is the token and account requests; with `prepare` it becomes the whole model build (seconds; a first follow-lineage scan plus a load). An edit applied in it would be shown by the UI but never reach the running bot's challenge policy, chat or mid-game refresh. So, after `prepare` returns (no suspension in between), the box is created from the **current** `self.settings`, not the copy taken at the start, and the rest of the runtime construction (`preventSleepWhileOnline`, …) reads that same value. `prepare` built the copy's source; if the source changed meanwhile, the first poll switches (above).
- `LichessBotControllerServices` (`:34-50`) gains `var makeModelFolderScanner: @Sendable () -> any LichessBotModelFolderScanning`, defaulting to `LichessBotModelsFolderScanner(directory: CheckpointPaths.modelsDir)` exactly as `replyToTerminate` (`:42-44`) defaults to the app's answer, so every existing `LichessBotControllerServices(makeTransport:readToken:)` in the tests compiles unchanged and a controller test of the follow-lineage source scans a temporary folder, never the real `Models/`.
- A failed build throws out of `startRuntime`; `goOnline`'s existing catch (`:822-829`) tears down, releases the instance lock (`:2552-2555`), sets `.error(<the build's error>)` and logs `[ALARM] LICHESS-BOT going online failed: …`. Nothing was accepted, declined or sent.
- Logged before `[LICHESS-BOT] online`: the existing generation-ready line (`[LICHESS-BOT] model generation 1 ready: <source> … (going online)`).
- Games left running from the last run stay unresumed while the bot is offline with the error, exactly as when going online fails for any other reason today; the status chip already reports them (`LichessBot/UI/LichessBotStatusChip.swift:64`). For that report to survive, `prepare` runs **before** `startRuntime` clears `leftoverGamesFromLastRun` / `finishedGamesAwaitingFilingFromLastRun` (`LichessBotController.swift:2593-2594`, "going online resumes … whatever the last run left") — i.e. between the account check (`:2587-2591`) and that clearing.

**The UI while building.** `connection` stays `.connecting` (no new `ConnectionState` case, so nothing that switches on it changes). A new published `goingOnlineStep: LichessBotGoingOnlineStep?` (`.verifyingAccount`, `.preparingModel(LichessBotModelSourceKind, detail: String)`, nil otherwise) drives the text in §3.9. The build reports its detail (reading headers, loading `<file>`, building the network) through a `@Sendable` progress closure hopping to the main actor. Separate `Task { @MainActor in … }` hops are not ordered with each other or with the end of going online, so each update carries the going-online attempt's `runtimeGeneration` and is applied only while that attempt is still `.connecting`; `goingOnlineStep` is set to nil on every exit from `goOnline` (online, error, cancelled, shut down).

**Cancellation.** `goOffline()` while `.connecting` sets `goingOnlineCancelRequested` (today it returns at once, `:1021`). The two Go Offline buttons become enabled while connecting: `.disabled(!controller.isRunning && controller.connection != .connecting)` (`LichessBotOverviewView.swift:87`, `LichessBotStatusChip.swift:47`). `isRunning` itself is unchanged — its other uses (the chip's Go Online, `LichessBotStatusChip.swift:39`; the live view's empty text, `LichessBotLiveView.swift:64`; the account field, `LichessBotAccountSettingsSection.swift:62-63`) keep today's meaning. `startRuntime` checks the flag after each suspension (token, account, `prepare`) and throws `LichessBotControllerError.goingOnlineCancelled`; `goOnline`'s catch maps that one error to `.offline` (not `.error`) and logs `[LICHESS-BOT] going online cancelled by the operator (<step>)`. A build already running on its `DispatchQueue` cannot be interrupted; it finishes and its generation is discarded.
- `connection` stays `.connecting` until the step in flight returns (the chip and Overview show "Cancelling…"), never `.offline` at once: `goOnline` is still suspended inside `startRuntime` holding the instance lock, and a second Go Online accepted meanwhile would race it for that lock. A second Go Offline press is a no-op.
- `goingOnlineCancelRequested` is cleared at the start of every `goOnline` and on each of its exits, so a cancel never leaks into the next attempt.
- `startRuntime` still suspends after `prepare` (`manager.setOneGameMode`, `setRateLimitHold`, `seedDailyCounts`, `:2645-2651`), so checking only after token, account and `prepare` would lose a press made there and go online anyway. The flag is checked after **every** suspension, the last one being `seedDailyCounts` (`:2651`) — before the event-stream handler, `manager.run()`, the reconciler, the poll loop and launch recovery are started (`:2673-2701`) and before `runtime` is assigned. From that check to `.online` (`:816`) nothing suspends (both run on the main actor), so a cancel is either honoured before any stream opens or arrives after the bot is online, where Go Offline works as today.
- Quitting while connecting: `performShutdown` closes the journal and file queues (`:1209-1210`) while `startRuntime` may still be inside `prepare`, and today's check (`:805-812`) runs only after `startRuntime` builds a runtime on the closed queues. Every one of those checks therefore also tests `isShutDown` and throws, and `goOnline`'s catch keeps today's shut-down outcome (`.error("The bot has shut down")`, `:809`).

**What becomes unreachable, and is removed** (the owner asked for the decline path to be kept only if a genuine case remains — none does):
- `LichessBotChallengeContext.modelReady` and the policy's `decline(.later, rule: "model not ready")` (`LichessBotChallengePolicy.swift:18-19`, `:53-55`): removed. Declines for `later` remain for not accepting (draining, 429 hold, `:50-52`).
- The accept path's build-and-decline (`LichessBotSessionManager.swift:653-665`): removed, and the accept path does **not** read the slots at all — it never used the generation (`_ = try await slots.ready(…)`, `:659`); the game reads `slots.current` at `gameStart`. With the build gone there is no suspension between the decision (`:646`) and `accept` (`:689`), so the two re-checks that existed only because the build suspended the actor — "stopped accepting new games while the model was prepared" (`:666-673`) and "the concurrent-game limit filled while the model was prepared" (`:674-687`) — are unreachable and are removed, along with the duplicate hold (`:657` and `:688` become one). The slot is still held from the decision until `gameStart` as today. (Reading `slots.current` here instead would keep a pointless actor hop and keep those re-checks reachable under rule texts that no longer describe anything.) This removes the behavior `LichessBotGroupBFixTests.testAcceptRechecksAcceptingAfterTheModelBuild` pins (§5.1).
- The `gameStart` path's "no model is available" branch (`:733-744`): becomes "read `slots.current`".
- `LichessBotGameSession`'s `latestMoveSource` becomes non-optional (`:45`, `:148`), and the "mid-game refresh: no model generation is built" anomaly (`:612-614`) goes with it.
- Remaining genuine model failures while online, all of which keep playing: a refresh or a source switch that fails (alarm, backoff), and a followed lineage's problem outcomes (alarm, §3.4). None declines anything.

---

## 4. Edge cases and their handling

| Situation | What happens |
|---|---|
| A save in progress | Not visible: staged under a hidden or `.tmp` name and renamed in (§1.4). |
| File copied in by Finder / `cp` / `rsync` | Header may be readable before the data is complete; the load's `content_sha256` check fails, the file is marked failed until its size or date changes, then retried. |
| File removed between check and load | Load fails (`readFailed`); marked failed; the next check no longer lists it. The playing generation is unaffected. |
| Rolling `--out-model` replaced between check and load | Loaded file verified to be the same lineage and not older; recorded as loaded; logged. |
| Rolling file and step copy of one save | One candidate (same `content_sha256`). |
| Run without `--enumerate-checkpoints` | The rolling file alone is followed; each save is a new inode, so it is re-read. |
| Run ends | Its newest file stays newest; the bot keeps playing it. Not a fallback: it is still the lineage's newest. |
| Newest file deleted (cleanup) | Never backward: keep the playing generation; log. |
| Every file of the lineage gone, or `Models/` unreadable | While online: keep playing the last good generation for new and in-progress games; alarm (status line + log), retried at the backoff's cadence; no declines (OD-8 as amended). Before going online: the first build fails and the bot stays offline with the error. |
| Exact resume of the run (new process, new `model_id`) | Followed automatically (its chain contains the anchor). |
| Branch from a file of the run (`--start-model` without `--resume-exact`) | A new run; not followed (OD-10). |
| Two resumes of one file after the anchor | `.fork`: keep playing the last good generation, alarm naming both segments; the operator re-anchors on one (OD-9 as amended). Before going online: stays offline with the error. |
| Resume of the parent from an earlier file, after the parent stopped | The new segment ranks above the old one by chain depth, even at a lower cumulative step; logged once (OD-17). |
| Second process `--resume-exact`-ed from a step file of a run that keeps training | `.fork` (§3.3 step 2b): keep playing the last good generation, alarm naming both; the operator anchors on one (OD-17, fork as amended OD-9). |
| Settings saved with a followed lineage, next launch | Loads (store fix, §3.1); without the fix the whole saved settings would be "unreadable". |
| `cum_trainer_step` null (run continues unrecorded history) | Ranking does not use it; display shows "cum —". |
| File written before lineage records (format < 7) | Not followable; counted as excluded; not selectable in follow mode. |
| GUI run's files in `Models/` | Not followable (OD-4); the in-process sources cover the GUI run. |
| Settings edited while online (interval, toggle) | No rebuild; following continues (P-ready ends the stall, P0 the needless rebuild). |
| Newest file fails to load while online | Keep playing the current generation for new and in-progress games; one alarm and a log line for that file; retried only when the file changes (OD-6 as amended). |
| Source changed while online (any source to any source) | The new source's generation builds in the background; new games keep starting on the old one until it is ready; a failed build keeps the old one, alarms and retries; never a decline (§3.10). |
| The model can't be built when going online (any source) | The bot stays offline in Error with the build's error; nothing is accepted, declined or sent (§3.10). |
| Operator presses Go Offline while the model builds | Going online is cancelled: "Cancelling…" until the build step returns, then Offline; the finished build is discarded (§3.10). |
| Settings applied while the model builds | Reach the runtime: its settings are taken after the build; a changed source is switched to at the first poll (§3.10). |
| Champion disappears / trainer gone while online | Keep playing the last generation; alarm via the failed refresh (§3.10). |

---

## 5. Tests (new files; existing tests change only where §5.1 lists, under OD-18)

Real model files are written with `SafetensorsModelIO.encode(…, lineage:)` (`Persistence/SafetensorsModelIO.swift:115-123`), as `DrewsChessMachineTests/CheckpointManagerSafetensorsTests.swift:246-250` does, with records from `LineageTracker` (`.fresh`, `.resume(parent:gaps:legacyTotals:)`, `.branch`) so chains are real, not hand-built. Temporary folders as in `ModelFileCatalogTests.swift:8-18`.

**P-ready — ready before play (§3.10), written first and shown failing before the change where they can compile against today's API:**
- `DrewsChessMachineTests/LichessBotGoOnlineModelFirstTests.swift` (controller, fake transport, shared fakes as the existing go-online tests use them):
  - `testGoingOnlineBuildsTheModelBeforeOpeningTheEventStream` — the fake transport records the provider's `snapshotCount` at the moment the event-stream request reaches it: 1. (Fails today: the stream opens with no generation, count 0.) The slots' log lines go to `SessionLogger.shared`, which this test does not capture; the order of the generation-ready line and `online` is checked live (§6, check 9).
  - `testModelBuildFailureLeavesTheBotOfflineWithTheError` — champion source, `LichessBotFakeModelProvider(snapshot: nil)`: `connection == .error(…No champion…)`, no event-stream request, no challenge accepted, declined or sent, instance lock released. (Fails today: the bot goes online.)
  - `testModelBuildFailureKeepsTheLeftoverGamesReport` — a leftover journal is still reported after the failed attempt.
  - `testGoOfflineWhileThePreparingModelCancelsGoingOnline` — a provider whose snapshot waits on a test-held latch; `goOffline()` while connecting → still `.connecting` (cancelling) while the latch holds; after the test opens it → `.offline`, no event-stream request, instance lock released, the late build discarded; a following `goOnline()` goes online (the flag did not leak).
  - `testGoOfflineAfterThePreparedModelStillCancels` — the press lands after `prepare` returns, while the leftover-journal read of `seedDailyCounts` is held (a journal queue blocked by the test): → `.offline`, no event-stream request.
  - `testSettingsAppliedWhilePreparingReachTheRuntime` — `updateSettings` with a changed challenge setting while the build is held: once online, the manager's challenge decisions use the new value.
  - `testShutdownWhilePreparingStopsGoingOnline` — `shutdown` while the build is held: after release, `.error("The bot has shut down")`, no runtime, no event-stream request.
  - `testGoingOnlineStepIsReportedWhilePreparing` — `goingOnlineStep == .preparingModel(.champion, …)` while the build waits; nil once online.
  - `testEverySourceIsBuiltBeforeGoingOnline` — champion, trainer snapshot and live trainer through fake providers, file through a temp file; follow lineage through a temp folder given by `LichessBotControllerServices.makeModelFolderScanner` (never the real `Models/`). The source and the services field do not exist until P2/P3, so P-ready writes the four existing sources and P3 adds the follow-lineage case as its own method, `testFollowLineageIsBuiltBeforeGoingOnline`, in this file (not an edit of an existing test).
- `DrewsChessMachineTests/LichessBotModelSlotsPrepareTests.swift`:
  - `testPrepareBuildsTheFirstGeneration` / `testPrepareThrowsWhenTheSourceCannotBuild` (each source).
  - `testSourceChangeKeepsTheOldGenerationUntilTheNewOneIsBuilt` — `refreshIfDue(B)` with B's build held: `current` is still A's generation throughout; after release, B's.
  - `testFailedSourceSwitchKeepsTheOldGenerationAndThrows`.
  - `testChampionGoneWhileOnlineKeepsPlayingAndThrows`.
  - `testIntervalEditKeepsTheLiveTrainerRefreshing` — the stall half of the P0 settings bug, ended by moving switching into `refreshIfDue`.
- `DrewsChessMachineTests/LichessBotReadyBeforePlayManagerTests.swift`:
  - `testChallengeDuringASourceSwitchIsAcceptedOnTheCurrentGeneration` — no decline, no build on the accept path.
  - `testGameStartDuringASourceSwitchPlaysTheCurrentGeneration`.
  - `testNoChallengeIsEverDeclinedForTheModel` — a run of challenges across a failing switch: every decline reason is a policy reason, none `model`.

**P0 — regression tests first (must fail before the fix, pass after, unmodified; written after P-ready, against its API):**
- `DrewsChessMachineTests/LichessBotModelFileLoadTests.swift`
  - `testRecordedHashIsOfTheBytesThatWereDecoded` — the loader reads through a byte-source seam that serves file A's bytes on the first read and file B's on the second (the seam is introduced first with today's two reads, no behavior change; the test then fails; the single-read fix makes it pass).
- `DrewsChessMachineTests/LichessBotModelSlotsSettingsChangeTests.swift`
  - `testIntervalEditDoesNotRebuild` — fake provider + `LichessBotManualTime` (`DrewsChessMachineTests/LichessBotGameSessionTests.swift:7`): `prepare(A)`, then `refreshIfDue(A with a new interval)` before it elapses → same generation (after P-ready, before the fix: rebuilt as "source changed").
  - `testTogglingMidGameRefreshDoesNotRebuild` — `prepare(A)`, `refreshIfDue(A with the toggle flipped)` → same generation.
  - `testLeftoverFilePathDoesNotRebuildTheLiveTrainer` — live-trainer settings whose `filePath` changes → same generation.
  - (The stall half of the bug — following stops after an interval edit — ends in P-ready, which moves switching into `refreshIfDue`; `testIntervalEditKeepsTheLiveTrainerRefreshing` in `LichessBotModelSlotsPrepareTests` pins it.)

**P1 — `DrewsChessMachineTests/ModelLineageTipTests.swift`** (pure selection):
- `testNewestIsTheHighestLocalStepInOneSegment`
- `testFilenamesAndModificationDatesNeverDecide` (names and dates reversed against the metadata)
- `testLaterSegmentOutranksAHigherCumulativeStepInItsParent` (resume from an earlier file)
- `testExactResumeIsFollowedAcrossANewModelID`
- `testBranchIsNotFollowed`
- `testFilesBeforeTheAnchorSegmentAreNotCandidates`
- `testForkAfterTheAnchorIsReportedWithBothSegments`
- `testForkBeforeTheAnchorIsNotAFork`
- `testByteIdenticalFilesAreOneCandidate` (rolling + step copy)
- `testDistinctWeightsAtOneRankAreAConflict`
- `testNeverSelectsBelowThePlayingGeneration`
- `testNullCumulativeStepStillRanks`
- `testExcludedFilesAreCountedByReason` (unrecorded, GUI path kind, no content hash, unreadable record)
- `testNoCandidatesIsNoFiles`
- `testParentThatKeptTrainingAfterAResumeIsAFork` (child resumed at step h; a parent file beyond h recorded after the child started)
- `testResumeFromAnEarlierFileAfterTheParentStoppedIsFollowed` (parent files beyond h all recorded before the child started; logged once)
- `testCandidateNotPrefixOrderedWithThePlayingGenerationIsAFork`

**P1 — `DrewsChessMachineTests/ModelFolderHeaderCacheTests.swift`** (filesystem):
- `testStagingFilesAreNeverListed` (`.<name>.<UUID>.tmp`, `<name>.safetensors.tmp`)
- `testUnchangedFilesAreNotReadAgain` (`headersRead == 0` on the second scan)
- `testAFileRenamedOverTheSamePathIsReadAgain` (new inode, same size, same second — written with `FileSafety.replaceRegularFile`)
- `testRemovedFileLeavesTheScan`
- `testTruncatedFileIsUnreadableUntilItChanges`
- `testUnlistableFolderThrows` (never "empty")
- `testCatalogScanMatchesTheUncachedScan` (same lines and unreadable list as `ModelFileCatalog.scanSynchronously` on the same folder)
- `testNonRegularItemsAreListedUnreadableNotDropped` (a folder named `x.safetensors`, a dangling link)
- `testLinkTargetChangeIsReadAgain` (a symbolic link whose target is replaced: the cache keys on what the link resolves to)
- `testLineageFactsComeFromTheRecordNotTheMirrorKeys` (mirror keys edited to disagree; facts follow the JSON)

**P2 — `DrewsChessMachineTests/LichessBotFollowLineageSettingsTests.swift`:**
- `testTodaysSaveWithoutTheNewFieldsLoads` — a literal JSON fixture of a settings blob as this build writes it today (all four sources' fields, no new keys): loads; `followedLineage == nil`; `filledFromDefaults == ["model.lineageCheckIntervalSeconds"]`; `ignoredSavedKeys == []`.
- `testFollowLineageRoundTrips`
- `testFollowLineageWithoutALineageIsInvalid` and `testCheckIntervalBelowTheMinimumIsInvalid` — both shown on the Play tab via `LichessBotSettingsTab.tabsWithProblems`.
- `testMinimumCheckIntervalIsThePollInterval` (pins the single source).
- `testSavedFollowedLineageLoadsThroughTheStore` — regression test for the store bug (§3.1): save settings with `followedLineage` set through `LichessBotSettingsStore.save` into a private suite (`makeTemporaryDefaults()`), then `loadReporting` → the same settings, `ignoredSavedKeys == []`. Written first; fails today with `.unreadable` (`keyNotFound`); passes after the `isKnownOptionalKey` fix, unmodified.
- `testAnUnrelatedMissingKeyIsStillUnreadable` — a `keyNotFound` at a path other than the probed one still fails the load (the fix is no catch-all).

**P3 — `DrewsChessMachineTests/LichessBotFollowLineageSlotsTests.swift`** (scripted `LichessBotModelFolderScanning` over real files in a temp folder, manual time):
- `testPrepareBuildsTheNewestFile`
- `testCheckRunsOnlyAfterTheInterval` (scan count by manual time)
- `testNewerFileRebuildsAndOlderDoesNot`
- `testSameContentDoesNotRebuild`
- `testNoFilesKeepsPlayingAndThrowsOnRefresh` / `testFolderUnreadableKeepsPlayingAndThrowsOnRefresh` / `testForkKeepsPlayingAndThrowsOnRefresh` (the playing generation is unchanged; every check throws)
- `testPrepareFailsWhenTheLineageHasNoFiles` / `…WhenTheFolderIsUnreadable` / `…OnAFork` / `…WhenTheNewestFileFailsToLoad`
- `testProblemClearingLogsAvailableAgain`
- `testRunEndKeepsTheLastFileAvailable`
- `testFailedNewestKeepsPlayingAndIsNotRetriedUntilItChanges`
- `testFailedNewestThrowsOnceThenReportsWithoutThrowing`
- `testReplacedRollingFileLoadsTheNewerSameLineageFile`
- `testFileOfAnotherLineageAtTheSamePathIsRefused`
- `testGenerationRecordsLineagePosition` (and the `.file` source records it too)
- `testConcurrentCallersShareOneScan`
- `testNetworkBuildFailureDoesNotMarkTheFileFailed`
- `testSiblingSegmentInTheRollingPathFailsVerification`
- `testInFlightBuildIsJoinedAcrossAnIntervalEdit` (two callers differing only in the interval → one build)
- `testOldGenerationInfoJSONStillDecodes` (a game record's generation JSON without `lineage`)

**P4 — `DrewsChessMachineTests/LichessBotFollowLineageMidGameTests.swift`** (harness from the shared fakes `LichessBotFakeGameServer`, `LichessBotManualTime`, `LichessBotRecordingGameObserver` in `LichessBotGameSessionTests.swift:7`, `:85`, `:428`):
- `testFollowedLineageGameSwitchesToANewerGenerationOfTheSameLineage`
- `testGameNeverSwitchesToAnotherFollowedLineage`
- `testGameNeverSwitchesWithMidGameRefreshOff`

**P5 — `DrewsChessMachineTests/LichessBotFollowLineageViewRenderTests.swift`:** the settings Play tab with each source, the follow status view in every outcome, the picker in follow mode with a non-followable row disabled.

**Existing suites that guard the change** (run; unmodified except the edits §5.1 lists): `ModelFileCatalogTests`, `ModelFileCatalogActivityOrderTests`, `ModelLineageTreeTests`, `LichessBotSettingsStoreTests`, `LichessBotSettingsStoreEncodingTests`, `LichessBotSettingsTabTests`, `LichessBotSettingsViewRenderTests`, `LichessBotGameSessionTests`, `LichessBotGameSessionFaultTests`, `LichessBotSessionManagerTests`, `LichessBotPhase5CoreTests`, `LichessBotDataLayerTests`, `LineageRecordTests`, `CheckpointManagerSafetensorsTests`.

### 5.1 Existing-test edits forced by the ready-before-play rule (OD-18; the by-construction items depend on OD-19)

Each edit only adapts a test to the new invariant (a generation exists before the bot is online; slots are built by `prepare`). None loosens an assertion about anything else. Line numbers are at `f5be524b`. **Approval:** the owner approved "the existing-test edits this forces" under OD-18; the second review (§11) found the list incomplete and corrected it. The two **deletions** and the provider swaps it added (marked *added in review*) need the owner's express approval before P-ready starts, and the deletions and the slot-harness rewrites exist only because of OD-19's by-construction choice, which has no recorded decision yet (§10).

- **Controller tests that go online with a provider that can build nothing** (`LichessBotFakeModelProvider(snapshot: nil)`): going online would now stop in Error. Each switches that provider to `try await LichessBotFakeModelProvider.randomChampion()` (`DrewsChessMachineTests/LichessBotSessionManagerTests.swift:103-113`), the fake the other go-online tests already use:
  - `LichessBotBotLimitTests.swift:41`
  - `LichessBotCasualFallbackTests.swift:62`
  - `LichessBotChallengeQueueControllerTests.swift:148` (the harness's default provider; `:233` and `:327` already pass `randomChampion()`)
  - `LichessBotLeftoverJournalReportTests.swift:113` (goes online at `:116`) and `:140` (goes online at `:146`); `:103` and `:126` never go online and stay as they are
  - `LichessBotOnlineBotsRaceTests.swift:70`
  - `LichessBotNotesAtGoOnlineTests.swift:47`
  - `LichessBotPollObservationTests.swift:81`
  - `LichessBotOfflineQuitTests.swift:36`
  - `LichessBotQuitWithdrawalTests.swift:106`, `:136`
  - (Not `LichessBotShutdownTests.swift:67`: that file's one `goOnline` (`:197`) runs after `shutdown` and is refused by the `isShutDown` guard (`LichessBotController.swift:781-785`) before any model work, so its assertions hold with the unbuildable provider. *Removed in review.*)
- **Manager / slot harnesses that construct `LichessBotModelSlots(provider:time:log:)` directly** — that initializer no longer exists; each uses `try await LichessBotModelSlots.prepare(for: settings.model, provider: …, time: …, folderScanner: …, log: …)` with a buildable provider (`randomChampion()`), and a synchronous harness becomes `async` (its call sites gain `await`):
  - `LichessBotSessionManagerTests.swift:157-175` (`makeHarness`; default provider `:159` becomes `randomChampion()`; call sites `:218`, `:236`, `:246`, `:260`, `:280`, `:312`, `:339`, `:358`, `:371`, `:382`). *Added in review:* `testDeclinesWhatDCMCannotPlay` (`:217`) and `testDrainingDeclinesLater` (`:245`) pass `unbuildableChampion()` explicitly, which `prepare` cannot build (empty weights), so both become `try await LichessBotFakeModelProvider.randomChampion()`.
  - `LichessBotPhase5CoreTests.swift:149-158` (`makeManager`; call sites `:195`, `:238`; `:197` passes `unbuildableChampion()` and becomes `randomChampion()`)
  - `LichessBotMultipleChallengesTests.swift:28` (`unbuildableChampion()` → `randomChampion()`); `:76` already buildable
  - `LichessBotReviewFixTests.swift:111`, `:213`, `:248` (`unbuildableChampion()` → `randomChampion()`), `:179`
  - `LichessBotGroupBFixTests.swift:61-69` (`makeHarness`; call sites `:109`, `:132` pass `unbuildableChampion()` → `randomChampion()`). The third call site, `:200`, is not adapted: its test is deleted (below).
- **Assertions about the removed behavior:**
  - `LichessBotSessionManagerTests.testDeclinesWhatDCMCannotPlay` (`:217-233`): `XCTAssertEqual(provider.snapshotCount.value, 0, "declines never build a network")` becomes `1` with the message "declines build nothing beyond the generation prepared before going online".
  - `LichessBotSessionManagerTests.testDeclinesLaterWithoutAModel` (`:235-242`): **deleted** — it pins the decline this rule removes; `LichessBotReadyBeforePlayManagerTests.testNoChallengeIsEverDeclinedForTheModel` replaces it.
  - `LichessBotPolicyTests.swift`: the context builder's `modelReady` parameter and argument (`:53`, `:62`) are removed with the field; `testDrainingAndModelNotReadyDeclineLater` (`:100-103`) loses its `modelReady: false` assertion (`:102`) and is renamed `testDrainingDeclinesLater`. (Removing an assertion: listed with the deletions for approval.)
  - `LichessBotGroupBFixTests.testAcceptRechecksAcceptingAfterTheModelBuild` (`:197-219`) and its private `GatedModelProvider` (`:168-195`, used by nothing else): **deleted** — *added in review.* It holds the champion snapshot on a latch until a challenge has arrived, then stops accepting and expects the post-build re-check to decline `later`. Under P-ready the snapshot is taken by `prepare` inside the harness, before the event stream exists, so the harness would wait on a latch the test opens only later (a hang), and the re-check it pins is removed as unreachable (§3.10, accept path). Nothing replaces it, because the behavior no longer exists; `LichessBotReadyBeforePlayManagerTests.testChallengeDuringASourceSwitchIsAcceptedOnTheCurrentGeneration` pins that no build sits on the accept path. The file's doc comment (`:4-6`, "the accept re-check after a model build") and the `// MARK: - Accept re-check after the model build` line (`:166`) go with it.
- **Game-session harness:** `LichessBotGameSessionFaultTests.swift:166` — the parameter's type changes with the initializer's, `@escaping @Sendable () async -> (any LichessBotMoveSource)?` → `… -> any LichessBotMoveSource`, and its default `{ nil }` (which no longer type-checks) becomes `{ LichessBotScriptedMoveSource() }` (`LichessBotGameSessionTests.swift:390`). Only the two tests with mid-game refresh on ever call it (`:299`, `:318`), and both pass their own closure, which already returns a non-optional source. The other sessions built in tests (`LichessBotMovePacingTests.swift:19`, `LichessBotDataLayerTests.swift:355`, `LichessBotGameSessionTests.swift:515`, `LichessBotReviewFixTests.swift:21`) pass `{ source }` and compile unchanged.
- **Not edited:** `LichessBotSessionManagerTests.swift:296` (`snapshotCount == 1`, "one snapshot serves both the acceptance and the game") still holds — the one snapshot is now the prepared one. Its doc comment (`:274-275`, "Accepting builds the model generation first (E15)") becomes inaccurate; it is left as it is unless the owner approves a comment edit.
- **Left unused, not deleted:** `LichessBotFakeModelProvider.unbuildableChampion()` (`LichessBotSessionManagerTests.swift:114-123`) has no caller after the swaps above; removing it is a test-file deletion and needs the owner's approval like the rest.
- **Exact list for approval.** Deletions: `LichessBotSessionManagerTests.testDeclinesLaterWithoutAModel`; `LichessBotGroupBFixTests.testAcceptRechecksAcceptingAfterTheModelBuild` with `GatedModelProvider`, its MARK and the doc-comment clause; the `modelReady: false` assertion of `LichessBotPolicyTests.testDrainingAndModelNotReadyDeclineLater` (plus its rename). Edits: the provider swaps (eleven go-online sites in nine files; nine `unbuildableChampion()` → `randomChampion()` swaps at `LichessBotSessionManagerTests.swift:217`, `:245`, `LichessBotPhase5CoreTests.swift:197`, `LichessBotMultipleChallengesTests.swift:28`, `LichessBotReviewFixTests.swift:111`, `:213`, `:248`, `LichessBotGroupBFixTests.swift:109`, `:132`), the slot constructions rewritten to `prepare` (nine sites in five files: three harnesses and six inline constructions), the `snapshotCount` 0 → 1 change, the `LichessBotPolicyTests` context-builder change and the `LichessBotGameSessionFaultTests` parameter type.

**Further existing-test edits the implementation needed** (approved by the owner on 2026-10-06 on condition that each is listed here with its reason; none weakens an assertion, none deletes a test):
- `LichessBotChallengeQueueControllerTests.makeOnlineController` and `LichessBotSessionManagerTests.makeHarness`: their provider parameter's default (`LichessBotFakeModelProvider(snapshot: nil)`) is removed rather than swapped, because a default argument cannot `await` `randomChampion()`. The call sites that relied on it now pass `try await LichessBotFakeModelProvider.randomChampion()` (four in each file: `LichessBotChallengeQueueControllerTests` `:206`, `:258`, `:277`, `:304`; `LichessBotSessionManagerTests` `testChallengeResponseBudget` and the three takeover tests).
- `LichessBotNotesAtGoOnlineTests.makeController` and `LichessBotOfflineQuitTests.makeController` were synchronous, so the swap made them `async` and their call sites `try await` (two each).
- `LichessBotQuitWithdrawalTests` (`:106`, `:136`): the controller factory closure is synchronous, so the provider is made just before it (`let model = try await …randomChampion()`) and the closure passes `model`.
- The inline slot constructions in `LichessBotMultipleChallengesTests` and `LichessBotReviewFixTests` prepare for `LichessBotModelSettings.testBaseline()` (the champion source their fake serves); the harnesses with a settings value in scope prepare for its `model`.

---

## 6. Validation

**No waiting on training runs** (owner, 2026-10-06: work does not wait for training runs; slowing them is acceptable). Builds, targeted test runs, the full suite and the live check run while the replay runs train. The one constraint: nothing touches the running trainers' processes or files.
- Tests write only into their own temporary folders (`ModelFileCatalogTests.swift:8-18` pattern) and never into `Models/` or `Sessions/`.
- The live check only **reads** `Models/` (header reads, one full read per loaded generation); it never renames, moves, deletes, copies over or `touch`es a file there, and never signals a trainer process.
- Launching the GUI runs the orphan-staging sweep over `Models/` (`CheckpointPaths.cleanupOrphans`); it removes only staging debris unmodified for `CheckpointPaths.orphanStagingMinimumAge` (`Persistence/CheckpointManager.swift:239`), far longer than any live save stages, identity-checked — so it cannot touch a running trainer's in-flight save.
- Expect slower tests and training while both run; that is accepted.

**Per phase:**
- Build succeeds.
- The phase's new tests pass, run with `-only-testing:DrewsChessMachineTests/<Class>`; P0's regression tests are shown failing before their fix and passing after, unmodified.
- The guarding suites above pass.

**Before merge:** the full suite (the change touches persistence: `ModelFileCatalog`, `CheckpointManager.loadModelFile`).

**Live check against a running replay (alongside it):**
1. Pick a live run, e.g. `20261005-lrBleaky-cyc1` (run `783BF744-5FCB-4869-BBE8-7126EDD0C72B`, `model_id` `20261006-6-Xxvl`). Record its newest step file's `model_id`, `training_step`, `content_sha256` and `dcm_lineage.run` from the header with a short script — the expected answer, taken independently of the app.
2. Settings → Follow lineage → Choose Lineage… → the `Xxvl` segment row. Go online in one-game mode.
3. `[LICHESS-BOT] lineage follow:` names that file (or its rolling copy — same `sha=`), `seg=0`, the same step; `[LICHESS-BOT] model generation 1 ready: followLineage 20261006-6-Xxvl …` follows. The Overview card shows the same.
4. Wait for the run's next `[REPLAY] enumerated checkpoint ->` (about 33–36 minutes). Within one check interval of the file appearing, a `lineage follow:` line names the new step and generation 2 is built; a game in progress with mid-game refresh on records both generations in its record.
5. Confirm no `headers read` above the number of new files on later checks (the cache works), and `scan ms` stays in tens of milliseconds.
6. Rename-free negative checks in a scratch folder copy are covered by tests; on the live folder, do **not** delete, move or write files.
6a. Chain following on real files (header-only, no app needed for the expected answer): anchor on segment 0 of a finished two-segment run on disk — e.g. run `9249756B-…` (`20261005-24-qx7x`, segment 0, handed off at step 36000 → `20261005-28-gGnA`, segment 1) — and confirm the follow status names segment 1's newest file, matching a script's reading of the headers. On 2026-10-06 four such runs exist; in every one the child resumed from the parent's newest file, so §3.3 step 2b reports no fork.
7. Confirm a finished game's record (`generations[].lineage`) and PGN `DCMSources` show `followLineage`, the run, segment and step.

**Live check of ready-before-play (any source; read-only toward `Models/`):**
8. Source = Model file with a path that does not exist (set in Settings; the file is never created). Go Online → the status shows "Preparing model…", then **Error** with the loader's message; the protocol log has no event-stream request and no challenge line; the session log has `[ALARM] LICHESS-BOT going online failed: …` and no `[LICHESS-BOT] online`.
9. Source = Follow lineage (valid). Go Online → in the session log, `[LICHESS-BOT] model generation 1 ready: followLineage … (going online)` comes **before** `[LICHESS-BOT] online`, and the protocol log's first event-stream request comes after it.
10. While online with a game in progress, switch the source to Champion and back: the Overview shows "Switching to …" until each build finishes; new challenges during the switch are accepted (never declined `later` for the model); the game in progress keeps its generation.
11. Go Online, then press Go Offline while "Preparing model" shows → "Cancelling…" until the build step returns, then **Offline** (not Error), `[LICHESS-BOT] going online cancelled by the operator …`, no event-stream request; Go Online right after goes online normally.
12. `grep -c "model not ready" ` over the session and protocol logs of the whole check → 0.

**Done when:** all phases' tests pass (including every §5.1-edited test), the full suite passes, the live check matches the independent header reading at two consecutive saves, checks 8–12 behave as written, `LICHESS_BOT_PLAN.md` §9 / §9.2 and `CHANGELOG.md` are updated, and the ROADMAP entry is marked complete.

---

## 7. Phasing (build + commit per phase)

Order: **P-ready, then P0 … P6.** (P-ready was added by the owner decisions of 2026-10-06; the later phases keep their numbers so the references above stay valid.)

- **P-ready — ready before play, every source (§3.10), first:**
  - Add the one-line ROADMAP.md entry under the Lichess bot item (OD-14, approved), worded: `- **Lichess bot: follow a model lineage on disk, and build the model before going online (planned 2026-10-06).** Plan: documentation/plans-active/LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md.` P6 marks it complete.
  - Write the P-ready tests that compile against today's API (`testGoingOnlineBuildsTheModelBeforeOpeningTheEventStream`, `testModelBuildFailureLeavesTheBotOfflineWithTheError`) and show them failing.
  - `LichessBotGenerationBuilder`, `LichessBotModelSlots.prepare`, non-optional `current`, switching in `refreshIfDue`, removal of `ready` / `sourceAvailable` / `modelReady` / the decline and no-model branches, non-optional `latestMoveSource`, removal of the accept path's post-build re-checks and of `LichessBotModelProvider.trainerAvailable()`, `startRuntime` ordering (settings box from the current settings after `prepare`), `goingOnlineStep`, cancellation (checks after every suspension, including `isShutDown`), the Go Offline buttons, `modelRefreshFailure` and the switch status view.
  - The §5.1 test edits (after the owner has approved §5.1's exact list and OD-19), then the remaining P-ready tests. The slots' settings comparison stays whole-settings here (P0 fixes it); `generationSource` is introduced for the switch status view only.
- **P0 — latent bugs** (§3.4 last paragraph, §3.5): byte-source seam → failing tests → single-read load; settings-comparison tests (against the P-ready API) → failing → `generationSource` comparison.
- **P1 — persistence:** `ModelFileEntry` lineage facts and content hash; `ModelFolderHeaderCache`; `ModelFileCatalog` over it; `ModelLineageTip`; tests.
- **P2 — settings:** enum case, `LichessBotFollowedLineage`, new fields, limits (poll cadence moved to `LichessBotLimits`), validation; the store regression test (`testSavedFollowedLineageLoadsThroughTheStore`) shown failing, then the `isKnownOptionalKey` fix (§3.1); tests.
- **P3 — slots:** scanner protocol and production scanner, `LichessBotControllerServices.makeModelFolderScanner` (§3.10), follow status, check/build/refresh (on the P-ready slots: `prepare` for the first build, switching in `refreshIfDue`), `LichessBotGenerationInfo.lineage`, logging, controller publishing; tests.
- **P4 — mid-game refresh** generalization; tests.
- **P5 — UI:** settings rows, picker follow mode, Overview status view, "Check Now"; render tests.
- **P6 — docs:** `LICHESS_BOT_PLAN.md` §9 table row + §9.2 pointing here, and §9's "Unavailable source means no silent fallback … New challenges are declined with `later`" paragraph marked superseded by §3.10 (text kept, with the new rule beside it); `CHANGELOG.md`; this file's status; the ROADMAP entry marked complete with nothing removed from it (OD-14).

---

## 8. Risks

- **First scan cost.** Reading 4.4k headers and decoding their records once per launch of the bot runtime: about 37 MB of reads plus JSON decode, estimated a few seconds, off the actor on a utility queue. It now happens inside "Preparing model" while going online (§3.10), so no challenge waits for it; going online takes that much longer, shown as the step.
- **Going online is slower for every source** by one build: a champion or trainer export plus a network build, or a file read and a network build — the cost the first accepted challenge paid before. Shown as the "Preparing model" step and cancellable.
- **Champion / trainer sources need the GUI session first.** Going online with such a source before a champion or trainer exists now stops in Error ("No champion network exists. Build or load one first.", `LichessBotModelSlots.swift:35`) instead of going online and declining. `autoConnectOnLaunch` is unused today (§1.6), so nothing goes online unattended.
- **Slower tests.** The §5.1 harnesses build a real random network (`randomChampion()`) where they used a fake that built nothing.
- **A failing switch is silent to opponents but loud to the operator:** new games keep starting on the old source's generation until the switch succeeds; the Overview's switch status and the alarm say so. Every move records its generation, so statistics stay attributable.
- **Disk contention with live training.** Header reads are small; a generation load is one 39 MB read per new checkpoint (every ~30 minutes). Negligible beside training I/O.
- **GPU contention.** Building a generation's network while three runs train is today's file-source cost, once per checkpoint.
- **Mid-game strength changes.** With mid-game refresh on, one game can mix checkpoints; every move records its generation, so statistics stay attributable (existing live-trainer behavior).
- **Downgrade** of saved settings (OD-15).
- **`ModelFileCatalog` refactor** touches the picker's only data path; guarded by its existing tests plus `testCatalogScanMatchesTheUncachedScan`.

---

## 9. Non-goals

- Following files outside `Models/` (another `--out-model` folder, train-vs-UCI session folders in `Sessions/`) (OD-5).
- Following across branches or derivations (OD-10).
- Following a GUI run's files (OD-4).
- File-system event watching (§2.2).
- Per-step statistics in the Stats view, new PGN tags, probe-pElo display (§9.1's deferred item).
- A "Re-snapshot now" button for the trainer-snapshot source (`forceRefresh` is unused today, §1.1) — noted, not in scope.
- Any search, opening book or change to move selection.

---

## 10. Owner decisions

- **OD-1 Lineage identity:** run (`lineage_run_id`) + anchor segment (`segment_id`). *Recommend* — handles exact resumes and sibling forks exactly (§2.1).
  - Decided (owner, 2026-10-06): accepted as recommended.
- **OD-2 Check interval:** a new `lineageCheckIntervalSeconds`, default 60 s, minimum = the 15 s poll. *Recommend* over reusing `liveTrainerRefreshIntervalSeconds` (its name and 30 s floor are about SGD lock contention, not disk checks).
  - Decided (owner, 2026-10-06): accepted as recommended.
- **OD-3 Mid-game toggle:** shared, label-only change, Swift name and saved key unchanged. *Recommend* (no rename, no migration, no test edits).
  - Decided (owner, 2026-10-06): accepted as recommended.
- **OD-4 Followable path kinds:** `replay` and `vsuci` only. *Recommend* (GUI clocks rewind on promotion; the GUI has its own sources).
  - Decided (owner, 2026-10-06): accepted as recommended.
- **OD-5 Folder:** `Models/` only. *Recommend* for v1.
  - Decided (owner, 2026-10-06): accepted as recommended.
- **OD-6 Newest file fails to load:** keep the playing generation, don't retry until the file changes, never step back to an older file; with nothing playing, unavailable. *Recommend.*
  - Decided (owner, 2026-10-06): **amended** — a later checkpoint that fails to load: keep playing the current generation for new and in-progress games, no declines; log; retry only when the file changes. There is no "nothing playing" case while online, because of OD-18. Applied in §3.4, §3.7, §4, §5.
- **OD-7 Newest on disk older than the playing generation:** keep playing, log. *Recommend.*
  - Decided (owner, 2026-10-06): accepted.
- **OD-8 No files of the lineage / folder unreadable:** decline new challenges, finish games in progress. *Recommend.*
  - Decided (owner, 2026-10-06): **amended** — while online: keep playing the last good generation + alarm (status line + log), don't decline. Before going online the same condition makes the first build fail, and the bot stays offline with the error. Applied in §3.4, §3.10, §4, §5.
- **OD-9 Fork after the anchor:** unavailable + alarm, operator re-anchors. *Recommend* over auto-picking the newest save (no silent choice).
  - Decided (owner, 2026-10-06): **amended** — while online: keep playing the last good generation + alarm; the operator re-anchors. (Before going online: stays offline with the error.) Applied in §3.4, §4, §5.
- **OD-10 Branches:** not followed. *Recommend.*
  - Decided (owner, 2026-10-06): accepted (don't follow branches).
- **OD-11 Lineage attribution for the fixed-file source too:** *Recommend yes* (same loader, free).
  - Decided (owner, 2026-10-06): accepted.
- **OD-12 "Check Now" button on the Overview:** *Recommend yes.*
  - Decided (owner, 2026-10-06): accepted.
- **OD-13 Fix the two latent bugs in P0** (hash of a second read; settings edits stall refresh): *Recommend yes*, regression tests first. (The settings-store bug of §3.1 is not optional: without its fix the new field makes saved settings unreadable; it is fixed in P2.)
  - Decided (owner, 2026-10-06): accepted. (Sequencing: P-ready runs first and ends the stall half; P0 fixes the needless-rebuild half and the double read, §3.4, §7.)
- **OD-14 ROADMAP:** a one-line pointer under the Lichess bot item to this plan. *Recommend yes* — needs your permission.
  - Decided (owner, 2026-10-06): approved — P-ready (the first phase) adds the one-line ROADMAP.md entry pointing at this plan; P6 marks it complete (§7).
- **OD-15 Downgrade of settings saved with the new source:** accepted as unreadable in older builds. *Recommend accept.* The same holds for game journals, records and the index written with `sourceKind = followLineage` (`LichessBotGenerationInfo.sourceKind` is the enum): an older build cannot decode them either.
  - Decided (owner, 2026-10-06): accepted.
- **OD-16 Architecture change inside a followed lineage:** build and log, don't refuse. *Recommend* (cannot happen for exact resumes; the bot is architecture-agnostic).
  - Decided (owner, 2026-10-06): accepted (build and log).
- **OD-17 A child resumed from an earlier file of its parent (§3.3 step 2):** if the parent kept training after the child started, it is a fork (unavailable, operator re-anchors); if the parent had stopped before the child started (a redo), follow the child and log it once. *Recommend.* Stricter alternative: treat both as a fork. Known limit either way: following the parent past a concurrent child needs a segment-exclusion setting this plan does not add.
  - Decided (owner, 2026-10-06): accepted, but its fork outcome follows amended OD-9: keep playing the last good generation + alarm, operator re-anchors.
- **OD-18 Ready before play, every source (new rule from the owner).** "We need to make sure we're ready to play before putting out challenges or receiving." Going online builds the model generation first for every source (champion, trainer snapshot, live trainer, file, follow lineage); a failed build keeps the bot offline with the error; nothing is accepted, declined-for-model or sent before a generation is ready; a source change while online builds the new generation in the background and keeps playing the old one until it is ready — never a decline for it. The "model not ready" decline path goes unless a genuine case remains.
  - Decided (owner, 2026-10-06): **required.** Designed in §3.10 (going online, the UI while building, cancellation, source change while online), tested in §5 (P-ready) and validated in §6 (checks 8–12). No genuine "model not ready" case remains: while online a generation always exists, so the decline (`LichessBotChallengePolicy.swift:53-55`), the accept path's decline (`LichessBotSessionManager.swift:660-664`), the `gameStart` "no model is available" branch (`:737-743`) and the mid-game "no model generation is built" anomaly (`LichessBotGameSession.swift:612-614`) are removed. The existing-test edits this forces are listed in §5.1 and are owner-approved under this decision.
- **OD-19 How the OD-18 invariant is enforced:** by construction — `LichessBotModelSlots.prepare` returns slots that already hold a generation, `current` is non-optional, and `ready` / `sourceAvailable` / `modelReady` are removed (§3.10). *Recommend.* Alternative: build first in the controller only and keep the manager's lazy build and its decline as unreachable code — fewer test edits (only the go-online provider swaps of §5.1, eleven sites in nine files, and no deletions), but it leaves a decline path for a case that cannot happen, which the owner asked not to keep.
  - **Not yet decided.** Every §5.1 deletion and slot-harness rewrite follows from this choice, so it needs the owner's decision together with the approval of §5.1's exact list before P-ready starts.
  - Decided (owner, 2026-10-06): **(a), by construction** — enforce "model built before going online" by construction: remove `ready(for:)`, `sourceAvailable`, `LichessBotChallengeContext.modelReady`, the "model not ready" decline and the accept path's unreachable re-checks (§3.10).
  - Decided (owner, 2026-10-06): **§5.1 approved in full.** Deletions approved: (1) `LichessBotSessionManagerTests.testDeclinesLaterWithoutAModel`; (2) `LichessBotGroupBFixTests.testAcceptRechecksAcceptingAfterTheModelBuild` with its `GatedModelProvider` helper, its MARK line and the doc-comment clause; (3) the `modelReady: false` assertion of `LichessBotPolicyTests.testDrainingAndModelNotReadyDeclineLater`, with the rename to `testDrainingDeclinesLater`; (4) the now-unused `LichessBotFakeModelProvider.unbuildableChampion()`, and the doc comment at `LichessBotSessionManagerTests.swift:274-275` corrected. Every other §5.1 edit is approved ("go ahead with all the test edits as needed"). Any further existing-test edit the implementation genuinely needs is approved too, on condition that each is listed with its reason (§5.1 / "Implementation decisions"), no assertion is weakened, and no other test is deleted without asking first.

---

## 11. Review (independent, 2026-10-06, against `f5be524b`)

About 45 `file:line` citations were checked against the code; all resolve and say what the plan claims, except the small offsets corrected below. The measured facts were re-taken read-only from `Models/` (4,462 files, 37.3 MB of headers, live run `783BF744…` with `…-latest` and `…-step31000` sharing `content_sha256` `1447ee85e98b…`). Both P0 bugs are real (`LichessBotModelSlots.swift:219-220`; `:103`, `:128`), and their regression tests as written fail before the fix.

Fixes made in this file:
- **Settings store bug (must-fix, §3.1, §5 P2, §7 P2).** A saved `followedLineage` would make the whole saved settings unreadable: `isKnownOptionalKey` does not catch `keyNotFound` from a struct-typed probe (checked with a standalone script of the same shapes). Added the fix, a failing-first regression test and a guard test.
- **Fork check (must-fix, §2.1, §2.3, §3.2, §3.3, §4, OD-17).** Prefix order alone missed a parent that kept training after a child resumed it (an A/B resume off a live run's step file); the child would silently become "newest" by chain depth. Added `handoffs` to the file position, step 2b/2c, tests and OD-17.
- **Never-backward key (must-fix, §3.3 step 6, §3.6).** `LichessBotGenerationLineage` lacked `recordedUnix` and the segment chain, so the playing generation could not be ranked with the selection's key. Added both, and a fork-against-the-playing-generation rule.
- **Scan filter and keys (must-fix, §3.2).** The plan said "regular files" and `lstat` via `FileSafety.existingItem`, which (a) gives no size or modification time and (b) would change the catalog's listing, contradicting "its result is unchanged". Now: today's filter, `stat` following links (the key describes the bytes read), a `FileSafety` helper for size and time, non-regular items listed as unreadable. Tests added.
- **Settings comparison fix spelled out (§3.4).** `generationSource` is per kind; the in-flight build join (`:111`, `:117`) uses it too; the controller's backoff reset keeps whole-settings comparison.
- **Follower details (§3.4).** `notBelow` read after the scan (actor reentrancy); `.noFiles` throws from `refreshIfDue`, so §4's alarm rows are true; the check runs before any generation exists (status only); a network-build failure does not mark the file failed; the post-load verification requires the loaded file's chain to equal or extend the candidate's and cites `ModelCheckpointFile.lineageParent`.
- **UI (§3.9).** The row shows the controller's status only when the draft's lineage is the applied one, and the offline resolution scan runs through an async `DispatchQueue` wrapper, never on the main actor; a failed scan shows its error.
- **Validation (§6), owner correction.** Work no longer waits for training runs: builds, tests and the live check run alongside them, read-only toward the trainers' files and processes. Added a header-only chain-following check on a real two-segment run.
- **Smaller.** No number in the proposed `.unrecorded` doc comment; OD-15 covers journals, records and the index; §1.5 re-measured (185 files carry a record: 180 `replay`, 5 `new_model`, no `vsuci`).

Should-fix (left to implementation, not edited):
- §3.3 step 4 usually keeps the rolling `…-latest` file (path order) — the copy most likely to be replaced before the load — so "file changed since the check" will be common. The verification handles it. Preferring the copy whose cached identity was seen earliest would avoid it without using names.
- The first follow-lineage check inside the controller's poll loop holds that loop (gate snapshot, challenge timeouts, matchmaking) for the scan, as a live-trainer rebuild does today. Measured at about 1 s (one header JSON parse per file plus 185 record decodes); acceptable, but worth a `scan ms` look in the live check.
- `ModelFileEntry.lineage == nil` (entries built outside the catalog) is excluded from candidates but not counted under any reason; a test-only path, so harmless.

### Second review (independent, 2026-10-06, after the owner decisions; against `f5be524b`)

Checked against the code: `goOnline` / `startRuntime` / `goOffline` / `tearDownRuntime` / `performShutdown` / the poll loop (`LichessBot/App/LichessBotController.swift`), `handleChallenge` / `startSessionIfNeeded` (`LichessBot/Play/LichessBotSessionManager.swift`), `LichessBotModelSlots`, `LichessBotChallengePolicy`, `LichessBotGameSession` (mid-game refresh, chat), the model card and the Go Offline buttons, and every test that constructs slots, goes online, builds a challenge context or a game session (grep of `LichessBotModelSlots(`, `goOnline(`, `snapshot: nil`, `unbuildableChampion`, `modelReady`, `latestMoveSource`, `LichessBotChallengeContext(`). More than 50 of the plan's `file:line` citations were spot-checked (slots, manager, policy, game session, controller, UI, settings, tests); all resolve and say what the plan claims. No path accepts, declines or sends a challenge before a generation exists once §3.10 is followed: sends need `.online` (`:1657`, `:1881`), and the manager, poll loop (queue pump, matchmaking) and reconciler start after `prepare`.

Fixes made in this file:
- **§5.1 was incomplete (must-fix).** (a) `LichessBotGroupBFixTests.testAcceptRechecksAcceptingAfterTheModelBuild` (`:197-219`) cannot survive P-ready: its latched provider would be consumed by `prepare` inside the harness (a hang), and the re-check it pins becomes unreachable. Listed as a deletion with its `GatedModelProvider`. (b) `LichessBotSessionManagerTests.swift:217` and `:245` pass `unbuildableChampion()` explicitly; `prepare` cannot build it, so both need the swap. (c) `LichessBotShutdownTests.swift:67` needs no edit (its only `goOnline` runs after shutdown) and was removed from the list. (d) The `LichessBotGameSessionFaultTests.swift:166` parameter's type changes, not only its default. Counts corrected (OD-19: eleven sites in nine files). An exact approval list was added; OD-19 has no recorded decision, and the deletions depend on it.
- **Accept path (must-fix, §3.10).** "Becomes read `slots.current`" kept a pointless actor hop and kept reachable two re-checks whose rule texts ("… while the model was prepared", `:666-687`) would no longer describe anything. The accept path now touches no slots, and the re-checks are removed as unreachable.
- **Settings edited while preparing were lost (must-fix, §3.10).** The runtime's settings box is built from the copy taken at the start of `startRuntime` (`:2538`, `:2597`) and `apply` writes only an existing box (`:620`); the plan widened that window from two HTTP requests to the whole model build. The box is now created from the current settings after `prepare`.
- **Cancellation (must-fix, §3.10).** A press after `prepare` (during `seedDailyCounts`, `:2645-2651`) would have been lost and the bot gone online; the flag is now checked after every suspension, before any task starts. Specified that `connection` stays `.connecting` ("Cancelling…") until the step returns (an immediate `.offline` would allow a second Go Online racing the held instance lock), that the flag is cleared per attempt, and that the same checks test `isShutDown` (a quit during `prepare` closes the queues, `:1209-1210`, before today's check runs).
- **Stale construction text (must-fix, §3.4).** It still said `init(provider:time:log:)` stays and existing tests compile unchanged, contradicting §3.10 and §5.1. Rewritten; `prepare` also hands the actor its scan cache and follow status.
- **Phase consistency (must-fix).** §3.10 used `generationSource` (a P0 change) for P-ready's switch; P-ready now keeps the whole-settings comparison in the slots (so P0's regression tests still fail first), introduces `generationSource` for the switch view only, and resets the poll's refresh time on any `settings.model` change. `testEverySourceIsBuiltBeforeGoingOnline` listed a follow-lineage case in P-ready, before that source exists; it moves to a P3 method.
- **Controller tests could not inject a folder (must-fix).** Added `LichessBotControllerServices.makeModelFolderScanner` with the live default (the `replyToTerminate` pattern), so no test reads the real `Models/` and no existing `LichessBotControllerServices(…)` call changes.
- **Smaller.** `trainerAvailable()` loses its only caller and is removed from the protocol and the app provider (fakes untouched); the model card's headline would label one source's model with another's name during a switch and now shows the playing generation's source; progress hops are tied to the attempt; the P-ready test that asserted log order through `SessionLogger.shared` (not capturable) now asserts the provider's count at the stream request, with log order left to live check 9; new tests for a late cancel, settings applied while preparing and a shutdown while preparing.

Should-fix (left to implementation, not edited):
- A source switch's build (and a follow-lineage refresh's scan and load) runs inside the poll loop's `await` (`:2882`), so matchmaking, the queue pump, challenge time-outs and `finishIfDrained` pause for it, as for a live-trainer rebuild today. Play is unaffected. If switches prove slow, give the loop one refresh task it starts when due and polls each tick.
- Going online only clears the launch-time leftover report when it succeeds, but `noteLeftoverJournalsAtLaunch` runs once per launch and skips while connecting (`:897-900`). If the operator goes online before it ran and the build fails, no leftover report is shown until relaunch. Pre-existing for every going-online failure; more likely now that a missing champion fails going online.
- `LichessBotFakeModelProvider.unbuildableChampion()` is left with no caller (removing it needs approval), and `testAcceptsThenPlaysTheGame`'s doc comment (`:274-275`) becomes inaccurate.


---

## 12. Implementation decisions

Decisions taken while implementing, where the plan left a choice open or the code needed something it did not spell out. Each follows the owner's rules and the plan's intent.

**P-ready**
- **Going-online progress is tied to an attempt counter, not `runtimeGeneration`.** `runtimeGeneration` moves only when a runtime starts or is torn down, and a failed attempt tears down nothing, so a late progress hop from a failed attempt could land in the next attempt before that one bumps it. `goingOnlineAttempt` is bumped at the start of every `goOnline`; a hop applies only to its own attempt while it is still preparing the model.
- **A third going-online step, `.startingSession` ("Starting the session").** After the model is built, going online still seeds the day's counts and starts the streams; the operator sees that the model step finished. It also gives `testGoOfflineAfterThePreparedModelStillCancels` a point to wait for (the press must land after `prepare` returned).
- **`LichessBotController.journalQueue` is internal, not private,** so that test can hold the journal queue and stop going online at the leftover-journal read. Nothing in the app uses it outside the controller.
- **`LichessBotModelSourceKind.displayName`** is the one place the sources' names live; the source picker, the model card headline, the going-online step and the switch status read it.
- **The switch status view also shows the last failed refresh** (with its retry time) when no switch is pending: a vanished champion or a failing live-trainer snapshot is then visible on the model card, not only in the alarms.
- **A failed build uses up no generation number** (as before): the builder returns the built network and the slots number it when they publish it.
- **`goingOnlineCancelled` / `shutDownWhileGoingOnline` count as "stopped"** in the challenge queue's error classification; no send can throw them, but the switch must be exhaustive and "stopped" keeps the entry.

**P0**
- **The byte-source seam is `LichessBotModelFileLoader`**, a value holding the one `readBytes` closure (`.live` reads through `CheckpointManager.readModelFileBytes(at:)`, which names a failed read `readFailed` exactly as `loadModelFile(at:)` always did). The seam went in first with today's two reads; `testRecordedHashIsOfTheBytesThatWereDecoded` failed against it; the single read then made it pass.
- **`refreshIfDue` adopts a same-source settings value** (`currentSettings = settings`) before its refresh test, so the interval and toggle in force are the newest; a finished build sets `currentSettings` to the settings it was built for (a caller that joined it with a newer non-weight value is brought up to date by the next poll).
- **Builds and tests run through `xcodebuild` with a private derived-data folder** (owner instruction, 2026-10-06): with several open workspaces named `DrewsChessMachine.xcodeproj` (agent worktrees), the Xcode tool resolved build and test commands to whichever window was frontmost. No Xcode window is opened or focused.
- **Fail-first evidence, re-taken with `xcodebuild`.** With the two reads and the whole-settings comparison temporarily restored, `testRecordedHashIsOfTheBytesThatWereDecoded` failed (hash of the second read; two reads) and all three `LichessBotModelSlotsSettingsChangeTests` failed (generation 2, a second snapshot); with the fix, all four pass unmodified.
