# Lichess bot: follow the newest checkpoint of one lineage on disk

Status (2026-10-06): **PLAN ONLY.** Nothing here is implemented.
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
- When the source cannot vouch for a file (no file of the lineage, folder unreadable, a fork after the chosen segment), new challenges are declined and the reason is shown and logged. It never switches to another source and never steps back to an older file.
- Fixes two latent bugs found while reading the code (P0, regression tests first): the file loader hashes a second read of the file, not the bytes it decoded; and any model-settings edit (even the refresh interval) stops `refreshIfDue` until the next game starts.
- Fixes a third latent bug the feature would trip (P2, regression test first): the settings store reports saved settings as unreadable when they hold a struct-typed optional whose default is nil — exactly what `followedLineage` is (§3.1).

Rules this plan follows (CLAUDE.md files and the owner's standing rules):
- checkpoints are identified by safetensors `__metadata__` (`model_id`, `training_step`, the `dcm_lineage` record), never by filename;
- the `dcm_lineage` JSON is the single source of truth; its flat mirror keys (`lineage_run_id`, `cum_trainer_step`, …) are never read back (`Persistence/LineageRecord.swift:876-890`);
- no silent defaults or fallbacks: every "can't" is a stated, logged state;
- no `try?`, no force unwraps; long file work runs on a `DispatchQueue` behind a continuation, never synchronously inside a `Task`;
- saved settings from today's build keep loading (owner's past pain: no forced "Reset to defaults");
- SwiftUI: one `View` per new file, no `some View` helper properties, no `AnyView`, `.shown(_:)` instead of `if`-gated content, `onChange` 0/2-arg only;
- tests are not modified or deleted; new tests go in new files; bug fixes get a failing regression test first.

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
- Construction: a new designated `init(provider:time:folderScanner:log:)`. The existing `init(provider:time:log:)` stays and passes the production scanner, so the existing tests that build slots (`DrewsChessMachineTests/LichessBotSessionManagerTests.swift:170`, `LichessBotPhase5CoreTests.swift:158`) compile unchanged; the app (`LichessBotController.swift:2600`) uses the designated init explicitly.

`LichessBotLineageFollowStatus` (Sendable, Equatable) — what the UI and logs show:
- `followed`, `checkedAt: Date`, `outcome` = `.following(newest:)` | `.keepPlaying(newestOnDisk:)` | `.noFiles` | `.fork` | `.conflict` | `.folderUnreadable(reason)` | `.newestFailedToLoad(file:reason:)`, plus scan counts and excluded-by-reason counts.

Behavior:
- **`checkLineage(settings, force:)`** — scans (joining an in-flight scan), selects with `notBelow` = the current generation's position when it is of this followed lineage, updates the status, logs on change only (§3.7). Throttled by `lineageCheckIntervalSeconds` measured on the injected `time` source unless forced.
  - Actor reentrancy: `current` can change while the scan is awaited (a build finishing), so `notBelow` is read **after** the scan returns, never captured before it.
- **`sourceAvailable`** (`:152-161`) for `.followLineage`: uses the status, checking first if there is none or it is older than the interval. Available when the outcome is `.following` / `.keepPlaying`, or `.newestFailedToLoad` while a generation of this lineage is playing. Unavailable for `.noFiles`, `.fork`, `.conflict`, `.folderUnreadable`, and `.newestFailedToLoad` with nothing playing (OD-6, OD-8). Per-challenge calls cost nothing between checks.
- **`build`** (`:163-207`) for `.followLineage`: takes the selected candidate (checking first if needed; throws a `LichessBotLineageFollowError` naming the outcome when there is none), loads it with the shared loader (§3.5), then verifies the **decoded** file (its `lineageParent.lineage`, `ModelCheckpointFile.swift:370-385`): same run, chain contains the anchor, path kind allowed, its chain equal to or extending the candidate's (the same segment or a descendant — a sibling segment of the same run that took over the same `--out-model` path must not pass), and ranks at or above the candidate. A rolling file replaced between scan and load passes (it moved forward within the same lineage) and the generation records what was actually loaded, with a log note; anything else throws `followedFileChangedDuringLoad` and the next check rescans. A **file** failure (read, decode, `content_sha256`, the verification above) records the file's fingerprint in `failedLineageFiles`, so it is not retried until it changes on disk; the selection treats a failed file as `.newestFailedToLoad`, never as absent (no stepping back past it). A failure after the file decoded (building the inference network) says nothing about the file and does not mark it: it throws, and the controller's backoff retries it.
- **`refreshIfDue`** (`:127-142`) for `.followLineage`: when the interval has elapsed, check; rebuild when the selection's `contentSHA256` differs from the playing generation's and ranks above it. "No newer file" is not an error. `.noFiles`, `.fork`, `.conflict` and `.folderUnreadable` throw, so the controller's existing backoff and alarm apply (`LichessBotController.swift:2887-2897`) — §4's "alarm" rows depend on `.noFiles` throwing too.
  - Today's guard returns early when there is no `current` generation (`:128`). For `.followLineage` the check (status and log only, no build) runs without one, so the Overview status and the "unavailable" alarm exist before the first challenge; this is also what makes the first check start at go-online (§8). The build still happens in `ready`, before a game.
- **Operator "Check now"** (OD-12): `checkLineageNow(for:)` = `checkLineage(force: true)` + the same rebuild rule. Never bypasses any rule.
- **Architecture changes**: an exact resume cannot change architecture, so within one run it should not happen. The bot builds a network per generation from the file's own architecture (`:186`), so a change would still play correctly; it is logged in the ready line, not refused (OD-16).

**Settings comparison fix (P0, latent bug).** `ready` and `refreshIfDue` compare whole `LichessBotModelSettings` (`:103`, `:128`). Editing the interval or the mid-game toggle while online therefore makes `refreshIfDue` return early until the next game calls `ready`, which then rebuilds needlessly ("source changed"). For a follower that means following silently stops after an interval edit. Fix: compare a `generationSource` value for "is this the same generation source", and adopt the other fields without a rebuild.
- `generationSource` holds only what selects the weights for the settings' kind: the kind, plus `filePath` for `.file` and `followedLineage` for `.followLineage`. (A `filePath` left over in the settings while the source is the live trainer must not rebuild the live trainer.)
- All three comparisons use it: `ready` (`:103`), `refreshIfDue` (`:128`) and the in-flight build join in `rebuild` (`:111`, `:117`) — otherwise two callers that differ only in the interval start two builds.
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
- Problems, on transition: `[LICHESS-BOT] lineage follow unavailable: <outcome with names>` and, when it clears, `[LICHESS-BOT] lineage follow available again: …`.
- Never-backward: `[LICHESS-BOT] lineage follow: newest on disk <file> (seg k step s) ranks below the playing generation N (seg k step s'); keeping generation N`.
- A failed load: `[LICHESS-BOT] lineage follow: <file> could not be loaded: <error>; not retried until it changes; still playing generation N` (or "no generation playing; declining new challenges").
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

**Overview → Model card** (`LichessBot/UI/LichessBotOverviewView.swift:262-283`): new `LichessBotLineageFollowStatusView` (new file), `.shown(source == .followLineage)`: "following run 783BF744 from Xxvl", "newest <file> · seg 0 · step 31000 · cum 31000", "checked 11:30:02", the outcome in orange (unavailable) or secondary (keep playing), and a "Check Now" button (OD-12) calling `controller.checkFollowedLineageNow()`.

**Controller** (`LichessBot/App/LichessBotController.swift`): publishes `lineageFollowStatus` next to `generation` in the poll loop (`:2843-2845`); `checkFollowedLineageNow()` runs on the runtime's slots and surfaces errors through the existing `raiseAlarm` path.

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
| Every file of the lineage gone, or `Models/` unreadable | Unavailable: new challenges declined (`later`), alarm; games in progress finish on their generation. |
| Exact resume of the run (new process, new `model_id`) | Followed automatically (its chain contains the anchor). |
| Branch from a file of the run (`--start-model` without `--resume-exact`) | A new run; not followed (OD-10). |
| Two resumes of one file after the anchor | `.fork`: unavailable, alarm naming both segments; the operator re-anchors on one. |
| Resume of the parent from an earlier file, after the parent stopped | The new segment ranks above the old one by chain depth, even at a lower cumulative step; logged once (OD-17). |
| Second process `--resume-exact`-ed from a step file of a run that keeps training | `.fork` (§3.3 step 2b): unavailable, alarm naming both; the operator anchors on one. |
| Settings saved with a followed lineage, next launch | Loads (store fix, §3.1); without the fix the whole saved settings would be "unreadable". |
| `cum_trainer_step` null (run continues unrecorded history) | Ranking does not use it; display shows "cum —". |
| File written before lineage records (format < 7) | Not followable; counted as excluded; not selectable in follow mode. |
| GUI run's files in `Models/` | Not followable (OD-4); the in-process sources cover the GUI run. |
| Settings edited while online (interval, toggle) | No rebuild; following continues (P0 fix). |

---

## 5. Tests (new files only; no existing test is modified)

Real model files are written with `SafetensorsModelIO.encode(…, lineage:)` (`Persistence/SafetensorsModelIO.swift:115-123`), as `DrewsChessMachineTests/CheckpointManagerSafetensorsTests.swift:246-250` does, with records from `LineageTracker` (`.fresh`, `.resume(parent:gaps:legacyTotals:)`, `.branch`) so chains are real, not hand-built. Temporary folders as in `ModelFileCatalogTests.swift:8-18`.

**P0 — regression tests first (must fail before the fix, pass after, unmodified):**
- `DrewsChessMachineTests/LichessBotModelFileLoadTests.swift`
  - `testRecordedHashIsOfTheBytesThatWereDecoded` — the loader reads through a byte-source seam that serves file A's bytes on the first read and file B's on the second (the seam is introduced first with today's two reads, no behavior change; the test then fails; the single-read fix makes it pass).
- `DrewsChessMachineTests/LichessBotModelSlotsSettingsChangeTests.swift`
  - `testChangingTheLiveTrainerIntervalKeepsRefreshing` — fake provider + `LichessBotManualTime` (`DrewsChessMachineTests/LichessBotGameSessionTests.swift:7`): `ready(A)`, then `refreshIfDue(A with a new interval)` after it elapses → a new generation (today: none).
  - `testTogglingMidGameRefreshDoesNotRebuild` — `ready(A)`, `ready(A with the toggle flipped)` → same generation (today: rebuilt).

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
- `testFirstUseBuildsTheNewestFile`
- `testCheckRunsOnlyAfterTheInterval` (scan count by manual time)
- `testNewerFileRebuildsAndOlderDoesNot`
- `testSameContentDoesNotRebuild`
- `testNoFilesIsUnavailable` / `testFolderUnreadableIsUnavailableAndThrowsOnRefresh` / `testForkIsUnavailable`
- `testRunEndKeepsTheLastFileAvailable`
- `testFailedNewestKeepsPlayingAndIsNotRetriedUntilItChanges`
- `testFailedNewestWithNothingPlayingIsUnavailable`
- `testReplacedRollingFileLoadsTheNewerSameLineageFile`
- `testFileOfAnotherLineageAtTheSamePathIsRefused`
- `testGenerationRecordsLineagePosition` (and the `.file` source records it too)
- `testConcurrentCallersShareOneScan`
- `testCheckRunsBeforeAnyGenerationExists` (status and log without `current`; no build)
- `testNetworkBuildFailureDoesNotMarkTheFileFailed`
- `testSiblingSegmentInTheRollingPathFailsVerification`
- `testInFlightBuildIsJoinedAcrossAnIntervalEdit` (two callers differing only in the interval → one build)
- `testOldGenerationInfoJSONStillDecodes` (a game record's generation JSON without `lineage`)

**P4 — `DrewsChessMachineTests/LichessBotFollowLineageMidGameTests.swift`** (harness from the shared fakes `LichessBotFakeGameServer`, `LichessBotManualTime`, `LichessBotRecordingGameObserver` in `LichessBotGameSessionTests.swift:7`, `:85`, `:428`):
- `testFollowedLineageGameSwitchesToANewerGenerationOfTheSameLineage`
- `testGameNeverSwitchesToAnotherFollowedLineage`
- `testGameNeverSwitchesWithMidGameRefreshOff`

**P5 — `DrewsChessMachineTests/LichessBotFollowLineageViewRenderTests.swift`:** the settings Play tab with each source, the follow status view in every outcome, the picker in follow mode with a non-followable row disabled.

**Existing suites that guard the change** (run, unmodified): `ModelFileCatalogTests`, `ModelFileCatalogActivityOrderTests`, `ModelLineageTreeTests`, `LichessBotSettingsStoreTests`, `LichessBotSettingsStoreEncodingTests`, `LichessBotSettingsTabTests`, `LichessBotSettingsViewRenderTests`, `LichessBotGameSessionTests`, `LichessBotGameSessionFaultTests`, `LichessBotSessionManagerTests`, `LichessBotPhase5CoreTests`, `LichessBotDataLayerTests`, `LineageRecordTests`, `CheckpointManagerSafetensorsTests`.

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

**Done when:** all phases' tests pass, the full suite passes, the live check matches the independent header reading at two consecutive saves, and `LICHESS_BOT_PLAN.md` §9 / §9.2 and `CHANGELOG.md` are updated.

---

## 7. Phasing (build + commit per phase)

- **P0 — latent bugs** (§3.4 last paragraph, §3.5): byte-source seam → failing tests → single-read load; settings-comparison tests → failing → `generationSource` comparison.
- **P1 — persistence:** `ModelFileEntry` lineage facts and content hash; `ModelFolderHeaderCache`; `ModelFileCatalog` over it; `ModelLineageTip`; tests.
- **P2 — settings:** enum case, `LichessBotFollowedLineage`, new fields, limits (poll cadence moved to `LichessBotLimits`), validation; the store regression test (`testSavedFollowedLineageLoadsThroughTheStore`) shown failing, then the `isKnownOptionalKey` fix (§3.1); tests.
- **P3 — slots:** scanner protocol and production scanner, follow status, check/build/availability/refresh, `LichessBotGenerationInfo.lineage`, logging, controller publishing; tests.
- **P4 — mid-game refresh** generalization; tests.
- **P5 — UI:** settings rows, picker follow mode, Overview status view, "Check Now"; render tests.
- **P6 — docs:** `LICHESS_BOT_PLAN.md` §9 table row + §9.2 pointing here; `CHANGELOG.md`; this file's status; ROADMAP only with permission (OD-14).

---

## 8. Risks

- **First scan cost.** Reading 4.4k headers and decoding their records once per launch of the bot runtime: about 37 MB of reads plus JSON decode, estimated a few seconds, off the actor on a utility queue. A challenge arriving during the first check waits for it. Mitigation: the check also runs when the source is selected in Settings (the row's resolution scan), and the bot's first check starts at go-online.
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
- **OD-2 Check interval:** a new `lineageCheckIntervalSeconds`, default 60 s, minimum = the 15 s poll. *Recommend* over reusing `liveTrainerRefreshIntervalSeconds` (its name and 30 s floor are about SGD lock contention, not disk checks).
- **OD-3 Mid-game toggle:** shared, label-only change, Swift name and saved key unchanged. *Recommend* (no rename, no migration, no test edits).
- **OD-4 Followable path kinds:** `replay` and `vsuci` only. *Recommend* (GUI clocks rewind on promotion; the GUI has its own sources).
- **OD-5 Folder:** `Models/` only. *Recommend* for v1.
- **OD-6 Newest file fails to load:** keep the playing generation, don't retry until the file changes, never step back to an older file; with nothing playing, unavailable. *Recommend.*
- **OD-7 Newest on disk older than the playing generation:** keep playing, log. *Recommend.*
- **OD-8 No files of the lineage / folder unreadable:** decline new challenges, finish games in progress. *Recommend.*
- **OD-9 Fork after the anchor:** unavailable + alarm, operator re-anchors. *Recommend* over auto-picking the newest save (no silent choice).
- **OD-10 Branches:** not followed. *Recommend.*
- **OD-11 Lineage attribution for the fixed-file source too:** *Recommend yes* (same loader, free).
- **OD-12 "Check Now" button on the Overview:** *Recommend yes.*
- **OD-13 Fix the two latent bugs in P0** (hash of a second read; settings edits stall refresh): *Recommend yes*, regression tests first. (The settings-store bug of §3.1 is not optional: without its fix the new field makes saved settings unreadable; it is fixed in P2.)
- **OD-14 ROADMAP:** a one-line pointer under the Lichess bot item to this plan. *Recommend yes* — needs your permission.
- **OD-15 Downgrade of settings saved with the new source:** accepted as unreadable in older builds. *Recommend accept.* The same holds for game journals, records and the index written with `sourceKind = followLineage` (`LichessBotGenerationInfo.sourceKind` is the enum): an older build cannot decode them either.
- **OD-16 Architecture change inside a followed lineage:** build and log, don't refuse. *Recommend* (cannot happen for exact resumes; the bot is architecture-agnostic).
- **OD-17 A child resumed from an earlier file of its parent (§3.3 step 2):** if the parent kept training after the child started, it is a fork (unavailable, operator re-anchors); if the parent had stopped before the child started (a redo), follow the child and log it once. *Recommend.* Stricter alternative: treat both as a fork. Known limit either way: following the parent past a concurrent child needs a segment-exclusion setting this plan does not add.

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
