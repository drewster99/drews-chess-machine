# Lichess bot: which model is in use, its file's age, and the last switch

Status (2026-10-12): owner-requested (GitHub issue #20, "great do that"; lineage-picker findability folded in, "do your best work"); implemented. Deviations: "As implemented" at the end.

Paths are relative to `DrewsChessMachine/DrewsChessMachine/LichessBot/` unless they start with `DrewsChessMachine/` or `DrewsChessMachineTests/`.

## Problem

- **Settings ▸ Model** ("applies to the next game") shows the configured file or the followed lineage's newest file (`LichessBotFollowedLineageRow`). It doesn't show which model the bot is actually playing, or any file's last-modified time. The newest file and the one in use can differ: right after a lineage update, after a failed load, or when the newest file ranks below the one playing.
- **The Online panel** (`LichessBotControlsCard`, `UI/LichessBotOverviewView.swift:51`) shows no model at all.

## What exists

- `LichessBotModelSlots` (actor, `Play/LichessBotModelSlots.swift:397`) holds `current: LichessBotModelGeneration`, the generation new games start with. `publish(_:settings:reason:)` (:550) is the single place `current` is replaced, and it logs `[LICHESS-BOT] model generation N ready: …`.
- `LichessBotGenerationInfo` (`Play/LichessBotGameInterfaces.swift:21`) holds the generation's identity: source, model ID, training step, `snapshotAt` (build time), `filePath`, `fileSHA256` and `lineage` (with `cumTrainerStep`). It has **no file modified time and no previous generation**. It is Codable and is stored in game records.
- `LichessBotController` (`@Observable`) mirrors `slots.current.info` into `generation` each poll tick (:3525) and in `checkFollowedLineageNow` (:3495). It also overwrites `generation` with a starting game's generation (:3811), so `generation` doesn't always mean "current".
- Model files are read by `LichessBotModelFileLoader` (:310) through `CheckpointManager.readModelFileBytes` (`Data(contentsOf:)`, no file times). The lineage scan knows each candidate's `fileModifiedAt` (`ModelLineageTip.Candidate.entry`) but `LichessBotLineageFile` drops it.

## Design

### One record of the model in use

- **`LichessBotGenerationInfo.fileModifiedAt: Date?`**: the modified time of the bytes the generation was built from. It is read **from the open file descriptor that read those bytes** (`fstat`), so it is the time of exactly that file. A rolling save renamed over the path in between can't make the time and the bytes disagree. Nil for sources without a file (champion, trainer snapshot, live trainer). Optional, so game records written before it decode (as nil).
- **`LichessBotModelSwitch`** (new, `Play/LichessBotModelSwitch.swift`): `{ at: Date, to: LichessBotGenerationInfo, from: LichessBotGenerationInfo?, reason: String }`. `LichessBotModelSlots.publish` creates it; its time is the generation's `snapshotAt` (see "As implemented"). `from` is the generation being replaced, nil for the first generation of a session. `reason` is the existing build reason ("going online", "newer file of the followed lineage", …).
- `LichessBotModelSlots.lastSwitch: LichessBotModelSwitch?`: set only in `publish`, the one mutation point.
- **`LichessBotController.modelInUse: LichessBotModelSwitch?`**: mirrored from `slots.lastSwitch`, on the same paths and at the same moments as `generation` from `slots.current` (poll loop, Check Now), and cleared where `generation` is cleared. `modelInUse.to` is always the slots' current generation; it is never overwritten by a game start (unlike `generation`, which other views use and which stays as it is).
- **The ready line** gains `replacing generation K <modelID> <file>` when a generation replaces another, so the log states the switch too.

### File times

- `CheckpointManager.readModelFileBytesAndModificationDate(at:) -> (bytes: Data, modifiedAt: Date)`: open the file once (`FileHandle(forReadingFrom:)`), take the descriptor's `fstat` modification time, read to end, close. `readModelFileBytes` stays for its other callers. `LichessBotModelFileLoader.readBytes` returns both, and the four test stubs supply a fixed date.
- `LichessBotLineageFile.fileModifiedAt: Date?` (from `candidate.entry.fileModifiedAt`), so the "newest" line can show it.

### Settings ▸ Model

- **New row "In use"** (`LichessBotModelInUseRow`, new file), below Source: the file name (or the source's name when there's no file), model ID, cumulative step (lineage `cumTrainerStep`, else the training step), "file modified <time>" when there's a file, and "playing since <switch time>". When the bot isn't online: "Not online — the model loads when the bot goes online."
- **The lineage "newest" line** (`LichessBotFollowedLineageSummary`) adds "modified <time>" for the newest file. While online with the followed lineage applied, it also adds **"(in use)"** when that file's content SHA-256 equals the in-use generation's `fileSHA256`, and otherwise **"(next game once loaded)"**. Offline it shows neither, since nothing is playing.

### Online panel

- **New line** (`LichessBotModelInUseLine`, new file) in `LichessBotControlsCard`, shown while online:
  `Playing <file or source> (<modelID>, cum <N>) · file modified <time> · switched <time> from <previous file or modelID>`
  - The first generation of the session reads "since going online <time>" in place of the "switched … from" part.
  - When games in progress are still on an older generation (mid-game switching off), a second line names how many: "2 games in progress still on generation 3 (<modelID>)". This comes from the live games' latest-move generations (`LichessBotLiveGame.generations`).

### Times

One formatter (`LichessBotModelTimeText`): local time, with the date when not today, plus a relative age ("13:05 · 23 min ago"). Digits are monospaced. Refreshed by the panel's existing `TimelineView` cadence, or a 30 s `TimelineView` where there is none.

### Lineage picker (owner, 2026-10-12: folded into #20)

`LichessBotModelLinePicker` (`UI/LichessBotModelLinePicker.swift`) over `ModelLineageTree` (`DrewsChessMachine/Persistence/ModelLineageTree.swift`). A branch like R-elitecyc (`20261012-2-dMNh`) sits six levels down: Ew5t → mIMw → Ea0A → abM2 → btGW → dMNh. Every level starts collapsed (`OutlineGroup`); each level lists its older files before its child lineages (20 file rows under abM2 before btGW); and search matches only model IDs and sessions, not file names.

- **Search matches file names too** (`ModelLineageTree.listedRows(of:matching:)`): a line matches when its model ID or any of its file names contains the text (case-insensitive). Matches are shown with their ancestors, and the matched lines and their ancestors are expanded (`OutlineGroup` → a recursive `DisclosureGroup` list with an expanded set, seeded from the search).
- **"Active lineages" section** above the tree: every branch tip (a segment node with no child segments), newest activity first. Each row shows:
  - its newest file name and modified time;
  - its model ID and cumulative step;
  - its chain of ancestor model IDs, written as a short path ("Ew5t › mIMw › Ea0A › abM2 › btGW › dMNh");
  - the model name.

  Selecting a row selects that lineage, exactly as selecting its tree node does: same selection ID, same purpose rules.
- **Child lineages before older files** in each segment's children: a segment's sub-branches lead, then its earlier files, then session champions.
- **Times:** each file row and each Active-lineages row shows the file's modified time (`ModelFileEntry.fileModifiedAt`), with the same formatter as the model-in-use rows.
- **Tests:**
  - tree builder: child segments come before earlier files; the branch-tip list is ordered by newest activity; a search on a file-name fragment finds a deep line and lists its ancestors; a model-ID search still works;
  - render: the picker with a six-deep chain shows the tip in Active lineages.

## Edge cases

- **No file** (champion / trainer / live trainer): no modified time; shows the source name.
- **A failed load:** `publish` isn't reached, so the switch record and "in use" stay on the old generation. The newest line shows "(next game once loaded)" plus the existing load-failure status.
- **Offline, or a different lineage in the draft settings than the applied one:** the row says not online, or shows the applied lineage's in-use model. "(in use)" is decided only against the applied lineage.
- **A rolling `-latest` file rewritten in place:** the modified time comes from the bytes actually read, so after a rewrite the in-use line still shows the time of the loaded version. The newest line (from the scan) shows the new time and "(next game once loaded)" until it is loaded.
- **Going offline and online again:** `generationID` restarts and the first switch has no `from`; it reads "since going online".
- **Old game records:** `fileModifiedAt` decodes as nil.

## Tests

- **Slots:** `prepare` gives a switch with no `from`; a second publish records `from` = the first generation, `at` = the test clock, and the reason. A failed build leaves `lastSwitch` unchanged.
- **Loader:** `fileModifiedAt` equals the file's modified time; the date is taken from the descriptor that read the bytes (stub returns a known date).
- **Lineage file:** carries `fileModifiedAt` from the candidate.
- **Controller:** after a lineage update, `modelInUse.to` changes at the publish, and `generation` set by a game start doesn't change `modelInUse`.
- **Render:** the In use row (online / offline / no-file source), the newest line with "(in use)" and "(next game once loaded)", and the Online-panel line (first generation, a switch, games on an older generation), through the existing `LichessBotFollowLineageViewRenderTests` harness; long file names fit (no overflow, cf. #19).
- **Codable:** a `LichessBotGenerationInfo` JSON without `fileModifiedAt` decodes; a round trip keeps the date.

## Validation

- All touched test classes plus the full suite pass (the full suite runs once with the other changes).
- Run against the live bot: the Online panel and Settings ▸ Model show the file in use with its time, and a lineage update shows "(next game once loaded)" and then "(in use)" with a new switch time. The bot is never restarted for this without the owner's go.

## As implemented

- **Switch record:** `LichessBotModelSwitch.at` is the generation's own `snapshotAt` (wall-clock `Date()` when the generation was built, just before `publish`), not a separate field, so tests don't control it (the slots' test clock is a `Duration`, not a date); `lastSwitch` is non-optional (`prepare` records the first).
- **One read:** the controller takes generation, switch and lineage status from `LichessBotModelSlots.shown` in one actor turn, so a publish can't land between them.
- **No line limits** on the In use details and the Online line: long file names wrap rather than truncate.
- **Search pruning:** a row matching on its own model ID keeps its whole subtree (a seed's ID shows its whole family); a row matching only on a file or session name whose subtree holds more matches keeps only the paths to them (`ModelLineageTree.pruned`), so the chain lists flat. Rows under a match that has several kept children still start collapsed (no expanded-set); during a search the "Recent branch tips" section lists every matching tip (up to 8) at the top.
- **Status log:** `LichessBotLineageFile` equality includes `fileModifiedAt`, so a newest file touched without a content change logs one more `lineage follow:` line. Accepted.
- **Text in one place:** `LichessBotModelInUseText` (`UI/`) builds every string the row, the Online line and the newest-file mark show; the views only lay them out.
- **One "is this file playing" rule:** `LichessBotGenerationInfo.plays(_:of:)`, used by the slots' retry check (`newestKnownFileIsNotPlaying`) and the "(in use)" mark. The mark also requires the applied source to be Follow lineage (`appliedFollowedLineage(of:)`), so a lineage left in the settings under another source shows no mark.
- **Times:** `LichessBotModelTimeText` (`absolute`, `withAge`, `ageText`); the picker's `dateText` delegates to it. The newest-file line shows the absolute time only (no age, so no timer); the In use row and the Online line refresh every 30 s.
- **Older generations:** compared by the whole generation info, since IDs restart per going-online; oldest first.
- **Picker:** the "Active lineages" section is named "Recent branch tips" (8 rows); the expanded-tree rewrite became a pruned tree (`ModelLineageTree.pruned`) for searches, with file names searchable.
- **Tests:** `LichessBotModelInUseTests` (text, times, marks, Codable, renders), `LichessBotModelSwitchTests` (slots), `LichessBotGoOnlineModelFirstTests.testThePollPublishesTheModelInUseAndGoingOfflineClearsIt` (controller mirroring and clearing), loader and `ModelLineageTreeTests` additions. The controller case "a game start doesn't change `modelInUse`" holds by construction (only the poll and Check Now assign it) and has no test of its own.
