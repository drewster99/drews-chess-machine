# Lichess bot: a durable challenge log, and how each game started

Status (2026-10-06): **PLAN ONLY.** Nothing here is implemented. P1 (file layer) may start now; P2 onward start after `LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md` has landed (§7, "Sequencing"). Reviewed 2026-10-06 against the code and the real bot data; the fixes are listed in §11.
- Every `file:line` was checked against `main` at `57af1480`.
- Paths are relative to `DrewsChessMachine/DrewsChessMachine/` unless they start with `DrewsChessMachineTests/` (= `DrewsChessMachine/DrewsChessMachineTests/`) or `documentation/`.
- Numbers about the bot's data were measured on this Mac on 2026-10-06, read-only, from `~/Library/Application Support/DrewsChessMachine/LichessBot/`. Sizes are base-2.
- Abbreviations: owner decision (OD), Coordinated Universal Time (UTC), JSON Lines (JSONL), Portable Game Notation (PGN), user interface (UI).

**The request (owner, 2026-10-06).**
- "Don't we already have a log of every challenge we have issued? If not, we should. Even if the log is just a JSONL somewhere that we ignore. Let's figure out how to retain that data and get it in the UI, and also back-fill if possible."
- Earlier: "For lichess bot games list in the UI, do we have a way to show HOW the game got started? I.e., matchmaking or accepted challenge?"

**Short answer to both questions: no.** There is no durable record of challenges, and no game record says how its game started (§1).

**What this plan does.**
- Adds an append-only **challenge log**: `Challenges/challenges-YYYYMMDD.jsonl`, one file per UTC day, kept forever. It gets one line per lifecycle fact of every challenge, in both directions:
  - outgoing created (with who sent it), not created (offline, refused, or no answer), withdrawn (why, and what Lichess said), declined, canceled;
  - incoming received, DCM's decision, a failed response, canceled by the challenger;
  - a game starting from the challenge.
- Records **how each game started** at game start: a new journal event, an optional field in the game record and in the index row. A resumed game takes it from its journal. When it can't be determined, the game says so and gives the reason.
- **UI:**
  - an Origin column (with a filter) in the All Games window;
  - an origin glyph in the Recent games list and on game tiles;
  - origin text in the Live picker and the game detail;
  - a new **Challenge Log** window (a table with filters), opened from the challenge outcomes card.
- **Back-fill:**
  - Rebuilds past challenges from the protocol log into a separate, regenerable file, `Challenges/reconstructed-from-protocol.json`. Each row carries its evidence and a confidence.
  - Past games get their origin by joining on the game id, which is the challenge id. No game record, PGN or journal is rewritten.
  - On this Mac it classifies **206 of 206** filed games: 12 accepted incoming, 132 matchmaking, 62 operator (inferred), 0 unknown. It also rebuilds 373 past challenges (§6, Appendix A).

Rules this plan follows (CLAUDE.md files and the owner's standing rules):
- one source of truth, and one point of mutation:
  - the challenge log is the record of challenges;
  - the controller's `recordChallengeEvent` is its only writer;
  - the display resolver (§3.6) is the one place that decides what origin a game shows;
- no silent defaults: an origin that isn't known is shown as **Unknown, with the reason**, never left blank and never guessed;
- no `try?`, no force unwraps, no stringly-typed values (Lichess's own strings go through `LichessBotOpenValue`); file writes go through `FileSafety`;
- old files stay readable: every existing record, index, journal, outcome file and protocol file still decodes, and none is rewritten or deleted;
- the derived `index.json` cache is rebuilt by its own existing rule (§3.5);
- SwiftUI: one `View` per new file, no `some View` helper properties, `.shown(_:)` instead of `if`-gated content, centrally defined glyphs and labels;
- tests are not modified or deleted; new tests go in new files. **No existing-test edits are planned** (§5.1 says how this was checked).

---

## 1. How things work today (verified)

### 1.1 What records challenges today

- **`challenge-outcomes.json`** (`Stats/LichessBotChallengeOutcomes.swift`):
  - Covers outgoing attempts only: created, offline, refused, plus the answer (`recordCreated` `:231`, `recordNotCreated` `:244`, `resolve` `:284`).
  - No sender (who sent it) and no incoming challenges.
  - **Pruned after 24 h** (`prune` `:292`, called on every update, `App/LichessBotController.swift:1425-1452`).
  - It exists to count Lichess's challenge credits (`LichessBotChallengeCredits` `:158` ff.).
- **The protocol log** (`Data/LichessBotProtocolLog.swift`):
  - `Protocol/events-YYYYMMDD.jsonl`, one file per UTC day, kept indefinitely, redacted.
  - **Never synchronized to disk** (`:48-50`, the type's doc comment), so a power loss can lose recent lines.
  - It stores every raw event-stream line (`App/LichessBotController.swift` `.eventStreamLine` in `handle`, `:3150` ff.), including `challenge`, `gameStart`, `challengeDeclined` and `challengeCanceled` with their full JSON.
  - It also stores free-text `.challenge` messages. Each send writes "challenge sent to <user>" (with `id`, `rated`, `clock`, `color`), and some senders then write a companion line beside it: "matchmaking sent a challenge to <user>", "challenge queue: sent <user>" or "matchmaking resent a challenge to <user> as casual". The other messages are incoming decisions "<challenger>: accept | decline (<key>): <rule> | ignore: <rule>" (field `challenge`), "outgoing challenge accepted|declined|canceled", withdrawals, and since 2026-10-01 "challenge outcome: …" lines.
  - It is free text for people, not a record to query, and the companion lines carry no challenge id.
- **Nothing else records challenges.** The controller's `pendingChallenges` (`:185`) and the manager's `pendingOutgoingChallenges` / `acceptedAwaitingStart` (`Play/LichessBotSessionManager.swift:179`, `:192`) are memory only, and are cleared as soon as the game starts (`:694`).

### 1.2 Who sends a challenge, in memory

- `LichessBotController.ChallengeOrigin` (`App/LichessBotController.swift:106-118`):
  - `.manual` covers the Challenge sheet, the queue and the manual Resend as Casual;
  - `.matchmaking(fillMode:opponent:)`;
  - `.matchmakingCasualResend`.
- It is kept on `PendingChallenge.origin` (`:120-128`) and only decides how a decline is answered (`matchmakingCasualResend(for:reasonKey:)`, `:2447-2459`).
- Every outgoing send goes through the private `sendChallenge(to:request:origin:)` (`:1713`). The callers:
  - the public `sendChallenge(to:request:)` (`:1705`, the Challenge sheet, `UI/LichessBotChallengeSheet.swift:337`);
  - `resendAsCasual` (`:1810`);
  - the queue pump (`:2025`);
  - `sendMatchmakingChallenge` (`:2290-2312`), for automatic passes (`:2167` ff.), Fill Open Slots (`:2119-2161`) and the casual resend (`:2512`).
- The queue and the manual resend reach the private send through the public one, so both are `.manual` today. An automatic pass and Fill Open Slots differ only by the `fillMode` they pass, and Fill always passes `.everyFreeSlot` — the same value an automatic pass passes when the operator's fill mode is `.everyFreeSlot`, so the two can't be told apart.

### 1.3 How a challenge's life reaches the controller

- The manager's events reach the controller through one `AsyncStream`, in order (`:2721`, consumed at `:2780`).
- For a challenge event, the manager emits:
  - `.challengeArrived(challengeID:challengerID:challengerTitle:)` (`Play/LichessBotSessionManager.swift:614`), only for the alert sound (`App/LichessBotController.swift:3162-3177`);
  - then `.challengeDecision` (`:616` for our own echo, `:643` otherwise).
- `challengeDeclined` / `challengeCanceled` are reported only for a challenge the manager holds as pending outgoing (`resolveChallenge`, `:598-608`). Answers to anything else are kept for a moment in `recentChallengeOutcomes` (`:195`) and never reported.
- `gameStart` resolves a pending outgoing challenge, then starts the session (`:573-581`). The controller hears `.gameSessionStarted(gameID:generation:origin:)` (`:740`). That `origin` is `LichessBotSessionOrigin` (new / resumed, `Play/LichessBotGameResume.swift:191-199`) and has nothing to do with challenges.
- **Race:** the POST that creates a challenge can return after Lichess has already started the game. The code handles it:
  - `noteOutgoingChallenge` reports "accepted" if the session already exists (`Play/LichessBotSessionManager.swift:291-303`);
  - the controller resolves the pending challenge in `.gameSessionStarted` (`App/LichessBotController.swift:3187-3197`).
  
  So "who sent it" can become known after the game started.
- Lichess's own data can't tell direction: every challenge game has `gameStart.source == "friend"` (210 of 210 here), and the export doesn't name the challenger.

### 1.4 Game data

- The game record (`Data/LichessBotGameRecord.swift`, `currentSchemaVersion = 1` at `:22`) is built purely from the journal plus Lichess's export (`LichessBotRecordBuilder.build`). It is "the single source of truth for everything the bot shows about it" (`:15-20`) and has no origin.
- The journal (`Data/LichessBotJournal.swift`):
  - `LichessBotJournalEvent` has 17 cases (`:17-48`), `schemaVersion = 1` (`:62`);
  - "a build that doesn't know a case can't decode a journal holding it" (`:7-12`).
  
  It is replayed by three exhaustive switches:
  - the record builder (`Data/LichessBotGameRecord.swift:505` ff.);
  - the resume carryover fold (`Play/LichessBotGameResume.swift:41-66`);
  - the live view's replay (`App/LichessBotLiveGame.swift:319` ff.).
- The controller already writes into a game's journal outside the session: `LichessBotJournalWriter.recordRequest` (`Data/LichessBotJournal.swift`, used for request records).
- The index (`Data/LichessBotIndex.swift`):
  - `LichessBotGameSummary` (`:6-56`);
  - `schemaVersion = 2` (`:66`);
  - a stored index whose schema differs is discarded and rebuilt from the records (`readStored`, `load`).
- Synthesized `Codable` decodes an absent `let x: T?` as nil, so a new optional field leaves every existing record and index row readable.
- Live games: `App/LichessBotLiveGame.swift`, listed by `listStartedGame` (`App/LichessBotController.swift:3507`).

### 1.5 The UI lists

- All Games window: `UI/LichessBotAllGamesWindow.swift`. It is a `Table` over `controller.index.rows` with columns result, When, Color, Opponent, Rating, Speed, End, Plies and Game.
- Recent games: `LichessBotRecentGamesList` (`UI/LichessBotRecordCard.swift:139` ff.).
- Live picker: `LichessBotLiveView.menuTitle` (`UI/LichessBotLiveView.swift:51-54`).
- Tiles: `UI/LichessBotGameTileView.swift`.
- Game detail: `UI/LichessBotGameDetailView.swift`.
- Outcomes card: `UI/LichessBotChallengeOutcomesCard.swift`.
- Central style precedent: `LichessBotStatusStyle`.

### 1.6 File writing today

- `LichessBotJSONLines.append(_:to:synchronize:)` (`Data/LichessBotJSONLines.swift:73-107`) is the one JSONL append helper, used by the journal and the protocol log:
  - it opens with `O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC` (`:78`), so it follows a symbolic link;
  - `synchronize` is `FileHandle.synchronize()` (plain `fsync`, which doesn't flush the drive's cache).
- `FileSafety` (`Utils/FileSafety.swift`) has `fullSync` (`F_FULLFSYNC`, `:732`, `:744`), `writeNewFile` (`:533`) and `replaceRegularFile` (`:597`), but **no append operation**.
- Appenders already cut an unterminated final line before appending and record the cut (`cutUnterminatedFinalLine`, `Data/LichessBotJSONLines.swift:116-152`). The cut opens the path separately, with `FileHandle(forUpdating:)` (`:117`), which also follows a symbolic link, so a cut can truncate a link's target.
- **More than one process appends.** Every DCM instance keeps a protocol log (`Data/LichessBotJSONLines.swift:70-71`), so two instances append to the same day file. The instance lock (`bot.lock`) is released on the journal queue at teardown (`App/LichessBotController.swift:2938`, in `tearDownRuntime`). Withdrawal tasks outlive the runtime (`withdraw`, `:1846-1863`), so a withdrawal's result can be logged after another instance has taken the lock. Today nothing serializes one process's tail cut against another process's append.

### 1.7 Measured on this Mac (2026-10-06)

- 206 game records, 2026-09-28 to 2026-10-06.
- 6 protocol files covering every game day: 5,909,379 bytes (5.64 MB), 37,096 entries.
- Raw `challenge` events: 319 distinct ids — 289 outgoing (challenger = `drewschessmachine`) and 30 incoming. By UTC day: 09-28 68, 09-29 73, 10-01 4, 10-02 40, 10-03 8, 10-06 126. A raw challenge event averages 645 bytes (maximum 709).
- Every one of the 206 game ids equals the id of a raw `challenge` event.
- `challenge-outcomes.json` holds 140 records, all from 2026-10-06.
- Parsing every protocol file and classifying (Appendix B): 0.23 s (0.19 s on the review rerun).
- No symbolic link, FIFO or other non-regular item under `LichessBot/` (212 `.jsonl` files, all regular files), so refusing them (§3.3) changes nothing on disk today.
- `F_FULLFSYNC` after a 1 KB append, 40 samples on this Mac with three training runs live (review, 2026-10-06): median 3.85 ms, p90 4.23 ms, max 7.94 ms.
- Lichess ids are lowercased usernames, and the protocol messages mix the two: "challenge outcome:" lines carry the opponent id (`maia1`), while "challenge sent to", "matchmaking send to … failed" and "withdrawing unanswered challenge to" carry the display name (`EdwardKillick`, `Cizme`). Pairing them by exact text pairs 25 of the 54 not-created attempts, and case-insensitive pairing pairs 42 (§3.7).

### 1.8 Do tests write into the real bot data folder? (checked; asked by the lead)

- **Not in the bot's data folder.** Every one of the 19 `LichessBotController(…)` constructions in `DrewsChessMachineTests/` passes a temporary `dataDirectory`. `.standard` is only the initializer's default (`App/LichessBotController.swift:415`), and nothing else in `LichessBot/` uses it.
- The real `Protocol/`, `Games/` and the other files there contain no fixture names (`FitBot`, `cbob`: no matches).
- **But the session log is polluted.** `SessionLogger.shared` writes to the real `~/Library/Logs/DrewsChessMachine` (`Logging/SessionLogger.swift:44`), so tests put `[LICHESS-BOT]` lines with fake opponents there. That folder holds 49 GB in 9,730 files. Out of scope here (§8, OD-14).

---

## 2. Design alternatives considered

### 2.1 Where the challenge record lives

- **A — a new append-only JSONL log, typed, one line per fact (chosen).** This is the record the owner asked for. Appending never rewrites earlier data, and a crash can only damage the last line, which the existing tail-cut handles.
- **B — extend `challenge-outcomes.json` and stop pruning it.** It is one JSON document rewritten on every change, so it grows without bound and is rewritten whole each time. It has no incoming side. Rejected.
- **C — keep relying on the protocol log.** Free text, no ids on the companion lines, never synchronized to disk. Fine as evidence for the back-fill (§3.7), not as the record. Rejected.

### 2.2 How a game's origin is stored

- **A — copy it into the game's journal at game start, carry it into the record and the index row, and join on the challenge id only for games without it (chosen).**
  - The record stays the one thing that describes its game (`Data/LichessBotGameRecord.swift:15-20`).
  - A resumed game reads it from its journal.
  - A tournament game, which has no challenge, still gets an origin.
  - A PGN can carry it.
- **B — store nothing on the game; always join game id ↔ challenge log.** One fewer copy, but every display depends on the challenge log having loaded, tournament games have nowhere to go, and records stop being self-contained. Kept as the fallback join for past games (§3.6), not as the primary store. OD-1.

### 2.3 Back-fill output

- **A — a separate, derived, regenerable file (chosen).** It is a pure function of the protocol files, rewritten whole when its inputs change, and written only when its bytes differ. That makes it idempotent and lets an improved algorithm replace it. The live log stays purely live.
- **B — append "reconstructed" lines into the day files.** Mixes inferred data into the append-only truth, and an append-only file can never correct a wrong inference. Rejected.

### 2.4 Who decides a game's origin

- **A — the controller, from its challenge ledger (chosen).** The ledger is the in-memory fold of the challenge log, loaded from disk, so it also knows challenges sent or accepted by an earlier run. The race in §1.3 is handled in one place (§3.4).
- **B — the session manager.** It knows pending outgoing and accepted incoming challenges, but not who sent them. It loses everything at a relaunch, and it hits the same race without a ledger to wait on. Rejected.

---

## 3. Design

### 3.1 Files

`Data/LichessBotDataDirectory.swift` gains:
- `challengesDirectory` (`<root>/Challenges/`);
- `challengeLogURL(for date: Date)` → `Challenges/challenges-YYYYMMDD.jsonl`, named in UTC with the same format style as `LichessBotProtocolLog.fileURL(for:)`, so a day's file never changes with the Mac's time zone;
- `reconstructedChallengesURL` → `Challenges/reconstructed-from-protocol.json`;
- the layout doc comment (`:5-17`) gains both entries; `createDirectories()` (`:82`) creates `Challenges/`.

Retention: **kept forever**; nothing prunes or rotates the day files away. Growth at the busiest measured day (126 challenges plus refusals, about 4 lines of about 1 KB each): about 0.5 MB/day, so at most about 180 MB/year at that pace (§8).

### 3.2 Line schema (`Data/LichessBotChallengeLog.swift`, new)

```swift
/// One line of `Challenges/challenges-YYYYMMDD.jsonl` (challenge-log plan §3.2).
struct LichessBotChallengeLogEntry: Sendable, Codable, Equatable {
    /// Bumped whenever a case or a field is added, so an older build can tell
    /// "written by a newer build" (skipped and counted) from corruption.
    static let currentSchemaVersion = 1
    let schemaVersion: Int
    let at: Date
    /// The writing build (`BuildInfo.buildNumber`, `BuildInfo.gitHash`).
    let build: Int
    let gitHash: String
    let event: LichessBotChallengeLogEvent
}

enum LichessBotChallengeLogEvent: Sendable, Codable, Equatable {
    // Outgoing: DCM's own sends.
    /// Lichess created the challenge (the POST answered with it).
    case outgoingCreated(challenge: LichessBotChallengeSnapshot, sender: LichessBotChallengeSender,
                         request: LichessBotOutgoingChallenge, opponentKind: LichessBotChallengeOpponentKind, creditCost: Int)
    /// A send that created no challenge. `attemptID` names the attempt
    /// (there is no challenge id).
    case outgoingNotCreated(attemptID: UUID, opponentID: String, sender: LichessBotChallengeSender,
                            request: LichessBotOutgoingChallenge, opponentKind: LichessBotChallengeOpponentKind?,
                            reason: LichessBotChallengeNotCreatedReason, creditCost: Int)
    /// Our own challenge echoed on the event stream with no created line for it
    /// in this run (§3.4, unmatched echoes).
    case outgoingSeenWithoutCreatedLine(challenge: LichessBotChallengeSnapshot, attribution: LichessBotEchoAttribution)
    case withdrawalRequested(challengeID: String, reason: LichessBotWithdrawalReason)
    case withdrawalResult(challengeID: String, result: LichessBotWithdrawalResult)
    // Incoming.
    case incomingReceived(challenge: LichessBotChallengeSnapshot)
    case incomingDecided(challengeID: String, decision: LichessBotIncomingDecisionRecord)
    case incomingResponseFailed(challengeID: String, error: String)
    // Either direction, as Lichess reported it on the event stream.
    case declinedOnLichess(challengeID: String, reason: LichessBotDeclineReasonRecord, text: String?)
    case canceledOnLichess(challengeID: String)
    /// A `gameStart` whose game id is a challenge in the log.
    case gameStarted(challengeID: String)
    // The writer's own housekeeping.
    /// An unterminated final line left by an interrupted write was cut
    /// before this append (the bytes are kept here).
    case unterminatedLineCut(byteCount: Int, base64: String)
}
```

The supporting types (new; the persistence schema stays separate from the API models, as the journal's does, `Data/LichessBotJournal.swift:3-6`):

| Type | Shape |
|---|---|
| `LichessBotChallengeSnapshot` | `id`; `challenger` as `LichessBotChallengeParty` (id, name, title, rating, provisional); `destUser: LichessBotChallengeParty?` (optional, as in `LichessBotChallenge`: an open challenge has none); `variant: LichessBotOpenValue<LichessBotVariantKey>`; `rated`; `speed: LichessBotOpenValue<LichessBotSpeed>`; time control (type as open value, limit and increment seconds, days per turn); `color: LichessBotOpenValue<LichessBotChallengeColorName>`; `finalColor: LichessBotOpenValue<LichessBotColorName>?`; `initialFen?`; `rematchOf?`. Built from `LichessBotChallenge` (`API/LichessBotAPIModels.swift:431-447`) in one `init(_:)`. |
| `LichessBotChallengeSender` | `.challengeSheet`, `.casualResendOffer` (the operator's Resend as Casual), `.challengeQueue`, `.matchmaking(trigger: LichessBotMatchmakingTrigger, fillMode: LichessBotMatchmakingSettings.FillMode)`, `.matchmakingCasualResend` |
| `LichessBotMatchmakingTrigger` | `.automaticPass`, `.fillOpenSlots` |
| `LichessBotChallengeNotCreatedReason` | `.opponentOffline`, `.refused(LichessBotChallengeRefusal)` (existing type), `.noAnswer(error: String)` — Lichess's answer never arrived, so a challenge may exist |
| `LichessBotEchoAttribution` | `.unansweredSend(attemptID: UUID, sender: LichessBotChallengeSender)` — exactly one `.noAnswer` send to that player within the window; `.notRecorded` |
| `LichessBotWithdrawalReason` | `.operatorCancel`, `.unansweredTimeout(seconds: Int)`, `.goingOffline`, `.wentOfflineWhileSending` |
| `LichessBotWithdrawalResult` | `.confirmed`, `.alreadyGone(message: String?)` (Lichess answered 400/404: expired or already answered), `.failed(error: String)`, `.abandonedAtShutdown` |
| `LichessBotIncomingDecisionRecord` | `.accept`, `.decline(reason: LichessBotDeclineReason, rule: String)`, `.ignore(rule: String)` — one mapping from `LichessBotChallengeDecision` |

**Not logged:** sends refused by DCM's own checks before any request (not online, slot or per-opponent limit, missing scope). They are not challenges; the protocol log and the UI already report them (OD-8).

**Decoding rule.**
1. Each line's `schemaVersion` is decoded first.
2. A line with a higher version than the build knows is **skipped and counted** (`skippedNewerLines`). Each load with skipped lines raises one alarm. A newer build's lines therefore never make an older build's day file unreadable.
3. A line at or below the build's version must decode. Anything else is corruption: the file is reported (file and line, like `LichessBotJSONLinesError.undecodableLine`) and left out of the ledger, with an alarm. Appends to it continue.
4. An unterminated final line is dropped and reported, as everywhere else.

Rationale: the journal's rule ("a newer case makes the file undecodable") would make one newer line cost a whole day.

### 3.3 Writer, durability, concurrency

`LichessBotChallengeLog` (`final class`, `Sendable`, `Data/LichessBotChallengeLog.swift`) mirrors `LichessBotProtocolLog`:
- `record(_ event:at:)` is synchronous and enqueues the append on the controller's general file queue (`LichessBotFileQueue`, the one the protocol log uses). Entries land in the order they were recorded.
- The first append to a file this launch hasn't vouched for runs `cutUnterminatedFinalLine`. Any cut bytes are written as an `unterminatedLineCut` line ahead of the entry, so the file itself records the repair. They are also logged as `[ALARM] LICHESS-BOT …`, as the protocol log does.
- **Durability: `F_FULLFSYNC` after every append** (OD-3). Challenge events are rare (Lichess caps sends at 25/min), and a full sync measured 3.85 ms median and 7.94 ms max here (§1.7). This is the one record the owner wants kept, so the protocol log's no-sync risk (§1.1) isn't repeated here. When an append **created** the day file, the `Challenges/` directory is also flushed (`FileSafety.fullSync(at:)` on the directory), so the new file's name survives a power loss along with its contents.
- **Where the sync runs.** It runs on the general file queue (`LichessBotFileQueue`, a serial `DispatchQueue` at `.utility`), never on the main actor and never on a cooperative-pool thread. `record` only enqueues. Nothing on the play path waits on that queue: journals use their own queue (`App/LichessBotController.swift:360-366`). So the cost is a few ms of delay to queued protocol-log and index work per challenge fact. Each sync is timed. One over 100 ms logs `[LICHESS-BOT] challenge log: slow sync <ms> ms`, and shutdown logs `[LICHESS-BOT] challenge log: <n> appends, slowest sync <ms> ms` (in memory only; nothing new is persisted for it).
- A failed append calls `onWriteFailure`, which the controller raises as an alarm. Nothing is dropped silently. After the queue is closed (shutdown), a refused append is written to the session log with its event, as `LichessBotFileQueue.enqueue` already does.
- Lock discipline: the "tail verified" set is a `SyncBox<Set<String>>` (`OSAllocatedUnfairLock`), read and changed only inside file-queue closures (the protocol log's pattern). No new `NSLock`, no actor.
- **Writers.**
  - Within the app, the controller's `recordChallengeEvent(_:)` is the only caller (§3.4).
  - The instance lock (`bot.lock`) allows one runtime per data folder, but it does **not** make the challenge log single-writer. A withdrawal's result (`withdrawalResult`, `.abandonedAtShutdown`) and the teardown's echo flush can be appended after the runtime's lock is released (§1.6). A second instance going online in that window appends to the same day file.
  - So every append takes an exclusive `flock` on the day file's own descriptor for the whole tail check, cut, write and sync (below). Two processes then never interleave a cut with an append. `O_APPEND` keeps every write at the file's end.

**FileSafety gains `openForAppending(at:)`** (OD-5):
- opens with `O_RDWR | O_APPEND | O_CREAT | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC`:
  - `O_RDWR`, so the tail cut reads and truncates through the **same** descriptor;
  - `O_NONBLOCK`, so a FIFO at the path is an error rather than a hang inside `open`, as in `openForWriting(.truncate)` (`Utils/FileSafety.swift:331`);
- `fstat`s the descriptor and refuses anything but a regular file with a `FileSafetyError` naming the path;
- returns the handle, the `FileIdentity` and whether this call created the file.

**`LichessBotJSONLines`** keeps one append path. `append(_:to:synchronize:)` becomes one locked step:
1. `openForAppending`;
2. `flock(LOCK_EX)`, blocking, on that descriptor (released by close);
3. if the caller hasn't vouched for the tail, the tail check and cut on that descriptor (`pread`, `ftruncate`), returning the cut bytes so the caller can record them;
4. write;
5. synchronize per `Synchronization` (`.none`, `.fsync`, `.fullSync`; `.fullSync` also flushes the directory when step 1 created the file);
6. close.

`cutUnterminatedFinalLine(of:)` stops using `FileHandle(forUpdating:)`. Today that open follows a symbolic link and runs **before** the append's own open, so a refusing open alone would come too late: the link's target would already have been truncated. Its core becomes descriptor-based, and the append calls that core on its locked descriptor.
- The path-taking `cutUnterminatedFinalLine(of:)` **keeps its signature**, because `LichessBotDataLayerHardeningTests.swift:162-166` calls it directly.
- It opens the existing file with `O_RDWR | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC`, with no `O_CREAT`, so a missing file is still an error and is never created. It refuses a non-regular file, takes the same `flock` and calls the core.

`LichessBotJournalWriter.append(_:gameID:synchronize: Bool)` also keeps its signature, because tests call it (`LichessBotDataLayerHardeningTests.swift:283`, `LichessBotGameIDPathSafetyTests.swift:88`). It maps `true` → `.fsync` and `false` → `.none`. Only `LichessBotJSONLines.append`, which no test calls, takes `Synchronization`.

The journal keeps `.fsync` where it passes `true` today and `.none` elsewhere, and the protocol log keeps `.none`. So the journal and the protocol log also stop following symbolic links, and they gain the cross-process lock. The behavior changes:
- a symbolic link, FIFO or directory at a journal or protocol-log path is now refused (an alarm) and left untouched, where today it is written through. None exists on disk today (§1.7);
- two instances' protocol-log appends now take turns per append, which costs microseconds.

### 3.4 The ledger and the one funnel (`App/LichessBotController.swift`, `Stats/LichessBotChallengeLedger.swift` new)

**`LichessBotChallengeLedger`** is a pure value: the fold of challenge-log entries into one row per challenge (keyed by challenge id) or per not-created attempt (keyed by `attemptID`). The controller holds it as `private(set) var challengeLedger: LichessBotChallengeLedger?`, nil until it has loaded. Each row holds:
- direction, opponent, terms, sender or decision;
- the times of each fact;
- the derived **state**, `LichessBotChallengeLogState`:

| State | When |
|---|---|
| `open` | No terminal fact. The UI shows **Waiting** when the id is in `pendingChallenges`, else **No answer recorded** — explicit, never blank. |
| `accepted(gameStarted:)` | Accepted, and whether its game was seen starting. |
| `declined(reason)` | Declined, with Lichess's reason. |
| `canceledByChallenger` | Incoming, withdrawn by the challenger. |
| `withdrawn(reason, result)` | DCM withdrew it; why, and what Lichess answered (`result` nil when no answer was recorded). |
| `notCreated(reason)` | The send created no challenge. |
| `incomingDecided(decision)` | Incoming, decided, no later fact. |
| `canceledOnLichessWithoutRecordedWithdrawal` | Outgoing, Lichess reported it canceled, and no withdrawal fact names it. Only the challenger can cancel, so DCM or another client using the token withdrew it, but why was not recorded. Happens live when a `withdrawalRequested` line was lost or another client canceled, and in reconstruction for 6 of today's challenges (§6.3). Shown as **Withdrawn (reason not recorded)**. |

**Fold precedence.** The fold is order-independent: facts may arrive in any order (the POST race, two day files), and the state is decided by the strongest fact present, never by the last one read:
1. a `gameStarted` → `accepted(gameStarted: true)`. It wins over every other fact. A game that started after a withdrawal attempt (the cancel answered 400/404 "already accepted", or the withdrawal raced the acceptance) or after a decline is still accepted, and the row keeps the other facts as notes ("withdrawal attempted: already gone"). An incoming challenge DCM declined or ignored that still started (the operator accepted it by hand on lichess.org) is `accepted` with the note "accepted outside DCM";
2. `notCreated` → `notCreated` (attempt rows have no other facts);
3. `declinedOnLichess` → `declined`;
4. a `withdrawalRequested` → `withdrawn(reason, result)`, where `result` is optional in the state: nil (shown as **result not recorded**) when no `withdrawalResult` line exists for it. `result == .alreadyGone` is shown as **No longer on Lichess** (expired or answered unseen), which replaces the draft's separate `noLongerOnLichess` state: one fact, one state;
5. `canceledOnLichess` → `canceledByChallenger` (incoming) or `canceledOnLichessWithoutRecordedWithdrawal` (outgoing);
6. an incoming decision → `incomingDecided`;
7. otherwise `open`.

A row's **sender** comes from its `outgoingCreated` when one exists. An `outgoingSeenWithoutCreatedLine` for the same id gives the sender only when there is no `outgoingCreated` (see the teardown case under "Unmatched echoes"). Two `outgoingCreated` lines for one id, which a code bug would cause, keep the first and add an anomaly to the row.

- **Load.** The ledger loads at go-online, before the runtime starts, beside player notes and outcomes (`:830-839`). It also loads when the bot window opens. Either load runs **only while `challengeLedger` is nil** (as `loadPlayerNotes` / `loadChallengeOutcomes` are guarded today), so a loaded ledger is never replaced by a re-read. Loading reads every day file on the general file queue, behind a continuation, and logs `[LICHESS-BOT] challenge log loaded: files=N lines=N bytes=… ms=… skipped_newer=N files_left_out=N`.
- **Events recorded while the load is in flight.** The load's read is enqueued on the file queue from the main actor. Every `recordChallengeEvent` before that point is in the files it reads. Every one after it is appended later, so it is not. Going online awaits the load before the runtime starts, so normally no event can precede it. The controller still keeps the events recorded after the read was enqueued in `eventsAwaitingLedger`, in order, and folds them on top of the loaded ledger when the load lands. The boundary is exact because both run on the main actor. If the load fails, they become the ledger's only content.
- **Load status.** The ledger carries `loadStatus`:
  - `complete`;
  - `partial(filesLeftOut: [file: reason])` — a file that failed to decode is left out, with an alarm naming it;
  - `failed(reason)` — the read itself failed. Going online still proceeds, because losing the challenge log must not stop the bot. The alarm stays up, appends still happen, and the in-memory ledger holds only this run's events.
- **What a miss means.** When a lookup misses on a ledger that is not `complete`, the origin is `undetermined(.challengeLogIncomplete)` rather than `.noChallengeRecord`: the challenge may be in the part that didn't load (OD-12 covers the load-time budget).
- **The funnel.** `recordChallengeEvent(_ event: LichessBotChallengeLogEvent)`, on the main actor, is the **only** mutation point. In one synchronous step it:
  1. applies the entry to the ledger, or adds it to `eventsAwaitingLedger` while the ledger is nil;
  2. enqueues the append;
  3. tells the origin resolver (§3.5).
  
  Its call sites:

| Fact | Where (today's code) | Event |
|---|---|---|
| POST answered with the challenge | `sendChallenge` right after `client.challenge` returns (`:1765`), beside `recordCreated` (`:1783-1785`) | `outgoingCreated` |
| Opponent offline | `:1751-1757` | `outgoingNotCreated(.opponentOffline)` |
| Lichess refused the POST | `:1772-1777` (`LichessBotChallengeRefusal.classify`) | `outgoingNotCreated(.refused)` |
| POST failed without an answer | `:1779` ("outcome not recorded") | `outgoingNotCreated(.noAnswer)` |
| Withdrawn: operator, timeout | `cancelChallenge(id:)` (`:1819-1841`) keeps its public signature (the UI's call) and forwards `.operatorCancel` to a new private `cancelChallenge(id:reason:)`; the timeout path (`:2984-2993`) calls the private one with `.unansweredTimeout`. `withdrawalRequested` is written after the existing `pending` guard, so a cancel of an id that isn't pending writes nothing, as today | `withdrawalRequested` then `withdrawalResult` (`.confirmed`, `.alreadyGone` on 400/404, `.failed`) |
| Withdrawn: going offline, created while going offline | `withdraw(challengeID:client:)` (`:1846-1863`) gains a `reason:` (`tearDownRuntime` `:2895` → `.goingOffline`; `:1790` → `.wentOfflineWhileSending`) | the same pair; `CancellationError` → `.abandonedAtShutdown` |
| Any challenge event on the stream | manager `.challengeArrived` (now carrying the `LichessBotChallenge`) | incoming: `incomingReceived` (once per id: Lichess replays open challenges on reconnect, and a replay writes nothing); our own echo: see "Unmatched echoes" below |
| Incoming decision | `.challengeDecision` (`:3178`) for a challenger ≠ us | `incomingDecided` |
| Accept/decline request failed | `.challengeResponseFailed` | `incomingResponseFailed` |
| `challengeDeclined` / `challengeCanceled` | new manager event `.challengeAnsweredOnStream` (every such line, either direction) | `declinedOnLichess` / `canceledOnLichess` |
| `gameStart` for a challenge id | new manager event `.gameStartReceived(LichessBotGameEventInfo)`, emitted before the session starts | `gameStarted` (once per id: skipped when the ledger already holds one, so a reconnect replay or a relaunch writes nothing) |

**When `gameStarted` is written.** In the POST race (§1.3), `gameStart` arrives while the challenge has no `outgoingCreated` line yet. A rule of "only for an id in the log" would then never record that the game started, and the row would stay `open`. Today's quick-accepting bots make this the common case for matchmaking. So `gameStarted` is written at `.gameStartReceived` when the id is in the ledger **or** among the held unmatched echoes (below). The echo of our own challenge precedes its `gameStart` on the same, ordered event stream. The controller also keeps the ids of this run's `gameStart`s in `gameStartsSeen`. When `outgoingCreated`, `outgoingSeenWithoutCreatedLine` or `incomingReceived` is recorded for an id in that set that has no `gameStarted` line, the `gameStarted` line is written right after it. That covers a `gameStart` whose echo was lost to a stream reconnect. The fold is order-independent (above), so the line order doesn't matter.

- **Manager event changes** (`Play/LichessBotSessionManager.swift`):
  - `.challengeArrived` carries the whole `LichessBotChallenge` instead of three fields (`:614`). No test matches its fields.
  - Two new cases are emitted from `handleEventLine` (`:562` ff.): `.challengeAnsweredOnStream(LichessBotChallengeReference, LichessBotChallengeStreamAnswer)` for every `challengeDeclined` / `challengeCanceled` line (`:586-590`), and `.gameStartReceived(LichessBotGameEventInfo)` before `startSessionIfNeeded` (`:573-581`).
  - `.gameSessionStarted` and `.challengeDecision` keep their shapes, because tests match them positionally (§5.1). The existing `.outgoingChallengeResolved` path is untouched; it still drives the pending bookkeeping and the outcome log.
- **Sender.**
  - `ChallengeOrigin` keeps `.manual`, which now means the Challenge sheet only (the doc comment says so), so `LichessBotCasualFallbackTests.swift:209` stays valid. It gains `.casualResendOffer` and `.challengeQueue`, and `.matchmaking` gains `trigger:`. `ChallengeOrigin.sender: LichessBotChallengeSender` is the one mapping to the persisted form.
  - `resendAsCasual` and the queue pump call the private send with their own origin.
  - `runMatchmakingPass(fillMode:)` (`:2224`) gains `trigger:`: Fill Open Slots passes `.fillOpenSlots`, the automatic pass `.automaticPass`. The casual-resend rules (`:2447-2459`) are unchanged: `.casualResendOffer` and `.challengeQueue` are the operator's, like `.manual`.
- **Unmatched echoes.**
  - Our own challenge is echoed on the stream, usually *before* the POST returns (measured: echo at 16:29:07.440, "challenge sent to" at .455). Writing a line for every echo would duplicate almost every send.
  - So the controller holds an unmatched echo in memory, keyed by id, and drops it when `outgoingCreated` for that id is recorded.
  - An echo whose id the ledger already holds (`outgoingCreated` or `outgoingSeenWithoutCreatedLine`, from this run or, after a relaunch, an earlier one) is **not held**. Lichess replays our open challenges when the stream reconnects, and those replays must write nothing.
  - An echo still unmatched after `2 × LichessBotRequestTimeouts.resource` (2 × 30 s: no POST of this run can still answer by then) is written as `outgoingSeenWithoutCreatedLine`. Its attribution is the single `.noAnswer` send to the same player (ids compared lowercased) inside that window, if there is exactly one, else `.notRecorded`. The poll loop checks this.
  - Teardown writes every echo still held with `.notRecorded`. Typical source: a challenge sent by another client using the token, or by an earlier run whose `outgoingCreated` line was lost.
  - **Teardown during a POST.** Going offline doesn't wait for sends in flight (`tearDownRuntime` runs at once; only `shutdown` waits, `awaitOutstandingChallengeTraffic`). A send whose echo arrived but whose POST hasn't returned is flushed as `.notRecorded`, and its `outgoingCreated` is recorded a moment later (`:1783`, before the went-offline check at `:1786`). Both lines then exist for one id, and the fold takes the sender from `outgoingCreated` (fold precedence, above). Nothing is rewritten.

### 3.5 Game origin

**Type** (`Play/LichessBotGameOrigin.swift`, new). It is named apart from `LichessBotSessionOrigin` (new / resumed session) and `ChallengeOrigin`, and the doc comment says so.

```swift
/// How a game DCM played on Lichess began (challenge-log plan §3.5).
enum LichessBotGameOrigin: Sendable, Codable, Equatable {
    /// DCM accepted a challenge someone sent it.
    case acceptedIncomingChallenge(challengeID: String, challengerID: String)
    /// A player accepted DCM's challenge; `sender` is who sent it.
    case outgoingChallengeAccepted(challengeID: String, sender: LichessBotChallengeSender)
    /// DCM's challenge (the echo proves the direction), sent by an earlier run
    /// or another client: who sent it was not recorded.
    case outgoingChallengeSenderNotRecorded(challengeID: String)
    /// Lichess paired DCM in a tournament.
    case tournament(source: LichessBotOpenValue<LichessBotGameSourceName>, tournamentID: String?)
    /// Not determined while the game was followed.
    case undetermined(source: LichessBotOpenValue<LichessBotGameSourceName>?, gap: LichessBotGameOriginGap)
}

enum LichessBotGameOriginGap: String, Sendable, Codable, Equatable {
    /// No challenge with the game's id was in the challenge log by the time
    /// the game's session ended.
    case noChallengeRecord
    /// The challenge log did not load completely (a day file left out, or the
    /// read failed), so the challenge may be in the part that is missing.
    case challengeLogIncomplete
}
```

- `LichessBotGameEventInfo.source` (`API/LichessBotAPIModels.swift:390`) becomes `LichessBotOpenValue<LichessBotGameSourceName>?`; it decodes the same string. The `LichessBotGameSourceName` cases are taken from Lichess's game-source enumeration (`lila`, verified at implementation time, not guessed). An unknown value is kept verbatim. Only `friend` has been observed here.

**Deciding it** (`LichessBotGameOriginResolver`, a pure `struct` held by the controller; every case is unit-tested):
- At `.gameStartReceived(info)`, remember `info` for the game.
- At `.gameSessionStarted(gameID, _, sessionOrigin)`:
  - **resumed, and the journal holds a determined origin** (`LichessBotResumedJournal.recordedOrigin`, below) → show it, write nothing;
  - otherwise, look the id up in the ledger:
    - an incoming row (the challenger is not us), **whatever DCM decided** → `acceptedIncomingChallenge`. A game started from it, so someone accepted it. When DCM's decision was not `accept`, the operator accepted it outside DCM (by hand on lichess.org), and the display detail says "accepted outside DCM (DCM decided: <decision>)";
    - `outgoingCreated` → `outgoingChallengeAccepted(sender)`;
    - `outgoingSeenWithoutCreatedLine(.unansweredSend(_, sender))` → `outgoingChallengeAccepted(sender)`;
    - `outgoingSeenWithoutCreatedLine(.notRecorded)` → `outgoingChallengeSenderNotRecorded`;
    - `info.source` known as arena or swiss → `tournament`;
    - **not known yet** (the POST race, or an echo still unmatched) → the game waits in `gamesAwaitingOrigin`.
- When `recordChallengeEvent` adds a row whose id is a waiting game → decide and write.
- At `.gameSessionEnded` for a game still waiting → `undetermined(source, .noChallengeRecord)`, or `.challengeLogIncomplete` when the ledger's `loadStatus` is not `complete` (§3.4). A resumed journal already ending in that same undetermined value is not written again.
- A determined origin is written **once** per game per run.

**Writing it.**
- New journal case `.gameOrigin(LichessBotGameOrigin)`, written by `LichessBotJournalWriter.recordOrigin(_:gameID:)`. This is the `recordRequest` pattern: the controller writing into a game's journal outside the session, through the same file queue, synchronized like a posted move (`.fsync`).
- A game already filed gets nothing written. The writer's existing "late journal entries not written: the game is filed" path logs it. The game then shows its origin through the challenge-log join (§3.6), so nothing is lost.
- `LichessBotJournal.schemaVersion` stays 1, per the journal's own rule that cases are added without changing it (`Data/LichessBotJournal.swift:7-12`). The downgrade cost is listed in §8.
- The three exhaustive switches handle the case:
  - the record builder keeps the **first determined** origin, else the last undetermined one (a determined origin always replaces an earlier undetermined one). A later determined origin that differs from the first adds an anomaly to the record;
  - the carryover fold ignores it;
  - the live view's replay sets the live game's origin.
- `LichessBotResumedJournal` gains `recordedOrigin: LichessBotGameOrigin?` (the same rule as the record builder: first determined, else last undetermined). Nil means "the journal holds none".

**Persisting it.**
- `LichessBotGameRecord` gains `let origin: LichessBotGameOrigin?`. Nil means "the journal held none": a game played before this feature, or a write that failed. The record schema stays 1: an optional field old builds ignore, as `LichessBotGenerationInfo.valueHeadRecenteredOnLoad` did.
- `LichessBotGameSummary` gains `let origin: LichessBotGameOrigin?`, copied in `init(record:)`.
- `LichessBotIndex.schemaVersion` goes up by one from its value at implementation time (2 → 3 if this lands before the record-stats plan's P2, 3 → 4 after it; §7), so the stored cache is discarded once and rebuilt from the records by the existing rule. It is a derived cache, never user data.
- **PGN:** new tag `DCMOrigin` for newly filed games only, beside `DCMModelIDs` / `DCMSources` (`Data/LichessBotPGNWriter.swift:46-47`). The value is a fixed token per case — `incoming`, `sheet`, `casual-resend-offer`, `queue`, `matchmaking-auto`, `matchmaking-fill`, `matchmaking-casual-resend`, `outgoing-sender-not-recorded`, `tournament`, `undetermined` — produced by one function. Existing PGNs are not rewritten (OD-10).

**The live game.** `LichessBotLiveGame` gains `private(set) var origin: LichessBotGameOriginDisplay?`. Nil means "not decided yet", shown as **Not yet known**. The controller sets it when it decides, and the replay sets it from a resumed journal.

**Logging.** Protocol `.game` "game origin: <label>", with fields `origin`, `challenge` and `basis`. Session log: `[LICHESS-BOT] game <id> origin: <label> (<detail>)`.

### 3.6 What a game shows: the one resolver (`Play/LichessBotGameOriginDisplay.swift`, new)

`LichessBotGameOriginDisplay.resolve(gameID:recorded:ledger:reconstructed:)` → `LichessBotGameOriginDisplay` holds:
- `category: LichessBotGameOriginCategory`: `incoming`, `challengeSheet`, `casualResendOffer`, `challengeQueue`, `matchmaking`, `matchmakingCasualResend`, `outgoingSenderNotRecorded`, `tournament`, `unknown`;
- `detail` (for example "automatic pass, every free slot · challenge `aBc123`");
- `basis: LichessBotOriginBasis`: `recorded`, `challengeLog`, `reconstructed(LichessBotReconstructionConfidence)`, `unknown(LichessBotOriginUnknownReason)`.

Order:
1. The record's determined origin → `recorded`.
2. Otherwise, the live ledger's row for the game id → `challengeLog`. This covers a lost journal write and a record that says `undetermined` when the log learned the answer later.
3. Otherwise, the reconstructed row (§3.7) → `reconstructed(confidence)`.
4. Otherwise, the record's `undetermined(gap)` → `unknown(.gap(gap))`.
5. Otherwise, the reason depends on when the game was created. Saying "played before origins were recorded" for every game without an origin would be a guess:
   - created before the challenge log's first entry (`liveLogFirstEntryAt`, §3.7), or while no challenge log exists → `unknown(.playedBeforeOriginsWereRecorded)`, shown as **Unknown — played before origins were recorded; no challenge with this id in the protocol log**;
   - otherwise → `unknown(.notRecorded)`, shown as **Unknown — no origin recorded for this game, and no challenge with this id in the challenge log**. This happens when the origin write failed, or when the game was filed by launch recovery without a session in this build.

- When the record and the ledger disagree, the record wins, and the disagreement is logged once per game at index load (`[LICHESS-BOT] origin disagreement for <id>: record … challenge log …`). It is never hidden.
- The controller computes `originsByGameID` whenever the index, the ledger or the reconstructed history changes, the way `recordsByOpponent` is computed in `index`'s `didSet` (`:208-217`). Views read that map and never resolve anything themselves.

### 3.7 Back-fill: reconstructing past challenges (`Stats/LichessBotChallengeReconstruction.swift`, new, pure)

**Inputs.**
- The protocol day files, in name order, lines in file order (file order is record order).
- Only entries timestamped **before the live log's first entry** (`liveLogFirstEntryAt`: the `at` of the first line of the oldest `challenges-*.jsonl`), so only files dated on or before that UTC day are read. With no live log, every entry is read.
  - The cutoff is a timestamp, not a day. On the live log's first day, the protocol file also holds that day's challenges recorded live. A not-created attempt has no challenge id, so a day cutoff would count it twice, once live and once reconstructed. Rows with an id would merely be shadowed.
  - Once the live log exists, the inputs that matter stop changing, and the parse cost doesn't grow with the protocol log.
- `ourAccountID` (the bot's stored account id). With none configured, reconstruction doesn't run, and the window says why.

**Algorithm v1** (the message forms are frozen: they are historical formats, matched exactly). **Every player-name comparison is case-insensitive**, comparing lowercased names: the messages mix Lichess ids (lowercase) and display names (§1.7). Exact-case matching pairs 25 of the 54 not-created attempts instead of 42.
1. **Raw stream events** (`kind == stream`, `fields.stream == event`, JSON message):
   - `challenge` → snapshot. Direction = outgoing if `challenger.id == ourAccountID`, else incoming (**certain**).
   - `challengeDeclined` / `challengeCanceled` → answers.
   - `gameStart` with a challenge's id → game started.
2. **"challenge sent to <name>"** with `fields.id` → an outgoing send with terms (`rated`, `clock` `L+I`, `color`).
3. **Companion lines** ("matchmaking sent a challenge to <name>", "matchmaking resent a challenge to <name> as casual", "challenge queue: sent <name>") pair with the latest unpaired send line to the same name, at most 5 s before. Each is written synchronously after its send on the main actor, with no suspension between (§1.2 call chain), so they sit right after it in the file.
   - An ambiguous match (two candidate sends) or no match leaves the companion unpaired and counted, never guessed.
   - **Matchmaking trigger and fill mode are not reconstructed.** A "Fill Open Slots" bracket can contain an automatic pass that was already running when the bracket opened (`:2129-2137`), so the bracket can't attribute sends.
4. **Incoming decisions** "<challenger>: accept|decline (<key>): <rule>|ignore: <rule>" with `fields.challenge`, only where `<challenger>` is not `ourAccountID`. Older builds logged "drewschessmachine: accept" on their own echoes (4 such lines here); those are skipped and counted.
5. **Not-created attempts:**
   - "challenge outcome: <opponent id> offline" / "challenge outcome: <opponent id> refused: …" (since 2026-10-01; the id is the opponent's, since there is no challenge). The sender comes from a failure companion within 5 s:
     - "matchmaking send to <name> failed:" → matchmaking (`paired`);
     - "challenge queue: skipped <name>:" / "challenge queue: dropped <name>:" (`:2036-2039`) → challenge queue (`paired`);
     - "matchmaking: casual resend to <name> not sent:" (`:2564`) → matchmaking casual resend (`paired`);
     - none → operator (`inferredFromAbsence`).
     
     Queue and casual-resend failures don't occur in today's data. Without these forms, they would be misread as operator sends.
   - Before the outcome lines existed: "<name> is at its bot-game limit …" (written only from a refused POST, `:1770`) with no outcome line within 5 s.
6. **Withdrawals:**
   - "withdrew challenge <id> on going offline" → `goingOffline, confirmed`.
   - "withdrawing unanswered challenge to <name> after N s" → `unansweredTimeout` for the newest open outgoing challenge to that name, with its result from the `challengeCanceled` that follows.
   - An outgoing `challengeCanceled` with neither line → `canceledOnLichessWithoutRecordedWithdrawal` (§3.4). Here these are probably the operator's cancels, but no line says so.
   - Today's data: 5 timeouts, 2 going offline, 6 without a recorded reason (§6.3).
7. **Confidence** of a reconstructed outgoing sender:
   - `certain`: a line carrying the id says it;
   - `paired`: a companion line, paired by name and adjacency;
   - `inferredFromAbsence`: no companion. Operator. A lost protocol write (never synchronized, §1.1) would make this wrong. The evidence is stronger than one missing line: every matchmaking pass writes "matchmaking pick: <name> …" before it sends (`:2272`), and so the row's `evidence` also records whether such a pick line for that name precedes the send within 60 s. Measured here: all 190 paired matchmaking sends have one, and none of the 99 inferred-operator sends do. So an inferred send is wrong only if two separate lines were both lost. Also, 98 of the 99 inferred sends (61 of the 62 inferred games) come before the first matchmaking line of any kind in the log (2026-09-29 16:14:40Z). The remaining one is AndoBot, 2026-10-02 22:49:08Z.
   
   Direction from a raw event is always `certain`.

**Output.** `Challenges/reconstructed-from-protocol.json` holds:
- `algorithmVersion`;
- `ourAccountID`;
- `inputs` (file, size, SHA-256);
- `liveLogFirstEntryAt` (nil, or the cutoff timestamp);
- `rows`, each a challenge or attempt with direction, opponent, terms, sender plus confidence, decision, state, and `evidence: [file:line]`;
- `counts` (including unpaired companions and skipped echo-accepts).

Rows are sorted by first evidence and encoded with sorted keys. There is no generation time in the file, so the same inputs give the same bytes.

**Running it (OD-6).**
- On the general file queue, behind a continuation, when the controller loads (bot window or go-online). It runs if the file is missing, or if its `algorithmVersion`, `ourAccountID` or `liveLogFirstEntryAt` differs from the current ones, or if the input list or any input's size differs (protocol files only grow). A changed account id changes every direction, so it must regenerate.
- The file is written with `FileSafety.writeNewFile` (absent) or `replaceRegularFile` (present), **only when the new bytes differ**. A rerun with unchanged inputs writes nothing, not even a modification-time change. When `writeNewFile` meets `alreadyExists`, the case where a second instance wrote it first, the file is read back. Identical bytes count as `unchanged`; different bytes go through the replace path. The race can never become an error alarm or a lost write.
- It is re-runnable by hand: a "Rebuild" button in the Challenge Log window.
- It never touches the live log, the records, the journals, the PGNs or `challenge-outcomes.json`.
- Logged: `[LICHESS-BOT] challenge history reconstructed: inputs=N files (X MB) rows=N (outgoing created N, not created N, incoming N); games: incoming N, matchmaking N, queue N, casual resend N, operator (inferred) N, unknown N; written|unchanged`.

**Game origins for past games** are the join of §3.6 step 3 (game id = challenge id), computed in memory. No second file.

### 3.8 The challenge outcome log after this (OD-2)

- The 24 h credit log (`challenge-outcomes.json`) records a subset of the same facts.
- **Through P5 both are fed from the one funnel.** `recordChallengeEvent` also applies the matching outcome-log change, so the two cannot disagree; the outcome log's own call sites move into the funnel.
- **P6 (if OD-2 is approved)** makes the outcome log an in-memory fold of the challenge log's last 24 h, plus the reconstructed rows for a day not yet covered by the live log. The controller then stops writing `challenge-outcomes.json`. The file is left on disk untouched, and `LichessBotChallengeOutcomeLog.load` keeps working.

### 3.9 UI

All new views get one file each under `LichessBot/UI/`. Glyphs and labels live in `LichessBotGameOriginStyle` (new, after `LichessBotStatusStyle`), the single place that maps a category to an SF Symbol, a short label and a long label.

| Category | Glyph | Short label |
|---|---|---|
| incoming | `arrow.down.left` | Incoming |
| challengeSheet | `arrow.up.right` | Sheet |
| casualResendOffer | `arrow.uturn.right` | Resend offer |
| challengeQueue | `list.bullet` | Queue |
| matchmaking | `wand.and.stars` | Matchmaking |
| matchmakingCasualResend | `arrow.uturn.right.circle` | Casual resend |
| outgoingSenderNotRecorded | `arrow.up.right` | Outgoing (sender not recorded) |
| tournament | `trophy` | Tournament |
| unknown | `questionmark` | Unknown |

- **Basis marker.** `inferredFromAbsence` adds a trailing "≈" and secondary color; `unknown` uses secondary color. `help()` always gives the long label, the detail, the basis and the evidence (for example "Reconstructed from the protocol log: 'challenge sent to X' with no matchmaking or queue line beside it (inferred)"). Glyph meanings stay consistent across all views.
- **`LichessBotGameOriginLabel`** (glyph + short label + marker) and **`LichessBotGameOriginGlyph`** (glyph only, with help) are the two building blocks.
- **All Games window:**
  - a sortable **Origin** column after Color (sort by category order);
  - an **Origin** filter menu in the header (All, or one category), held in `@State`;
  - the filtered rows are computed once per change into a `@State` array (not in `body`);
  - the summary line gains the per-category counts of the shown rows.
- **Recent games list:** a glyph column after the color disc.
- **Live picker:** `menuTitle` appends " · <short label>", or " · origin not yet known".
- **Tile:** a glyph beside the opponent's name.
- **Game detail:** an "Origin" row with the long label and the detail.
- **Challenge Log window** (`LichessBotChallengeLogWindowController` + `LichessBotChallengeLogView` + `LichessBotChallengeLogFilterBar`; the row model `LichessBotChallengeLogRow` is a plain struct):
  - Opened from a **Challenge Log…** button in the outcomes card's header; one window at a time, like the All Games window.
  - Columns: When (monospaced), Direction (glyph), Opponent (favorite star, title, name), Rating, Terms ("3+2 · rated", color), Sender / Decision, State (with decline reason key), Game (link when it started), Credits, Source ("Live", or "Reconstructed (certain | paired | inferred)").
  - Filters: date range (24 h, 7 days, 30 days, All), direction (All / Outgoing / Incoming), state, sender, an "Include reconstructed" toggle (default on), and a search on the opponent.
  - A footer with counts for the shown rows, the ledger's load status (files, lines, skipped newer lines, files left out and why), and the reconstruction status with its **Rebuild** button.
  - Rows are built from the ledger plus the reconstructed rows; live rows win on an id both hold.

### 3.10 Documentation

- `documentation/plans-active/LICHESS_BOT_PLAN.md` §10.1's layout gains `Challenges/` (the text is kept, with the new rows added).
- `CHANGELOG.md`.
- This file's status.
- ROADMAP.md: **nothing until the owner expressly approves OD-11** (the standing rule: new plans are added to the ROADMAP only with express permission). Today OD-11 is a recommendation, not a decision. If approved, the line is added in P1's commit and marked complete in P7, with nothing removed from it. If not approved, P7 skips this item.

---

## 4. Edge cases

| Case | Handling |
|---|---|
| `gameStart` handled before the POST that created its challenge returns (§1.3) | The game waits in `gamesAwaitingOrigin`. `gameStarted` is written at once, because the echo is held (§3.4, "When `gameStarted` is written"). `outgoingCreated` arrives a moment later and the origin is written then. |
| Echo of a challenge an earlier run sent, replayed on reconnect | The ledger, loaded from disk, already holds its `outgoingCreated`, so the echo is not held and nothing is written. |
| Going offline while a send's POST is in flight | The teardown flushes the echo as `.notRecorded`, and the POST's `outgoingCreated` follows. The fold takes the sender from `outgoingCreated`. |
| Operator cancels, but the challenge was already accepted (cancel → 400/404, then `gameStart`) | `withdrawalRequested(.operatorCancel)` and `withdrawalResult(.alreadyGone)`, then `gameStarted`. The state is `accepted` (it beats withdrawal), with the note "withdrawal attempted: already gone". The origin is `outgoingChallengeAccepted(.challengeSheet)`. |
| Withdrawn while going offline, but accepted first | The same: `accepted`, origin with the sender from `outgoingCreated`. The game is resumed or filed by launch recovery at the next go-online. |
| Incoming challenge DCM declined or ignored, accepted by hand on lichess.org | `acceptedIncomingChallenge`, with the detail "accepted outside DCM (DCM decided: …)". The row's state is `accepted`, with the decision kept as a note. |
| Outgoing challenge canceled on Lichess with no withdrawal line (another client, or a lost line) | `canceledOnLichessWithoutRecordedWithdrawal`, shown as **Withdrawn (reason not recorded)**. |
| Ledger still loading when a challenge fact is recorded | Held in `eventsAwaitingLedger` and folded on top when the load lands (§3.4). Nothing is lost or double-counted. |
| A day file left out (undecodable) and a game whose challenge is in it | `undetermined(.challengeLogIncomplete)` at session end, never `.noChallengeRecord`. The display's step 2 finds the row once the file is repaired. |
| Two instances: one's withdrawal result lands after the other took the lock | Each append holds `flock(LOCK_EX)` on the day file for the cut, write and sync (§3.3), so the lines interleave whole and nothing is cut from under the other process. |
| POST failed with no answer, but Lichess created the challenge | `outgoingNotCreated(.noAnswer)`. The echo stays unmatched, then is written with `.unansweredSend(sender)`, so the game's origin keeps its sender. |
| Challenge sent by an earlier run, accepted after a relaunch | The ledger (loaded from disk) has its `outgoingCreated` → origin with sender. |
| Incoming challenge accepted, app relaunched before `gameStart` | The ledger has `incomingDecided(accept)` → `acceptedIncomingChallenge`. |
| Reconnect replays open challenges and running games | `incomingReceived` and `gameStarted` are written once per id. Decisions are written each time, since each is a real request. |
| Tournament game | `source` arena / swiss → `tournament`. No challenge row is involved. |
| A game nobody can explain | `undetermined(.noChallengeRecord)` at session end, shown as **Unknown** with the reason. |
| Resumed game whose journal predates this feature | `recordedOrigin` is nil → ledger lookup → written if found, else waits like a new game. |
| Resumed game whose journal can't be read | Ledger lookup. The journal writer appends as it does for every other event. |
| Origin decided after the game was filed | The journal write is refused by the existing filed-game rule (logged). The display falls back to the challenge-log join (§3.6 step 2). |
| A challenge's facts span two UTC days | The fold reads all files, so rows cross files naturally. |
| Crash mid-append | An unterminated final line, cut before the next append and recorded as `unterminatedLineCut`. Reading drops it and reports it. |
| Corrupt complete line | That file is left out of the ledger, with an alarm. Appends continue. |
| Line written by a newer build | Skipped and counted, with an alarm per load (§3.2). |
| Two DCM instances | `bot.lock` lets one runtime record live sends. Late withdrawal results are covered by the per-append `flock`. Reconstruction in both produces identical bytes, and the loser of a create race reads the file back and reports `unchanged` (§3.7). |
| No account configured | Reconstruction doesn't run. The window says "No Lichess account is configured". |
| Two sends to the same player within 5 s in old logs | The companion pairing is ambiguous, so it is left unpaired and counted, never guessed. |
| Rematch challenge | Incoming or outgoing like any other; `rematchOf` kept in the snapshot. |
| Clock, time zone | File names in UTC; times shown in local time. |

---

## 5. Tests (new files only)

Pure logic first. All file tests use `FileManager.default.temporaryDirectory`, and the controller tests pass a temporary `dataDirectory`, as all 19 existing ones do.

- **`FileSafetyAppendTests`**:
  - `openForAppending` creates and appends;
  - it refuses a symbolic link, a directory and a FIFO, each with a `FileSafetyError` naming the path; the FIFO case returns promptly (`O_NONBLOCK`), not a hang; the symbolic link's target is byte-for-byte unchanged afterwards, **including when its end is unterminated** (the cut must not reach it);
  - the creation flag is true only for the call that created the file;
  - two handles both land at the end (`O_APPEND`);
  - the `.fullSync` path issues `F_FULLFSYNC`, through a seam that records the call, and also flushes the directory when the call created the file;
  - the append holds `flock(LOCK_EX)`: a second descriptor's `flock(LOCK_EX | LOCK_NB)` taken from inside the append's write seam fails with `EWOULDBLOCK`, and succeeds after the append returns;
  - the tail cut runs on the append's own descriptor: an unterminated tail is cut and returned, and a terminated file is untouched.
- **`LichessBotJSONLinesSynchronizationTests`**: `.none` / `.fsync` / `.fullSync` reach the right call; a symbolic link at a journal path and at a protocol-log path is refused, with an alarm, and its target unchanged.
- **`LichessBotChallengeLogSchemaTests`**:
  - a golden JSON line for every event case decodes and re-encodes byte-for-byte (sorted keys), which pins the format;
  - a line with `schemaVersion = current + 1` is skipped and counted;
  - a complete bad line at the current version is reported with file and line;
  - an unterminated tail is dropped and reported.
- **`LichessBotChallengeLogWriterTests`**:
  - per-UTC-day file naming across a day boundary in a non-UTC time zone;
  - tail cut plus `unterminatedLineCut` line;
  - a write failure reaches `onWriteFailure`;
  - an append after queue close is not written and is logged.
- **`LichessBotChallengeLedgerTests`** (pure fold): every state in §3.4, including:
  - accepted with and without a seen game start;
  - declined with a known, unrecognized and unstated key;
  - each withdrawal reason × result;
  - not created × each reason;
  - incoming accept, decline and ignore, then canceled by the challenger;
  - replayed `incomingReceived` written once;
  - facts out of order (`gameStarted` before `outgoingCreated`);
  - rows across two day files;
  - every fold-precedence pair of §3.4, including `gameStarted` after `withdrawalResult(.alreadyGone)` → `accepted` with the note; `gameStarted` after an incoming `decline` → `accepted` with "accepted outside DCM"; `withdrawalRequested` with no result → `withdrawn(_, nil)`; outgoing `canceledOnLichess` alone → `canceledOnLichessWithoutRecordedWithdrawal`;
  - `outgoingSeenWithoutCreatedLine(.notRecorded)` followed by `outgoingCreated` for the same id → the sender from `outgoingCreated`; two `outgoingCreated` → the first kept, plus an anomaly;
  - every permutation of a row's facts folds to the same row (order independence);
  - `loadStatus` `partial` / `failed` turns a miss into `.challengeLogIncomplete`.
- **`LichessBotUnmatchedEchoTests`**:
  - an echo matched by `outgoingCreated` writes nothing;
  - unmatched past the window with exactly one `.noAnswer` send to that player → `.unansweredSend`;
  - with zero or two such sends → `.notRecorded`;
  - teardown writes the remaining echoes;
  - an echo whose id the ledger already holds (an earlier run's `outgoingCreated`, loaded from disk) is not held, and nothing is written;
  - `.noAnswer` attribution matches the player case-insensitively (`EdwardKillick` vs `edwardkillick`).
- **`LichessBotGameOriginResolverTests`**:
  - incoming accepted;
  - each `LichessBotChallengeSender` (sheet, resend offer, queue, matchmaking automatic and fill, casual resend);
  - the POST race: `gameStartReceived` and `gameSessionStarted` before `outgoingCreated`, so `gameStarted` is written at once (the echo is held) and the origin is written when `outgoingCreated` arrives;
  - an incoming row decided `decline` or `ignore` whose game starts → `acceptedIncomingChallenge` with the outside-DCM detail;
  - the echo-only cases;
  - tournament (arena, swiss);
  - an unknown `source` kept verbatim;
  - undetermined at session end;
  - resumed with a determined origin (nothing written), with none, and with an undetermined one later determined;
  - written once per game per run;
  - a miss on an incomplete ledger at session end → `.challengeLogIncomplete`.
- **`LichessBotGameOriginJournalTests`**:
  - the record builder keeps the first determined origin, and a determined origin replaces an earlier undetermined one;
  - two different determined origins → anomaly, first kept;
  - an old journal without the event → `origin == nil`;
  - `LichessBotResumedJournal.recordedOrigin`;
  - the live view's replay sets the origin;
  - the carryover fold is unchanged by the new case.
- **`LichessBotIndexOriginTests`**:
  - a stored `index.json` at `LichessBotIndex.schemaVersion - 1` is rebuilt at `LichessBotIndex.schemaVersion` (symbolic, §7: never literal version numbers);
  - an old record without `origin` gives a nil row field;
  - a new record's origin reaches its row.
- **`LichessBotGameOriginDisplayTests`**: the §3.6 order, each step; step 5's two reasons, split by the game's creation time against `liveLogFirstEntryAt` (and no live log at all); the disagreement log line; every category's labels come from `LichessBotGameOriginStyle`.
- **`LichessBotChallengeReconstructionTests`** (synthetic protocol lines built in the test, one per message form):
  - direction from the challenger;
  - each companion pairing, including ambiguous and missing, with names that differ only in case;
  - not-created pairing with each failure companion (matchmaking, queue skipped/dropped, casual resend not sent) and with none;
  - the pick-line evidence flag;
  - an outgoing cancel with no withdrawal line → `canceledOnLichessWithoutRecordedWithdrawal`;
  - old echo-accept lines skipped;
  - outcome-line and bot-limit refusals, without double counting;
  - withdrawals;
  - input cut-off at the live log's first **entry**: a not-created attempt on the live log's first day, before the cutoff, is reconstructed, and one after it is not;
  - a changed `ourAccountID` or `liveLogFirstEntryAt` regenerates;
  - a create race (the file appears between the check and `writeNewFile`) with identical bytes → `unchanged`, no alarm;
  - same inputs → identical bytes;
  - a rerun with unchanged inputs writes nothing (modification time unchanged);
  - a grown input regenerates.
- **`LichessBotChallengeLogControllerTests`** (fake Lichess, as in `LichessBotCasualFallbackTests`):
  - a send from the sheet, the queue, Resend as Casual, Fill Open Slots, an automatic pass and the automatic casual resend writes `outgoingCreated` with the right sender;
  - offline, refused and no-answer sends;
  - operator cancel, timeout withdrawal and going-offline withdrawal;
  - incoming accept, decline and ignore;
  - a game from each path shows the right origin in `originsByGameID` and in its record after filing;
  - the POST race end to end: the fake Lichess streams the echo and `gameStart` and accepts before answering the POST, and the game's record gets the right sender;
  - relaunch mid-game: a second controller on the same temporary `dataDirectory` resumes a game whose challenge the first sent. The origin comes from the journal when it was written, else from the ledger loaded from disk;
  - an operator cancel that Lichess answers 400 (already accepted), followed by `gameStart` → state `accepted`, origin `challengeSheet`;
  - going offline during a POST → both lines; the fold's sender is from `outgoingCreated`;
  - an event recorded while the ledger load is held (the file queue blocked by the test) is in the ledger after the load, exactly once;
  - every file written lies under the temporary `dataDirectory`, and nothing under `LichessBotDataDirectory.standard`.
- **`LichessBotPGNOriginTagTests`**: `DCMOrigin` for each case (if OD-10).
- **`LichessBotChallengeLogViewRenderTests` / `LichessBotGameOriginViewRenderTests`**: the new views render, in the existing render-test pattern, with empty, live-only, reconstructed-only and mixed data, and with every category and basis.

### 5.1 Existing-test edits: none planned

Checked against `DrewsChessMachineTests/`:
- `.gameSessionStarted` and `.challengeDecision` are matched positionally (`LichessBotReadyBeforePlayManagerTests.swift:59`, `:127`; `LichessBotMultipleChallengesTests.swift:95`; `LichessBotGroupBFixTests.swift:101`), so their shapes stay.
- No test matches `.challengeArrived`.
- New manager cases don't break `if case` matching, and no test switches exhaustively over `LichessBotManagerEvent` or `LichessBotJournalEvent` (the one `switch event` in `LichessBotDataLayerHardeningTests.swift:393` is over reconciler events and has a `default`).
- `ChallengeOrigin.manual` keeps its meaning for the public send (`LichessBotCasualFallbackTests.swift:209`).
- No test constructs `LichessBotGameRecord` or `LichessBotGameSummary` directly, or pins `LichessBotIndex.schemaVersion`.
- No test calls `LichessBotJSONLines.append`. The two APIs of the append path that tests do call keep their signatures (§3.3): `LichessBotJSONLines.cutUnterminatedFinalLine(of:)` (`LichessBotDataLayerHardeningTests.swift:162-166`; its three expectations hold under the new implementation) and `LichessBotJournalWriter.append(_:gameID:synchronize:)` (`LichessBotDataLayerHardeningTests.swift:283`, `LichessBotGameIDPathSafetyTests.swift:88`).
- No test calls `cancelChallenge(id:)`, `runMatchmakingPass` or the private `sendChallenge(to:request:origin:)`. The public `sendChallenge(to:request:)` and `fillOpenSlots()` keep their signatures (callers: `LichessBotOfflineQuitTests.swift:89`, `LichessBotChallengeQueueControllerTests.swift:209`, `:312`, `LichessBotQuitWithdrawalTests.swift:113`, `:145`, `LichessBotCasualFallbackTests.swift:100`, `:208`, `LichessBotBotLimitTests.swift:95`). `LichessBotCasualFallbackTests.swift:154` compares to `.matchmakingCasualResend`, which keeps its shape.
- No test asserts `LichessBotGameEventInfo.source` (`LichessBotAPIModelsTests.swift:125-165` decodes `"friend"` and `"ai"` lines and asserts other fields), so it can become an open value.
- (Re-checked in review, 2026-10-06, against `dad00b85`.)

If implementation finds an edit is needed after all, it stops and lists it for the owner's approval, as the follow-lineage plan's §5.1 did.

---

## 6. Validation

1. **Targeted tests.** Every new test class passes, run by `-only-testing:` per phase. **Full suite** before merging, because this touches persistence. All tests pass, none modified.
2. **Builds** without new warnings, through drews-xcode-mcp.
3. **Back-fill on this Mac (P4).** Open the bot window offline. `[LICHESS-BOT] challenge history reconstructed …` must report exactly what Appendix B reports on the same inputs. On today's files (if no new games are played first), that is:
   - **rows 373**:
     - outgoing created 289: accepted 194, declined 82, withdrawn 13 (unanswered timeout 5, going offline 2, canceled on Lichess without a recorded withdrawal 6), open 0. The review corrected this line: the two "no answer" rows were the going-offline withdrawals `TTuj7kiR` and `l2RWQlRP` (no `challengeCanceled` came back, since the stream was closing), and "canceled 11" is the 5 timeouts plus the 6 unexplained;
     - not created 54: all refused (39 outcome lines + 15 bot-limit lines before outcome lines existed); sender paired 42 (all matchmaking), operator inferred 12. This needs case-insensitive pairing (§3.7); exact case gives 25/29;
     - incoming 30: accepted 12, declined 18 by DCM's decision (17 with Lichess's `challengeDeclined`; `l4qASCK5` has the decision line only);
   - companion lines unpaired: 0; own-echo accept lines skipped: 4;
   - **games 206**: incoming 12, matchmaking 132, queue 0, casual resend 0, operator (inferred) 62, unknown 0.
   
   If games are played before P4 lands, rerun Appendix B (games) and Appendix B.2 (row states) on the then-current files and compare with those.
4. **Idempotence.** Close and reopen the window: the log says `unchanged`, and the file's SHA-256 and modification time are unchanged. Remove nothing; protocol files only grow.
5. **Read-only proof.** Before and after P4, `shasum` every file under `Games/`, plus `index.json` and `challenge-outcomes.json`: identical. The exception is P3, where the schema bump rebuilds `index.json` once. After P3, its `schemaVersion` is the version after the bump (one more than the stored file had before P3; no literal number, §7), each row has an `origin` key or none, and a second launch does not rebuild it. For `Protocol/`, which the app itself may append to while the window is open, each pre-existing file's first *old-size* bytes hash the same (only appended, never rewritten).
6. **Live (P2–P3), on lichess.org.** Go online and send:
   - one challenge from the sheet;
   - one queued challenge;
   - one by Fill Open Slots;
   - let an automatic pass send;
   - let a rated decline with `casual` trigger the automatic resend (with `fallBackToCasual` on);
   - cancel one by hand;
   - let one time out.
   
   Accept one incoming challenge from the owner's own account and decline one (an unaccepted time control). Check:
   - each line in today's `Challenges/challenges-*.jsonl` (sender, state);
   - the Challenge Log window;
   - each game's Origin in All Games, Recent, Live picker, tile, detail, its filed record (`origin`) and PGN (`DCMOrigin`).
   - every accepted challenge has a `gameStarted` line, including matchmaking games accepted before the POST returned (§3.4), and no challenge from this session is left `open` once it is answered.
7. **Crash.** `kill -9` the app mid-game after a challenge line. Relaunch and go online. The resumed game keeps its origin (from the journal). Any cut tail shows as `unterminatedLineCut` plus an `[ALARM]`.
8. **Downgrade check (documented, not run on real data).** In a temporary data folder: a journal with `.gameOrigin` can't be decoded by a pre-feature build (the expected, documented cost, §8); a day file with a newer-schema line is skipped and counted by this build.
9. **Load cost.** The `challenge log loaded` line's `ms` is recorded in the CHANGELOG entry; OD-12's trigger is checked against it.

---

## 7. Phasing (build + commit per phase)

**Sequencing** (revised in review, 2026-10-06). Two other active plans touch the same code: `LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md`, being implemented now (P-ready landed at `a7803ac0`; P0 is in the working tree, dirty in `Play/LichessBotModelSlots.swift` and `UI/LichessBotModelSwitchStatusView.swift`), and `LICHESS_BOT_RECORD_STATS_PLAN.md` (in review).

- **P1 may start now.** It touches only `Utils/FileSafety.swift`, `Data/LichessBotJSONLines.swift`, `Data/LichessBotJournal.swift` (the Bool → `Synchronization` mapping inside the writer), `Data/LichessBotProtocolLog.swift`, `Data/LichessBotDataDirectory.swift` and new files. Follow-lineage edits none of these: its §3.2 only reads `FileSafety.FileIdentity` and `existingItem`. Record-stats doesn't edit them either. Before starting, confirm that with `git status` / `git diff`, and never stage another agent's dirty files.
- **P2 onward start after follow-lineage has landed.** Its P6 is committed. Both plans touch:
  - `App/LichessBotController.swift`: `startRuntime`, the poll loop, `handle`, the go-online loads, game-start logging. Follow-lineage P3 publishes `lineageFollowStatus` from the poll loop and logs `step=`/`cum=` at game start;
  - `Play/LichessBotSessionManager.swift`;
  - `Play/LichessBotGameInterfaces.swift`: follow-lineage P3 adds `LichessBotGenerationInfo.lineage`. This plan adds nothing there, but its record and journal changes decode beside it;
  - `Data/LichessBotGameRecord.swift` (generations vs. origin);
  - `Data/LichessBotPGNWriter.swift` (if it adds tags);
  - `UI/LichessBotOverviewView.swift` and the model card; `UI/LichessBotGameDetailView.swift`;
  - `documentation/plans-active/LICHESS_BOT_PLAN.md` and `CHANGELOG.md`.
- **Record-stats.** It also edits `Data/LichessBotIndex.swift` (`LichessBotGameSummary.facts`, the schema bump), the controller's index pipeline and `UI/LichessBotRecordCard.swift`, where this plan's P5 adds the Recent list glyph. Whichever of record-stats P2 and this plan's P3 lands second rebases on the first. Neither waits on the other's design: each adds one optional row field.
- **Index schema version, one rule for both plans.** `LichessBotIndex.schemaVersion` is bumped by **exactly one from the value in the tree when the change is implemented**: 3 for whichever of the two lands first, 4 for the second. A third index change later takes the next number the same way.
  - Both plans' index tests refer to the version **symbolically**. The fixture is a stored index at `LichessBotIndex.schemaVersion - 1`, and the assertion is that the rebuilt file has `LichessBotIndex.schemaVersion`, never a literal 2, 3 or 4. So the second landing doesn't break the first's test, which would be a test edit needing the owner's approval.
  - The validation steps say "the version after the bump" rather than a number.
  - This plan's `LichessBotIndexOriginTests` and §6.5 are worded that way, and the same rule was sent to the record-stats plan's reviewer for its `LichessBotIndexFactsTests` and validation step 2.
  - A single combined bump was considered and rejected: it would couple two independently approved plans' phases for no gain, since each rebuild costs one pass over the records (about 0.2 s here).

Re-check every `file:line` here against the tree at that time before each phase.

| Phase | Work | Tests run |
|---|---|---|
| **P1 — file layer** | `FileSafety.openForAppending`; `LichessBotJSONLines.append` on it with `Synchronization`; data-directory paths; the challenge-log entry schema, writer and reader; `LichessBotChallengeLedger` (pure). Nothing calls the writer yet. | P1 test classes; existing `LichessBotDataLayer*` and journal tests |
| **P2 — recording** | `ChallengeOrigin` cases and trigger; manager event changes; `recordChallengeEvent` funnel at every §3.4 site (the outcome log moves into it); ledger load at go-online and window open; unmatched echoes. | P2 classes; existing `LichessBot*Controller*`, `CasualFallback`, `ChallengeQueue`, `QuitWithdrawal`, `MultipleChallenges`, `SessionManager` tests |
| **P3 — game origin** | Origin types; `source` as open value; resolver; journal case and `recordOrigin`; record field and builder; resume fold; live game; index field and the schema bump (+1 from the then-current version, §7 "Index schema version"); PGN tag (OD-10); logging. | P3 classes; existing record, resume, index and PGN tests |
| **P4 — back-fill** | Reconstruction (pure) plus its runner and file; resolver step 3; validation 3–5 on this Mac. | P4 classes |
| **P5 — UI** | Style, label, glyph; All Games column and filter; Recent list; Live picker; tile; detail; Challenge Log window and card button. | render tests |
| **P6 — outcome log from the challenge log** (only if OD-2) | The fold, and stopping `challenge-outcomes.json` writes. | existing `LichessBotChallengeOutcomeTests` unchanged and passing |
| **P7 — docs** | §3.10 (the ROADMAP item only if OD-11 was approved). | full suite |

Each phase: all work → recheck → build → its tests → commit (`git add` of its own files only). The full suite runs at P3 (persistence) and at P7.

---

## 8. Risks

- **Downgrade.**
  - A journal holding `.gameOrigin` can't be decoded by a pre-P3 build, so that build can't file or resume the game. This follows the journal's own stated rule (`Data/LichessBotJournal.swift:7-12`); it applies only to games played on the new build and left unfiled when downgrading.
  - A pre-P3 build seeing the bumped `index.json` rebuilds it at its own version: harmless cache churn, repeated on each switch between two builds.
- **Growth and load time.** About 0.5 MB/day at the busiest measured day; the ledger loads every day file. Measured and logged on every load. A cached index of rows (like `index.json`) is deferred until a load exceeds 1 s (OD-12).
- **Inferred operator sends** (62 games here) would be wrong wherever a matchmaking or queue companion line was lost. The protocol log is never synchronized to disk, so a power loss can lose lines. They are marked "≈" and explained; never shown as certain.
- **The POST race** leaves a game briefly "Not yet known" in the live view. Bounded by one POST.
- **Message-format coupling.** Reconstruction parses historical protocol messages, which are frozen in this plan's text and covered by fixture tests. Later wording changes don't matter: once the live log exists, inputs stop at its first entry.
- **Merge conflicts** with the follow-lineage and record-stats work in the same files. Mitigated by sequencing (§7): P1 is disjoint from both, P2+ wait for follow-lineage, and the index bump follows one +1 rule.
- **`F_FULLFSYNC`** on the general file queue delays queued protocol-log and index work by a few milliseconds per challenge fact (measured 3.85 ms median, 7.94 ms max, §1.7). Acceptable at ≤ 25 sends/min. Nothing on the play path waits on that queue. A sync over 100 ms is logged (§3.3).
- **Per-append `flock` on the journal and protocol log** (§3.3). Each append adds a lock and unlock (microseconds) on the play path's journal queue. The lock is uncontended for a journal, which has one writer.
- **Reconstruction cutoff by timestamp** relies on the live log's first line being written by the build that started recording. A pre-feature build run after that point (another worktree's build, a downgrade) sends challenges that appear in neither the live log nor the reconstruction. Those games show **Unknown — no origin recorded …** (§3.6 step 5), never a guess.
- **Related, not fixed here:** tests write `[LICHESS-BOT]` lines with fake opponents (FitBot, cbob) into the real `~/Library/Logs/DrewsChessMachine` through `SessionLogger.shared` (`.userLibraryLogs`, `Logging/SessionLogger.swift:44`). That folder is 49 GB in 9,730 files. The bot's own data folder is not written by tests (§1.8). OD-14.

---

## 9. Non-goals

- No Lichess API calls to back-fill. Lichess has no endpoint listing a bot's past challenges.
- No rewrite of any existing game record, journal, PGN, protocol file, index row or `challenge-outcomes.json` (the index is a derived cache, rebuilt by its own rule).
- No change to how challenges are decided, sent, withdrawn or limited.
- No deletion or rotation of challenge-log files.
- Not fixing the test session-log pollution (OD-14).

---

## 10. Owner decisions

| OD | Question | Recommendation |
|---|---|---|
| OD-1 | Store a game's origin on the game (journal event + record + index row), with the challenge-log join only as a fallback (§2.2 A), or join only (B)? | **A.** Keeps the record the game's single description; tournament games and resumed games work without the log. |
| OD-2 | Make the 24 h credit log a fold of the challenge log and stop writing `challenge-outcomes.json` (P6), or keep both fed by one funnel? | **Fold (P6).** One source of truth. The old file stays on disk, readable. |
| OD-3 | Challenge-log durability: `F_FULLFSYNC` per append, plain `fsync`, or none? | **`F_FULLFSYNC`.** Rare events, the record the owner wants kept. |
| OD-4 | File granularity: per UTC day, one file, or per month? | **Per UTC day**, like `Protocol/`. |
| OD-5 | Add `FileSafety.openForAppending` and move the shared JSONL append helper (journal, protocol log, challenge log) onto it, with the tail cut on the same descriptor under a per-append `flock` (§3.3)? | **Yes, in P1.** One append path. For the journal and protocol log, the changes are refusing a symbolic link, FIFO or directory (none exist today, §1.7) and the cross-process lock. |
| OD-6 | Back-fill: automatic when stale (controller load) plus a Rebuild button, or manual only? | **Automatic + button.** Idempotent and cheap (0.23 s here). |
| OD-7 | How inferred origins look: "≈" + secondary color + explanatory help, or a separate "(inferred)" word? | **"≈" + help.** Short in tables; the help text says it in words. |
| OD-8 | Log sends DCM refused itself before any request (limits, offline, scope)? | **No.** Not challenges; already in the protocol log and the UI. |
| OD-9 | Store typed snapshots only, or also each raw Lichess JSON line, in the challenge log? | **Typed only.** The protocol log keeps the raw lines; this keeps the challenge log compact and its schema deliberate. |
| OD-10 | Add `DCMOrigin` to newly filed PGNs? | **Yes**, new games only. |
| OD-11 | Add a one-line ROADMAP.md entry under the Lichess bot item? **Needs the owner's express permission** (standing rule: new plans go into the ROADMAP only with it); nothing is written to ROADMAP.md until then (§3.10). | **Yes:** `- **Lichess bot: durable challenge log and game origins (planned 2026-10-06).** Plan: documentation/plans-active/LICHESS_BOT_CHALLENGE_LOG_PLAN.md.` Marked complete at P7, nothing removed. |
| OD-12 | Load every day file at launch, with a row cache only when a load exceeds 1 s? | **Yes, defer the cache.** Measured on every load. |
| OD-13 | Record the matchmaking trigger (automatic pass vs. Fill Open Slots) on live sends? | **Yes.** Cheap and otherwise unknowable; reconstruction can't recover it. |
| OD-14 | Fix the tests' session-log pollution (§1.8, §8) in a separate small change? | **Yes, separately:** give tests a temporary log folder. Not part of this plan. |

---

## 11. Review (2026-10-06, against `dad00b85`)

What was checked:
- every claim of §1 and §3.3–§3.7 against the code;
- Appendix B, rerun read-only against the real `Protocol/` and `Games/` files: identical output, 206/206 games (incoming 12, matchmaking 132, operator inferred 62, unknown 0), 319 raw challenges;
- the 373-row breakdown, recomputed by Appendix B.2 (new);
- `F_FULLFSYNC` latency (§1.7);
- the bot data folder, for symbolic links and FIFOs: none;
- every existing test that touches a changed API (§5.1).

**Must-fixes applied.**
1. **Case-insensitive name matching in reconstruction** (§1.7, §3.7). The protocol messages mix ids and display names. The plan's "42 paired / 12 inferred" holds only with case-insensitive pairing; exact text gives 25 / 29. Every name comparison is now lowercased.
2. **§6.3's row breakdown corrected.** "No answer 2" were two withdrawals on going offline, and "canceled 11" is 5 timeout withdrawals plus 6 withdrawals with no recorded reason. Incoming "declined 18" is by DCM's decision, and one of them has no `challengeDeclined` echo. Appendix B.2 reproduces every count.
3. **The ledger's states were incomplete and the fold's order was unspecified** (§3.4).
   - Added `canceledOnLichessWithoutRecordedWithdrawal` (6 rows today) and an optional withdrawal result.
   - Added an order-independent precedence: `gameStarted` beats withdrawal and decline, which covers "accepted after a withdraw attempt" and "accepted by hand outside DCM".
   - Added the sender rule for an id that has both an echo line and `outgoingCreated`.
4. **`gameStarted` was lost in the POST race** (§3.4). The plan wrote it only "for a logged challenge id", and in the race the id isn't logged yet. It is now also written when the echo is held, or retroactively via `gameStartsSeen`.
5. **Unmatched echoes** (§3.4):
   - a reconnect replay of an earlier run's challenge (already in the ledger from disk) would have been written as "seen without a created line". It is now not held;
   - a teardown during a POST writes both lines, and the fold resolves them.
6. **Ledger load** (§3.4):
   - it loads only while nil, so a window open no longer replaces the live ledger with a re-read that misses in-flight appends;
   - events recorded during the load are buffered and folded on top;
   - `loadStatus` turns a miss on a partial ledger into `undetermined(.challengeLogIncomplete)` (renamed from `.challengeLogNotLoaded`), never `.noChallengeRecord`.
7. **Incoming origin** (§3.5): any incoming row whose game started is `acceptedIncomingChallenge`, not only one DCM decided to accept.
8. **The append path** (§3.3):
   - `openForAppending` lacked `O_NONBLOCK`: a FIFO would hang in `open`, and the plan's own FIFO test could never pass;
   - the tail cut opened the path separately with `FileHandle(forUpdating:)`, which follows symbolic links and runs **before** the refusing open, so it would truncate a link's target. It now runs on the same `O_RDWR` descriptor;
   - "one writer" was not true. Every instance appends to the protocol log, and withdrawal results can land after the runtime's lock is released (`:2938`). Each append now holds `flock(LOCK_EX)` across cut, write and sync;
   - creating a day file also flushes `Challenges/`;
   - the two path-taking APIs that tests call keep their signatures (§5.1).
9. **Reconstruction cutoff** (§3.7). A day cutoff double-counts not-created attempts (which have no id) on the live log's first day; the cutoff is now the live log's first entry's timestamp. A changed account id or cutoff now regenerates. A two-instance create race reads back and reports `unchanged` instead of failing.
10. **Display step 5** (§3.6) labeled every origin-less game "played before origins were recorded", a guess for a new-build game whose origin write failed. It is now split by creation time.
11. **OD-11** (§3.10, §7 P7, §10): nothing goes into ROADMAP.md without the owner's express approval. P7 skips it otherwise.
12. **Index schema coordination and sequencing** (§7):
    - one rule: +1 from the then-current version, and tests refer to `LichessBotIndex.schemaVersion` symbolically;
    - the same rule was sent to the record-stats reviewer;
    - P1 may start now (its files are disjoint from follow-lineage and record-stats), and P2+ wait for follow-lineage.
13. **Tests** (§5): added cases for every fix above, plus controller-level tests of the POST race end to end, a relaunch mid-game, cancel-then-accepted, going offline during a POST, and recording during a held load.

**Smaller corrections.**
- `destUser` is optional in the snapshot.
- "challenge outcome: <id>" is the opponent id.
- Queue and casual-resend failure lines now pair not-created attempts, so they are no longer read as operator sends.
- The public `cancelChallenge(id:)` keeps its signature.
- The record builder rule is now "first determined" (the old "last determined, keep first on conflict" contradicted itself).
- Measured fsync numbers replace "milliseconds".
- `noLongerOnLichess` is folded into `withdrawn(_, .alreadyGone)`: it had no fact of its own to derive from.
- The read-only proof allows the protocol log to be appended.

**The "inferred" rule (owner question).** It stands, with stronger evidence than the plan claimed:
- every matchmaking send in the log is preceded by a "matchmaking pick" line for that player (190/190), and no inferred-operator send is (0/99). The pick-line check is now recorded as evidence (§3.7 step 7);
- 61 of the 62 inferred games were sent before the first matchmaking line of any kind (2026-09-29 16:14:40Z).
- The remaining one is AndoBot (2026-10-02 22:49:08Z).

**Idempotence and non-rewrite** (checked):
- the reconstructed file is written only when its bytes differ, and has no generation time;
- the live log is append-only;
- no record, journal, PGN, protocol file or `challenge-outcomes.json` is rewritten;
- `index.json` is rebuilt only by its own schema rule.

**Durability and concurrency** (checked):
- `F_FULLFSYNC` runs on the general file queue (a serial `DispatchQueue`, never the main actor or a `Task`);
- nothing on the play path waits on it;
- the lock discipline is `SyncBox` / `flock`, with no `NSLock`.

**Open points for the owner.**
- OD-1 … OD-14 are all still undecided. OD-11 (ROADMAP) needs express permission.
- New with this review, folded into OD-5: the per-append `flock` and the `O_NONBLOCK` / same-descriptor cut also change the journal and protocol-log append path. The recommendation is yes: it fixes a latent cross-instance cut race that exists today.
- Whether "Withdrawn (reason not recorded)" is acceptable wording for the 6 historical outgoing cancels, which were probably the operator's but have no line that says so.

## Appendix A. Measured data (2026-10-06, read-only)

- Protocol files:

  | File | Bytes |
  |---|---|
  | `events-20260928.jsonl` | 1,049,025 |
  | `events-20260929.jsonl` | 1,396,503 |
  | `events-20261001.jsonl` | 116,940 |
  | `events-20261002.jsonl` | 701,547 |
  | `events-20261003.jsonl` | 195,049 |
  | `events-20261006.jsonl` | 2,450,315 (still growing) |

  Total 5,909,379 bytes (5.64 MB).
- Entry kinds: request 32,457; challenge 2,136; stream 1,819 (event-stream `challenge` 319, `gameStart` 210, `gameFinish` 206, `challengeDeclined` 100, `challengeCanceled` 11); game 636; lifecycle 35; anomaly 11; account 2.
- `.challenge` messages:
  - "challenge sent to" 289 (all with `id`), "matchmaking sent a challenge to" 190, "challenge queue: sent" 0, "matchmaking resent … as casual" 0;
  - "<us>: ignore: our own outgoing challenge" 285, "<us>: accept" (older builds' echo) 4;
  - "outgoing challenge accepted" 194, "declined" 82, "canceled" 8;
  - "withdrawing unanswered challenge" 5, "withdrew challenge … on going offline" 2;
  - "challenge outcome:" 373 (first at 2026-10-01 04:01:16Z), "… is at its bot-game limit" 54 (15 before the first outcome line).
- `gameStart.source`: `friend` 210 of 210; no `tournamentId`.
- Feature commits (for reading the logs; a running build may predate or postdate its commit): challenge queue and matchmaking `baa32bd3` (2026-09-29 05:06Z); outcome tracking `1c26491d` (2026-09-29 17:15Z); automatic casual resend `9a36f9fd` (2026-10-03 00:19Z).

## Appendix B. Prototype classifier used for the numbers above

It reads only. P4's Swift implementation must reproduce its game counts exactly on the same inputs. The full algorithm is §3.7; this prototype covers its game and send counts.

```python
import json, glob, os, re, collections, datetime
ROOT = os.path.expanduser('~/Library/Application Support/DrewsChessMachine/LichessBot')
US = 'drewschessmachine'
def ts(s): return datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
entries = []
for f in sorted(glob.glob(ROOT + '/Protocol/events-*.jsonl')):
    lines = open(f, 'rb').read().split(b'\n')
    for l in lines[:-1]:              # an unterminated last line is dropped
        if l.strip(): entries.append(json.loads(l))
chal, answers, gamestart = {}, collections.defaultdict(list), {}
for e in entries:
    if e['kind'] == 'stream' and e['fields'].get('stream') == 'event' and e['message'].startswith('{'):
        o = json.loads(e['message']); t = o.get('type')
        if t in ('challenge', 'challengeDeclined', 'challengeCanceled'):
            c = o['challenge']; answers[c['id']].append(t); chal.setdefault(c['id'], c)
        elif t == 'gameStart':
            gamestart.setdefault(o['game']['gameId'], o['game'])
SENT = re.compile(r'^challenge sent to (.+)$')
COMP = [(re.compile(r'^matchmaking sent a challenge to (.+)$'), 'matchmaking'),
        (re.compile(r'^matchmaking resent a challenge to (.+) as casual$'), 'matchmakingCasualResend'),
        (re.compile(r'^challenge queue: sent (.+)$'), 'challengeQueue')]
sends, pending, unpaired = [], {}, 0
for e in entries:
    if e['kind'] != 'challenge': continue
    m = e['message']; s = SENT.match(m)
    if s:
        d = dict(user=s.group(1), id=e['fields'].get('id'), at=e['at'], sender=None)
        sends.append(d); pending[d['user']] = d; continue
    for rx, kind in COMP:
        c = rx.match(m)
        if c:
            d = pending.pop(c.group(1), None)
            if d is None or d['sender'] or (ts(e['at']) - ts(d['at'])).total_seconds() > 5: unpaired += 1
            else: d['sender'] = kind
by_id = {d['id']: d for d in sends}
games = collections.Counter()
for f in glob.glob(ROOT + '/Games/*/*/*.json'):
    gid = json.load(open(f))['gameID']
    c = chal.get(gid)
    if c is None: games['unknown'] += 1
    elif c['challenger']['id'] != US: games['incoming'] += 1
    else:
        d = by_id.get(gid)
        games['outgoing, no send line' if d is None else (d['sender'] or 'operator (inferred)')] += 1
out = [c for c in chal.values() if c['challenger']['id'] == US]
print('challenges', len(chal), 'outgoing', len(out), 'incoming', len(chal) - len(out), 'unpaired companions', unpaired)
print('games', sum(games.values()), dict(games))
```

Output on 2026-10-06: `challenges 319 outgoing 289 incoming 30 unpaired companions 0` and `games 206 {'incoming': 12, 'matchmaking': 132, 'operator (inferred)': 62}`.

Rerun in review (2026-10-06, read-only, same files): identical output.

## Appendix B.2. Row-state check (review, 2026-10-06)

This reads only. It checks the §6.3 row counts: the outgoing states with §3.7 step 6's withdrawal rules, the not-created pairing with case-insensitive names (step 5), and the incoming decisions. P4's Swift implementation must reproduce these counts on the same inputs.

```python
import json, glob, os, re, collections, datetime
ROOT = os.path.expanduser('~/Library/Application Support/DrewsChessMachine/LichessBot')
US = 'drewschessmachine'
def ts(s): return datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
entries = []
for f in sorted(glob.glob(ROOT + '/Protocol/events-*.jsonl')):
    for l in open(f, 'rb').read().split(b'\n')[:-1]:
        if l.strip(): entries.append(json.loads(l))
chal, ans, started = {}, collections.defaultdict(set), set()
for e in entries:
    if e['kind'] == 'stream' and e['fields'].get('stream') == 'event' and e['message'].startswith('{'):
        o = json.loads(e['message']); t = o.get('type')
        if t == 'challenge': chal.setdefault(o['challenge']['id'], o['challenge'])
        elif t in ('challengeDeclined', 'challengeCanceled'): ans[o['challenge']['id']].add(t)
        elif t == 'gameStart': started.add(o['game']['gameId'])
ch = [e for e in entries if e['kind'] == 'challenge']
offline = {m.group(1) for e in ch for m in [re.match(r'^withdrew challenge (\S+) on going offline$', e['message'])] if m}
timeout = []  # (lowercased name, at)
for e in ch:
    m = re.match(r'^withdrawing unanswered challenge to (\S+) after \d+ s$', e['message'])
    if m: timeout.append((m.group(1).lower(), e['at']))
def out_state(i):
    if i in started: return 'accepted'
    if 'challengeDeclined' in ans[i]: return 'declined'
    if i in offline: return 'withdrawn: going offline'
    name = chal[i]['destUser']['id']
    if 'challengeCanceled' in ans[i]:
        return 'withdrawn: unanswered timeout' if any(n == name for n, _ in timeout) else 'withdrawn: reason not recorded'
    return 'open'
decision = {}
for e in ch:
    m = re.match(r'^(\S+): (accept|decline|ignore)\b', e['message'])
    if m and 'challenge' in e['fields'] and m.group(1).lower() != US: decision[e['fields']['challenge']] = m.group(2)
out = [i for i, c in chal.items() if c['challenger']['id'] == US]
inc = [i for i, c in chal.items() if c['challenger']['id'] != US]
first_outcome = min(ts(e['at']) for e in ch if e['message'].startswith('challenge outcome:'))
failed = [(e['message'].split()[3].lower(), ts(e['at'])) for e in ch if re.match(r'^matchmaking send to \S+ failed:', e['message'])]
nc = []
for e in ch:
    m = e['message']
    if re.match(r'^challenge outcome: \S+ (offline|refused: )', m): nc.append((m.split()[2].lower(), ts(e['at'])))
    elif 'is at its bot-game limit' in m and ts(e['at']) < first_outcome: nc.append((m.split()[0].lower(), ts(e['at'])))
paired = sum(1 for n, t in nc if any(fn == n and abs((ft - t).total_seconds()) <= 5 for fn, ft in failed))
print('outgoing', len(out), dict(collections.Counter(out_state(i) for i in out)))
print('not created', len(nc), 'paired', paired, 'inferred', len(nc) - paired)
print('incoming', len(inc), dict(collections.Counter('accepted' if i in started else decision.get(i, 'undecided') for i in inc)))
print('rows', len(out) + len(nc) + len(inc))
```

Output on 2026-10-06: `outgoing 289 {'accepted': 194, 'declined': 82, 'withdrawn: reason not recorded': 6, 'withdrawn: unanswered timeout': 5, 'withdrawn: going offline': 2}`, `not created 54 paired 42 inferred 12`, `incoming 30 {'decline': 18, 'accepted': 12}`, `rows 373`.
