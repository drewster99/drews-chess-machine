# Lichess bot: a durable challenge log, and how each game started

Status (2026-10-06): **PLAN ONLY.** Nothing here is implemented. Implementation starts after `LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md` has landed (§7, "Sequencing").
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
- Appenders already cut an unterminated final line before appending and record the cut (`cutUnterminatedFinalLine`, `Data/LichessBotJSONLines.swift`).

### 1.7 Measured on this Mac (2026-10-06)

- 206 game records, 2026-09-28 to 2026-10-06.
- 6 protocol files covering every game day: 5,909,379 bytes (5.64 MB), 37,096 entries.
- Raw `challenge` events: 319 distinct ids — 289 outgoing (challenger = `drewschessmachine`) and 30 incoming. By UTC day: 09-28 68, 09-29 73, 10-01 4, 10-02 40, 10-03 8, 10-06 126. A raw challenge event averages 645 bytes (maximum 709).
- Every one of the 206 game ids equals the id of a raw `challenge` event.
- `challenge-outcomes.json` holds 140 records, all from 2026-10-06.
- Parsing every protocol file and classifying (Appendix B): 0.23 s.

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
| `LichessBotChallengeSnapshot` | `id`; `challenger` / `destUser` as `LichessBotChallengeParty` (id, name, title, rating, provisional); `variant: LichessBotOpenValue<LichessBotVariantKey>`; `rated`; `speed: LichessBotOpenValue<LichessBotSpeed>`; time control (type as open value, limit and increment seconds, days per turn); `color: LichessBotOpenValue<LichessBotChallengeColorName>`; `finalColor: LichessBotOpenValue<LichessBotColorName>?`; `initialFen?`; `rematchOf?`. Built from `LichessBotChallenge` (`API/LichessBotAPIModels.swift:431-447`) in one `init(_:)`. |
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
- **Durability: `F_FULLFSYNC` after every append** (OD-3). Challenge events are rare (Lichess caps sends at 25/min), and a full sync costs milliseconds. This is the one record the owner wants kept, so the protocol log's no-sync risk (§1.1) isn't repeated here.
- A failed append calls `onWriteFailure`, which the controller raises as an alarm. Nothing is dropped silently. After the queue is closed (shutdown), a refused append is written to the session log with its event, as `LichessBotFileQueue.enqueue` already does.
- Lock discipline: the "tail verified" set is a `SyncBox<Set<String>>` (`OSAllocatedUnfairLock`), read and changed only inside file-queue closures (the protocol log's pattern). No new `NSLock`, no actor.
- **One writer.**
  - Live entries are written only while a runtime exists, and the instance lock (`bot.lock`) allows one runtime per data folder.
  - Within the app, the controller's `recordChallengeEvent(_:)` is the only caller (§3.4).
  - `O_APPEND` keeps every write at the file's end anyway.

**FileSafety gains `openForAppending(at:)`** (OD-5):
- opens with `O_WRONLY | O_APPEND | O_CREAT | O_NOFOLLOW | O_CLOEXEC`;
- `fstat`s the descriptor and refuses anything but a regular file with a `FileSafetyError` naming the path;
- returns the handle and the `FileIdentity`.

`LichessBotJSONLines.append` switches to it and takes `synchronize: LichessBotJSONLines.Synchronization` (`.none`, `.fsync`, `.fullSync`). The journal keeps `.fsync` where it has `true` today and `.none` elsewhere; the protocol log keeps `.none`. So the journal and the protocol log also stop following symbolic links. The behavior change is that a symbolic link at a journal or protocol-log path is now refused (an alarm), where today it is written through.

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
| `withdrawn(reason, result)` | DCM withdrew it; why, and what Lichess answered. |
| `noLongerOnLichess` | Lichess no longer knows it: expired or answered unseen. |
| `notCreated(reason)` | The send created no challenge. |
| `incomingDecided(decision)` | Incoming, decided, no later fact. |

- **Load.** The ledger loads at go-online, before the runtime starts, beside player notes and outcomes (`:830-839`). It also loads when the bot window opens. Loading reads every day file on the general file queue, behind a continuation, and logs `[LICHESS-BOT] challenge log loaded: files=N lines=N bytes=… ms=… skipped_newer=N`.
- **A file that fails to decode** is left out, with an alarm. The ledger still loads, and the alarm names what is missing.
- **If the ledger itself can't load,** going online still proceeds: losing the challenge log must not stop the bot. The alarm stays up, appends still happen, and the in-memory ledger starts empty. Game origins then fall back to `undetermined(.challengeLogNotLoaded)` where the ledger would have been needed (OD-12 covers the load-time budget).
- **The funnel.** `recordChallengeEvent(_ event: LichessBotChallengeLogEvent)`, on the main actor, is the **only** mutation point. In one synchronous step it:
  1. applies the entry to the ledger;
  2. enqueues the append;
  3. tells the origin resolver (§3.5).
  
  Its call sites:

| Fact | Where (today's code) | Event |
|---|---|---|
| POST answered with the challenge | `sendChallenge` right after `client.challenge` returns (`:1765`), beside `recordCreated` (`:1783-1785`) | `outgoingCreated` |
| Opponent offline | `:1751-1757` | `outgoingNotCreated(.opponentOffline)` |
| Lichess refused the POST | `:1772-1777` (`LichessBotChallengeRefusal.classify`) | `outgoingNotCreated(.refused)` |
| POST failed without an answer | `:1779` ("outcome not recorded") | `outgoingNotCreated(.noAnswer)` |
| Withdrawn: operator, timeout | `cancelChallenge(id:)` (`:1819-1841`) gains a `reason:` (the public one passes `.operatorCancel`, the timeout path `:2984-2993` passes `.unansweredTimeout`) | `withdrawalRequested` then `withdrawalResult` (`.confirmed`, `.alreadyGone` on 400/404, `.failed`) |
| Withdrawn: going offline, created while going offline | `withdraw(challengeID:client:)` (`:1846-1863`) gains a `reason:` (`tearDownRuntime` `:2895` → `.goingOffline`; `:1790` → `.wentOfflineWhileSending`) | the same pair; `CancellationError` → `.abandonedAtShutdown` |
| Any challenge event on the stream | manager `.challengeArrived` (now carrying the `LichessBotChallenge`) | incoming: `incomingReceived` (once per id: Lichess replays open challenges on reconnect, and a replay writes nothing); our own echo: see "Unmatched echoes" below |
| Incoming decision | `.challengeDecision` (`:3178`) for a challenger ≠ us | `incomingDecided` |
| Accept/decline request failed | `.challengeResponseFailed` | `incomingResponseFailed` |
| `challengeDeclined` / `challengeCanceled` | new manager event `.challengeAnsweredOnStream` (every such line, either direction) | `declinedOnLichess` / `canceledOnLichess` |
| `gameStart` for a logged challenge id | new manager event `.gameStartReceived(LichessBotGameEventInfo)`, emitted before the session starts | `gameStarted` (once per id) |

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
  - An echo still unmatched after `2 × LichessBotRequestTimeouts.resource` (2 × 30 s: no POST of this run can still answer by then) is written as `outgoingSeenWithoutCreatedLine`. Its attribution is the single `.noAnswer` send to the same player inside that window, if there is exactly one, else `.notRecorded`. The poll loop checks this.
  - Teardown writes every echo still held with `.notRecorded`. Typical source: a challenge sent by an earlier run, or by another client using the token.

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
    /// The challenge log could not be loaded, so it could not be consulted.
    case challengeLogNotLoaded
}
```

- `LichessBotGameEventInfo.source` (`API/LichessBotAPIModels.swift:390`) becomes `LichessBotOpenValue<LichessBotGameSourceName>?`; it decodes the same string. The `LichessBotGameSourceName` cases are taken from Lichess's game-source enumeration (`lila`, verified at implementation time, not guessed). An unknown value is kept verbatim. Only `friend` has been observed here.

**Deciding it** (`LichessBotGameOriginResolver`, a pure `struct` held by the controller; every case is unit-tested):
- At `.gameStartReceived(info)`, remember `info` for the game.
- At `.gameSessionStarted(gameID, _, sessionOrigin)`:
  - **resumed, and the journal holds a determined origin** (`LichessBotResumedJournal.recordedOrigin`, below) → show it, write nothing;
  - otherwise, look the id up in the ledger:
    - incoming received and decided `accept` → `acceptedIncomingChallenge`;
    - `outgoingCreated` → `outgoingChallengeAccepted(sender)`;
    - `outgoingSeenWithoutCreatedLine(.unansweredSend(_, sender))` → `outgoingChallengeAccepted(sender)`;
    - `outgoingSeenWithoutCreatedLine(.notRecorded)` → `outgoingChallengeSenderNotRecorded`;
    - `info.source` known as arena or swiss → `tournament`;
    - **not known yet** (the POST race, or an echo still unmatched) → the game waits in `gamesAwaitingOrigin`.
- When `recordChallengeEvent` adds a row whose id is a waiting game → decide and write.
- At `.gameSessionEnded` for a game still waiting → `undetermined(source, .noChallengeRecord)`. A resumed journal already ending in that same undetermined value is not written again.
- A determined origin is written **once** per game per run.

**Writing it.**
- New journal case `.gameOrigin(LichessBotGameOrigin)`, written by `LichessBotJournalWriter.recordOrigin(_:gameID:)`. This is the `recordRequest` pattern: the controller writing into a game's journal outside the session, through the same file queue, synchronized like a posted move (`.fsync`).
- A game already filed gets nothing written. The writer's existing "late journal entries not written: the game is filed" path logs it. The game then shows its origin through the challenge-log join (§3.6), so nothing is lost.
- `LichessBotJournal.schemaVersion` stays 1, per the journal's own rule that cases are added without changing it (`Data/LichessBotJournal.swift:7-12`). The downgrade cost is listed in §8.
- The three exhaustive switches handle the case:
  - the record builder keeps the **last determined** origin, else the last undetermined one. Two *different* determined origins add an anomaly to the record and keep the first;
  - the carryover fold ignores it;
  - the live view's replay sets the live game's origin.
- `LichessBotResumedJournal` gains `recordedOrigin: LichessBotGameOrigin?` (same "last determined" rule). Nil means "the journal holds none".

**Persisting it.**
- `LichessBotGameRecord` gains `let origin: LichessBotGameOrigin?`. Nil means "the journal held none": a game played before this feature, or a write that failed. The record schema stays 1: an optional field old builds ignore, as `LichessBotGenerationInfo.valueHeadRecenteredOnLoad` did.
- `LichessBotGameSummary` gains `let origin: LichessBotGameOrigin?`, copied in `init(record:)`.
- `LichessBotIndex.schemaVersion` goes 2 → 3, so the stored cache is discarded once and rebuilt from the records by the existing rule. It is a derived cache, never user data.
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
5. Otherwise → `unknown(.notRecordedAndNotInProtocolLog)`, shown as **Unknown — played before origins were recorded; no challenge with this id in the protocol log**.

- When the record and the ledger disagree, the record wins, and the disagreement is logged once per game at index load (`[LICHESS-BOT] origin disagreement for <id>: record … challenge log …`). It is never hidden.
- The controller computes `originsByGameID` whenever the index, the ledger or the reconstructed history changes, the way `recordsByOpponent` is computed in `index`'s `didSet` (`:208-217`). Views read that map and never resolve anything themselves.

### 3.7 Back-fill: reconstructing past challenges (`Stats/LichessBotChallengeReconstruction.swift`, new, pure)

**Inputs.**
- The protocol day files, in name order, lines in file order (file order is record order).
- Only files dated on or before the first day of the live log (the oldest `challenges-*.jsonl`), or every file while there is no live log. So once the live log has existed for a day, the inputs stop changing, and the parse cost doesn't grow with the protocol log.
- `ourAccountID` (the bot's stored account id). With none configured, reconstruction doesn't run, and the window says why.

**Algorithm v1** (the message forms are frozen: they are historical formats, matched exactly):
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
   - "challenge outcome: <id> offline" / "… refused: …" (since 2026-10-01). The sender is matchmaking when paired with "matchmaking send to <name> failed:" within 5 s, else operator (inferred).
   - Before the outcome lines existed: "<name> is at its bot-game limit …" (written only from a refused POST, `:1770`) with no outcome line within 5 s.
6. **Withdrawals:** "withdrew challenge <id> on going offline" → `goingOffline, confirmed`. "withdrawing unanswered challenge to <name> after N s" → `unansweredTimeout` for the newest open outgoing challenge to that name.
7. **Confidence** of a reconstructed outgoing sender:
   - `certain`: a line carrying the id says it;
   - `paired`: a companion line, paired by name and adjacency;
   - `inferredFromAbsence`: no companion. Operator. A lost protocol write (never synchronized, §1.1) would make this wrong.
   
   Direction from a raw event is always `certain`.

**Output.** `Challenges/reconstructed-from-protocol.json` holds:
- `algorithmVersion`;
- `ourAccountID`;
- `inputs` (file, size, SHA-256);
- `liveLogFirstDay` (nil or a day);
- `rows`, each a challenge or attempt with direction, opponent, terms, sender plus confidence, decision, state, and `evidence: [file:line]`;
- `counts` (including unpaired companions and skipped echo-accepts).

Rows are sorted by first evidence and encoded with sorted keys. There is no generation time in the file, so the same inputs give the same bytes.

**Running it (OD-6).**
- On the general file queue, behind a continuation, when the controller loads (bot window or go-online), if the file is missing, its `algorithmVersion` is older, or the input list or any input's size differs (protocol files only grow).
- The file is written with `FileSafety.writeNewFile` (absent) or `replaceRegularFile` (present), **only when the new bytes differ**. A rerun with unchanged inputs writes nothing, not even a modification-time change.
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
- ROADMAP.md: one line, only with the owner's permission (OD-11).

---

## 4. Edge cases

| Case | Handling |
|---|---|
| `gameStart` handled before the POST that created its challenge returns (§1.3) | The game waits in `gamesAwaitingOrigin`. `outgoingCreated` arrives a moment later and the origin is written then. |
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
| Two DCM instances | `bot.lock` lets one runtime write live lines. Reconstruction in both writes identical bytes, so a replace race is harmless. |
| No account configured | Reconstruction doesn't run. The window says "No Lichess account is configured". |
| Two sends to the same player within 5 s in old logs | The companion pairing is ambiguous, so it is left unpaired and counted, never guessed. |
| Rematch challenge | Incoming or outgoing like any other; `rematchOf` kept in the snapshot. |
| Clock, time zone | File names in UTC; times shown in local time. |

---

## 5. Tests (new files only)

Pure logic first. All file tests use `FileManager.default.temporaryDirectory`, and the controller tests pass a temporary `dataDirectory`, as all 19 existing ones do.

- **`FileSafetyAppendTests`**:
  - `openForAppending` creates and appends;
  - it refuses a symbolic link, a directory and a FIFO, each with a `FileSafetyError` naming the path;
  - two handles both land at the end (`O_APPEND`);
  - the `.fullSync` path issues `F_FULLFSYNC`, through a seam that records the call.
- **`LichessBotJSONLinesSynchronizationTests`**: `.none` / `.fsync` / `.fullSync` reach the right call; a symbolic link at a journal path is refused.
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
  - rows across two day files.
- **`LichessBotUnmatchedEchoTests`**:
  - an echo matched by `outgoingCreated` writes nothing;
  - unmatched past the window with exactly one `.noAnswer` send to that player → `.unansweredSend`;
  - with zero or two such sends → `.notRecorded`;
  - teardown writes the remaining echoes.
- **`LichessBotGameOriginResolverTests`**:
  - incoming accepted;
  - each `LichessBotChallengeSender` (sheet, resend offer, queue, matchmaking automatic and fill, casual resend);
  - the POST race;
  - the echo-only cases;
  - tournament (arena, swiss);
  - an unknown `source` kept verbatim;
  - undetermined at session end;
  - resumed with a determined origin (nothing written), with none, and with an undetermined one later determined;
  - written once per game per run.
- **`LichessBotGameOriginJournalTests`**:
  - the record builder keeps the last determined origin;
  - two different determined origins → anomaly, first kept;
  - an old journal without the event → `origin == nil`;
  - `LichessBotResumedJournal.recordedOrigin`;
  - the live view's replay sets the origin;
  - the carryover fold is unchanged by the new case.
- **`LichessBotIndexOriginTests`**:
  - a stored schema-2 `index.json` is rebuilt to schema 3;
  - an old record without `origin` gives a nil row field;
  - a new record's origin reaches its row.
- **`LichessBotGameOriginDisplayTests`**: the §3.6 order, each step; the disagreement log line; every category's labels come from `LichessBotGameOriginStyle`.
- **`LichessBotChallengeReconstructionTests`** (synthetic protocol lines built in the test, one per message form):
  - direction from the challenger;
  - each companion pairing, including ambiguous and missing;
  - old echo-accept lines skipped;
  - outcome-line and bot-limit refusals, without double counting;
  - withdrawals;
  - input cut-off at the live log's first day;
  - same inputs → identical bytes;
  - a rerun with unchanged inputs writes nothing (modification time unchanged);
  - a grown input regenerates.
- **`LichessBotChallengeLogControllerTests`** (fake Lichess, as in `LichessBotCasualFallbackTests`):
  - a send from the sheet, the queue, Resend as Casual, Fill Open Slots, an automatic pass and the automatic casual resend writes `outgoingCreated` with the right sender;
  - offline, refused and no-answer sends;
  - operator cancel, timeout withdrawal and going-offline withdrawal;
  - incoming accept, decline and ignore;
  - a game from each path shows the right origin in `originsByGameID` and in its record after filing;
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

If implementation finds an edit is needed after all, it stops and lists it for the owner's approval, as the follow-lineage plan's §5.1 did.

---

## 6. Validation

1. **Targeted tests.** Every new test class passes, run by `-only-testing:` per phase. **Full suite** before merging, because this touches persistence. All tests pass, none modified.
2. **Builds** without new warnings, through drews-xcode-mcp.
3. **Back-fill on this Mac (P4).** Open the bot window offline. `[LICHESS-BOT] challenge history reconstructed …` must report exactly what Appendix B reports on the same inputs. On today's files (if no new games are played first), that is:
   - **rows 373**: outgoing created 289 (accepted 194, declined 82, canceled 11, no answer 2); not created 54 (refused; sender paired 42, operator inferred 12); incoming 30 (accepted 12, declined 18);
   - companion lines unpaired: 0; own-echo accept lines skipped: 4;
   - **games 206**: incoming 12, matchmaking 132, queue 0, casual resend 0, operator (inferred) 62, unknown 0.
   
   If games are played before P4 lands, rerun Appendix B on the then-current files and compare with that.
4. **Idempotence.** Close and reopen the window: the log says `unchanged`, and the file's SHA-256 and modification time are unchanged. Remove nothing; protocol files only grow.
5. **Read-only proof.** Before and after P4, `shasum` every file under `Games/`, `Protocol/`, `index.json` (index excepted at P3, where its schema bump rebuilds it) and `challenge-outcomes.json`. Identical.
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
7. **Crash.** `kill -9` the app mid-game after a challenge line. Relaunch and go online. The resumed game keeps its origin (from the journal). Any cut tail shows as `unterminatedLineCut` plus an `[ALARM]`.
8. **Downgrade check (documented, not run on real data).** In a temporary data folder: a journal with `.gameOrigin` can't be decoded by a pre-feature build (the expected, documented cost, §8); a day file with a newer-schema line is skipped and counted by this build.
9. **Load cost.** The `challenge log loaded` line's `ms` is recorded in the CHANGELOG entry; OD-12's trigger is checked against it.

---

## 7. Phasing (build + commit per phase)

**Sequencing.** Start after `LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md` lands. Its P-ready landed at `a7803ac0`, and its P0 is in the working tree now. Both plans touch:
- `App/LichessBotController.swift` (`startRuntime`, poll loop, `handle`, go-online loads);
- `Play/LichessBotSessionManager.swift`;
- `Play/LichessBotGameInterfaces.swift` (it adds generation lineage; this plan adds nothing there but reads near it);
- `Data/LichessBotGameRecord.swift` (generations vs. origin);
- `Data/LichessBotPGNWriter.swift` (if it adds tags);
- `UI/LichessBotOverviewView.swift` and the model card;
- `UI/LichessBotGameDetailView.swift`;
- `documentation/plans-active/LICHESS_BOT_PLAN.md` and `CHANGELOG.md`.

Re-check every `file:line` here against the tree at that time before P1.

| Phase | Work | Tests run |
|---|---|---|
| **P1 — file layer** | `FileSafety.openForAppending`; `LichessBotJSONLines.append` on it with `Synchronization`; data-directory paths; the challenge-log entry schema, writer and reader; `LichessBotChallengeLedger` (pure). Nothing calls the writer yet. | P1 test classes; existing `LichessBotDataLayer*` and journal tests |
| **P2 — recording** | `ChallengeOrigin` cases and trigger; manager event changes; `recordChallengeEvent` funnel at every §3.4 site (the outcome log moves into it); ledger load at go-online and window open; unmatched echoes. | P2 classes; existing `LichessBot*Controller*`, `CasualFallback`, `ChallengeQueue`, `QuitWithdrawal`, `MultipleChallenges`, `SessionManager` tests |
| **P3 — game origin** | Origin types; `source` as open value; resolver; journal case and `recordOrigin`; record field and builder; resume fold; live game; index field and schema 3; PGN tag (OD-10); logging. | P3 classes; existing record, resume, index and PGN tests |
| **P4 — back-fill** | Reconstruction (pure) plus its runner and file; resolver step 3; validation 3–5 on this Mac. | P4 classes |
| **P5 — UI** | Style, label, glyph; All Games column and filter; Recent list; Live picker; tile; detail; Challenge Log window and card button. | render tests |
| **P6 — outcome log from the challenge log** (only if OD-2) | The fold, and stopping `challenge-outcomes.json` writes. | existing `LichessBotChallengeOutcomeTests` unchanged and passing |
| **P7 — docs** | §3.10. | full suite |

Each phase: all work → recheck → build → its tests → commit (`git add` of its own files only). The full suite runs at P3 (persistence) and at P7.

---

## 8. Risks

- **Downgrade.**
  - A journal holding `.gameOrigin` can't be decoded by a pre-P3 build, so that build can't file or resume the game. This follows the journal's own stated rule (`Data/LichessBotJournal.swift:7-12`); it applies only to games played on the new build and left unfiled when downgrading.
  - A pre-P3 build seeing the schema-3 `index.json` rebuilds it as schema 2: harmless cache churn.
- **Growth and load time.** About 0.5 MB/day at the busiest measured day; the ledger loads every day file. Measured and logged on every load. A cached index of rows (like `index.json`) is deferred until a load exceeds 1 s (OD-12).
- **Inferred operator sends** (62 games here) would be wrong wherever a matchmaking or queue companion line was lost. The protocol log is never synchronized to disk, so a power loss can lose lines. They are marked "≈" and explained; never shown as certain.
- **The POST race** leaves a game briefly "Not yet known" in the live view. Bounded by one POST.
- **Message-format coupling.** Reconstruction parses historical protocol messages, which are frozen in this plan's text and covered by fixture tests. Later wording changes don't matter: once the live log exists, inputs stop at its first day.
- **Merge conflicts** with the follow-lineage work in the same files. Mitigated by sequencing (§7).
- **`F_FULLFSYNC`** on the general file queue delays queued protocol-log and index work by milliseconds per challenge fact. Acceptable at ≤ 25 sends/min.
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
| OD-5 | Add `FileSafety.openForAppending` and move the shared JSONL append helper (journal, protocol log, challenge log) onto it? | **Yes, in P1.** One append path. Refusing symbolic links changes journal and protocol-log behavior only in that refusal. |
| OD-6 | Back-fill: automatic when stale (controller load) plus a Rebuild button, or manual only? | **Automatic + button.** Idempotent and cheap (0.23 s here). |
| OD-7 | How inferred origins look: "≈" + secondary color + explanatory help, or a separate "(inferred)" word? | **"≈" + help.** Short in tables; the help text says it in words. |
| OD-8 | Log sends DCM refused itself before any request (limits, offline, scope)? | **No.** Not challenges; already in the protocol log and the UI. |
| OD-9 | Store typed snapshots only, or also each raw Lichess JSON line, in the challenge log? | **Typed only.** The protocol log keeps the raw lines; this keeps the challenge log compact and its schema deliberate. |
| OD-10 | Add `DCMOrigin` to newly filed PGNs? | **Yes**, new games only. |
| OD-11 | Add a one-line ROADMAP.md entry under the Lichess bot item? | **Yes:** `- **Lichess bot: durable challenge log and game origins (planned 2026-10-06).** Plan: documentation/plans-active/LICHESS_BOT_CHALLENGE_LOG_PLAN.md.` Marked complete at P7, nothing removed. |
| OD-12 | Load every day file at launch, with a row cache only when a load exceeds 1 s? | **Yes, defer the cache.** Measured on every load. |
| OD-13 | Record the matchmaking trigger (automatic pass vs. Fill Open Slots) on live sends? | **Yes.** Cheap and otherwise unknowable; reconstruction can't recover it. |
| OD-14 | Fix the tests' session-log pollution (§1.8, §8) in a separate small change? | **Yes, separately:** give tests a temporary log folder. Not part of this plan. |

---

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
