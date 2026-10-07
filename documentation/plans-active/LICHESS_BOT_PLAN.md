# Lichess Bot — native Swift client

Status (2026-09-28; updated 2026-10-01):
- **Phases 1–5 implemented** (commits `4ebcbfb`…`e66fc02`). The bot is live on lichess.org, and the live verification findings are recorded below.
- **Implemented alongside the live testing:** the additions of 2026-09-28 in §7.2, §9.1 (first version), §12.5a and §14.3b (account refresh), as each section notes. §9.1's lineage tree and §14.3b's opponent card are planned.
- **§7.3 shipped 2026-09-29** (commit `baa32bd`): the challenge queue, matchmaking and the finished-game hold in the grid (§14.3a).
- **Defaults are now the owner's running configuration** (commit `c5542b8`, confirmed intentional 2026-10-01). Every default this document states matches `LichessBotSettings.swift` as of that commit. Where a section explained why an earlier, more conservative default was chosen, the explanation stays and the new value is given beside it. The same commit added challenge alert tones (§12.7). Settings saved before it no longer decode: use **Reset to defaults** once after updating (§12.1).
- Phases 6–7 are not started.

*(Original status line, kept for history: "planned, not started. Nothing in this document is implemented.")*

Research behind it:
- `documentation/research/lichess-bot/chess-bot-api-integration-options.md`
  and `documentation/research/lichess-bot/notes/`.
- The Lichess OpenAPI spec, fetched directly from `github.com/lichess-org/api`
  (`doc/specs/…`) on 2026-09-28.
- Lichess server source, read directly from `github.com/lichess-org/lila`
  and `github.com/lichess-org/scalachess` on 2026-09-28, where the spec was
  silent (§4 draw rules).

Bot account: **`DrewsChessMachine`**. It was created on lichess.org 2026-09-28 05:37 UTC and upgraded to BOT through the app the same day. *(Before the upgrade: 0 games played, so it was eligible; "play no games on it before the upgrade" applied then.)*

## 1. Summary

DCM plays on lichess.org as a BOT account, through a native Swift client
built into the app. There is no Python bridge and no UCI subprocess. The
client talks to Lichess's HTTP/NDJSON Bot API directly and asks DCM's own
network for moves in-process. It runs inside the normal GUI app, next to
(and independent of) Play-and-Train. It has its own window with live games,
settings that apply immediately, per-game records, a protocol event log, and
statistics that can be filtered and grouped.

Why native rather than the `lichess-bot` Python reference client: DCM already
has an in-process move API. The UCI protocol exists in DCM only so that
*external* tools like cutechess can talk to it. `lichess-bot` would contribute
its hardened HTTP/stream handling, not chess logic. Its community bug history
is used here as a checklist (§4, §6) instead of a runtime dependency.

## 2. Goals

1. Play rated or casual games on Lichess as a BOT account. The operator
   controls which challenges are accepted.
2. Choose which model plays, following the human-play pattern: champion,
   trainer snapshot, live trainer, or a model file. This stays a separate
   system from human play (§9).
3. Settings take effect immediately, or at the next decision point that
   depends on them. No reconnect is needed except for a token change (§12).
4. Keep a complete local record of every game: moves, clocks, DCM's per-move
   internals, chat, and every protocol anomaly. Reconcile it against
   Lichess's authoritative export (§10).
5. A durable data layer. Streams reconnect on their own with heartbeat
   detection, games resume after a crash or relaunch, and nothing is lost on
   an unclean exit (§6, §10).
6. **Stay within Lichess's rate limits.** On any 429, stop all requests for at
   least a full minute (§5).
7. Statistics that can be filtered and grouped by model, time control,
   opponent type and rating, color, termination and date range, including
   ML-specific views such as value-head calibration on real games (§11).
8. A clean on/off control with graceful drain, and safe quit/shutdown
   behavior (§13).

## 3. Non-goals (explicit)

- **No search, no opening book, no tablebases.** This follows the project's
  core design (CLAUDE.md "What this project is"). The bot plays the network's
  single forward pass, the same as every other DCM move.
- **v1 does not feed Lichess games into training.** Records are for
  analysis, and nothing flows into `ReplayBuffer` or a corpus automatically.
  This is a v1 choice, not a project rule: corpus replay already trains on
  human PGN games (the elite corpus), and Train-vs-UCI trains on external
  engine games. The saved PGNs are importable through the existing
  `--import-pgn` path if that is ever wanted, as a separate decision.
- **Standard chess only.** Every variant is declined. DCM has no Chess960 or
  variant support (`MoveGenerator`; see §4 castling).
- No `lichess-bot`, no Python, no UCI subprocess.
- Tournaments (bots are excluded from Arena/Swiss), correspondence, and
  outgoing matchmaking are out of v1. Matchmaking is an optional later phase
  (§16, Phase 8).

## 4. Grounding facts this plan depends on

Every item was checked against code, the primary-source spec, or Lichess's
server source. The design
choices that follow from each are noted.

**Lichess protocol (OpenAPI spec, `lichess-org/api` `doc/specs`)**
- Event stream `GET /api/stream/event`:
  - Sends an empty keep-alive line **every 7 seconds** (documented).
  - "When the stream opens, all current challenges and games are sent."
  - "**Only one global event stream can be active at a time.** When the
    stream opens, the previous one with the same access token is closed."
  - Consequences: a heartbeat watchdog is justified. Resume-on-reconnect
    comes for free. Two clients using one token will knock each other off in
    a loop, and this must be detected (§6).
- Game stream `GET /api/bot/game/stream/{gameId}`:
  - "The first line is always of type `gameFull`", so every reconnect gives a
    complete resync.
  - Line types are `gameFull`, `gameState`, `chatLine` and `opponentGone`.
  - **No keep-alive interval is documented** for this stream. It is
    measured from the first live sessions (the protocol log records every
    keep-alive gap, §20.11). Until then, the game-stream watchdog uses the
    conservative, turn-aware rule in §6.
- `gameState` fields (all required unless marked optional):
  - `moves`: space-separated UCI, described as "**King to rook for
    Chess960-compatible castling notation**"
  - `wtime`, `btime`, `winc`, `binc` (ms)
  - `status` (enum below)
  - optional: `winner`, `wdraw`/`bdraw`, `wtakeback`/`btakeback`
  - optional `expiration {idleMillis, millisToMove}`: the countdown before
    first-move abort
- `gameFull`:
  - `id`, `variant`, `clock {initial, increment}`, `speed`, `perf`, `rated`,
    `createdAt`
  - `white`/`black` `{id, name, title, rating, provisional}`
  - `initialFen` (default `"startpos"`), `state`
  - optional: `daysPerTurn`, `tournamentId`
- Status enum:
  - `created`, `started`, `aborted`, `mate`, `resign`, `stalemate`
  - `timeout`, `draw`, `outoftime`, `cheat`, `noStart`, `unknownFinish`
  - `insufficientMaterialClaim`, `variantEnd`
- `opponentGone {gone, claimWinInSeconds?}`.
- Move endpoint `POST /api/bot/game/{id}/move/{uci}`:
  - Optional `offeringDraw=true`, so a draw offer can ride on a move.
  - Returns 400 with an `Error` body on failure.
- Game action endpoints:
  - `…/draw/{yes|no}`, `…/takeback/{yes|no}`, `…/resign`, `…/abort`
  - `…/claim-victory` and `…/claim-draw`. The spec describes both as for
    when "the opponent has left the game for a while". **lila's source
    confirms it:** `BotPlayer.claimDraw` sends `RoundBus.DrawForce`, which
    `RoundAsyncActor` honors only when:
    - `forceDrawable && hasClock && !isMyTurn`, and
    - the opponent `isLongGone`.

    It then ends the game as a rage-quit draw. **It is not a
    threefold/50-move claim.**
  - `…/chat` (`room` = `player`|`spectator`, plus `text`)
- **Draw rules for BOT games (lila + scalachess source, not in the spec):**
  - **Threefold is claimed automatically for bots.** On a threefold,
    `RoundAsyncActor` runs `Drawer.autoThreefold`. That claims the draw for
    any player who has the auto-claim preference, recently offered a draw,
    **or is a BOT** (`pov.player.userId.exists(isBotSync)`). Every game our
    bot plays therefore ends at the first threefold. Neither side can play
    on.
  - **50-move, fivefold and insufficient material end automatically.**
    scalachess `Variant.autoDraw` = `isInsufficientMaterial ||
    fiftyMoves (halfMoveClock >= 100) || fivefoldRepetition`, and
    `Position.end` includes `autoDraw`.
  - `draw/yes` or `offeringDraw` in a threefold position ends the game as a
    draw at once (`Drawer.yes`). Offering a draw when our side cannot lose
    becomes an `insufficientMaterialClaim`.
  - Net: Lichess's automatic draw conditions for bot games **match DCM's
    self-play adjudication in kind** (threefold, 50 moves, insufficient
    material). The differences that remain are in the *definitions* (§20.2).
- Challenges:
  - `POST /api/challenge/{id}/accept`
  - `POST /api/challenge/{id}/decline` with a `reason` enum: `generic`,
    `later`, `tooFast`, `tooSlow`, `timeControl`, `rated`, `casual`,
    `standard`, `variant`, `noBot`, `onlyBot`.
  - A challenge carries:
    - `challenger {id, name, rating, title, provisional, online, lag}`
    - `variant`, `rated`, `speed`
    - `timeControl` (`clock {limit, increment}` | `correspondence` | `unlimited`)
    - `color`, `initialFen?`, `rematchOf?`, `direction`
    - `compat {bot, board}`
- Account:
  - BOT upgrade `POST /api/bot/account/upgrade`: "cannot have played any
    game … **irreversible** … will only be able to play as a Bot".
  - The upgrade is confirmed through `GET /api/account` (`title == "BOT"`).
  - `POST /api/token/test` returns `{userId, scopes, expires}`. **Tokens can
    expire**, which needs monitoring (§12).
- Rate limits: see §5.
- Export: `GET /game/export/{id}` (JSON or PGN; `clocks`, `opening`, `evals`
  flags) is the authoritative after-the-fact record. The live stream does
  not reliably deliver a final `gameState` on resignation (lichess-org/api
  #60, closed "not planned").

**DCM code (file:line as of `bc7d145`)**
- **Move parsing: reuse, don't duplicate.**
  - `ChessMove.parseUCI(_:legal:)` (`App/UCI/ChessMove+UCI.swift:49-78`)
    resolves a UCI token against the legal-move list. `ChessMove.uci`
    (`:22-37`) formats a move.
  - Castling is stored king-to-destination (`e1g1`).
  - Consequence: Lichess's `e1h1` form **will not match** today, so §8.1
    adds an alias.
  - Two non-authoritative UCI helpers exist and must **not** be used here:
    `LichessProbeData.parseUCI` (no legality check) and
    `MoveGenerator.uciString` (writes non-standard `=queen`).
- **Game state.**
  - `ChessGameEngine` + `applyMoveAndAdvance` (`Chess/ChessGameEngine.swift:177-258`).
  - Replaying the move list from `startpos` produces exactly the encoder
    history self-play uses (same `BoardEncoder.encode(state, history: recentStates)`
    call as `UCIEngine.swift:276-281` and `BatchedSelfPlayDriver.swift:354-359`).
  - Starting from a FEN leaves history empty, and the history/repetition
    planes are zero-filled. That is out of distribution, so from-position
    games are always declined (§7; there is no setting to accept them).
- **Draw adjudication.** `ChessGameEngine` **ends the game itself** on
  threefold repetition, 50 moves and insufficient material
  (`updateResult`, `:262-335`), and later moves then throw
  `.gameAlreadyOver`.
  - Lichess ends bot games on the same *kinds* of conditions (see above).
  - DCM's *definitions* differ in places: the threefold key includes the
    en-passant square, and the insufficient-material cases can differ.
  - **Dangerous direction:** DCM's local rules end a game that Lichess says
    is still `started`. The bot would then refuse to move and lose on time.
  - §8.2 adds a server-authoritative mode so the local engine never refuses
    to move while the server says the game is live.
- **W/D/L is already computed on every move and thrown away.**
  - The single-position inference graph targets
    `[policyOutputReadback, valueOutput, valueProbs]` (`Network/ChessNetwork.swift:1080`).
  - `internalEvaluate` (`:1154-1198`) reads back only policy and the scalar.
  - `evaluateValueDistribution` (`:1218-1259`) runs a **second** full
    forward pass to get W/D/L.
  - §8.3 surfaces `valueProbs` from the existing pass, so there is no second
    pass.
- **Move sampling.**
  - `MoveSampler` (`Network/MoveSampler.swift:45`) gives `move` and
    `chosenProbability`. `--uci` calls it directly (`UCIEngine.swift:284-375`),
    and the bot does the same. It does not go through `MPSChessPlayer`, which
    discards the value (`MPSChessPlayer.swift:363`).
  - τ=0 is unsupported. `SamplingSchedule.argmax` is τ=0.01 and still
    samples (`MPSChessPlayer.swift:153-165`).
- **Human-play model slots** (`App/UpperContentView/PlayController.swift`):
  - `HumanPlayOpponentChoice` has four cases: `championSnapshot`,
    `trainerSnapshot`, `liveTrainer`, `loadedFile` (`:7-28`).
  - Resolution happens in `materializeOpponentSource` (`:912-981`), via
    `buildInferenceNetwork` (`:989-998`).
  - An unavailable source disables Start and raises an error. There is no
    fallback.
  - **Live trainer re-exports the trainer's weights before every AI move**
    (`LiveTrainerMoveEvaluationSource`, `Network/MoveEvaluationSource.swift:198-222`).
    `exportWeights` holds `weightAccessLock`, the same lock SGD holds
    (`ChessTrainer.swift:6562-6566`).
  - Measured export+load during training (LichessProbeWatcher logs, n=87):
    median ~505 ms, max ~1059 ms.
  - Consequence: the bot must **not** copy this per-move pattern (§9).
- **Networks.**
  - Each `ChessNetwork` has a serial `executionQueue`, and `evaluate` goes
    through `enqueue`.
  - The doc comment at `ChessNetwork.swift:1137-1142` warns that
    "two games concurrently on one network" need separate networks "or
    explicit serialization". The serial queue appears to already provide
    that serialization. **Do not trust the comment either way.** Phase 1
    proves it with a test (§16).
- **App shell.**
  - There is one `WindowGroup`. Secondary windows use
    `NSWindowController` + `NSHostingController` + a singleton registry +
    an `enum …Launcher` (template: `App/LichessProbeMonitorWindow.swift:10-97`).
  - Menus reach controllers through `AppCommandHub` closures, wired in
    `UpperContentView.wireMenuCommandHub()`.
  - `TitleBarView` has an always-visible right-side status area, which is
    where the status chip goes.
  - **No Keychain use exists anywhere.** This is the first use of the
    Security framework.
  - Versioned Codable-in-UserDefaults precedent: `LastSessionPointer` key
    `DrewsChessMachine.LastSessionPointer.v1`.
  - The Xcode project uses file-system-synchronized groups, so new files
    under `DrewsChessMachine/` are picked up automatically.
- **Naming.** The existing "Lichess" types (`LichessProbe*`,
  `lichessProbeInferenceNetwork`) belong to the **puzzle probe** feature.
  Every new type uses the prefix **`LichessBot`** so the two are never
  confused.
- **Shutdown.**
  - `AppDelegate.applicationShouldTerminate` never confirms today.
  - SIGUSR1, SIGHUP and SIGUSR2 end in `_exit(0)`, which bypasses
    `applicationWillTerminate`. §13 covers this.

## 5. Rate limits — design constraint on every request

**Lichess's rules (verbatim, spec + api-tips page):**
- "Only make one request at a time."
- "If you receive an HTTP response with a 429 status … waiting one minute
  before retrying will be sufficient, but some limits may require longer.
  Reduce your request frequency before retrying."
- No numeric limits are published, and Lichess reserves the right to change
  them.
- A 429 threatens every live game (moves can't be sent) and, if repeated,
  the account itself. The design therefore puts **not triggering 429** ahead
  of throughput.

**5.1 Interpretation of "one request at a time"**

Long-lived NDJSON streams (one event stream, plus one game stream per active
game) are the API's *intended* mechanism. They are open connections, not
repeated requests, and the reference client works the same way. **Every
other request goes through one account-wide single-flight gate**: moves,
challenge accept/decline, chat, draw/takeback/resign/abort/claims, account
and token checks, and exports. **Never more than one non-stream request is
in flight.** The *opening* of a stream (the GET until response headers
arrive) also goes through the gate. Once open, the stream releases the gate.
Stream opens are therefore serialized too, which prevents a reconnect burst
after a network blip.

**5.2 `LichessBotRequestGate`: single flight, strict priority**

1. **Move**: posting our move.
2. **Game-critical action**: claim-victory, resign, abort, draw and takeback
   responses.
3. **Challenge response**: accept/decline.
4. **Stream open**: event stream and game streams.
5. **Chat**.
6. **Housekeeping**: account/ratings refresh, token test, game export for
   reconciliation.

**Housekeeping is dispatched only when no game is waiting on us to move.** A
housekeeping request already in flight can delay a move by at most its own
latency, and it gets a short timeout (default 10 s). Every request records
its category, status, latency, and `Retry-After` in the protocol log (§10.3).

**5.3 Keep the request count low**

- **No polling anywhere.** All game and challenge state comes from streams.
- **Account/ratings refresh:**
  - Runs on connect and after a game finishes.
  - Throttled to at most once per `accountRefreshMinIntervalSec`
    (default 300).
  - Never on a timer while idle.
- **Exports** (reconciliation, §10.2):
  - One per finished game.
  - Lowest priority, and spaced at least `exportMinSpacingSec` apart
    (default 10).
  - Retried with backoff. A failed export never blocks play.
- **Chat:**
  - At most a greeting and a goodbye per game, with a hard cap per game.
  - Never an automated reply to opponent chat, so there is nothing an
    opponent can use to make the bot spam.
- **Challenges:**
  - Every accept/decline costs a request.
  - Spam protection: past `challengeResponseBudgetPerMinute` (default 10),
    further challenges are **left unanswered and logged**. They expire on
    Lichess's side, and this spends no requests.
- **Moves.** One request per move, which cannot be reduced. For this reason
  `maxConcurrentGames` originally defaulted low (2) and bullet was off by
  default (§7). Since commit `c5542b8` the defaults are the owner's running
  configuration: `maxConcurrentGames` 12, with ultraBullet and bullet
  accepted. The Stats view shows requests per game and per minute, so the
  operator can see the budget and tune concurrency against it.
- **Reconnects** (§6):
  - Exponential backoff with jitter (equal jitter as implemented; §6): start 2 s, cap 60 s.
  - At most one stream open in flight (via the gate).
  - Takeover detection stops ping-pong loops.

**5.4 On a 429 (any request, any category)**

1. The gate **closes account-wide** for `max(Retry-After, 60 s)`. The 60 s
   floor is a constant (`LichessBotRateLimit.minimumCooldownSec`) and is not
   configurable below 60. Nothing is sent, moves included; Lichess would
   reject them anyway.
2. The bot automatically **enters Drain**: it stops accepting new games for
   `postRateLimitDrainMinutes` (default 15) after the cooldown ends.
   In-progress games continue once the gate reopens.
3. A **`[ALARM] LICHESS-BOT rate limited`** line goes to SessionLogger. The
   UI shows a red status chip and a cooldown countdown. The protocol log
   records the triggering request, its category, and the **request rate over
   the preceding 60 s** by category, for diagnosis.
4. **Circuit breaker:** a second 429 within `rateLimitBreakerWindowMinutes`
   (default 60) takes the bot **fully offline**. The event stream closes, no
   reconnect happens, and manual re-enable is required. Repeated 429s risk
   longer lockouts or account action, so this is deliberately not
   automatic.
5. Requests are **never retried in a tight loop**. After the gate reopens,
   anything still relevant is re-derived from the latest stream state and
   not replayed blindly. A move is re-sent only if it is still our turn at
   the same ply.

**5.5 Other failures, which are not 429**

- **5xx or network errors:** retry through the gate with backoff (start
  1 s, ×2, cap 30 s, bounded attempts per category).
- **400 on a move:** re-sync. Treat the game stream as possibly stale and
  reopen it; `gameFull` gives the truth. Log the full error body. Never
  resend the same move blindly.
- **401/403:** the token was revoked, expired, or lacks a scope. Go offline
  immediately, alarm, and show the token panel.

**5.6 Tests** (§17). A stub server returns 429 with and without
`Retry-After`. The tests assert:
- Zero requests are sent during cooldown.
- Drain engages.
- A second 429 trips the breaker.
- With N concurrent games, the maximum number of simultaneous non-stream
  requests observed is **exactly 1** (an instrumented stub).
- Housekeeping is never dispatched while a move is pending.
- Reconnect spacing follows the backoff schedule.

## 6. Streams, heartbeat, reconnect

`LichessBotStreamReader` is built on `URLSession.bytes(for:)`, the async
streaming API. It uses no blocking calls, as the concurrency rules require.

**Two separate `URLSession`s** keep long-lived streams from starving move
POSTs of connections:
- **Streams** session: raised per-host connection limit, very high idle
  timeout. Liveness is owned by our watchdog, not by URLSession.
- **Requests** session: normal timeouts, its own connection pool.

See §20.5 (E32–E34).
- The NDJSON splitter is a pure, tested function. It buffers across chunk
  boundaries, splits on `\n`, tolerates `\r\n`, **skips empty keep-alive
  lines, which still count as heartbeat**, and caps line length.
- **Heartbeat watchdog:**
  - Event stream: no bytes (keep-alives count) for
    `eventStreamStallTimeoutSec` (default 25, about 3.5× the documented 7 s
    keep-alive) means the stream is dead. Tear it down and reconnect.
  - Game streams: same mechanism. The threshold is set from the first live
    sessions' measured gaps (§20.11) and is configurable. If Lichess sends
    no keep-alives on game streams during a long think, a stall is expected.
    The watchdog then triggers a cheap **resync** (reopen and receive
    `gameFull`) only when it is the opponent's turn and the silence exceeds
    a TC-scaled bound.
  - This matters because ISP IP rotation silently kills long-lived
    connections (lichess-bot #135).
- **Stream end ≠ game over** (lichess-bot #1184). A game is over **only**
  when:
  - (a) a `gameState`/`gameFull` status other than `created`/`started`
    arrives, or
  - (b) a `gameFinish` arrives on the event stream, or
  - (c) the export API reports a finished status.

  Any other stream end is a reconnect.
- **Resync is idempotent.** Game state is a pure function of
  `(initialFen, moves)` (§8.2). A reconnect's `gameFull` rebuilds from
  scratch, and duplicate or out-of-order `gameState` lines cannot corrupt
  anything.
- **Event-stream takeover:** if the event stream is closed by the server
  within `takeoverDetectWindowSec` (default 30) of opening, three times
  running, the most likely cause is another client using the same token.
  Go offline with the alarm "another client appears to be using this token"
  and stop reconnecting.
- **On connect or reconnect**, the event stream replays current challenges
  and games. Every `gameStart` for a game without an active session starts
  or resumes one, which covers app relaunch.
- **Reconnect backoff** is exponential with jitter (start 2 s, ×2, cap
  60 s) and resets after 5 min of healthy streaming. Every attempt is
  logged.
  - *As implemented:* **equal** jitter rather than full jitter — each delay
    is uniform in `[base/2, base]` (`LichessBotBackoff`). Full jitter can
    draw a near-zero delay, which after a server-side close would hammer
    the stream-open path; equal jitter keeps a floor while still
    decorrelating reconnects.
  - *As implemented:* a **game** stream that ends normally reconnects
    starting from the first backoff step, and a resync (a `400` on a move,
    or a move list that fails to replay) reopens at once with no backoff:
    the game's clock is running. Stream opens still pass through the
    gate, so this cannot bypass rate limiting.

### 6.1 Protection against a second bot instance

There are two different threats, and they need different defenses.

**A. Another DCM process on this Mac** (a second GUI instance, or a
relaunch overlapping a dying process).
- Going Online takes an **exclusive `flock()`** on
  `LichessBot/bot.lock` and holds it for as long as the bot is Online.
- The kernel releases it automatically if the process dies or crashes, so
  there is no stale-lockfile problem (unlike PID files).
- The lock file records the holder's PID, launch time, build and host name
  for the error message only. It is never used as the lock itself.
- A second process that fails the lock refuses to go Online and says:
  "Bot is already online in another DrewsChessMachine process (pid …,
  started …)".
- The same lock also protects the data directory from two writers, since
  both processes would otherwise append to the same journals.
- CLI modes never construct the bot (E37).

**B. Another machine** (or anything else holding credentials for the
account). No local lock can see this. Lichess's rules make it visible in two
ways, and each is detected and acted on:
1. **Same token elsewhere.** Only one event stream is allowed per token, so
   each side's connect closes the other's. Detection: the event stream is
   closed by the server within `takeoverDetectWindowSec` of opening, three
   times running (§6). Action: go Offline with the alarm "another client is
   using this token", and do **not** fight back.
2. **Different token, same account.** Both event streams stay up, both
   clients receive the same challenges and `gameStart`s, and **both try to
   play the same games.** Detection signals:
   - a move appears on **our side** of a game that we did not send
     (the move list shows our-color ply *p*, but our journal has no POST
     for *p*)
   - repeated 400 "not your turn" on POSTs that our state says are legal

   A game we never accepted is **not** a signal. The operator can
   legitimately accept a challenge on lichess.org while logged in as the
   bot (E20), so that event is only logged.

   Action: immediately stop moving in that game (never race another
   engine), go Offline, and raise the alarm "another client is playing on
   this account".
- Operational rule, stated in the Settings token panel: **one bot token,
  one machine.** Revoke tokens you aren't using (lichess.org → Preferences →
  API access tokens).

## 7. Challenge policy

This is a pure decision function: `(challenge, settings, liveState) →
.accept | .decline(reason) | .ignore(why)`. It is unit-tested for every rule,
and every decision is logged with the rule that fired. Each rule maps to the
most specific Lichess decline reason.

| Rule | Default | Decline reason |
|---|---|---|
| Bot is Offline or Draining, or rate-limit cooldown/drain is active | — | `later` |
| Model source unavailable (§9) | — | `later` |
| Variant ≠ standard (not configurable, DCM plays standard only) | decline | `standard` |
| `initialFen` present and ≠ startpos | decline | `standard` |
| Rated (setting: accept rated) | **off, casual only** | `casual` |
| Casual (setting: accept casual) | on | `rated` |
| Speed allowlist: ultraBullet / bullet / blitz / rapid / classical | all five on (was blitz, rapid on; bullet, classical off) | `tooFast` / `tooSlow` |
| Correspondence / unlimited | decline | `timeControl` |
| Clock limit below min / above max (s) | 15 / 1800 (min was 180) | `tooFast` / `tooSlow` |
| Increment below min / above max (s) | 0 / 30 | `timeControl` |
| Opponent is BOT (setting: accept bots) | on | `noBot` |
| Opponent is human (setting: accept humans) | on | `onlyBot` |
| Opponent rating outside [min, max] | 0 / 2500 (max was 4000) | `generic` |
| Provisional opponents (setting) | on | `generic` |
| Blocklisted username | — | `generic` |
| At `maxConcurrentGames` | 12 (was 2) | `later` |
| `gamesReservedForHumans`: a bot challenge when only reserved slots remain | 2 (was 0) | `later` |
| Already `maxSimultaneousGamesPerOpponent` vs this opponent | 1 | `later` |
| Daily cap: total games today / per opponent today | 2000 / 5 (was 200 / 20) | `later` |
| Rematch of the previous game (setting: accept rematches) | on | `generic` |
| `compat.bot == false` (Lichess says the challenge isn't playable through the Bot API) | decline | `timeControl` |
| `direction == "out"` (a challenge **we** sent, echoed on the stream) | never answered | — (no request) |
| Over `challengeResponseBudgetPerMinute` (§5.3) | 10 | **ignore** (no request) |

The original defaults were conservative on purpose:
- **Casual only:** the research found humans rating-farming a weak bot got
  bot rated play restricted. *Still the incoming default (`acceptRated`
  off).* Matchmaking's own outgoing challenges are now rated by default
  (§7.3 B).
- **No bullet:** HTTP round-trip time alone flagged a winning 1+0 game on
  time (lichess-bot #36). DCM's fast inference doesn't help, because the
  bottleneck is the network.

**Since commit `c5542b8` (confirmed intentional 2026-10-01) the defaults are
the owner's running configuration**: every speed from ultraBullet to
classical, a 15 s minimum clock, opponents rated up to 2500, 12 concurrent
games with 2 of them reserved for humans, and 2000 games a day with at most
5 against any one opponent. The bullet risk above still holds; the §14.4
lost-on-time breaker and the Stats lost-on-time counts are how it shows up.
Unchanged by that commit: casual accepted, rated declined, bots and humans
both accepted, provisional opponents accepted, rematches accepted, increment
0–30 s, clock maximum 1800 s, 1 simultaneous game per opponent, and a
challenge-response budget of 10 a minute.

### 7.1 Picking opponents: one game at a time, and outgoing challenges (decided 2026-09-28)

Lichess never assigns a bot an opponent. BOT accounts can't use the lobby,
seeks or pools, so every game begins as a challenge, either incoming (the
policy above) or outgoing. Outgoing challenges were deferred to Phase 8;
they now move into **Phase 5**.

- **Play one game.** Sets concurrency to 1 for this run and goes Online.
  It takes the next game that starts, from an accepted incoming challenge
  or an outgoing challenge accepted by its recipient. Once that game has
  started, the bot moves to Draining, and it goes Offline when the game
  ends. The normal concurrency setting is untouched.
- **Challenge… sheet.**
  - Pick an opponent from **online bots** (`GET /api/bot/online`,
    unauthenticated, streamed as NDJSON, refreshed on demand), or type any
    username (for example the operator's own human account).
  - Shows each candidate's ratings and our head-to-head record (§14.3a).
  - Choose time control, color and rated/casual. The defaults come from the
    §7 policy (blitz/rapid, casual), and choices outside that policy are
    allowed only with a visible warning. *(As built, the sheet opens at 5+3,
    random color, casual; these are not settings and `c5542b8` did not
    change them.)*
  - Sends `POST /api/challenge/{username}` through the gate at
    challenge-response priority.
  - Shows the pending challenge with a **Cancel** button
    (`POST /api/challenge/{id}/cancel`).
  - The game starts through the normal `gameStart` path when they accept.
    A decline is shown with Lichess's reason.
- **Limits:**
  - outgoing challenges to one bot count toward the same
    `botPairDailyStop`, short of Lichess's 100-per-pair daily cap *(not
    built as a local stop; see the §7 table)*
  - an outgoing challenge nobody answers is withdrawn after
    `outgoingChallengeTimeoutSeconds`, default 999 s (was 180 s before
    `c5542b8`); 0 waits indefinitely. Busy bots often leave challenges
    unanswered rather than declining them
  - at most one outgoing challenge pending at a time
  - no automatic matchmaking loop (that remains Phase 8)
  - an outgoing challenge echoed back on the event stream
    (`direction == "out"`) is never answered (§7 table)
- **Token scopes:** `bot:play` **and** `challenge:write`. The token check
  requires `bot:play` and reports whether `challenge:write` is present. The
  Challenge… sheet is disabled, with the reason shown, when it isn't.

### 7.2 Finding bots to challenge: search, rating range, favorites, daily limits (added 2026-09-28)

**Status: implemented 2026-09-28** (`LichessBotBotList`, `LichessBotPlayerNotes`, `LichessBotChallengeSheet`; tests in `LichessBotBotListTests`).

**Added the same day:**
- **Username tab:**
  - DCM's past opponents with nothing typed.
  - Lichess's username autocomplete after 3+ characters (`GET /api/player/autocomplete?object=true`). It is prefix-only, about 10 results, and shortest names first (lila `userIdsLikeFilter`, `regexStart`).
  - Selecting a row loads the player.
- **Leaderboard tab:**
  - `GET /api/player/top/{nb}/{perf}`, where `nb` is at most 100 (spec).
  - It ranks players with a stable rating who played a rated game in that speed within 7 days (lila `RankingApi`: each rated game writes an entry that expires in 7 days).
  - Every entry carries `online`, so there is an "online only" filter.
- **Online Players tab:** lila's undocumented `GET /player/online?nb=50` with `Accept: application/json`. It returns the 50 highest-rated online humans, bots excluded (`byIdsSortRatingNoBot`, cached for 2 min).
  - It was added in 2015 (lila PR #541) for the old mobile app. It has no per-route rate limit and isn't blocked for crawlers. Nothing forbids it, but it is undocumented, and staff say they'd "much rather add an endpoint than have people use web-scraping" (api-tips page).
  - So it is called at most once per 2 min, at housekeeping priority, and a failure says the undocumented route may have changed.
  - There is no documented way to list online humans.
- **Bots may challenge humans** (verified in lila `ChallengeGranter`):
  - The human's preferences decide: a block or **Never** denies; **Friends** requires the human to follow the bot; **Rating** needs non-provisional ratings on both sides within ±300; **Registered** (the default) and **Always** allow it.
  - Rate limits for a bot challenging a human who doesn't follow it are about **5 a minute and 40 a day** (5 credits each, from 25 a minute and 200 a day), against 25 and 200 for bot targets.
  - Humans can't pick `noBot` in the web decline menu, but can decline, block, or change preferences.
  - `GET /api/user/{name}?challenge=true` (used by the mobile app, not in the spec) returns `canChallenge`, without the rating-range check.

**Source facts** (verified in `lichess-org/api` and lila source, 2026-09-28):
- `GET /api/bot/online?nb=N` has one parameter, `nb`, the most bots to return (1–512, default 100). It needs no login and returns NDJSON, one full `User` per line: perfs, flair, verified, `createdAt`, `seenAt`, `playTime`, and `profile` (`bio`, `realName`, `links`, `flag`).
- lila builds the list from bots seen in the last 10 s, caches it for 10 s, and then **takes the first N lines**. There is no paging, sorting or filtering. With more than 512 bots online, the API returns an arbitrary 512 (the web page `/player/bots` lists them all). Favorites therefore never depend on this list; see below.
- A perf's `prog` is lila's `Perf.progress`: the newest minus the oldest of the last (up to 12) recorded ratings, i.e. the rating change over roughly the last 12 rated games. Recording starts after the player's 10th game in that perf. It is 0 when there is no history.
- **Bot-vs-bot daily limit:** no endpoint reports it. A challenge beyond it is refused with HTTP 400 and the text `<user> played 100 games against other bots today, please wait until <ISO-8601 time> to challenge them.` (observed 2026-09-28). The time is exact.

**Sheet** (`LichessBotChallengeSheet`):
- **Tabs:** Online Bots · Favorites · Username.
- **Star column** on the left of the Online Bots table. Starred bots sort to the top, ahead of the table's chosen sort.
- **Search:** case-insensitive substring over username, real name and bio.
- **Rating range:** min and max fields applied to the rating for the *selected time control's speed* (5+3 filters on blitz, 10+5 on rapid). The fields follow the time control, and "hide provisional" is a toggle. Blank means no bound.
- **Detail line** under the table for the selected bot: real name, the first line of the bio, account age, and our head-to-head.
- **Freshness:**
  - "updated N min ago".
  - The list refreshes when the sheet opens, on Refresh, and automatically once it is more than 5 minutes old while the sheet is open, checked every 60 s.
  - Always at `.housekeeping` priority, so a fetch never starts while a move is due.
- **Daily limits:**
  - A refusal in that form records the bot's "available again at" time, and the bot shows "limit until 1:57 AM" in orange until then.
  - The same parse applies when the named user is *us*.
  - The header shows DCM's own count: bot games in the last 24 h, from our records, against Lichess's 100.

**Favorites and limit times** persist in `LichessBot/player-notes.json` (atomic write, through the file queue), not in Settings, so the settings schema is untouched. Ids are lowercased Lichess user ids.

The Favorites tab lists every favorite, online or not:
- Online status for favorites missing from the online list comes from one `GET /api/users/status?ids=…` (up to 100 ids, `.housekeeping`).
- Offline favorites are shown dimmed.

**Validation.**
- Unit tests:
  - refusal parsing: exact text, fractional-second ISO time, a different count, a non-matching message
  - notes round-trip and a missing file (empty notes)
  - filtering: search, range on the selected speed, provisional hidden, favorites first
  - our 24 h bot-game count
- Live checks:
  - star a bot; it sorts first and persists across relaunch
  - a refused challenge marks the bot with the parsed time
  - the Favorites tab shows an offline favorite dimmed
  - the auto-refresh fires after 5 minutes at housekeeping priority

### 7.3 Challenging several players, and matchmaking (added 2026-09-28; approved)

**Status: implemented 2026-09-29** (`LichessBotChallengeQueue`, `LichessBotMatchmaking`, `LichessBotController` queue pump and matchmaking passes, `LichessBotChallengeQueueList`, Settings ▸ Matchmaking; tests in `LichessBotChallengeQueueTests`, `LichessBotMatchmakingTests`, `LichessBotChallengeQueueControllerTests`). Where the build chose among readings of this section:
- One selected player is still sent at once, with any refusal shown in the sheet; two or more go into the queue.
- A skipped entry stays listed with its reason and is not retried; choosing that player again re-queues them at the end.
- "Fill Open Slots" fills every open slot outside those reserved for humans, whatever the fill mode, one send at a time with the same spacing and per-hour cap.
- DCM's own rating counts for the relative window only when it is established (not provisional).
- "Prefer favorites" defaulted to on. *(Off since `c5542b8`; see B.)*
- The per-hour cap counts challenges that reached Lichess; the spacing applies to every attempt.
- Matchmaking also pauses during "Play one game".
- An accepted challenge's game now holds its slot while its session is being set up (the manager reports starting sessions with accepted challenges), so an automatic send can't take that slot in the gap.

This supersedes §7.1's "no automatic matchmaking loop". All sends still go one at a time through the gate, and every check of a single send still applies: online status, the concurrent-game and per-opponent limits, and the bot daily limit.

**A. Challenge queue (multi-select).**
- **Multi-select.** The Challenge sheet's player tables (Online Bots, Favorites, Online Players, Leaderboard, History) allow multiple selection (⌘/⇧-click). Send becomes "Send N Challenges", with the one clock, color and rated setting applied to all of them.
- **The queue.** Selected players go into an ordered **challenge queue** owned by the controller (its single source).
  - It sends the next entry whenever a slot is free: games in progress + accepted + pending + in flight < `maxConcurrentGames`.
  - A player already at the per-opponent limit, offline, or at their bot limit is skipped with a reason shown, and the queue moves on.
  - A send that fails for another reason (a Lichess refusal) is dropped with the reason. A 429 stops the queue until the gate reopens, and the entry stays.
  - Duplicates of a queued or pending opponent are not added.
- **UI.** The Overview lists the queue: opponent, clock, and why it's waiting ("waiting for a free slot") or was skipped. It has per-entry Cancel and Clear Queue.
- **Lifetime.** The queue is in memory, cleared by Go Offline (along with the pending-challenge withdrawal) and by quitting. Draining stops sending but keeps the queue until offline.

**B. Matchmaking (auto-challenge). Planned off by default; on by default since `c5542b8`.**

The defaults below are the owner's running configuration (commit `c5542b8`, confirmed intentional 2026-10-01). The planned defaults, which kept a fresh install from sending anything until the operator opted in, are kept in parentheses. With matchmaking on and rated by default, a fresh install (or a Reset to defaults) starts sending rated challenges as soon as the bot goes Online.
- **Settings** (Settings ▸ Matchmaking, live):
  - enabled (default **on**; planned off);
  - fill mode: **every free slot** (default) or **only when idle** (no game in progress);
  - time controls to use (a subset of the sheet's clock choices; default ¼+0, 1+0, 1+1, 2+1, 3+0, 3+2, 5+0, 5+3, 10+0; planned 3+2, 5+3). Not in the default set: 10+5, 15+10, 30+0, 30+20;
  - rated (default **on**; planned off);
  - opponent rating window relative to DCM's own rating for that speed (default −300…+300). DCM's rating comes from `/api/account`. When DCM has no rating at that speed, separate absolute bounds apply (default 0…2200; planned 1000…2200); the log says which bounds were used;
  - "prefer favorites" (default off; planned on);
  - a cap on challenges sent per hour (default 10; planned 20);
  - a decline cool-down per bot (default 2 h; planned 6 h).
- **Slots.** Matchmaking never takes the `gamesReservedForHumans` slots, so a human can always challenge DCM.
- **When it runs.** Only while Online (not Draining, not in a 429 hold, not after a breaker trip), and only when the challenge queue (A) is empty, because the operator's own picks go first.
- **Bot list.** While matchmaking is on, the online-bots list is refreshed on its own cadence at housekeeping priority (a named constant, a few minutes), since the Challenge sheet's refresh only runs while the sheet is open. A pass with a list older than that refreshes it first.
- **Picking.** From that list, a uniformly random bot that:
  - fits the rating window at the chosen speed and is not provisional there;
  - is not blocked;
  - has no game, pending or queued challenge with us;
  - is not at its bot daily limit (known limit time);
  - did not decline us within the cool-down;
  - is under the per-opponent daily limit.

  "Prefer favorites" picks among fitting favorites first. The time control is chosen uniformly from the configured ones.
- **Rate.** At most one send in flight, a minimum spacing between sends (named constant), and the per-hour cap. This stays far below Lichess's challenge limits (25 credits a minute and 200 a day; a challenge to a bot costs 1). It also leaves DCM's own bot-games budget (100 a rolling day) as the binding limit, and matchmaking stops at it.
- **Declines.** A decline records the bot's cool-down. A `casual`-reason decline for a rated request records it too; matchmaking does not resend.
- **"Fill Open Slots" button** on the Overview: runs one fill pass on demand with the matchmaking criteria, even when matchmaking is off.
- **Logging.** Every pick and skip reason, and every send, goes to the protocol log (`challenge`), and a `[LICHESS-BOT]` session-log line records each send.

**Validation.**
- **Unit tests** (pure logic, a new `LichessBotMatchmaking` type):
  - candidate filtering, one test per exclusion rule;
  - reserved-human slots respected;
  - the per-hour cap and spacing;
  - queue ordering, duplicate rejection, and skip reasons;
  - the queue stops on a 429 and resumes after;
  - fill mode "idle" vs "every free slot".
- **Controller-level tests** with the existing fakes:
  - N selected with k free slots sends k and queues N−k;
  - a game ending sends the next queued entry;
  - Go Offline clears the queue and withdraws pending challenges.
- **Live:** select several online bots and confirm the sends are staggered and the queue drains as games end. Then enable matchmaking for a session and confirm slots stay filled, favorites and cool-downs are honored, and the per-hour cap holds.

## 8. Engine-side prerequisites (changes to shared code, tests first)

Each item keeps one source of truth. Existing behavior is unchanged for
every current caller.

**8.1 Castling alias in `ChessMove.parseUCI`.** Accept the king-to-rook form
(`e1h1`, `e1a1`, `e8h8`, `e8a8`) **only when** the matching castling move
(`e1g1`, `e1c1`, …) is in the legal list. This is unambiguous in standard
chess: apart from castling, a king never moves more than one square, and
`h1`/`a1` (`h8`/`a8`) hold its own rook, so no ordinary king move can have
those squares as a destination. It affects `--uci` harmlessly: UCI GUIs send `e1g1` in standard
mode. Outgoing moves stay `e1g1`; the first live games confirm Lichess
accepts that (§20.11).
**Add the missing round-trip tests first:** no test currently covers
`parseUCI`/`uci` for castling, en passant, or (under)promotion.

**8.2 Server-authoritative adjudication in `ChessGameEngine`.** Add an init
option, `adjudication: .automatic` (default, today's behavior) or
`.serverAuthoritative`.

In `.serverAuthoritative`, the engine sets `result` **only** for checkmate
and stalemate (no legal moves, so we literally can't move). Threefold,
50-move and insufficient material never end the game locally. Lichess
decides, and ends bot games on those conditions automatically (§4).

This mode exists for the case where DCM's local definitions are stricter
than Lichess's: DCM must never stop moving while the server says the game
is live.

A pure helper, `localDrawCondition(engine) -> LocalDrawCondition?`, still
reports when DCM *believes* threefold, 50-move or insufficient material
holds. It is used only for **disagreement logging** (E10): DCM says draw
but the server says `started`, or the reverse. It never drives a request. Known
discrepancy to document: `PositionKey` includes the en-passant square even
when no en-passant capture is possible (`ChessGameEngine.swift:42-47`).
DCM can therefore under-detect a threefold that Lichess recognizes. This
only affects disagreement logging, because Lichess's own key decides the
game. Self-play keeps `.automatic` and is untouched.

**8.3 Single-pass W/D/L.** `ChessNetwork.internalEvaluate` also reads back
`results[valueProbs]`, which is already computed in the same `graph.run`,
and passes it to the consumer. The new `evaluate` variant returns
`(policy, wdl)`. The existing `(policy, scalar)` signature stays for current
callers.

Factor the `.wdlSoftmax` / `.scalarTanh` projection
(`ChessNetwork.swift:1242-1258`) into **one** helper used by both
`evaluateValueDistribution` and the new path. `evaluateValueDistribution`
can then be reimplemented on top of the single pass, or left as is. No
behavior change either way.

Test: the W/D/L from the new path equals `evaluateValueDistribution`'s
output for the same board (bit-exact, or within fp tolerance for the
network's dtype).

**8.4 SAN formatter + PGN writer.** Neither exists; the importer only parses
SAN.
- `SANFormatter` is pure. It covers piece letters, disambiguation (file,
  then rank, then both), captures, en passant, promotion (`=Q`), castling
  (`O-O`, `O-O-O`), and check `+` / mate `#`.
- **Validated by round-trip through the existing importer's resolver**
  (`PGNImporter.resolveLegalSANMove`): every legal move in the perft test
  positions and a sample of corpus games must round-trip.
- `LichessBotPGNWriter` writes:
  - the Seven Tag Roster plus `Site`, `TimeControl`, `WhiteElo`/`BlackElo`,
    `WhiteTitle`/`BlackTitle`, `Termination`
  - DCM-specific tags `[DCMModelID]`, `[DCMModelSource]`, `[DCMBuild]`
  - per-move comments with `[%clk h:mm:ss]` (standard) and
    `[%dcm p=… wdl=w,d,l lat=…ms]`. Unknown `%` commands are ignored by
    PGN tools.

**8.5 Shared network build helper.** Lift `PlayController.buildInferenceNetwork`
and `buildBareInferenceNetwork` (`:989-1011`) into one shared helper used by
human play and the bot. This is a pure move; human play's behavior is
unchanged.

**8.6 Prove concurrent evaluate is safe.** A test fires N concurrent
`evaluate` calls on one network with different boards. The results must
equal N sequential calls. If the test passes, update the stale doc comment
(`ChessNetwork.swift:1137-1142`) to state the actual guarantee: the serial
`executionQueue` serializes callers, and the consumer must copy out before
returning. If it fails, each model slot gets a per-game network pool
instead (§9).

## 9. Model sources: separate from human play, same user experience

`LichessBotModelSource` mirrors the human-play picker: a radio group, with
unavailable rows disabled and a `.help` reason.

| Source | Snapshot semantics | Refresh |
|---|---|---|
| **Champion** (follows promotions) | Export of `session.network`, keyed by champion ModelID | Re-snapshot **only when the champion ModelID changes** (a promotion). Champion weights change only on promotion, so re-exporting per game would be wasted work. The change is detected when it happens (heartbeat check of the ModelID), and the new generation is built **proactively**, so one is ready before the next challenge (E15). New games always start on the **newest ready** generation, and the record says exactly which. |
| **Trainer snapshot** | Export of `trainer.network`, pinned | Only when the operator clicks **Re-snapshot now** (or selects the source) |
| **Live trainer** | Export of `trainer.network` into a **shared mirror** | On a cadence, `liveTrainerRefreshIntervalSec` (default 120). **Not per move**: human play's per-move export costs about 0.5 s per move and contends with SGD, which multiplies with concurrent games (§4). |
| **Model file** | `.safetensors`/`.dcmmodel` via `CheckpointManager.loadModelFile` (the single loader) | Pinned. Records path, file SHA-256 and embedded ModelID. |
| **Follow lineage** (added 2026-10-06, §9.2) | The newest `.safetensors` file in `Models/` of one training run (`lineage_run_id`), from a chosen segment on, through every exact resume; loaded through the same loader | Checked every `lineageCheckIntervalSeconds` (default 60, minimum the 15 s poll); a newer file of the lineage builds a new generation. Never steps back to an older file. Records the file's lineage position (run, segment, local and cumulative step, `content_sha256`). |

**Default source: Model file** (since commit `c5542b8`, confirmed intentional 2026-10-01; it was **Champion** before). The default path is the owner's chosen checkpoint, hard-coded as an absolute path:
`/Users/andrew/Library/Application Support/DrewsChessMachine/Models/20260713-v5cont-resume-replay-step270000.safetensors`
- **Portability caveat:** the path is specific to the owner's account and machine. On any other machine or account, or after that file is moved or pruned, the default does not exist. Availability only checks that a path is set, so the failure shows up when the generation is built, as the loader's error. There is no fallback to another source (below); choose another file or source in Settings.
- Live-trainer defaults are unchanged: refresh every 120 s, `midGameRefresh` off.

**Generations, not in-place mutation.** Each refresh creates a new
*slot generation*, a network with its own weights. New games (and, only if
`midGameRefresh` is on for Live trainer, new moves) use the latest
generation. Games in progress keep their generation until they end. A
generation is freed once no game references it. At most two or three
networks exist at once, and each is memory-reported with sizes in base-2
units.

**Every move records its generation:** `{sourceKind, modelID, trainingStep
at snapshot, snapshotAt, arch description, build}`. Every game's stats row
is attributable to exact weights.

**Unavailable source means no silent fallback.** For example, Trainer is
selected but no trainer exists. New challenges are declined with `later`.
The status chip shows "Model source unavailable: no trainer", and the log
records it. Games in progress keep their generation and finish normally.

*Superseded 2026-10-06 (owner rule OD-18; `LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md` §3.10):* the bot builds its model generation before it goes online, for every source, and a failed build keeps it offline with the error. While online a generation always exists, so nothing is declined for the model: a source that becomes unavailable (a vanished champion, a failed switch, a followed lineage with no file it can vouch for) keeps the last good generation playing, raises an alarm and is retried at the poll loop's backoff. There is still no fallback to another source.

**Snapshot cost** is logged per refresh (`[LICHESS-BOT] snapshot ms=…`) and
summed in Stats as "training time spent on bot snapshots", so the cost to
training is visible.

### 9.1 Choosing a model file by lineage (added 2026-09-28)

**Status: tree implemented 2026-09-28** (`ModelLineageTree`, `LichessBotModelLinePicker`; tests in `ModelLineageTreeTests`). Deviations:
- **Probe pElo ("strongest (probe)") is not shown.** The app has no access to the dashboards' `data/*.csv`, which live in the repo, not in the app's data. Showing it needs an import step or the ROADMAP's in-file provenance, so it is deferred rather than guessed.
- Session champions hang under their base ModelID's segment, labeled with the session folder.
- A parent cycle, which should never happen, is broken by placing each id once, and ids reachable only through a cycle are listed as roots.

**Status.** A first version shipped with the live-testing fixes: "Latest by lineage" (`ModelFileCatalog`, `LichessBotModelLinePicker`). It groups files by `model_id`, shows the highest-step file first, and lists the lineage's earlier files underneath. Live use showed the grouping is too fine, and the result misleads:
- The untrained seed "Qeu8" appears as its own lineage, and its trained continuations appear as unrelated rows (GLu5, Lnji, PVZp, Ejp0).
- The operator picked the seed by mistake: game qgwVLXEV was played by an untrained network.

**Design (no dependence on `cumstep_base`).** Everything comes from the files' own metadata: `model_id`, `parent_model_id`, `training_step`, `created_at_unix`, `creator` and `replay_*`. The dashboards' registries are not read. A cumulative step is shown only once files record it (ROADMAP "Lineage provenance in every checkpoint"); until then it is left out, not estimated.
- **Tree by seed.** Follow `parent_model_id` up to a root, a file with no parent or whose parent isn't in the folder. The top level is one row per root, e.g. "Qeu8 · untrained seed · built 2026-07-02".
  - Checked 2026-09-28: 18 roots across 3,754 files, and no `model_id` has conflicting parents.
  - If one ever does, show it as an error row naming both parents rather than picking one.
- **Branches.** Under a root, each chain from the root to a leaf is a branch, labeled with its path (Qeu8 → GLu5 → Lnji → PVZp → Ejp0) and its training kind from `creator` (replay, vs-UCI, manual).
  - Segments shared by several branches (e.g. X79T under Qeu8) are not duplicated. A branch lists only its own segments beyond the fork point.
- **Per segment:** the `model_id`, its step range (first–last `training_step` in the folder), its date range, and the number of files. Expanding it lists the files, newest step first. Each file appears exactly once, and the segment row itself selects that segment's latest file.
- **Which file to suggest.** The branch tip, i.e. the leaf segment's highest-step file, is the default, marked "latest".
  - When `data/<run>.csv` in the dashboards has pElo for a file, show it next to the file, with a rolling mean of ±25 checkpoints next to the raw value. These values are keyed by the file's `model_id` plus `training_step`, never by filename.
  - The best-by-rolling-mean file is marked "strongest (probe)". Raw probe scores scatter by about ±35 between checkpoints, so a single spike is not the peak.
  - When there is no probe data the columns stay blank.
- **Untrained seeds are labeled** "untrained (no training_step)" and sorted after trained branches, so they can't be mistaken for a line's latest model.
- **Session champions** (`Sessions/*.dcmsession`, e.g. Ejp0-3 … Ejp0-66) are shown as further branches.
  - A session's `champion.safetensors` records an **empty** `parent_model_id` (checked 2026-09-28 on `…-Mh5n-sigusr2.dcmsession`: champion `20260921-2-Mh5n-3`, parent `""`; its trainer file does name the champion as parent).
  - So a champion is placed by its **base ModelID**, the ID minus the `-N` promotion suffix (Ejp0-66 → `20260727-1-Ejp0`). That branch is labeled "self-play from <base>", and which checkpoint of the base it started from is shown as unknown rather than guessed.
  - The Ejp0 self-play runs 1 and 2 both minted Ejp0-1…-10 from the same seed. They are told apart by the session folder, and shown as separate runs.
  - The ROADMAP item "Lineage provenance in every checkpoint" makes this exact for future saves.
- **Search** matches any `model_id` in a branch's path, so "Qeu8" finds Ejp0.

**Validation.**
- Unit tests cover the tree builder:
  - roots, including a parent missing from the folder
  - shared prefixes, not duplicated
  - a conflicting parent, reported as an error
  - a seed with no step, labeled untrained
  - ordering: each file in exactly one place
- A test on probe data keyed by `model_id` plus `training_step` checks that a filename which disagrees with the metadata is ignored.
- Live check on this Mac's Models folder:
  - Qeu8 shows the replay branch ending in Ejp0 (step 1,397,000), the Qeu8e branch ending in sFzi, and the vs-UCI branch ending in syxR.
  - Selecting the Qeu8 root row makes it obvious that it is untrained.
  - The count of files listed equals the file count in the header.

**Suggested extra (optional, Phase 8): Rotation / A-B.** Alternate between
two sources per game, for example champion vs. a pinned baseline file.
Lichess's human pool then becomes a live head-to-head comparison of models,
and Stats-by-ModelID shows the result.

### 9.2 Following a model lineage on disk (added 2026-10-06)

The **Follow lineage** source plays the newest checkpoint of one training run written by a separate process (`--replay-corpus`, `--train-vs-uci`), which the in-process sources can't see. Identity comes only from each file's `dcm_lineage` record (run + chosen segment), never from filenames or modification times; the models folder is re-checked on a cadence with a header cache, and forks, conflicts and missing files are reported (alarm, last good generation keeps playing) rather than resolved silently. Design, owner decisions and implementation notes: `documentation/plans-active/LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md`.

## 10. Data layer

The root is `CheckpointPaths.rootURL/LichessBot/`, using the central root.
The existing Corpora/SessionIndex/Performance code rebuilds the root on its
own; this feature does not add to that duplication.

**10.1 Layout**
```
LichessBot/
  Games/YYYY/MM/<YYYYMMDD-HHMMSS>-<gameId>.json   final record (single source of truth)
  Games/YYYY/MM/<YYYYMMDD-HHMMSS>-<gameId>.pgn    PGN with DCM annotations
  Games/YYYY/MM/<YYYYMMDD-HHMMSS>-<gameId>.journal.jsonl  raw per-game journal, kept
  InProgress/<gameId>.journal.jsonl               journal while the game is live
  Protocol/events-YYYYMMDD.jsonl                  account-level protocol event log
  Challenges/challenges-YYYYMMDD.jsonl            challenge log, one file per UTC day, kept forever (challenge-log plan)
  Challenges/reconstructed-from-protocol.json     past challenges rebuilt from Protocol/ (derived, regenerable)
  index.json                                      rebuildable stats cache (never authoritative)
```

**10.2 Game lifecycle**

1. **During play:** an append-only journal records every received line (raw
   JSON, plus our receive timestamp), every decision (move chosen with
   internals, draw/resign/claim decisions with the W/D/L that drove them),
   every request and response for this game, and every anomaly. It is
   appended through a serial `DispatchQueue` work executor (the same pattern
   as `SessionLogger`), so file I/O never runs on a cooperative-pool thread.
   It is synchronized at the end of each move cycle.
2. **Game end** (§6 definition), then **finalize**:
   - Build the record from the journal.
   - Queue an **export reconciliation** (JSON with `clocks`, `opening`; §5.3
     spacing).
   - Compare our moves, result, status and clocks against the export. The
     export wins for result, status, rating diffs and opening. Any mismatch
     is recorded in the record's `reconciliation` block **and** flagged in
     the protocol log. This is where the resignation-stream gap (#60) gets
     caught.
   - Write `.json` + `.pgn` atomically (tmp + rename).
   - Move the journal into `Games/`.
   - Update `index.json`.
3. **Launch-time recovery:** every leftover `InProgress/*.journal.jsonl` is
   either a live game, resumed when the event stream replays its
   `gameStart`, or a game that finished while we were down, finalized from
   the journal plus the export. Nothing is dropped. A journal whose export
   fails stays in `InProgress/` with a visible "unreconciled" badge and
   retries later.
   - While the bot is still offline after a launch, the status chip reports
     the leftovers from each journal's recorded finish (review 2026-10-03):
     a game whose journal records its finish only waits to be filed (a quit
     doesn't wait for the post-game chat fetches), so it is counted as
     "to file" with a session-log line and no alarm; a game with no
     recorded finish may still be live on DCM's clock and raises an alarm;
     a journal that can't be read raises an alarm naming why. A report
     computed while the bot was going online is dropped, since going
     online resumes or files those games itself.
4. **Integrity:**
   - Journals tolerate a truncated last line (a crash mid-append). The
     reader drops only an incomplete final line and logs it.
   - Records carry a `schemaVersion`, and decoding uses tolerant enums
     (`unknown(String)`) so a new Lichess status value is logged and never
     crashes the app or drops a game.

**10.3 Protocol event log (`Protocol/*.jsonl`)**

One JSON object per line, for every:
- stream open/close/stall/reconnect with duration and reason
- keep-alive gap statistics
- request, with category, status, latency and `Retry-After` (the **token is
  never logged**; a redaction helper covers every logged request line)
- 429 and cooldown
- breaker trip
- challenge decision with rule and reason
- account/token check result
- snapshot refresh
- negotiated HTTP protocol version per connection (`networkProtocolName`,
  E32)
- unknown event type or field value

Files rotate daily. Retention: **keep indefinitely** (decided 2026-09-28),
the same as the per-game journals. They are small, and
the UI shows total size in base-2 units.

**10.4 SessionLogger integration.** High-level events only, tagged
`[LICHESS-BOT]`: connect, disconnect, and a one-line summary per game with
result, opponent, TC, model and moves. Alarms use `[ALARM] LICHESS-BOT …`.
Per-move lines go to the journal only. The existing session log keeps its
signal-to-noise.

**10.5 `index.json`** follows the `SessionIndex` precedent: a derived cache
of the per-game summary fields that Stats needs. It is rebuilt from
`Games/**/*.json` when missing, stale (count or mtime mismatch), or on
request ("Rebuild index"). There is a single source of truth, and the cache
can always be discarded.

**10.6 What a game record contains**
- **Identity:** gameId, URL, createdAt, start and end timestamps.
- **Setup:** speed/perf, clock, rated, variant, initialFen.
- **Players:**
  - our color
  - opponent id/name/title, bot or human, rating before and after,
    provisional flag
  - our rating before and after
- **Result:** result, status, winner, termination (from the export), move
  count.
- **Model:** the model generation block (§9).
- **Per move:**
  - ply, side, SAN
  - `uciAsGiven`: the raw token exactly as Lichess sent it (the opponent's
    moves) or exactly as we sent it (ours). Never normalized. See E1.
  - server clocks from the state (both sides)
  - receive time, and our latency breakdown (encode, inference, sample,
    POST round-trip)
  - `chosenProbability`, top-5 with probabilities, W/D/L, τ used, legal move
    count, generation id
- **Game events:** draw and takeback offers with our responses, chat log
  (both rooms), `opponentGone` events and claims.
- **Anomalies:** stream reconnects during the game, rejected moves.
- **Reconciliation** block.

## 11. Statistics

**Implemented in part (2026-10-06):** the Overview's Record card now holds
the period table (last hour … all time, with performance rating, average
opponent rating and rating change), a Rated / Casual / All filter, and panes
for time controls, models, self-assessment, endings, opponent strength, move
choice, clock, game length, openings, opponents and bot health. Design,
definitions and decisions: `LICHESS_BOT_RECORD_STATS_PLAN.md`. The
combinable filters and group-by tables below remain to be built.

**Filters** (combinable):
- date range
- model source kind and ModelID
- build
- speed / perf
- rated / casual
- our color
- opponent kind (human / bot / titled)
- opponent rating range
- termination
- game-length bucket

**Group-by tables** (sortable; SwiftUI `Table`, see §14). Group by any one
of: ModelID, source, speed, color, opponent kind, opponent-rating bucket,
termination, day, or week. Columns:
- games, W / D / L, score %
- **performance rating** (standard formula, shown with its sample size)
- average opponent rating
- average length
- time-forfeit rate
- median / p95 move latency
- requests per game

**Charts:**
- **Lichess rating over time** per perf, from
  `GET /api/user/{id}/rating-history` (E49; it includes later
  rating-refund adjustments). The per-game "at the time" diffs are an
  overlay, and promotions are marked as vertical rules. Uses
  `FastLineChart`, the project's time-series convention.
- **Score vs. opponent-rating bucket** (bars with expected-score overlay).
- **W/D/L by ModelID** (stacked bars): did newer champions do better against
  the same Lichess population?
- **Value-head calibration on real games** (reliability diagram): bucket
  every position by predicted expected score `E = p_win + ½·p_draw`, and
  plot the actual game score per bucket. This is the learning-oriented view,
  and it is the evidence needed before turning on eval-driven resign/draw
  (§12.4).
- Move-latency distribution and percentiles, and clock remaining at game
  end.
- Request rate by category (from §5 telemetry).

Aggregation is **pure** functions over the index rows, which makes it
unit-testable. It is computed off the main actor and published to the view.
**CSV export** covers the filtered game rows.

## 12. Settings

**12.1 Persistence**

`LichessBotSettings` is a `Codable` struct stored in UserDefaults under
`DrewsChessMachine.LichessBotSettings.v1` (versioned-key precedent).

**Not `TrainingParameters`.** These aren't training knobs, they don't belong
in `.dcmsession`, and putting them there would misapply the 9-step parameter
checklist.

Decode failure is **not** a silent default. The bot stays offline, and the
window shows "Saved settings unreadable". An explicit **Reset to defaults**
button preserves the unreadable blob under a `.unreadable` key for
forensics.

**12.2 Token: accepting and storing it**

*Entry, in the app only (single source of truth):*
- Settings ▸ Account has a masked `SecureField` ("Paste token") and a
  **Save & Validate** button.
- Nothing is stored until validation passes. Validation costs two
  housekeeping requests through the gate (§5), only on an explicit click:
  - `POST /api/token/test` → user id, scopes, expiry
  - `GET /api/account` → username, `title`, `count.all` (the upgrade
    guard needs it)
- It is **accepted** only if:
  - the token is valid
  - its scopes include `bot:play` (and `challenge:write`, which is optional:
    without it the Challenge… sheet is disabled; §7.1)
  - it is not expired
  - its user id is **`drewschessmachine`** (the configured account, so a
    token for a different account is refused)
- `title != "BOT"` is shown as a warning, not a refusal, because the token
  is needed to run the upgrade itself.
- On acceptance:
  - it is written to the Keychain
  - the field is cleared
  - the panel shows "Token saved for DrewsChessMachine · bot:play,
    challenge:write · expires <date | never>", with **Replace** and
    **Remove** buttons
- The token is **never shown again**, never logged (a redaction helper
  covers every logged request), never placed in `UserDefaults`, settings
  JSON, records or crash text, and never written to disk outside the
  Keychain.

*Storage:* `LichessBotTokenStore`, the first Security-framework use in
the project.
- Service `com.drewben.DrewsChessMachine.lichess-bot`, account = the
  Lichess user id.
- Add/update/read/delete with every `OSStatus` surfaced and none swallowed.
- **Keychain choice** (settled, §18):
  - **(A) Data-protection keychain. CHOSEN 2026-09-28.**
    `kSecUseDataProtectionKeychain = true` with
    `kSecAttrAccessibleAfterFirstUnlock`, exactly the requested semantics.
    It works with the screen locked (E36), the item is private to DCM, and
    there are no per-build access prompts. Cost: a one-time
    **Keychain Sharing entitlement** added to the Xcode target (the project
    has no entitlements file today). Automatic signing manages the
    provisioning. Items there are **not visible to the `security`
    command-line tool**.
  - **(B) Legacy login keychain.** No project change, and it is shared with
    the `security` CLI. There is no per-item after-first-unlock attribute:
    the item is readable while the login keychain is unlocked, which by
    default is from login until logout, screen lock included. That breaks if
    the login keychain is set to lock on sleep or inactivity. After a
    signing-certificate change, an "Always Allow" prompt may appear.

*Use:*
- The token is read **once** at Go Online and held in memory for
  reconnects, so an overnight reconnect never touches the Keychain.
- **Remove** deletes the item and takes the bot Offline.

*Expiry:*
- A token within 7 days of expiry raises a warning chip.
- An expired or revoked token (401/403) means Offline plus an alarm (§5.5).

*No shell step, ever:*
- The token is never handled outside the app. There are no curl scripts
  and no shell variables, and nothing is stored anywhere but the app's
  Keychain item.
- All protocol verification happens inside the app, from its protocol log
  and journals (§20.11).
- If a copy was stored earlier with `security add-generic-password` (as
  suggested on 2026-09-28, before this section was written), delete it. It
  sits in the login keychain, invisible to option (A):
  `security delete-generic-password -s com.drewben.DrewsChessMachine.lichess-bot -a drewschessmachine`.

*Guarded BOT upgrade (Settings ▸ Account):*
- The **Upgrade account to BOT…** button is enabled only when:
  - a validated token is saved
  - `GET /api/account` shows `title != "BOT"` **and** `count.all == 0`
    (Lichess refuses accounts that have played any game)
- It opens a sheet stating in plain words that this is **irreversible** and
  that the account can then never play as a human.
- Confirming requires **typing the account name exactly**.
- On confirm, it sends `POST /api/bot/account/upgrade` through the gate,
  then re-reads `/api/account` to verify `title == "BOT"`.
- Every step is logged to the protocol log and as
  `[LICHESS-BOT] upgrade …` in SessionLogger.
- Once the account is a BOT, the button is hidden (opacity-0 + zero-frame
  per UI convention) and the account card shows the BOT title.

**12.3 Live-apply semantics.** Every setting is read from the `@Observable`
store at its decision point, so a change takes effect at the next occurrence
of that decision. Each group in the UI says which:

| Group | Applies at |
|---|---|
| Challenge policy | next incoming challenge |
| Play style (τ, draw/resign/claim/takeback, think delay) | next move / next offer |
| Model source | next game (Live trainer with `midGameRefresh`: next refresh) |
| Concurrency / caps | next challenge (never aborts running games) |
| Alert tones | next incoming challenge |
| Chat templates | next game |
| Heartbeat / backoff / rate-limit spacing | next connection / next request |
| Token | requires reconnect (explicit button) |

**12.4 Play-style options and defaults**
- **Temperature schedule:** `SamplingSchedule` start/decay/floor. Defaults
  are start 0.11, decay 0.02/ply, floor 0.01 (the owner's running
  configuration since commit `c5542b8`, confirmed intentional 2026-10-01).
  τ is driven by game-total ply, so it reaches the floor at ply 5: DCM's
  first two or three moves get a little variety and everything after is
  ≈argmax. The planned defaults were start 0.5, decay 0.05/ply, floor 0.01
  (≈argmax by about ply 10), chosen to give opening variety without
  weakening later play. The new start keeps the opening much closer to
  argmax, so E45's repeated-line risk is larger.
  The minimum is 0.01; τ=0 is unsupported. No Dirichlet noise, ever.
- **Minimum think time:** default 0 ms. Optional, for spectator
  friendliness.
- **Resign:** default **off**. When enabled: `p_loss ≥ threshold` (0.95)
  for N consecutive own moves (6), and only after ply ≥ 40.
- **Offer draw:** default **off**. When enabled: `p_draw ≥ threshold` (0.85)
  for N moves (8) after ply ≥ 60. The offer rides on the move through
  `offeringDraw`, which saves a request.
- **Accept draw offer:** default **decline all**. When enabled: accept if
  expected score `E ≤ threshold` (0.45).
  - Declining costs **no request**. On Lichess, making a move implicitly
    declines a pending offer, so the bot never sends `draw/no`.
  - Accepting rides on our move as `offeringDraw=true`, which is an
    agreement when the opponent is offering.
  - The only standalone `draw/yes` is an accept sent while it is **not**
    our turn. That is rare, and it goes through the gate at game-critical
    priority.
- **Threefold / 50-move:** **no knob.** Lichess ends bot games on these
  automatically (§4, lila `autoThreefold` claims for any BOT player). There
  is nothing to claim and no way to play on. So DCM never reaches positions
  past a threefold or past 50 moves on Lichess, which matches self-play.
- **Claim victory / claim draw when the opponent has left**
  (`opponentGone`), default **on**:
  - Once `claimWinInSeconds` elapses, **claim victory**. Lichess decides
    what the claim is worth. If it refuses the victory claim (400, "You
    cannot claim victory"), claim a draw once. Both outcomes are logged.
  - Re-verify with a fresh stream state first, because `opponentGone` is
    documented as unreliable across reconnects (lila #18664/#18665).
  - If `gone:false` arrives, cancel any pending claim.
- **Takebacks:** accept up to N per game (default 0, i.e. decline).
- **Abort:** there is no bot-initiated abort. Lichess auto-aborts no-start
  games. `expiration` is tracked for display only.

**Rationale for the eval-driven defaults:** resign and draw decisions trust
the value head's calibration. The §11 calibration chart on real games is how
to earn that trust before turning them on.

**12.5 Chat:** a greeting template and a goodbye template. Placeholders are
`{modelID}`, `{source}`, `{build}` and `{opponent}`. The greeting defaults
to on: "Hi, I'm DrewsChessMachine ({modelID}), a from-scratch neural net,
no search. Type !help for commands." (since commit `c5542b8`; before it,
"DrewsChessMachine: a from-scratch neural net, no search. Model {modelID}.
Type !help for commands."). The goodbye defaults to off ("Thanks for the
game, {opponent}!"). The room is selectable and defaults to the player
room.

**12.5a Chat commands (added 2026-09-28; supersedes E53's "no chat commands").** Research: `documentation/research/lichess-bot/chat-commands.md`, verified in lila, lichess-bot and BotLi source.

**Status: implemented 2026-09-28** (`LichessBotChatCommands`, `HardwareInfo`; tests in `LichessBotChatCommandsTests`).

- **Commands:** `!help`, `!name`, `!about`, `!motor`, `!cpu`, `!gpu`, `!ram`.
  - Every command is safe to answer in both rooms: none reveals the value head's evaluation or the policy for the current position. `!eval` and `!top` are deliberately out of scope for now, because they leak engine help to the opponent.
  - A reply goes to the room the command came from.
- **Parsing:**
  - A line counts as a command when, after trimming, it starts with `!` followed by a known command word, matched case-insensitively as a *prefix*. So `!name2` or `!help please` still work: Lichess's duplicate filter blocks a human who repeats the exact same text, and lichess-bot's PR #967 does the same.
  - Lines from our own account are ignored, because our greeting mentions `!help`. So are lines from the system user `lichess`.
  - Anything else is displayed as before and never interpreted.
- **Replies:** each is a single message, except `!about`, which sends three, each at most 140 UTF-16 code units (Lichess counts `String.length`).
  - `!about` is three messages, not one message with newlines. lila keeps a single `\n` through cleanup, but how the chat renders it is unverified, and each message gets its own 140.
  - Replies are built by a pure function and checked against the limit. An optional tail, such as `!name`'s "no search" note, is appended only when it fits. A reply whose core doesn't fit is not sent, and an anomaly is logged.
  - Replies avoid emoji, chess symbols, anything link-like (including "github.com") and all-caps text. lila strips or lowercases those, or drops the message silently with a 200.
- **Budget:** replies are requests through the one request gate, so an opponent must not be able to spend our budget.
  - Per game: at most one command reply every 3 s, and at most 20 command replies per game.
  - No replies when our clock is under 30 s.
  - A skipped command is logged as a note, not answered.
  - The greeting and goodbye are separate and unchanged.
- **Content:**
  - `!name`: `<our username> running DCM <modelID> step <n> (build <b>) · no search, one forward pass per move`. The step is omitted when the model has none.
  - `!motor`: `DCM <modelID> step <n>`.
  - `!about`:
    1. "Drew's Chess Machine is a from-scratch chess engine by drewster99 (GitHub). A neural net picks each move in one forward pass: no search."
    2. "It learns from self-play and from replayed games. It's written in Swift and runs on Apple silicon, on the GPU through Metal's MPSGraph."
    3. "This machine: <cpu>, <cores>-core CPU (<per-level counts>), <gpu cores>-core GPU, <memory> GB unified memory."
  - `!cpu`, `!gpu` and `!ram` are the corresponding pieces of message 3.
  - `!help`: "Commands: !name, !about, !motor, !cpu, !gpu, !ram. I answer in the chat you ask in."
- **Hardware facts** are read once at launch (`HardwareInfo`), never per command:
  - the CPU brand (`machdep.cpu.brand_string`)
  - per-performance-level core counts and names (`hw.nperflevels`, `hw.perflevel<i>.physicalcpu` / `.name`)
  - memory (`hw.memsize`, reported in base-2 GB)
  - the GPU model and core count (IORegistry `AGXAccelerator` `model` / `gpu-core-count`)
  - A fact that can't be read makes the command's reply say "unknown" for that fact, and the failure is logged. Nothing is guessed.
- **Record:**
  - Each incoming command line is already journaled as chat.
  - Each reply is journaled as its request, plus an action note: "replied to !name (spectator)".
  - A budget skip is journaled as a note.
- **Validation:**
  - Unit tests cover:
    - parsing: prefix match, case, ignoring our own lines, non-commands
    - every reply built from maximal-length inputs stays ≤ 140 UTF-16 units
    - the optional tail is dropped when it doesn't fit
    - budget: cooldown, per-game cap, and the low-clock skip
    - `LichessBotChat.message` / `templateProblem` counting UTF-16 units rather than `Character`s. The old `Character` count undercounted surrogate pairs; fixed with this change.
  - Live checks:
    - `!help` and `!about` in the player room and from a logged-in spectator, with replies arriving in the same room
    - a repeated `!name` still answered
    - `!about`'s three messages shown intact on lichess.org

**12.6 Connection / safety:**
- **Auto-connect on launch:** default **off**.
- **Prevent system sleep while online:** default **on** (§13).
- **Reconnect backoff:** start / cap.
- **Stall timeouts** (§6).
- **Rate limits** (§5): the minimum-cooldown floor is fixed and not
  editable.
- **Circuit breakers** (§14.4).

**12.7 Challenge alert tones (added 2026-09-29, commit `c5542b8`):**
- Two settings (Settings ▸ Alerts): a system sound for challenges from BOT
  accounts and one for challenges from humans, each with a preview.
- **Default: none for both** (no sound plays).
- Every incoming challenge sounds, whatever the policy then does with it:
  the tone means "someone wants a game", not "a game is starting".
- Our own outgoing challenges, echoed back on the event stream, never sound
  (the challenger is our own account), or every matchmaking send would.
- Read live on each arrival (`LichessBotChallengeAlert`, tests in
  `LichessBotChallengeAlertTests`).

## 13. On/off, drain, quit, signals

States: **Offline**, then **Connecting**, then **Online** (accepting). From
Online the bot can move to:
- **Draining** (not accepting new games; games continue), then Offline when
  they finish.
- **Cooldown** (429, §5), then Draining.
- **Error** (token, takeover, breaker). It stays Offline until the operator
  acts.

Controls live in the window, a Chess menu section, and the status chip menu:
- **Go Online**
- **Stop Accepting (Drain)**
- **Go Offline**: allowed only with no games in progress; otherwise it
  offers Drain or Resign all
- **Resign all games…**: confirmation, then a resign request per game
  through the gate at game-critical priority

**Quit** (`AppDelegate.applicationShouldTerminate`, currently never
confirms). *Decided 2026-09-28: the default is to drain.* With games in
progress, return `.terminateLater`, move the bot to **Draining** at once
(new challenges are declined with `later`), and show a **"Finishing games"
sheet**:
- It lists each game in progress with live status: opponent, move number,
  both clocks, and whose turn.
- The app quits by itself when the last game ends. That is the default
  path, with no click needed.
- **Abort**: resign every game (through the gate at game-critical
  priority), then quit.
- **Quit now**: games are abandoned. The opponent can claim after the
  timeout, and launch recovery reconciles them. Behind a confirmation.

Without games in progress the quit stops the runtime and is answered
once the bot has shut down. That holds while offline too (review
2026-10-03): going offline started withdrawing our unanswered
challenges, and the shutdown waits for those withdrawals and for queued
protocol-log, player-notes and outcome-log writes. Only a launch that
never went online or loaded the bot's notes quits at once.
- **Cancel**: don't quit; the bot stays Draining. Go Online again from the
  window if wanted.

The same sheet serves **Go Offline** with games in progress, minus the
quit.

**Sleep and App Nap.**
- Setting: **"Prevent system sleep while the bot is online"**, default
  **on**. While it is on and the bot is Online or holding any game, the bot
  holds a
  `ProcessInfo.beginActivity([.userInitiated, .idleSystemSleepDisabled])`
  assertion.
  - The app holds none today, and an idle sleep mid-game forfeits every game
    (E35).
  - With the setting off, the bot still holds `.userInitiated` (App Nap
    off), and the Overview shows "sleep not prevented".
- The same assertion for **training** runs is a separate roadmap item
  (ROADMAP, 2026-09-28).
- A forced sleep (lid close) can't be blocked. Wake recovery handles it
  (E34), and the Overview warns about it.

**Signal exits** (SIGUSR1, SIGHUP and SIGUSR2 end in `_exit`, bypassing
AppKit). The bot registers a best-effort, *network-free* pre-exit hook
with `EarlyStopCoordinator`. The hook closes the event stream and writes an
`abandoned-at-exit` marker into each live journal, so the deploy is never
delayed on the network. Launch recovery finalizes those games. The Chess
menu also gets a **"Drain bot for deploy"** action, so the operator can
empty the bot before sending SIGUSR2.

## 14. UI

The UI follows project conventions:
- one `View` per file
- no `some View` helper properties (child structs instead)
- no `AnyView`
- no `if`-gated visible content (opacity-0 plus zero-frame in lockstep)
- `@Observable` controllers
- monospaced, padded digits everywhere
- aligned tabular layout
- full light/dark support
- accessibility labels

**14.1 Window.** `LichessBotWindow` follows the `LichessProbeMonitorWindow`
template: `NSWindowController` + `NSHostingController`, a single-instance
registry, `isReleasedWhenClosed = false`, and a `LichessBotWindowLauncher`.
Closing the window does **not** stop the bot; the bot is app-level. It opens
from **Chess ▸ Lichess Bot…** through a new `AppCommandHub` closure slot,
following the `openLichessProbeMonitor` pattern.

**14.2 Status chip** (`LichessBotStatusChip`) in `TitleBarView`'s right
status area, which is always visible:
- a state dot, "Online · 2 games · 5-2-1 today"
- red for cooldown or error, amber for draining or token expiring
- click to open the window; context menu with the §13 controls

**14.3 Window sections** (sidebar):
1. **Overview:**
   - the state machine and big on/off/drain controls
   - account card (username, BOT title, ratings per perf, token
     expiry/scopes)
   - current model generation card (source, ModelID, step, snapshot age,
     **Re-snapshot now**)
   - gate/cooldown status, and requests per minute by category
   - **live game cards**, one per active game, containing:
     - a read-only board (a `HumanPlayBoardView` variant with last-move and
       check highlight, oriented to our color) with move list
     - both clocks, ticking locally between server updates
     - a W/D/L bar (a new shared view; `ArenaHistoryView`'s `WLDBar` is
       private)
     - opponent line and TC
     - per-game **Resign** / **Open on lichess.org** buttons
2. **Games:** a sortable `Table` of all games (date, opponent, rating,
   color, TC, result, termination, moves, model, reconciled ✓/⚠). This is
   the project's first SwiftUI `Table`; sortable multi-column data is the
   core need. A detail pane shows:
   - board replay with a SAN move list
   - per-move DCM data (probability, top-5, W/D/L, latency, clocks)
   - chat
   - game anomalies and the reconciliation diff
   - buttons for PGN copy/open, JSON open, and lichess.org
3. **Stats:** the §11 filter bar, group-by table, charts and CSV export.
4. **Events:** the protocol log viewer (filter by category or severity,
   game id, text search; follow-tail toggle; size readout).
5. **Settings:** the §12 groups. Each group carries an "applies at …"
   caption (§12.3), inline validation, and a **Reset group** button.
   Changes apply immediately, with no Save button. Invalid input is
   rejected inline and never partially applied.

**14.3a Live game viewing and statistics (decided 2026-09-28)**
- **Fonts:** the system font the rest of the app uses, with monospaced
  variants for digits. No custom font.
- **Live view:** one game shown large, **defaulting to the first-started
  game still in progress**, with a picker for the others. When that game
  ends, the view moves to the next-oldest game in progress.
- **Grid view:** a toggle that shows every game in progress at once, one
  tile each with a small board, clocks, W/D/L bar and opponent. Clicking a
  tile focuses that game in the large view.
  - **Finished games stay in the grid for a while.** A tile gets a result
    badge and remains for a configurable time (default 10 min) or until
    dismissed; a **Clear finished** button removes them all. A popped-out
    window keeps its game until the window is closed.
- **Pop-out:** any live game opens in its own standalone single-game window,
  and several can be open at once. Closing one affects nothing but that
  window.
- **Single-game view contents:**
  - a read-only board oriented to our color, with last-move and check
    highlights
  - both clocks, ticking locally between server updates
  - a W/D/L bar
  - a **PGN move list** like the human game view's
  - DCM's per-move data for the selected move
  - the game chat
  - the **protocol transcript** (below)
- **Browse-only stepping, everywhere (decided 2026-09-28).** Every game
  view, live or finished, can step through earlier positions *for viewing
  only*. It never changes the game.
  - A view cursor sits apart from the game position. Use ◀ ▶ and Home/End,
    or click a move in the list.
  - While browsing, a banner reads "Viewing ply N · live at ply M" with a
    **Back to live** button (End does the same).
  - New moves don't pull the view away from the ply being browsed. The
    live ply, both clocks and the W/D/L bar keep updating.
  - Live views have no way to change the game's position.
  - This is a shared component: a browse cursor plus board plus move list.
    The **human game view adopts it too**. Today, clicking a move there only
    selects it for **Revert**, and the board never shows that position.
    Revert stays a separate, explicit action.
- **Protocol transcript (chat style), per game, both directions.** Lichess
  bots don't speak UCI: traffic is NDJSON lines streamed from Lichess and
  HTTP requests from DCM.
  - Incoming stream lines are shown on one side with receive times.
  - DCM's outgoing requests are shown on the other: method, path, form
    fields, status, latency and the negotiated protocol.
  - Keep-alives are shown collapsed into a counter, but **the full
    transcript is logged** for debugging:
    - every game-stream line, including each keep-alive with its receive
      time, goes into the game's journal
    - every event-stream line (challenges, `gameStart`/`gameFinish`, …)
      goes into the protocol log
    - event-stream keep-alives are logged as per-minute gap statistics,
      plus an individual entry for any gap longer than twice the keep-alive
      interval (at one every 7 s, logging each one individually would add
      about 12k lines a day; a gap past the stall limit ends the stream and
      is logged as a stall)
  - The token and the `Authorization` header are never recorded.
  - This needs one addition to the Phase 2 and 3 code: the API client
    reports each request (with its game id) to an observer. Requests go
    into the game's journal (a new `request` journal event) and the
    protocol log, so the transcript also exists for finished games.
- **Statistics:** full §11 stats, plus **head-to-head records**. An
  **Opponents** table lists every opponent with games, W/D/L, score %,
  their current and first-seen rating, last played, and our performance
  rating against them. Opponent detail lists every game against them. The
  record against an opponent is also shown on that opponent's live game
  card and in the challenge log.
- **One game at a time:** see §7.1.

**14.3b Opponent card, and keeping our own account current (added 2026-09-28).**

**Status: implemented 2026-09-28** (`LichessBotOpponentCard`; tests in `LichessBotOpponentProfileTests`).
- Placement: a third panel ("Opponent") beside Transcript and Chat in the game view, and a compact summary under the Challenge sheet's table.
- Fetch triggers: sending a challenge; accepting one (before the game starts); selecting a bot in the sheet; opening the Opponent panel.
- Lichess's own AI has no account, so it shows a note instead of a card.
- The rating-history trend and per-speed stats remain optional, not built.

*Our account.*
- The Overview's Account card (ratings, rated and unrated counts) is refreshed after **every filed game**: one `GET /api/account` once the reconciler finalizes the record. **Implemented 2026-09-28.**
- It runs at housekeeping priority, and a failure is logged while the previous values stay on screen.

*Opponent card.* Shown in the Challenge sheet (for the selected or looked-up player) and beside the opponent in the game view. It is built from public endpoints, with no extra token scope. Fields are from the `User` / `UserExtended` schemas in `lichess-org/api`, checked 2026-09-28:
- **Identity:** username, title (BOT, GM, …), flair, verified; patron (`patronColor` present; the `patron` boolean is deprecated); account age (`createdAt`); last seen (`seenAt`); and `disabled` / `tosViolation` flags when present.
- **Ratings:** per speed (bullet, blitz, rapid, classical, correspondence): rating, provisional mark, `rd`, recent `prog`, rated `games`, and global `rank` when present.
- **Record:**
  - their `count` totals (all, rated, W/D/L, and vs humans: `winH` / `lossH` / `drawH`), and `playTime.total`
  - **our head-to-head**, from both DCM's own records and Lichess's crosstable (`GET /api/crosstable/{us}/{them}`). The crosstable covers games DCM never recorded.
- **Profile** (humans; shown as plain text, links never followed): flag, location, OTB ratings when set.
- **Later, optional:** the rating-history trend (`GET /api/user/{u}/rating-history`) and per-speed stats (`GET /api/user/{u}/perf/{perf}`: best wins, streaks). Game exports are out of scope; they are the most rate-limited call.

*Fetching: background only, never blocking.*
- Each opponent is fetched **at most once per app session**, into an in-memory cache keyed by lowercased user id. The cache is not persisted, so each launch gets fresh data.
- Fetches run in a detached background task at **`.housekeeping` priority** through the one request gate. The gate never *starts* housekeeping while any game awaits our move, and in a low-clock game only moves and game-critical actions are eligible (E18).
- The irreducible cost: a housekeeping request **already in flight** when an opponent moves makes our move POST wait for that one round trip (about 100–400 ms observed). To keep even that out of games:
  - Prefer fetching when a challenge **arrives** or is **sent**, before the game exists.
  - In the game view, fetch only if the cache has no entry.
- A card never waits on a fetch. It shows what's cached and fills in when the fetch lands. A failed fetch shows "couldn't load (reason)" with a Retry button; it is never retried automatically.
- Nothing about an opponent affects challenge acceptance or play.

*Validation.*
- Unit tests:
  - decoding the full `UserExtended` / crosstable example payloads, including absent optional fields
  - the cache: one fetch per id per session; concurrent requests for the same id share one fetch
  - a fetch never enqueued while a move is awaited (gate eligibility, with a fake gate)
- Live checks:
  - send a challenge and watch the protocol log: the opponent fetch is `housekeeping` and lands before the game starts
  - during a game, no opponent fetch starts while a move is due
  - the Account card updates after a filed game

**14.3c Chat panel: wider bubbles, operator chat, per-game move delay or hold (added 2026-09-28).**

**Status: implemented 2026-09-28.** Deviations from the text below:
- Operator messages are journaled as a general `chatSent(room:text:origin:)` event, with origin greeting, goodbye, command reply or operator, rather than an operator-only event. It is sent with the request label "operator chat".
- That event also fixed the missing-goodbye bug (live findings).
- "Play move" also ends a plain delay early.
- Tests: `LichessBotMovePacingTests`, `LichessBotSentChatTests`.

- **Bubble width.** In both the Transcript and the Chat panel, each bubble may take up to **75%** of the panel's width, aligned left (Lichess) or right (DCM), so long lines wrap less.
- **Operator chat input**, under the Chat panel of a live game:
  - A text field, a **room picker** (Player / Spectator), and Send.
  - A live counter shows UTF-16 units against 140; Send is disabled over the limit.
  - A warning appears when the text looks like a link (a URL, or `name.tld`), because Lichess silently drops link-bearing messages from bots.
  - Sending uses the same `POST /api/bot/game/{id}/chat` through the request gate at `.chat` priority.
  - The message is journaled as an **operator** message (a new `operatorChat(room:text:)` journal/transcript event), distinct from DCM's automatic greeting, goodbye and command replies. The game record keeps who said what.
  - Lichess allows this: bot rules restrict moves, not chat (research doc, "Can a human type chat through the BOT account?").
- **Per-game move delay / hold**, in the game view:
  - **Delay:** a per-game "wait before moving" of 0–30 s (default 0), applied before posting DCM's move.
  - **Hold:** DCM decides its move, shows it, and posts only when the operator clicks **Play move**.
  - **Clock safety** for both: the wait ends early, and a held move is posted automatically, once our clock falls to **30 s + 2 × the last move round trip**. The auto-post is logged.
  - Neither applies before DCM's first move, so a game can't be aborted by our own wait.
  - Neither is persisted: they belong to that one game and end with it. Changing them mid-game takes effect on the next move.
- **Validation:**
  - Unit tests:
    - the link heuristic and the UTF-16 counter
    - the clock-safety rule (release time from clock and round trip)
    - delay/hold never applied at ply 0 or 1 of our moves
    - the journal round-trip of the operator-chat event
  - Live checks:
    - an operator message in each room shows on lichess.org and in the transcript as an operator message
    - Hold with a short clock auto-plays at the threshold and logs it

**14.4 Circuit breakers** are configurable and shown in Overview. Each trip
logs, alarms, and moves the bot to Draining or Offline:
- ≥ 1 × 429 → cooldown + drain; 2 within 60 min → Offline (§5)
- ≥ 3 lost-on-time in the last 10 games → Draining (a latency problem)
- ≥ 5 move-POST failures within 10 min → Draining
- a reconnect storm (≥ 10 reconnects within 10 min) → Offline
- event-stream takeover (§6) → Offline
- 401/403 → Offline
- model source unavailable → decline-only until it returns

## 15. Code organization

The new top-level folder is `DrewsChessMachine/LichessBot/`, picked up
automatically by the synchronized group:
- `API/`
  - `LichessBotAPIModels` (tolerant `Codable`)
  - `LichessBotNDJSON` (splitter)
  - `LichessBotStreamReader`
  - `LichessBotRequestGate`
  - `LichessBotAPIClient`
  - `LichessBotTokenStore` (Keychain)
  - `LichessBotUnits` (`Seconds` / `Milliseconds` wrapper types, E17)
- `Play/`
  - `LichessBotGameSession` (per-game state machine)
  - `LichessBotGameStateReducer` (pure)
  - `LichessBotMoveChooser` (encode → evaluate(policy + W/D/L) → `MoveSampler` → top-k)
  - `LichessBotModelSlots` (generations)
  - `LichessBotChallengePolicy` (pure)
  - `LichessBotPlayPolicy` (pure: draw/resign/claim/takeback)
  - `LichessBotChat`
- `Data/`
  - `LichessBotRecordStore` (journal/finalize/reconcile/atomic writes)
  - `LichessBotProtocolLog`
  - `LichessBotIndex`
  - `LichessBotPGNWriter`
  - `LichessBotSettingsStore`
  - `LichessBotInstanceLock` (`flock`, §6.1 A)
- `Stats/`: `LichessBotStats` (pure aggregation)
- `UI/`: window, launcher, chip, and one file per view
- `LichessBotController`: `@MainActor @Observable`, owns the lifecycle and
  publishes UI state
- `LichessBotSleepAssertion`: `ProcessInfo` activity, held while online
  (§13, E35)

Shared code touched by §8:
- `ChessMove+UCI.swift`
- `ChessGameEngine.swift`
- `ChessNetwork.swift` (+ `ChessMPSNetwork` wrapper)
- a new `Chess/SANFormatter.swift`
- a shared network-build helper
- `AppCommandHub`, `UpperContentView.wireMenuCommandHub`,
  `DrewsChessMachineApp` (menu)
- `TitleBarView`, `AppDelegate`, `EarlyStopCoordinator`

**Concurrency model** (actors: settled, §18):
- The network-facing async state machines are Swift **actors**:
  `LichessBotAPIClient`/gate, each `LichessBotGameSession`,
  `LichessBotModelSlots`. They are async-first around
  `URLSession.bytes`, need no synchronous access, and never block.
- **Actor contention.** No actor is a single global hub for game traffic.
  - Each game has its own `LichessBotGameSession` actor, so games never wait
    on each other's message processing.
  - The only deliberately shared serialization point is the request gate,
    which is Lichess's single-flight rule (§5) and is required.
  - Actors never run slow work inline. GPU inference and file I/O leave the
    actor through their executors (`executionQueue`, the file queue), so an
    actor is never busy while it waits on them.
  - `LichessBotModelSlots` hands out generation references. It does no
    evaluation, so a snapshot refresh never blocks a move on another game.
  - UI reads come from `SyncBox` snapshots the actors publish, not from
    `await`ing actors on the main actor.
- **Actor reentrancy** (the actor-specific bug class):
  - While an actor `await`s (a POST, an inference), other messages for it
    can run, and its state may change underneath.
  - Rule: after every `await`, re-validate what the next step depends on
    before acting. For example, after inference, confirm the position is
    still the same ply and still our turn before posting the move.
  - Posting is guarded by the per-ply idempotency check (§16, Phase 3).
  - **[T]** a stub delivers a `gameState` while inference is in flight.
- File I/O goes through a serial `DispatchQueue` work executor.
- GPU work goes through the existing network `executionQueue` +
  continuation bridge.
- Small synchronous snapshots read by the UI use `SyncBox`.
- No `Task { try … }` swallows errors. Every task catches and routes errors
  to the protocol log and the controller.
- No `try?`, no force-unwraps. Unknown JSON values surface as
  `unknown(raw)` and are logged.

## 16. Phases

Each phase runs implement → **build** → targeted tests → commit
(CHANGELOG entry per project convention, no attribution lines) → next phase.
The full test suite runs after Phase 1 (shared network/engine code) and
before final sign-off.

**Operator prerequisites** (no code, no token handling outside the app)
- The account `DrewsChessMachine` exists with 0 games. **Play no games on
  it.**
- When Phase 5's Account panel exists:
  1. mint a token on lichess.org with only "Play games with the bot API"
     (`bot:play`) and "Create, accept, decline challenges"
     (`challenge:write`); §7.1
  2. paste it into the app
  3. run the guarded upgrade (§12.2)
- Protocol verification (formerly a separate curl step before any code) now
  happens in the app during Phase 5's first live sessions (§20.11). None of
  it blocks earlier phases:
  - castling is parsed in both forms regardless
  - the two-session design is safe under HTTP/1.1 and h2
  - the threefold question is already answered from lila's source

**Phase 1: engine-side prerequisites (§8)**
- Tests first: UCI round-trips (castling, en passant, promotion,
  underpromotion); castling alias; `ChessGameEngine` `.serverAuthoritative`
  (threefold/50-move/insufficient material continue; mate/stalemate end;
  `.automatic` unchanged); W/D/L single pass equals the second-pass result;
  SAN round-trip over perft positions + corpus sample; concurrent-evaluate
  equals sequential.
- Then the implementation. Update the `ChessNetwork` doc comment per the
  §8.6 result.
- *Validation:*
  - all new tests pass
  - **the full suite passes with no existing test modified**
  - a self-play smoke run (`--train`, 3 min) shows unchanged `[STATS]`
    behavior
  - `--uci` still plays

**Phase 2: protocol layer**
- Models:
  - tolerant decoding, verified against the Lichess spec's example payloads
    (vendored into the test target, source path cited). Fixtures captured
    in Phase 5's live sessions are added to the same suite.
  - NDJSON splitter
  - stream reader + heartbeat
- Request gate: single flight, priority, 429 cooldown and breaker, backoff.
- API client and token store.
- URLProtocol-stub test harness.
- *Validation:* the §5.6 and §17.2 stub tests pass, including the
  single-flight invariant under load and zero requests during cooldown.

**Phase 3: play layer**
- Game-state reducer (pure; incremental when the move list extends, full
  rebuild on shrink or resync).
- Game session state machine, including the move idempotency guard: post
  only when it's our turn at ply *p* and nothing has been posted for *p*.
- Move chooser, model slots and generations.
- Challenge, play and chat policies.
- *Validation:* the §17.1 policy/reducer tests pass. Scripted-stub
  full-game tests play complete games end to end, including a takeback, a
  mid-game stream drop with resync, a draw offer, and the resignation-gap
  scenario.
- *As implemented (2026-09-28):*
  - The pure reducer is `LichessBotPositionTracker`. It replays the
    server's move list through `ChessGameEngine` in `.serverAuthoritative`
    mode: incrementally when the list extends, and from scratch on a shrink
    or divergence.
  - A game session reads its stream **sequentially**. A line that arrives
    while inference or a POST is in flight waits in the stream buffer until
    the current step finishes. So the §15 "[T] a stub delivers a
    `gameState` while inference is in flight" case cannot interleave, and it
    is not tested as such. The post-`await` re-validation stays, because
    other entry points (operator resign, the opponent-gone claim task) do
    run on the actor during those awaits.
  - The full-game tests run against an in-process fake Lichess (an actor
    implementing `LichessBotGameAPI`) rather than a `URLProtocol` stub. The
    wire format is covered separately by the Phase 2 client tests.
  - E22: a `gameFinish` on the event stream makes the game's session drop
    and reopen its stream (`requestResync`). The fresh `gameFull` carries
    the finished status. Export reconciliation, in Phase 4, remains the
    backstop.
  - Takeover detection counts only streams that the **server** closed
    within the window. Stalls and transport errors are network problems and
    go to the backoff.

**Phase 4: data layer**
- Journal, finalize, export reconciliation, PGN writer, protocol log,
  index, launch recovery, SessionLogger integration.
- *Validation:*
  - a crash-injection test (truncated journal) recovers
  - the finished-while-offline path finalizes from the export
  - an index rebuild matches the incremental index
  - atomic writes leave no partial files
- *As implemented (2026-09-28):*
  - The journal (`LichessBotJournalWriter`, a game observer) stores the raw
    stream lines as text, plus DCM's own decisions, POSTs, rejections,
    actions, anomalies and the finish. It does not duplicate what the raw
    lines already hold (`gameFull`, chat). Each launch that writes to a
    game's journal first appends a header naming the build and whether it
    resumed an existing file. The file is synchronized after every posted
    move and at the finish.
  - `LichessBotRecordBuilder` is pure: it replays the journal's raw lines to
    rebuild the move list, SAN, retractions (E30) and offer events, and
    attaches our decisions by ply. A move on our side with no POST from
    this client becomes an anomaly, which is the §6.1 B signal. The export
    wins for status, winner, rating changes and opening.
  - Some aborted games have no export (a `404`). If the journal says
    `aborted` or `noStart`, the record is finalized from the journal alone,
    with reconciliation outcome `exportUnavailable`.
  - `LichessBotReconciler` (actor) fetches exports one at a time, at least
    `exportMinimumSpacingSeconds` apart.
    - A still-live export (E23) is retried with backoff for up to 10 min.
    - After that the game is marked unreconciled and retried every 30 min.
    - A game that still has a session is dropped from the queue; its end
      enqueues it again. A session that saw its game finish hands it over
      through the journal's finish (after the post-game chat fetches, or at
      once in a drain); one that ended without a finish (its game stream
      answered 404: a game Lichess deleted) is enqueued by the controller
      when the manager reports the session's end (review 2026-10-03).
    - Launch recovery enqueues every leftover `InProgress/` journal. The
      controller wiring is Phase 5.
  - `LichessBotRecordStore` is a stateless `Sendable` class whose work all
    runs on the one file queue, not an actor: the files are the state.
    `finalize` returns the record decoded from the bytes it wrote, so it is
    identical to the one on disk.
  - The `[LICHESS-BOT]` per-game summary line comes from
    `LichessBotRecordStore.summaryLine`; the Phase 5 controller writes it
    when a game is finalized.

**Phase 5: controller + core UI (usable bot)**
- Controller state machine.
- Window, menu, status chip.
- Overview (live games), Settings (including the token panel and the
  guarded BOT upgrade, §12.2), the §13 controls and quit handling
  (drain by default, with the "Finishing games" sheet).
- Added 2026-09-28:
  - **Play one game** and the **Challenge…** sheet (§7.1)
  - the live single-game view (first-started game by default), grid view
    with finished-game retention, and pop-out windows (§14.3a)
  - the shared browse-only stepping component, adopted by the human game
    view as well (§14.3a)
  - the per-game protocol transcript: the API client reports every request
    to the journal and protocol log (§14.3a)
- *First live sessions:*
  - run the §20.11 **live verification checklist**
  - copy representative captured lines into the test target as fixtures
  - append a "Live verification findings" section to this plan
  - any finding that contradicts §4 or §20 is fixed before Phase 6
- *Validation:* the §17.3 **live acceptance checklist**, run on real
  Lichess.

- *As implemented (2026-09-28):*
  - **Controller and window.** `LichessBotController` (`@MainActor
    @Observable`, app-level) owns the lifecycle. The window has Overview,
    Live Games and Settings; Games, Stats and Events are Phase 6.
  - **Status chip:** in the title bar.
  - **Menu:** Chess ▸ Lichess Bot… (⇧⌘L).
  - **Quit:** `AppDelegate` routes quit through the controller. With games
    in progress it returns `.terminateLater`, drains, and brings up the
    bot window with the "Finishing games" sheet. Without games it returns
    `.terminateLater` and answers after the bot's shutdown, offline
    included, unless the bot was never used this launch.
  - **Player notes and challenge outcomes** load when the bot window
    opens and, if they aren't loaded yet, when the bot goes online (it can
    go online from the status chip without the window). Each loads once
    per launch; a failed load is retried on the next call. Loading the
    notes merges bot limits both ways with the live limits and saves any
    the notes gained (review 2026-10-03).
  - **Protocol transcript:** the API client reports every request, and
    each game's journal records every request, stream line and keep-alive.
    Event-stream lines and keep-alive gap statistics go to the protocol
    log.
  - **Browse-only stepping (`GameBrowseCursor`):** shared by the bot's
    game views and the human game window.
    - In the human window the live board stays mounted and animating under
      a read-only browsed board, because its animation callbacks pace the
      game.
    - Arrow keys are not claimed there (the τ slider uses them). In the bot
      window they are claimed only while the Live section's single view is
      showing.
  - **Pre-ship review fixes:**
    - Resyncs back off, and stop after repeated tries without progress.
    - Repeated move rejections at one ply stop moving in that game.
    - A move on our side that this client didn't send is detected live,
      stops moving, and takes the bot to Error (§6.1 B).
    - A failed sync resets the position tracker.
    - Stream opens wait at most 15 s for headers, and ordinary requests
      time out after 10 s idle and 30 s total.
    - One request gate lives for the controller's lifetime, so cooldowns
      and the breaker's history survive going offline. A 429 now holds new
      games for `postRateLimitDrainMinutes`, then resumes; a breaker trip,
      a rejected token or a takeover goes to Error.
    - Accepted-but-not-started games count against capacity.
    - A challenge answer that arrives before its POST returns is still
      matched.
    - Daily counts are seeded from today's records.
    - A late journal write can never recreate a filed journal, and a
      fragment never overwrites a record.
    - `fullId` and any token-bearing text are redacted.
    - Game streams resync on wake (E34).
    - File stems use UTC (E50).
  - **Second review (recheck) fixes:**
    - A 5xx on a move is retried as a Lichess failure. It is never counted
      as a rejection, and only disagreements count toward giving up on a
      game.
    - Foreign-move checks cover only plies the session has never seen, so
      a resync in a resumed game can't flag DCM's own earlier moves.
    - Events and polls from a torn-down runtime are dropped.
    - The finishing sheet appears even when quitting opens the window.
    - A quit during record filing is still answered.
    - Accepted-but-not-started games count as games in play for drain and
      quit.
    - The sleep option really allows idle sleep when off.
    - The daily seed counts only today's leftover journals.
    - The post-429 hold is a separate manager flag, so it can't undo a
      drain, and it survives going offline and back.
    - Resetting settings reaches the running bot.
  - ~~**Still open:** scaling the game-stream stall limit to the time
    control (§6).~~ Not needed: game streams carry keep-alives (live
    verification findings, below).

**Phase 6: games, stats, events UI**
- *Validation:*
  - stats aggregation unit tests (performance rating, calibration buckets,
    group-by) against hand-computed fixtures
  - UI checked in light and dark, and at a narrow window width

**Phase 7: hardening**
- Circuit breakers, token-expiry monitoring, the signal pre-exit hook,
  "Drain bot for deploy".
- *Validation:* a **soak test of ≥ 24 h unattended** (§17.4).

**Phase 8: optional; needs separate approval**
- Automatic outgoing matchmaking (a loop that picks opponents and sends
  challenges on its own). It respects the 100/day-per-pairing cap and the
  §5 budget. Manual outgoing challenges and "Play one game" moved to
  Phase 5 (§7.1).
- Rotation/A-B model source.
- Headless `--lichess-bot` CLI mode + a launchd `KeepAlive` recipe for
  process-level supervision.

## 17. Test plan

**Coverage rule:**
- Every §20 item tagged **[T]** has at least one named test. The test's
  doc comment cites the E-number, so the mapping can be grepped.
- Every **[LIVE]** item has a captured fixture and a line in the "Live
  verification findings" (Phase 5).
- Every **[D]** item is noted in code at the enforcement point.

A Phase 6 review walks the §20 table and checks all three off.

**17.1 Unit tests (pure, deterministic)**
- NDJSON splitter: chunk boundaries, `\r\n`, keep-alive blanks, partial
  last line, oversize line.
- Decoding every spec example, plus the live-captured fixtures from
  Phase 5. Unknown `type` or enum values decode as `unknown` and are
  logged.
- Reducer:
  - `gameFull` → state
  - `gameState` extension
  - duplicate `gameState`
  - takeback (list shrink)
  - reconnect `gameFull` after divergence
  - castling alias in incoming moves
- Turn/idempotency guard.
- Challenge policy: every §7 rule and its decline reason; budget-exceeded
  means ignore.
- Play policy: resign / draw-offer / draw-accept over synthetic W/D/L
  sequences, including the N-consecutive and ply-minimum edges;
  opponent-gone claims (victory, then draw on refusal);
  `localDrawCondition` disagreement logging (E10).
- Gate: priority ordering, housekeeping deferral, backoff schedule, 429
  cooldown floor (Retry-After shorter than 60 still waits 60), breaker.
- Records: journal round-trip, truncated-line tolerance, reconciliation
  diff rules (export wins).
- Stats: performance rating, calibration buckets, filters, group-by.
- Settings: `Codable` round-trip, validation, unreadable blob → error state
  (no silent default).

**17.2 Integration tests (URLProtocol stub server, no network)**
- A full game from `gameStart` to `gameFinish`.
- A stall with no keep-alives → watchdog → reconnect → `gameFull` resync →
  the game continues.
- 429 on a move → zero requests during cooldown → drain → resume.
- A second 429 → offline.
- 400 on a move → resync, not resend.
- 401 → offline + alarm.
- Event-stream takeover ×3 → offline.
- `gameFinish` without a final `gameState` (resignation) → export
  reconciliation sets the correct result.
- N concurrent games: at most 1 simultaneous non-stream request.
- App "relaunch" (new controller instance on the same data dir) with a
  live journal → resumes.

**17.3 Live acceptance checklist** (real Lichess; BOT account vs. the
operator's human account; casual; each item recorded as pass/fail with its
game id)

1. Accept blitz; decline bullet, rated and variant. The Lichess UI shows the
   correct decline reasons. *(Written for the planned defaults. Since commit
   `c5542b8` bullet is accepted by default, so test the speed decline with a
   speed unchecked.)*
2. Castle both sides; en passant; promotion and underpromotion. All moves
   are accepted and recorded, and the PGN opens cleanly in a PGN viewer.
3. Draw offer declined (default). With the accept policy on, it is
   accepted.
4. A takeback request is declined.
5. The opponent resigns; the bot resigns (threshold temporarily enabled).
   Results match the export.
6. The opponent abandons. The bot claims victory after `claimWinInSeconds`.
7. Wi-Fi off for 30 s mid-game, then back on. Reconnect, resync, the game
   completes, and the anomaly is logged.
8. Quit mid-game with **Quit now**, relaunch. The game resumes, or is
   finalized from the export if it ended meanwhile.
9. Two concurrent games against two browser sessions. Both play; the gate
   log shows single flight.
10. Switch the model source mid-session. The running game keeps its
    generation, and the next game uses the new one. Records show both.
11. Live trainer with training running: refreshes appear at the configured
    cadence, and the snapshot cost is visible in Stats.
12. Drain: the running game finishes, new challenges are declined with
    `later`, then the bot goes Offline.
13. Every §10 artifact is present and correct for every game above. The
    Stats slices agree with a hand count.

**17.4 Soak test (Phase 7)**: ≥ 24 h Online with Play-and-Train running.
- Accept real opponents under the default policy.
- Zero 429s.
- Zero lost-on-time at the default TCs.
- All games reconciled.
- Memory (RSS) flat within tolerance, with sizes in base-2 units.
- Reconnects logged and recovered.
- Training `[STATS]` throughput within 2% of a no-bot baseline.

## 18. Decisions

**Settled (2026-09-28)**
- **Account:** `DrewsChessMachine`. It exists, has 0 games and is not yet a
  BOT.
- **Token:** scopes `bot:play` and `challenge:write` (§7.1; decided
  2026-09-28, which replaces `bot:play` only). It is entered **only through the app's
  Settings ▸ Account panel** and stored in the Keychain (§12.2). It is never
  pasted into chat, a shell, a file, or the repo. The BOT upgrade is a
  guarded button in the same panel.
- **Actors** for the new network code, with the contention and reentrancy
  rules in §15.
- **Casual only** by default. *(Still the incoming default. Since commit `c5542b8`, matchmaking's outgoing challenges are rated by default, §7.3 B.)*
- **Raw journals and protocol logs kept indefinitely** for now.
- **Research files** moved to `documentation/research/lichess-bot/`.
- **Castling recorded as given** (E1).
- **Prevent-sleep option**, default on (§13). The same option for training
  is a separate ROADMAP item.
- **Keychain: option (A)**, the data-protection keychain with
  `kSecAttrAccessibleAfterFirstUnlock` and a one-time Keychain Sharing
  entitlement (§12.2).
- **Default challenge rules approved** as listed in §7. *(Superseded: since commit `c5542b8`, confirmed intentional 2026-10-01, the defaults are the owner's running configuration. §7 lists the current and the approved values.)*

**Still open**
1. **Font of the new Lichess Bot window and status chip.** Your global rules
   prefer Apple Garamond. The DCM app uses the system font everywhere, and
   Apple Garamond is not installed on macOS by default. Choose one:
   - system font + monospaced digits (consistent with the rest of the app)
   - Apple Garamond for text in the new window only (it must be installed,
     with the system font as a declared fallback)
2. *(Approved 2026-09-28; kept for reference.)* **Default challenge
   rules** (§7):
   - casual only
   - standard chess only
   - speeds: **blitz and rapid accepted; bullet and classical declined**
   - clock 3–30 min, increment 0–30 s; correspondence and unlimited
     declined
   - humans and bots both accepted, any rating, provisional OK
   - at most 2 games at once, 1 per opponent
   - at most 200 games/day, 20 per opponent/day
   - rematches accepted

   *Current defaults (commit `c5542b8`, confirmed intentional 2026-10-01):*
   casual only and standard only (unchanged); every speed from ultraBullet
   to classical accepted; clock 15 s–30 min, increment 0–30 s;
   correspondence and unlimited declined; humans and bots both accepted,
   opponents rated 0–2500, provisional OK; at most 12 games at once, 2 of
   them reserved for humans, 1 per opponent; at most 2000 games/day, 5 per
   opponent/day (Lichess itself caps games against bots at 100 a day and
   enforces it; there is no local cap, by owner decision 2026-10-01); rematches accepted.
3. **Default in-game behavior** (§12.4):
   - τ schedule 0.5 → 0.01 by ply 10 *(now 0.11 → 0.01 by ply 5, commit
     `c5542b8`)*
   - never resign, never offer a draw, decline all draw offers
   - decline takebacks
   - claim victory when the opponent leaves (a draw only if Lichess refuses
     the victory)
   - greeting message on, auto-connect off
4. **Signal exits / deploys** (§13). DCM has three Unix-signal paths that
   exit immediately through `_exit` and skip normal quit:
   - **SIGUSR2** = "save session, then exit". It is used before swapping in
     a new build.
   - **SIGUSR1** and **SIGHUP** = exit now.

   Bot games in progress at that moment are abandoned: the opponent can
   claim after a timeout, and launch recovery records the outcome. Choose
   one:
   - (a) **Proposed:** exit immediately, plus a **"Drain bot"** action to
     use beforehand when a clean handoff is wanted.
   - (b) SIGUSR2 waits for bot games to finish before saving and exiting.
     This could delay the exit by a whole game.

## 19. Risks

- **Lichess API changes.** Undocumented rate limits can shift, and new
  enum values appear. Mitigations: tolerant decoding, conservative request
  budget, telemetry to see drift early.
- **Account action.** Farming or abuse by opponents against a weak model.
  Mitigations: casual-only default for incoming challenges, daily caps,
  blocklist, breakers. *(Since commit `c5542b8`, matchmaking sends rated
  challenges by default; turn its Rated setting off to stay fully casual.)*
- **Value-head miscalibration** makes eval-driven resign/draw
  unsafe. Mitigation: off by default, and the calibration chart gates
  turning it on.
- **GPU contention with training.** Bot inference is one forward pass per
  move, which is negligible. Live-trainer snapshots briefly block SGD.
  Mitigations: cadence-based refresh, and cost reported in Stats.
- **macOS 27 beta MPSGraph behavior.** The bot uses only the existing,
  already-exercised inference path, plus a readback of a tensor that is
  already computed. No new graph construction.

## Live verification findings (2026-09-28)

These come from the first live session: account `DrewsChessMachine`, upgraded to BOT through the app, and game `ZQRPySw4`, a DCM win by mate as Black against dala-700, casual 5+3.

- **Our own outgoing challenge is echoed on the event stream without a `direction` field**, unlike the spec's example. The bot tried to accept its own challenge and got a harmless 404. The fix recognizes our own challenges by `challenger.id` before the policy runs; `direction` alone can't be relied on.
- **Event-stream response headers arrive after about 7.4 s**: Lichess holds them until the first keep-alive. Stream opens hold the request gate until headers arrive, so a move could wait behind an event-stream reconnect. Game streams answered in 288 ms. *Open:* hold the gate only to dispatch a stream request, not for its header wait.
- **Keep-alives:** every 7.0 s on the event stream (maximum 7.4 s). Game streams send them too; one arrived in a 9-second game. The §6 "scale the game-stream stall limit to the time control" item is therefore not needed: after the first keep-alive, the event-stream limit applies.
- **Latency:** a move POST takes about 107 ms round trip over h2, and a challenge POST about 394 ms. `accept` returns 404 for a challenge the bot itself sent.
- **The challenge-creation response decoded** through `decodeCreated`, and `gameStart.fullId` appears in the event stream and is redacted in the log (E26).
- **A new BOT account shows provisional 3000? ratings** in every perf.
- **The token-test response** reported the scopes as a comma-separated list, as assumed.
- **Filing:** the game was reconciled against the export with no mismatches, and written as `.json`, `.pgn` and `.journal.jsonl` under `Games/2026/09/`.
- **Bot-vs-bot limit:** Lichess limits each BOT account to 100 games against other bots per rolling day, counted **in total, not per pairing**. It refuses further challenges with its own message ("… played 100 games against other bots today, please wait until …"). DCM's per-pairing stop was removed; Lichess's message is shown instead.
- **A game-ending move arrives twice.** For a threefold (game zZMhCXrX), the `gameState` carrying the move came with status `started`, and a second `gameState` with the same 60-ply move list and status `draw` followed about 1 ms later. The final move list is the only "ended at ply N" marker. DCM logged a false local-draw-rule disagreement on the first message. **Fixed 2026-09-28:** the disagreement is reported only if the next `gameState` is still live. The regression test is `testThreefoldEndedInTheNextStateIsNotAnAnomaly`.
- **Our own chat is echoed only while the game stream is open.** The greeting's echo arrived about 220 ms after its POST. The goodbye, POSTed after the game finished (the session stops reading then), was never echoed, so it was missing from the Chat tab and the record. **Fixed:** sent messages are recorded from their successful POST (`chatSent`) and matched to echoes.
- **Lichess closes the game stream right after a game's final state.** lila's `GameStateStream` pushes the final state, then stops the stream (`PoisonPill` → `queue.complete()`); the 10 s delay there only affects the rematch signal. Reopening the stream for a finished game sends `gameFull` and closes. So post-game chat can't arrive on the stream. **Fixed:** the player-room chat is fetched with `GET /api/bot/game/{id}/chat` (`[{text, user}]`) at +60 s and +5 min, and unseen lines are added and journaled (`chatFetched`). Filing waits for the first fetch unless going offline or quitting.
- **A declined "classical" challenge (2026-09-28) was correspondence.** The challenge had `"timeControl":{"type":"unlimited"}`, `speed: correspondence`, so it was declined with `timeControl`. That was correct; there was no stale-settings bug. Separately, a game session did keep one settings copy per stream. **Fixed:** settings are read per stream line.
- **Several pending outgoing challenges** are now allowed (bounded by `maxConcurrentGames` together with games in progress). A pending outgoing challenge never blocks accepting incoming ones: the policy counts only games in progress and accepted-awaiting-start.
- **`+` in a form body:** `URLComponents.percentEncodedQuery` leaves `+` literal, which the server reads as a space. **Fixed:** form bodies are strictly percent-encoded.
- **Draw statuses:** Lichess reports every agreed or rule draw as `draw`. DCM names the rule from its own engine's view of the final position.

## 20. Where DCM's normal handling and Lichess disagree: edge cases

DCM's code was built for self-play, where DCM controls both sides, the
rules, the clock (there is none), and the lifetime of the game. On Lichess,
the server controls all of those. Each item below lists what DCM normally
does, what Lichess does, and the handling. Tags show how each item is
covered:
- **[LIVE]** verified empirically in Phase 5's first live sessions, from
  the protocol log and journals. The captured lines become test fixtures.
- **[T]** a unit or integration test (§17)
- **[D]** a design rule, enforced in code review and noted in the code

Any [LIVE] finding that contradicts an assumption here is fixed before
Phase 6.

### 20.1 Move notation and position reconstruction

| # | DCM normally | Lichess | Handling |
|---|---|---|---|
| E1 | Castling stored and parsed as king-to-destination (`e1g1`) | `moves` is described as king-to-rook (`e1h1`); for standard games it may be either form | **Accept both forms.** The alias in `parseUCI` is gated on a legal castling move (§8.1) and resolves to DCM's internal move for the engine only. **Records keep every move exactly as given:** `uciAsGiven` is the raw token Lichess sent, and our own moves are recorded exactly as we sent them. The normalized form is internal and never replaces the raw one in journals, records or the protocol log. PGN uses SAN (`O-O`), so it is unaffected. **[LIVE]** which form arrives, and which outgoing form Lichess accepts (send `e1g1` unless live games show otherwise). **[T]** both forms, and raw preserved |
| E2 | A move list is never empty in practice | `moves` is `""` at game start; `"".split(" ")` yields `[""]` in some splits | Reducer treats an empty or whitespace string as zero moves. **[T]** |
| E3 | Positions always start from the standard start | `initialFen` may be `"startpos"` **or** a full FEN that happens to equal the start position | Normalize: a FEN equal to the start position (ignoring move counters) counts as startpos. Any other FEN is declined (§7). If a game with one starts anyway (E20), abort or resign. **[T]** |
| E4 | Incremental `applyMove` only, never undo | Takebacks **shrink** the move list. A reconnect's `gameFull` can differ from local state | State is a pure function of `(initialFen, moves)`. Extend incrementally when the new list has the old one as a prefix; otherwise rebuild in full. Encoder history is then always identical to a replay from start, which matches self-play. **[T]** |
| E5 | Local state is truth | The server is truth. A local bug or a missed line could diverge | After every `gameState`, compare the server list to the local list. On a mismatch, rebuild, log a `divergence` anomaly, and **never** post a move computed from a diverged state. **[T]** |
| E6 | Promotions formatted lowercase; `parseUCI` accepts either case | Sends lowercase | Fine. Covered by the round-trip tests. **[T]** |
| E7 | Two non-authoritative UCI helpers exist (`LichessProbeData.parseUCI`, `MoveGenerator.uciString` with `=queen`) | — | **[D]** The bot uses only `ChessMove.parseUCI` / `ChessMove.uci`. A test asserts that the bot's outgoing strings round-trip through `parseUCI`. |

### 20.2 Game-ending rules

| # | DCM normally | Lichess | Handling |
|---|---|---|---|
| E8 | `ChessGameEngine` **auto-ends** at threefold, halfmove clock ≥ 100, and insufficient material | **Source-confirmed (§4):** bot games end automatically at threefold (lila `autoThreefold` claims for any BOT player), 50 moves, fivefold, and insufficient material (scalachess `autoDraw`). Neither side can play on | The kinds of condition match. The engine still runs in `.serverAuthoritative` mode (§8.2), so a *definition* difference (E10, E11) can never make DCM stop moving while the server says `started`. **[LIVE]** a fixture of one threefold ending, to confirm the source reading. **[T]** engine modes |
| E9 | Self-play never produces a position after a threefold or past 50 moves | **Also true on Lichess for bot games**, because they end automatically (E8) | No out-of-distribution exposure from continued play, so no claim policy is needed. **Residual:** because of E11, DCM's repetition planes can differ from what a rules-correct key would show in rare en-passant-square cases. The same thing happens in self-play, so it is consistent with training. **[D]** |
| E10 | DCM's insufficient-material rule: K v K, K+minor v K, all bishops on one colour with no knights | scalachess `InsufficientMatingMaterial`, plus the flag-against-insufficient-material case (`insufficientMaterialClaim` → draw, not loss) | The server status decides the result. The record stores the server status **and** `localDrawCondition` (§8.2). Any disagreement in either direction goes to the log with the FEN, for later comparison of the two rule sets. **[T]** mapping and disagreement logging |
| E11 | `PositionKey` includes the en-passant square even when no en-passant capture is possible, so DCM can under-detect a threefold | Lichess uses the rules-correct key | Known and documented. It only affects disagreement logging (E10); Lichess ends the game on its own key. **Not changed**, because self-play encoding depends on it and changing it would move the training distribution. **[D]** |
| E12 | Checkmate or stalemate: the self-play driver stops asking for moves | After the opponent's move leaves us with no legal moves, the server status flips to `mate`/`stalemate` | Never call `MoveSampler` with zero legal moves. Wait for the server status. **[T]** |
| E13 | One `GameResult` enum | Status `draw` covers agreement, claimed threefold, claimed 50-move and more. `timeout` (opponent left, claimed) ≠ `outoftime` (flag). Also `cheat`, `unknownFinish`, `variantEnd` | Store the raw server status plus a derived detailed reason from the final position and the event trail (agreement, claim, flag and so on). Termination stats use both. **[T]** |
| E14 | Every self-play game counts | `aborted` and `noStart` games are not real games | Recorded, but **excluded** from W/D/L, performance and calibration by default (filter toggle). **[T]** |

### 20.3 Turn, time and clocks

| # | DCM normally | Lichess | Handling |
|---|---|---|---|
| E15 | Moves have no deadline | The first move has a **first-move deadline** (`expiration.millisToMove`) before an abort. Every move is on a clock | The model slot must be **ready before accepting**: pre-warm at Go Online and on every source change. Never build a network lazily at first move (a network build takes seconds). **[T]** readiness gate on accept |
| E16 | Inference latency ≈ milliseconds | Same GPU as a live training step, which uses about 1.1–1.4 s of GPU time per step (`gpu=` in `[STATS]`). A small bot inference may queue behind it | **[LIVE]** measure bot inference p50/p99 **with training running**. The record stores per-move latency split (encode, inference, sample, POST). If p99 threatens the minimum TC, raise the default minimum TC. Optional knob (off by default): "yield training while it's our move in a game with clock < X s". **[T]** latency accounting |
| E17 | — | Challenge `timeControl.limit/increment` are in **seconds**, while `gameFull.clock` and `gameState` times are in **milliseconds** | Typed units in the model layer (`Seconds`, `Milliseconds` wrapper types). Never pass raw `Int`s across. **[T]** |
| E18 | — | Our clock can get low while the gate is busy (housekeeping or chat) | When **any** game has our clock below `lowClockThresholdMs` (default 10 000), suspend all non-move traffic (chat, challenge responses, housekeeping) until it recovers. **[T]** |
| E19 | Wall-clock timestamps | Server clocks are snapshots at the last move. Local wall time can jump (NTP, DST) | Timers and latency use `ContinuousClock` (monotonic). The wall clock is used only for record stamps. Displayed clocks tick locally from the last server snapshot. **[D]** |

### 20.4 Protocol and stream semantics

| # | DCM normally | Lichess | Handling |
|---|---|---|---|
| E20 | Every game is one DCM chose to start | `gameStart` can arrive for games the policy never approved: accepted from lichess.org while logged in as the bot, a leftover from before a restart, or a game from a changed policy | Every `gameStart` creates a session. If it would fail the policy **for playability** (variant, from-position, correspondence), abort if allowed (before our 2nd move), otherwise resign, and log it. A game that is merely out of policy (rated, speed) is **played**, never thrown. **[T]** |
| E21 | — | Our move POST can race the opponent resigning, flagging or aborting, and returns 400 "game over" | That 400 is benign when the next state shows a finished game. It is logged as info, not as an anomaly. **[T]** |
| E22 | — | Resignation may not produce a final `gameState` (api #60) | Game end also comes from `gameFinish` and the export (§6). **[T]** |
| E23 | — | The export of a just-finished game can lag and briefly still show `started` | Reconciliation retries with backoff until a terminal status appears, up to a bounded window, then marks the game `unreconciled` for a later retry. **[LIVE]** measure the lag. **[T]** |
| E24 | — | The opponent can be Lichess's own AI (`aiLevel` set, no rating) | Opponent kind `lichessAI`, a separate Stats slice. Nil ratings are handled everywhere. **[T]** |
| E25 | — | Player `id` is lowercase; `name` has display casing | Identity compares on `id` only (our id comes from `/api/account`). **[T]** |
| E26 | — | `gameStart.game.fullId` embeds a player-specific secret suffix | Store and log only `gameId`. `fullId` never enters records, PGN or logs. **[D]** |
| E27 | — | `Retry-After` may be delta-seconds **or** an HTTP-date. Error bodies may be HTML from a proxy (502/503) | The parser handles both forms (the 60 s floor still applies). Non-JSON bodies are logged truncated and never decoded as JSON. **[T]** |
| E28 | — | New enum values and fields appear over time | Tolerant decoding: `unknown(raw)` cases, logged once per distinct value. Never crash, never drop a game. **[T]** |
| E29 | — | `wdraw`/`bdraw`/`wtakeback`/`btakeback` are *omitted* when false, not sent as `false` | Missing means false. **[T]** |
| E30 | — | A takeback, if accepted, shrinks the list. Our "already posted for ply p" guard must follow it | The guard resets for every ply ≥ the new length, and taken-back moves are kept in the journal as `retracted`. **[T]** |
| E31 | — | An opponent draw offer stays pending until a move. Our `offeringDraw=true` while they offer is an **agreement** | `offeringDraw` is set only when the policy decided to accept or offer, never as a side effect. **[T]** |

### 20.5 Networking stack (URLSession)

| # | Risk | Handling |
|---|---|---|
| E32 | **Connection starvation:** over HTTP/1.1, URLSession caps connections per host. Long-lived streams (1 event + N games) can occupy them all, and move POSTs then **queue behind the streams** | Two `URLSession`s: **streams** (`httpMaximumConnectionsPerHost` raised) and **requests** (a separate session and pool), so streams can never starve moves. **[LIVE]** log `URLSessionTaskTransactionMetrics.networkProtocolName` to see HTTP/1.1 vs h2. **[T]** a stub test with a saturated stream pool |
| E33 | `timeoutIntervalForRequest` (default 60 s) is an **idle** timeout. A quiet game stream (the opponent thinking in rapid or classical, if the game stream has no keep-alives) times out and errors | Stream session idle timeout is set very high. Liveness is owned by our own watchdog (§6), which knows whose turn it is. `timeoutIntervalForResource` is raised too (streams can outlive the default). **[T]** |
| E34 | Sleep/wake, network change, VPN toggle and ISP IP rotation silently kill connections | Watchdog (§6), plus: on `NSWorkspace.didWakeNotification`, tear down and reconnect **immediately** (skip backoff), then resync every game. **[T]** simulated wake |

### 20.6 macOS process environment

| # | DCM normally | Risk | Handling |
|---|---|---|---|
| E35 | The app holds **no** App Nap or sleep assertion (verified: no `beginActivity` or IOPM use anywhere) | App Nap throttles timers and the network when the window is hidden. **Idle system sleep mid-game loses every game on time** | While the bot is Online or has games: `ProcessInfo.beginActivity(options: [.userInitiated, .idleSystemSleepDisabled], reason:)`, ended when Offline. A forced sleep (lid close) can't be prevented; E34 recovers, and the Overview warns "on battery / lid-close will forfeit games". **[T]** assertion held iff online |
| E36 | — | The Keychain item is unreadable while the screen is locked (default accessibility), so an overnight reconnect fails | `AfterFirstUnlock`, and the token is held in memory once online (§12.2). **[T]** |
| E37 | Several DCM processes are common (GUI, `--train`, `--uci`, CLI tools) | A second process opening the event stream **kicks the first off**, because only one stream is allowed per token | **[D]** The bot never starts in any CLI mode (only the GUI path constructs `LichessBotController`). Auto-connect is off by default. Instance protection is layered; see §6.1 and E54–E55. **[T]** CLI entry points never instantiate the bot |
| E38 | `SessionLogger.log` never fsyncs (doc says 0.5 s idle flush, code doesn't schedule it). Recent lines can be lost on a crash | Bot forensics need durability | Journals and the protocol log synchronize at the end of each move cycle and on every anomaly. SessionLogger remains best-effort, and bot-critical facts never live **only** there. **[D]** |
| E39 | SIGUSR1, SIGHUP and SIGUSR2 end in `_exit` | Bot state is not flushed | The network-free pre-exit hook (§13). Launch recovery (§10.2). **[T]** recovery from `abandoned-at-exit` |
| E54 | One bot per process was assumed everywhere | A second local DCM process goes Online | `flock` on `LichessBot/bot.lock` (§6.1 A). **[T]** a second controller against the same data dir is refused, and the lock is released when the holder dies |
| E55 | The bot assumes it is the only thing moving for this account | Another machine with a *different* token plays the same games | Foreign-move detection (§6.1 B2): stop moving, go Offline, alarm. **[T]** a stub injects an our-side move we never sent |

### 20.7 Model and session lifecycle

| # | Situation | Handling |
|---|---|---|
| E40 | The operator loads another session, builds a new network, or changes architecture while Online with a Champion/Trainer source | In-progress games keep their generation (an independent network copy). The next game snapshots the new source, building a network of the **new** architecture via the shared helper. A lineage change is logged. **[T]** |
| E41 | A promotion happens mid-game (Champion source) | That game finishes on its generation, and the next game picks up the new champion (ModelID key changed). **[T]** |
| E42 | The trainer network is training-mode (BN) and its working weights can lag the fp32 masters by one step | Snapshot via the **same** export → inference-mode `loadWeights` path human play already uses. Never evaluate on `trainer.network` directly, and never include velocity. **[D]** |
| E43 | Live trainer with `midGameRefresh` on changes the model mid-game | Per-move generation id in the record (§10.6). Stats can exclude mixed-generation games. **[T]** |
| E44 | Trainer selected but Play-and-Train stopped: `trainerAvailable` is still true (`session.trainer != nil`, a leftover) | That is valid: the snapshot is of a static trainer. The UI labels it "trainer (not training)" so a frozen trainer isn't mistaken for a live one. **[T]** |

### 20.8 Play quality

| # | Situation | Handling |
|---|---|---|
| E45 | With τ near argmax (0.01), DCM is nearly deterministic. A human (or bot) that beats it once can **replay the same winning line** | The opening τ schedule varies early moves (§12.4). Optional per-game τ jitter. Stats flag repeated identical games vs the same opponent (same move-list hash). Rotation/A-B (Phase 8) also breaks repetition. |
| E46 | Two near-deterministic bots can play the same game over and over | Per-opponent daily cap (default 5 since commit `c5542b8`; was 20) (Lichess enforces its own 100/day bot-pair limit; there is no local cap, by owner decision 2026-10-01). Repeated-game detection (E45) can auto-decline that opponent for the day. **[T]** |
| E47 | Human openings produce positions self-play never reaches | Expected, and not a protocol bug. Stats by opening (from the export's `opening`) show where DCM struggles. That is useful learning signal, never training data (§3). |
| E48 | `SamplingSchedule` τ decays by game-total ply | Lichess games start from startpos (from-position games are declined), so `ply = moves.count`. That matches self-play. **[T]** |

### 20.9 Stats and data integrity

| # | Situation | Handling |
|---|---|---|
| E49 | **Rating refunds:** when a cheater is banned, Lichess later adjusts ratings, so the stored per-game `ratingDiff` goes stale | The rating-over-time chart uses `GET /api/user/{id}/rating-history` (housekeeping, **once per day at most**, one request). Per-game diffs are kept as "at the time", and the two are labeled distinctly. |
| E50 | Day boundaries | Filenames and stamps use UTC (the existing `CheckpointPaths` formatter). Stats' "day/week" grouping uses the **local** time zone, labeled as such. **[D]** |
| E51 | Reconciliation mismatch (local vs export) | The export wins for result, status, ratings and opening. The mismatch stays visible (⚠ in Games), is counted in Stats, and never silently disappears. **[T]** |

### 20.10 Chat

| # | Situation | Handling |
|---|---|---|
| E52 | Lichess limits chat message length (believed to be 140 characters) | Templates are validated against the limit **after** placeholder expansion, and a message is never truncated mid-word silently: an over-limit template is rejected in Settings. **[LIVE]** confirm the limit. **[T]** |
| E53 | Opponent chat is arbitrary untrusted text | Stored verbatim and displayed as plain text. *(Original decision: no chat commands at all, so no opponent could trigger requests or actions through chat, §5.3. Superseded 2026-09-28 by §12.5a.)* The only behavior chat can trigger is a §12.5a command reply: a fixed, read-only text within a per-game budget (a cooldown, a cap, and no replies when our clock is low). Nothing else an opponent types changes any state. **[D]** **[T]** |

### 20.11 Live verification checklist (consolidated [LIVE] items; run in Phase 5)

1. The castling form in `moves` for standard games, and whether `e1g1` is
   accepted (E1).
2. Whether the game stream sends keep-alives, and at what interval (§6,
   E33).
3. Confirm from one real game that a threefold ends the game
   automatically (E8). The lila and scalachess source already answer this,
   so this is a cheap confirmation, not a blocker.
4. Export lag after a game ends (E23).
5. The HTTP protocol version negotiated by URLSession (E32). The protocol
   layer logs `networkProtocolName` from the first request (Phase 2). The
   two-session design is safe either way.
6. The chat length limit (E52).
7. The event stream and challenges with a `bot:play` + `challenge:write`
   token (§4, §7.1).
8. Bot inference latency with training running (E16).
