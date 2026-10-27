# Lichess bot: circuit breakers that never abandon a game (owner decisions 2026-10-26)

## Owner decisions

1. **Cardinal rule: no automatic action ever abandons a game in progress.** A breaker stops *new* games; games in progress keep playing, within whatever request limits are in force, until they end. Abandoned games count against the account. The only automatic stops that still end play are the ones that can't play anyway: a rejected token (401/403) and an event-stream takeover (another client owns the token). The operator's own "Go Offline without finishing" stays as it is (an explicit choice, not automatic).
2. **429:** respect the wait Lichess gives — `Retry-After`, or the body's `ratelimit.seconds` — never less than 60 s, plus a configurable margin. Games in progress continue once each cooldown ends.
3. **Repeated 429s** (2 within 60 min) no longer close the gate and stop the bot. They put the bot in a **new-games hold**: no incoming challenges accepted (declined `later`), no outgoing challenges or matchmaking, for 2 hours, then it resumes by itself. Games in progress finish, respecting every cooldown.
4. **Lost on time:** 3 of the last 30 games → full stop without auto-resume: no new games; once the games in progress finish, the bot goes offline in Error until the operator goes online again.
5. **Move-POST failures** (5 within 10 min): alert and hold new games; moves in games in progress keep being posted and retried. Resumes after 30 min.
6. **Reconnect storm** (10 event-stream reconnects within 10 min): alert and hold new games; resumes after 30 min.
7. **Each breaker is configurable:** its threshold, its window, and whether it resumes by itself and after how long. A breaker set not to resume drains: no new games, games in progress finish, then the bot goes offline in Error naming the breaker, until the operator goes online.
8. **Event-stream takeover** keeps today's handling (three immediate server closes → Error, no reconnect).

## What exists (code map, 2026-10-26)

- `LichessBotRequestGate.beginCooldown`: cooldown = `max(Retry-After, 60 s)` (`LichessBotRateLimit.cooldown`); a 2nd 429 inside `breakerWindow` emits `.breakerTripped` and **closes the gate**. The controller answers `.breakerTripped` / `.closed` with `failRuntime`, whose `tearDownRuntime` calls `manager.abandonAllSessions()`: **every live game is abandoned.**
- The 429 body is read only for DCM's daily bot-game limit (`LichessBotOwnBotGameLimitRefusal.parse`); an ordinary 429's `ratelimit.seconds` is ignored.
- One 429 starts `rateLimitHoldUntil` (controller, `postRateLimitDrainMinutes` from the 429, not from the cooldown's end) and `manager.setRateLimitHold(true)`. That hold already blocks incoming accepts (`isAcceptingNewGames`), outgoing sends (`outgoingSendBlockedReason`), the challenge queue and automatic matchmaking (`PassConditions.rateLimitHoldActive`). Starting a second hold *replaces* the end time (a shorter one could cut a longer one short).
- A game session's `.stoppedMoving` (3 refusals at one ply, a client fault, 8 consecutive disagreements) → `failRuntime` → every other live game abandoned.
- `lostOnTimeBreakerCount`, `lostOnTimeBreakerWindowGames`, `movePostFailureBreakerCount`, `reconnectStormBreakerCount`, `breakerWindowMinutes` are declared and validated but read nowhere; no UI.
- Settings load by overlaying the saved JSON onto the defaults, nested objects key by key (`LichessBotSettingsStore.loadReporting`), so settings must be plain fields (no enums with associated values, which would merge into two keys).

## Design

### A. New-games holds (one mechanism for every breaker)

`LichessBotNewGameHolds` (pure value, `Play/LichessBotNewGameHolds.swift`) replaces `rateLimitHoldUntil`. One entry per `LichessBotNewGameHoldReason`:

- `rateLimit` — one 429 (`postRateLimitDrainMinutes` after its cooldown ends)
- `rateLimitBreaker`, `lostOnTime`, `movePostFailures`, `reconnectStorm` — the breakers
- `gameStoppedMoving` — a game session stopped moving (see D)

Each entry ends `.at(Date)` or `.whenOperatorGoesOnline`. Holding the same reason again keeps the later end (`.whenOperatorGoesOnline` outlasts any date), so a short hold never cuts a longer one short. `expire(now:)` drops dated entries that are over and returns them; `isHolding`; `summary` (for the UI and `outgoingSendBlockedReason`); `requiresOperator` (any `.whenOperatorGoesOnline` entry).

The controller is the single owner:
- Any hold → `manager.setNewGameHold(true)` (renamed from `setRateLimitHold`; same flag, same effect on `isAcceptingNewGames`), connection `.online` → `.draining`, outgoing sends, the queue and matchmaking blocked (each existing `rateLimitHoldUntil != nil` check becomes `newGameHolds.isHolding`).
- A hold that requires the operator also sets `drainRequested`, so the bot goes offline once the games in progress finish; `fileRecordsThenStop` then leaves the connection in `.error("<reason>; go online to resume")` instead of `.offline`.
- The poll loop (where `rateLimitHoldUntil` expired) calls `expire`; when nothing is held any more, the manager accepts again and the state returns to `.online` (unless the operator drained), as `endRateLimitHold` does today.
- Going online clears the holds that require the operator (the operator's explicit resume) and keeps dated ones still running (as the 429 hold outlives offline/online today).
- Every hold start and end is logged: `[ALARM] LICHESS-BOT breaker <reason>: … holding new games until <time | the operator goes online>` and `[LICHESS-BOT] new-games hold <reason> over`.

### B. 429s

- `LichessBotTooManyRequestsCause.requestRate` gains the body's wait: `case requestRate(bodyWait: Duration?)`. The API client parses `ratelimit.seconds` from any 429 body (`LichessBotRateLimit.bodyWait(from:)`, beside `parseRetryAfter`); a stream open (no body read) passes nil.
- `LichessBotRateLimit.cooldown(retryAfter:bodyWait:margin:)` = `max(retryAfter, bodyWait, 60 s) + margin`. The one cooldown rule.
- The gate takes a `LichessBotRateLimitPolicy` (`breakerCount`, `breakerWindow`, `cooldownMargin`) at init and through `setRateLimitPolicy` (replaces `setBreakerWindow`). At `breakerCount` 429s inside the window it emits `.breakerTripped` and **stays in cooldown** (never closes). Only a failed cooldown timer still closes it (it can't know when to send again; that `failRuntime` remains, and is the one remaining automatic stop besides 401/403 and takeover).
- Controller: `.rateLimited(cooldown:…)` → hold `rateLimit` until cooldown end + `postRateLimitDrainMinutes`; `.breakerTripped` → hold `rateLimitBreaker` per its resume setting (default 2 h). No `failRuntime`.
- New setting `connection.rateLimitCooldownMarginSeconds` (default 10).

### C. Breaker counters (pure, `Play/LichessBotBreakers.swift`)

- `LichessBotEventRateBreaker` (count within a time window): `note(at: Date) -> Bool` (tripped). Used for move-POST failures and reconnects. After a trip the count restarts, so a new trip needs a fresh run of events; each trip extends the hold.
- `LichessBotLostOnTimeBreaker` (count within the last N finished games): `noteFinished(gameID:lostOnTime:) -> Bool`; a game is counted once (`gameID` set). Counts games finished since the bot went online (it restarts at every go-online: the operator's resume is the reset).
- Lost on time = DCM's score 0 and status `outoftime`: one rule, `LichessBotGameEnding.isTimeForfeitLoss(status:ourScore:)`, also used by `LichessBotRecordStatistics`'s flagged count.

### D. Wiring each breaker

- **Move-POST failures:** a new game event `.movePostFailed(ply:error:)` emitted by `LichessBotGameSession.post` for a POST that got no answer or a 5xx (the transient branch only: a 4xx is Lichess disagreeing, handled by resync; a 429 is the rate-limit breaker's). Not journaled (`.moveRejected` beside it already is); the live game ignores it. The controller counts it.
- **Reconnect storm:** the controller counts `.eventStreamOpened` after the first of each runtime.
- **Lost on time:** the controller notes each game's `.finished` with the live game's `ourScore`.
- **A game that stopped moving** (`.stoppedMoving`): that game stops (as now); instead of `failRuntime`, hold `gameStoppedMoving` until the operator goes online: the other games finish, then Error. Not configurable (it signals a disagreement with Lichess about a game, plan §6.1 B, which needs a look).
- **Takeover, 401/403, failed cooldown timer:** unchanged (`failRuntime`).

### E. Settings

New section `breakers` (`LichessBotBreakerSettings`):

| Breaker | Fields (defaults) |
|---|---|
| `rateLimit` | `count` 2, `windowMinutes` 60, `resumesAutomatically` true, `resumeAfterMinutes` 120 |
| `lostOnTime` | `count` 3, `windowGames` 30, `resumesAutomatically` false, `resumeAfterMinutes` 120 |
| `movePostFailures` | `count` 5, `windowMinutes` 10, `resumesAutomatically` true, `resumeAfterMinutes` 30 |
| `reconnectStorm` | `count` 10, `windowMinutes` 10, `resumesAutomatically` true, `resumeAfterMinutes` 30 |

- A timed resume runs from the trip (each later trip of the same breaker extends it).
- Removed from `connection`: `rateLimitBreakerWindowMinutes` (carried forward into `breakers.rateLimit.windowMinutes` by `carryForwardReplacedFields`, so a saved value keeps its meaning), and the five never-read keys (dropped; reported as ignored saved keys at load).
- Added to `connection`: `rateLimitCooldownMarginSeconds` 10. `postRateLimitDrainMinutes` stays.
- Validation: counts ≥ 1; `windowGames` ≥ `count`; windows ≥ 1 min; the rate-limit window longer than the minimum cooldown (today's rule, moved); `resumeAfterMinutes` ≥ 1; margin 0…600 s.
- UI: a "Circuit breakers" section in the settings window: per breaker its count, window, "Resume by itself" toggle and "after … min" field; the margin and the post-429 hold in Connection. Overview shows each active hold (reason and end) under the connection controls.
- Live: `apply` pushes the rate-limit policy to the gate and the counters' thresholds to the controller's breakers on save.

### F. Decisions made in planning (owner may revisit)

- **Lost on time counts games finished since the bot went online**, not the filed history: the operator's go-online is the resume after a lost-on-time stop, and a history-seeded count would trip again on the first time loss after it.
- **Move-POST failures count only POSTs that got no answer or a 5xx.** A 4xx is Lichess disagreeing about the game (resync handles it; three at one ply stop that game), and a 429 is counted by the rate-limit breaker.
- **A game that stopped moving holds new games until the operator goes online** (not configurable): it means DCM and Lichess disagree about a game, which needs a look, but the other games still finish.
- **The post-429 hold runs from the end of the cooldown** (plan §5.4 said so; the code counted from the 429).
- **Cooldown margin default 10 s** ("respect the time Lichess tells us and sometimes add to it").

## Tests

Pure:
- `LichessBotNewGameHoldsTests`: later end wins; operator end outlasts dates; expire drops only finished dated holds; going online clears only operator holds; summary.
- `LichessBotBreakersTests`: event-rate trips at the count inside the window, not across it, restarts after a trip; lost-on-time counts within the last N games, counts a game once.
- `LichessBotRateLimitTests` additions: body `ratelimit.seconds` read (and absent, malformed); cooldown = max of the three + margin.

Existing tests touched only where the approved design changes an API or a behavior they assert:
- every `LichessBotRequestGate(time:breakerWindow:…)` construction takes `rateLimitPolicy:` (breaker count 2, the same window, margin zero, so their timings are unchanged);
- `.requestRate` becomes `.requestRate(bodyWait: nil)`;
- `setRateLimitHold` becomes `setNewGameHold`.

Gate (existing tests that assert the closing breaker change with the design — owner-approved behavior change):
- `testSecondRateLimitInsideTheWindowTripsTheBreaker` becomes: the breaker trips and the gate stays in cooldown, then sends again after it.
- New: a 429 body's `ratelimit.seconds` longer than `Retry-After` sets the cooldown; the margin is added.

Session / controller:
- Game session: a transient move-POST failure emits `.movePostFailed`; a 4xx does not.
- Manager: `setNewGameHold` declines new challenges `later` (existing hold tests keep passing under the rename).
- Settings: an old save with `connection.rateLimitBreakerWindowMinutes` 90 loads as `breakers.rateLimit.windowMinutes` 90; the never-read keys are reported ignored; defaults validate.

## Validation

1. Build: no errors, no new warnings.
2. Full test suite passes (gate tests updated for the approved behavior change only).
3. /recheck, independent review.
4. Deploy: new frozen Release build; bot swapped per the handoff (drain, finish games, relaunch the new build with `--lichess-bot`, go online); the session log shows the new build online.
5. Live: the next 429 / reconnect storm shows the hold lines and no `Bot stopped:` alarm with games in progress.

## Implementation notes (2026-10-26)

- **The gate's breaker count starts over after a trip**, like the other breakers' (independent review): kept, every later 429 inside the window re-tripped and restarted the two-hour hold. Test `testTheBreakerCountStartsOverAfterATrip`.
- **The manager's hold flag is pushed by one function** (`pushNewGameHold(to:)`) that re-reads `newGameHolds` after each push, so a hold starting while another ends can't leave the manager accepting under a hold.
- **Every breaker restarts at go-online** (the time-window ones too), and settings saves reach the gate in order (chained pushes).
- **Validation bounds**: counts 1…1000, windows and resume times 1 min…1 week, lost-on-time window ≤ 1000 games; the rate-limit window must exceed the shortest cooldown including the margin.
- `stoppedMoving` and a repeated operator hold drain once.
- `LichessBotMatchmaking.PassConditions.rateLimitHoldActive` is now `newGameHoldActive` (any hold), and `setRateLimitHold` is `setNewGameHold`.
- Controller test `testALostOnTimeTripFinishesTheOtherGameThenStopsUntilTheOperatorGoesOnline` pins the cardinal rule end to end. No controller test drives a 429: the controller's gate runs on the real clock and a 429 costs a real minute.
