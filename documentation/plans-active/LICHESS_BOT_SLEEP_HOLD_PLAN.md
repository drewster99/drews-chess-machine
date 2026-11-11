# Lichess bot: no new games while the Mac is asleep (plan, 2026-11-10)

Status: **plan only, not started.** Implementation waits for the owner's "start".

## Problem

On 2026-11-09 and 2026-11-10 the bot lost three games on time, and the third tripped the lost-on-time breaker (14:07, Nov 10). In all three, the lid was closed on battery: macOS forces sleep then, whatever assertion the bot holds (`preventSleepWhileOnline` / `caffeinate -i`). The Mac then makes "maintenance" DarkWakes of about 5 s every few minutes. During those wakes the bot's timers run, so it keeps playing matchmaking:

| Time (Nov 10) | Power state (`pmset -g log`) | Bot |
|---|---|---|
| 12:19:57–12:20:02 | Clamshell sleep starts, on battery | — |
| 12:29:58 | DarkWake (12:29:57, 6 s) | matchmaking challenge sent to karl-engine-v1 |
| 12:38:30 | DarkWake (12:38:27, 5 s) | challenge sent to croco_little_bot, accepted, game XSyD9cqf started (30+0; its long clock survived the sleep, won by mate 13:11) |
| 12:43:34 | DarkWake (12:43:33, 8 s) | challenge sent to kbelludoo (2+1) |
| 12:47:37 | DarkWake (12:47:37, 5 s) | kbelludoo accepted → game GaX1h6Ty; another challenge sent at 12:47:38 |
| 12:58:49 | asleep | GaX1h6Ty lost on time at ply 24 |
| 13:43:11 → 14:07:51 | DarkWake 13:43:09–13:43:14, then asleep | 3hOyU7UN started in a DarkWake, lost on time at ply 14 → breaker |
| 14:41:10 | `DarkWake to FullWake` | — |

Nov 9's aayhCxxI (5+3, 12:40–13:03) fits the same pattern during the 11:46 lid-closed sleep.

**`NSWorkspace.didWakeNotification` fires only on a full wake.** The bot's protocol log (`LichessBot/Protocol/events-2026110{8,9,10}.jsonl`) has exactly one `system woke` (Nov 10 03:57:54Z = Nov 9 22:57:54 EST). That matches the one `DarkWake to FullWake` in the window while the observer was registered (Nov 9 22:57:53). None of the 426 DarkWakes while the observer was registered (bot online Nov 8 03:04 → offline Nov 10 14:16) posted it. So the existing notifications can tell "asleep, including its maintenance wakes" from "awake" without IOKit.

**Unverified assumption:** `NSWorkspace.willSleepNotification` is posted before a clamshell sleep on battery, as it is for other sleeps. The live check (Validation 4) confirms it; if it is not, the hold must come from IOKit's `IORegisterForSystemPower` (`kIOMessageSystemWillSleep`) instead.

## Owner rule this must keep

The cardinal rule (owner decision 2026-10-26, `LICHESS_BOT_CIRCUIT_BREAKERS_PLAN.md`): no automatic action abandons a game in progress. This plan only stops *new* games. A game already running when the Mac sleeps keeps its session, and it may still flag. Nothing here resigns or aborts it.

## Design

### A. A new hold reason: `systemAsleep`

- Add `LichessBotNewGameHoldReason.systemAsleep`, labelled "the Mac is asleep".
- Add `LichessBotNewGameHoldEnd.whenSystemWakes`. Only a full wake or the operator going online ends it.
- Because it goes through the one mechanism every breaker uses (`LichessBotNewGameHolds` → `manager.setNewGameHold`), its effect is the breakers' effect:
  - incoming challenges declined `later` (`isAcceptingNewGames`);
  - no outgoing sends;
  - the challenge queue doesn't pump;
  - matchmaking and Fill Open Slots pause (`PassConditions.newGameHoldActive`).
- `hold(_:until:)` ordering: `whenOperatorGoesOnline` stays the longest, then `whenSystemWakes` outlasts any date. A sleep hold never shortens or replaces another reason's hold; each reason has its own entry.
- `requiresOperator` stays false for it: the bot drains to `.draining`, not offline, and comes back by itself on wake.
- `expire(now:)` ignores it, since it has no date.
- New `systemWoke()` returns and removes `whenSystemWakes` entries.
- `operatorWentOnline()` also removes them. Going online is a user action, so the Mac is awake. This also covers a bot that went offline while asleep, whose wake observer was torn down with its runtime.

### B. Wiring (controller, beside the existing wake observer)

- Add an `NSWorkspace.willSleepNotification` observer to `Runtime`, next to `wakeObserver`, removed in `tearDownRuntime`. It calls `holdNewGames(.systemAsleep, until: .whenSystemWakes)`, then withdraws pending outgoing challenges (C).
- The existing `didWakeNotification` handler (`handleWake`) first ends the sleep hold, the way `releaseExpiredHolds` ends dated holds: log, push to the manager, `.draining` → `.online` when nothing else holds. Then it resyncs the sessions as now.
- Order matters: on a cold full wake the resync and the release both run. Releasing first lets matchmaking restart on the next poll, and the resync doesn't need the hold.
- Logs:
  - `[LICHESS-BOT] new-games hold: the Mac is asleep until it wakes`
  - `[LICHESS-BOT] new-games hold over: the Mac is asleep`
  - protocol-log lifecycle entries for both.
  - No `[ALARM]`: sleeping is normal, not a fault.

### C. Withdraw outgoing challenges on sleep

GaX1h6Ty was a challenge sent in one DarkWake (12:43:34) and accepted in a later one (12:47:37). Holding new sends isn't enough: an outstanding challenge can still become a game while the Mac sleeps.

- On `willSleep`, withdraw every pending outgoing challenge, using the existing fire-and-forget `withdraw(challengeID:client:reason:)` path that going offline uses.
- New `LichessBotWithdrawalReason.systemSleep`. It's a `Codable` case in the challenge log, so older builds can't read logs that contain it. That is the normal cost of a new case; no migration.
- Withdrawing a challenge is not abandoning a game: nothing has started.
- **Limit:** the system may suspend the process before the request leaves. Then Lichess's own challenge expiry or the next wake's state handles it. A challenge accepted anyway produces a game that the hold can't stop. It would be logged as now; the breaker remains the backstop.

### D. Not changed

- Games in progress at sleep: untouched (cardinal rule).
- `preventSleepWhileOnline` and its assertion: unchanged. They still prevent idle sleep on AC; they can't prevent clamshell sleep on battery.
- No new setting. Starting games while the Mac can't play them is never wanted, so this is always on.
- The lost-on-time breaker: unchanged.

## Edge cases

| Case | Expected |
|---|---|
| Lid closed, external display, on AC (clamshell mode, no sleep) | No `willSleep` → no hold; bot keeps playing |
| Idle sleep blocked by the assertion | No `willSleep` → no hold |
| Sleep, DarkWakes, full wake | Hold from `willSleep` through every DarkWake; released at `didWake` |
| `willSleep` then sleep cancelled (no `didWake`) | Hold stays. The UI shows "the Mac is asleep" and going online clears it. Safe: no games, never a lost game. Risk noted in Questions |
| Bot drains to offline while asleep (another breaker) | The sleep hold remains in `newGameHolds`; the next go-online clears it |
| Sleep hold active and a 429 hold or breaker | Both held; each ends by its own rule; accepting resumes only when none is left |
| `willSleep` while going online (runtime not built yet) | No observer yet. The going-online step needs a full wake, so it isn't reachable in practice. Going online already checks `newGameHolds.isHolding` |
| Full wake with games in progress | The resync runs as now; the hold release doesn't touch sessions |
| Two `willSleep`s without a wake | Idempotent: same reason, same end |

## Tests (test-authoring gate; regression first)

1. `LichessBotNewGameHoldsTests` (new test methods only; existing tests untouched):
   - a `whenSystemWakes` hold isn't ended by `expire(now:)` at any date;
   - `systemWoke()` ends exactly the sleep entries and returns them, leaving dated and operator holds;
   - `operatorWentOnline()` ends operator and sleep entries, leaving dated ones;
   - ordering: `whenOperatorGoesOnline` > `whenSystemWakes` > any `.at`, for holding the same reason again;
   - `requiresOperator` is false for a sleep-only hold;
   - `summary` names "the Mac is asleep until it wakes".
2. Matchmaking pass with `newGameHoldActive` is already covered (`LichessBotMatchmakingTests`). No duplicate.
3. **Regression (controller level):** write the hold-on-sleep test first and show it fails on the current code. The test must drive the controller through its real notification path: post `willSleepNotification` / `didWakeNotification` on an injected `NotificationCenter`. Today the controller hard-codes `NSWorkspace.shared.notificationCenter`; making the center an init parameter is a design seam (production passes the workspace center). If the owner would rather not add it, the controller-level behavior is validated by logs instead (step 4 below), and the gap is reported.

## Validation

1. Build clean through drews-xcode-mcp: no errors, no new warnings.
2. New tests and their sibling classes pass. Per owner rule (2026-11-05), the full suite runs only when the owner asks.
3. Bump and freeze a build; the owner relaunches the bot. The bot is never relaunched by me.
4. Live check: close the lid on battery with the bot online and idle. In the session log expect:
   - `new-games hold: the Mac is asleep` at sleep;
   - no `matchmaking challenge sent` and no `game … started` until the full wake;
   - `new-games hold over` at the full wake;
   - `pmset -g log` DarkWakes in between with no bot activity except stream keep-alives.
5. Success: zero games started between `willSleep` and `didWake` across at least one real lid-closed sleep. Failure: any game or challenge sent in that window.

## Questions (decided provisionally, for the owner)

1. Withdraw pending outgoing challenges at sleep (C)? Decided: yes. A challenge isn't a game; GaX1h6Ty shows the gap.
2. A setting to turn this off? Decided: no.
3. Release on `operatorWentOnline` as well as on wake? Decided: yes. It fixes the "bot offline during sleep" and "sleep cancelled" cases.
4. Injected `NotificationCenter` for the controller test (Tests 3)? Decided: yes, as a design seam. Without it, the controller wiring is tested only live.
