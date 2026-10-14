# Lichess bot: DCM's own daily bot-game limit is not a request-rate limit

## Incident (2026-10-14)

- 06:40:22Z and 06:55:24Z: matchmaking challenges to DarkOnWeakBot and RatKingBot got HTTP 429 with
  `{"error":"You played 100 games against other bots today, please wait before challenging another bot.","ratelimit":{"key":"bot.vsBot.day","seconds":N}}`.
- The request gate treated both as request-rate 429s: a 60 s cooldown and a 15-minute hold on new games after the first; the second, inside the 1-hour breaker window, tripped the breaker, closed the gate and took the bot offline (Error) for about 9 hours, human games included.

## What Lichess sends (lila sources, master, read 2026-10-14)

- `modules/bot/src/main/BotLimit.scala`: `RateLimit[UserId](100, 1.day, "bot.vsBot.day")`, hit for each player when an all-bot game starts (`StartGame`).
  - `challengeLimitError(orig, dest)`: the challenger at its limit → `Limited("bot.vsBot.day", "You played 100 games against other bots today, please wait before challenging another bot.", until)`; else the challenged bot at its limit → `OpponentLimit("<name> played 100 games … please wait until <ISO time> to challenge them.", until)`.
  - `acceptLimitError(opponent)` is the same check with the accepting bot as `orig`: accepting a bot's challenge while DCM is at its limit gets the same `Limited`.
- `app/controllers/Challenge.scala` `eitherBotLimitResponse`: `Limited` → `JsonLimited` = **HTTP 429**; `OpponentLimit` → **HTTP 400**. Used by `apiCreate` (challenge) and `apiAccept` (accept).
- `modules/web/src/main/CtrlErrors.scala` `Writes[Limited]` and `BotLimit` `Writes[OpponentLimit]`: both `{"error": msg, "ratelimit": {"key": "bot.vsBot.day", "seconds": until − now (whole seconds, ≥ 0)}}`. No `Retry-After` header.
- Request-rate 429s (`limit.challenge`, `limit.challengeUser`, …) answer `{"error":"Too many requests. Try again later."}`, no `ratelimit` object.

## Fix

1. **Classify the 429** (`LichessBotOwnBotGameLimitRefusal.parse(status:body:respondedAt:)`, `Play/LichessBotOwnBotGameLimit.swift`): own-limit iff status 429 and (`ratelimit.key == "bot.vsBot.day"` or `error` matches `You played <n> games against other bots today`). `seconds` is taken only with that key. Anything else stays a request-rate 429. The opponent-limit 400 carries the same key but is a 400 and keeps its existing handling (`LichessBotBotLimitRefusal`).
2. **The gate does not treat it as a rate limit.** `LichessBotRequestGate.perform` takes a classifier of a 429's response (`causeOf429`; default: request rate, which streams and other callers keep). An own-limit 429 starts no cooldown, does not count toward the breaker, emits `.ownBotGameLimitRefused(refusal, label, priority)` and is returned like any other non-2xx answer; the client throws `LichessBotAPIError.ownBotGameLimit(refusal)`.
3. **One record of "DCM at its own limit until T"**: the controller's `botLimitUntil` map (Lichess's bot-limit ends per bot account, persisted in `player-notes.json`) under DCM's own account id — DCM is a bot like the others there. Written only by the gate event handler (`recordOwnBotGameLimit`), which logs once per new end: `[LICHESS-BOT]` line + protocol `challenge` event.
4. **T** (decision):
   - `seconds` present: `respondedAt + seconds + 1 s`. lila floors both instants to whole seconds, so the true end is up to 1 s after `respondedAt + seconds`; the extra second keeps DCM from asking a moment early.
   - `seconds` absent (not seen from current lila; a format change): `respondedAt + 1 h` (`LichessBotOwnBotGameLimit.recheckInterval`), logged as a recheck time, not Lichess's. Not the replayed window's clear time: the replay has just been shown wrong (Task 2: lila restarts reset the count), so its clear time is no better than a guess. A recheck costs one refused challenge, which no longer affects the gate.
5. **While limited** (`now < T`):
   - `LichessBotBotGameWindow` treats Lichess's report as truth: the replayed entry becomes count 100 clearing at T; games after T replay on top (the first resets it, as in lila). Its `count(asOf:)` and the Challenge sheet's status line therefore show 100/100 until T.
   - `LichessBotBotGameBudget.block` returns `.lichessReportedLimit(until:)` first, for every sender, including the operator's own challenges (Lichess would refuse them). Matchmaking passes are blocked with "Bot games paused: Lichess daily limit until HH:MM"; the queue skips bot entries (humans still go); operator challenges to bots fail at once with the same text.
   - Incoming bot challenges are declined with `later` (`LichessBotChallengePolicy`, context field `ownBotGameLimitUntil` from the controller through a manager provider). Human challenges are unaffected.
   - The Overview's matchmaking line shows "Bot games paused: Lichess daily limit until HH:MM".
6. **Resume at T**: nothing to schedule. Every check compares with `now`; matchmaking's retry after a blocked pass (`retryAfterUnproductivePass`) runs again after T.
7. A relaunch keeps the limit (player notes); the notes' load-time merge keeps the later end.

## Tests

- Classification: own-limit 429 (full body; key only; message only), plain 429, opponent-limit 400, undecodable 429 body.
- Gate: an own-limit 429 starts no cooldown, emits no `rateLimited`/`breakerTripped`, the next request goes straight out; a plain 429 still cools down.
- Regression (fails on the old code): two own-limit 429 challenge refusals through the real client and gate within an hour; the breaker does not trip, the gate stays open, a later request succeeds.
- Window: the report overrides a lower replayed count until T; a game after T resets it.
- Budget: `.lichessReportedLimit` blocks matchmaking, queue and operator senders until T, none after.
- Matchmaking `passBlockedReason` skips while limited, resumes after T.
- Policy: incoming bot challenge declined `later` while limited; human challenge accepted; bot accepted after T.

## Task 2 finding (why DCM's estimate said ~67, Lichess 100)

See the investigation report in the hand-off; summary: DCM's records are complete and `LichessBotBotGameWindow` matches lila's `RateLimit`; lila keeps the count in an in-memory Caffeine cache that a lila restart empties. Replaying DCM's games with the cache emptied at the Lichess outages seen in the protocol log (HTTP 502 bursts 2026-10-07 06:26Z, 10-08 06:24Z, 10-11 06:26Z, 10-12 06:40Z and 07:12Z) reproduces Lichess exactly: count 100 reached by the game at 2026-10-14 06:40:02Z, limited until 08:25:33Z, and it also explains three bot games Lichess let start on 2026-10-07 that the restart-free replay said were impossible. Restarts make the estimate wrong in both directions, which is why Lichess's answer must override it (step 5).

## Follow-ups approved 2026-10-14

### Matchmaking peak window (`LichessBotMatchmakingSchedule`)

- Settings ▸ Matchmaking: `peakStartHour` (2), `peakEndHour` (12, exclusive; may wrap past midnight; equal hours = always peak), `offPeakRatePercent` (40). Rationale in `LichessBotMatchmakingSettings.peakStartHour`'s doc (refusal-for-opponent-cap rates 0–12% 02:00–10:00 vs 37–50% 16:00–02:00 CDT, caps clearing 01:20–03:00).
- One mechanism: the rate limiter's hourly cap becomes `round(maxChallengesPerHour × percent)` off-peak (`LichessBotMatchmakingRateLimiter.decision(now:rate:minimumSpacing:)`), read by every pass, Fill Open Slots and the casual resend; a 0% rate gives `.offPeakPaused`.
- Status line: "n/cap this hour (peak 10/h | off-peak 40%: 4/h)".
- Local hour from `Calendar.component(.hour:)`; window boundaries from `nextDate(…, .nextTime)` (DST-safe).

### Challenge Log rows for sends that created no challenge

- `outgoingNotCreated` gains optional `opponent: LichessBotChallengeParty?` (name, title, rating and provisional at the challenge's speed, from the online check plus matchmaking's pick / online-bots list / loaded profile). Older lines decode with nil; schema version unchanged (older builds ignore the key).
- Rows show the recorded party; without one, the id with the BOT badge when the recorded kind is bot.

## Review changes (2026-10-14)

- Off-peak cap: any rate above 0% sends at least one challenge an hour (`scaledCap`); only 0% stops off-peak sends. Found by `LichessBotCasualFallbackTests`, whose 1/h cap rounded to 0 off-peak.
- `testClientStillTreatsAPlain429AsARequestRate` reads the gate's events, not its phase: the virtual clock's sleep ends the cooldown at once.
- Fail-first: with the gate's own-limit branch disabled, `LichessBotOwnBotGameLimitRegressionTests` failed (`rateLimited(cooldown: 60.0 seconds)`); with it, all 993 Lichess bot tests pass.
