# Lichess bot: stuck games (watchdog, manual abort, victory claim)

Status (2026-10-12): owner-requested ("build the watchdog, the manual abort button, the automatic victory claim. we should probably do the abort some number of minutes past the full clock running down").

## The case that prompted it

Game 0qgNk2Ex (rated bullet 0+30 s, DrewsChessMachine vs SybAIBot): DCM played 1. e4 at 12:24; SybAIBot never moved. Lichess's own no-first-move abort never fired: five hours later its export still says `started`, and the bot's game stream carried only keep-alives. The bot mirrored it faithfully, held a game slot, and blocked a quit that waits for running games. A game where a side never made its first move is aborted, not lost (Lichess status `noStart`, unrated); a game Lichess leaves open with the opponent's clock spent can only be closed by a client action.

## What exists

- `opponentGone` stream events schedule `claimVictory` after `claimWinInSeconds` (+1 s) when `play.claimWhenOpponentGone` is on; a refused claim falls back to `claimDraw` once (`LichessBotGameSession.handleOpponentGone` / `claimAfterOpponentLeft`). Unchanged.
- `api.abort(gameID:)` is used only for unplayable games (`handleUnplayable`).
- Keep-alives reach `LichessBotGameSession.handle(_:settings:)` as `.keepAlive`.

## Design

### Stuck-game watchdog (in `LichessBotGameSession`)

- **When it fires:** the opponent is to move, the game is not finished, and the time since the opponent's turn began (the local time the state making it their turn arrived) exceeds the opponent's clock at that state plus `play.stuckGameGraceSeconds` (new setting, default 300 s, range 60…3600). "Full clock running down" is the opponent's remaining time as Lichess reported it when their turn began; the grace comes on top. Checked on every stream item (keep-alives included), so a silent stream still drives it; the session's injected clock (`time`) measures it, so tests control it.
- **What it does:**
  - **Abortable** (fewer than 2 plies played: a side hasn't made its first move): `api.abort`. Rating-neutral.
  - **Otherwise:** `api.claimVictory`. A refusal is logged and retried no sooner than one grace period later; never a draw claim, never a resignation (owner: resigning or leaving a game outstanding counts badly; a draw claim would give a won game away).
- **Once per turn:** an action, its outcome and any retry are logged as game events (`.action` / `.anomaly`) and appear in the session log; a new opponent move resets the watchdog.
- **Not in scope:** our own clock running out (Lichess ends that), games Lichess has finished (the stream reports them).

### Manual abort

- `LichessBotGameSession.abortNow()` (mirrors `resignNow()`): `api.abort`, logged `.action("aborted (operator)")`.
- `LichessBotController.abortGame(id:)` routes to the session through the session manager, like resign.
- **Abort** button on the game tile and the game detail view, shown only while the game is in progress and abortable (fewer than 2 plies), with a confirmation. Never offered once both sides have moved.

### Settings

- `LichessBotPlaySettings.stuckGameGraceSeconds: Int` (default 300), validated 60…3600, with a field in Settings ▸ Play beside "Claim when opponent gone". Persisted settings without it decode to the default (owner rule: saved settings survive changes; carry-forward overlay as for other added fields).

## Tests

- Watchdog (scripted fake stream + manual time):
  - a game stuck at ply 1 aborts exactly once after clock + grace, not before;
  - a game stuck at ply 10 with the opponent's clock spent claims victory once after clock + grace; a refusal retries no sooner than one grace later, never claims a draw or resigns;
  - an opponent move before the deadline resets it;
  - our own turn never triggers it;
  - a finished game never triggers it.
- Manual abort: `abortNow` posts abort and logs; the tile shows Abort only while abortable (render or view-model test).
- Settings: decode without the field gives 300; validation refuses 59 and 3601.

## Validation

- Touched test classes and the full suite pass.
- Live: after restarting the bot, game 0qgNk2Ex (still `started` on Lichess) is aborted by the watchdog within one grace period of going online; Lichess's export then reads `aborted`.

## As implemented (2026-10-12)

- **Watchdog** (`LichessBotGameSession`, `StuckOpponentTurn` + `trackStuckOpponentTurn` + `actOnStuckOpponentTurnIfDue`): `handleState` records the opponent's turn as (ply, position FEN, `time.now()` when the state arrived, the opponent's `wtime`/`btime` in that state). A later state for the same ply **and** position (a reopened stream's `gameFull`, a draw offer, a takeback proposal) keeps the original start and clock, so reconnects can't postpone the deadline; a new ply or a different position starts a new turn; our turn or a finished game clears it. After every stream item (keep-alives, lines, oversize/truncated reports) the session acts when `now >= startedAt + opponentClock + grace` (first attempt) or `now >= lastAttemptAt + grace` (later attempts), the grace read from the current settings.
- **Action, off the stream loop:** the check logs `.action("opponent still to move at ply N after S s, past their clock of C s plus the G s grace; aborting|claiming victory[ (attempt k)]")` and starts one tracked task (`stuckGameAction`; never two at once). The task re-checks, with no suspension before the request, that the game is live at the same ply and FEN, cancels any opponent-gone `pendingClaim` (otherwise a lost race leaves that claim refused while the game still looks live, and it falls back to a draw claim), then posts `LichessBotPlayPolicy.allowsAbort(pliesPlayed:)` (`< 2`, the one rule — `handleUnplayable` and `LichessBotLiveGame.isAbortable` use it too) ? `api.abort` : `api.claimVictory`. Outcome: `.action("aborted (stuck game)")` / `.action("claimed victory (stuck game)")`, `.anomaly("stuck game: abort|victory claim failed: …")`, or `.action("stuck game: … at ply N cancelled: …")`. A new opponent turn or our turn, `finish()`, and `run()`'s exit cancel it. Awaited inline, a request held in the account gate (cooldowns ~60 s) would have stopped the stream being read while DCM's clock ran.
- **Deviations / decisions:**
  - Any attempt that doesn't end the game — refused, failed (network, 5xx), or **accepted but the stream still says `started`** — is retried a grace period later (the plan named only refusals). Never a draw claim or a resignation.
  - The watchdog does not act after the session stopped moving (`movingStopped`: another client on the account, repeated rejections); that game is left to the operator, who now has Abort (while abortable) and Resign.
  - A state with an unknown status leaves the timed turn as it was.
- **Manual abort:** `LichessBotGameSession.abortNow()` → `LichessBotSessionManager.abort(gameID:)` (ended session or refusal → manager anomaly) → `LichessBotController.abortGame(id:)` (protocol log line; logged skip when the bot isn't running). `LichessBotAbortGameButton` (own file) on the tile header and the game view header, `.shown(game.isAbortable)`, with a confirmation dialog.
- **Setting:** `LichessBotPlaySettings.stuckGameGraceSeconds = 300`, range `LichessBotLimits.stuckGameGraceSecondsRange` (60…3600) in `validationProblems()`, field in Settings ▸ Play under "Claim when the opponent leaves". A save without it loads through the existing overlay (filled from the default, reported in `filledFromDefaults`); no carry-forward entry is needed since it replaces no field.
- **Tests:** `LichessBotStuckGameTests` (watchdog at ply 1 / ply 10 with refusal retry / reset by an opponent move / kept across a stream reopen with a lower clock / kept across a same-ply draw offer / our turn / a game finishing after the deadline / stream read while the request is held / pending opponent-gone claim cancelled; `abortNow`; `isAbortable`; settings decode and range). The shared fake server gained `setClockMilliseconds(white:black:)`, `sendKeepAlive()`, `holdVictoryClaims()` / `releaseHeldVictoryClaims()`, `opponentOffersDrawWithoutMoving()` and `finishGame(status:winner:)`.
