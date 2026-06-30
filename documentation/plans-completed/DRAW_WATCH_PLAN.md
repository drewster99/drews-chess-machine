# DRAW_WATCH_PLAN.md

> **Status: SHIPPED** (v1 2026-06-13 audit; extended through v2/v3 by 2026-06-23).
> `Training/DrawWatchTracker.swift` is live (`SessionController.drawWatchTracker` /
> `drawWatchSnapshot`). The shipped feature now exceeds this v1 plan — several "future
> non-goals" listed below have since landed:
> - **Game termination on flag** (the Stage-2 non-goal) shipped as a toggle: the
>   `drawWatchTerminateGames` `@TrainingParameter` (`Training/TrainingParameters.swift`)
>   gates early-stopping a flagged game as a draw.
> - **Tunable streak length:** the v1 compile-time `N = 8` constant was promoted to the
>   `drawWatchStreakLength` `@TrainingParameter` (`Training/TrainingParameters.swift`),
>   so the consecutive-ply window is now user-configurable.
> - **Histogram bucket width is 40 plies**, not the 20 plies documented below
>   (`DrawWatchTracker.histogramBucketWidthPlies = 40`; 10 buckets × 40 = 0–400 plies).
>
> The rest of this doc describes the v1 design as drafted; read the `N = 8`, "constant
> for now", "20-ply bucket", and "termination is a non-goal" passages as superseded by
> the above.

Stealth-mode per-game `pDraw` monitor during self-play. **No game termination in this phase — every game still plays out to its natural conclusion (mate / stalemate / 3-fold / 50-move / ply-cap). Flagging a game does NOT change the game's behavior in any way; the worker just records that the flag fired and the game continues playing exactly as before.** Termination policy is an explicit non-goal of this plan and would be a separate plan layered on top.

A key v1 metric is **the flag's predictive accuracy as a draw signal**: of the games we flag, what percentage actually go on to end in a draw (vs end decisively despite the network thinking they were drawn). This requires the game to finish, which is one more reason termination stays off in this phase.

Author handoff note: this plan is for an implementer. Read it end to end before touching code; the per-file edits at the bottom assume the design decisions above are settled. Wait for explicit "start" before implementing.

## Goal

For each self-play game, watch the network's W/D/L distribution per ply. When `pDraw ≥ 0.95` holds for `N` consecutive plies on the same game, raise an in-memory flag describing where in the game it occurred. Surface those flags through a new chart tile showing the by-ply-bucket histogram and per-game / per-ply percentages.

The user-facing question this answers: *"Are we producing games where the network has effectively decided the position is drawn long before the game-length cap, and if so where (early / mid / endgame) and how often?"*

## Detection rule

- Source value: `pDraw = softmax(valueLogits)[draw_slot]`, where `draw_slot = 1` (per the W/D/L slot order `[win, draw, loss]` documented in `Network/ChessNetwork.swift` and `wdl-value-head.md`).
- Threshold: `pDraw ≥ 0.95` (constant for now; promote to `@TrainingParameter` only if the user asks).
- Window: `N = 8` consecutive plies (constant for now; same caveat).
- STM independence: `pDraw` for a position does not depend on whose turn it is to move (the WDL distribution describes the same position regardless of STM), so the consecutive-plies counter is a single counter per game, not split by side.
- Reset on dip: any ply with `pDraw < 0.95` resets the streak counter to 0.
- Flag fires on threshold crossing: when the streak counter transitions from `7 → 8`, emit one flag. The streak counter continues to increment past 8 but no additional flag fires until the streak first drops below the threshold and then re-reaches 8 (so a single sustained-draw stretch produces exactly one flag).
- Flag payload (per fire):
  - `workerId: UInt16`
  - `intraWorkerGameIndex: UInt32`
  - `plyIndex: UInt16` — 0-indexed; the ply at which the streak hit 8 (i.e. the 8th ply of the streak)
  - `streakStartPly: UInt16` — `plyIndex - 7`
  - `pDrawAtFire: Float` — `pDraw` value at the firing ply (always `≥ 0.95`)
  - `timestamp: Date`

## Data path — extend the batched forward pass

The self-play hot path goes through `ChessMPSNetwork.evaluateBatched(batchBoardsPointer:floatCount:count:consume:)` (the pointer-flavored overload at `ChessMPSNetwork.swift:258`). The current `consume` closure receives `(policy, values)`:

```swift
consume: @Sendable @escaping (
    UnsafeBufferPointer<Float>,   // policy: count * policySize logits
    UnsafeBufferPointer<Float>    // values: count scalars (p_win - p_loss)
) -> Void
```

The W/D/L probabilities are already computed inside the graph (per `ChessNetwork.valueProbs` exposed for the trainer). Per-ply pDraw on the inference path currently requires a second forward pass via `evaluateValueDistribution(board:)` — too expensive at 4000 workers × every ply.

**The change**: extend the batched `consume` signature to also deliver the W/D/L slot per position. The proposed contract:

```swift
consume: @Sendable @escaping (
    UnsafeBufferPointer<Float>,   // policy:    count * policySize logits
    UnsafeBufferPointer<Float>,   // values:    count scalars (p_win - p_loss)
    UnsafeBufferPointer<Float>    // wdlProbs:  count * 3 floats in slot order [win, draw, loss]
) -> Void
```

Cost analysis: `count × 3 floats` extra readback per batch. At `count = 4000`: 48 KB per forward pass, negligible vs the existing `count × 4864` policy readback. The slot probabilities are computed in-graph already; this just exposes them.

Both `evaluateBatched` overloads (`[Float]` and pointer) get the same signature change. `MoveEvaluationSource.evaluate(...)` — the unrelated non-batched path used by Play Game / Human-vs-Network — is **not touched** in this plan; that single-position path can stay scalar-only. (If a future feature wants WDL on the interactive path, that's an additive change to `MoveEvaluationSource`'s protocol.)

All existing call sites of `evaluateBatched` (arena's `TickTournamentDriver`, batched-eval correctness probe, anyone else) get the new closure argument and may either consume the new buffer or ignore it (Swift's closure-conformance for unused params is just `_ in`).

## Per-game streak tracking

Each self-play worker already holds a `ChessMachine` (the game state) plus per-game accumulators (boards, policy indices, ply indices, sampling taus, state hashes, material counts) that the batched driver hands to `ReplayBuffer.append(...)` on game completion. Add **three** more per-game scalars carried alongside those:

- `consecutivePliesAboveDrawThreshold: UInt16` — reset to 0 at game start, incremented on each ply where `pDraw ≥ 0.95`, reset on any ply below.
- `drawWatchFiredThisStreak: Bool` — set true when the streak first hits 8, cleared whenever the streak resets to 0. Prevents re-firing within the same sustained stretch.
- `drawWatchEverFiredThisGame: Bool` — set true on **any** flag fire during the game; cleared only at game start. Sticky across multiple flagged streaks within one game. Used at game completion to decide whether to record this game in the "flagged games" tally and to attribute its eventual outcome to flag precision.

These live on the worker / `BatchedSelfPlayDriver`'s per-slot state next to the existing per-game arrays. Lifetime is per game; zero at construction, drop on completion.

When the streak transitions `7 → 8`, the worker submits a flag to the session-wide `DrawWatchTracker` (see next section). Submission happens off-main; the tracker uses its own lock. **The game continues playing as normal** — no ply-loop short-circuit, no early-stop, no behavior change of any kind. The fire is observability only.

At game completion (the point where the worker hands the finished game to `ReplayBuffer.append(...)`), the worker also calls `tracker.recordGameCompleted(plyCount:, wasFlagged: drawWatchEverFiredThisGame, outcome:)` with the game's final outcome (`+1`, `0`, or `−1`). This is the bridge that lets the tracker compute flag→draw precision.

## Session-wide aggregation: `DrawWatchTracker`

New file: `DrewsChessMachine/DrewsChessMachine/Training/DrawWatchTracker.swift`.

- `final class @unchecked Sendable` with internal `OSAllocatedUnfairLock` (project standard — match `ParallelWorkerStatsBox` / `GameDiversityTracker` exactly).
- Holds:
  - `private var flags: [DrawWatchFlag] = []` — bounded ring (cap ~10_000 flags; old flags drop oldest-first on overflow so a long session doesn't grow unboundedly).
  - `private var totalGamesObserved: Int = 0` — incremented on every game-completion observation submitted by a worker (workers call `recordGameCompleted(...)` regardless of whether the game raised a flag).
  - `private var totalPliesObserved: Int = 0` — incremented by `plyCount` on each game-completion observation.
  - `private var totalPliesInFlaggedStreaks: Int = 0` — sum of streak lengths (`finalStreakLength`) at the end of each flagged streak; reported as a percentage of `totalPliesObserved`.
  - `private var flaggedGamesObserved: Int = 0` — number of completed games where `wasFlagged == true` was reported.
  - `private var flaggedGamesEndedInDraw: Int = 0` — subset of `flaggedGamesObserved` whose final outcome was `0`.
  - `private var flaggedGamesEndedInDecisive: Int = 0` — subset whose final outcome was `±1` (kept as a single counter; the win/loss split isn't useful here — we care about draw vs not-draw).
- API (all thread-safe):
  - `func recordFlag(_ flag: DrawWatchFlag)` — append + bound. Also extend `DrawWatchFlag` with `finalStreakLength` (filled on the streak-end observation, see below) so `totalPliesInFlaggedStreaks` can be incremented at streak break rather than fire time. Alternatively pass `finalStreakLength` to a separate `recordStreakEnded(length:)` call from the worker when the streak drops back below threshold or the game ends mid-streak — implementer's choice, but document which.
  - `func recordGameCompleted(plyCount: Int, wasFlagged: Bool, wasCapTerminated: Bool, outcome: Float)` — bumps `totalGamesObserved` + `totalPliesObserved`; when `wasFlagged`, bumps `flaggedGamesObserved`. **Only when `wasFlagged && !wasCapTerminated`**, bumps either `flaggedGamesEndedInDraw` (`|outcome| < 0.5`, matching `ReplayBuffer.append`'s draw test) or `flaggedGamesEndedInDecisive`. Cap-terminated games are excluded from the precision ratio — see locked decision #5.
  - `func snapshot() -> DrawWatchSnapshot` — Sendable struct: copy of flags array + the six counters + a precomputed ply-bucket histogram (10 buckets of equal width; bucket boundaries derived from session-observed max ply, or fixed 0–20, 20–40, ..., 180–200 — implementer's call, document it) + derived `flagDrawAccuracy: Double?` (`nil` until `flaggedGamesObserved > 0`, else `Double(flaggedGamesEndedInDraw) / Double(flaggedGamesObserved)`).
  - `func reset()` — for session-start.

The tracker is owned by `SessionController`. The heartbeat does NOT poll it — instead, the new chart tile uses the same `@Observable` + `chartCoordinator` pattern already used by `DiversityHistogramBar` (the heartbeat mirrors the snapshot into a published property only when the totals change, dirty-checked).

## Visualization: new chart tile

New file: `DrewsChessMachine/DrewsChessMachine/Views/Charts/TrainingChartGridView/DrawWatchHistogramChart.swift`. Place in the chart grid alongside the existing diversity histogram tile.

Four readouts:

1. **Bucket histogram**: x = ply bucket (0–20, 20–40, …, 180+); y = flag count. Each bar is the number of flag-fires whose `plyIndex` falls in that bucket across the session. Reuse the existing `DiversityHistogramBar` rendering style for visual consistency.
2. **% of games flagged**: `100 × flaggedGamesObserved / totalGamesObserved`. Single scalar in the header.
3. **% of plies in flagged streaks**: `100 × totalPliesInFlaggedStreaks / totalPliesObserved`. Single scalar in the header.
4. **Flag → draw precision**: `100 × flaggedGamesEndedInDraw / flaggedGamesObserved` — of games we flagged as "looks drawn", how many actually finished as draws. This is the v1 calibration metric for the rule: a precision near 100% says `pDraw ≥ 0.95 × 8 plies` is a reliable draw predictor; precision near the buffer's overall draw rate (currently ~10–13%) says the network's confident-draw calls have no predictive power yet. Single scalar in the header; shows `--` until the first flagged game completes.

Header line: `flags=N · games=A.A% · plies=B.B% · →draw=C.C%` (formatting matches the existing chart-header style in `SmallProgressRateChart.headerString`).

Tile size and placement: match the existing tile-grid layout (`SmallProgressRateChart` is the reference for height/chart-card chrome).

## Log emission

Two log channels, both via `SessionLogger`:

- `[DRAW-WATCH]` per-fire: `[DRAW-WATCH] flag worker=X game=Y ply=Z streakStart=W pDraw=0.97x` — one line per flag at fire time. Useful for grep; expected rate is low (only sustained-draw games fire).
- `[DRAW-WATCH] summary` periodic: piggyback on the existing 15-minute `[STATS]` cadence — append a one-line summary at the end of each stats interval: `[DRAW-WATCH] summary flags=N games=A.A% plies=B.B% →draw=C.C%`. Cheap snapshot read.
- `[DRAW-WATCH] outcome` per-flagged-game: at game completion, if `wasFlagged`, one line: `[DRAW-WATCH] outcome worker=X game=Y plies=Z flags=F outcome=draw|win|loss` so a grep can correlate fire-time pDraw against eventual result without parsing the chart.

## Concurrency / threading

- `DrawWatchTracker`: standard project pattern — `final class @unchecked Sendable` + internal `OSAllocatedUnfairLock`. No actor isolation. Workers call from off-main; the heartbeat reads via `asyncSnapshot()` (matching `ParallelWorkerStatsBox.asyncSnapshot()` shape — wrap the lock acquire in `DispatchQueue.global` + `CheckedContinuation` per the same off-main pattern just landed for `ReplayBuffer.asyncCompositionSnapshot()`).
- Per-game state lives on the worker / `BatchedSelfPlayDriver` slot; only the owning worker mutates it, so no cross-worker contention.
- The extended `consume` closure runs on the network's execution queue (same as today's `consume`). The per-position pDraw read is a single `wdlProbs[i*3 + 1]` lookup — sub-microsecond.

## Files touched (concrete list)

**Modified**:
- `Network/ChessNetwork.swift` — extend `evaluateBatched(batchBoards:count:consume:)` and `evaluateBatched(batchBoardsPointer:...:consume:)` to read back the W/D/L slot tensor and pass it as the third `consume` argument.
- `Network/ChessMPSNetwork.swift` — thread the new closure signature through both `evaluateBatched` overloads (the file is a thin wrapper; same change in two places).
- `Training/BatchedSelfPlayDriver.swift` — consume the new `wdlProbs` buffer; add per-slot streak counter + `drawWatchFiredThisStreak` flag; submit to `DrawWatchTracker` on streak transition `7 → 8`; submit `recordGameCompleted` on game end.
- `Arena/TickTournamentDriver.swift` — closure signature update (ignore the new arg; arena games are short and not the watch target).
- `Training/BatchedEvalCorrectnessProbe.swift` (if present — there's a `[BATCHER]` startup probe) — closure signature update.
- `App/SessionController.swift` — own a `var drawWatchTracker: DrawWatchTracker?` and instantiate alongside the other session-scoped boxes; pass it through to the self-play driver.
- `App/SessionController+Heartbeat.swift` — once-per-tick mirror of `tracker.asyncSnapshot()` into a `@State`-mirrored `DrawWatchSnapshot?` for the chart tile (dirty-checked on flag count).
- `App/SessionController+Training.swift` — emit the periodic `[DRAW-WATCH] summary` line inside the existing `[STATS]` block.
- `Charts/ChartCoordinator.swift` — add `currentDrawWatchSnapshot: DrawWatchSnapshot?` published property + setter (dirty-checked like `currentDiversityHistogramBars`).
- `Views/Charts/TrainingChartGridView/TrainingChartGridView.swift` — slot the new tile into the grid.

**New**:
- `Training/DrawWatchTracker.swift` — the lock-protected aggregator + `DrawWatchFlag` + `DrawWatchSnapshot` types.
- `Views/Charts/TrainingChartGridView/DrawWatchHistogramChart.swift` — the chart tile.
- `DrewsChessMachineTests/DrawWatchTrackerTests.swift` — unit tests (see below).

**NOT touched** (deliberately, scope-limited):
- `Network/MoveEvaluationSource.swift` — the single-position path. Interactive Play Game / Human-vs-Network don't need pDraw monitoring.
- `Training/ReplayBuffer.swift` — flags are observability, not replay state.
- `Persistence/SessionCheckpointFile.swift` — flags are session-scoped, not persisted across resume (start a fresh tracker on every session start).

## Tests

Add `DrawWatchTrackerTests.swift` with at least:

- `testStreakFiresAtEight` — feed a synthetic series `[0.99]*8 → 1` flag fired, payload correct.
- `testStreakResetOnDip` — feed `[0.99]*7, 0.5, [0.99]*8 → 1` flag, not 2.
- `testNoRefireWithinSameStreak` — feed `[0.99]*20 → 1` flag.
- `testSecondFlagAfterStreakBreaks` — feed `[0.99]*8, 0.5, [0.99]*8 → 2` flags.
- `testGameCompletionUpdatesCounters` — submit 10 game completions of varying plies; assert totals.
- `testRingBoundedAtCap` — submit 20_000 flags; assert resident is 10_000 and oldest dropped.
- `testHistogramBucketing` — submit flags at known plies; assert bucket counts match.

No new tests required for the `evaluateBatched` closure signature change — existing self-play / arena tests exercise the path; the third closure arg is consumed (or ignored via `_ in`) and the build is the proof.

## Validation (how to know it works)

1. Build green; 356/356 (or whatever the current count is) tests still pass; new `DrawWatchTrackerTests` pass.
2. Run a Play-and-Train session at the current `selfPlayMaxPliesPerGame=150` cap on the current champion (lineage `sMe9-1X+`, which holds `pD ≈ 0.50` on the trainer side but the *champion* may produce different per-game pDraws).
3. After ~15 min, expect:
   - `[DRAW-WATCH] flag …` lines appearing for at least some games (the current sp regime produces 60–70% draw_rate at arenas, so a non-trivial fraction of self-play games likely flag).
   - `[DRAW-WATCH] summary` lines emitted in the periodic stats.
   - Histogram chart populates with at least one non-empty bucket.
   - `% of games flagged` and `% of plies in flagged streaks` both > 0 and < 100.
4. Sanity-check the histogram shape against the user's intuition: draws-by-shuffle-marathons should cluster in mid-to-late-ply buckets; early flags would be more surprising and may warrant a tighter pDraw threshold or longer N.
5. No regression in tickMs (the heartbeat mirror is dirty-checked; histogram render uses the same `FastLineChart`/`FastBarChart` infrastructure as existing tiles).

## Locked design decisions (approved)

1. **Buckets**: fixed 20-ply width (`[0–20], [20–40], …`). Survives a future raise of `selfPlayMaxPliesPerGame` without re-bucketing logic.
2. **Per-game vs per-streak**: one game with two flagged streaks counts as 1 in "% games flagged" but 2 in the flag-count histogram. Confirmed.
3. **Reset**: only on session start (no cross-session persistence).
4. **Threshold + N**: compile-time constants for v1 (`pDraw ≥ 0.95`, `N = 8`). Promote to `@TrainingParameter` only when needed.
5. **Cap-terminated games (the precision-metric trap)**: **excluded from the precision denominator**. A game that hit the ply cap is counted in `totalGamesObserved`, in `flaggedGamesObserved` (if it flagged), and in the bucket histogram — but does NOT participate in `flaggedGamesEndedInDraw` / `flaggedGamesEndedInDecisive` and therefore does NOT participate in `flagDrawAccuracy`. Implementation needs a `wasCapTerminated: Bool` argument to `recordGameCompleted(...)`; the worker computes it as `plyCount >= TrainingParameters.shared.selfPlayMaxPliesPerGame` at completion time.
6. **Reset across promotion**: cumulative across promotions (no per-champion reset). The chart shows the session arc.
7. **Worker pDraw read path**: inline inside the `consume` closure on the network's executionQueue. The same closure already reads policy + values; one more `wdlProbs[slot*3 + 1]` lookup + a UInt16 increment + occasional `tracker.recordFlag(...)` call. The tracker's lock acquire is the only cross-thread thing and is trivially fast.
8. **Hover support on the new tile**: yes, hover-to-see-bucket-count enabled (match the existing `DiversityHistogramBar`-style tile behavior).
9. **[DRAW-WATCH] log format**: human-readable `key=value` lines (not JSON). Three line types as documented in the "Log emission" section.

## Future stages (explicit non-goals for THIS plan)

- **Stage 2**: actual game termination on flag. Likely implemented as an early-stop hook in the worker tick driver, with the game outcome set to `0` (draw). Will need a new param to gate it (`drawWatchTerminateOnFlag: Bool`). Independent plan.
- **Stage 3**: feed flag-density signals back into the training loop (e.g. as a buffer-side label or an alarm). Out of scope.
- **Stage 4**: per-streak `meanPDraw` or `minPDraw` payload for richer histograms. Trivial additive once the v1 pipeline exists.
