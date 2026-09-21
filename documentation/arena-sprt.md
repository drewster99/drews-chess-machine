# Arena promotion: SPRT mode — design

Status: **implemented, phases 1–5.** Written 2026-09-20.

All five phases have landed (`703dc77`, `26a747e`, `f21ff44`, `ea85025`, and
this one). Everything under "Where the changes land" now describes shipped
code. **The default is still the score threshold**, so an arena behaves exactly
as it did until someone picks SPRT in the Arena settings popover.

The one design question this plan left open and the implementation did not
settle is "Zero-variance records", below — found in phase 3, pinned by test,
not fixed.

## Goal

Add a selectable promotion criterion to the arena. Today promotion is a fixed
score threshold over a fixed number of games. Add Sequential Probability Ratio
Test (SPRT) as an alternative, chosen by a dropdown in the Arena settings
popover, with its parameters (`elo0`, `elo1`, `alpha`, `beta`) editable only
when SPRT is selected. Changes take effect on the next arena.

## What exists today

Verified in source, not docs:

- **The gate** (`SessionController+Arena.swift:258`):
  ```swift
  let shouldPromote = !aborted
      && playedGames >= totalGames
      && score >= TrainingParameters.shared.arenaPromoteThreshold
  ```
  Raw score `(W + 0.5·D)/N` against a flat threshold. No confidence interval,
  no significance test.

- **Elo is display-only.** `ArenaEloStats` computes score, Elo
  (`400·log₁₀(p/(1−p))`) and a Wald 95% CI using the empirical variance over
  the three outcome buckets — which is the statistically correct treatment of
  draws. Every call site is a formatter or a view. Nothing reads it to decide.

- **Colors are balanced.** `candIsWhite = (i % 2 == 0)` on initial fill, and
  recycled games alternate. Games are *not* paired (no shared opening between a
  colour-swapped pair) — there is no opening book.

- **Config is read at arena start**, not per tick
  (`SessionController+Arena.swift:611–612, 793`). Existing arena parameters are
  `liveTunable: false`, which is exactly the "takes effect next arena"
  semantics we want.

- **Early termination is already representable.** `TickTournamentDriver`
  returns `gamesPlayed = aWins + bWins + draws`, and its own doc notes
  "`TournamentGameRecord`s are NOT emitted for unfinished games — `gamesPlayed`
  will be less than the requested total."

## Constraint: the parameter system has no enum type

```swift
public enum ParameterType: String, Codable, Sendable {
    case bool
    case int
    case double
}
```

`ParameterValue` mirrors it. There is no string or enum case, and the type
flows through the macro, Codable, `CheckpointController`, the `.dcmsession`
saved-config block, and the parameter display in `UpperContentView`.

**Decision: do not extend `ParameterType`.** Store the mode as an `Int` and
expose it through a real Swift enum at every use site:

```swift
enum ArenaPromotionCriterion: Int, CaseIterable, Sendable {
    case scoreThreshold = 0
    case sprt = 1
    var displayName: String { ... }
}
```

The registry stores an `Int`; `TrainingParameters.arenaPromotionCriterion`
returns the enum. Use sites never see a raw integer or a string, so this does
not become a stringly-typed switch. Extending `ParameterType` is a much larger
change for no benefit at two cases, and can be done later if a third
persistence-visible enum ever appears.

## New parameters

All `category: "Arena"`, all `liveTunable: false` (read at arena start, so they
take effect on the next arena and cannot mutate mid-test).

| id | type | default | range | meaning |
|---|---|---|---|---|
| `arena_promotion_criterion` | Int | `0` | 0…1 | 0 = score threshold, 1 = SPRT |
| `arena_sprt_elo0` | Double | `0.0` | −50…50 | H₀: candidate is this many Elo better |
| `arena_sprt_elo1` | Double | `10.0` | −50…50 | H₁: candidate is this many Elo better |
| `arena_sprt_alpha` | Double | `0.05` | 0.001…0.5 | type I error (false promote) |
| `arena_sprt_beta` | Double | `0.05` | 0.001…0.5 | type II error (false reject) |
| `arena_sprt_min_games` | Int | `32` | 2…10000 | games before the test may fire |
| `arena_sprt_max_games` | Int | `20000` | 0…1000000 | runaway guard; `0` = unbounded |

`arena_games_per_tournament` is **not used in SPRT mode** and is disabled in the
UI when SPRT is selected. A sequential test decides when it has enough evidence;
fixing the sample size in advance discards the entire benefit. In SPRT mode the
tournament runs until the LLR crosses a bound.

That leaves a runaway case: if true strength sits between `elo0` and `elo1`, the
test can continue indefinitely. This matters operationally rather than
statistically — the arena competes with self-play and training for the GPU, and
`arena_auto_interval_sec` schedules the next arena, so an arena that never ends
stalls the loop. See "Runaway guard" below.

Validation beyond per-field ranges, enforced on Save:
- `elo1 > elo0` (otherwise the likelihood ratio is degenerate or inverted)
- `alpha + beta < 1`
- `min_games <= arena_sprt_max_games` when the guard is non-zero

## The statistics

### Bounds

With type I error α and type II error β, the log-likelihood-ratio bounds are

```
lower (reject H₁, do not promote) = log( β / (1 − α) )
upper (accept H₁, promote)        = log( (1 − β) / α )
```

At α = β = 0.05 that is `[−2.94, +2.94]`.

### The LLR itself — needs pinning, not assuming

Two candidate formulations. **This is the part of the plan that must be
verified against published reference values before it is trusted**, not
reasoned about in a review:

1. **BayesElo trinomial.** Model W/D/L probabilities from an Elo difference and
   a `drawelo` nuisance parameter estimated from the running tally, then sum
   `log P(outcome | H₁) / P(outcome | H₀)` per game. This is the classic
   fishtest formulation. More faithful to how draws behave; more moving parts,
   and the drawelo estimator is the fiddly bit.

2. **Generalised SPRT (GSPRT) on the score.** Treat each game's score as a
   variable on `{0, 0.5, 1}` and test its mean:
   ```
   mu_k = 1 / (1 + 10^(-elo_k / 400))            for k in {0, 1}
   xbar = (W + 0.5·D) / N
   var  = (W(1-xbar)² + D(0.5-xbar)² + L(0-xbar)²) / N
   LLR  = N · (mu1 - mu0) · (xbar - (mu0+mu1)/2) / var
   ```
   No nuisance parameter. This is the form modern fishtest uses (in its
   pentanomial variant).

**Status of its inputs — corrected.** An earlier draft claimed this "reuses code
that is already tested". That was overstated. Of the three inputs:
`ArenaEloStats.score` exists; the variance expression exists **inside**
`summary()` as a local used for the confidence interval and then discarded —
`Summary` does not expose it; the LLR does not exist at all. Expect to expose
the variance (or recompute it in `ArenaSPRT`) rather than to reuse a tested
value.

**Recommendation: GSPRT (2) for v1** — no nuisance parameter to estimate, and
it simulates as correctly calibrated (below). Implement (1) later only if (2)
proves mis-calibrated in practice.

### Simulated calibration and duration

Synthetic outcomes only — no games are played. Draw a true Elo, convert to
W/D/L at a chosen draw rate, sample outcomes, feed the decision function. At a
0.85 draw rate (this engine's actual regime), α = β = 0.05, open-ended:

| elo0 → elo1 | true Elo | accept | reject | median games |
|---|---|---|---|---|
| 0 → 10 | 0 | **0.05** | 0.95 | 722 |
| 0 → 10 | 10 | **0.95** | 0.05 | 731 |
| 0 → 10 | 5 | 0.47 | 0.53 | 1,138 |
| 0 → 20 | 0 | 0.03 | 0.97 | 182 |
| 0 → 20 | 20 | 0.93 | 0.07 | 190 |
| 0 → 35 | 0 | 0.02 | 0.98 | 58 |
| 0 → 35 | 35 | 0.90 | 0.10 | 63 |

Accept rates land on α and 1−β exactly, which is the property that matters.
Inconclusive was 0.00 in every cell — the runaway guard never fired even at a
true Elo sitting midway between the hypotheses.

**Note on a fixed count.** An earlier draft proposed reusing the 400-game
tournament size as a cap. Simulated that way, the test accepts ~0.6% of the time
when H₁ is true — truncation destroys the calibration entirely. The open-ended
form is not a refinement; it is the thing that makes the test work.

**Default choice: `elo0 = 0`, `elo1 = 10`** (decided 2026-09-20). Median ~720
games at an 0.85 draw rate — slower than today's fixed 400, deliberately: it
detects roughly half the edge, with a controlled false-promote rate. `elo1 = 20`
(~190 games) and `elo1 = 35` (~60) remain available as settings.

**Not doing: pentanomial.** It models *game pairs* (same opening, both colours)
and is the lower-variance modern standard — but this engine has no opening book,
so there is nothing to pair on. Colour alternation gives balance, not pairing.
Revisit if an opening book is ever added.

### Deciding

After each completed game, once `gamesPlayed >= min_games`:

```
llr >= upper  ->  accept H₁  ->  promote, stop the tournament early
llr <= lower  ->  reject H₁  ->  do not promote, stop the tournament early
otherwise     ->  continue
```

### Zero-variance records — an open question, found in phase 3

The GSPRT's denominator is the **empirical** score variance, and that is
exactly zero when every game had the same result. `logLikelihoodRatio` returns
`nil` there, `decide` never reaches the boundary check, and the test runs to
`arena_sprt_max_games` and returns **inconclusive — which does not promote**.

So, under the statistic as implemented:

| record | LLR | decision |
|---|---|---|
| 64W / 0D / 0L | undefined (var = 0) | inconclusive → **no promotion** |
| 1000W / 0D / 0L | undefined (var = 0) | inconclusive → **no promotion** |
| 0W / 1000D / 0L | undefined (var = 0) | inconclusive |
| 31W / 0D / 1L | +7.02 | accept |
| 39W / 0D / 1L | +11.05 | accept |
| 400W / 0D / 1L | +1137.16 | accept |

A flawless candidate is the most promotable one there is, and this refuses it.
The saving grace is that the degeneracy breaks the instant a single game
differs — 31W/1L already accepts comfortably — so it can only bite a candidate
that stays perfect for an entire run. Against a weak early champion over a
`min_games` of 32 that is not impossible; in the engine's usual ~85%-draw
regime it is vanishingly unlikely, and the draw-side version (an all-draw
opening stretch) resolves on the first decisive game.

This is pinned by `testPerfectRecordNeverDecidesUntilTheGuardFires` and
`testAllDrawRecordIsUnscoreableUntilOneDecisiveGame` — as a record of current
behaviour, **not** as a blessing of it. Changing it means changing the
statistic, which is a decision to take deliberately rather than patch around:

- Use the variance implied by the hypotheses rather than the sample
  (`mu·(1−mu)`-style), which is never zero. Changes the LLR everywhere, so the
  calibration simulation must be re-run.
- Clamp the empirical variance to a small floor. Cheap, but the floor is an
  arbitrary constant that sets how fast a near-degenerate record decides.
- Leave it, and rely on the guard. Defensible if `min_games` is large enough
  that a perfect record over it is not credible.

Not decided. Nothing in phases 1–3 depends on which way it goes.

### Runaway guard

If `arena_sprt_max_games` is reached with the LLR still between the bounds, the
result is **inconclusive → do not promote**. A test that did not reach
significance is not evidence of improvement, and promoting on it would
reintroduce exactly the noise-promotion problem SPRT exists to remove.

The guard is a safety valve, not a sample size: it should be set far above the
expected decision point, so that hitting it is a signal that `elo0`/`elo1` are
too close together rather than a normal outcome.

## Where the changes land

1. **`Training/TrainingParameters.swift`** — seven `@TrainingParameter`
   declarations, seven accessors, and the `ArenaPromotionCriterion` enum.
   Registry count **62 → 69**; `TrainingParametersTests.test_registry_size`
   pins it and must be updated. *(An earlier draft said "36 → 43" and "six new
   parameters" — both were wrong; the registry held 62 keys when phase 1
   landed, and the table above lists seven.)*

2. **`Arena/ArenaSPRT.swift`** — **done** (`703dc77`). Pure, dependency-free
   statistics, no actors, no UI, no I/O, mirroring how `ArenaEloStats` is
   structured:
   ```swift
   enum ArenaSPRT {
       enum ConfigError: Error, Equatable, CustomStringConvertible { ... }
       struct SPRTConfig: Equatable, Sendable {   // throwing init validates
           let elo0, elo1, alpha, beta: Double
           let minGames, maxGames: Int
           var bounds: (lower: Double, upper: Double)
       }
       enum Decision: String, Equatable, Sendable {
           case accept, reject, continueTesting, inconclusive
           var promotes: Bool      // only .accept
           var isFinal: Bool       // != .continueTesting
       }
       static func expectedScore(forElo:) -> Double
       static func bounds(alpha:beta:) -> (lower: Double, upper: Double)
       static func logLikelihoodRatio(wins:draws:losses:config:) -> Double?
       static func decide(wins:draws:losses:config:) -> Decision
   }
   ```
   Two deviations from this plan as written: the decision type is nested as
   `ArenaSPRT.Decision` rather than a free-standing `SPRTDecision`, and it
   carries a fourth case, `.inconclusive`, so the runaway guard is
   distinguishable from a real rejection at every call site rather than only in
   the log line. `SPRTConfig`'s init throws `ConfigError` so an invalid
   hypothesis pair cannot reach the decision function at all.

3. **`Arena/TickTournamentDriver.swift`** — the change is smaller than it looks.
   The loop already gates spawning on a single predicate:
   ```swift
   if nextGameIndexToSpawn < totalGames { /* recycle slot */ }
   else                                  { /* retire slot */ }
   ```
   Replace that integer comparison with a `shouldContinue` decision that the
   caller supplies: the existing fixed-count rule in threshold mode, the SPRT
   verdict in SPRT mode. Two setup lines also need adjusting — the
   `guard totalGames > 0` early return and `initialK = min(concurrency,
   totalGames)`, which in SPRT mode is just `concurrency`.

   Slot retirement already gives the correct drain behaviour for free: when the
   predicate flips, slots retire as their games finish, so in-flight games still
   complete and still count. No cancellation, no wasted GPU work.

   `candIsWhiteNext = (nextIdx % 2 == 0)` continues to alternate correctly since
   `nextIdx` simply keeps incrementing.

   **The verdict must be latched at the crossing, not recomputed at the end.**
   This is the one place the drain behaviour is not free. With `concurrency`
   games in flight, the tally when the LLR crosses a bound is not the tally the
   driver returns: up to `K − 1` further games finish while the slots retire,
   and `TournamentStats` reports all of them. Re-running `decide` on that final
   tally would be wrong in both directions — the extra games are a variable
   number of observations admitted *because* the test already stopped, which is
   precisely the optional-stopping bias SPRT's calibration assumes away, and a
   re-decision can land on `.continueTesting`, for which there is no action once
   the tournament is over. So the driver latches the first `isFinal` decision
   together with the `(W, D, L)` it was made on, and that is what the gate and
   the record read; the drained games are still reported, in the Elo summary and
   as `gamesPlayed`, as description rather than as evidence. The same applies to
   the runaway guard: `n >= maxGames` is evaluated per completed game, so the
   returned `gamesPlayed` can exceed `maxGames` by up to `K − 1` without that
   being a violation.

   Tests should pin this directly — a driver run at `concurrency > 1` whose
   latched decision disagrees with `decide(...)` applied to the final tally.

4. **`App/SessionController+Arena.swift`** — snapshot all arena config into a
   value struct at run start (it already reads from `TrainingParameters.shared`
   there), then branch:
   ```swift
   let shouldPromote: Bool
   switch criterion {
   case .scoreThreshold:  // unchanged, byte-for-byte
   case .sprt:            // driver's decision == .accept
   }
   ```
   The abort path must keep precedence over both.

5. **`Arena/TournamentRecord.swift`** — new optional fields: `promotionCriterion`,
   `sprtLLR`, `sprtDecision`, `sprtBounds`, and the config used. Optional so
   existing persisted history still decodes — `ArenaHistoryCodableTests` covers
   this and must be extended with a round-trip for both old and new shapes.

6. **`Arena/ArenaLogFormatter.swift`** — extend the `[ARENA]` line with
   `crit=`, `llr=`, `sprt=` (accept/reject/inconclusive). The existing
   `elo=/elo_lo=/elo_hi=` fields stay: they remain the right descriptive
   summary regardless of which criterion decided.

   **Shipped wider than planned**, because three non-promoting SPRT outcomes
   have to stay distinguishable and a bare `kept` collapses them:
   `formatVerdict` now renders `kept (SPRT reject)`, `kept (SPRT inconclusive
   — guard, not a rejection)` and `kept (SPRT undecided — run ended early)`.
   A multi-line `formatSPRTBlock` carries the decision, the ratio, the
   boundaries, the full hypothesis set, and — when they differ — both the
   tally at the crossing and the count of games drained after it. The KV line
   gains `crit=` plus a `sprt_*` group including every config field, so a
   persisted LLR stays interpretable without knowing what the parameters were
   that day; `sprt=none` marks an SPRT run that never decided, which `crit=`
   alone cannot express. Threshold-mode output is unchanged apart from
   `crit=score`.

7. **`App/UpperContentView/ArenaSettingsPopover.swift` +
   `ArenaSettingsPopoverModel.swift`** — a `Picker` bound to the criterion, and
   the SPRT fields dimmed when it is not selected. The popover is hand-built
   (`ArenaPopoverField` rows, transactional parse/validate/apply on Save), so
   this fits the existing shape; `Picker` is already used elsewhere in the app
   (`ArenaSurfaceView`, `BoardSideView`).

   **Shipped with four decisions worth recording:**

   - The picker + hypothesis fields live in their own
     `ArenaPromotionCriterionSection` view struct, per the project's
     one-view-struct-per-file rule (a `@ViewBuilder` helper property would
     have been the smaller diff and is not blessed).
   - **Both groups stay visible; the inactive one dims.** `# of games` and
     `Promote threshold` grey out under SPRT, and the hypothesis block greys
     out under the threshold. Hiding either would resize the popover on every
     criterion change and, worse, remove the only on-screen explanation for
     why a setting appears to do nothing.
   - **The SPRT fields are validated even while inactive.** Letting a bad
     `alpha` save because SPRT is not currently selected defers the failure to
     the start of the first SPRT arena, which is the most expensive place to
     discover a statistics misconfiguration.
   - **The criterion is written last, after every field is accepted.** A flip
     to `.sprt` alongside a rejected hypothesis set would leave the next arena
     running the new rule against the old numbers. Cross-field failures report
     on their own line rather than reddening one of the two fields in a
     relation, and they are checked by asking `ArenaSPRT.SPRTConfig` to
     construct itself — the same validator the arena uses — so the popover
     cannot drift out of agreement with what the arena will accept.

8. **`Persistence`** — confirm the new ids are carried in the `.dcmsession`
   saved-config block, or deliberately excluded, matching how
   `value_label_smoothing_epsilon` was handled (it is *not* in that block — a
   known gap, not a precedent to copy blindly).

   **Deliberately deferred to phase 4, not skipped.** `CLAUDE.md`'s
   add-a-parameter checklist puts session save/load in the same step as the
   declaration, but until the gate branch exists these seven parameters change
   nothing a resumed session could observe — an `Optional` field and a
   `[RESUME-PARAM]` block written in phase 2 would be dead code that the phase-4
   tests could not distinguish from correct. They land with the gate, in
   `SessionCheckpointState` + `buildCurrentSessionState` + the
   `[RESUME-PARAM]` block in `SessionController+Training.swift`, mirroring
   `batchStatsInterval`, with both the "from session" and "saved=nil
   (defaulted)" branches logged.

## Tests

New `ArenaSPRTTests.swift`:
- Bounds at α = β = 0.05 are `[−2.94, +2.94]`, and are asymmetric when α ≠ β.
- A lopsided win record drives the LLR above the upper bound; a lopsided loss
  record below the lower one.
- An even record with many games stays between the bounds — the test does not
  drift into a decision from sample size alone.
- `decide` returns `.continueTesting` below `minGames` regardless of record,
  including for a record that would otherwise cross immediately.
- Draw-heavy records reduce |LLR| per game relative to decisive ones.
- `elo1 <= elo0` is rejected at construction.
- **Calibration:** simulate many tournaments at a known true Elo and confirm the
  false-accept rate approximates α. This is the test that actually proves the
  implementation, and the one most likely to fail first.

Existing suites to extend:
- `TickTournamentDriverTests` — early stop fires; in-flight games still counted;
  with no SPRT config the driver is unchanged.
- `TournamentRecordTests` / `ArenaHistoryCodableTests` — old records decode.
- `ArenaLogFormatterTests` — new fields in both modes.
- `TrainingParametersTests` — registry size, unique ids, ranges, defaults.

## Risks

- **The LLR formula is the whole feature.** Bounds and plumbing are easy; a
  mis-specified likelihood silently mis-calibrates the error rates and looks
  like it works. The calibration test is not optional.
- **Fewer games per tournament** changes arena wall-clock and therefore the
  cadence of the whole train/arena loop. SPRT accepting at 60 games where 400
  were budgeted is the *point*, but it will shift timings elsewhere.
- **Inconclusive runs.** If `elo0`/`elo1` are set close together, most
  tournaments will hit the cap undecided and nothing will ever promote. The UI
  hint should say so, and the log should make "inconclusive" visually distinct
  from "rejected".
- **Registry size is pinned by a test** — the seven new parameters will fail it
  until updated. Expected, not a surprise.

## Phasing

1. ~~`ArenaSPRT.swift` + its tests, including calibration. No wiring. Fully
   verifiable in isolation.~~ **Done** — `703dc77`, 24 tests.
2. ~~Parameters + registry + validation, with tests.~~ **Done** — `26a747e`.
   Registry 62 → 69. Also fixed `testZeroMaxGamesMeansUnbounded`, which had
   been committed red in phase 1.
3. ~~Driver early-stop, behind an optional config, with tests proving the
   no-config path is unchanged.~~ **Done** — `f21ff44`. Added
   `ArenaSPRT.Monitor` / `Verdict` for the latching, which this plan had not
   anticipated; found the zero-variance degeneracy above.
4. ~~Gate branch in `SessionController+Arena`, record, log formatter.~~
   **Done.** Also covered persistence (item 8) and the busy label, which
   needed to stop printing `game 37/400` for a run that had no schedule.
5. ~~UI: picker + conditional enabling.~~ **Done** — plus the last-arena
   summary in the popover, which showed every non-promotion as a neutral
   "kept" and now distinguishes an inconclusive run (orange — usually says the
   hypotheses need widening) from a rejection.

Phase 1 was where the risk was; the rest was plumbing.

## Open questions

1. ~~**GSPRT or BayesElo trinomial** for the LLR.~~ — **decided: GSPRT**,
   shipped in phase 1. Revisit the BayesElo trinomial only if the shipped form
   proves mis-calibrated against real arena records.
2. ~~Defaults for `elo0`/`elo1`~~ — **decided: `0 / 10`**.
3. ~~**Should an inconclusive run be visually distinct from a rejection** in
   the arena history UI, or is the log line enough?~~ — **decided: yes,
   distinct, everywhere.** The log block spells out that the guard is not a
   rejection; `formatVerdict` renders three different strings; and the
   popover's last-arena line colours inconclusive orange rather than the
   neutral secondary used for "kept". The three states are not
   interchangeable, and a reader who cannot tell them apart will read a
   too-narrow hypothesis gap as a bad candidate.

4. **New, unresolved: the zero-variance degeneracy.** See the section above.
   A candidate that wins every game does not promote. Pinned by test, not
   fixed, because fixing it means changing the statistic and re-running the
   calibration.
