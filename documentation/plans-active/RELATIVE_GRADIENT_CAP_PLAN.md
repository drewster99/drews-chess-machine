# Relative gradient-norm cap plan: clip each step at min(hard max, max(floor, k × recent median))

Status (2026-10-07): **approved by the owner** ("ok approved keep it going", 2026-10-07 ~03:55). Every owner decision below is recorded as decided as recommended. **P1–P4 implemented** (CHANGELOG 2026-10-07 "Relative gradient-norm cap"); the arena capture and rewind of the history (T12) landed in P1 so no intermediate commit breaks a GUI promotion. P5 (flip the default mode to `clip`) waits on validation V-1 and V-3.
- Every `file:line` was checked against `main` at `9e344de2`.
- Paths are relative to `DrewsChessMachine/DrewsChessMachine/` unless they start with `DrewsChessMachine/` (project folder), `DrewsChessMachineTests/` (= `DrewsChessMachine/DrewsChessMachineTests/`), `documentation/` or `experiments/`.
- Session logs are under `~/Library/Logs/DrewsChessMachine/`. Every number quoted from a log was re-measured for this plan from the log itself (script: Part E, "Reproduce").

**The request (owner, 2026-10-06/07).** After E-0020 (a fixed cap of 1.0 stopped B-silu's step-20,600 blowup): "if it stays that way that'd be great but we can't do a cap of 1 all the time". Early training runs gNorm about 30 at step 1 and 1–3 for hundreds of steps, so a fixed cap of 1.0 would clip nearly every early step. The general form is a cap relative to the run's own recent gradient norms.

**Owner direction (2026-10-07), built in below.**
- Cap formula: `cap = min(hard max, max(floor, k × median of the last N pre-clip norms))`. The existing `grad_clip_max_norm` stays the hard maximum, always in force; the relative cap only tightens it.
- The floor is a first-class parameter ("worst case we might end up with an enforced minimum for the cap").
- Warm-up: start the relative cap after a short minimum history (about 100 steps), using the median of the available history until the window fills. The hard max covers the steps before that.
- The history is saved with the trainer state, so an exact resume needs no second warm-up.
- k's final value is gated on a per-step measurement (V-1), with k = 4 or 5 as fallbacks.
- A fresh-start validation run (V-3) exercises the warm-up.

**What this plan does.**
- Adds one pure type, `GradientCapPolicy`, that turns a configuration, the hard max and the pre-clip norm history into the cap for the next step (and says which term bound it).
- Keeps the history — every real-data SGD step's pre-clip global norm and the cap that step used — inside `ChessTrainer`, next to the trainer clock. It is trainer state: saved in every trainer-state file, restored by every exact resume, rewound by a GUI promotion with the weights, velocity, clock and dropout state.
- Feeds the cap through the **existing** `gradClipMaxNorm` scalar placeholder (`Training/ChessTrainer.swift:1693`, written every step at `:6249`). No graph change, no rebuild.
- Shares the trailing-median definition with the alarms' `gradient_spike` rule (`TrainingHealthReference.make`, `Training/TrainingHealth.swift:809-839`): one function, parameterized by a span policy.
- Logs every clip event (`[GRAD-CLIP]`), and adds the per-window maximum pre-clip norm, the clip count and the cap to every step line (`[REPLAY]`, `[VS-UCI]`, `[STATS]`).
- Five new parameters, through the full CLAUDE.md checklist.
- Works the same on GUI Play-and-Train (including GUI `--train`), `--replay-corpus` and `--train-vs-uci`: the logic is inside the one trainer all three use.

Rules this plan follows (CLAUDE.md files and the owner's standing rules):
- one source of truth: the history lives only in the trainer; the median definition only in `TrainingHealthReference`;
- one code path for GUI, corpus replay and train-vs-UCI;
- no silent defaults, no fallbacks: a history that disagrees with the trainer clock is an error, never patched;
- no `try?`, no force unwraps;
- the full parameter checklist, each parameter with a declared `absentValue`;
- probe isolation: the health monitor stays an observer; the cap reads nothing from it;
- tests are never modified or deleted without the owner's approval (the two needed edits are OD-10, approved);
- one SwiftUI `View` per file, no helper `some View` properties, no `if`-gated visible content.

---

## Summary

| # | Item | Phase |
|---|---|---|
| 1 | `TrailingReferencePolicy` + `TrainingHealthReference.make(_:windowStart:policy:)`: the one trailing-median definition, used by rules 6/9 and the cap | P1 |
| 2 | `RelativeGradientCapMode` (`off` / `logOnly` / `clip`), `RelativeGradientCapConfiguration` (validated), `GradientCapDecision`, `GradientCapPolicy.decide(...)` (pure) | P1 |
| 3 | `GradientNormHistory` (pure value type): contiguous per-step ring of (pre-clip norm, applied cap), capacity = the window parameter's declared maximum | P1 |
| 4 | `ChessTrainer`: owns the history; decides the cap on its queue before each real-data step; feeds it through the existing placeholder; records the step's norm and cap in the same block as the clock increment; `TrainStepTiming` carries the decision | P1 |
| 5 | Persistence: `trainer_grad_norm_history` metadata key in every trainer-state file; `TrainerResumeSnapshot.gradNormHistory`; `restoreExactly` restores it; new `ResumeGap.gradNormHistory` | P1 |
| 6 | Five `TrainingParameters` (mode, k, N, minimum history, floor), full checklist; `TrainerHyperparameters` carries them | P2 |
| 7 | CLI paths: `[GRAD-CLIP]` lines, step-line fields, `[REPLAY-HPARAMS]` / `[VS-UCI-HPARAMS]` fields, `results.json` fields, resume gap handling | P2 |
| 8 | GUI: settings popover section, `[STATS]` fields, session save/load fields, promotion rewind of the history | P3 |
| 9 | Docs: CLAUDE.md tag list, `documentation/training-health-alarms.md` cross-reference, `--help`, CHANGELOG | P4 |
| 10 | Flip the default mode from `logOnly` to `clip` with the k chosen by V-1/V-3 | P5 (after V-1, V-3) |

---

# Part E — Evidence

## E1. The incident (E-0020, `experiments/20261005-lr-schedule-ab/README.md` "Arm B-silu-clip1" to "Gradient-cap experiment: results")

- B-silu (SiLU tower, LR cycle 1.0 ↔ 0.001, period 10,000, `grad_clip_max_norm` 15) blew up at trainer step ~20,600: logged gNorm 2.566 against a trailing median of 0.359, LR 0.881, illegal mass 0.80, 20 of 128 policy pre-BN channels parked, pElo 1571.9 at 19k → 457.4 at 21k.
- ctl15 (`--resume-exact` from the 18k checkpoint, cap 15) reproduced B-silu bit for bit: all 80 `[REPLAY]` lines 18,050–22,000 equal, the 22,000 checkpoint `[LAYER-HEALTH]` equal.
- clip1 (same resume, cap 1.0) first differed at 19,800, so the cap first acted on an unlogged step in 19,751–19,799, where the pre-clip norm exceeded 1.0, at least 2.8× the logged 0.33–0.38 around it. clip1 never blew up and finished level with B (ReLU) at 40,000. Its largest logged gNorm over the whole run is 0.462.
- gNorm is logged every 50 steps, so neither the precursor's value nor the healthy per-step spread is known. That is why k is gated on V-1.
- clip2 / clip5 (fixed caps 2.0 / 5.0 from the same 18k state) are running now (README "Arms B-silu-clip2 and B-silu-clip5"); V-2 compares against them.

## E2. A 3× trailing-1000 cap on the every-50-step logged norms (owner's simulation, 2026-10-07)

| Run | 3× cap range (median) | Steps over cap | Largest ratio |
|---|---|---|---|
| B | 0.72–3.20 (0.86) | 0 | 1.40 |
| B-leakyall | 0.73–2.94 | 0 | 1.38 |
| B-silu | 0.76–3.42 | 1 | 7.15 at 20,600 |
| clip1 | 0.69–1.15 | 0 | 1.32 |

Re-measured for this plan with the same method (median of the logged norms in the 1,000 trainer steps before each logged step, steps after 1,000), extended to the other runs:

| Run | Log(s) | Rows | Steps over 3× | Largest ratio (step, gNorm, median, LR) | Lowest trailing median (step) |
|---|---|---:|---:|---|---|
| A (constant LR 0.01) | `dcm_log_20261005-013220-2.txt`, `-171451.txt` | 802 | 0 | 1.25 (14,750; 1.42; 1.14; 0.01) | 0.983 (36,300) |
| B | `dcm_log_20261005-013235.txt`, `-171541.txt` | 802 | 0 | 1.40 (29,950; 0.417; 0.297; 0.472) | 0.240 (35,800) |
| B-leaky | `dcm_log_20261005-204108.txt` | 801 | 0 | 1.47 (20,800; 0.553; 0.378; 0.956) | 0.241 (35,800) |
| B-leakyall | `dcm_log_20261005-234434.txt` | 801 | 0 | 1.38 (10,050; 0.489; 0.355; 0.546) | 0.242 (35,800) |
| B-silu | `dcm_log_20261005-234437.txt` | 722 | 1 | 7.15 (20,600; 2.566; 0.359; 0.881) | 0.254 (14,350) |
| clip1 | `dcm_log_20261006-170000.txt` | 441 | 0 | 1.32 (29,950; 0.379; 0.287; 0.472) | 0.229 (35,750) |
| C (cycle 10 ↔ 0.01, diverged at LR 3) | `dcm_log_20261005-090417.txt`, `-121841.txt` | 124 | 3 | 13.43 (300; 14.325; 1.067; 3.0) | 0.013 (4,663) |

- At 2× the healthy runs still have no step over the cap; B-silu has 2 (20,600 and 21,050, the latter after the blowup) and C 11.
- A window of N = 5,000 instead of 1,000 raises the healthy runs' largest ratio from 1.47 to 2.00 (B-leaky at 20,800), because the median then spans LR phases (OD-3).
- GUI `[STATS]` `gNorm=` is the rolling mean, not the step's norm (`App/SessionController+Training.swift:1587`), so the two fresh GUI logs (`dcm_log_20260921-000925.txt`, `dcm_log_20260927-211635.txt`) say little about per-step spread; neither shows a ratio above 1.12.
- **Caveat (owner):** per-step ratios will be larger than every-50-step ratios, because the maximum of 50 draws exceeds one draw. Both tables are lower bounds on the per-step ratio. V-1 measures it.

## E3. The LR cycle: in these runs gradient norms grow slightly at high LR, they do not shrink

The brief's premise was that norms shrink at high LR. In B and B-silu after the 1,000-step warmup they grow slightly instead. Median logged gNorm by the step's LR:

| LR band | B: rows, median (min–max) | B-silu: rows, median (min–max) |
|---|---|---|
| < 0.01 | 317, 0.257 (0.232–0.320) | 278, 0.280 (0.239–0.441) |
| 0.01–0.1 | 171, 0.265 (0.236–0.367) | 149, 0.292 (0.243–0.691) |
| 0.1–0.5 | 149, 0.303 (0.251–0.429) | 131, 0.328 (0.242–0.663) |
| ≥ 0.5 | 144, 0.358 (0.283–0.605) | 143, 0.382 (0.264–2.566) |

- Trough to peak is a 1.4× change in the median over half a period (5,000 steps). A 1,000-step window sees about a fifth of that half-period (an estimate of ~1.07× if the change is spread evenly; not measured per window). The measured N = 1,000 ratios in E2, at most 1.47 in a healthy run, already include this lag. Against k = 3 it is negligible in either direction.
- Shrinking norms at high LR do occur during the first ~1,000 steps (B-silu: 3.574 at step 50 → 0.541 at step 1,000 while LR rises 0.05 → 1.0), but that is training progress as much as LR. A falling norm only loosens a trailing-median cap (the median lags high), so it cannot cause a false clip.
- **Why a relative cap suits the cycle where a fixed one cannot:** the update is `lr × clipped gradient`. A fixed cap allows the same gradient norm at LR 1.0 as at LR 0.001 — a 1,000× larger update. The relative cap holds the gradient to k times what the run has been producing in the current LR phase, so no per-phase retuning is needed. The blowups seen so far (B-silu 0.88, C at LR 3) happened at high LR, where an over-sized gradient does the most damage.
- Default momentum follows the LR cycle (`momentum_cycle_period_steps` 1,000 in `parameters-B.json`). With N = 1,000 the window spans one momentum period, so the median averages over it rather than chasing it.

## E4. Early training (fresh start): preview and caveat

- Owner's preview from every-50-step logs, with the median of the available history from step 100: the 3× cap is 46–54 at step 100, 8–13 at steps 150–200 and 5–9 at steps 250–450. No logged early step would clip, on B, B-leakyall or B-silu.
- **Caveat:** the in-app history is per step. Step 1's 27–31 is one entry among ~100 by step 101, not one of two logged rows, so the real early medians are much lower than the 46–54 above (the logged steps 50–450 run 0.6–3.7), and the per-step early spread is unknown. The early cap will be tighter than the preview says. V-3 (fresh start, every clip logged) is the real measurement.
- Before the warm-up completes, only the hard max applies: step 1's 27–31 is clipped to 15 by `grad_clip_max_norm`, exactly as today.

## E5. Reproduce

- Method for every number in E2–E3: read the `[REPLAY] step=` lines (GUI: `[STATS] elapsed=`) of the named logs; key each row by `trainerStep=` (GUI: `steps=`), keeping the last row per step; for each row take the median of the rows in the 1,000 trainer steps before it, requiring at least 5 rows spanning at least 200 steps (the `gradient_spike` reference definition, `Training/TrainingHealth.swift:823-839`); the ratio is the row's `gNorm` over that median. E3 buckets rows after step 1,000 by the row's `lr=`.
- The one-off script used for this plan is not in the repository; V-1's write-up adds it as `experiments/20261005-lr-schedule-ab/relcap_sim.py` with its exact command lines, so the tables can be regenerated.

---

# Part R — The rule

## R1. The cap

For the real-data SGD step that will be trainer step `s` (`s = completedTrainSteps + 1`):

```
hardMax  = grad_clip_max_norm                                  (existing, always in force)
history  = pre-clip norms of steps s−N … s−1 that the trainer recorded (contiguous)
if mode == off, or history has fewer than W entries:
    cap = hardMax                                              binding = hard
else:
    m        = median(history)                                 (mean of the two middle values for an even count)
    relative = max(floor, k × m)
    cap      = min(hardMax, relative)
    binding  = hard     if hardMax ≤ relative
               floor    if floor ≥ k × m and floor < hardMax
               relative otherwise
fed cap  = cap        in mode clip
           hardMax    in mode logOnly (the decision is still computed and logged)
```

- The graph is unchanged: `clipScale = cap / max(norm, cap)` (`Training/ChessTrainer.swift:4049-4058`). A step whose norm is at or below the fed cap gets a scale of exactly 1.0 (`x / x` is exact in IEEE fp32), so an unclipped step is **bit-identical** whatever cap was fed. This is what makes "identical to the control until the first clip" a valid comparison in V-1 to V-3 (X3 pins it).
- `clipped` for logging is `preClipNorm > fedCap`, the same comparison the graph's `maximum` makes.
- All arithmetic is in `Double` on the trainer's queue and converted to `Float` once for the feed. It is deterministic: the same history and configuration give the same cap.

## R2. Warm-up (minimum history W)

- The relative term applies once the window holds at least **W** entries (default 100, OD-4). Until the window fills (N entries), the median is over the entries it has.
- A fresh run therefore has no relative cap for its first 100 steps (the hard max covers them), then a median over 100 → 1,000 entries.
- An exact resume restores the history, so there is no second warm-up. A resume without history (a checkpoint written before this plan, a branch, a GUI resume of a session without a trainer file) starts an empty history: 100 steps without the relative term, logged and, in `clip` mode, a resume gap (D4).
- `W ≤ N` is required; a configuration with `W > N` (the relative term could never apply) is refused (D1).

## R3. Floor

- Default **0.5** (OD-5). It bounds the relative term from below; the hard max still bounds everything from above.
- Why 0.5: every healthy run's lowest trailing median is 0.229–0.983 (E2), so `3 × median` never falls below 0.69 and the floor never binds in a healthy run at k = 3 (nor at k = 2.5: 0.57). It binds only when the median falls below 0.167, which no healthy run reached. The runs that did are sick: C after its divergence sat at a median of 0.013–0.015 with `gradient_collapse` territory (rule 5 raises at a window median of 0.1). Without a floor the cap there would be 0.04: clipping a nearly-dead network at 3× its tiny norms would only slow any recovery. With the floor it stays at 0.5, which still clips a 33× spike.
- A floor at or above the hard max is allowed (the formula is well-defined: the hard max wins) but means the relative term can never act; it is logged once per configuration as `[GRAD-CLIP] config … relative_cap_inert=floor_at_or_above_hard_max` (OD-14).

## R4. Median source: pre-clip norms

- The history stores each step's **pre-clip** norm (`TrainStepTiming.gradGlobalNorm`, read back every step, `Training/ChessTrainer.swift:7110`).
- Post-clip norms would ratchet: every clip records a value at the cap, pulling the median down, which lowers the next cap and clips more. Under `min(hardMax, …)` a post-clip history can only shrink while clipping.
- The pre-clip median is robust: a median ignores up to half the window, so an isolated spike (B-silu's 20,600) moves it by one rank, not by its size.
- Known limit: if more than half the window is elevated (a sustained regime shift), the median follows it and the relative cap loosens. That is intended for a genuine shift (the run's gradient scale changed), and the hard max still bounds a runaway. Rule 9 (`gradient_spike`, 5×) and rule 5 (`gradient_collapse`) still report from pre-clip norms, so a clip never hides a spike from the alarms.

## R5. What is recorded per step

Every real-data step records `(trainerStep, preClipNorm, fedCap)` in the history, whatever the mode (so `logOnly` and `off` runs also save a history, and switching to `clip` later — live in the GUI, or on a resume — starts warm). The synthetic `trainStep(batchSize:)` path (GPU sweeps, smoke tests; `Training/ChessTrainer.swift:4386`) feeds the hard max and records nothing, just as it does not advance the clock (`:4959-4976`).

---

# Part D — Design

## D1. Types (new file `Training/RelativeGradientCap.swift`, pure, no Metal)

- `enum RelativeGradientCapMode: Int, Sendable, CaseIterable { case off = 0, logOnly = 1, clip = 2 }` with `token` (`off` / `log_only` / `clip`) for log lines. A mode, not an enable flag plus a log flag: three states, one parameter (OD-7).
- `struct RelativeGradientCapConfiguration: Sendable, Equatable { mode, multiple (k), windowSteps (N), minimumHistorySteps (W), floor }` with a throwing `init` that refuses `W > N`, non-finite or non-positive k / floor, and N above `GradientNormHistory.capacity`. Thrown error names both parameter ids. This is the one place the cross-parameter rule lives.
- `struct GradientCapDecision: Sendable, Equatable { fedCap: Float, decidedCap: Float, binding: Binding (hard / relative / floor), referenceMedian: Double?, referenceCount: Int, mode }`. `decidedCap` is the rule's cap; `fedCap` equals it in `clip` and equals the hard max otherwise. `func clipped(preClipNorm:) -> Bool` (against `fedCap`) and `wouldClip(preClipNorm:)` (against `decidedCap`, for `logOnly`).
- `enum GradientCapPolicy { static func decide(configuration:hardMax:history:nextTrainerStep:) -> GradientCapDecision }` — the R1 formula. The median comes from `TrainingHealthReference.make(_:windowStart:policy:)` with `TrailingReferencePolicy(lookbackSteps: N, minimumRecords: W, minimumSpanSteps: 0)` (D3).
- `struct GradientNormHistory: Sendable, Equatable, Codable`:
  - storage: `lastTrainerStep: Int?`, `preClipNorms: [Float]`, `fedCaps: [Float]` (parallel, oldest first), capacity `GradientNormHistory.capacity` = `RelativeGradClipWindowSteps.declaredClosedRange.upperBound` (10,000; derived from the declaration, not a second constant);
  - `mutating func append(trainerStep:preClipNorm:fedCap:) throws` — requires `trainerStep == lastTrainerStep + 1` (or an empty history) and a finite norm; drops the oldest entry past capacity. A gap or repeat throws `GradientNormHistoryError.discontinuity(expected:got:)`: it means the clock moved without the history (a bug), and the step stops with the error rather than mixing two trajectories (D2);
  - `func window(endingBefore step: Int, count: Int) -> [(trainerStep: Int, value: Float?)]`;
  - `func summary(trainerSteps: ClosedRange<Int>) -> (maxPreClipNorm: Float?, clipped: Int, steps: Int)` for the step lines (D5);
  - JSON form: `{"version":1,"last_trainer_step":…,"pre_clip_norms":[…],"fed_caps":[…]}`, floats in shortest round-trip form like `TrainerScheduleState` (`Training/TrainerResumeState.swift:142-157`); decode refuses unequal array lengths, non-finite values, more than `capacity` entries, a negative step, or `last_trainer_step` smaller than the entry count.

## D2. `ChessTrainer` integration (`Training/ChessTrainer.swift`)

- New state: `private var gradNormHistory: GradientNormHistory` and `var relativeGradientCap: RelativeGradientCapConfiguration` (live, set like `gradClipMaxNorm` at `:1352`). Both are touched only on `executionQueue` (the history) or under the same discipline as the other live scalars (the configuration); the house comment style documents it.
- **Decide** in the real-data `trainStep(replayBuffer:batchSize:)` phase 3 block, on the trainer queue, immediately before its `buildFeeds` call (`:4931`); decision, feed, `graph.run`, readback, record and clock increment are then one block on one queue: `let decision = GradientCapPolicy.decide(configuration: relativeGradientCap, hardMax: gradClipMaxNorm, history: gradNormHistory, nextTrainerStep: _completedTrainSteps.value + 1)`.
- **Feed**: `BatchFeedsInput` gains `gradientCapFeed: Float`; `buildFeeds` writes it to `gradClipMaxNormNDArray` instead of reading `gradClipMaxNorm` directly (`:6249`). The synthetic path passes `gradClipMaxNorm`. One writer, one placeholder, no graph change.
- **Record**: after `runPreparedStep` returns (norm already read back and checked finite, `:7110-7160`), in the same block as `_completedTrainSteps.modify { $0 += 1 }` (`:4976`): `try gradNormHistory.append(trainerStep: newClock, preClipNorm: timing.gradGlobalNorm, fedCap: decision.fedCap)`. Recording and the clock increment are one block on one queue, so an export between them is impossible (the same guarantee `exportWeightsWithCompletedSteps` relies on, `:4414-4430`). The append runs before the increment; if it throws (D1's discontinuity, a bug), the error leaves `trainStep` and stops training, as the non-finite-loss path does (`:7146-7160`).
- **Timing**: `TrainStepTiming` gains `gradientCap: GradientCapDecision` (non-optional; the synthetic path's decision is `binding: .hard, mode: .off`). Every consumer (step lines, recorder, health monitor, GUI stats box) reads it from there.
- **Lifecycle** — the history follows the trainer's optimizer state exactly:

| Event | Where | History |
|---|---|---|
| Fresh weights | `resetNetwork(initialization:)` (`:2367`, zeroes the clock at `:2525`) | cleared |
| Branch: new weights, zero velocity | `loadBaseWeightsResetVelocity` | cleared |
| Exact resume (GUI session, corpus replay, train-vs-UCI) | `restoreExactly(from:)` (`Training/TrainerResumeState.swift:299`) | set from the snapshot, or cleared and reported (D4) |
| GUI promotion rewind | `App/SessionController+Arena.swift:200-215` (capture) and `:493-504` (restore) | captured with weights, velocity, clock and dropout state at arena start; restored with them on promotion |
| GUI resume of a session without a trainer file | `App/SessionController+Training.swift:950-962` (after `resetNetwork`) | stays cleared; the clock is set to the session's step count and the next append starts a new history at that step |
| Stop + continue | — | kept |

- New trainer API (queue-hopping like `captureDropoutState`, `:6061-6075`): `exportGradNormHistory() async throws -> GradientNormHistory`, `restoreGradNormHistory(_:) async throws` (checks `lastTrainerStep ≤ completedTrainSteps`; a history ahead of the clock throws). The bare `completedTrainSteps` setter (`:4403`) is unchanged; the append contiguity check turns any future clock move without a history move into a loud error on the next step.

## D3. One trailing-median definition (shared with `gradient_spike`)

- `TrainingHealthReference.make` (`Training/TrainingHealth.swift:823-839`) gains a `policy: TrailingReferencePolicy` argument (`lookbackSteps`, `minimumRecords`, `minimumSpanSteps`). `TrailingReferencePolicy.spikeRules` is built from the existing constants (`spikeReferenceLookbackSteps` 1,000, `spikeReferenceMinimumRecords` 5, `spikeReferenceMinimumSpanSteps` 200, `:485-490`). The existing two-argument `make(_:windowStart:)` stays, defined as `make(_:windowStart:policy: .spikeRules)` — the rules' named policy, not a default — so `TrainingHealthMonitorTests.swift:148-154` keep calling it unchanged.
- The cap calls it with `TrailingReferencePolicy(lookbackSteps: N, minimumRecords: W, minimumSpanSteps: 0)`. The median, the look-back bounds and the finite-value filter are therefore one implementation.
- **The data is not shared, by design (OD-9).** The health monitor's history (`Training/TrainingHealthMonitor.swift:161-162`) is an observer's: it is filled at evaluation time every 50 steps, reset per generation on rewinds, empty at process start, rebuilt from log rows by `--replay-health-log`, and may be disabled. The cap needs a per-step, persisted history that is training state. Both are derived from the one origin of the value, `TrainStepTiming.gradGlobalNorm`. Reading the cap from the monitor would make an observer drive training math (CLAUDE.md "Probe isolation"); making the monitor read the trainer's ring would change rule 9's tested behavior and split it from the offline replay. Neither is done.
- Cost: one copy and sort of at most N = 1,000 doubles per step, on the order of tens of microseconds against a ~660 ms step.

## D4. Persistence and exact resume

- **Where (OD-8).** One place: a new `__metadata__` key `trainer_grad_norm_history` in every trainer-state file, written next to the `trainer_*` schedule keys (`TrainerScheduleState.MetadataKey`, `Training/TrainerResumeState.swift:134-140`), but decoded on its own — `TrainerScheduleState.decode` requires all four schedule keys together, and a file written before this plan has the schedule but no history.
  - Files: corpus replay's rolling and enumerated trainer files, train-vs-UCI session `trainer.safetensors`, GUI session `trainer.safetensors`. Champion files and plain model files carry no trainer state and no history.
  - **Not in the lineage record.** The brief asked for the history "in the trainer state and in lineage schema 3". Storing it twice would give two sources of truth for one value. The lineage record is strict (every key required, `Persistence/LineageRecord.swift:25-28`), so a new key would also need schema 4 with schema-3 records reading it as unrecorded. The trainer file is where every other piece of optimizer state lives (velocity tensors, clock, warmup, cycle); the lineage record already carries the five new parameters through its parameter snapshot (macro-generated, Part K), which is what provenance needs. Decided (owner, 2026-10-07): as recommended.
  - Size: at most 10,000 × 2 floats, about 200 KB of header JSON per trainer file, against a 5.1M-parameter trainer file of ~40 MB (weights + velocity).
- **Snapshot.** `TrainerResumeSnapshot` (`:199-203`) gains `gradNormHistory: GradNormHistoryResumeState` (`.restored(GradientNormHistory)` / `.notInCheckpoint`), parallel to `DropoutRNGResumeState`. `exportResumeSnapshot()` fills it under the same training pause; `init(checkpoint:fileName:)` reads it from the file; `restoreExactly` restores it and logs `[RESUME] grad-norm history: restored entries=… last_trainer_step=…` or `… not in checkpoint (relative cap mode=<m>)`.
  - A restored history must end exactly at the restored clock (`last_trainer_step == completedTrainSteps`); otherwise the restore throws (the file is internally inconsistent).
- **Resume gap.** New `ResumeGap.gradNormHistory = "grad_norm_history"` (`Training/ResumeExactness.swift`): the checkpoint has no history **and** the resumed run's mode is `clip`. In `off` and `logOnly` the history does not change training math, so its absence is logged, not a gap. Corpus replay and train-vs-UCI `--resume-exact` then refuse unless `--accept-inexact grad_norm_history`; the GUI reports it NOT EXACT. Every checkpoint written before this plan (including B-silu's 18k) is history-less.
- **Parameters on resume.** The five parameters ride the lineage parameter snapshot and `session.json` like every other key (Part K). Their `absentValue`s keep a pre-feature checkpoint's resume at `mode = off`, which is what that run did.
- **Behavior fingerprint (OD-11).** `Training/BehaviorFingerprint.swift` trains one SGD step from an empty history, so the relative term never applies there and the fingerprint does not cover it. The recipe is not changed: a recipe change makes every saved fingerprint incomparable (a `build` gap on every resume across builds). The cap's math is covered by unit tests (X1) and the graph-level scale by X3.

## D5. Logging

- **Every clip event, one line (OD-13):**
  `[GRAD-CLIP] trainerStep=<s> preNorm=<%.4f> cap=<fed %.4f> decided=<%.4f> binding=hard|relative|floor applied=true|false mode=off|log_only|clip median=<%.4f|none> n=<entries> k=<k> floor=<f> hardMax=<h> ratio=<preNorm/median %.2f|none> lr=<%.3g>`
  - written when `preNorm > fedCap` (`applied=true`; any mode, so hard-max clips such as step 1's are logged too) or, in `logOnly`, when `preNorm > decidedCap` (`applied=false`);
  - one shared formatter in `Training/RelativeGradientCapLog.swift`, used by all three paths; the GUI and the CLI runners log it from the step's `TrainStepTiming` (the trainer stays free of `SessionLogger` calls for this, like the other per-step lines).
  - Volume: a cap that binds every step writes one ~250-byte line per step (about 1.3 MB per hour at 0.66 s/step). Expected volume in healthy training is near zero (E2).
- **Step lines** (`[REPLAY]` `CLI/CorpusReplayRunner.swift:2142`, `[VS-UCI]` `CLI/TrainVsUciRunner.swift:904`, GUI `[STATS]` `App/SessionController+Training.swift:1928`): append ` gNormMax=<max pre-clip norm since the previous step line> clips=<clipped steps since the previous line> gCap=<this step's fed cap>`, from `GradientNormHistory.summary(trainerSteps:)` over the steps since the previous line. This answers E1's blind spot: an unlogged step's spike shows in the next line's `gNormMax`.
- **`[HEALTH] check`** is not changed. It already reports `gradMaxRatio=` (the window's largest pre-clip norm over its reference, `Training/TrainingHealthLog.swift:186`), and the step lines above carry the clip counts. Adding clip fields there would change `TrainingHealthStepRecord`'s initializer, which four test sites call; the step lines make that unnecessary.
- **Configuration:** `[REPLAY-HPARAMS]` (`CLI/CorpusReplayRunner.swift:1185`) and `[VS-UCI-HPARAMS]` (`CLI/TrainVsUciRunner.swift:404`) append `relClip=<mode>/k<k>/N<N>/W<W>/floor<f>`; the GUI logs a `[PARAM]` line on change (the popover's existing pattern) and `[GRAD-CLIP] config …` at every Play-and-Train start.
- **`results.json`** (`CLI/CliTrainingRecorder.swift`): per row `grad_norm_max`, `grad_clip_events`, `grad_clip_cap`; in the parameters block the five new keys (Part K step 5).

## D6. Paths

- All three paths share `ChessTrainer`, `TrainerHyperparameters` (`Training/TrainerHyperparameters.swift:39,70,96,124,163`) and `restoreExactly`, so the cap, the history, its persistence and its restore are one code path. Per-path work is only the log fields and the GUI promotion rewind (the one GUI-only trainer-state move). Decided (owner, 2026-10-07): as recommended (OD-6).
- Live tunability: the five parameters are `liveTunable: true`. The GUI pushes edits to the trainer through `TrainerHyperparameters` exactly as `gradClipMaxNorm` is pushed today; the next step's decision uses them. The CLI paths read them once at start (they are not on the reconcile list in CLAUDE.md, and need not be).

---

# Part K — Parameters (5), with the full CLAUDE.md checklist

| Swift type | id | Type, default, range | `absentValue` | Why that absence |
|---|---|---|---|---|
| `RelativeGradClipMode` | `relative_grad_clip_mode` | Int, **1** (`log_only`) until P5, then **2** (`clip`); 0…2 | `.preFeature(0)` | before the feature no run had a relative cap: `off` is what it factually did |
| `RelativeGradClipMultiple` | `relative_grad_clip_multiple` | Double, **3.0**; 1.0…20.0 | `.currentSetting` | inert while the mode resolves to `off`, the same convention as the LR-cycle dependents (`Training/TrainingParameters.swift:1067-1110`) |
| `RelativeGradClipWindowSteps` | `relative_grad_clip_window_steps` | Int, **1,000**; 100…10,000 | `.currentSetting` | as above |
| `RelativeGradClipMinHistorySteps` | `relative_grad_clip_min_history_steps` | Int, **100**; 10…10,000 | `.currentSetting` | as above |
| `RelativeGradClipFloor` | `relative_grad_clip_floor` | Double, **0.5**; 0.01…100.0 | `.currentSetting` | as above |

All `category: "Optimizer"`, `liveTunable: true`. Descriptions state the formula `min(Gradient Clip Max Norm, max(floor, k × median))`, that the median is of pre-clip norms over the last N real-data steps, and that W ≤ N is required.

The checklist, walked for all five:
1. **Declare.** Five `@TrainingParameter` declarations after `GradClipMaxNorm` (`Training/TrainingParameters.swift:449-458`), each with the `absentValue` above; add to `allKeys` (`:2880` neighborhood).
2. **Singleton.** Stored properties, `collectValues` / `applyOne` entries, snapshot accessors (pattern at `:1641`, `:1809`, `:1944`, `:2059`, `:2205`). `TrainerHyperparameters` gains the five fields (`init(_:)` `Training/TrainerHyperparameters.swift:64`, `init(currentlyAppliedTo:)` `:90`, `apply(to:)`) and builds `RelativeGradientCapConfiguration` through its throwing `init`; a refusal surfaces as the configuration error at CLI start (exit 2 with the message) and as a popover validation error in the GUI.
3. **`parameters.json`.** Verify the five keys in `--show-default-parameters` and the `--create-parameters-file` → edit → reload round trip, including a file with `W > N`, which must be refused naming both ids.
4. **Session (`.dcmsession`).** Optional fields on `SessionCheckpointState` (`Persistence/SessionCheckpointFile.swift`, next to `gradClipMaxNorm` at `:299`), passed through `buildCurrentSessionState` (`App/SessionController+Checkpoint.swift:1295` neighborhood), one `resume.restore(…)` line each in `SessionParameterResume.applyGuiSession` (`App/SessionParameterResume.swift:136` neighborhood; `savedFloat:` is not needed — none is a trainer `Float`).
5. **`results.json`.** The five keys in the recorder's hyperparameter block (next to `grad_clip_max_norm`, `CLI/CliTrainingRecorder.swift:1045`), plus the per-row fields of D5.
6. **Runtime log.** `relClip=` on `[REPLAY-HPARAMS]` / `[VS-UCI-HPARAMS]`; `gCap=` / `clips=` / `gNormMax=` on every step line; `[GRAD-CLIP] config …` at GUI start; `[PARAM]` lines on GUI edits.
7. **UI.** A new `RelativeGradientCapSection` `View` in its own file under `App/UpperContentView/`, placed in the Optimizer tab directly under the "Clip:" row (`App/UpperContentView/TrainingSettingsPopover.swift:1015-1031`): a segmented Off / Log only / Clip picker and four `PopoverRow`s (k, N, W, floor). Bindings and validation entries in `TrainingSettingsPopoverModel.swift` next to `gradClipText` (`:49`, `:73`, `:477`, `:1141-1150`), including the W ≤ N pair check. The rows stay in the hierarchy whatever the mode (no `if`), dimmed when the mode is Off.
8. **Live tunability.** Consumers re-read through `TrainerHyperparameters.apply(to:)` on the existing GUI edit path; nothing caches them in a session-start snapshot inside the trainer.
9. **Renames.** None; new ids.

---

# Part T — Touch points

| # | File | Change | Phase |
|---|---|---|---|
| T1 | `Training/RelativeGradientCap.swift` (new) | D1 types | P1 |
| T2 | `Training/TrainingHealth.swift` | `TrailingReferencePolicy`; `make(_:windowStart:policy:)`; `.spikeRules` | P1 |
| T3 | `Training/ChessTrainer.swift` | D2: history, decision, feed via `BatchFeedsInput`, record next to the clock increment, `TrainStepTiming.gradientCap`, export / restore API, clear in `resetNetwork` / `loadBaseWeightsResetVelocity` | P1 |
| T4 | `Training/TrainerResumeState.swift` | `trainer_grad_norm_history` key; `GradNormHistoryResumeState`; snapshot field; `restoreExactly`; `init(checkpoint:)` | P1 |
| T5 | `Persistence/SafetensorsModelIO.swift:185-200` (the one trainer-state writer, all three paths) and its callers | write the history key next to `schedule.metadataEntries()`; refuse a non-empty history whose `last_trainer_step` differs from `schedule.completedTrainSteps`, the same rule the writer already applies to the lineage step and `training_step` | P1 |
| T6 | `Training/ResumeExactness.swift` | `ResumeGap.gradNormHistory`; the gap rule (mode `clip` and no history) | P1 |
| T7 | `Training/RelativeGradientCapLog.swift` (new) | `[GRAD-CLIP]` event and config lines, step-line field formatter | P2 |
| T8 | `Training/TrainingParameters.swift`, `Training/TrainerHyperparameters.swift` | Part K steps 1–2 | P2 |
| T9 | `CLI/CorpusReplayRunner.swift`, `CLI/TrainVsUciRunner.swift` | step-line fields, `[GRAD-CLIP]` lines, HPARAMS fields, resume gap wiring | P2 |
| T10 | `CLI/CliTrainingRecorder.swift` | results.json fields | P2 |
| T12 | `App/SessionController+Arena.swift` | capture and restore the history with the arena-start snapshot | P3 |
| T13 | `App/SessionController+Training.swift`, `App/SessionController+Checkpoint.swift`, `App/SessionParameterResume.swift`, `Persistence/SessionCheckpointFile.swift` | `[STATS]` fields and `[GRAD-CLIP]` lines, session fields and restore | P3 |
| T14 | `App/UpperContentView/RelativeGradientCapSection.swift` (new), `TrainingSettingsPopover.swift`, `TrainingSettingsPopoverModel.swift` | UI | P3 |
| T15 | `CLAUDE.md` (tag list: `[GRAD-CLIP]`), `App/CommandLineHelp.swift` (`grad_norm_history` accept token), `documentation/training-health-alarms.md` (cross-reference from rule 9), `CHANGELOG.md` | docs | P4 |
| T16 | `Training/TrainingParameters.swift` | P5: `relative_grad_clip_mode` default 1 → 2, k default per V-1/V-3 | P5 |

Verified untouched: `TrainingHealthStepRecord` and the `[HEALTH] check` line (D5), the training graph (`ChessTrainer` graph builder, `:3865` placeholder and `:4049-4058` clip math), `BehaviorFingerprint` (OD-11), `LineageRecord` schema (OD-8), the health monitor's own history (OD-9).

---

# Part X — Tests

### X1. Pure math (`DrewsChessMachineTests/RelativeGradientCapPolicyTests.swift`, new)
- Formula: for a table of (history, k, N, W, floor, hardMax) cases, `decide` returns the expected `decidedCap` and `binding` — each of hard / relative / floor binding; floor ≥ hardMax → hard; mode `off` → hard max, `binding = .hard`; `logOnly` → `fedCap == hardMax` and `decidedCap` = the rule's cap.
- Warm-up: W−1 entries → no relative term; W entries → median of W; more than N entries → only the last N count (an outlier older than N is ignored); even count → mean of the two middle values.
- Window bounds: entries are taken from `s−N … s−1` exactly (an entry at `s−N−1` is excluded).
- Pre-clip source: a history whose clipped steps recorded large pre-clip norms gives the median of the pre-clip values (a test that would fail if fed caps were used).
- Configuration: `W > N`, k ≤ 0, NaN floor, N above capacity → each refused with an error naming the ids.
- `[GRAD-CLIP]` line format: golden strings for an applied clip, a `logOnly` event and a hard-max clip with `median=none`.

### X2. History (`DrewsChessMachineTests/GradientNormHistoryTests.swift`, new)
- Append contiguity: a gap, a repeat or a non-finite norm throws; capacity drops the oldest; `summary(trainerSteps:)` max / clip count over a range.
- JSON round trip is bit-exact for 10,000 random `Float`s including subnormals and values near `Float.greatestFiniteMagnitude`; decode refuses unequal lengths, non-finite text, over-capacity, and `last_trainer_step` < count.
- Trailing-median sharing: `TrainingHealthReference.make(_:windowStart:)` equals `make(_:windowStart:policy: .spikeRules)` on the existing monitor cases (the existing tests `TrainingHealthMonitorTests.swift:148-154` stay as they are and must still pass).

### X3. Graph: the fed cap is what clips (`DrewsChessMachineTests/GradientCapGraphTests.swift`, new; tiny architecture, seeded, fp32)
- Three trainers from one init seed, one identical batch, zero initial velocity, momentum μ:
  1. cap 1e9 (never binds): velocity after one step is the raw gradient `g`, its norm `‖g‖` equals `TrainStepTiming.gradGlobalNorm`;
  2. cap `c = ‖g‖ / 4`: velocity equals `(c / ‖g‖) · g` element-wise within fp32 tolerance, and its L2 norm equals `c` within tolerance;
  3. cap `c = 2‖g‖` (does not bind): velocity and weights are **bit-identical** to trainer 1 — the property V-1 to V-3 rely on.
- A two-step test with mode `clip`, k = 1.0, W = 1, N = 1: step 2's fed cap equals step 1's pre-clip norm (the decision reached the placeholder), and `TrainStepTiming.gradientCap.fedCap` reports it.

### X4. Exact resume (`DrewsChessMachineTests/RelativeGradientCapResumeTests.swift`, new, on the `ExactResumeTests` harness, `ExactResumeTests.swift:280-330`)
- Uninterrupted: train 2K real-data steps with mode `clip`, k = 1.0, W = 2, N = 4 (k = 1 makes about half the steps clip, so the cap acts often). Interrupted: train K, `exportResumeSnapshot` → write the trainer file → read it back → fresh trainer → `restoreExactly` → train K more. Assert: the history after 2K steps is equal; the fed cap of every step K+1…2K is equal; the schedule clock is equal; weights are equal where the existing harness already asserts them (MPSGraph-deterministic configuration).
- The same with a history-less checkpoint: the resumed run reports `grad_norm_history` in its gaps in mode `clip`, not in `logOnly` / `off`; its first W steps after resume feed the hard max.
- Corpus replay and train-vs-UCI CLI resume tests (`assertCLIExactResume`, `ExactResumeTests.swift:289-299`): add the history to the round trip; a `--resume-exact` from a history-less file in mode `clip` is refused without `--accept-inexact grad_norm_history` and proceeds with it.
- GUI promotion rewind (`DrewsChessMachineTests/GuiLineageLifecycleTests.swift` style, new test file `GradientNormHistoryPromotionRewindTests.swift`): after a simulated promotion the trainer's history equals the arena-start capture, and the next step appends at the rewound clock + 1 without a discontinuity error.
- Discontinuity guard: setting `completedTrainSteps` backwards without restoring the history makes the next step throw `GradientNormHistoryError.discontinuity`.

### X5. Parameters (`DrewsChessMachineTests/RelativeGradientCapParameterTests.swift`, new)
- Each key's `absentValue` resolves as declared (a pre-feature parameter snapshot resumes at mode `off`); `parameters.json` round trip; `W > N` refused at load; session field round trip (`SessionCheckpointState` encode/decode) and `applyGuiSession` restore lines (`[RESUME-DIFF]`).

### X6. Existing tests that must change (OD-10, approved)
- `TrainStepTiming` gains a non-optional `gradientCap`, so the two test constructions must pass it: `DrewsChessMachineTests/TrainingHealthTestSupport.swift:152` and `DrewsChessMachineTests/TrainingLiveStatsGatingTests.swift:29`. The edit is mechanical (add `gradientCap: .unclipped(hardMax:)`, a named test-support constructor of a `.hard` / `.off` decision); no assertion changes.

### X7. Run scope
- Targeted classes after each phase (CLAUDE.md "Running the tests"); the full suite once at the end of P1 (trainer and persistence change) and at the end of P3, with the slow gated suites as the test plan has them.

---

# Part V — Validation

All runs use the build that contains P1–P3, a frozen app copy, `--seed 20261005`, `--policy-tail-precision fp32_from_pre_bn` and the README's flags. They share the GPU with any live runs (allowed: CLAUDE.md memory "Live training: never touch it, but don't wait on it"). Each run gets an `experiments/20261005-lr-schedule-ab/README.md` section and probes via `experiments/probe_loop.sh` (`PROBE_SEGMENT=1` for resumes).

- **V-1. Per-step measurement — the gate for k (OD-2).** `--resume-exact` from `20261005-lrBsilu-cyc1-replay-step18000.safetensors` with `parameters-B.json` plus `relative_grad_clip_mode=1` (log only) and `relative_grad_clip_multiple=1.0` (so every step above its median writes a `[GRAD-CLIP] … applied=false` line: the upper half of the per-step ratio distribution), `--accept-inexact` naming whatever gaps the refusal lists (expected: `params` as for clip1, `build` if the behavior fingerprint differs; no `grad_norm_history` gap in log-only mode), `--training-step-limit 3000` (to 21,000).
  - Check first: its `[REPLAY]` lines equal B-silu/ctl15's through 20,600. Log-only feeds the hard max, so if this build computes what build 2330 computed, V-1 is ctl15 again. If the lines differ, report the first differing step; V-1 then serves as this build's own control for V-2.
  - Report: per-step ratio p50 / p99 / p99.9 / max over 18,100–19,750 (healthy) and over B-silu's healthy span; every step with ratio > 2 in a table; the precursor's exact pre-clip norm and ratio in 19,751–19,799; 20,600's ratio.
  - Decision rule for k: the smallest of 3, 4, 5 with no healthy per-step ratio above it except isolated single steps, also reported against whether it would have clipped the precursor and 20,600.
- **V-2. k = 3 from 18k against the fixed-cap arms.** Same resume, mode `clip`, k = 3 (or V-1's k), defaults otherwise, `--accept-inexact` with `grad_norm_history` added to V-1's tokens, `--training-step-limit 5000` (to 23,000, the clip2/clip5 limit). The relative term first applies at trainer step 18,101 (W = 100 entries, from an empty history).
  - Compare with ctl15, clip1, clip2, clip5: loss, gNorm, `gNormMax`, illegal mass at every step line; every `[GRAD-CLIP]` event; policy pre-BN parked channels at the 22,000 checkpoint; pElo at 19k–23k.
  - Pass: no blowup (no `illegal_mass` or `gradient_spike` alarm after 20,000 beyond a single spike, pElo at 21k–23k within noise of clip1 / B), and the clip events listed are few and isolated.
- **V-3. Fresh start, B's recipe (owner, required).** `--replay-corpus 20260624-192615-w3aA5b`, `--start-model 20261005-r7b24-fresh.safetensors` (B's own start net), `parameters-B.json` plus mode `clip`, k = 3, W = 100, `--training-step-limit 3000`, `--enumerate-checkpoints`.
  - Report: every clip (the `[GRAD-CLIP]` lines, so the cap at each event), the fed cap trajectory from the step lines (`gCap=`) during warm-up and to 3,000, the gNorm and `gNormMax` trajectory, pElo at 1k / 2k / 3k against B's own probes (`probes-B.jsonl`).
  - The step-1 hard-max clip (27–31 → 15) is expected and is not a relative clip.
  - Report the first step whose `[REPLAY]` line differs from B's (build 2320) and whether a `[GRAD-CLIP] applied=true` relative event precedes it; a difference with no preceding relative clip is a build difference, not the cap.
  - Pass: no relative clipping in healthy early training beyond isolated spikes; pElo within noise of B at 1k / 2k / 3k.
- **V-4. Exact resume, live.** Stop V-3's recipe at 1,500 (a separate run to 1,500, then `--resume-exact` to 3,000). `[RESUME] EXACT` with `grad-norm history: restored entries=1500 last_trainer_step=1500`; every `[GRAD-CLIP]` line and every fed cap from 1,501 on equal V-3's; the trainer files at 3,000 carry identical `trainer_grad_norm_history` (header compare). Weights bit for bit only where MPSGraph is deterministic (shared GPU: not required).
- **V-5. GUI.** Build New Model → Play-and-Train with mode `clip` for 30 minutes: `[STATS]` carries `gNormMax=` / `clips=` / `gCap=`; early clip count; File ▸ Save Session, quit, resume: `[RESUME]` restores the history (no `grad_norm_history` gap); a promotion (Promote Trainee Now) rewinds the history with the clock (no discontinuity error on the next step; next `[STATS]` `gCap` consistent).
- **V-6. train-vs-UCI smoke.** A short `--train-vs-uci` run with mode `clip`, one periodic session save, `--resume-exact` from the session folder: history restored, `[VS-UCI]` lines carry the fields.
- **V-7. Default flip (P5).** After V-1 and V-3 pass, set `relative_grad_clip_mode` default to 2 and k to V-1's value; rerun `--show-default-parameters` and the targeted parameter tests.

---

# Part P — Phasing

Each phase: implement, build once (drews-xcode-mcp), run the phase's targeted tests, commit, continue (owner's standing order: no pause between approved phases).

| Phase | Content | Tests |
|---|---|---|
| P1 | T1–T6: pure core, trainer integration, persistence, resume gap | X1–X4 (trainer-level), X6, full suite once |
| P2 | T7–T10: parameters, CLI paths, recorder | X5, X4 CLI cases, recorder tests |
| P3 | T12–T14: GUI promotion rewind, session fields, `[STATS]`, popover | X4 promotion case, GUI session tests, full suite once |
| P4 | T15: docs | — |
| — | V-1 to V-6 | — |
| P5 | T16 after V-1 and V-3 pass (V-7) | X5 |

---

# Owner decisions

All decided (owner, 2026-10-07: "ok approved keep it going"; coordinator: record each as decided as recommended).

- **OD-1 Default mode.** Recommended: ship with `log_only` (1), so every run starts collecting per-step evidence without changing training math and no existing resume-gap test changes; flip the default to `clip` (2) in P5 once V-1 and V-3 pass. **Decided (owner, 2026-10-07): as recommended.**
- **OD-2 Default k.** Recommended: 3 (E2: the healthy logged maximum is 1.47; B-silu's 20,600 is 7.15). The final value is gated on V-1's per-step measurement, with 4 or 5 as fallbacks by V-1's decision rule. Note: the precursor in 19,751–19,799 is only known to be above 2.8×, so k = 3 may not clip it; V-1 measures it and V-2 shows whether clipping 20,600 alone is enough. **Decided (owner, 2026-10-07): as recommended.**
- **OD-3 N.** Recommended: 1,000 — equal to the `gradient_spike` reference span, one tenth of B's LR period, one momentum period; N = 5,000 raised the healthy maximum ratio from 1.47 to 2.00 (E2). **Decided (owner, 2026-10-07): as recommended.**
- **OD-4 Warm-up W.** Recommended: 100 entries, the median of the available history until the window fills (owner direction); the hard max covers the steps before; an exact resume restores the history (no second warm-up). **Decided (owner, 2026-10-07): as recommended.**
- **OD-5 Floor.** Recommended: a first-class parameter, default 0.5 (R3: never binds in a healthy run at k ≥ 2.5; holds a collapsed run's cap at 0.5 instead of 0.04). **Decided (owner, 2026-10-07): as recommended.**
- **OD-6 GUI and train-vs-UCI.** Recommended: yes, one shared code path inside `ChessTrainer` (D6). **Decided (owner, 2026-10-07): as recommended.**
- **OD-7 Mode instead of an enable flag.** Recommended: `off` / `log_only` / `clip` in one parameter, because log-only is needed for V-1 and for OD-1, and two booleans would interact. **Decided (owner, 2026-10-07): as recommended.**
- **OD-8 Where the history is saved.** Recommended: only in the trainer-state file's `trainer_grad_norm_history` key, not also in the lineage record (D4: one source of truth; lineage would need schema 4; the parameters already ride the lineage parameter snapshot). This narrows the brief's "and in lineage schema 3". **Decided (owner, 2026-10-07): as recommended.**
- **OD-9 Shared median.** Recommended: share the definition (`TrainingHealthReference.make` with a policy) and keep two histories derived from the one value: the monitor's (observer, per-generation, offline-replayable) and the trainer's (training state, persisted) (D3). **Decided (owner, 2026-10-07): as recommended.**
- **OD-10 Test edits.** Recommended: approve the mechanical addition of the new `gradientCap` argument at `TrainingHealthTestSupport.swift:152` and `TrainingLiveStatsGatingTests.swift:29` (X6); no assertion changes. **Decided (owner, 2026-10-07): as recommended.**
- **OD-11 Behavior fingerprint.** Recommended: leave the recipe unchanged (D4); the cap is covered by X1–X4. **Decided (owner, 2026-10-07): as recommended.**
- **OD-12 GUI gNorm chart.** Recommended: keep the chart's reference line at the hard max for now; plotting the per-step fed cap needs chart data plumbing and is a follow-up (Non-goals). **Decided (owner, 2026-10-07): as recommended.**
- **OD-13 Every clip event logged.** Recommended: one `[GRAD-CLIP]` line per clipped step in every mode, hard-max clips included (D5; volume bounded at ~1.3 MB/hour in the worst case). **Decided (owner, 2026-10-07): as recommended.**
- **OD-14 Floor at or above the hard max.** Recommended: allowed, logged once as inert, not refused (refusing would couple validation of the existing live-tunable hard max to a new parameter). **Decided (owner, 2026-10-07): as recommended.**
- **OD-15 Fresh-start validation (V-3).** Recommended: required before P5, with the owner's report and pass criteria. **Decided (owner, 2026-10-07): as recommended.**

---

# Risks

| Risk | Mitigation |
|---|---|
| k too tight per step: healthy steps clipped, training slowed | log-only default (OD-1); V-1 measures the per-step spread; V-3 checks early training; every clip is a log line |
| k too loose: misses the precursor (> 2.8×, exact value unknown) | the hard max still applies; V-1 measures the precursor; V-2 shows whether clipping 20,600 alone prevents the blowup; fallbacks are tighter fixed caps |
| A sustained rise (> half the window) loosens the relative cap | intended for a genuine scale shift; hard max bounds a runaway; rules 5 and 9 still report |
| Median lag across the LR cycle | measured: ≤ 1.4× across half a period, ≤ ~1.1× within N = 1,000 (E3) |
| Clip spiral | the median uses pre-clip norms (R4) |
| Clipping a collapsing net harder | the floor (R3) |
| History and clock diverge (a new code path moves the clock without the history) | contiguity check throws on the next step (D2, X4) — loud, never mixed |
| Exact-resume regressions | X4 round trips on all three paths; V-4 live |
| GUI promotion rewind forgets the history | T12, X4 promotion test, V-5 |
| Header growth (~200 KB per trainer file) | negligible against ~40 MB files; header-scanning tools read it once |
| Fingerprint does not cover the cap | X1–X4; OD-11 |
| Bit-identity claim for unclipped steps is wrong on some GPU path | X3 asserts it; V-1's line equality checks it at scale |

---

# Non-goals

- An update-norm cap (`lr × g` relative to its median): the gradient-norm cap already scales with LR through its median (E3); revisit only if V-2 fails.
- Per-layer or per-tensor clipping; adaptive clipping by parameter norm (AGC).
- A clip-rate alarm rule in `TrainingHealthEvaluator`; the step lines' `clips=` counts and the `[GRAD-CLIP]` lines (D5) are the data for deciding one later.
- Plotting the fed cap on the GUI gNorm chart (OD-12).
- Any change to the lineage schema (OD-8) or the behavior fingerprint recipe (OD-11).
- Any change to rule 9's thresholds or its own history (OD-9).

---

# Review (2026-10-07, against `main` at `9e344de2`)

A hard re-check of every claim against the code and the logs. Must-fixes found and fixed in this file:

1. **Wrong decision site.** The first draft put the cap decision next to `nextStep` at `Training/ChessTrainer.swift:4683`, which is phase 1 (sampling), a different `enqueue` block from the step itself. Fixed: the decision is made in phase 3 immediately before its `buildFeeds` call (`:4931`), so decision, feed, run, readback, record and clock increment are one block on one queue (D2).
2. **Nonexistent function name.** `exportTrainerWeightsWithClock` does not exist; the function is `exportWeightsWithCompletedSteps` (`:4414`). Fixed.
3. **Wrong `TrainerHyperparameters` initializers.** They are `init(_:)` (`:64`) and `init(currentlyAppliedTo:)` (`:90`), not `init(parameters:)` / `init(trainer:)`. Fixed (Part K step 2).
4. **Unnamed writer.** T5 named "the `CheckpointManager` trainer-state write path"; the one trainer-state writer is `Persistence/SafetensorsModelIO.swift:185-200`, which already refuses a lineage step or `training_step` that disagrees with the schedule clock. Fixed: the history is written there and gets the same clock check.
5. **Hidden test breakage.** Adding clip fields to `[HEALTH] check` would have changed `TrainingHealthStepRecord`'s initializer, which the tests call in four places, without an OD. The brief allows the step line *or* `[HEALTH] check`; the step lines carry `gNormMax=` / `clips=` / `gCap=` and the check line already has `gradMaxRatio=`. Fixed: `[HEALTH] check` and `TrainingHealthStepRecord` are untouched; T11 removed.
6. **Wrong history size in V-4.** The history keeps up to 10,000 entries, so at trainer step 1,500 it holds 1,500, not 1,000. Fixed.
7. **Unmeasured number stated as measured.** "Within any 1,000-step window at most about 1.1×" (E3) was an estimate. Fixed: stated as an estimate, with the measured N = 1,000 ratios (E2) as the evidence.
8. **Reproduce pointed at a nonexistent path.** E5 named a script path not in the repository. Fixed: the method is stated exactly, and the script is added with V-1's write-up.
9. **First relative step.** "Starts at 18,100" was off by one: with W = 100 from an empty history, trainer step 18,101 is the first with 100 entries. Fixed.
10. **Wrong range accessor.** `declaredRange` → `declaredClosedRange` (the macro's accessor, as used for `GradClipMaxNorm` at `App/UpperContentView/TrainingSettingsPopover.swift:1028`). Fixed.

Checked and correct as written:
- The cap is already fed per step through a scalar placeholder (`:1693`, written at `:6249` from `gradClipMaxNorm`), so no graph rebuild is needed; the clip math is `cap / max(norm, cap)` in fp32 (`:4049-4058`), which makes an unclipped step bit-identical for any cap (X3 pins it).
- The pre-clip norm is read back every step and checked finite before the clock increment (`:7110-7160`, `:4976`); the synthetic `trainStep(batchSize:)` path does not advance the clock (`:4959-4975`), and the plan keeps it out of the history.
- The `gradient_spike` reference (`TrainingHealthReference.make`, `Training/TrainingHealth.swift:823-839`; constants `:485-490`) takes `(trainerStep, value)` pairs and a window start, so a policy argument is enough to share it; its existing two-argument call sites (`:914-917`, `TrainingHealthMonitorTests.swift:148-154`) keep compiling unchanged.
- The monitor's history (`Training/TrainingHealthMonitor.swift:161-162`) is filled at evaluation time and cleared on rewinds (`:300-304`), so it cannot serve as the per-step, persisted history the cap needs (OD-9).
- `TrainerScheduleState.decode` requires all four schedule keys together (`Training/TrainerResumeState.swift:159-188`), so the history key must be decoded separately (D4).
- The GUI promotion rewind restores weights, velocity, clock and dropout state from the arena-start snapshot (`App/SessionController+Arena.swift:200-215`, `:493-504`); the history must join that snapshot (T12).
- The lineage record requires every key on decode (`Persistence/LineageRecord.swift:25-28`), so a history key there would need schema 4 (OD-8).
- `absentValue` choices follow the LR-cycle precedent: the switch `.preFeature(off)`, its dependents `.currentSetting` (`Training/TrainingParameters.swift:1056-1110`); an Int-coded mode follows `PolicyLabelSmoothingModeParameter` (`:404-414`).
- Every number in E1–E4 was re-measured from the named logs or taken from the README section cited, and agrees with the owner's simulation where they overlap (largest ratios B 1.40, B-leakyall 1.38, B-silu 7.15, clip1 1.32; steps over 3× 0 / 0 / 1 / 0).

Open, not must-fix:
- The precursor's ratio (> 2.8×) is unknown, so whether k = 3 clips it is unknown until V-1 (OD-2 states it).
- Rule 9's own history still empties at a non-exact process start while the cap's history persists; the two can therefore differ in the first 200 steps after a resume. Intended (OD-9).

