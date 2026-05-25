# Comment Audit — DrewsChessMachine

A categorized punch list of correctness and fragility issues across the project's Swift `//` and `/* */` comments. Generated from a parallel sweep of every `.swift` file under `DrewsChessMachine/DrewsChessMachine/` (excluding `BuildInfo.swift`, `DrewsChessMachineTests/`, and `Packages/`). 13 partition agents read every file completely; this report is the merged output.

**This file is the deliverable. No source files were modified.** Triage at your pace.

## Scope

- **In:** every Swift file under `DrewsChessMachine/DrewsChessMachine/` — App/ (including the full `App/UpperContentView/` subtree), Training/, Arena/, Network/, Persistence/, Encoding/, Chess/, Views/, Charts/, CLI/, Logging/, Utils/.
- **Out:** `BuildInfo.swift` (auto-generated), `DrewsChessMachineTests/`, `Packages/` (the local SwiftPM macro package), and all markdown design docs (`chess-engine-design.md`, `sampling-parameters.md`, `mpsgraph-primitives.md`, `ROADMAP.md`, `CHANGELOG.md`, `*_PLAN.md`).
- **Total files audited:** ~179 across 13 partitions; ~17,000 comment lines.

## Categories and severities

| Category | Severity | What it catches |
|---|---|---|
| WRONG | Critical | Comment makes a factual claim the current code contradicts. |
| DRIFTED-REF | Critical | Names a symbol, function, file, or constant that's been renamed/moved/deleted. |
| DEAD-RETROSPECTIVE | High | Describes a prior state of the code ("previously", "used to", "was changed from"). |
| FRAGILE-LITERAL | Medium | Quotes a numeric value that's also defined as a nearby constant — two sources of truth. |
| DETACHED-LITERAL | Medium | Quotes a numeric value with no corresponding constant; will rot silently. |
| STALE-DATE | Medium | Inline calendar date (e.g., "2026-04-15 incident") that may no longer apply. |
| WHAT-COMMENT | Low | Pure restatement of a self-explanatory identifier. |
| MARK-LITERAL | Low | `// MARK:` lines with numeric counts (e.g., "39 parameter keys"). |
| OK-LOAD-BEARING | Info | Multi-paragraph design block (≥6 lines) verified accurate; listed so it isn't accidentally trimmed. |

## Summary counts

| Category | Count |
|---|---|
| WRONG (Critical) | 27 |
| DRIFTED-REF (Critical) | 22 |
| DEAD-RETROSPECTIVE (High) | 39 |
| FRAGILE-LITERAL (Medium) | 49 |
| DETACHED-LITERAL (Medium) | 24 |
| STALE-DATE (Medium) | 3 |
| WHAT-COMMENT (Low) | 7 |
| MARK-LITERAL (Low) | 1 |
| **Total actionable findings** | **172** |
| OK-LOAD-BEARING (Info, preserve) | ~140 |

## Cross-cutting patterns

A few patterns showed up in *multiple* partitions; worth fixing in one sweep:

1. **`inputPlanes: 20 → 30` propagation never finished.** When the v2→v3 architecture refresh added the 10 temporal-repetition history planes (planes 20–29), three comment sites kept v2 arithmetic:
   - `Network/ChessNetwork.swift:59–62` — class-level "Architecture v2" summary still describes "20×8×8 board tensor" while the live `inputPlanes = 30`.
   - `Network/ChessNetwork.swift:578` and `Network/ChessMPSNetwork.swift:212` — both say "`inputPlanes`×8×8 = 1,280 floats". Correct value is **1,920** (= 30 × 64).
   - `Persistence/ModelCheckpointFile.swift:156` — stem-conv illustration says "`inputPlanes × channels × 9 = 23,040`". Correct value is **34,560** (= 30 × 128 × 9). The runtime arithmetic on line 168 uses the live `inputPlanes` so the cap itself is still correct.

2. **Arena cadence and tournament-size literals are stale across at least 8 sites.** The defaults moved (or moved into `TrainingParameters`), but several comments still quote the old hard-coded numbers:
   - "200 games" → actual default is `TrainingParameters.shared.arenaGamesPerTournament = 400`. Stale in `SessionController+Arena.swift:23,70`, `SessionController+Training.swift:24`, and `TournamentRecord.swift:25` (which also names a non-existent symbol `Self.tournamentGames`).
   - "0.55 threshold" → actual default `arenaPromoteThreshold = 0.53`. Stale in `SessionController+Arena.swift:28`. (CLAUDE.md is also stale on this one.)
   - "30-minute auto-fire" → actual default `arenaAutoIntervalSec = 900.0` (= 15 minutes), and live-tunable. Stale in `SessionController+Training.swift:22,1163,1297` and `ArenaTriggerBox.swift:5,163`.

3. **`ReplayBuffer.fileVersion` migrated v4 → v7, but five field doc-blocks still say "not persisted in v4".** `Training/ReplayBuffer.swift:56,61,66,72,78` all carry the same stale tail. Sections were added in v5 (`plyIndices`, `gameLengths`, `samplingTaus`, `stateHashes`, `workerGameIds`) and v6 (`materialCounts`); current file version is 7. Same partition's "four sections" / "four section writes" claims at lines 1829–1831 and 1903–1905 are the same v4-era drift — the file now writes/reads nine column sections.

4. **"Engine menu" doesn't exist; the menu is "Train".** Comments in `AppCommandHub.swift:93` and (out-of-partition references) several files say "Engine ▸ Promote Trainee Now" but the actual menu name in `DrewsChessMachineApp.swift:396` is `CommandMenu("Train")`. Looks like a planned rename never landed.

5. **Two unrelated CLAUDE.md drift items surfaced during the audit.** Reporting for visibility only (CLAUDE.md is out of scope for edits):
   - **`BatchedSelfPlayDriver`'s `stopAll` and `expectedSlotCount`** are referenced in CLAUDE.md but no longer exist in the source — the file's own `History` block explicitly calls out the move away from that design.
   - **`arenaPromoteThreshold` "default 0.55"** in CLAUDE.md doesn't match the live default of `0.53`.

---

# Critical findings

## WRONG (27)

### App/SessionController.swift:406–417
> Subsystem implementations live in extension files:
>   • batch-size sweep actions       — SessionController+Sweep.swift
>   • engine diagnostics             — SessionController+Diagnostics.swift
>   • arena tournament + log-recovery — SessionController+Arena.swift
>   • candidate-probe + inference     — SessionController+CandidateProbe.swift
>   • checkpoint save/load/snapshot   — SessionController+Checkpoint.swift
>   • Play-and-Train orchestration    — SessionController+Training.swift
> The heartbeat (processSnapshotTimerTick / __processSnapshotTimerTick /
> periodicSaveTick / refreshChartZoomTick / refresh{Memory,TrainingChart,
> ProgressRate,Usage}IfNeeded) is still below in this file.

**Code context:** Extensions table of contents in `SessionController` class body.
**Issue:** Wrong on two counts. (1) Omits two extension files that exist on disk: `SessionController+ManualPromote.swift` and `SessionController+TacticalProbe.swift`. (2) Claims the heartbeat is "still below in this file" — but a different comment block at lines 467–470 of the same file (and `SessionController+Heartbeat.swift` on disk) shows the heartbeat moved out. The two claims in this file contradict each other.
**Suggested fix:** Add lines for `SessionController+ManualPromote.swift` and `SessionController+TacticalProbe.swift`, and rewrite the "still below in this file" claim to point at `SessionController+Heartbeat.swift`.

### App/SessionController.swift:798–802
> Network built in \(String(format: "%.1f", net.buildTimeMs)) ms
> ID: \(idStr)
> Parameters: ~2,400,000 (~2.4M)
> Architecture: 20x8x8 -> stem(128)
>   -> 8 res+SE blocks -> policy(4864) + value(1)

**Code context:** `buildNetwork()` networkStatus user-visible string literal.
**Issue:** Architecture string embeds the obsolete `20x8x8` (now `30x8x8`) AND `value(1)` (now W/D/L 3-logit head per the post-2026-05-12 WDL rewrite). User-facing.
**Suggested fix:** Wire to `ChessNetwork.inputPlanes` / `policySize` and change `value(1)` → `value(3 W/D/L)`.

### App/SessionController+Arena.swift:23
> Run one arena tournament in parallel mode — 200 games between

**Code context:** `runArenaParallel(...)` doc comment.
**Issue:** Claims "200 games" but the actual game count comes from `TrainingParameters.shared.arenaGamesPerTournament`, default **400** (`Training/TrainingParameters.swift`, `ArenaGamesPerTournament` default 400; read live at line 785).
**Suggested fix:** "a configurable number of games (default 400)" or just "the configured number of games".

### App/SessionController+Arena.swift:28
> Promotes the candidate into the real champion iff the score
> meets the 0.55 threshold.

**Code context:** `runArenaParallel(...)` doc comment.
**Issue:** Claims "0.55" threshold but actual default for `ArenaPromoteThreshold` is **0.53**. Live-read at line 264. CLAUDE.md is also stale on this.
**Suggested fix:** "meets the configured `arenaPromoteThreshold` (default 0.53)" — or drop the literal.

### App/SessionController+Arena.swift:70
> can continue through the 200 games.

**Code context:** Inside `runArenaParallel`, trainer-snapshot section.
**Issue:** Same stale "200 games" as line 23.
**Suggested fix:** Drop the literal — "the tournament's games" or "the full tournament".

### App/SessionController+Training.swift:22
> (either by the 30-minute auto-fire or the Run Arena button),

**Code context:** `startRealTraining` doc comment.
**Issue:** "30-minute auto-fire" — actual `arenaAutoIntervalSec` default is **900.0 s = 15 min**. Live-read each iteration of the training-worker loop.
**Suggested fix:** "either by the configurable auto-fire interval or by the Run Arena button" — no literal.

### App/SessionController+Training.swift:24
> then runs 200 games between the candidate inference network

**Code context:** `startRealTraining` doc comment.
**Issue:** Same stale "200 games".
**Suggested fix:** "the configured number of games".

### App/SessionController+Training.swift:1163–1165
> // candidate probe at its own 15 s cadence between
> // steps, and nudges the arena trigger box when the
> // 30 min auto cadence elapses.

**Code context:** TaskGroup comment introducing the training-worker task.
**Issue:** "15 s" matches `candidateProbeIntervalSec = 15` (currently right but tunable). "30 min" is wrong — `arenaAutoIntervalSec = 900` (= 15 min), tunable.
**Suggested fix:** Use the configured-parameter names.

### App/SessionController+Training.swift:1297
> // execution. Both the 30-minute auto-fire and the

**Code context:** Arena-coordinator task introduction.
**Issue:** Same stale "30-minute".
**Suggested fix:** "Both the auto-fire and the Run Arena button enter here via `triggerBox.trigger()`."

### App/AppCommandHub.swift:93
> /// Engine menu > Promote Trainee Now — promote the *current* trainer

**Code context:** `var promoteTrainerNow: () -> Void = {}` declaration.
**Issue:** No "Engine" menu in this app. Actual menu is **Train** (`DrewsChessMachineApp.swift:396` `CommandMenu("Train")`; the button is at `:425`). Same misnomer appears in `SessionController+Arena.swift:248`, `UpperContentView.swift:671/2568`, `ArenaOverrideBox.swift:10` — a planned rename never landed.
**Suggested fix:** Sweep "Engine ▸" → "Train ▸" across the codebase.

### App/LowerContentView.swift:16–21
> /// Promotion threshold drawn as a horizontal reference line on
> /// the arena-activity chart. Lives on `UpperContentView` (it
> /// is a tunable training parameter), forwarded here as a
> /// `let` so the chart grid stays decoupled from
> /// `TrainingParameters`.

**Code context:** `let promoteThreshold: Double` property of `LowerContentView`.
**Issue:** Comment says the value "Lives on `UpperContentView`" but actual source of truth is `TrainingParameters.shared.arenaPromoteThreshold` — `ContentView.swift:114` forwards directly from `TrainingParameters` (not via `UpperContentView`). The "decoupled from `TrainingParameters`" parenthetical is also wrong: the forwarding is *from* `TrainingParameters` to the chart grid.
**Suggested fix:** "Sourced from `TrainingParameters.shared.arenaPromoteThreshold`, forwarded here as a `let` so the chart grid stays decoupled from the singleton."

### App/UpperContentView/UpperContentView.swift:456
> Upper bound on the adjustable training-step delay. 500 ms
> already turns a ~60 steps/s training worker into roughly
> 2 steps/s, which is as slow as anyone reasonably wants to
> crawl the learning rate while still making progress.

**Code context:** `nonisolated static let stepDelayMaxMs: Int = 3000`.
**Issue:** Cited "500 ms" upper-bound rationale is 6× smaller than the actual 3000 ms ceiling — the steps/s calculus no longer matches.
**Suggested fix:** Update to 3000 ms, or rephrase the rationale around the current ceiling.

### App/UpperContentView/UpperContentView.swift:468
> ...with N
> workers, a 500 ms rung removes roughly N × 2 games/sec of
> aggregate production, which is usually more than enough to
> bring the ratio back. 2000 ms is the ceiling so a runaway
> auto-adjust can't stall the session outright.

**Code context:** `nonisolated static let selfPlayDelayMaxMs: Int = 3000`.
**Issue:** "2000 ms is the ceiling" — actual is 3000 ms.
**Suggested fix:** Update the ceiling to 3000 ms.

### App/UpperContentView/TrainingSettingsPopover.swift:1641 (and again at 1903)
> ...would push the row over the popover's 540 pt width budget.
> ... blow out the popover's fixed 540 pt width.

**Code context:** `.frame(width: 616)` (line 291).
**Issue:** Two comments cite a "540 pt" popover-width budget against the actual `.frame(width: 616)` declared in the same file.
**Suggested fix:** Replace with 616, or factor into a `static let popoverWidth` and reference by name.

### App/UpperContentView/UpperContentView.swift:546–549 (DEAD-RETROSPECTIVE; flagged here too because the dead-list is also factually stale)
> chart-related `@State` that used to live here — the rings,
> decimated frame, scroll position, hover position, zoom
> state, arena events, diversity bars — moved onto the
> coordinator.

(See DEAD-RETROSPECTIVE section.)

### Training/ChessTrainer.swift:1013
> `v_new = v - lr * (clipped_grad + weightDecayC * v)`,
> i.e. `(1 - lr*c) * v - lr * clipped_grad`. Decay is applied only to

**Code context:** `static let weightDecayCDefault: Float = 1e-4` doc comment.
**Issue:** Two-level formula bug. (a) Conflates velocity `v` with the weight (decoupled-decay term `weightDecayC * weight` is multiplied by `lr` and subtracted from the **weight**, not the velocity — see actual ops at 3296–3334). (b) Within the same comment, "v_new = v - lr * …" reads as a weight update masquerading as a velocity update. The corrected sub-formula at line 1014 only makes sense if `v` means the weight, which is then inconsistent with naming `v` and `clipped_grad` separately.
**Suggested fix:** Replace with the actual update under μ=0: `weight_new = (1 − lr·weightDecayC) · weight − lr · clipped_grad`. Or reuse the canonical block at 1159–1161 / 3176–3177 (which is correct).

### Training/ChessTrainer.swift:1029
> `maxNorm / globalNorm` so the effective step is capped. 5.0 is
> a conservative value that sits well above steady-state norms

**Code context:** `static let gradClipMaxNormDefault: Float = 30.0`.
**Issue:** Quotes "5.0" but the constant on the next line is `30.0`. 6× discrepancy — a reader reasoning about clip frequency will be misled.
**Suggested fix:** Update to "30.0" (or drop the literal and reference the constant name).

### Training/ChessTrainer.swift:2937–2938
> // frac(|A| < 0.05): "near-zero-signal" positions whose
> // policy-gradient contribution is tiny. Threshold 0.05
> // picked to match the default `drawPenalty`

**Code context:** `let smallThreshold = graph.constant(0.05, dataType: dtype)`.
**Issue:** Claims 0.05 matches the default `drawPenalty`, but actual default `drawPenalty: Float = 0.1` (init signature at line 1519). Differ by 2×; justification is wrong.
**Suggested fix:** Pick a justification that's actually true ("half the default drawPenalty"), or describe the threshold's intent in calibration-independent terms.

### Training/TrainingParameters.swift:467 (MaxPliesFromAnyOneGame description)
> default: 10. Range capped at 400 — with typical batch sizes (e.g. 4096) the cap is always active.

**Code context:** `MaxPliesFromAnyOneGame` parameter, `default: 10`, `range: 1...400`.
**Issue:** "Cap is always active" depends on the *value*, not the *range*. With cap at default 10, true. With cap near the range max (400), the cap essentially never binds at batch=4096. Reads as if range-max is the operative comparison.
**Suggested fix:** "At the default (10) and typical batch sizes, the cap is essentially always active for long games. Setting near the range max (400) effectively disables it."

### Training/ReplayBuffer.swift:75–78
> /// Per-position 32-bit packed identity: high 8 bits = `workerId`,
> /// low 24 bits = `intraWorkerGameIndex`. Broadcast across every
> /// row of a single appended game. Same ring index as `boardStorage`.
> /// Observability-only; not persisted in v4.

**Code context:** `private let workerGameIdStorage: UnsafeMutablePointer<UInt32>`.
**Issue:** Two drifts in one block.
  1. "high 8 bits / low 24 bits" contradicts the actual `packWorkerGameId` (lines 297–298): `(UInt32(workerId) << 16) | (gameIndex & 0x0000_FFFF)` — that's 16/16. The matching docstring on `packWorkerGameId` at lines 286–288 itself correctly says 16/16. Misleads anyone reasoning about whether `intraWorkerGameIndex` overflow can alias across workers (it can — at 65,536 games per worker, not 16M).
  2. "not persisted in v4" — this field IS persisted starting in v5. Current file format is v7.

**Suggested fix:** "Per-position 32-bit packed identity: high 16 bits = `workerId`, low 16 bits = `intraWorkerGameIndex` (masked). Broadcast across every row of a single appended game. Persisted in v5+."

### Training/BatchedSelfPlayDriver.swift:124–125
> /// Per-tick sampler scratches (one slice per game per pass).
> /// Sized `tickScratchCapK × MoveSampler.scratchCapacity` floats
> /// each. Game `i`'s sample uses bytes `i*256..<(i+1)*256` —

**Code context:** `MoveSampler.scratchCapacity = 256` (verified at `Network/MoveSampler.swift:50`); the pointer is `UnsafeMutablePointer<Float>` and the slicing at lines 375–382 advances by *floats*, not bytes.
**Issue:** Says "bytes" but means "floats". Byte offsets would be `i*1024..<(i+1)*1024`. Self-contradicting (the preceding sentence uses floats correctly).
**Suggested fix:** Change "bytes" to "floats" or just drop the example since the per-slot size is stated above.

### Network/ChessMPSNetwork.swift:212
> /// - Parameter board: `inputPlanes`×8×8 = 1,280 floats (from `BoardEncoder.encode`).

**Code context:** `evaluate(board:consume:)` doc.
**Issue:** Wrong arithmetic. With `inputPlanes = 30`, the product is **1,920**, not 1,280. 1,280 = 20 × 64 is the pre-refresh v2 value.
**Suggested fix:** Either drop the explicit "= 1,280" or write "= 1,920". Better: drop the literal and let the formula stand.

### Network/ChessNetwork.swift:578
> /// - Parameter board: `inputPlanes`×8×8 = 1,280 floats in NCHW order (planes, rows, cols).

**Code context:** Batch-eval entry point doc.
**Issue:** Identical bug to ChessMPSNetwork.swift:212. Should be 1,920.
**Suggested fix:** Drop the literal or write 1,920.

### Network/MoveSampler.swift:26–28
> /// - `ply` is the side-relative ply index (0 for this player's first
> ///   move, 1 for their second, etc.) — drives both the tau schedule
> ///   and the Dirichlet ply-limit check.

**Code context:** Header doc on the sampler's `ply` parameter.
**Issue:** Wrong: parameter is **game-total ply**, not side-relative. The same file's `sampleMove` `- parameter ply:` doc at lines 64–69 ("Game-total ply count of the position being sampled, 0-indexed half-move count from the start") is correct, as is the caller at `MPSChessPlayer.swift:417` which passes `2 * gamePliesRecorded + (isWhite ? 0 : 1)`. Under the wrong reading, tau and Dirichlet ply-limit get applied at half the expected rate.
**Suggested fix:** "`ply` is the game-total (half-move) ply index" — make this doc match the parameter-level doc 40 lines down.

### Persistence/ModelCheckpointFile.swift:153–161
> /// Current largest tensors at the post-refresh architecture:
> /// - residual conv weights: `channels × channels × 9 = 147,456`
> /// - stem conv: `inputPlanes × channels × 9 = 23,040`
> /// - policy 1×1 conv: `channels × policyChannels = 9,728`
> /// - SE FC: `channels × (channels / r) = 4,096`

**Code context:** `maxTensorElementCount` doc.
**Issue:** Stem-conv illustration uses obsolete `inputPlanes = 20`. With `inputPlanes = 30`, stem is `30 × 128 × 9 = 34,560`. The runtime arithmetic at line 168 reads live `inputPlanes`, so the cap is still correct — only the doc-comment literal is stale.
**Suggested fix:** Change "23,040" to "34,560".

### Views/Charts/TrainingChartGridView/WDLProbabilityChart.swift:6–7
> /// 0.75 dashed reference marks the draw-bias init — `pD` trending
> /// up to and staying there is the regression-toward-collapse signal.

**Code context:** Class doc.
**Issue:** Grammatically ambiguous in a load-bearing way: "there" naturally refers to "0.75", but 0.75 is the **init** value of pD (correctly described elsewhere as softmax of `[0, ln 6, 0]`). The actual collapse signal — stated correctly two paragraphs down at line 22 (`"pD → 1 is the regression-toward-collapse signal"`) and in CLAUDE.md — is `pD → 1.0`. The class doc reads as if 0.75 itself is the collapse endpoint.
**Suggested fix:** "pD trending up toward 1.0 and staying there is the regression-toward-collapse signal; the 0.75 line is just the init reference so post-init movement is legible."

### CLI/CliTrainingConfig.swift:11–15
> /// Unknown keys are tolerated quietly so older and newer parameter
> /// files can coexist; a typo in a recognized id surfaces as an
> /// `unknownParameter` error from `TrainingParameters.apply` later,
> /// since this loader does not pre-validate ids — it just collects
> /// them.

**Code context:** Loader header.
**Issue:** The "tolerated quietly" claim contradicts actual behavior. `TrainingParameters.apply` (lines 820–914) routes every id through a switch with `default: throw TrainingConfigError.unknownParameter(id:)`, so any unknown key throws at apply time. The loader doesn't filter — it passes everything through. A file written by a newer build adding a new key, loaded by an older build, will fail the run. So the design is strict, not forward/backward-tolerant.
**Suggested fix:** Either (a) actually filter unknowns against `TrainingParameters.shared.knownIDs()` in the loader (or soft-skip in apply), or (b) rewrite the comment to describe the actual strict pass-through behavior.

### Utils/DiagSampler.swift:68–72 (and 55–58)
> // VM_MEMORY_IOACCELERATOR is 273. The macro lives in
> // <mach/vm_statistics.h>; reproducing the literal here avoids
> // an import dance for one constant.
> let ioAcceleratorTag: UInt32 = 273

**Code context:** GPU mapping tag.
**Issue:** The SDK header (`MacOSX26.5.sdk/usr/include/mach/vm_statistics.h:642`) defines `#define VM_MEMORY_IOACCELERATOR 100`, not 273. The code uses `273` directly so the comment matches the code's behavior, but it misnames that literal as the macro's value. Either the tag actually observed at runtime via `vm_region_submap_info_data_64_t.user_tag` is `273` (a privately-assigned IOKit tag distinct from the macro), in which case the comment shouldn't equate the literal with the macro; or the literal is wrong and should be `100`.
**Suggested fix:** Drop the "VM_MEMORY_IOACCELERATOR is 273" identity claim. Describe the literal as the observed runtime tag for AGX mappings on Apple Silicon, distinct from the macro.

### Chess/ChessGameEngine.swift:98–101
> and the engine rejects anything illegal with
> `ChessGameError.illegalMove` instead of trusting the caller and
> potentially trapping inside `MoveGenerator.applyMove`'s force unwrap.

**Code context:** Class-level doc.
**Issue:** `MoveGenerator.applyMove` no longer has a force unwrap. The source-square retrieval is now `guard let piece = board[fromIndex] else { preconditionFailure(...) }` (`MoveGenerator.swift:73–77`). Trap behavior is the same (hard crash on bad caller), but the phrase "force unwrap" is wrong and misleading to anyone grepping for `!`-shaped issues.
**Suggested fix:** Replace "force unwrap" with "`preconditionFailure`" / "trap" / "hard crash", or delete the implementation-detail half of the sentence.

---

## DRIFTED-REF (22)

### App/SessionController+Training.swift:1130
> // function (lines ~7694-7703); should be non-nil here but

**Code context:** Inside `startRealTraining`, near `probeInferenceForProbes` capture.
**Issue:** Line numbers don't exist — this file is 2,320 lines total. Leftover from when the code lived inside `UpperContentView.swift` (and even there the count no longer matches; UpperContentView.swift is now 3,425 lines).
**Suggested fix:** Drop the line numbers — refer to "the candidate/probe/arena setup block above".

### App/SessionController+Training.swift:954
> // reset, so warmup picks up mid-session as the line-287 seed

**Code context:** Inside `startRealTraining`, after the trainer-network reset.
**Issue:** "line-287 seed" — line 287 in this file is `TrainingParameters.shared.selfPlayDrawKeepFraction = v`, not the warmup seed. Stale from a prior file shape.
**Suggested fix:** Drop the line number — say "the earlier seed of `trainer.completedTrainSteps` from `rs.trainingSteps`".

### App/LowerContentView.swift:22–25
> /// Target replay ratio rendered as a dashed horizontal
> /// reference line on the replay-ratio tile. Same forwarding
> /// pattern as `promoteThreshold`.

**Code context:** `let replayRatioTarget: Double`.
**Issue:** Inherits the wrong premise from the `promoteThreshold` comment above ("Same forwarding pattern"). `ContentView.swift:115` forwards `TrainingParameters.shared.replayRatioTarget`, not anything from `UpperContentView`. Same applies to the `gradClipMaxNorm` doc two lines down.
**Suggested fix:** Fix the `promoteThreshold` comment above first; this line becomes correct as a consequence.

### Encoding/BoardEncoder.swift:340
> // pattern (no skip-if-zero optimization) so each plane is
> // self-contained and doesn't depend on the leading clear at
> // line 136 — easier to reason about and immune to the silent

**Code context:** Comment on planes 18 and 19.
**Issue:** "Leading clear" `base.update(repeating: 0, count: tensorLength)` is actually on **line 272**, not 136. Hard-coded line citation rotted.
**Suggested fix:** Reference the call by name ("the leading `base.update(repeating: 0, ...)` above") rather than by line number.

### Encoding/BoardEncoder.swift:356
> // (unlike planes 18/19) because the leading `base.update` at
> // line 192 already cleared the full tensorLength region — we

**Code context:** Comment on planes 20–29.
**Issue:** Same defect — the referenced `base.update` is on line 272, not 192. The semantic claim ("the leading clear is what makes skip-if-zero safe") is correct, just the locator is wrong.
**Suggested fix:** Drop the line number; reference the call by name.

### Network/ChessNetwork.swift:59–62
> /// Architecture v2 (post-refresh — see dcm_architecture_v2.md):
> /// - Input: 20x8x8 board tensor (NCHW layout). Planes 18 and 19 are
> ///   threefold-repetition signals (≥1× before, ≥2× before).
> /// - Stem: 3x3 conv (20 -> 128 channels), batch norm, ReLU

**Code context:** Class-level architecture summary.
**Issue:** Still describes the 20-plane v2 input, but `static let inputPlanes = 30` (line 124) and the immediately-adjacent doc at 116–124 explicitly says "v3 architecture: 20 baseline planes ... plus 10 binary temporal-repetition-history planes". Class summary and constant doc contradict each other. The "see dcm_architecture_v2.md" reference does point to an existing file but it documents the prior v2 design.
**Suggested fix:** Rewrite the class summary to v3 (30 planes, with 20 baseline + 10 temporal-repetition history planes).

### Arena/ArenaTriggerBox.swift:5
> "Trigger inbox for the arena coordinator task. The training worker
>  fires the trigger when the **30-minute auto cadence** elapses; the UI
>  fires it via the Run Arena button."

**Code context:** Class doc.
**Issue:** `ArenaAutoIntervalSec` default is **900.0** s (15 minutes), not 30. The constant is also user-tunable and re-read live — the literal is doubly misleading.
**Suggested fix:** "the configured auto cadence (`arenaAutoIntervalSec`, default 15 min)".

### Arena/ArenaTriggerBox.swift:163–164
> "the prefill/warmup phase, so the **30-minute clock** only
>  begins once the model is stable."

**Code context:** Inline comment.
**Issue:** Same drift.
**Suggested fix:** "the auto-fire clock".

### Arena/TickTournamentDriver.swift:34
> "**Public contract** mirrors `TournamentDriver.run` so the call
>  site in `SessionController+Arena.swift` can branch on the
>  `arenaUseTickDriver` flag and feed either driver into the same
>  downstream `TournamentStats` consumer."

**Code context:** Class header.
**Issue:** Both references are dead. `TournamentDriver` (singular) doesn't exist (only `TickTournamentDriver`). `arenaUseTickDriver` doesn't exist either — the call site at `SessionController+Arena.swift:203` unconditionally constructs `TickTournamentDriver()`.
**Suggested fix:** Drop the mirror/flag language entirely.

### Arena/TournamentRecord.swift:25
> "Number of arena games that actually completed before the
>  tournament ended. May be less than **`Self.tournamentGames`** if
>  the user clicked Abort or Promote mid-tournament, or if the
>  session was stopped while the arena was in flight."

**Code context:** Field doc.
**Issue:** Inside `struct TournamentRecord`, `Self.tournamentGames` doesn't exist. The actual constant is `UpperContentView.tournamentGames` (UpperContentView.swift:268), but the run-time source of truth is the live `TrainingParameters.shared.arenaGamesPerTournament` (default 400). Comment also mentions "Promote mid-tournament" but `ArenaOverrideBox.swift:5–14` and `SessionController+Arena.swift:244–251` state plainly there is no force-promote override anymore — only `abort()`.
**Suggested fix:** Refer to `TrainingParameters.shared.arenaGamesPerTournament` (or "the configured tournament size") and drop the Promote-mid-tournament mention.

### Arena/TournamentProgress.swift:15 and Arena/TournamentRecord.swift:69
> "captured once in **`runArenaTournament`** before the first game"
> "see **`runArenaTournament`**)"

**Issue:** `runArenaTournament` doesn't exist. Arena entry point is `SessionController+Arena.swift:42 func runArenaParallel(...)`.
**Suggested fix:** Replace with `runArenaParallel`.

### Chess/ChessRunner.swift:24–29
> `pieces` is the unflipped display board used to look up ghost-piece
> icons for each arrow. `state` is the source GameState the board was
> encoded from; `PolicyEncoding.decode` uses it both to interpret the
> (channel, row, col) cells back into absolute coordinates AND to
> filter to the legal subset (so absurd top-K cells like "knight jump
> from a square with no piece" don't appear as ghost arrows).

**Code context:** `evaluate(board:state:pieces:)` docstring.
**Issue:** Describes a function (`PolicyEncoding.decode`) that the code path doesn't call. `extractTopMoves` uses `PolicyEncoding.geometricDecode` (line 158) plus a separately-computed `legalSet` (line 133). `PolicyEncoding.decode` does exist (PolicyEncoding.swift:289) but isn't on this call path. Reader risk: looks for legality filter inside `PolicyEncoding.decode` when investigating an "illegal arrows" bug.
**Suggested fix:** Rewrite the doc to describe `geometricDecode + legalSet`, or change the code to call `decode`.

### Charts/AttributedMetricColor.swift:27
> /// Matches `ContentView.policyEntropyAlarmThreshold` (1.0
> /// in-repo, calibrated for post-mask / legal-only entropy).

**Code context:** Threshold field comment.
**Issue:** Constant moved out of `ContentView` to `TrainingAlarmController` (`App/UpperContentView/TrainingAlarmController.swift:43`). Value `1.0` is still correct; only the symbol path is stale. The same file's actual code at line 2808 already uses the new path.
**Suggested fix:** Update path to `TrainingAlarmController.policyEntropyAlarmThreshold`.

### Persistence/CheckpointManager.swift:749–750
> // wrote into a scratch ReplayBuffer. The scratch restore
> // runs the full v4 validation stack — magic, version,

**Code context:** Replay-buffer scratch verify.
**Issue:** `ReplayBuffer.fileVersion = 7` (line 1544). Stale by three revisions.
**Suggested fix:** "the full current-version validation stack" — let the version live in `ReplayBuffer`.

### Persistence/ModelCheckpointFile.swift:61–62
> /// Source of the save: `manual`, `promote`, or `session-autosave`.

**Code context:** `ModelCheckpointMetadata.creator` doc.
**Issue:** No call site writes `"session-autosave"`. Actual values are `manual` / `promote` / `periodic` (via `diskTag`). Verified call sites: `SessionController+Checkpoint.swift:86,325,331`, `SessionController+Arena.swift:465,471`.
**Suggested fix:** Replace `session-autosave` with `periodic`.

### Persistence/LastSessionPointer.swift:38–41
> /// Which save path wrote this pointer. One of `"manual"`,
> /// `"post-promotion"`, `"periodic"`. Purely informational —
> /// the resume flow treats all three the same way.

**Code context:** `trigger` field.
**Issue:** Actual value space is **four** values: `manual` / `periodic` / `promote` / `post-promotion`. Missing `promote` (from the manual-promote path at `SessionController+Arena.swift:517`).
**Suggested fix:** "One of `manual` / `periodic` / `promote` / `post-promotion`."

### Training/ReplayBuffer.swift:1903–1905
> /// Only after all eight pass does the function mutate any live
> /// state (taking the buffer's `lock`, resetting counters,
> /// re-seeking to the header end, and reading the four sections
> /// into the ring storage).

**Code context:** `restore(from:)` doc.
**Issue:** Reads NINE column sections, matching the file format at lines 1523–1532 (boards, moves, outcomes, plyIndices, gameLengths, samplingTaus, stateHashes, workerGameIds, materialCounts). "Four sections" is v4-era.
**Suggested fix:** "reading the nine column sections" or, more durable, "all column sections" with pointer to the file-format spec.

### Training/ReplayBuffer.swift:1829–1831
> /// Passing the hasher inout (rather than capturing it in
> /// an escaping closure) lets the single hasher object accumulate
> /// across all four section writes from a single `_writeLocked`
> /// call.

**Code context:** `_writeLocked` body calls `writeRange` nine times, not four.
**Issue:** Same v4-era drift.
**Suggested fix:** "across all column-section writes".

### Training/ReplayBuffer.swift:56,61,66,72,78
> /// Observability-only; not persisted in v4.

(Five separate occurrences on different field declarations.)

**Code context:** Field doc-blocks for `plyIndexStorage`, `gameLengthStorage`, `samplingTauStorage`, `stateHashStorage`, `workerGameIdStorage`.
**Issue:** All five fields ARE persisted, starting in v5 (current file format is v7). Pattern-bug from a long-stale documentation pass. Critically misleading: a reader pruning unused state would conclude wrong. The lone fresh comment is `materialCountStorage` at line 85 ("Persisted in v6+") — use that as the template.
**Suggested fix:** Replace each "Observability-only; not persisted in v4." with "Observability; persisted in v5+."

### Training/ReplayBuffer.swift:286–298
> /// Pack a worker_id (0..65_535) and an intra-worker game index
> /// (0..65_535) into a single UInt32 for storage in
> /// `workerGameIdStorage`. Top 16 bits = worker, low 16 bits = game.

**Code context:** `packWorkerGameId`.
**Issue:** Function accepts `gameIndex: UInt32` (not `UInt16`), with a silent `& 0xFFFF` mask inside the pack. `ActiveGame.intraWorkerGameIndex` is `UInt32` incrementing with `&+= 1`, so values up to 2³²−1 are reachable. The doc's "(0..65_535)" range is correct only if the caller enforces the cap — it doesn't. A single slot at >65,536 games (≈22 days continuous at ~30 s/game) silently aliases.
**Suggested fix:** Note the silent truncation explicitly, or change the parameter type to `UInt16` so the truncation is visible at the call site.

### Training/WorkerCountBox.swift:18–19 (LOW severity)
> ... the upper bound is similarly enforced by the Stepper and the spawn
> loop's `absoluteMaxSelfPlayWorkers` constant, not here.

**Issue:** Bare-name reference to a constant that's on `UpperContentView`, not `ContentView` as CLAUDE.md states. Comment doesn't qualify the type, so technically searchable but reader has to know where to look.
**Suggested fix:** Qualify: `` `UpperContentView.absoluteMaxSelfPlayWorkers` ``.

---

# High findings

## DEAD-RETROSPECTIVE (39)

These comments describe a previous state of the code ("previously", "used to", "was changed from", etc.). Even when accurate, they're WHAT-was-changed boilerplate the user's rules forbid and they rot fast.

### App/SessionController.swift:48
> Whether a champion network exists. (Was `UpperContentView.networkReady`.)

**Suggested fix:** Drop the parenthetical.

### App/SessionController.swift:275–276
> `UserDefaults`-backed (was `@AppStorage("lastAutoComputedDelayMs")` on
> the view). Not a training parameter — intentionally NOT in

**Suggested fix:** Drop the "was @AppStorage…" parenthetical; the "Not a training parameter — intentionally NOT in `TrainingParameters`" rationale is load-bearing and can stay.

### App/SessionController.swift:324
> long derivation comment that was on `UpperContentView` for the full why.

**Suggested fix:** Either inline the derivation here, or drop the dangling pointer.

### App/SessionController+Arena.swift:89–91
> // (Earlier behavior
> // zeroed velocity on promotion, throwing away accumulated
> // gradient signal.)

**Suggested fix:** Drop the parenthetical; the preceding paragraphs already explain why the current behavior is correct.

### App/SessionController+Arena.swift:186
> // from a single task (legacy: parent harvest loop; tick:

**Suggested fix:** Drop "(legacy: ... ; tick: ...)" — just describe how the current driver fires the callback.

### App/SessionController+CandidateProbe.swift:287–289
> Sum of the K largest values in `values`, as Double. O(N log K)
> via a fixed-size min-heap — replaces the previous full
> `.sorted(by: >).prefix(K).reduce(0, +)` which is O(N log N)
> plus a full sorted-copy allocation.

**Suggested fix:** Drop the "replaces the previous…" sentence; keep the O(N log K) explanation.

### App/SessionController+Diagnostics.swift:138–148
> // SyncBox rather than `nonisolated(unsafe) var`: today
> // `evaluate(board:consume:)` invokes `consume`
> // synchronously inside ChessNetwork's serial
> // executionQueue, so writes happen-before the
> // post-await read — no actual race exists. But the
> // annotation silenced Swift 6's Sendable-capture check
> // for a guarantee that lives outside this file; if
> // `consume` ever becomes truly async or escapes, the
> // happens-before chain breaks silently. SyncBox makes
> // the lock discipline explicit and removes the
> // unsafe-annotation requirement.

**Suggested fix:** Trim to "Uses `SyncBox<[Float]>` so the lock discipline is explicit — the `consume` closure's happens-before guarantee with the post-await read lives outside this file and would break silently if `consume` ever became truly async."

### App/SessionController+Diagnostics.swift:228–231
> // SyncBox over the (policy, value) pair so the post-await
> // read sees both fields under a single lock, with no
> // nonisolated(unsafe) capture. See the matching note in
> // runEngineDiagnostics above for why this is preferred
> // over the prior `nonisolated(unsafe) var` shape.

**Suggested fix:** Drop the "over the prior `nonisolated(unsafe) var` shape" half-sentence.

### App/SessionController+Heartbeat.swift:77–80
> Extracted out of the inline
> `.onReceive(snapshotTimer)` closure so `body`'s expression
> type-check stays cheap — the closure used to be ~140 lines and
> dragged the whole modifier chain past the
> `-warn-long-expression-type-checking` budget.

**Suggested fix:** Drop the "the closure used to be ~140 lines…" trailing sentence.

### App/SessionController+Heartbeat.swift:150–154
> // Pass the locally-snapshotted step count so the LR uses the
> // same observation rather than re-acquiring the SyncBox; the
> // count and LR in the published snapshot are then guaranteed
> // consistent (they were previously two independent reads with
> // a one-step disagreement window).

**Suggested fix:** Drop the parenthetical.

### App/SessionController+Training.swift:644–646
> // The previous
> // per-worker-Task gate array is gone along with the legacy
> // task-per-game topology.

**Suggested fix:** Drop the two trailing sentences. The "Single self-play gate. `BatchedSelfPlayDriver` is one driver task..." opening is sufficient.

### App/ChartZoomControlRow.swift:3–6
> /// Compact row — same font size and weight for every element so
> /// it lays out as one tight cluster on the left rather than the
> /// bold-zoom + tiny-hint + far-right-Auto layout it used to be.

**Suggested fix:** Rewrite without the "used to be" comparison.

### App/UpperContentView/LiveBoardWithNavigationView.swift:215
> Chevrons were previously flanking the
> board to its left and right; moving them underneath
> gives the board more horizontal room without the side
> gutters and lets us slot the Reset action between them.

**Suggested fix:** Delete the retrospective; keep only the forward-looking layout note.

### App/UpperContentView/ArenaSurfaceView.swift:59
> 1. Bucket-width drift. Arenas run before a recent change emitted
> `valueByPly` in 5-ply buckets; current arenas use 20-ply buckets

**Suggested fix:** "Some persisted arena summaries used 5-ply buckets; current ones use 20-ply" — without the "before a recent change" temporal frame.

### App/UpperContentView/HumanPlayWindow.swift:48–50
> Wider than the pre-history-panel layout: the move list now
> sits to the right of the board and needs a fixed strip
> (`HumanPlayWindowView.historyPanelWidth`) without squeezing...

**Suggested fix:** "The move-list strip on the right needs `historyPanelWidth` of fixed width without squeezing the board."

### App/UpperContentView/HumanPlayWindow.swift:214–219
> ...the original
> 10 Hz `.throttle` from the pre-pacer flow is no longer
> necessary at this volume.

**Suggested fix:** Drop the "pre-pacer flow / 10 Hz throttle" sentence.

### App/UpperContentView/AutoResumeSheetView.swift:4–8
> Extracted out of `UpperContentView` (it used to be a `autoResumeSheetContentView() -> AnyView`
> helper, written that way only to keep the `.sheet { … }` call site from
> inflating the already-huge body's type-inference cost — a concrete `View`
> struct does that job better and drops the `AnyView`).

**Suggested fix:** Trim to "Extracted from `UpperContentView`'s `.sheet { … }` call site to keep the body's type-inference cost down and avoid `AnyView`."

### App/UpperContentView/TrainingSettingsPopoverModel.swift:8–11
> Editing a field clears its own error via `didSet` (this replaces the ~19
> per-field `.onChange` handlers that previously had to be split into an
> `AnyView` chain to stay under the type-checker's per-expression budget).

**Suggested fix:** Trim to the present-tense fact: "Editing a field clears its own error via `didSet`."

### App/UpperContentView/TrainingSettingsPopover.swift:1759
> Single merged W/D/L row — replaces the prior W: / D: / L:
> triple.

**Suggested fix:** Drop the "replaces the prior … triple" clause.

### App/UpperContentView/UpperContentView.swift:84–105
> These start at the compile-time defaults (the matching `Self.X` static constants) and remain unchanged throughout a normal interactive run. When `--parameters` specifies an override, `applyCliConfigOverrides` writes the new value into the matching field here. Every runtime read site that formerly read `Self.X` now reads `effectiveX` instead, ...
> // Migrated to TrainingParameters.shared (see below). Properties formerly stored here as @State / @AppStorage are now accessed via `trainingParams.<name>`. ...

**Suggested fix:** Delete the entire `effectiveX` block; the surviving migration sentence is the only useful information.

### App/UpperContentView/UpperContentView.swift:546–549
> chart-related `@State` that used to live here — the rings,
> decimated frame, scroll position, hover position, zoom
> state, arena events, diversity bars — moved onto the
> coordinator. See `ChartCoordinator.swift`.

**Suggested fix:** Replace with "All chart-layer state (rings, decimated frame, scroll/hover/zoom, arena events, diversity bars) lives on `ChartCoordinator` — see `ChartCoordinator.swift`."

### App/UpperContentView/UpperContentView.swift:1092–1100
> The chart layer (zoom-control row + chart grid) is no
> longer rendered here — it lives in `LowerContentView`,
> which `ContentView` mounts as a sibling of
> `UpperContentView` ...

**Suggested fix:** "The chart layer is rendered by `LowerContentView`; `UpperContentView` mirrors `realTraining` into `chartCoordinator.isActive` via the `.onChange` below."

### App/UpperContentView/UpperContentView.swift:1115–1117
> Driven off the single `MenuHubSignature`
> Equatable so this is one `.onChange`, not the 13-deep chain it used
> to be — `body` tolerates it directly now that the surrounding view
> is far smaller, so the old `MenuHubSyncProbe` carrier is gone.

**Suggested fix:** "Driven off a single `MenuHubSignature` Equatable so any field change in the hub-watched set fires one handler."

### App/UpperContentView/UpperContentView.swift:1351–1360
> The main window's inline board no longer accepts
> human-play taps — human games render in their own
> window (HumanPlayWindow) ...

**Suggested fix:** "The inline board mirrors the live game position for at-a-glance visibility; human-play taps and the picker live in `HumanPlayWindow`, so all human-play wiring here is no-op."

### App/UpperContentView/UpperContentView.swift:1383–1389
> The Reset / Stop toolbar now lives inside the human-
> play window (HumanPlayWindow). The inline placement
> below the mini board was confusing operator-side
> ("am I supposed to use that?"). Kept the view-tree
> slot as an opacity-0 / height-0 spacer ...

**Suggested fix:** Trim to "View-tree slot preserved as an opacity-0/height-0 spacer to keep the surrounding VStack shape stable across builds."

### App/UpperContentView/UpperContentView.swift:1594–1597
> ...the
> formerly-used `.onAppear` hook can stall indefinitely
> waiting for a user click on the dock icon.

**Suggested fix:** "Using `.onAppear` instead would stall indefinitely on dock-icon-foreground."

### App/UpperContentView/UpperContentView.swift:1769–1775
> The session time / GPU RAM / CPU / GPU block that used to
> live here moved into the top-bar status chip + chart-grid
> tiles (App memory, GPU, CPU) once those existed. Real
> training intentionally returns empty here so the busy row
> collapses to nothing during a session ...

**Suggested fix:** Trim to: "Real training intentionally returns empty so the busy row collapses to nothing during a session — the top-bar chip + chart tiles carry the live information."

### App/UpperContentView/UpperContentView.swift:2988–2999
> Header is labelled with the trainer's model ID — the
> moving SGD copy that arena promotion turns into a
> champion. The separate Trainer ID / Champion ID rows
> are dropped: ... // SP tau / Arena tau / clip / decay are now surfaced as editable
> text fields above the body, so they are not duplicated here. /
> Learning rate likewise lives in the interactive text field.

**Suggested fix:** Trim to the forward-looking header rationale; delete "are dropped" and "no longer duplicated".

### App/UpperContentView/UpperContentView.swift:3126–3128
> Ent reg / Grad clip / Weight dec / Draw pen previously
> listed here are duplicates of the editable fields shown
> above the loss section. Removed to avoid redundancy.

**Suggested fix:** Delete the comment entirely.

### App/UpperContentView/UpperContentView.swift:3256–3261
> Arena history used to be appended here as a multi-line
> block. It now lives in the Arena History sheet ...

**Suggested fix:** Delete.

### App/UpperContentView/UpperContentView.swift:3266–3274
> Session rates and the per-outcome Results
> breakdown moved to the `SelfPlayStatsCard` and `ResultsCard`
> SwiftUI views in the new card column ...; this function now only emits the Status
> line for single-worker mode.

**Suggested fix:** "Emits only the single-worker Status line; counts/averages/rates/Results live in `SelfPlayStatsCard` and `ResultsCard`."

### App/UpperContentView/UpperContentView.swift:3354–3361
> Extracted out of `body` so each chunk type-checks independently. Before
> these existed, `body` was ~1020 lines of nested generics and clocked in
> at ~16 seconds in the type-checker ...

**Suggested fix:** Trim to "Each chunk type-checks independently. The previous monolithic body blew past `-warn-long-function-bodies`; splitting keeps each piece well under the budget."

### App/UpperContentView/UpperContentView.swift:3397–3403
> ... (The old form had to split a
> 19-deep `.onChange` chain into five `AnyView` chunks to stay under
> the type-checker's per-expression budget; that's all gone.)

**Suggested fix:** Drop the parenthetical retrospective.

### Arena/TickTournamentDriver.swift:30–33
> "one driver Task pumps K
>  active games in lockstep with parallel encode + parallel sample
>  across P CPU workers, and two batched `network.evaluateBatched`
>  calls per tick — one per network (candidate / champion) —
>  **instead of K Swift tasks each parking in an actor barrier.**"

**Suggested fix:** Drop the "instead of …" tail; the prior design is deleted.

### Arena/TickTournamentDriver.swift:51–52
> "Checked at top of each tick body — same gate as
>  **the legacy driver's `Task.isCancelled || isCancelled?() == true`**."

**Suggested fix:** Drop the "same gate as the legacy driver" framing.

### Arena/TickTournamentDriver.swift:69
> "Arena's UX
>  surfaces the per-game progress through `onGameCompleted` (chip
>  + countdown), not a watcher board. **Matches the legacy driver.**"

**Suggested fix:** Drop "Matches the legacy driver."

### Training/ChessTrainer.swift:1016–1019
> /// (Decaying BN gamma toward zero zeros out a channel and reduces effective
> /// capacity — the prior "L2 on all params" decision was reverted after the
> /// deep ML review.)

**Suggested fix:** Drop the parenthetical, or move to CHANGELOG.

### Training/ChessTrainer.swift:3190–3194
> // The earlier coupled form had a documented "μ near 0.9
> // amplifies effective step size by ~10×" trap because the
> // decayC · weight term lived inside the velocity buffer, so
> // raising μ silently amplified decay by the same factor. The
> // decoupled form fixes that.

**Suggested fix:** Drop the paragraph (or compress to "decoupled form, NOT the legacy coupled-decay form").

### Chess/ChessMachine.swift:188–198
> // `engine` is set unconditionally in `beginNewGame` at the
> // single call site that drives this method; reaching here
> // with `engine == nil` means a future refactor moved the
> // assignment or introduced a second entry point and the
> // ordering guarantee broke. The old silent-stalemate
> // fallback would inject a fake game result into whatever
> // consumer the player/delegate sees, which is exactly the
> // class of bug we want surfaced loudly.

**Suggested fix:** Drop the "The old silent-stalemate fallback would inject…" historical sentence; the forward-looking guidance is the load-bearing part.

### Chess/ChessMachine.swift:289–298
> // The loop only exits when `engine.result != nil` (or via
> // throw — handled inside the loop body), so `engine.result`
> // is always non-nil here. The old `?? .stalemate` fallback
> // is gone with the max-plies cap; if a future change brings
> // back an "exit without engine result" path it should add a
> // distinct termination cause rather than re-bucketing into
> // stalemate.

**Suggested fix:** Trim the "The old `?? .stalemate` fallback is gone with the max-plies cap" historical sentence; keep the forward-looking guidance.

### Persistence/CheckpointStatusKind.swift:7–13 (soft)
> the original flow — "Saving session (manual)…" then a same-styled
> gray "Saved <filename>" that cleared after 6 seconds — was easy to
> miss, leaving the user unsure whether the save had actually
> completed.

**Suggested fix:** Borderline — the historical contrast motivates the current "green checkmark plus longer dwell" design. Reasonable to keep, but could be trimmed to "Success messages get a green checkmark and a longer dwell so they're hard to miss."

---

# Medium findings

## FRAGILE-LITERAL (49) and DETACHED-LITERAL (24)

A FRAGILE-LITERAL is a number in a comment that matches a constant nearby — drift risk on either. A DETACHED-LITERAL is a number in a comment with no corresponding constant — will rot silently as adjacent code changes. Both violate "NEVER put numbers or values in comments". Grouped here for compactness.

### App/SessionController+Checkpoint.swift:122–126 — FRAGILE
> Upper bound on how long a save path will wait for a
> worker to acknowledge a pause request. Has to cover one
> in-flight self-play game or training step, so 15 s is a
> comfortable margin above the worst-case game length at
> typical self-play rates.

**Anchor:** `saveGateTimeoutMs: Int = 15_000` two lines below.
**Suggested fix:** Drop the literal: "...a comfortable margin above the worst-case game length at typical self-play rates."

### App/SessionController+Training.swift:1356 — FRAGILE
> the view; at default 500 steps this covers

**Anchor:** `bootstrapStatsStepCount: Int = 500`, used one line above as `let bootstrapSteps = UpperContentView.bootstrapStatsStepCount`.
**Suggested fix:** Drop the literal.

### App/SessionController+Training.swift:1362 — FRAGILE
> // steady-state log file grows at a manageable rate
> // (~60 lines/hr) while still capturing drift

**Anchor:** `steadyInterval: TimeInterval = 60` on the next line.
**Suggested fix:** Drop "(~60 lines/hr)" or rephrase as "one line per `steadyInterval`".

### App/SessionController+Heartbeat.swift:543 — FRAGILE
> // session length (~60 at the current 60s window / 1s refresh).

**Anchor:** `progressRateWindowSec = 60.0`, `progressRateRefreshSec = 1.0`.
**Suggested fix:** "bounded at `progressRateWindowSec / progressRateRefreshSec` iterations per call".

### App/SessionController.swift:798–802 — DETACHED
> Network built in \(net.buildTimeMs) ms / ID / Parameters: ~2,400,000 / Architecture: 20x8x8 -> stem(128) -> 8 res+SE blocks -> policy(4864) + value(1)

(Already in WRONG; the multiple stale architectural literals here are detached.)

### App/SessionController+Arena.swift:154 — DETACHED
> // out is one in-flight arena game (~400 ms).

**Suggested fix:** Either drop the literal or change to "tens to hundreds of ms".

### App/SessionController+CandidateProbe.swift:46–48 — DETACHED
> // network, then immediately run the probe. Doing the ~11.6 MB
> // trainer → probe copy here — not after every training block —
> // means it happens only when the probe is actually about to

**Suggested fix:** Drop the size.

### App/SessionController+Checkpoint.swift:323 — DETACHED
> // responsive during the ~150 ms scratch-network build.

**Suggested fix:** Drop or generalize to "sub-second".

### App/SessionController+Diagnostics.swift:185, 331, 335, 342, 352 — DETACHED (5 sites)
> If the policy outputs are essentially identical (avg per-cell
> |Δ| < 1e-4) the policy head has collapsed to a position-agnostic
> constant

**Issue:** The `1e-4` threshold appears in 5 places (comment + 4 code lines). No constant.
**Suggested fix:** Extract a `nonisolated static let policyConditionalNoiseFloor: Float = 1e-4` and reference everywhere.

### App/SessionController+TacticalProbe.swift:33–37 — DETACHED
> The thresholds are deliberately conservative — `correctAndConfident`
> requires the network to put more than half its legal-masked mass on
> the expected move(s). At current training state (`pEnt ≈ 1.7`, ~5
> effective moves) we expect mostly `correctButFlat` results

**Issue:** "more than half" is the hard-coded `0.5` at line 166; `pEnt ≈ 1.7` / `~5 effective moves` are point-in-time training-state observations.
**Suggested fix:** Extract `correctAndConfidentMassThreshold: Float = 0.5`; drop the training-state snapshot.

### App/SessionController+TacticalProbe.swift:201 — DETACHED
> distribution (the diagnostic-only path). Total cost ~3ms.

**Suggested fix:** Drop the wall-clock estimate.

### App/UpperContentView/AutoResumeController.swift:8 — FRAGILE
> ...the sheet is shown with a 30-second countdown; if the
> user neither resumes nor dismisses, the countdown fires the resume
> automatically.

**Anchor:** `nonisolated static let countdownStartSec: Int = 30`.
**Suggested fix:** Reference `countdownStartSec` by name.

### App/UpperContentView/CheckpointController.swift:9, 41, 47, 50 — FRAGILE
> ...flips a still-running save to a `.slowProgress` row after 10 s ...

**Anchor:** `slowSaveWatchdogSeconds: Int = 10`.
**Suggested fix:** Reference the constant; keep the literal only at definition site.

### App/UpperContentView/CheckpointController.swift:54–55, 61, 71–80 — FRAGILE
> Success lifetime is 20 s — long enough for the user to glance up
> and confirm the save actually landed — versus 6 s for progress
> lines and 12 s for errors.

**Anchor:** Lifetime switch — `.progress = 6`, `.success = 20`, `.error = 12`, `.slowProgress = 120`.
**Suggested fix:** Reference cases by name ("`.success` lifetime is the longest").

### App/UpperContentView/CheckpointController.swift:41–46 — DETACHED
> ...a healthy session save (two ~10 MB `.dcmmodel` files plus a 35
> MB replay buffer at 500k positions) takes well under a second on SSD ...

**Suggested fix:** Qualify as approximate or drop figures, keep "well under a second".

### App/UpperContentView/HumanPlayPacer.swift:14–15, 39, 119–120 — FRAGILE
> ...the AI's pre-move "absorb your move" sleep ... the 2-second timer started ... // ...scheduling the 2s timer ... // Production value is 2 seconds; tests inject a much smaller value...

**Anchor:** `init(postHumanDelay: Duration = .seconds(2))`.
**Suggested fix:** Refer to `postHumanDelay` by name; let the init carry the literal.

### App/UpperContentView/TacticalProbeWatcher.swift:14–15 — FRAGILE (and DETACHED for the "21ms")
> Probe cost is ~3ms × 7 ≈ 21ms / per tick (a small fraction of the 15s cadence).

**Anchor:** `init(... intervalSec: TimeInterval = 15.0)`. "15s" also restated in `TacticalProbeMonitorView.swift:5,166`, `TacticalProbeHistory.swift:41`, `ControlSideEffectsProbe.swift:47`, `UpperContentView.swift:2257` — five places total.
**Suggested fix:** Sweep "15s" → `intervalSec`/`candidateProbeIntervalSec` reference.

### App/UpperContentView/TacticalProbeHistory.swift:41–42 — FRAGILE
> Cap per series so a long-running monitor session doesn't grow
> unboundedly. 120 entries × 15-second cadence ≈ 30 minutes of history...

**Anchor:** `init(maxEntriesPerProbe: Int = 120)`.
**Suggested fix:** Reference the constants by name.

### App/UpperContentView/TacticalProbeMonitorView.swift:5, 166 — FRAGILE
> ...updating every 15s via the owning window controller's watcher.
> // Footer: "Ticks every 15s..."

**Suggested fix:** Pull cadence in from the watcher.

### App/UpperContentView/ControlSideEffectsProbe.swift:47 — FRAGILE
> ... otherwise the user would wait up to 15s for the interval probe...

**Anchor:** Cross-file `UpperContentView.candidateProbeIntervalSec = 15`.
**Suggested fix:** Reference the constant by name.

### App/UpperContentView/ArenaHistoryView.swift:16 — FRAGILE
> A 60pt sparkline above the list...

**Anchor:** `.frame(height: 60)`.
**Suggested fix:** Drop the height number, or extract `sparklineHeight: CGFloat = 60`.

### App/UpperContentView/ArenaHistoryView.swift:333 — FRAGILE
> "14pt capture radius" (above `if abs(location.x - dotX) <= 14`)

**Suggested fix:** Promote to a named constant.

### App/UpperContentView/ArenaHistoryView.swift:412 — FRAGILE
> // 4pt is wide enough to catch the eye when scanning ... (above: `frame(width: 4)`)

**Suggested fix:** Promote or drop.

### App/UpperContentView/TrainingSettingsPopover.swift:164–168, 320–323, 386–387 — FRAGILE
> ... 6-pt red dot next to a tab's label ... (above: `frame(width: hasError ? 6 : 0, ...)`)

**Suggested fix:** Extract `errorDotSize: CGFloat = 6`.

### App/UpperContentView/TrainingSettingsPopover.swift:1093–1094 — FRAGILE
> Rolling rates (plies / hour) over the box's `recentWindow`
> (1-minute) window.

**Suggested fix:** Drop the "(1-minute)" annotation.

### App/UpperContentView/Cards/SelfPlayStatsCard.swift:17–18 — FRAGILE
> Rates are 1-minute rolling — `ParallelWorkerStatsBox.recentWindow` is 60 s ...

**Suggested fix:** "Rates are rolling over `ParallelWorkerStatsBox.recentWindow`."

### App/UpperContentView/UpperContentView.swift:308–312 — FRAGILE
> ... 30 minutes is the default — long enough that arenas are consequential events ...

**Anchor:** `static let secondsPerTournament: TimeInterval = 30 * 60`.
**Suggested fix:** Drop "30 minutes"; the expression is self-documenting.

### App/UpperContentView/UpperContentView.swift:389–391 — FRAGILE
> ...500 picked so the bootstrap window covers the first few minutes of training ...

**Anchor:** `bootstrapStatsStepCount: Int = 500`.
**Suggested fix:** Drop "500".

### App/UpperContentView/UpperContentView.swift:2257 — FRAGILE
> ...the 15-second interval has elapsed since the last probe.

**Anchor:** `candidateProbeIntervalSec: TimeInterval = 15`.
**Suggested fix:** Reference by name.

### Arena/ArenaTriggerBox.swift:13–14 — FRAGILE
> The earlier design polled `consume()` every **500 ms** inside a `Task.sleep` loop; that woke the cooperative pool **~28k times per 4-hour session** for work that is 'false' 99.99% of the time.

**Issue:** Derived numbers from a deleted polling design, kept as justification. Soft.

### Arena/TickTournamentDriver.swift:411–413 — DETACHED
> Capture into a Sendable flag and rethrow after `await` —
> the call site at **L401/402** already propagates errors up to
> SessionController+Arena's do/catch, which aborts the
> arena cleanly without taking the app down.

**Issue:** `L401/402` line numbers don't match either this file's or `SessionController+Arena.swift`'s lines 401–402; the actual rethrow site is this file's `try await candDone / try await champDone` at lines 454–455.
**Suggested fix:** Drop the embedded line numbers.

### Arena/TickTournamentDriver.swift:155–156 — FRAGILE
> Realistic chess games end well within this bound via 50-move-rule / 3-fold-repetition / normal termination... `(newCap + 1) / 2` step.

**Issue:** `arenaCapPlies = 1024` lives only here; the comment names `ActiveGame.resetForNewGame`'s `(newCap + 1) / 2` step without matching to a constant on `ActiveGame`. Borderline soft.

### Arena/ArenaExtendedSummary.swift:213–217 — FRAGILE
> // Bucket widths chosen to be human-readable in the log block:
> //   - length:   20 plies (roughly one 'phase' of a chess game)
> //   - ply:      20 plies (matches user's explicit request)
> //   - progress: 5%       (matches user's explicit request → 20 buckets exactly covering [0, 100])

**Anchor:** `lengthBucketWidth = 20`, `plyBucketWidth = 20`, `progressBucketWidth = 5`.
**Suggested fix:** Reference constants by name.

### Arena/ArenaExtendedSummary.swift:219–220 — FRAGILE
> length buckets are 0-19, 20-39, ..., 180-199, then an open-ended overflow bucket '200+'. **10 closed buckets + 1 overflow**.

**Issue:** Labels derive from `lengthBucketWidth = 20` AND `closedLengthBucketCount = 10`. Drift if either constant changes.

### Arena/ArenaExtendedSummary.swift:233–235 — FRAGILE
> // Total-material axis: bucket width in standard piece-value
> // points. Total non-king material starts at **78** and can exceed it after promotions

**Issue:** 78 is the expected starting non-king material with `BoardEncoder.materialValue` (P×16 + N×4 + B×4 + R×4 + Q×2 = 78). If the material table changes (e.g. king != 0), drift.

### Training/ChessTrainer.swift:73 — FRAGILE
> /// [0, log(policySize)] ≈ [0, 8.49] for the current 4864-logit head.

**Anchor:** `ChessNetwork.policySize = policyChannels × boardSize² = 76 × 64 = 4864`.
**Suggested fix:** "≈ [0, log(ChessNetwork.policySize)] nats".

### Training/ChessTrainer.swift:146–147, 162, 164, 172, 499, 2757, 2760, 2761 — FRAGILE (8 sites)
> /// `1/policySize ≈ 0.0002` even when training is perfectly healthy,
> /// and can rise spuriously on outcome-skewed batches where the

**Issue:** Multiple sites mirror `1/4864 ≈ 0.0002`.
**Suggested fix:** Replace with `1/ChessNetwork.policySize` or drop the numeric estimate.

### Training/ChessTrainer.swift:199 — FRAGILE
> /// batch=512) — trivial against the existing per-step readback

**Anchor:** Current `trainingBatchSize` default is 4096.
**Suggested fix:** "~16 KB at the current 4096-position default" or drop the size estimate.

### Training/ChessTrainer.swift:638–641 — FRAGILE
> /// that's eyeballed in logs. Sized so that at the default batch
> /// of 4096 the ring still holds 8 full batches' worth of raw
> /// advantages; at smaller batches the ring is effectively the
> /// `rollingWindow` * batchSize product anyway.

**Anchor:** `advRawRingMaxCapacity = 32_768`; 32,768 / 4096 = 8.
**Suggested fix:** "at the configured `TrainingBatchSize` default the ring holds `advRawRingMaxCapacity / batchSize` full batches".

### Training/ChessTrainer.swift:645–652 — FRAGILE
> /// 512 steps at typical Play-and-Train throughput (~30 steps/min on the M-series
> /// dev machine) is ~17 minutes of history — long enough...

**Anchor:** `static let rollingTimingWindow: Int = 512` on the next line.
**Suggested fix:** Name the constant: "the `rollingTimingWindow`-step window".

### Training/ChessTrainer.swift:1167–1169 — FRAGILE
> /// μ=0.9 still behaves like ~10× the LR on the gradient term —
> /// known to push this network into the one-hot-illegal collapse
> /// mode at the empirical sweet-spot LR of 5e-5. Decay is now

**Anchor:** `learningRate` default at line 1517.
**Suggested fix:** "at the default `learningRate`".

### Training/ChessTrainer.swift:2196–2197 — FRAGILE
> // Either can be transiently *large* — up to ≈ log(policySize)
> // ≈ 8.5 nats while raw softmax mass is still mostly on illegal

**Issue:** Same `log(ChessNetwork.policySize)` derivation as line 73.
**Suggested fix:** Inline via the constant.

### Training/ChessTrainer.swift:2186–2192 — FRAGILE
> // settles near the positive-target entropy (`H(positive) ≈
> // 0.64 nats` for ε=0.1, |legal|≈30) ... (e.g. `H(complement) ≈
> // 3.4 nats` at ε=0.1, |legal|≈30).

**Anchor:** ε is `policyLabelSmoothingEpsilon = 0.1`.
**Suggested fix:** Keep the example but phrase as "for the current ε=`policyLabelSmoothingEpsilon` default".

### Training/ChessTrainer.swift:5186, 5303 — FRAGILE (2 sites)
> // estimated: the trainer literally uploads a [batch, 128, 8, 8]
> /// this batch size — one [batch, 128, 8, 8] float32 activation tensor.

**Anchor:** `ChessNetwork.channels = 128`, `ChessNetwork.boardSize = 8`.
**Suggested fix:** "[batch, `ChessNetwork.channels`, `ChessNetwork.boardSize`, `ChessNetwork.boardSize`]".

### Training/ChessTrainer.swift:1038–1040 — DETACHED
> /// Default per-head loss coefficients in `total_loss =
> /// valueLossWeight · valueLoss + policyLossWeight · policyLoss
> /// − entropyCoeff · policyEntropy`.

**Issue:** Formula omits the `illegalMassWeight · illegalMassPenalty` term added at lines 3039–3057.
**Suggested fix:** Append "+ illegalMassWeight · illegalMassPenalty" to the formula.

### Training/ChessTrainer.swift:634–637 — DETACHED
> /// At 32 K floats the copy + sort inside `percentiles()` runs in ~1 ms
> /// (vs. ~150 ms at the prior 2 M-entry ceiling), yet 32 K samples already pin the empirical p05/p50/p95 to within ~0.5%

**Suggested fix:** Acceptable as illustrative narrative, but flag that these are empirical estimates, not contracts.

### Training/ChessTrainer.swift:4821–4823 — DETACHED
> // massive VM-range allocations (seen as ~420 GB virtual vs
> // ~5 GB resident) and the main thread spends progressively
> // more time in deferred Obj-C releases.

**Suggested fix:** Acceptable as narrative.

### Training/ReplayRatioController.swift:215 — FRAGILE
> /// Subsampling keeps the buffer bounded at ~240 entries instead
> /// of the ~162k we'd accumulate sampling every barrier tick.

**Anchor:** `rateWindowSec = 60.0`, `rateSampleIntervalSec = 0.25`. 60/0.25 = 240.
**Suggested fix:** Parameterize as `rateWindowSec / rateSampleIntervalSec`, or name both constants.

### Training/ReplayRatioController.swift:187–192 — FRAGILE
> /// Typical steady-state occupancy under 32 self-play slots + 3 SGD steps/sec is ~14-57k entries in the 20 s window...

**Issue:** Operating-point estimates, not durable. Low severity.

### Training/GameDiversityTracker.swift:30 — FRAGILE (soft)
> static let histogramBounds: [Int] = [0, 1, 2, 3, 4, 5, 7, 10, 20, 40]
> (preceded by prose listing "6-7, 8-10, 11-20, 21-40, 41+")

**Issue:** Triple-redundant numeric literals (bounds / labels / prose).
**Suggested fix:** Note "labels listed in `histogramLabels` below" rather than re-listing buckets.

### Training/ParallelWorkerStatsBox.swift:101 — FRAGILE
> /// 1 minute so the displayed rate reacts quickly to throughput

**Anchor:** `static let recentWindow: TimeInterval = 60`.
**Suggested fix:** Reference the constant by name.

### Network/MoveSampler.swift:119 — DETACHED
> // a Swift Array per ply, and at K=4096 × ~50 ticks/sec this path runs 200k+ times per second.

**Issue:** K=4096 doesn't appear elsewhere; actual cap is `UpperContentView.absoluteMaxSelfPlayWorkers = 8192`. Tick-rate ~50 has no anchor.

### Network/ChessNetwork.swift:787–788 — DETACHED
> // site in the app (roughly once per barrier cycle at ~20-40
> // Hz across concurrent slots)

**Issue:** No constant for the rate.

### Network/ChessMPSNetwork.swift:104–107 — DETACHED (soft)
> /// Number of plies in the BN warmup batch. ~64 plies × 64 spatial
> /// cells = 4096 samples per channel, plenty for stable batch-stat estimation.

**Issue:** Self-contained inside file; the chain holds.

### Network/MPSChessPlayer.swift:193 — DETACHED (well-documented domain literal)
> /// Upper bound on the number of legal moves in any chess position.
> /// The mathematical maximum is around 218, so 256 leaves a safety margin

### Network/ChessNetwork.swift:1236–1237 — DETACHED (soft)
> /// EMA momentum = 0.99 (i.e. tracks roughly the last ~100 batches).

**Issue:** `0.99^100 ≈ 0.366`; real half-life ~69 steps. "Roughly" works.

### Network/ChessNetwork.swift:1505–1508, :78 — DETACHED (pre-refresh parameter counts)
> /// the new head uses ~50× fewer parameters (~9.8 K vs ~528 K).
> /// Total parameters: ~2.47M (down from ~2.92M pre-refresh ...)

**Issue:** Pure retrospective context.

### Encoding/BoardEncoder.swift:234 — FRAGILE
> /// Number of floats one encoded position occupies: `inputPlanes` × 64 squares. With the v3 architecture refresh (10 binary temporal-repetition planes added on top of the v2 baseline) this is 30 × 64 = 1920.

**Issue:** Hard-coded `30 × 64 = 1920` derived from `ChessNetwork.inputPlanes = 30`. Same concern at line 238 and line 372.
**Suggested fix:** Drop the explicit arithmetic; the expression below is self-documenting.

### Encoding/PolicyEncoding.swift:44 (and :5–7, :189) — FRAGILE
> // Total: 56 + 8 + 9 + 3 = 76 channels × 64 squares = 4864 logits.
> // ... where `policySize = policyChannels * 64 = 4864`.
> // Index range: `0..<policySize` (= 4864).

**Issue:** 76/4864 totals duplicated across the file; matched by `channelCount = 76` and `ChessNetwork.policyChannels = 76` / `policySize = 4864`. Currently consistent, three sources of truth.

### Encoding/PolicyEncoding.swift:24 — FRAGILE
> //   `channel = direction_index * 7 + (distance - 1)` for
> //   `direction_index ∈ 0..<8`, `distance ∈ 1..7`.

**Issue:** "1..7" isn't valid Swift range syntax (neither open nor closed). Should be `1...7`.

### Chess/ChessGameEngine.swift:120–132 — FRAGILE
> /// Ordered window of up to the 10 most recent prior positions...
> /// `recentPositionKeyWindow: Int = 10`

**Issue:** "10" literal in docstring duplicates the constant; plane numbers "20–29" also live in `BoardEncoder.swift:351`.
**Suggested fix:** "the `recentPositionKeyWindow` most recent prior positions, with index 0 = position 1 ply ago, index `recentPositionKeyWindow - 1` = oldest".

### Chess/ChessGameEngine.swift:200–217 — FRAGILE
> // Compute the 10-bit temporal-repetition mask: bit `i` is set
> // iff `recentPositionKeys[i] == key`.

**Issue:** "10-bit" couples to `recentPositionKeyWindow = 10` and the `UInt16` mask type. If window grows past 16 the mask type must also change.
**Suggested fix:** "`recentPositionKeyWindow`-bit" or just "one bit per windowed prior position".

### Chess/ChessRunner.swift:18–22 — FRAGILE
> The network emits raw policy logits (no softmax in the graph). For
> the UI demo we softmax over all `policySize` (4864) slots once here...

**Suggested fix:** Drop the parenthesized "(4864)".

### Chess/ChessRunner.swift:138–146 — FRAGILE
> // headroom. Sorting the full 4864-cell vector costs nothing
> // (policy-size array, one pass) and guarantees we'll always

**Suggested fix:** Replace "4864" with `ChessNetwork.policySize`.

### Chess/ChessMachine.swift:122–130 — DETACHED
> ... `stopAll`-style pauses no longer have to wait out an entire 300-ply game before a slot can exit. That was the mechanism behind the 15-second save-session pause timeouts observed in the session log.

**Issue:** "300-ply game" — actual max-plies cap is `SelfPlayMaxPliesPerGame` (default 150, range 25...500), and per file's own comment block at 147–155 doesn't apply to `ChessMachine`. "15-second" timeout is a runtime symptom, not a constant.
**Suggested fix:** Drop both literals.

### Persistence/PeriodicSaveController.swift:3 (and 35–39) — FRAGILE
> /// Pure-logic scheduler for the 4-hour periodic session autosave.
> /// Interval between scheduled saves... 4 hours per the spec...

**Anchor:** `UpperContentView.periodicSaveIntervalSec = 4 * 60 * 60`.
**Suggested fix:** Drop "4-hour" / "4 hours" — the constant lives elsewhere.

### Persistence/SessionSaveTrigger.swift:18–20 — FRAGILE
> /// Fired by `PeriodicSaveController` when its 4-hour deadline elapsed.

**Suggested fix:** Same as above — drop "4-hour".

### Persistence/CheckpointStatusKind.swift:7–13 — FRAGILE (also DEAD-RETROSPECTIVE — see above)
> ... "Saved <filename>" that cleared after 6 seconds...

### Utils/DiagSampler.swift:28–30 (and 62–63) — FRAGILE
> /// Cost is roughly O(regions) — ~1-3 ms per call at typical app sizes (~2K regions). Calling once per `[STATS]` emit (60 s cadence) is well below 1% overhead;

**Issue:** Empirical figures with no anchor.

### Logging/SessionLogger.swift:45–47 — FRAGILE
> /// pending flush and schedules a new one 0.5 s out. A burst of
> /// writes (per-step STATS at 5–20 Hz, BATCH-STATS at ~1 Hz) therefore

**Issue:** `[STATS]` fires per-step only during bootstrap (first 500 steps); after that it's per 60s. `[BATCH-STATS]` cadence is governed by `batchStatsInterval` (default 10 batches). The "5–20 Hz / ~1 Hz" guess is flavor; the load-bearing 0.5 s flush claim is correct.

### CLI/CliTrainingRecorder.swift:67–72 — FRAGILE (also slightly imprecise)
> /// Encode the Codable snapshot to `Data`. Shared back-end of `writeJSON(to:)` and `writeJSONToStdout(...)` so both paths emit byte-identical output.

**Issue:** Not byte-identical — `writeJSONToStdout` appends a trailing newline (line 121).
**Suggested fix:** "byte-identical JSON payload".

---

## STALE-DATE (3)

### Training/ChessTrainer.swift:1032
> (see 2026-04-15 incident). Under heavy policy-collapse

**Code context:** `gradClipMaxNormDefault` doc.
**Issue:** Inline calendar date with no in-repo link. Likely belongs to the regime when the clip was 5.0 (see WRONG finding at line 1029); may be stale alongside the literal.
**Suggested fix:** Replace with a CHANGELOG/ROADMAP pointer, or remove the parenthetical if the 30.0 threshold was chosen for a different reason.

### Utils/SyncBox.swift:1–6 (soft)
> //  Created by Andrew Benson on 5/3/26.

**Issue:** Xcode template header; creation date doesn't update. Project convention seems to be to leave these. Flagged only because it's the only date in the file.

### App/UpperContentView/AutoResumeFormat.swift:77
> "Started Apr 30, 2026 at 8:00 AM (10h ago)" line. Rendered
> only when the session.json peek succeeded...

**Issue:** Embeds a calendar date as the example output. Will look misleading once that date is far past.
**Suggested fix:** Replace with a generic format token, e.g. "Started <Date> (<Nh ago>) line."

---

# Low findings

## WHAT-COMMENT (7)

### Network/ChessNetwork.swift:350
> // Input: [batch, inputPlanes, 8, 8]
> let input = g.placeholder(shape: [-1, NSNumber(value: Self.inputPlanes), 8, 8], ...)

**Issue:** Restates the placeholder shape. Benign as a section marker.

### Network/ChessNetwork.swift:1600
> // Flatten: [batch, 1, 8, 8] -> [batch, 64]
> x = graph.reshape(x, shape: [-1, 64], name: "value_flatten")

**Issue:** Restatement. Benign.

### Network/MoveSampler.swift:165
> // Inverse-CDF sampling from probabilities in probsScratch[0..<n].

**Issue:** Section label.

### Encoding/BoardEncoder.swift:211–212
> /// - Plane 12-13: current player's castling rights (kingside, queenside)
> /// - Plane 14-15: opponent's castling rights (kingside, queenside)

**Issue:** Should be "Planes 12–13" / "Planes 14–15" (plural) to match style elsewhere.

### Arena/ArenaExtendedSummary.swift:213–217
> // (matches user's explicit request)

**Issue:** Historical breadcrumb that doesn't carry forward design intent.

### Persistence/PeriodicSaveController.swift:208–221 (internal contradiction)
> // We intentionally do NOT clear pendingFire here —
> // [...]
> // But to avoid immediate re-firing on the very next
> // tick (while the save task is in flight), clear it now ...

**Issue:** First sentence contradicts the action two lines down. Comment was patched in the second paragraph rather than fixing the first.
**Suggested fix:** Collapse to a single coherent explanation.

### Persistence/ModelCheckpointFile.swift:454–457
> // Explicitly qualify with `Swift.` because the unqualified
> // `withUnsafeBytes` inside a `Data` extension resolves to the
> // instance method on `self`, not the global function.

**Issue:** Useful — explains a non-obvious shadowing pitfall in a `Data` extension. Keep.

## MARK-LITERAL (1)

### Training/TrainingParameters.swift:164
> // MARK: - 39 parameter keys (macro-driven)

**Issue:** Actual `@TrainingParameter` count is **42**, verified via `grep -c '^@TrainingParameter'`. The "macro-driven" claim itself is correct.
**Suggested fix:** Drop the number: `// MARK: - Parameter keys (macro-driven)`.

---

# OK-LOAD-BEARING (Info — preserve)

These are multi-paragraph design blocks (≥ 6 lines) that were audited and confirmed accurate. They're load-bearing knowledge that the project's house style intentionally preserves — listed here so a future cleanup doesn't trim them by mistake. Each entry is `file:lines — one-line topic`.

## App/

- `App/SessionController.swift:1–25` — Stage 4a scope (what was lifted out of UpperContentView vs. what stays).
- `App/SessionController.swift:285–324` — `updateReplayRatioCompensator` integral-compensator derivation.
- `App/SessionController.swift:371–395` — `TrainingStartMode` enum per-case Stop/Start preservation rules.
- `App/SessionController+Arena.swift:23–40` — `runArenaParallel` synchronization model + arena-vs-self-play interaction.
- `App/SessionController+Arena.swift:84–91` — optimizer-velocity snapshot at arena start.
- `App/SessionController+Arena.swift:436–456` — post-promotion autosave detached-task rationale (deadlock avoidance).
- `App/SessionController+Arena.swift:563–583` — `[ARENA]` log-format example.
- `App/SessionController+ManualPromote.swift:7–24` — asymmetry vs arena promotion (no velocity rewind).
- `App/SessionController+Heartbeat.swift:413–429` — `refreshProgressRateIfNeeded` rolling-window math.
- `App/SessionController+Heartbeat.swift:596–621` — `formatElapsedAxis` cascade (avoiding "0.0/0.0" early labels).
- `App/SessionController+TacticalProbe.swift:6–43` — ProbeCategory/Verdict/Fixture model.
- `App/SessionController+Training.swift:243–322` — `[RESUME-PARAM]` audit-line pattern.
- `App/AppCommandHub.swift:4–19` — Bridging SwiftUI `.commands` to ContentView state.
- `App/AppCommandHub.swift:43–58` — `canResumeFromAutosave` semantics.
- `App/AppCommandHub.swift:120–143` — `runPolicyConditioningDiagnostic` / `runTacticalProbe` probe contract.
- `App/AppDelegate.swift:4–17` — Lifecycle bridging contract, `Darwin._exit(0)` note.
- `App/ContentView.swift:3–22` — Composer view's role, child decomposition.
- `App/ContentView.swift:35–43` — `chartCollectionEnabled` semantics.
- `App/ContentView.swift:82–89` — `upperPane` flexible-height rationale.
- `App/DrewsChessMachineApp.swift:61–89` — CLI argument-parsing preamble.
- `App/DrewsChessMachineApp.swift:90–135` — `takeValue(for:)` design.
- `App/DrewsChessMachineApp.swift:147–162` — `--parameters` validation.
- `App/DrewsChessMachineApp.swift:198–218` — Unknown-arg `_exit(2)` rationale.
- `App/TacticalProbeData.swift:2–14` — Probe-fixture data: hand-built positions rationale.

## App/UpperContentView/

- `App/UpperContentView/AutoResumeController.swift:1–13` — Launch-time auto-resume flow.
- `App/UpperContentView/CheckpointController.swift:1–13` — Stage 3c part 1 extraction scope.
- `App/UpperContentView/ControlSideEffectsProbe.swift:1–16` — Hidden zero-sized view funneling `.onChange` side effects.
- `App/UpperContentView/HumanPlayPacer.swift:1–55` — State machine + AI permission gate.
- `App/UpperContentView/HumanPlayBoardView.swift:1–28` — Coordinate convention + slide animation.
- `App/UpperContentView/HumanPlayWindow.swift:5–24` — Window lifecycle + registry pattern.
- `App/UpperContentView/HumanPlayWindow.swift:139–165` — Three-region SwiftUI content.
- `App/UpperContentView/TacticalProbeWatcher.swift:1–20` — Driver lifecycle + @MainActor isolation.
- `App/UpperContentView/TacticalProbeHistory.swift:1–14` — Ring buffer design + actor isolation.
- `App/UpperContentView/TrainingAlarmController.swift:1–31` — Two raise paths + thresholds preamble.
- `App/UpperContentView/TrainingAlarmController.swift:36–95` — Threshold rationale (entropy / value saturation / draw collapse).
- `App/UpperContentView/TrainingSettingsPopover.swift:1–35` — Tab + live-propagation exception contract.
- `App/UpperContentView/TrainingSettingsPopoverModel.swift:1–34` — Live-propagation exception + cancel-stash contract.
- `App/UpperContentView/PlayController.swift:30–37` — Controller ownership + UI bindings.
- `App/UpperContentView/PlayController.swift:296–309` — `start(...)` argument contract for Revert-to-here.
- `App/UpperContentView/ArenaSurfaceView.swift:48–72` — Surface aggregation + bucket-width rationale.
- `App/UpperContentView/ArenaBreakdownsView.swift:4–50` — Ten-chart catalog with per-chart purpose.
- `App/UpperContentView/UpperContentView.swift:895–928` — `ControlSideEffectsProbe` rationale.

## Training/

- `Training/ChessTrainer.swift:1922–2006` — Policy-loss block (label smoothing + signed-advantage split + complement CE).
- `Training/ChessTrainer.swift:2120–2206` — Per-position CE clamp discussion + rejected alternatives.
- `Training/ChessTrainer.swift:2207–2280` — Advantage baseline + RMS-vs-σ standardization.
- `Training/ChessTrainer.swift:2281–2317` — Signed-advantage + complement CE equilibria.
- `Training/ChessTrainer.swift:2407–2441` — Value loss (categorical CE over W/D/L), `idx = 1 − z` mapping.
- `Training/ChessTrainer.swift:2510–2540` — Value-head diagnostics (post-WDL).
- `Training/ChessTrainer.swift:2542–2585` — Policy entropy block (masked vs raw logits).
- `Training/ChessTrainer.swift:3156–3215` — **Authoritative** SGD update with decoupled weight decay and Polyak momentum.
- `Training/ChessTrainer.swift:3591–3618` — `trainStep` fresh-baseline doc.
- `Training/ChessTrainer.swift:4353–4370` — Trainer state persistence (weights + BN + velocity).
- `Training/ChessTrainer.swift:1198–1238` — `drawPenalty` threshold-on-salvaged-draws rationale.
- `Training/ReplayBuffer.swift:5–23` — Class docstring (ring buffer, lock discipline).
- `Training/ReplayBuffer.swift:1106–1134` — Under-fill guard rationale (K-cap break to honor dst contract).
- `Training/BatchedSelfPlayDriver.swift:5–82` — Class docstring (tick-based driver topology); the `History` section calls out the move away from `expectedSlotCount`/`stopAll`.
- `Training/BatchedSelfPlayDriver.swift:332–345` — Parallel sample/apply discipline.
- `Training/ReplayRatioController.swift:4–102` — Class docstring (four hooks + closed-form delay).
- `Training/ReplayRatioController.swift:268–306` — `recordSelfPlayBarrierTick` unit-conversion derivation.
- `Training/ReplayRatioController.swift:580–606` — `smoothedSelfPlayPerGameDelayMs` target-aware factor derivation.
- `Training/ActiveGame.swift:3–31` — Class doc (Option A layout, `@unchecked Sendable`).
- `Training/CancelBox.swift:3–14` — Locked-Bool vs `Task.isCancelled` rationale.
- `Training/GameDiversityTracker.swift:88–94` — In-place `[Int16]` slot rewrite vs CoW.
- `Training/GameDiversityTracker.swift:115–124` — Per-snapshot divergence-ply recompute (O(n²) cost note).
- `Training/EarlyStopCoordinator.swift:1–24` — SIG_IGN before DispatchSource race.
- `Training/ParallelWorkerStatsBox.swift:5–10` — Phase-histogram alignment with `ReplayBuffer.computeBatchStats`.
- `Training/ParallelWorkerStatsBox.swift:113–126` — `RawGameResult` vs `GameResult`.
- `Training/ParallelWorkerStatsBox.swift:208–216` — `_sessionStart` rate-denominator nuance.
- `Training/ParallelWorkerStatsBox.swift:222–228` — Back-compat seeded-init Optional fields.
- `Training/SamplingScheduleBox.swift:3–15` — Take-effect semantics + lock discipline.
- `Training/TrainingChartSample.swift:5–13` — On-disk schema compatibility + `formatVersion` rule.
- `Training/WorkerPauseGate.swift:3–22` — Request/ack gate semantics.

## Arena/

- `Arena/ArenaActiveFlag.swift:3–9` — Lock-protected arena-vs-probe mutex.
- `Arena/ArenaChartEvent.swift:3–7` — Arena → chart duration band rationale.
- `Arena/ArenaEloStats.swift:1–18` — Elo formula, candidate-perspective convention, empirical-variance Wald CI.
- `Arena/ArenaEloStats.swift:31–34, 56–61` — Endpoint nil-handling, point-CI degeneracy.
- `Arena/ArenaExtendedSummary.swift:159–168` — Field layout + append-only forward-compat decoder.
- `Arena/ArenaExtendedSummary.swift:189–206` — Custom `init(from:)` rationale (legacy sessions lack material arrays).
- `Arena/ArenaExtendedSummary.swift:486–507` — `ArenaMaterial.summary` purpose + king-cancellation.
- `Arena/ArenaLogFormatter.swift:1–17, 138–151, 234–241` — Extracted formatter rationale + calibration story + `"nan"` vs `"—"` asymmetry.
- `Arena/ArenaOverrideBox.swift:1–19` — Set-once abort semantics + lock discipline.
- `Arena/ArenaTriggerBox.swift:12–21, 30–37, 77–97, 129–137` — AsyncStream wakeup model + coalescing + tri-state enum + race prevention.
- `Arena/PromotionKind.swift:3–10` — `.automatic` vs `.manual` provenance, `gamesPlayed == 0` invariant.
- `Arena/TickTournamentDriver.swift:5–15` — `nilEvaluationBuffer` error rationale.
- `Arena/TickTournamentDriver.swift:25–69` — Public contract (K behavior, cancellation, color alternation). (Legacy-driver mentions flagged DEAD-RETROSPECTIVE above; surrounding contract is current.)
- `Arena/TickTournamentDriver.swift:117–128` — Per-slot state tracking invariants.
- `Arena/TickTournamentDriver.swift:204–215` — Game-end pass: natural vs max-plies-drop.
- `Arena/TickTournamentDriver.swift:340–345` — `===` identity precondition.
- `Arena/TickTournamentDriver.swift:386–402` — `async let cand / champ` parallel-submission.
- `Arena/TickTournamentDriver.swift:457–473` — Pre-apply candidate-value snapshot.
- `Arena/TickTournamentDriver.swift:683–700` — `NetworkRefCarrier` / `ArenaPointerCarrier` Sendable shims.
- `Arena/TournamentLiveBox.swift:1–7` — Single-writer/multi-reader, `asyncSnapshot` off-main pattern.
- `Arena/TournamentProgress.swift:14–19` — `startTime` once-captured rationale.
- `Arena/TournamentRecordsBox.swift:1–13` — `@Sendable` callback defensive belt.
- `Arena/TournamentTypes.swift:5–12, 115–125` — Player-A == candidate convention + validity-sweep origin.

## Network/

- `Network/ChessMPSNetwork.swift:53–81` — Warmup rationale (random-init inference network instability).
- `Network/ChessNetwork.swift:597–606` — Autoreleasepool VM-bloat retrospective.
- `Network/ChessNetwork.swift:1673–1692` — He-init fan-in correctness; OIHW vs `[in, out]` FC, prior `dropFirst()` bug.
- `Network/ChessNetwork.swift:1736–1748` — `vDSP_vclip` rationale vs `arc4random_buf` zeros.
- `Network/ChessNetwork.swift:1359–1369` — SE module placement rationale.
- `Network/ChessNetwork.swift:115–124` — `inputPlanes` invariant chain (downstream sites that auto-propagate).
- `Network/MoveEvaluationSource.swift:80–94` — Caller-owned destination contract.
- `Network/MPSChessPlayer.swift:205–215` — `schedule` mutation contract (between-games-only invariant).
- `Network/MoveSampler.swift:249–278` — Marsaglia–Tsang acceptance-rate notes.

## Persistence/

- `Persistence/CheckpointManager.swift:286–301` — `fullSyncPath` doc (open RO + F_FULLFSYNC + fallback).
- `Persistence/CheckpointManager.swift:689–700` — fsync staging rationale.
- `Persistence/CheckpointManager.swift:601–667` — `writtenSnap` override semantics (post-write supersedes pre-pause counters).
- `Persistence/CheckpointManager.swift:760–767` — Scratch capacity sized to `storedCount` not `capacity`.
- `Persistence/SessionCheckpointFile.swift:122–123, 401–409` — `currentFormatVersion = 1` + `unsupportedVersion` reject.
- `Persistence/SessionCheckpointFile.swift:144–349` (many lines) — Optional-field back-compat annotations across `drawPenalty`, `arenaConcurrency`, etc.
- `Persistence/SessionCheckpointFile.swift:261–267` — `maxPliesPerGame` JSON-key rename note.
- `Persistence/SessionCheckpointFile.swift:419–434` — `withTrainingSegments` / `withChartData` type-checker workarounds.
- `Persistence/SessionCheckpointFile.swift:507–514` — `readAll(from:)` URL normalization for file-importer URLs.
- `Persistence/ModelCheckpointFile.swift:75–137` — Binary layout, format-version, magic, FNV-1a archHash.
- `Persistence/ModelCheckpointFile.swift:147` — `minimumEncodedSize` arithmetic.
- `Persistence/ModelCheckpointFile.swift:179–187` — `weights` sub-array contract (champion vs trainer).
- `Persistence/PeriodicSaveController.swift:5–19, 200–225` — Arena-defer/pendingFire/save-during-window swallow.
- `Persistence/ModelID.swift:5–80` — ID format spec + UTC-counter reset + POSIX-locale rationale.
- `Persistence/SessionSaveTrigger.swift:11–13, 22–31` — Why `.postPromotion` is intentionally not in the enum.
- `Persistence/SessionSaveTrigger.swift:36–44` — `diskTag` rationale (`manualPromote` reuses `"promote"` tag).
- `Persistence/LastSessionPointer.swift:9–14` — Path-vs-bookmark rationale.
- `Persistence/LastSessionPointer.swift:67–84` — Decode-failure deliberately doesn't clear (future-schema-tolerance).

## Encoding/

- `Encoding/BoardEncoder.swift:205–229` — **30-plane spec table** (canonical readable spec; verified end-to-end against code).
- `Encoding/PolicyEncoding.swift:19–55` — **"Locked spec — do not reorder"** 76-channel layout (verified bijection).

## Chess/

- `Chess/ChessMachine.swift:42–49` — `ChessMachineDelegate` queue contract (cited in CLAUDE.md).
- `Chess/ChessMachine.swift:308–314` — `emit` strong-self via `DelegateBox` pattern.
- `Chess/ChessGameEngine.swift:39–47` — `PositionKey` FIDE-rule deviation note.
- `Chess/ChessGameEngine.swift:164–194` — Irreversible-move clearing block.
- `Chess/ChessMove.swift:12–21` — **Deliberate absence** of `policyIndex` property (cited in CLAUDE.md).
- `Chess/GameWatcher.swift:5–17, 23–35` — Class doc (not @Observable, polled by ContentView) + `eventSeq` monotonicity.
- `Chess/HumanChessPlayer.swift:43–57, 100–118` — Continuation cancel-race + install-with-cancel-check.
- `Chess/MoveGenerator.swift:61–68` — `applyMove` precondition contract.
- `Chess/MoveVisualization.swift:11–18` — `isLegal` diagnostic-intent rationale.

## Views/ and Charts/

- `Views/Charts/TrainingChartGridView/NonNegChart.swift:17` — 4864 policy-cells citation.
- `Views/Charts/TrainingChartGridView/EntropyChart.swift:18–22` — `log(30) ≈ 3.40` legal-move baseline.
- `Views/Charts/TrainingChartGridView/WDLProbabilityChart.swift:30–36` — lineWidth choices grounded in init values.
- `Views/Board/TensorChannelNames.swift:9–11` — Lockstep-with-`ChessNetwork.inputPlanes` assertion-pointer.
- `Charts/ChartZoom.swift:51–53` — Default zoom index = 30 min reconciliation.
- `Views/PolicyChannelsPanel.swift:36–40` — Section index ranges (matches `PolicyEncoding` base constants).
- `Views/HoverPolicyOverlay.swift:189–205` — Direction order lockstep with `PolicyEncoding.queenDirections` / `knightJumps`.

## CLI/, Logging/, Utils/

- `CLI/CliTrainingConfig.swift:6–9, 22–25` — `training_time_limit` semantics (session-time budget pulled out before apply).
- `CLI/CliTrainingConfig.swift:52–56` — NSNumber objCType disambiguation rationale.
- `CLI/CliTrainingRecorder.swift:9–12` — `OSAllocatedUnfairLock<State>` pattern reference.
- `CLI/CliTrainingRecorder.swift:312–326` — `BatchStatsSnapshot.SamplingConstraintsSnapshot` field semantics.
- `CLI/CliTrainingRecorder.swift:435–437` — `dup_distribution` denominator note.
- `CLI/CliTrainingRecorder.swift:358–362` — `bufferUniquePct` formula.
- `CLI/CliTrainingRecorder.swift:510–518` — `policyLossWin` / `policyLossLoss` semantics.
- `CLI/CliTrainingRecorder.swift:555–565` — `ratioProducedRate` / `ratioProductionRate` distinction (raw vs emitted).
- `CLI/CliTrainingRecorder.swift:567–574` — `selfPlayMovesPerHour` / `trainingMovesPerHour` (3600× rates).
- `Logging/SessionLogger.swift:12–17, 44–48, 137, 188–191` — 0.5 s flush deadline (load-bearing constant, correctly documented 5× in file).
- `Logging/SessionLogger.swift:25–26` — Filename pattern `dcm_log_yyyymmdd-HHMMSS.txt`.

---

# Appendix A — Files with zero findings

Files audited in full where no comment problems were identified.

## App (10 files)

- `App/AppDelegate.swift`
- `App/BuildInfo.swift` (auto-generated; quick read confirmed no audit-worthy content)
- `App/CliParametersDocument.swift`
- `App/DrewsChessMachineApp.swift`
- `App/EvaluationResult.swift`
- `App/MemoryStatsSnapshot.swift`
- `App/PlayAndTrainBoardMode.swift`
- `App/ProbeNetworkTarget.swift`
- `App/TacticalProbeData.swift`
- `App/TacticalProbeMonitorWindow.swift`

## App/UpperContentView (30 files)

`AboutPopoverContent.swift`, `ArenaCountdownChip.swift`, `ArenaPopoverField.swift`, `ArenaSettingsPopover.swift`, `ArenaSettingsPopoverModel.swift`, `AutoResumeBuildBlockView.swift`, `AutoResumeProgressBlockView.swift`, `AutoResumeSheetView.swift`*, `AutoResumeStatRowView.swift`, `AutoResumeTriggerBadgeView.swift`, `BoardSideView.swift`, `Cards/ResultsCard.swift`, `Cards/SelfPlayStatsCard.swift`*, `CheckpointStatusLineView.swift`, `CumulativeStatusBar.swift`, `HumanPlayBoardView.swift`*, `MainTextPanel.swift`, `MenuHubSignature.swift`, `PlayController.swift`*, `SelfPlayStatsColumn.swift`, `SessionStatusChipView.swift`, `TacticalProbeRowView.swift`, `TacticalProbeSparkView.swift`, `TitleBarView.swift`, `TrainingAlarmBanner.swift`, `TrainingSettingsChip.swift`, `TrainingStatsColumn.swift`, `UpperCumulativeStatusBar.swift`, `UpperTrainingStatsColumn.swift`, `ValueLabelSmoothingInfoButton.swift`

(* — some of these were also surfaced under OK-LOAD-BEARING for specific design blocks; the rest of the file is clean.)

## Training (11 files)

`ActiveGame.swift`*, `CancelBox.swift`*, `DiversityHistogramBar.swift`, `EarlyStopCoordinator.swift`*, `GameDiversityTracker.swift`* (one soft FRAGILE only), `ProgressRateSample.swift`, `SamplingScheduleBox.swift`*, `SweepProgress.swift`, `TrainingAlarm.swift`, `TrainingChartSample.swift`*, `WorkerPauseGate.swift`*

## Arena (8 files)

`ArenaActiveFlag.swift`, `ArenaChartEvent.swift`, `ArenaEloStats.swift`, `ArenaOverrideBox.swift`, `PromotionKind.swift`, `TournamentLiveBox.swift`, `TournamentRecordsBox.swift`, `TournamentTypes.swift`

## Views (43 files)

All 22 files in `Views/Charts/TrainingChartGridView/` (one exception: `WDLProbabilityChart.swift` has a WRONG and `Context.swift` has a FRAGILE), all 10 files in `Views/LogAnalysis/`, all 3 in `Views/Common/`, both `Views/Board/ChessBoardView.swift` and `ChannelBoardView.swift`, and `Views/HoverPolicyOverlay.swift` + `PolicyChannelsPanel.swift`.

## Charts (5 files)

`ChartCoordinator.swift`, `ChartDecimation.swift`, `ChartFileFormat.swift`, `ChartSampleRing.swift`, `ChartZoom.swift` (one OK-LOAD-BEARING; otherwise clean).

(`Charts/AttributedMetricColor.swift` and `Views/Charts/TrainingChartGridView/Context.swift` and `WDLProbabilityChart.swift` are the only 3 files in this partition with non-Info findings.)

---

# Appendix B — Audit methodology

13 parallel agent partitions, each given the same brief: read every Swift file in their partition completely, classify findings into the categorical scheme above, emit verbatim quotes (no paraphrasing), and write to a per-partition file at `/tmp/audit-<partition>.md`. Each agent grepped for matching constants before flagging FRAGILE-LITERAL vs DETACHED-LITERAL, and verified named symbols/files exist on disk before NOT flagging DRIFTED-REF. The 13 partition files were then merged into this report.

Spot-check verification (per the plan): 7 Critical findings and 5 High findings were re-opened against actual source to confirm quoted comments are verbatim and the issue claims reproduce. All 12 spot-checks reproduced cleanly; none of the partition outputs were pulled. Findings preserve agent quotes verbatim; agent analysis and suggested fixes are also preserved.

Categories were chosen to align with the user's standing rules from `CLAUDE.md`:
- "NEVER blindly believe a source code comment" → drives DRIFTED-REF + WRONG.
- "NEVER put numbers or values in comments. They always change." → drives FRAGILE-LITERAL + DETACHED-LITERAL.
- "NEVER add a comment describing WHAT was changed" → drives DEAD-RETROSPECTIVE.

OK-LOAD-BEARING is informational only — those blocks are explicitly preserved per the house style note in CLAUDE.md ("Most source comments are multi-paragraph design explanations, not function-summary boilerplate"). They're listed so a sweep doesn't accidentally trim them.

---

*End of audit.*
