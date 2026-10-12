# LR keyframes plan: peak and trough set by a list of keyframes, and a chart of the schedule

Status (2026-10-12): plan written from a code survey; owner decisions K1–K4, O-1 … O-6 and Q-1 … Q-4 recorded below (O-1 revised by Q-4). Revised the same day after an independent review (17 findings, all addressed below). Q-5 and Q-6 decided by the owner the same day. **Approved for implementation by the owner, 2026-10-12.**

Paths are relative to `DrewsChessMachine/DrewsChessMachine/` unless they start with `DrewsChessMachine/`, `DrewsChessMachineTests/`, `documentation/`, `scripts/` or `experiments/`.

## Problem

The LR cycle (`Training/LRMomentumCycle.swift`) oscillates between a trough and a peak; both bounds decay geometrically from a start value (`lr_cycle_min` / `lr_cycle_max`) to an end value (`lr_cycle_trough_end` / `lr_cycle_peak_end`) over one shared horizon (`lr_cycle_decay_horizon_steps`, counted in cycle steps = trainer step − warmup), then hold. That can't express the schedules the 2026-10-11/12 runs point to:

- A two-stage peak (owner's example: 1.0 → 0.1 over 50k steps, then → 0.01 over 150k, then hold).
- A trough on its own timeline (0.01 → 0.0001 over 100k while the peak follows another).
- A floor. Measured weight motion (2026-10-12, consecutive R-replay checkpoints): at a trough LR of ≈ 0.0007–0.0013, 50% of bf16 working weights change per 1,000 steps (median relative change 0.23%); motion scales with LR, so extending the trend 0.00001 gives ~0.5% and 0.000001 ~0.05% — the working network is effectively frozen. The current single decay drives R-replay's trough to 1e-6 at step 1M.

The schedule is also invisible. On 2026-10-11 an exact resume silently kept the checkpoint's schedule over the configured one (`[REPLAY-RESUME] WARNING lr_momentum_cycle: … restoring the checkpoint's value`), and the gentle cycle needed a branch (zero velocity, warmup, new lineage).

## Owner decisions (2026-10-12)

- **K1 — keyframes.** The envelope becomes a list of keyframes; each sets the peak and the trough at a trainer step.
- **K2 — one interpolation between keyframes: geometric.** No per-segment function choice. A hold is two keyframes with equal values; a step change is two keyframes one step apart.
- **K3 — no separate momentum list.** Momentum keeps following the LR cycle's phase, inverted: lowest at each LR peak, highest at each LR trough.
- **K4 — visualize** the keyframe envelope over time and where the actual oscillating LR lands.
- **O-1 — follow-momentum bounds: revised by Q-4.** First decided as two fixed values with the start → end drift dropped; Q-4 (a resume of an old run must not change at all) needs the drift representable, so each keyframe carries the momentum at the LR peak and at the LR trough, interpolated linearly between keyframes (the old drift was linear). Still one list (K3): no momentum timeline of its own.
- **O-2 — keyframe steps are trainer steps.**
- **O-3 — the first keyframe is at step 0**, required by validation.
- **O-4 — on a CLI exact resume, the configured schedule wins.** The resume is NOT EXACT with a `schedule` gap that `--accept-inexact schedule` must name; weights, velocity, clock, RNG and corpus position are still restored.
- **O-5 — a trough below 1e-4 logs a one-line `[PARAM]` warning** citing this measurement; never a refusal.
- **O-6 — GUI editing is a keyframe table** (step / peak / trough rows, add / delete) beside the chart; chart dragging may come later.
- **Q-1 — the list-valued parameter is declared through the `@TrainingParameter` macro** (a new `ValueKind`), not a hand-written conformance.
- **Q-2 — older builds can't open new model files**; accepted.
- **Q-3 — at most 64 keyframes.**
- **Q-4 — converting an old schedule changes nothing.** A resume of a file written before this change, configured with the same schedule it trained under, continues bit for bit as it would have on the old build: same LR and momentum at every step, no `schedule` gap. (A resume configured with a *different* schedule follows O-4; see Q-5.)

## Design

### The schedule types (`TrainingParametersMacroSupport/LRKeyframeSchedule.swift`, new)

They live in the macro's support package (`DrewsChessMachine/Packages/TrainingParametersMacro/Sources/TrainingParametersMacroSupport/`), because `ParameterValue` / `ParameterType` are `public` and the macro's declaration overloads (`Macros.swift`) must name the default's type (Q-1). Both are `public`.

- `LRKeyframe: Sendable, Equatable, Codable { step: Int; peak: Double; trough: Double; momentumAtPeak: Double; momentumAtTrough: Double }`. The momentum pair is **always present** (it is inert unless momentum follows the LR cycle, as the four follow keys are today); following stays gated by `momentum_cycle_enabled` and `momentum_follows_lr_cycle`, exactly as `LRMomentumCycle.values` gates it now (`LRMomentumCycle.swift:279–280`). No cross-key validation is needed.
- `LRKeyframeSchedule: Sendable, Equatable, Codable { keyframes: [LRKeyframe] }`, with one validator `validated() throws -> LRKeyframeSchedule`, the only check used by every writer:
  - 1 … 64 keyframes (Q-3); first `step == 0`; steps strictly increasing;
  - every value finite; `trough > 0`; `peak >= trough`; `peak <= 10` (today's `lr_cycle_max` ceiling); momentum values in 0 … 0.99 (today's follow-key range); `momentumAtPeak <= momentumAtTrough`.
- **Segment selection and evaluation**, defined once: for `s_k <= s < s_{k+1}`, segment k with `f = Double(s − s_k) / Double(s_{k+1} − s_k)`; for `s >= s_last`, the last segment (k = n−2) at `f = 1.0`; a single keyframe is evaluated as `b * pow(b / b, 0.0)` (= `b`).
  - LR bounds geometric: `b_k * pow(b_{k+1} / b_k, f)`.
  - Momentum bounds linear: `m_k + (m_{k+1} − m_k) * f`.
  - Evaluating the last segment at `f = 1.0` (instead of returning the stored value) reproduces the old envelope's arithmetic after its horizon, `start * pow(end / start, 1.0)`, which can differ from `end` in the last bit. So values are exact at every keyframe except the last, where they are `b_{n−2} * pow(b_{n−1} / b_{n−2}, 1.0)`.

### Where it sits in `LRMomentumCycle`

`LRMomentumCycle.values(...)` stays the one function every feed and readout uses (`LRMomentumCycleReadout`, `ChessTrainer.fedSchedule`, `schedule_at_save`). Changes:

- `lrMin`, `lrMax` and `LRMomentumCycleEnvelope` are replaced by `lrKeyframes: LRKeyframeSchedule`.
- The bounds are read at the **trainer step**; the cosine phase stays on the **cycle step** (trainer step − warmup), as today, so the cycle still starts after warmup and an existing run's phase doesn't move.
- `LR = trough · (peak/trough)^frac`, `frac` from `cycleFraction` (period, count, invert) — unchanged.
- Momentum: unchanged gates and guard — `momentumEnabled` (from `momentum_cycle_enabled`) first; then, when following, `high − (high − low) · frac` with `low` / `high` the keyframe-interpolated momentum at the LR peak / trough, and `nil` (static momentum) when the LR channel is off or `high < low` (the guard stays even though the validator makes it unreachable for new schedules); when not following, the independent `momentum_cycle_*` cycle, unchanged.
- `LRMomentumCycle.disabled` keeps both channels off (the behavior fingerprint sets it, `Training/BehaviorFingerprint.swift:274`; a change there would put a `build` gap on every resume).
- Warmup is unchanged: the multiplier ramps the cycled LR; during warmup the phase is held at cycle step 0.

### Converting an old schedule (exact)

One converter, `LRKeyframeSchedule.converting(legacyCycle:envelope:warmupSteps:)`, used for trainer files, sessions, lineage parameter snapshots, `parameters.json` and `UserDefaults`. With warmup `w`, horizon `H` (cycle steps), LR start (max, min), LR end (peakEnd, troughEnd), follow start (startLow, startHigh), follow end (endLow, endHigh):

- `H == 0` (or no envelope at all — a session or file written before the envelope existed, whose `lr_cycle_decay_horizon_steps` resolved `.preFeature(0)`): `[(0, max, min, startLow, startHigh)]`.
- `H > 0, w > 0`: `[(0, max, min, startLow, startHigh), (w, max, min, startLow, startHigh), (w + H, peakEnd, troughEnd, endLow, endHigh)]`.
- `H > 0, w == 0`: `[(0, max, min, startLow, startHigh), (H, peakEnd, troughEnd, endLow, endHigh)]` (no duplicate step).

Arithmetic, step by step against `LRMomentumCycle.swift:259–288`: before `w`, `max * pow(max / max, f)` = `max` (C99 Annex F: `pow(1, y) = 1`) vs today's `max * pow(peakEnd / max, 0.0)` = `max` (`pow(x, ±0) = 1`); on `[w, w + H)` the same expression with `f = Double(s − w) / Double(H)` = today's `decayFraction(cycleStep)`; at and after `w + H`, `max * pow(peakEnd / max, 1.0)` both ways. Momentum: `m + (m − m) · f` = `m` before `w`, then today's lerp. So LR and momentum are **bit-identical at every step** (V-2).

**Old schedules the old code treated as "no cycle".** Today `values` falls back to static LR when `lrMin <= 0`, `lrMax < lrMin`, or `H > 0` with `peakEnd < troughEnd` (or either end ≤ 0); the per-key ranges allow these because nothing but the GUI popover cross-checks. The new validator refuses such keyframes, so the converter maps them to what the old code did:

- an unusable LR pair → `lr_cycle_enabled = false` plus the default keyframes (inert while off) — static LR and static momentum, exactly as before; logged `[PARAM] legacy LR cycle was inactive (<reason>); converted as disabled`.
- follow momentum with `high < low` at **every** step (start and end both inverted) → `momentum_follows_lr_cycle = false` and the follow pair set to the start values (inert) — static momentum, as before; logged likewise.
- follow momentum inverted on **part** of the horizon (old code: static momentum on those steps only) → cannot be expressed; the load refuses with the reason. None of the 34 repo parameters files has this; Phase 3 scans every trainer file and session on this Mac before shipping (V-6) and the plan is revisited if one does.

### The parameters

| Removed (9) | Added (1) |
|---|---|
| `lr_cycle_min`, `lr_cycle_max`, `lr_cycle_peak_end`, `lr_cycle_trough_end`, `lr_cycle_decay_horizon_steps`, `momentum_follow_start_low`, `momentum_follow_start_high`, `momentum_follow_end_low`, `momentum_follow_end_high` | `lr_cycle_keyframes` (list of step, peak, trough, momentum at peak, momentum at trough) |

Kept: `lr_cycle_enabled`, `lr_cycle_period_steps`, `lr_cycle_count`, `lr_cycle_invert`, `momentum_follows_lr_cycle`, every `momentum_cycle_*`, `learning_rate`, `lr_warmup_steps`, `sqrt_batch_scaling_lr`. Registry: 108 → 100 keys, 89 → 81 live-tunable (all nine removed keys are live-tunable); `TrainingParametersTests.test_registry_size` changes with the design (named in the commit).

Default: today's defaults converted (warmup default 1,000), so a fresh run on default settings is unchanged — `(0, 0.1, 0.001, 0.85, 0.95)`, `(1000, 0.1, 0.001, 0.85, 0.95)`, `(1_001_000, 0.0001, 0.000001, 0.90, 0.95)`.

**Value-type plumbing** (CLAUDE.md "Adding / removing / renaming a parameter — full checklist", every step, for each of the 10 keys):

- **Macro (Q-1):** a `@TrainingParameter(default: LRKeyframeSchedule, …)` overload in `Macros.swift` (no `range:`; the schedule's validator is the range); `ValueKind.lrKeyframes` in `TrainingParameterMacro.swift`, inferred from a `LRKeyframeSchedule(...)` default, generating `encode` / `decode`. A `.declaredRangeMaximum` `absentValue:` on it is a macro diagnostic.
- `TrainingParameterDefinition` (`Training/TrainingParameters.swift:132`) gains a `keyframesValidation: Bool` marker (set only by the generated definition), and `validate` calls `LRKeyframeSchedule.validated()` for a `.lrKeyframes` value — the one validator, reached through the same path as every range check.
- `ParameterType` / `ParameterValue` gain `.lrKeyframes(LRKeyframeSchedule)`: `type`, `init(from:)` / `encode(to:)`, `jsonValue` (array of `{"step","peak","trough","momentum_at_peak","momentum_at_trough"}`), `init(jsonValue:id:)`, `displayText`, `storedKind`, `InvalidStoredSetting` display. Every exhaustive switch over them (TrainingParameters.swift, `Training/InvalidStoredSetting.swift`, `Persistence/LineageRecord.swift`) — the compiler finds them.
- `UserDefaults`: a plist array of dictionaries with those five keys; `inspectStored` / `store` gain the case; an unreadable value logs `[PARAM-REJECTED]` and uses the default, as today.
- `parameters.json`: `parametersObject(fromJSON:)` reads nested numbers through the same exact-double path (`ExactJSONNumber`) as top-level ones; `defaultsJSON()`, `defaultsDescriptionLines()`, `defaultsMarkdown()`, `save(to:)` print the list.
- Lineage parameter snapshot: `LineageRecord.Parameters.init(values:)` writes the list as JSON objects through its existing `JSONSerialization` `.sortedKeys` writer (17-digit doubles), so the hash is stable and computed the same way as every other key.
- `ResumeParameterDifferences` compares lists element-wise; `ParameterChange.old` / `new` carry the list.
- `scheduleKeyIDs` (`LRMomentumCycle.swift:375`) and `SafetensorsModelIO.checkScheduleAgrees` (`:300`) move to the new key set.
- `absentValue` (static, per key, `TrainingParameterResolution`): `lr_cycle_keyframes` is `.currentSetting` (the value the old `lr_cycle_min` / `lr_cycle_max` declared). It applies only when a record has neither the new key nor the old ones; a record with old keys is converted and the result is treated as its **saved** value (`SessionParameterResume`, `differences(fromLineage:)`). Never `.refuseExact`, which would mark every old GUI resume not exact (`SessionParameterResume.swift:86`).

### Persistence

- **Trainer state in model files** (`Training/TrainerResumeState.swift`, `TrainerScheduleState`): new metadata key `trainer_lr_schedule` — JSON `{schedule_format: 1, lr_enabled, period_steps, count, invert, keyframes, momentum_enabled, momentum_follows_lr_cycle, momentum_cycle: {...}}` (one object, so no half-written schedule). It replaces `trainer_lr_momentum_cycle` + `trainer_lr_momentum_cycle_envelope` in every new file; `trainer_completed_steps` and `trainer_lr_warmup_steps` are unchanged. A file carrying the new key and either old key is refused. `TrainerScheduleState.MetadataKey` keeps two lists — current and legacy — so `ModelDerivation.requirePlainModelSource` (`Persistence/ModelDerivation.swift:515`) still refuses a trainer file of either form, and `TrainerResumeSnapshot`'s missing-key message names the right keys.
- **session.json**: optional `lrSchedule` field; `lrMomentumCycle` + `lrMomentumCycleEnvelope` still decoded from older sessions and converted. The test-only helpers `resolvedLRMomentumCycle*` (`Persistence/SessionCheckpointFile.swift:521/537`) are removed; their tests' assertions move to the converter.
- **Train-vs-UCI session.json** (`CLI/TrainVsUciSession.swift:208`): same field.
- **Lineage schema 5.** `LineageRecord.currentSchema` 4 → 5 (2–4 still read; never written back as older). Its parameter snapshot may hold the list and lacks the nine removed keys; `configuration.schedule_at_save` adds `lr_peak` and `lr_trough`.
- **Architecture format stays v12** (the schedule isn't architecture; schema 5 already makes older builds refuse new model files, `Persistence/LineageRecord.swift:1255` — Q-2).
- `results.json` (`CLI/CliTrainingRecorder.swift`): the keyframes go in one top-level `lr_schedule` field (not on every row); the per-row `lr_cycle_decay_horizon_steps` (`:988`, `:1119`) is dropped; per-row `lr_cycle_peak` / `lr_cycle_trough` stay.

### Old files (always readable, never rewritten)

- **Trainer files / sessions with the old keys:** converted on load (above); logged `[RESUME] legacy schedule converted: <keyframes>`.
- **`parameters.json` with old keys** (34 repo files, e.g. `parameters.json`, `documentation/parameters.json`, `experiments/*/parameters-*.json`): `--parameters` converts them; `w` is the file's `lr_warmup_steps`, or the current setting when the file omits it (today's rule for missing keys, `CliTrainingConfig.loadAndApplyTransiently`). A file with only some of the nine old keys converts with the missing ones taken from the **old declared defaults** and says so (`[PARAM] legacy LR schedule keys converted (missing …: old defaults)`) — the old build would have taken them from `UserDefaults`, which no longer holds them. Two repo files are partial in this way: `documentation/dashboards/data/v5-source/v5-run-parameters.json` and `v5-bundle-staging-parameters.json` (records of past runs, `lr_cycle_min` / `lr_cycle_max` only). A file with both an old key and `lr_cycle_keyframes` is refused. `--create-parameters-file` writes only the new keys.
- **GUI settings in `UserDefaults`:** see Q-6.
- **Lineage differences:** `differences(fromLineage:)` converts a parent snapshot's old keys before comparing and drops the nine old ids from the parent-only list, so a resume logs no spurious `[RESUME-DIFF]` lines for the renamed keys. (Those lines are informational; renamed keys were never a `params` gap — `.params` comes only from a missing snapshot, a changed feed per step or a changed buffer capacity, `CLI/CorpusReplayRunner.swift:1442–1486`.)

### Resume (O-4)

- **CLI** (`CLI/CorpusReplayRunner.swift:1199–1206`, `CLI/TrainVsUciRunner.swift:280–283`): the checkpoint's schedule and the configured one are both converted, then compared in **behavior form** — the fields that change a fed value: with the LR cycle off, only `learning_rate` / warmup / √batch; while momentum follows, not the independent `momentum_cycle_*` fields; while momentum is off, no momentum fields. Equal → the run continues under it, EXACT. Different → the configured schedule is used, each difference is logged `[RESUME] schedule: <field> checkpoint=… configured=…`, and `ResumeGap.schedule` is added; `--resume-exact` refuses unless `--accept-inexact schedule`.
- **`ChessTrainer.restoreExactly`** (`Training/TrainerResumeState.swift:340–341`) today installs the checkpoint's warmup and cycle; it is called after configuration (`CorpusReplayRunner.swift:1681`, `TrainVsUciRunner.swift:504`). It gains a parameter so the CLI paths restore only the clock (`completedTrainSteps`) and keep the configured schedule; the GUI path keeps installing the session's. V-7 asserts the fed LR follows the configured schedule after a differing resume.
- `ResumeGap.schedule` (`Training/ResumeExactness.swift`): parsing, `[RESUME] NOT EXACT`, `not_exact_items`, the refusal text and the GUI note follow from `allCases`.
- **GUI**: a resume restores the session's schedule into the settings (the configured schedule at load, `SessionParameterResume`); later edits are live changes journalled in `parameter_changes`. No gap.

### Visualization (K4)

1. **`LRScheduleSampler`** (pure, `Training/LRScheduleSampler.swift`): for a step range and bucket count, returns per bucket the peak and trough bounds and the min / max / last **fed** LR and momentum, computed through `LRMomentumCycleReadout.values` (the fed `Float`, with warmup and √batch) — no reimplemented formula — plus the keyframes. Min/max per bucket keeps the oscillation's extremes when a pixel spans many steps. Runs off the main thread (a 2M-step range is 2M evaluations), cancelled and restarted on each draft edit.
2. **GUI chart** — `LRScheduleChart` (new `View`, `App/UpperContentView/LRScheduleChart.swift`) in the LR/Momentum Cycling tab: log LR axis; trainer-step axis with zoom (whole schedule / around the current step); peak and trough bounds as lines with keyframes as labelled points; the fed LR as a min–max band, a line below ~2 px per step; a rule at the current step with its LR, peak and trough; a dashed 1e-4 guide (O-5) with help text; momentum below on a linear axis. It renders the draft before Save. Beside it, `LRKeyframeTable` (new `View`): rows of step / peak / trough / momentum at peak / momentum at trough, add / delete, per-cell messages from the validator (O-6). The "Peak LR start/end", "Trough LR start/end", "Decay steps" and four "Low/High μ start/end" controls are removed.
3. **CLI `--show-lr-schedule [--parameters <file>] [--from <step>] [--to <step>] [--buckets <n>] [--csv <path>]`**: keyframes and a table (or CSV: step_from, step_to, peak, trough, lr_min, lr_max, lr_last, mom_min, mom_max) from the sampler; no GUI; exit 0; defaults without `--parameters`. The HTML step tables and experiment write-ups read this CSV.

### Logging

- `[REPLAY-CYCLE]` / `[VS-UCI-CYCLE]` / the GUI `[PARAM] lr_momentum_cycle` line print the keyframes (`LRMomentumCycleLogFormat`); step lines keep `lrCyc[pk=…,tr=…]`.
- `[PARAM]` warning for a trough below 1e-4 (O-5), once per commit of the key.

### Python

- `scripts/dcm_lineage.py`: `TRAINER_SCHEDULE_KEYS` gains `trainer_lr_schedule` (the new key alone is complete; either old pair is complete); schema-5 reading; `lr_schedule(header)` returning keyframes for either form.
- `documentation/dashboards/tests/test_lineage.py` fixtures for the new key.

## Phases

Each phase: implement, build (no new warnings), run the touched test classes, recheck, commit and push. The full suite runs once at the end, queued for a low-contention window (owner rule 2026-10-10).

1. **Schedule core.** `LRKeyframe` / `LRKeyframeSchedule` (support package; validator, segment evaluation), `LRMomentumCycle` on keyframes with the momentum pair on keyframes, the converter (including `w == 0`, `H == 0` / no envelope, the inactive-schedule cases), `LRScheduleSampler`. Tests V-1, V-2, V-3, V-9.
2. **Parameter value type.** Launch-time conversion of legacy `UserDefaults` (Q-6), macro overload and `ValueKind`, `ParameterValue` / `ParameterType` case, `TrainingParameterDefinition` hook, the 10-key change, `UserDefaults`, `parameters.json` read/write (nested exact doubles), legacy `parameters.json` conversion, lineage snapshot, `parameter_changes`, `ResumeParameterDifferences`. Tests V-4, V-5.
3. **Persistence and resume.** `trainer_lr_schedule` (current and legacy key lists), session.json field (GUI, train-vs-UCI), lineage schema 5, `checkScheduleAgrees`, `ResumeGap.schedule`, behavior-form comparison, `restoreExactly` clock-only on the CLI, `ModelDerivation` refusal; scan this Mac's trainer files and sessions with the converter. Tests V-6, V-7.
4. **CLI `--show-lr-schedule`.** Test V-8.
5. **GUI.** `LRScheduleChart`, `LRKeyframeTable`, popover tab and model validation (`TrainingSettingsPopover.swift` `CyclingTab`, `TrainingSettingsPopoverModel.swift`). View-render test; screenshot compared with `--show-lr-schedule` (V-10).
6. **Docs and readers.** CLAUDE.md (100 / 81 parameters, lineage schema 5, the schedule rules, the `schedule` gap, the owner decisions), a schedule section in `documentation/sampling-parameters.md`, `documentation/UCI.md` resume text, CHANGELOG, `scripts/dcm_lineage.py`, `--help`.

## Validation

- **V-1** Schedule math: exact at every keyframe except the last (which is the last segment at `f = 1.0`); geometric midpoints; segment selection at `s == s_k` exactly; hold after the last; single keyframe; step change (keyframes one step apart). Validator refuses empty, > 64, first step ≠ 0, unsorted, duplicate, non-finite, trough ≤ 0, peak < trough, peak > 10, momentum outside 0…0.99, momentum at peak > at trough — each with its reason.
- **V-2** Equivalence, bit for bit (`==` on the `Double` bits, against an old-envelope reference implementation kept in the test target), LR and momentum, every trainer step 0 … (w + H + 1,000,000): R-replay (`experiments/20261008-zlra-selfplay-lr/parameters-zlra.json`), B-siluall (`experiments/20261005-lr-schedule-ab/parameters-B-relcap-v3.json`), every other repo parameters file with a cycle, the defaults, and synthetic cases: `w == 0`; `H == 0`; `s == w + H` exactly; drifting follow bounds; follow on with `momentum_cycle_enabled` off; the inactive-LR and fully inverted-follow conversions (old code static = new code static).
- **V-3** Follow momentum: the keyframe-interpolated value at the LR peak at every LR peak, at the LR trough at every trough; inverted relative to LR at every step.
- **V-4** Round trips: `parameters.json` write → read → equal (`0.001`-style values exact); `UserDefaults`; lineage snapshot hash stable; `--show-default-parameters` lists the key; `--create-parameters-file` writes only new keys.
- **V-5** Legacy `parameters.json`: every repo file with old keys loads on the new build; the 32 complete ones convert to the schedule V-2 pins; the two partial v5 files convert with the old defaults and say so; a file mixing old and new keys is refused.
- **V-6** Persistence: a new trainer file and session round-trip the schedule bit-exactly; an old trainer file (copy of a current R-replay checkpoint) and an old session decode and convert; a file with both forms is refused; `--derive-model` refuses a trainer file of either form; the scan of this Mac's `Models/` and `Sessions/` converts every file (or names any that can't be).
- **V-7** Resume, run with the **new build** against copies of real R-replay and B-siluall checkpoints and their own parameters files: logs `EXACT`, no `schedule` gap, no `[RESUME-DIFF]` lines for the renamed keys, and the first 1,000 fed LR / momentum values and `[BATCH-HASH]` lines equal an old-build (2503) resume of the same copies. A same-schedule CLI resume is EXACT; a differing one refuses without `--accept-inexact schedule` and with it feeds the configured schedule (not the checkpoint's: `restoreExactly` on the CLI restores only the clock), restores velocity and clock, and records `schedule` in `not_exact_items`; a GUI resume restores the session's schedule into the settings.
- **V-8** `--show-lr-schedule` on `experiments/20261008-zlra-selfplay-lr/parameters-zlra-elite-gentlecycle.json`: fed LR 0.002 at trainer steps 1,000 and 11,000, 0.02 at 6,000 and 16,000; every per-step value lies within its bucket's min/max.
- **V-9** Sampler vs trainer: per-step sampler LR and momentum equal `LRMomentumCycleReadout.values` and `ChessTrainer.fedSchedule` for the same step.
- **V-10** GUI: the chart renders the draft; an edit updates it before Save; per-cell validation messages; screenshot matches the CLI table at three marked steps.
- **V-11** Behavior fingerprint unchanged by the change (no `build` gap on resume): `LRMomentumCycle.disabled` has both channels off.
- **Existing tests.** All pass, including the lineage schema-2/3 fixtures (old keys in their snapshots still decode and hash-verify). Tests that change with the design, each named in its commit, keeping expected numbers wherever the design allows (they are the regression evidence): `LRCycleDecayEnvelopeTests`, `LRMomentumCycleTests`, `LRMomentumCycleReadoutPinTests` (its fed values stay unchanged), `LRCycleWarmupOffsetTests`, `TrainerEffectiveScheduleTests`, `SessionCycleEnvelopeRoundTripTests`, `SessionResumeParameterFallbackTests`, `SessionCheckpointSchemaExpansionTests`, `TrainingSettingsCycleValidationTests`, `ReplayResumeRecordedParametersTests` (its "checkpoint wins" tests encode the rule O-4 changes), `ExactResumeTests` (schedule tests), `TrainerHyperparametersTests`, `TrainingParametersTests.test_registry_size`.

## Decided after the review (owner, 2026-10-12)

- **Q-5 — confirmed.** Q-4 (zero change) applies to a resume whose converted schedule matches the checkpoint's, compared in behavior form (inert differences such as a disabled cycle's numbers don't count). A resume configured with a different schedule follows O-4: the configured schedule is used and the resume reports the `schedule` gap (`--accept-inexact schedule`).
- **Q-6 — agreed.** At launch, when `lr_cycle_keyframes` isn't stored but the old keys are, the GUI converts them once with the same converter, stores the result, logs `[PARAM] legacy LR schedule settings converted`, and leaves the old entries in place (read, never deleted). Test: a `UserDefaults` suite holding only the old keys (`makeTemporaryDefaults()`) yields the converted schedule after launch-time load, and the old entries are still there.

No questions outstanding.
