# LR keyframes plan: peak and trough set by a list of keyframes, and a chart of the schedule

Status (2026-10-12): plan written from a code survey; owner decisions K1–K4 and O-1 … O-6 recorded below. Not approved for implementation. Remaining questions: "Open questions" at the end.

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
- **O-1 — follow-momentum bounds are two fixed values** (momentum at the LR peak, momentum at the LR trough); the slow start → end drift of those bounds is dropped.
- **O-2 — keyframe steps are trainer steps.**
- **O-3 — the first keyframe is at step 0**, required by validation.
- **O-4 — on a CLI exact resume, the configured schedule wins.** The resume is NOT EXACT with a `schedule` gap that `--accept-inexact schedule` must name; weights, velocity, clock, RNG and corpus position are still restored.
- **O-5 — a trough below 1e-4 logs a one-line `[PARAM]` warning** citing this measurement; never a refusal.
- **O-6 — GUI editing is a keyframe table** (step / peak / trough rows, add / delete) beside the chart; chart dragging may come later.

## Design

### The schedule (`Training/LRKeyframeSchedule.swift`, new)

- `struct LRKeyframe: Sendable, Equatable, Codable { step: Int; peak: Double; trough: Double }`.
- `struct LRKeyframeSchedule: Sendable, Equatable, Codable { keyframes: [LRKeyframe] }` with one validator (`validated(_:) throws`), used by every writer:
  - at least one keyframe, at most 64 (a bound for UI and logging; see Q-3);
  - first `step == 0`; steps strictly increasing;
  - every value finite; `trough > 0`; `peak >= trough`; `peak <= 10` (today's `lr_cycle_max` ceiling).
- `bounds(atTrainerStep s) -> (peak, trough)`: between keyframes k and k+1 each bound is `b_k · (b_{k+1}/b_k)^((s − s_k)/(s_{k+1} − s_k))`; at or after the last keyframe, the last values. Computed in `Double`, the same precision as today's `pow`.

### Where it sits in `LRMomentumCycle`

`LRMomentumCycle.values(...)` stays the one function every feed and readout uses (`LRMomentumCycleReadout`, `ChessTrainer.fedSchedule`, `schedule_at_save`). Changes:

- `lrMin`, `lrMax` and `LRMomentumCycleEnvelope` are replaced by `lrKeyframes: LRKeyframeSchedule` and `momentumFollow: (atPeak: Double, atTrough: Double)?` (nil = momentum doesn't follow).
- The bounds are read at the **trainer step**; the cosine phase stays on the **cycle step** (trainer step − warmup), exactly as today, so the cycle still starts after warmup and an existing run's phase doesn't move.
- `LR = trough · (peak/trough)^frac`, `frac` from `cycleFraction` (period, count, invert) — unchanged.
- Follow momentum = `atTrough − (atTrough − atPeak) · frac`.
- The independent momentum cycle (`momentum_cycle_*`) is unchanged.
- Warmup is unchanged: the multiplier ramps the cycled LR, and during warmup the phase is held at cycle step 0.

**Today's envelope is a keyframe list, exactly.** With warmup `w`, horizon `H` (cycle steps), start bounds (max, min), end bounds (peakEnd, troughEnd):

- `H == 0`: `[(0, max, min)]`.
- `H > 0`: `[(0, max, min), (w, max, min), (w + H, peakEnd, troughEnd)]` — the keyframe at `w` holds the start values through warmup, as today's decay fraction is 0 for every cycle step ≤ 0.

On the decay segment the new code evaluates the same expression as today (`start * pow(end / start, Double(s − w) / Double(H))` vs `lrMax * pow(peakEnd / lrMax, decayFraction)`), so LR is bit-identical there and before it. At and after `w + H` the old code returns `start · (end/start)^1`, which can differ from `end` by one unit in the last place; the new code holds the keyframe's value exactly. A test pins both for R-replay's and B-siluall's parameter files at every step 0…2,000,000 (Validation V-2).

Follow momentum converts **exactly only when start == end** for both bounds. Otherwise the converted schedule uses the **start** values, and the change is reported (see "Old files" below).

### The parameters

Every training parameter today is a scalar; `lr_cycle_keyframes` is the first structured one. The new key set:

| Removed (9) | Added (3) |
|---|---|
| `lr_cycle_min`, `lr_cycle_max`, `lr_cycle_peak_end`, `lr_cycle_trough_end`, `lr_cycle_decay_horizon_steps` | `lr_cycle_keyframes` (list) |
| `momentum_follow_start_low`, `momentum_follow_start_high`, `momentum_follow_end_low`, `momentum_follow_end_high` | `momentum_at_lr_peak`, `momentum_at_lr_trough` (double, 0…1) |

Kept: `lr_cycle_enabled`, `lr_cycle_period_steps`, `lr_cycle_count`, `lr_cycle_invert`, `momentum_follows_lr_cycle`, every `momentum_cycle_*`, `learning_rate`, `lr_warmup_steps`, `sqrt_batch_scaling_lr`. Registry: 108 → 102 keys (89 → 83 live-tunable); `TrainingParametersTests.test_registry_size` changes with the design (an edit forced by the approved change, reported in the commit).

Defaults: `lr_cycle_keyframes = [(0, 0.1, 0.001), (1000, 0.1, 0.001), (1_001_000, 0.0001, 0.000001)]` — today's defaults converted, so a fresh run on default settings is unchanged. `momentum_at_lr_peak = 0.85`, `momentum_at_lr_trough = 0.95` (today's start values).

**Value-type plumbing** (CLAUDE.md "Adding / removing / renaming a parameter — full checklist", every step, for each of the 12 keys):

- `ParameterType` / `ParameterValue` gain `.lrKeyframes([LRKeyframe])` (`Training/TrainingParameters.swift`): `type`, `init(from:)` / `encode(to:)`, `jsonValue` (array of `{"step","peak","trough"}` objects), `init(jsonValue:id:)`, `displayText`, `storedKind`.
- The macro (`Packages/TrainingParametersMacro`) gains `ValueKind.lrKeyframes`, inferred from a `default: LRKeyframeSchedule(...)` literal, generating `encode` / `decode` and a definition whose validation is the schedule's validator. One key uses it; it stays inside the one mechanism rather than a hand-written conformance (Q-1).
- `TrainingParameterDefinition.validate` gains the case; `commitAssignment`, `applyOne`, `read` call it like every other key.
- `UserDefaults`: stored as a plist array of `{step, peak, trough}` dictionaries (native, typed); `inspectStored` / `store` gain the case. An unreadable stored value logs `[PARAM-REJECTED]` and uses the default, as today.
- `parameters.json`: `ParameterValue.parametersObject(fromJSON:)` reads nested numbers through the same exact-double path (`ExactJSONNumber`) as top-level ones, so `0.001` in a keyframe isn't rounded on the way in. `defaultsJSON()`, `defaultsDescriptionLines()`, `defaultsMarkdown()`, `save(to:)` print the list.
- Lineage parameter snapshot: `LineageRecord.Parameters.init(values:)` writes the list in canonical JSON (sorted keys, shortest round-trip doubles), so the snapshot hash is stable; `ResumeParameterDifferences` compares lists element-wise.
- `parameter_changes` journal: `ParameterChange.old` / `new` carry the list.
- `TrainingParameterResolution`: `.declaredRangeMaximum` is refused at compile time for a list (the macro rejects it); the new keys use `.preFeature` with the converted value (see "Old files").
- `scheduleKeyIDs` (`LRMomentumCycle.swift:375`) and `SafetensorsModelIO.checkScheduleAgrees` move to the new key set.

### Persistence

- **Trainer state in model files** (`Training/TrainerResumeState.swift`, `TrainerScheduleState`): new metadata key `trainer_lr_schedule` — JSON `{schedule_format: 1, period_steps, count, invert, lr_enabled, keyframes, momentum: {...}}` (the cycle and the follow bounds in one object, so no half-written schedule is possible). It replaces `trainer_lr_momentum_cycle` + `trainer_lr_momentum_cycle_envelope` in every new file. `trainer_completed_steps` and `trainer_lr_warmup_steps` are unchanged. A file carrying both the new key and either old key is refused (no file is ever written that way).
- **session.json**: optional `lrSchedule` field (same JSON); `lrMomentumCycle` + `lrMomentumCycleEnvelope` are still decoded from older sessions and converted. `SessionParameterResume.applyGuiSession` restores the new keys from it. The dead test-only helpers `resolvedLRMomentumCycle*` (`Persistence/SessionCheckpointFile.swift:521/537`) are removed with their tests' assertions moved to the converter.
- **Train-vs-UCI session.json** (`CLI/TrainVsUciSession.swift:207`): same field.
- **Lineage: schema 5.** `LineageRecord.currentSchema` 4 → 5 (schema 2–4 still read; never written back as an older schema). Schema 5's parameter snapshot may hold the list value and lacks the nine removed keys; `configuration.schedule_at_save` adds `lr_peak` and `lr_trough` (the bounds at the save) beside the fed LR and momentum.
- **Architecture format stays v12.** The schedule is training state, not architecture; a version bump would add no gate the lineage schema doesn't already give. Lineage schema 5 already makes every older build refuse new model files (`LineageRecord` decode checks `oldestDecodableSchema...currentSchema`, `Persistence/LineageRecord.swift:1255`), the same consequence a format bump would have — see Q-2.
- `results.json` (`CLI/CliTrainingRecorder.swift`): `lr_cycle_decay_horizon_steps` is replaced by `lr_cycle_keyframes`; `lr_cycle_peak` / `lr_cycle_trough` keep reporting the current bounds.

### Old files (always readable, never rewritten)

- **Trainer files / sessions with the old keys:** decoded and converted to keyframes on load (exact for LR; follow momentum takes the start values). When the old record's follow bounds drift (start ≠ end), the load logs `[RESUME] legacy schedule: momentum follow bounds start=(…) end=(…) → fixed (…)`.
- **`parameters.json` files with the old keys** (41 in this repo, including `parameters.json`, `documentation/parameters.json` and every `experiments/*/parameters-*.json`): `--parameters` accepts them, converts the five LR keys and four follow keys as one group (all nine or none of each group; a partial group is refused, naming the missing keys), logs `[PARAM] legacy LR schedule keys converted: …`, and refuses a file that carries both an old key and its replacement. Experiment "Reproduce" recipes keep working unchanged. `--create-parameters-file` writes only the new keys.
- **Absent keys on resume:** `lr_cycle_keyframes`' `absentValue` is `.preFeature` of the converted value of the record's own old keys (a session or snapshot that predates the cycle entirely resolves to a disabled cycle, as today).
- **Exactness of resuming an old run:** with fixed follow bounds the momentum differs from what the old schedule would have fed when start ≠ end. Both live lines carry such a drift:
  - R-replay: `momentum_follow_start_low` 0.85 → `end_low` 0.90 over 1M cycle steps — at step ~86k the old schedule feeds 0.854 at a peak; the converted one 0.85.
  - B-siluall: 0.85 → 0.90 over 10M — 0.8504 at its 76k resume point.
  
  Such a resume reports the `schedule` gap (O-4 path below), so B-siluall's queued `--resume-exact` (`experiments/20261005-lr-schedule-ab/launch/Bsiluall_resume5_launch.sh`) needs `--accept-inexact schedule` on the new build. An alternative is to convert the drift's value **at the resume step** instead of the start values (Q-4).

### Resume (O-4)

- **CLI** (`CLI/CorpusReplayRunner.swift:1199–1206`, `CLI/TrainVsUciRunner.swift:280–283`): `scheduleDifferences` still compares the configured schedule with the checkpoint's; on a difference the configured one is used (`adoptingSchedule` is no longer called for these keys), the difference is logged per field as `[RESUME] schedule: <field> checkpoint=… configured=…`, and `ResumeGap.schedule` is added. `--resume-exact` refuses unless `--accept-inexact schedule`. Warmup length is part of the schedule comparison; the clock (`trainer_completed_steps`) is always restored.
- `ResumeGap.schedule` (`Training/ResumeExactness.swift`): a new case; parsing, `[RESUME] NOT EXACT`, lineage `not_exact_items`, the refusal text and the GUI status note follow from `allCases`.
- **GUI**: unchanged in kind. A GUI resume restores the session's schedule into the settings (that *is* the configured schedule at load; `SessionParameterResume`), and any later edit is a live change journalled in `parameter_changes` at its trainer step. No gap is needed; the 2026-10-11 trap was CLI-only.

### Visualization (K4)

One computation, two renderers:

1. **`LRScheduleSampler`** (pure, `Training/LRScheduleSampler.swift`): given the cycle (period, count, invert, enabled), keyframes, warmup and the follow bounds, a step range and a bucket count, returns per bucket: peak and trough bounds, min / max / last fed LR across the bucket's steps, the same for momentum; plus the keyframes. It calls `LRMomentumCycle.values` per step (no reimplemented formula). Min/max per bucket means the oscillation's extremes are drawn even when one pixel spans hundreds of steps.
2. **GUI chart** — `LRScheduleChart` (new `View`, `App/UpperContentView/LRScheduleChart.swift`) in the LR/Momentum Cycling tab of the training settings popover:
   - log LR axis, trainer-step axis, zoom (whole schedule / around the current step);
   - peak and trough bounds as lines, keyframes as points labelled with their step;
   - the fed LR as a filled min–max band per bucket, becoming a line when zoomed in below ~2 px per step;
   - a vertical rule at the current trainer step with its LR, peak and trough;
   - a dashed guide at 1e-4 (O-5) with its explanation as help text;
   - momentum below on a linear axis, same x-axis.
   It renders the draft being edited, before Save. Beside it, `LRKeyframeTable` (new `View`): rows of step / peak / trough, add / delete, per-cell validation with the schedule validator's messages (O-6). The current "Peak LR start/end", "Trough LR start/end" and "Decay steps" controls and the four follow start/end fields go; "Momentum at LR peak" / "at LR trough" replace the latter.
3. **CLI `--show-lr-schedule [--parameters <file>] [--from <step>] [--to <step>] [--buckets <n>] [--csv <path>]`**: prints the keyframes and a table (or writes CSV: step_from, step_to, peak, trough, lr_min, lr_max, lr_last, mom_min, mom_max) from the sampler; no GUI, exit 0. With no `--parameters`, the defaults. The HTML step tables and experiment write-ups read this CSV instead of re-deriving the schedule.

### Logging

- `[REPLAY-CYCLE]` / `[VS-UCI-CYCLE]` / the GUI `[PARAM] lr_momentum_cycle` line print the keyframes and follow bounds (`LRMomentumCycleLogFormat`).
- Step lines keep `lrCyc[pk=…,tr=…]` (the current bounds).
- `[PARAM]` warning for a trough below 1e-4 (O-5), once per commit of the key.

### Python

- `scripts/dcm_lineage.py`: `TRAINER_SCHEDULE_KEYS` gains `trainer_lr_schedule` (the "half a schedule" check treats the new key alone as complete, either old pair as complete); schema-5 reading; a `lr_schedule(header)` helper returning the keyframes for either form.
- `documentation/dashboards/tests/test_lineage.py` fixtures for the new key.

## Phases

Each phase: implement, build (no new warnings), run the touched test classes, recheck, commit and push. The full suite runs once at the end, queued for a low-contention window (owner rule 2026-10-10).

1. **Schedule core.** `LRKeyframe`, `LRKeyframeSchedule` (validator, `bounds`), `LRMomentumCycle` on keyframes + fixed follow bounds, the old-envelope converter, `LRScheduleSampler`. Tests V-1, V-2, V-3, V-9.
2. **Parameter value type.** `ParameterValue` / `ParameterType` case, macro `ValueKind`, the 12-key change, UserDefaults, `parameters.json` read/write (nested exact doubles), legacy-key conversion, lineage parameter snapshot, `parameter_changes`, `ResumeParameterDifferences`. Tests V-4, V-5.
3. **Persistence and resume.** `trainer_lr_schedule` metadata, session.json field (GUI and train-vs-UCI), lineage schema 5, `checkScheduleAgrees`, `ResumeGap.schedule`, CLI "configured wins". Tests V-6, V-7.
4. **CLI `--show-lr-schedule`.** Test V-8.
5. **GUI.** `LRScheduleChart`, `LRKeyframeTable`, settings-popover tab, popover model validation. View-render test; screenshot check of the chart against `--show-lr-schedule` output for the same parameters (V-10).
6. **Docs and readers.** CLAUDE.md (parameter count 102 / 83 live, lineage schema 5, the schedule rules, the `schedule` gap), `documentation/sampling-parameters.md`-style schedule section, `documentation/UCI.md` resume text, CHANGELOG, `scripts/dcm_lineage.py`, `--help` text.

## Validation

- **V-1** Schedule math: exact values at keyframes; geometric midpoints; hold after the last; single keyframe = constant; step change (two keyframes one step apart); validator refuses empty, > 64, first step ≠ 0, unsorted, duplicate, non-finite, trough ≤ 0, peak < trough, peak > 10 — each with its reason.
- **V-2** Equivalence: for `experiments/20261008-zlra-selfplay-lr/parameters-zlra.json` (R-replay) and `experiments/20261005-lr-schedule-ab/parameters-B-relcap-v3.json` (B-siluall), and the defaults, the converted keyframes give **bit-identical LR** to the old envelope at every trainer step before `w + H`, and within one unit in the last place at and after it, for steps 0…2,000,000 (old envelope code kept in the test target as the reference). Momentum identical when start == end; when not, equal to the start-value schedule.
- **V-3** Follow momentum: equals `atPeak` at every LR peak and `atTrough` at every LR trough; inverted relative to LR at every step.
- **V-4** Parameter round trips: `parameters.json` write → read → equal (including `0.001`-style values exactly); UserDefaults store → read; lineage snapshot hash stable across encode/decode; `--show-default-parameters` lists the key; `--create-parameters-file` writes only new keys.
- **V-5** Legacy `parameters.json`: each of the 41 repo files loads on the new build and yields the converted schedule; a partial old group and a mixed old+new file are refused with the named keys.
- **V-6** Persistence: a trainer file and a session written by the new build round-trip the schedule bit-exactly; an old trainer file (fixture from a current R-replay checkpoint header) and an old session decode and convert; a file with both the new and an old key is refused.
- **V-7** Resume: CLI exact resume with the same schedule is EXACT; with a different schedule refuses without `--accept-inexact schedule` and with it uses the configured schedule, restores velocity and clock, and records `schedule` in `not_exact_items`; GUI resume restores the session's schedule into the settings.
- **V-8** `--show-lr-schedule`: its CSV for R-elitecyc's parameters shows LR 0.002 at trainer steps 1,000 and 11,000 and 0.02 at 6,000 and 16,000; every per-step LR in a bucket lies within that bucket's min/max.
- **V-9** Sampler vs trainer: `LRScheduleSampler` per-step LR equals `LRMomentumCycleReadout.values` (and so the fed LR) for the same step.
- **V-10** GUI: the chart renders the draft; a keyframe edit updates it before Save; validation messages show per cell; screenshot matches the CLI table at three marked steps.
- All existing tests pass, including the lineage schema-2/3 fixtures (their snapshots keep the old keys and must still decode and hash-verify); the pinned tests listed in the survey (`LRCycleDecayEnvelopeTests`, `ReplayResumeRecordedParametersTests`' "checkpoint wins" tests, `ExactResumeTests` schedule tests, `TrainerHyperparametersTests`, `test_registry_size`) change only as this design requires, each change named in its commit.

## Open questions

- **Q-1 — macro vs hand-written conformance** for the one list-valued key. Recommendation: extend the macro, so every parameter is declared one way.
- **Q-2 — older builds can't open new model files** (lineage schema 5). Consequence: anything running a frozen older build (the Lichess bot, `FrozenBuilds/` runs) can't load a file a new build writes; it must move to the new build first. Same consequence as every past schema bump; noted, not avoidable without writing the old keys too (rejected: two sources of truth).
- **Q-3 — keyframe count limit.** 64 proposed; any real schedule needs a handful. Raise if you foresee more.
- **Q-4 — converting an old drifting follow bound:** the start values (proposed; one fixed rule, reported as a `schedule` gap on exact resume) or the drift's value at the resume step (closer to what the run was feeding, but a converted file's schedule then depends on where it was resumed). Recommendation: start values.
