# LR keyframes plan: peak and trough set by a list of keyframes, and a chart of the schedule

Status (2026-10-12): sketch, owner-requested. Not approved for implementation. Open decisions are listed at the end.

Paths are relative to `DrewsChessMachine/DrewsChessMachine/` unless they start with `DrewsChessMachine/`, `DrewsChessMachineTests/`, `documentation/` or `scripts/`.

## Problem

The LR cycle (`Training/LRMomentumCycle.swift`) oscillates between a trough and a peak; both bounds decay geometrically from a start value (`lr_cycle_min` / `lr_cycle_max`) to an end value (`lr_cycle_trough_end` / `lr_cycle_peak_end`) over one shared horizon (`lr_cycle_decay_horizon_steps`), then hold. That can't express the schedules the 2026-10-11/12 runs point to:

- A two-stage peak (owner's example: 1.0 → 0.1 over 50k steps, then → 0.01 over 150k, then hold).
- A trough that decays on its own timeline (0.01 → 0.0001 over 100k while the peak follows a different one).
- A floor: measured weight motion (`weight_motion.py`, 2026-10-12) shows a trough below about 0.0001–0.0003 barely moves the bf16 working weights (R-replay trough at LR ≈ 0.0007–0.0013: 50% of bf16 weights change per 1,000 steps; extending the measured trend, 0.00001 gives ~0.5% and 0.000001 ~0.05%). The current single decay drives R-replay's trough to 1e-6 by step 1M.

The schedule is also invisible: the only way to see where the LR will be at step N is to compute it by hand, and the evening of 2026-10-11 showed how easy it is to get the phase or the restored schedule wrong.

## Owner decisions (2026-10-12)

- **K1 — keyframes, yes.** The envelope becomes a list of keyframes; each sets the peak and the trough at a trainer step.
- **K2 — one interpolation between keyframes: geometric.** No per-segment function choice (LR acts on ratios; weight motion measured proportional to LR). A hold is two keyframes with equal values; a step change is two keyframes one step apart.
- **K3 — no separate momentum list.** Momentum keeps following the LR cycle's phase (`momentum_follows_lr_cycle`); see open decision O-1 for its low/high bounds.
- **K4 — visualize both** the keyframe envelope over time and where the actual oscillating LR lands.

## Design

### The schedule

`LRKeyframe { step: Int, peak: Double, trough: Double }` (trainer steps; see O-2). `LRKeyframeSchedule { keyframes: [LRKeyframe] }`, validated:

- at least one keyframe; steps strictly increasing; the first at step 0 (see O-3);
- every `trough > 0`, every `peak >= trough`;
- a single keyframe = constant bounds (today's no-decay cycle).

Bounds at step s: before the first keyframe impossible (first is 0); between keyframes k and k+1, each bound interpolates geometrically, `b = b_k · (b_{k+1}/b_k)^((s − s_k)/(s_{k+1} − s_k))`; after the last keyframe, hold the last.

The in-cycle shape is unchanged: `LR = trough · (peak/trough)^frac`, `frac` the cosine fraction from `lr_cycle_period_steps`, `lr_cycle_invert`, `lr_cycle_count` (`LRMomentumCycle.cycleFraction`). Warmup unchanged (`lr_warmup_steps` ramps to the schedule's step-0 value, see O-2).

`LRMomentumCycle.values(forCycleStep:)` keeps its role as the one function every feed and readout uses (`LRMomentumCycleReadout`); only the bound computation changes. The existing envelope is exactly the two-keyframe case:

- `[ (0, lr_cycle_max, lr_cycle_min), (horizon, lr_cycle_peak_end, lr_cycle_trough_end) ]` (in cycle steps; trainer steps add the warmup), or one keyframe when the horizon is 0.

So the keyframes **replace** `lr_cycle_min`, `lr_cycle_max`, `lr_cycle_peak_end`, `lr_cycle_trough_end` and `lr_cycle_decay_horizon_steps`; there is no second mechanism.

### The parameter

Every training parameter today is a scalar (`ParameterValue`: bool / int / double / uint64; the macro's `ValueKind` matches). The keyframe list is the first structured value:

- `ParameterValue.lrKeyframes([LRKeyframe])` and a macro `ValueKind` for it, with `Codable` JSON `[{"step": 0, "peak": 1.0, "trough": 0.001}, …]` in `parameters.json`, `UserDefaults` (JSON data), sessions and the lineage parameter snapshot. Validation is the schedule's own validator, called by the one declared-range path (`commitAssignment`, `applyOne`, `read`).
- One new key `lr_cycle_keyframes`; the five scalar keys above are removed (CLAUDE.md "Adding / removing / renaming a parameter — full checklist" walked for each). `absentValue`: a session or `parameters.json` without the key resolves to the keyframes its five scalar keys describe (the exact two-keyframe equivalence above), so every existing session, checkpoint and parameters file keeps its schedule bit for bit. This is reading old files (owner rule: old files always readable), not a migration that rewrites them.
- Live-tunable, like the scalars it replaces: the schedule is a pure function of the step, so an edit takes effect at the next step and is journalled in `configuration.parameter_changes`.

### Momentum

Unchanged in shape (K3): follow mode reads the LR cycle's phase. The follow bounds (`momentum_follow_start_low/high`, `momentum_follow_end_low/high`) currently drift linearly over the same `lr_cycle_decay_horizon_steps` that is being removed — see O-1.

### Resume

Exact resumes restore the checkpoint's whole schedule (`TrainerScheduleState`), which is why `--resume-exact --parameters <new cycle>` silently kept the old schedule on 2026-10-11 (`[REPLAY-RESUME] WARNING lr_momentum_cycle: … restoring the checkpoint's value`) and the gentle cycle needed a branch. Keyframes make an intentional mid-run schedule change common; see O-4.

### Visualization (K4)

One computation, two renderers, so the chart can't disagree with the trainer:

1. **`LRScheduleSampler`** (pure): given the schedule, period, invert, count, warmup and a step range, returns the keyframes, the peak and trough envelope sampled per step bucket, the actual LR per step (min/max per bucket when a pixel spans many steps, so the oscillation's extremes are drawn, not aliased), and follow-mode momentum. Calls `LRMomentumCycle` itself — no reimplemented formula.
2. **GUI chart** (Swift Charts, new `View` in `App/UpperContentView/`, in the LR/Momentum Cycling section of the training settings popover):
   - log-scale LR axis, trainer-step axis;
   - peak and trough envelopes as lines, keyframes as marked points (editable in the keyframe table beside it);
   - the actual LR as a filled band per bucket (or the line itself when zoomed in to < ~2 px per step), so where each oscillation lands is visible;
   - the current trainer step as a vertical rule, with its LR;
   - optional horizontal guide at the practical floor (≈ 1e-4) with a tooltip pointing at this plan's measurement;
   - momentum (follow mode) on a second small chart below, sharing the x-axis.
   It previews the draft being edited, before Save.
3. **CLI `--show-lr-schedule [--parameters <file>] [--steps <n>] [--csv <path>]`**: prints the keyframes and a step / LR / peak / trough / momentum table (or CSV) from the same sampler, no GUI. The HTML step tables and experiment write-ups read this CSV instead of re-deriving the schedule in Python.

### Logging and lineage

- `[REPLAY-CYCLE]` / `[STATS]` / the GUI cycle readout print the keyframes in place of `trough … peak …^period decay=…`.
- `configuration.schedule_at_save` (lineage schema 3) records the fed LR, momentum, and the current peak and trough; the keyframe list itself is in the parameter snapshot. Python reader `scripts/dcm_lineage.py` learns the new value type.

### Tests

- Sampler and bounds: exact values at keyframes, geometric midpoints, hold before/after, single keyframe, step change (two keyframes one step apart).
- Two-keyframe equivalence: for R-replay's and B-siluall's parameter files, the keyframe path gives bit-identical LR and momentum to the current envelope at every step 0…2M (regression guard for every existing run).
- Validation: empty list, unsorted / duplicate steps, first step ≠ 0, trough ≤ 0, peak < trough — each refused with the reason.
- Persistence round trips: `parameters.json`, `UserDefaults`, session save/load, lineage snapshot; absent key → scalar-derived keyframes.
- Resume: exact resume restores the saved keyframes; the O-4 behavior, whichever is chosen.
- Chart: the sampler's bucketed min/max contain every per-step LR in the bucket.

### Phases

1. Schedule type + sampler + `LRMomentumCycle` on keyframes, with the equivalence tests.
2. Parameter value type, persistence, absent-value resolution, logging, lineage.
3. `--show-lr-schedule` CLI.
4. GUI chart + keyframe table editor.
5. Docs: `documentation/sampling-parameters.md`-style section for the schedule, CLAUDE.md parameter list, CHANGELOG.

## Open decisions

- **O-1 — momentum follow bounds.** Today they drift linearly from start to end values over the decay horizon being removed. Options: (a) fixed low/high (drop the drift: two parameters instead of four); (b) drift across the keyframes' span (first to last keyframe); (c) keep a separate `momentum_follow_drift_steps`. Recommendation: (a) — no run has shown the drift matters, and (b)/(c) reintroduce a second timeline.
- **O-2 — keyframe step axis.** Trainer steps (matches every log line, table and checkpoint name; warmup ramps to keyframe 0's value) or cycle steps (today's horizon counts from the end of warmup). Recommendation: trainer steps.
- **O-3 — first keyframe at step 0 required, or hold the first keyframe's values before it?** Recommendation: required, so the schedule has no implicit region.
- **O-4 — changing the schedule on an exact resume.** Today the checkpoint's schedule always wins. Options: (a) keep that, and a new schedule needs a branch (zero velocity, warmup, new lineage); (b) the configured schedule wins and the resume is NOT EXACT with a `schedule` gap that `--accept-inexact schedule` must name. Recommendation: (b) — it's the honest record of what happened and keeps velocity and the corpus position.
- **O-5 — a floor guard.** Warn (not refuse) when a keyframe's trough is below 1e-4, citing this plan's measurement? Recommendation: a one-line `[PARAM]` warning, no refusal.
- **O-6 — GUI editing.** A keyframe table (step / peak / trough rows, add / delete) beside the chart, or drag points on the chart. Recommendation: the table first; dragging later if wanted.
