# Training Dynamics Plan — cyclical LR/momentum + measurement

Status: **PARTIALLY SHIPPED** (2026-06-13 audit). Item 0 (EMA overlay) AND
Section 3 (cyclical LR / inverse-coupled momentum, `Training/LRMomentumCycle.swift`)
are **implemented**; the Tier-1/2 measurement scaffolding is designed, not all built. Separate from `GPU_UTILIZATION_PLAN.md` (that's throughput; this is
learning dynamics + how we measure them).

## Motivation

Overnight analysis showed a long plateau: arenas hovered ~50–53% (candidate ≈
champion) before one promotion landed; the trainer's tactical strength
(Lichess probe) dipped and recovered rather than climbing. We want a principled
lever to push off plateaus — **cyclical LR with inverse-coupled momentum** (Leslie
Smith super-convergence / 1cycle: [arXiv 1708.07120](https://arxiv.org/abs/1708.07120),
[arXiv 1803.09820](https://arxiv.org/abs/1803.09820)) — and, crucially, a
**measurement** we can trust to tell whether a change helped.

## 1. Measurement — the foundation

**The problem:** Smith-style papers compare hyperparameters by *test loss on a
fixed dataset after a fixed epoch budget*. We have neither — the self-play buffer
is non-stationary and our primary signal (arena candidate-vs-champion) is
**relative and biased**. The literature agrees: training/self-play Elo "can be a
misleading indicator … influenced by training bias" (KataGo, Wu 2019,
[arXiv 1902.10565](https://arxiv.org/abs/1902.10565); also the adversarial-Go
NeurIPS'22 workshop paper). More reliable: fixed anchors and held-out sets.

Plan, in tiers:

- **Tier 1 — make the Lichess probe a real test-loss instrument** (it's our best
  held-out, self-play-independent signal):
  - Report **NLL as the headline** metric, not argmax-count — NLL is continuous
    and far lower-variance than discrete "N/200 hits" (which bounced ±2.5pp while
    NLL barely moved).
  - **EMA-smooth the trend** — DONE (see Item 0).
  - **Grow the set to 1000 puzzles**, evaluated as a **single forward batch**
    (the inference executable already caches per batch size, so it's one extra
    infrequent forward). Cuts sampling variance at the root.
- **Tier 2 — fixed-anchor gauntlet:** alongside candidate-vs-champion, periodically
  play the live net vs a **frozen reference net** (pinned at experiment start) for
  an *absolute* Elo trajectory. The moving-champion arena structurally can't tell
  you "is it stronger in absolute terms."
- **Tier 3 (optional, later):** ancestor rating pool (KataGo-style). 
- **NOT useful now:** external engines (Stockfish / Sloppy). The net loses ~100%,
  so the metric is *floored* — zero gradient, can't distinguish better from worse.
  Revisit only once we score > 0 against them.

## 2. Prepared-batch replay → deterministic A/B + param sweeps

The cleanest answer to "we're not supervised": **save a fixed sequence of prepared
batches to disk**, then **replay the identical sequence** under different
LR/momentum cycles. That converts the comparison into the fixed-dataset test-loss
A/B the papers do, and **removes the self-play-data confound** (cycle-on and
cycle-off would otherwise generate different games).

- A **CLI/JSON-driven sweep mode** runs the same saved sequence under N parameter
  sets and records results (probe NLL, etc.) — the direct analog of "run the same
  epochs, compare params."
- **Storage cost is the main concern.** A prepared batch at B=4096 is ~111 MB, and
  the legal mask (4096×4864 fp32 ≈ 80 MB) dominates. Mitigate: **bit-pack the mask**
  (0/1 → ~2.5 MB) or regenerate it from boards on load; store boards compactly.
  Size the saved sequence to a *comparison window* (hundreds of batches), not
  millions.

## 3. Cyclical LR + inverse-coupled momentum

**Key enabler:** the trainer already feeds **both `lr` and `momentum` as live
scalar placeholders every step** (`buildFeeds`, alongside the warmup × √batch
multipliers). So cycling is purely *what values we feed* — **no graph change, no
recompile.** The existing warmup-multiplier code is the pattern to extend.

Design:

**Deterministic phase from the global step (no cycle state).** The phase is a
pure function of the trainer's global step number — `phase = (globalStep mod
period) / period ∈ [0,1)` — so the cycle carries **zero state of its own**.
Stop/resume is automatic: `globalStep` is already persisted, so on resume the
phase continues with no discontinuity and nothing extra to rebuild. (Mirrors how
the warmup multiplier already keys off the step counter.) Repeating cycles
(original CLR style), **not** 1cycle — 1cycle needs a *known total training
length* to place its single cycle + annihilation tail, which open-ended self-play
doesn't have. We adopt the repeating form as a plateau-breaking lever instead.

**Smooth up-then-down via cosine, per cycle.** With
`frac = 0.5·(1 − cos(2π·phase))` the value sits at `min` at the period
boundaries and reaches `max` at the midpoint, smoothly (no triangular corner).
An `invert` flag uses `1 − frac` to flip the waveform — that's how momentum is
made inverse to LR at equal periods (see below).

**Interpolation differs by parameter — LR geometric, momentum linear:**
- **LR (geometric / multiplicative):** `lr = lrMin · (lrMax/lrMin)^frac`. LR's
  effect is ~scale-invariant, so a *log-space* sweep spends equal time per
  multiplicative octave; a linear ramp would spend almost all its time at high LR.
  Requires `lrCycleMin > 0` (geometric interpolation is undefined at zero) —
  enforce via the `@TrainingParameter` range (`lrCycleMax ≥ lrCycleMin > 0`).
- **Momentum (linear):** `m = momMin + (momMax − momMin)·frac`. Momentum's useful
  range (~0.85–0.95) is modest, so linear interpolation is fine (no need to think
  in `1/(1−μ)` window-space unless we push μ toward 0.99).

**Two fully independent blocks (LR, momentum), each self-contained.** Deliberately
*not* sharing period/count between LR and momentum, so either can run without the
other, or one can cycle at a multiple of the other's period for experiments. Smith's
inverse coupling (high LR ↔ low momentum) is therefore **not automatic** — it's
recovered by setting momentum's `invert = true` at an equal period, which flips its
waveform so it troughs when LR peaks. At unequal periods you get whatever phase
relationship the ratio produces.

**Absolute endpoints, not multipliers over a base.** `lrCycleMin/Max` and
`momentumCycleMin/Max` are **absolute values**, computed and assigned each step,
and the UI shows the **live effective value** directly (no "calculated range"
widget needed). Consequences:
- **Warmup and √batch still multiply on top** of the cycled absolute LR:
  `effectiveLR = cycledAbsLR · warmupMul · √batchMul`. Warmup still protects the
  first-N-steps start (during warmup the displayed LR sits *below* `lrCycleMin`,
  then settles into the [min,max] band × the constant √batch factor). The UI
  reads this final `effectiveLR`. Momentum has no warmup/√batch multipliers, so
  its absolute min/max assign directly.
- **Enabling LR cycling overrides the exponential-decay baseline** — with absolute
  endpoints the LR oscillates in a *fixed* band with no long-term downward drift.
  Decay-of-the-cycle (Smith `triangular2` / shrinking endpoints) is a **follow-on**,
  not in v1.

**`cycleCount` completion:** `0 = unbounded` (the default for open-ended
self-play). When `count ≠ 0`, once `globalStep / period ≥ count` the channel
**freezes at the cycle boundary** (`frac = 0`) thereafter — i.e. clamp phase to 0,
which is deterministic and state-free like the rest. Note this respects `invert`:
the frozen state is LR = `lrMin` and momentum = `momMax` (when `momentumCycleInvert
= true`) — exactly the low-LR / high-momentum converged regime you want training to
settle into after cycling stops. (Freezing at a literal `min` for both would leave
momentum pinned *low*, which is backwards under inverse coupling.)

**Params** (`@TrainingParameter`, liveTunable), two independent blocks:
```
// LR cycle — geometric interpolation, absolute LRs
lrCycleEnabled       : Bool
lrCyclePeriodSteps   : Int      // full up-down period
lrCycleCount         : Int      // 0 = unbounded
lrCycleMin           : Double   // absolute LR
lrCycleMax           : Double   // absolute LR
lrCycleInvert        : Bool     // default false

// Momentum cycle — linear interpolation, absolute momentum
momentumCycleEnabled     : Bool
momentumCyclePeriodSteps : Int
momentumCycleCount       : Int  // 0 = unbounded
momentumCycleMin         : Double
momentumCycleMax         : Double
momentumCycleInvert      : Bool // default true → inverse of LR at equal period
```

**Picking a period.** Self-play has no epoch, but **replay-buffer turnover** is the
natural analog: one "pass over the current data distribution" ≈ `bufferSize / batch`
steps. Smith set CLR `stepsize` to 2–8× iterations-per-epoch, so a sensible default
is `period ≈ 2–8 × (bufferSize / batch)`. Tune from there by watching the probe.

**Caveats:** the high-LR phase generates noisier self-play during its window (the
promotion gate bounds the risk — a degraded net won't promote); phase-align the
probe measurement to read peaks/troughs deliberately.

## 0. EMA overlay on the Lichess trend charts — IMPLEMENTED

`LichessProbeOverallTrendChart` now has an **"EMA overlay" toggle** (on by default)
+ a **span stepper** (3–200, default 25). When on, an EMA-smoothed line is drawn
over the raw NLL and puzzle-Elo series in a contrasting color, and the EMA values
are included in the y-domain. Pure `ema(_:span:)` helper (static, testable). This
is the prerequisite for reading any cycling experiment off the probe.

## Sequencing

1. EMA overlay — **done**.
2. Probe as NLL-headline test-loss + grow to 1000 puzzles + fixed-anchor gauntlet.
3. Prepared-batch replay + CLI/JSON sweep mode.
4. Cyclical LR + inverse momentum (cheap given the live scalars), then measure.

The **GPU pipeline (`GPU_UTILIZATION_PLAN.md` Phase 3)** is orthogonal (throughput,
not dynamics) and proceeds in parallel.
