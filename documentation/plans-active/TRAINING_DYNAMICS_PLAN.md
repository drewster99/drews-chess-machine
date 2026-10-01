# Training Dynamics Plan — cyclical LR/momentum + measurement

Status: **PARTIALLY SHIPPED** (re-audited 2026-10-01; previous audit 2026-06-13).
Separate from `GPU_UTILIZATION_PLAN.md` (that's throughput; this is learning
dynamics + how we measure them).

Shipped:
- **§0 EMA overlay** on the Lichess trend charts — `540bbda`.
- **Tier 1, NLL as the headline** — the probe reports NLL and puzzle-Elo
  (`bd6d144`), and the trend charts plot them.
- **Tier 1, bigger puzzle set** — **shipped differently**: instead of growing
  the 200-puzzle set to 1000, a separate ~4,435-puzzle **wide** set (rating
  400–3200) runs alongside the unchanged 200-set, both evaluated in one batched
  forward pass (`9818b66`; see `../plans-completed/LICHESS_WIDE_PROBE_PLAN.md`).
- **§2 prepared-batch replay** — **shipped differently** as **corpus replay**
  (`--replay-corpus` / `--epochs`, `c4243b5`): a fixed *game corpus* is replayed
  through the self-play staging path into the replay buffer, rather than saving
  prepared batches to disk. This removes the self-play-data confound the section
  was after, without the per-batch storage cost discussed below.
- **§3 cyclical LR + inverse-coupled momentum** — `95e0d54`
  (`Training/LRMomentumCycle.swift`).
- **§3 follow-on: decaying cycle** — `0a4792d` (2026-09-29): the LR cycle's peak
  and trough decay geometrically over a horizon of cycle steps and then hold;
  momentum can follow the LR cycle's phase (inverted) with bounds that drift
  linearly over the same horizon. Horizon 0 is the original fixed-band behaviour.

Not started:
- **Tier 2 fixed-anchor gauntlet** (live net vs a frozen reference net).
- **Tier 3 ancestor rating pool.**
- **Native N-arm sweep mode** (one CLI/JSON invocation running the same sequence
  under N parameter sets). Partially covered today by running corpus replay once
  per `parameters.json` file and comparing the results; there is no built-in
  multi-arm driver. (The existing `--arch-sweep` is a block-count depth sweep,
  not this.)

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
    — **DONE** (`bd6d144`: NLL + puzzle-Elo metrics on the probe).
  - **EMA-smooth the trend** — DONE (see Item 0).
  - **Grow the set to 1000 puzzles**, evaluated as a **single forward batch**
    (the inference executable already caches per batch size, so it's one extra
    infrequent forward). Cuts sampling variance at the root.
    — **DONE, differently** (`9818b66`): the 200-set was left untouched and a
    separate ~4,435-puzzle wide set was added; both run in one batched forward.
    See `../plans-completed/LICHESS_WIDE_PROBE_PLAN.md`.
- **Tier 2 — fixed-anchor gauntlet** (*not started*)**:** alongside candidate-vs-champion, periodically
  play the live net vs a **frozen reference net** (pinned at experiment start) for
  an *absolute* Elo trajectory. The moving-champion arena structurally can't tell
  you "is it stronger in absolute terms."
- **Tier 3 (optional, later; not started):** ancestor rating pool (KataGo-style).
- **NOT useful now:** external engines (Stockfish / Sloppy). The net loses ~100%,
  so the metric is *floored* — zero gradient, can't distinguish better from worse.
  Revisit only once we score > 0 against them.

## 2. Prepared-batch replay → deterministic A/B + param sweeps

> **Shipped differently** as corpus replay (`--replay-corpus` / `--epochs`,
> `c4243b5`). A fixed game corpus — not a fixed sequence of prepared batches — is
> replayed through the self-play `ActiveGame` staging path into the replay
> buffer, so the per-batch storage concern below never arose. The CLI/JSON
> N-arm sweep driver was **not** built; A/B comparisons are run as separate
> corpus-replay runs, one per `parameters.json`.

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
  not in v1. — **Follow-on SHIPPED** (`0a4792d`, 2026-09-29): a decaying envelope
  shrinks the LR peak and trough geometrically over a horizon of cycle steps and
  then holds; momentum can follow the LR cycle's phase (inverted) with bounds that
  drift linearly over the same horizon. A horizon of 0 keeps the fixed band.

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

1. EMA overlay — **done** (`540bbda`).
2. Probe as NLL-headline test-loss + grow to 1000 puzzles + fixed-anchor gauntlet.
   — NLL headline **done** (`bd6d144`); bigger set **done differently** as the
   ~4,435-puzzle wide set (`9818b66`); fixed-anchor gauntlet **not started**.
3. Prepared-batch replay + CLI/JSON sweep mode.
   — **done differently** as corpus replay (`c4243b5`); native N-arm sweep
   **not started** (per-file runs instead).
4. Cyclical LR + inverse momentum (cheap given the live scalars), then measure.
   — **done** (`95e0d54`), plus the decaying-cycle follow-on (`0a4792d`).

The **GPU pipeline (`GPU_UTILIZATION_PLAN.md`)** is orthogonal (throughput, not
dynamics). *(The original note here said "Phase 3 proceeds in parallel"; that
was a scheduling note from 2026-06-03 and no longer describes anything — see
that plan's own status line for where it stands.)*
