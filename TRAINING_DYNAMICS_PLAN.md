# Training Dynamics Plan — cyclical LR/momentum + measurement

Status: **PLAN.** Item 0 (EMA overlay) is **implemented**; the rest is designed,
not started. Separate from `GPU_UTILIZATION_PLAN.md` (that's throughput; this is
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
- **Deterministic phase from the step counter:** `phase01 = f(completedTrainSteps)`
  — a pure function of the step number, so stop/resume is seamless with no
  discontinuity (mirrors how the warmup multiplier already keys off
  `completedTrainSteps`).
- **Shape: repeating cosine** (smooth; Smith's original CLR was triangular/linear,
  Gugger's 1cycle / SGDR is cosine — cosine is the modern default and avoids the
  triangular peak's corner). **Multiplicative LR range** (`lr_max = N · lr_min`).
- **LR stays the source-of-truth scalar** fed to the graph; `lrMin`/`lrMax` + phase
  *drive* it, and the UI reads the live computed value. (Easier wiring + visible.)
- **Momentum inverse:** `μ_max` (~0.95) at `lr_min`, `μ_min` (~0.85) at `lr_max`,
  interpolated — high LR gets low momentum (stability), low LR gets high momentum
  (acceleration), per Smith.
- **Params** (`@TrainingParameter`, liveTunable): `cycleEnabled`,
  `cycleLengthSteps`, `lrMin`/`lrMax` (or a multiplier range over base),
  `momentumMin`/`momentumMax`. Compose *multiplicatively* with the existing
  warmup/√batch multipliers.
- **Caveats:** the high-LR phase generates noisier self-play during its window
  (the promotion gate bounds the risk — a degraded net won't promote); phase-align
  the probe measurement to read peaks/troughs deliberately.

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
