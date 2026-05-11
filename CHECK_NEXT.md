# CHECK_NEXT — diverging trainer in build 873 (git 19d8ab6)

Captured 2026-05-09 ~15:40 CDT, mid-session. Session log:
`~/Library/Logs/DrewsChessMachine/dcm_log_20260509-135538.txt`
(launched 13:55, build 873, branch `tmp/possibly-crap`, no resume — built fresh, then Play-and-Train).

## Update (2026-05-11): a *separate* pathology — illegal-mass saturation — found and fixed

The rest of this doc is about the **divergence** failure mode (logit blowup → `pLoss = −64,868`), which the `19d8ab6` / `4ef1a26` / `419c07f` fixes cured: in the affected runs `pLogitAbsMax≈6.9`, `gNorm≈1.1`, `pLoss≈+0.43` — bounded and healthy. But those same runs sit at `pIllM=0.9967` / `legalMass≈0.003` / `pLossWin≈+9.1`, flat across sessions and resumes — a *different*, structurally simpler failure: the policy head never learns down from its uniform-init illegal mass (~99.4% of a 4864-cell softmax is on illegal cells at init). Visible symptom: the probe shows one move-type-shaped illegal cell ("f4-e6") as the top policy output with 70%+ across unrelated positions, because the fully-conv head's per-channel bias is shared across all 64 squares so one inflated channel dominates everywhere.

Root cause: of the four `totalLoss` terms, three give ~zero gradient on illegal logits — the CE was over `maskedLogits` (commit `acc5340`, masking it *backfired*: `∂CE/∂(illegal logit) ≈ softmax_masked − target ≈ 0`), the entropy bonus is over the masked softmax (correctly — `19d8ab6` — so it can't reward spreading onto illegal cells), and the softmax-mass `illegalMassPenalty` has `∂/∂logit ∝ p`, which → 0 once illegal mass ≈ 1 (it saturates at exactly the pathology it exists to fix). Decoupled weight decay doesn't favor legal cells. So nothing pushed illegal logits down.

Fix (2026-05-11): the CE is back over the **raw** policy logits (`network.policyOutput`). The smoothed target is zero on illegal cells so the CE still can't reward illegal mass, but now `∂CE/∂(illegal logit) = softmax_raw(illegal) ∈ [0, 1]` — bounded, always points down, vanishes only at the correct fixed point. `maskedLogits` stays for the entropy bonus / move selection / diagnostics. `illegal_mass_weight` left at 1.0 (now mostly redundant, harmless). Regression test: `DrewsChessMachineTests/PolicyLossIllegalMassGradientTests.swift`. See `CHANGELOG.md` (2026-05-11 entry) and the structural-piece-3 comment block in `ChessTrainer.buildTrainingOps`. Watch the next fresh run: `pIllM` should fall from ~0.994 toward ≪0.1, `legalMass` rise toward ~1, `pLossWin` start ~9 and decrease, `playedMoveProb` rise toward ~0.90, `pLogitAbsMax`/`gNorm` stay bounded.

## Validation result (2026-05-10): hypothesis CONFIRMED — diverges with no promotions

**Experimental run:** build 875, session log `~/Library/Logs/DrewsChessMachine/dcm_log_20260509-155952.txt`. Launched 16:00 on 2026-05-09, stopped 14:59 on 2026-05-10 (~23 hours, single Play-and-Train). 3 automatic arenas fired (scores 0.43 / 0.48 / 0.41, all kept) — **no promotion event ever occurred**. Trainer ID stayed `20260509-12-o1mU-1` and champion `20260509-12-o1mU` for the entire run (no `-N` generation increments).

The divergence happened anyway:

| step | t-since-start | pLoss | gNorm | pLogitAbsMax | pwNorm | pEnt | top1Legal | event |
|------|---------------|-------|-------|--------------|--------|------|-----------|-------|
| 500 (warmup end) | 0:14 | +0.53 | 4.1 | 8.4 | 12.23 | 1.90 | 0.00 | healthy |
| 2675 | 1:11 | -27 | 40 | 75 | 12.38 | 0.94 | 0.58 | first `[ALARM] policy may be collapsing` |
| ~2860 | 1:16 | — | 130 | — | — | 0.50 | — | `[ALARM] Critical Training Divergence` |
| 3304 | 1:28 | -571 | **462** | **473** | 13.57 | 0.17 | 0.51 | past the inflection |
| 33971 (final) | 22:57 | **-64,868** | **34,160** | **45,710** | **68.09** | 0.13 | 0.62 | sustained runaway |

**vs. the previous (with-promotion) run:**

- Previous run: gNorm 5 → 173 over steps 1918 → 2962, reached gNorm 344 by step 3864.
- This run: gNorm 4 → 130 over steps 500 → 2860, reached gNorm 462 by step 3304.

The no-arena run diverged **faster and harder** than the with-promotion run. The "promotion as 2-3× amplifier" caveat in the hypothesis below was wrong — without arenas, the divergence is more aggressive, not less. Working theory: arena pauses (~38s each) actually act as inadvertent cooldowns, and the post-promotion buffer churn introduces enough new-distribution positions to slow the over-fitting-to-a-static-buffer mode.

**Differences in the failure mode (interesting):**

- Previous run had `pEnt` floored around 1.4 even at gNorm=300+ (policy still spread across many cells).
- This run had `pEnt` fully collapsing to 0.13, with `top1Legal=0.76` at the end — the policy degenerated to a near-delta on a single legal cell.

So promotion *was* doing one useful thing: periodically refreshing the buffer with the new champion's games kept the trainer from over-fitting to a static distribution. Without that perturbation, the trainer over-fits the static buffer and collapses to a degenerate delta policy. Both modes are pathological; the no-promotion mode is structurally simpler.

**Alarm system changes between runs:**

- `policyEntropyAlarmThreshold` was retuned from 5.0 → 1.0 (someone fixed the masked-entropy unreachability issue noted in the original hypothesis). Alarm correctly fired at step 2675.
- New alarms `[ALARM] Training Divergence Warning` (gNorm trigger ~50) and `[ALARM] Critical Training Divergence` (gNorm trigger ~130) appeared in this run. These are exactly the kind of gNorm-tripwire alarms recommended below.
- `arenaAutoSec=` is now in the `[STATS]` line in build 875 (per user request after this analysis — the field was added to `cfgStr` in `UpperContentView.swift` ~line 7378).

**Conclusion:** the runaway is purely internal to the trainer. The dropped per-position CE clamp from `19d8ab6` is the prime suspect, with momentum integration amplifying the consistent gradient direction past the clip threshold. Promotion is not the cause and is not even a meaningful amplifier — it may, paradoxically, have been a mild *stabilizer* via buffer-distribution refresh.

The candidate fixes below (cap `pLogitAbsMax` directly, or per-position CE-*gradient* clamp instead of value clamp) remain the right interventions.

### Side effect: divergence also slows the training loop itself

The 23-hour no-arena run degraded the training loop's *throughput* on top of destroying the model. This is a secondary reason to bound logit magnitude.

**Training rate degradation:**

| t | step | trainRate | prod (pos/sec) | trMs/move | gpu/step |
|---|------|-----------|----------------|-----------|----------|
| 0:13 (warmup end) | 500 | **9.4M/hr (peak)** | 2,651 | 0.38 | 1129 μs |
| 8:42 | 15,855 | 5.9M/hr | 1,691 | 0.47 | 1093 μs |
| 17:16 | 26,708 | **4.9M/hr** | 1,425 | 1.22 | 1087 μs |

A 47% drop in training throughput from peak. GPU per-step is essentially flat — the GPU compute itself isn't the bottleneck. `trMs/move` (CPU-side around-step plumbing) tripled. Replay ratio controller is hitting its bounds (`cur=0.96` against an inflated `target=4.34` — auto-adjuster wants way more consumption than it gets).

**Slow-tick growth:** 22,831 `[TICK-SLOW]` reports in the session (50ms threshold). Mean tickMs = 3,228 ms; max = 13,760 ms. 76% of slow ticks were >1s; 82% >500ms. `mainActorEnqueueWaitMs` stays at 1-2ms throughout — main actor is **not** contended, the tick-work itself is growing slow.

| t | tickMs |
|---|--------|
| 16:00 (start) | 53 |
| 16:53 (~1h) | 934 |
| 19:45 (~4h) | 2,819 |
| 01:20 (~9h) | 3,605 |
| 09:18 (~17h) | 6,527 |
| 14:59 (~23h) | 7,822 |

Stable-GPU + growing-CPU-overhead is compatible with two hypotheses, neither pinned down: (a) GPU command-encoding / driver-side bookkeeping scaling with extreme weight magnitudes even when per-op compute is constant, or (b) accumulating heartbeat state (rolling windows, segment histories, chart buckets) growing without trimming. Worth a profile pass once the divergence itself is fixed; if (a), bounded logits eliminate it for free, and if (b), it's a separate cleanup but exposed by the divergence.

---

## Context: today's commits

This session ran on the head of these recent commits (newest first):

- `ba2ad3f` misc (xcscheme: ASan/TSan)
- `3f7e6da` Use single source of truth for arena-countdown warmup gate
- **`19d8ab6` Fix post-promotion collapse: masked entropy + illegal-mass penalty, drop CE clamp**
- `407ab25` Fix promotion checkpoint state coherence
- `30c684c` ChessTrainer comment refresh (RMS advantage)
- `c507fa5` Delay automatic arena triggers until model reaches stability
- **`4ef1a26` Stabilize trainer promotion: sync step-counter, add policy loss clip, switch to RMS advantage normalization**
- **`901898f` Anti-collapse: drop advantage mean-centering, entropy bonus over unmasked logits** (subsequently re-masked in 19d8ab6)

The two load-bearing commits for this issue are `19d8ab6` (dropped per-position CE clamp) and `901898f` / `4ef1a26` (sign-preserving RMS advantage normalization, no mean-centering).

## What we observed in this session

### Arena history (all in this run)

| # | step | score | result | draw % | trainer→champion |
|---|------|-------|--------|--------|------------------|
| 1 | 718 | 0.470 | kept | 85.3% | gycg-1 → gycg |
| 2 | 1316 | 0.468 | kept | 78.3% | |
| 3 | 1904 | 0.495 | kept | 82.3% | |
| 4 | 2492 | **0.512** | **PROMOTED** | 86.3% | → gycg-1 |
| 5 | 3054 | 0.468 | kept | 91.0% | |
| 6 | 3630 | 0.483 | kept | **95.5%** | |

Promotion threshold was lowered to 0.51 (default 0.55). #4 cleared by 0.0017.

### Headline metrics over the run

| step | t-since-promote | pLoss | pLossWin / pLossLoss | gNorm | pLogitAbsMax | pwNorm Δ | ‖v‖ | legalMass | top1Legal | pEnt |
|------|-----------------|-------|----------------------|-------|--------------|----------|-----|-----------|-----------|------|
| 30 | — | +0.55 | +7.6 / -4.8 | 5.7 | 7.27 | 0.000 | small | 0.002 | 0.00 | 1.97 |
| 500 (warmup end) | — | +0.10 | +10 / -9 | 4.4 | ~7 | 0.000 | small | 0.002 | 0.00 | 1.97 |
| 1918 | — | -0.32 | +51 / -53 | 6.9 | small | 0.04 | small | 0.04 | 0.05 | 1.87 |
| 2454 | pre-arena #4 | -9.5 | +156 / -200 | 43.6 | 32.8 | 0.078 | rising | 0.14 | 0.20 | 1.38 |
| 2493 | last step before promote | -9.4 | +157 / -200 | 44.6 | 33.8 | 0.092 | ~75 | 0.29 | 0.31 | 1.32 |
| 2505 | +13 steps post-promote | -1.6 | +107 / -120 | 34.3 | 35.6 | 0.207 | — | 0.29 | 0.30 | 0.91 |
| 2581 | +89 | -2.4 | +120 / -125 | 42.4 | 39.3 | 0.210 | — | 0.23 | 0.23 | 1.06 |
| 2657 | +165 | -14 | +137 / -181 | 115 | 55.9 | 0.219 | — | 0.13 | 0.12 | 1.16 |
| 2962 | +470 | -22 | +141 / -238 | 173 | 89.4 | 0.280 | — | 0.14 | 0.14 | 1.45 |
| 3301 | +809 | -63 | +168 / -449 | 265 | — | — | — | 0.08 | 0.08 | 1.54 |
| 3752 | +1260 | -146 | +300 / -1008 | 337 | — | — | — | 0.13 | 0.13 | 1.42 |
| 3864 (latest, ~1h44m) | +1372 | **-179** | +281 / **-1158** | **344** | **325.87** | **+1.034** | — | 0.10 | 0.10 | 1.41 |

`pwNorm Δ` is cumulative drift from session start. `‖v‖` (optimizer velocity L2) values come from the dashboard chart, not the [STATS] line.

### Dashboard observations at promote (from screenshot at t=1:06:54)

- gNorm hockey-stick inflection is **left** of the promote playhead.
- ‖v‖ already at ~75 at promote, climbing from baseline 0.
- pLossWin/pLossLoss split already widening pre-promote (+157 vs -200).
- "Above-uniform policy count": **legal 12 / illegal 615** — the cells with above-uniform probability are mostly illegal.
- Legal-mass-sum chart spikes briefly to ~0.75 around promote, then falls back to ~0.25.
- Policy entropy briefly drops to ~1.0 around promote.

## Hypothesis (stated, contested by user)

**Claim:** The runaway is driven by an internal trainer instability that started ~step 1900, well before arena #4. Promotion is a passenger, not the driver.

**Mechanism (positive-feedback loop made possible by 19d8ab6):**

1. Policy concentrates as training proceeds (top1Legal: 0 → 0.31 by step 2454). Normal.
2. Concentration makes `−log p(played)` heavy-tailed for surprising played moves.
3. Sign-preserving RMS advantage multiplies each tail term by `adv ∈ [-2, +2]`. The 5%-tail of strongly-negative-advantage positions (`p05 ≈ -0.97` at latest) drives `pLossLoss` deeply negative.
4. **The per-position CE clamp that previously bounded this was dropped in 19d8ab6.** Justification was "gNorm clip handles it." That fails when raw gNorm hits 11× the clip — clip normalizes magnitude but **preserves direction**, so every clipped step pushes weights the same direction.
5. Momentum integrates the consistent direction. ‖v‖ climbs from ~0 to 150+. Applied update is `LR · v`, not `LR · grad_clipped`. Even with the gradient clip pegged, velocity keeps growing.
6. Bigger updates → bigger logits → more peaked policy → heavier tails. Loop closes.

**What promotion adds (secondary contributors, not initiators):**

- Self-play workers swap to new champion. New games enter the buffer with a slightly different distribution, increasing the surprise-tail mass for the trainer. `playedMoveProbNegAdv` rises from 0.16 → 0.30 over the post-promote window.
- The 13 post-arena steps land with the pre-arena momentum direction still loaded — first observable post-arena pwNorm is +0.115 above its pre-arena value (8.8e-3/step rate vs the 4e-5/step rate before).

The post-promote rate is ~2-3× the pre-promote rate, but the underlying divergence is internal and was already running.

## The user's experimental challenge

> If your logic is correct, then we should see the same explosion by the 2 hour point if we don't do any promotions at all. Is that right?

**Answer: Yes — but with a caveat on rate.**

If the hypothesis is correct, the explosion **must** also occur without arenas/promotions, because the mechanism is purely internal to the trainer. The expected timeline:

- gNorm grew **6.9 → 44.6 over steps 1918→2493 (575 steps)** = ~6× growth in that window, all pre-promotion.
- Continuing the pre-promotion exponential rate (one doubling per ~225 steps), by step ~3864 (the 1h44m mark in this session) gNorm should be in the **100-200 range** even with no promotions.
- That is somewhat *less* than the 344 we see with the promotion-amplified version, but still well past the clip=30 threshold and well into divergence.

**Strong predictions for a no-arena 2-hour run** (start fresh, build network, Play-and-Train, do not press Run Arena, leave auto-arena off or set to a very long interval):

- By ~step 2500 (~30-40 min in): gNorm crosses 30 (the clip threshold).
- By ~step 3000: pLogitAbsMax > 50, ‖v‖ > 30.
- By ~step 4000 (≈ 2h): gNorm in 100-200 range, pLogitAbsMax > 100, pLoss strongly negative, pLossWin/pLossLoss split >5×.
- pwNorm drift past 0.3 cumulative.

**Falsification:** If after 2 hours of no-arena training:

- gNorm stays bounded under ~30,
- pLogitAbsMax stays under ~50,
- pLoss stays in the ±5 range,

…then the hypothesis is wrong and promotion really is the driver. In that case, suspect the candidate→trainer weight copy mechanism, the LR-warmup-counter resync from `4ef1a26`, or self-play/buffer distribution shift at champion swap.

**Weak signal that would *partially* confirm but not fully:** If divergence appears but at a much slower rate (e.g., gNorm only reaches ~50 by 2 hours), the internal mechanism is real but promotion is a major amplifier — both fixes needed.

## Metrics to watch in the no-arena run

In priority order:

1. **`pLogitAbsMax`** — the cleanest signal. Goes from ~7 at init to >300 in this session. Should not exceed ~10-15 in a healthy run.
2. **`gNorm` vs `gradClipMaxNorm` (=30)** — if raw gNorm sits above clip for many steps, clip is masking divergence not preventing it.
3. **‖v‖ (optimizer velocity L2 norm)** — currently visible on the dashboard. Healthy is small/stable; runaway has it climbing past 30, then 100+.
4. **`pwNorm` Δ-cumulative** — pre-promote drift was ~3.6e-5/step, post-promote ~8.8e-4/step. If even no-arena drift exceeds ~1e-4/step, that's the internal mechanism showing.
5. **`pLossWin` / `pLossLoss` split** — symmetric (1:1) is healthy, asymmetric (3:1+) is the heavy-tail loss imbalance.
6. **`adv mean`** — drifted from 0 to -0.044 in this run. Sign-preserving RMS does nothing to pull this back; it's a direct readout of value-head bias.

## Candidate fixes (only if hypothesis confirmed)

These are the two interventions worth trying, in order of bluntness:

1. **Cap `pLogitAbsMax` directly** — soft penalty `λ · max(0, max|logit| − τ)²` on the policy-head pre-softmax output, with τ ~ 10-15. Cheap, doesn't fight any other loss term. Doesn't zero recovery gradients (unlike the dropped clamp).
2. **Per-position CE-gradient clamp (not value clamp)** — bound the gradient of CE w.r.t. logits on a per-position basis. The 19d8ab6 commit dropped a value-clamp because it zeroed gradients for surprising-played-move positions. A gradient clamp targets the unbounded-magnitude failure mode without zeroing anything.

Either way, **alarm system needs**:
- `pLogitAbsMax > 50` trigger (would have fired ~step 2700, ~1100 steps before runaway was visible in pLoss).
- `‖v‖ > 30` trigger (similar timing).
- The existing `policyEntropyAlarmThreshold = 5.0` is unreachable now that entropy is masked (ceiling ~3.5 over ~35 legal moves). Either retune to ~1.0 or remove.

Also: the `[ALARM] legal-mass probe ok ...` lines (92 in this session) say "ok" — they're not alarms. Retag as `[PROBE]` or only emit on actual threshold crossings.

## Other things noted

- **Arena draw rate climbing 85% → 95.5%** across the 6 arenas. Symmetric "play it safe" without decisive improvement.
- **Promotion threshold 0.51** (lowered from default 0.55) — arena #4 cleared by 0.0017. With the threshold at 0.55 it would not have promoted, and we'd have a natural no-promotion comparison run already.
- **Concurrency healthy:** GPU utilization 93-97% in arenas, replay ratio holding near target 1.0, no batched-eval drift.
- **Memory growth ~2GB across the run** but stable, not leaking aggressively.

## Files / line refs to check if hypothesis confirmed

- `DrewsChessMachine/DrewsChessMachine/Training/ChessTrainer.swift` — the loss graph, advantage normalization, dropped CE clamp.
- `gradClipMaxNorm` parameter (currently 30) — the clip that's saturating.
- `illegalMassWeight` parameter (currently 1.0) — the new penalty from 19d8ab6.

## Saved checkpoint from this session

`[CHECKPOINT] Saved session (post-promotion): 20260509-200246-20260509-11-znso-promote.dcmsession` (at the promote moment, step 2492).
