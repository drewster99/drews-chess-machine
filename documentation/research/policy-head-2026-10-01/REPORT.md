# Policy-head health survey (2026-10-01)

Read-only, weights-only survey of the policy head of every saved DCM checkpoint: the live leaky-FC1 run, the SE-style experiment (all arms and seeds), and every lineage whose longest segment passed 75,000 training steps. No source file, model file or git state was changed; nothing ran on the GPU. Checkpoints are identified by their `__metadata__` (`model_id`, `training_step`, `architecture`), never by filename. The leaky-FC1 run was live; its latest checkpoint at analysis time was `20261001-43-NbWz` @ 12,000.

## Policy head architectures and terms used here

**What the policy head does.** It turns the tower's output (C channels × 8 × 8
squares) into 4,864 move logits: 76 move-type channels × 64 from-squares. Logit
index = channel × 64 + row × 8 + col, in the side-to-move frame
(`PolicyEncoding.swift`). Illegal moves are masked on the CPU after the softmax.

**Three styles exist** (`ChessNetwork.policyHead`); every saved checkpoint uses
the first two:

| style | layers | used by (in this survey) |
|---|---|---|
| `simple_conv` | 1×1 conv C → 76 (+ bias) | ykkk; KbHZ and sMe9 (fp32 self-play) |
| `intermediate_conv` | **pre-block** (1×1 pre-conv C → K, no bias → BatchNorm → activation) → 1×1 final conv K → 76 (+ bias) | every other line |
| `fc_bottleneck` | pre-block → fully connected K·64 → 4,864 (+ bias) | none |

**The pre-block** is the first half of `intermediate_conv`: the pre-conv, its
BatchNorm and the activation (ReLU on every line surveyed). It turns the tower
output into K "policy feature" channels, which the final conv then combines
into move logits. Tensors: `policy.pre_conv.weight` [K, C, 1, 1],
`policy.pre_bn.{weight (γ), bias (β), running_mean, running_var}` [K], and the
final conv `policy.conv.weight` [76, K, 1, 1] + `policy.conv.bias` [76].

Example — the current SE-experiment / leaky-FC1 head (C = 128, K = 128):

| step | layer | shape per position | params |
|---|---|---|---:|
| 1 | tower output | 128 × 8 × 8 | |
| 2 | pre-conv 1×1, 128 → 128, no bias | 128 × 8 × 8 | 16,384 |
| 3 | BatchNorm (γ, β; running mean/var) | 128 × 8 × 8 | 256 |
| 4 | ReLU | 128 × 8 × 8 | |
| 5 | final conv 1×1, 128 → 76, + bias | 76 × 8 × 8 | 9,804 |
| 6 | flatten → logits | 4,864 | |
| | **total** | | **26,444** |

K per lineage: 128 on v5, mini2b, coxw, the SE / leaky runs and most self-play
lines; **512** on the qeu8 → Ejp0 family (C = 64, so the pre-conv widens 64 → 512)
and nt8y; 32 on LMGh.

**`k`** is a pre-block channel index, 0 … K−1 (e.g. "Ejp0 k=273" = channel 273 of
512). Channel k has pre-conv row `pre_conv.weight[k, :]`, BN values γ[k], β[k],
running_mean[k], running_var[k], and is read by final-conv column
`conv.weight[:, k]`. Indices carry no meaning beyond identity; lines forked from
one seed share indices (both Ejp0 self-play runs inherit the seed's channels).

**β/|γ|, "dead", "always on".** ReLU sets negative values to 0 and passes
positives unchanged. After BatchNorm, channel k's values are modelled as normal
with mean β and spread |γ|, so β/|γ| says how many spreads its typical value
sits above 0 — and therefore how often ReLU zeroes it:

| β/\|γ\| | share of values ReLU zeroes | label used here |
|---:|---:|---|
| −3 | 99.87% | dead (≈ never on) |
| −2 | 97.7% | mostly off |
| 0 | 50% | |
| +0.6 … +0.9 | 27% … 18% | typical channel on these lines |
| +2 | 2.3% | |
| +3 | 0.13% | always on (≈ never zeroed) |
| +10 | ~0 | (Ejp0 k=273) |

A healthy channel switches on and off across positions; that switching is the
nonlinearity that makes it a feature. An **always-on** channel is never zeroed,
so ReLU does nothing to it: it becomes a linear pass-through riding on a large
constant (its mean). The final conv multiplies that constant by the channel's
column and adds the result to every logit — a hidden second bias. Softmax
ignores anything added to every logit, so the loss never pulls it back; that is
how these channels came to carry ~40% of the shared policy offset on the long
replay lines.

**Other columns:** *running var / median* — how much more variable the
channel's pre-conv output is than the median channel (BN normalizes it away,
but it shows the pre-conv row grew); *final-conv column norm* — ‖conv.weight[:, k]‖,
how hard the final conv reads channel k; *added to every logit* — that
channel's constant contribution to the shared logit level (column mean × the
channel's mean activation).

### Always-on channels, every one (lineage-latest)

**v5** (0pTW, cum step 859,769; 11 of 128; median final-conv column norm 4.03;
β/|γ| median over all channels +0.94)

| k | β/\|γ\| | P(on) | running var / median | final-conv column norm | added to every logit |
|---:|---:|---:|---:|---:|---:|
| 34 | +9.60 | 1.0000 | 1,343 | 17.40 | −17.20 |
| 119 | +9.16 | 1.0000 | 598 | 14.30 | −11.70 |
| 62 | +6.53 | 1.0000 | 789 | 14.61 | −11.87 |
| 111 | +5.49 | 1.0000 | 1,373 | 16.67 | −15.16 |
| 126 | +4.49 | 1.0000 | 1.5 | 6.50 | −3.01 |
| 97 | +4.18 | 1.0000 | 1,248 | 15.95 | −13.53 |
| 42 | +3.98 | 1.0000 | 664 | 14.15 | −10.48 |
| 28 | +3.65 | 0.9999 | 139 | 10.84 | −9.33 |
| 105 | +3.52 | 0.9998 | 1,123 | 15.33 | −12.13 |
| 65 | +3.35 | 0.9996 | 0.8 | 5.19 | −1.48 |
| 61 | +3.20 | 0.9993 | 481 | 13.49 | −9.21 |

**Ejp0 replay** (@ 1,397,000; 20 of 512; median column norm 0.85; β/|γ| median +0.64)

| k | β/\|γ\| | P(on) | running var / median | final-conv column norm | added to every logit |
|---:|---:|---:|---:|---:|---:|
| 273 | +10.12 | 1.0000 | 1,220 | 8.08 | −8.57 |
| 163 | +8.33 | 1.0000 | 813 | 7.11 | −6.52 |
| 230 | +7.91 | 1.0000 | 823 | 7.04 | −6.38 |
| 244 | +6.38 | 1.0000 | 269 | 6.32 | −4.98 |
| 52 | +5.97 | 1.0000 | 266 | 5.79 | −4.15 |
| 332 | +5.78 | 1.0000 | 238 | 5.46 | −3.67 |
| 415 | +5.69 | 1.0000 | 207 | 4.80 | −2.84 |
| 79 | +5.58 | 1.0000 | 231 | 5.59 | −3.82 |
| 155 | +5.44 | 1.0000 | 203 | 4.88 | −2.94 |
| 387 | +5.44 | 1.0000 | 219 | 5.01 | −3.09 |
| 347 | +4.89 | 1.0000 | 194 | 4.87 | −2.88 |
| 2 | +4.34 | 1.0000 | 183 | 4.57 | −2.50 |
| 350 | +4.19 | 1.0000 | 182 | 4.53 | −2.45 |
| 489 | +4.10 | 1.0000 | 215 | 4.91 | −2.87 |
| 157 | +3.97 | 1.0000 | 148 | 4.06 | −1.95 |
| 133 | +3.50 | 0.9998 | 186 | 4.35 | −2.20 |
| 359 | +3.49 | 0.9998 | 182 | 4.25 | −2.10 |
| 169 | +3.24 | 0.9994 | 159 | 3.84 | −1.71 |
| 150 | +3.18 | 0.9993 | 132 | 3.51 | −1.42 |
| 312 | +3.17 | 0.9992 | 126 | 3.58 | −1.51 |

**Ejp0 self-play run 2 champion** (@ 1,186,322; 16 of 512 — the same channel
indices as the replay line, inherited from the 1.3M seed, each sitting less far
positive; median column norm 0.83)

| k | β/\|γ\| | P(on) | running var / median | final-conv column norm | added to every logit |
|---:|---:|---:|---:|---:|---:|
| 273 | +7.79 | 1.0000 | 1,696 | 6.75 | −7.01 |
| 163 | +6.61 | 1.0000 | 1,307 | 5.94 | −5.34 |
| 230 | +6.25 | 1.0000 | 1,099 | 5.90 | −5.22 |
| 244 | +5.13 | 1.0000 | 477 | 5.26 | −4.01 |
| 52 | +4.82 | 1.0000 | 382 | 4.87 | −3.38 |
| 332 | +4.73 | 1.0000 | 341 | 4.58 | −2.99 |
| 415 | +4.64 | 1.0000 | 303 | 3.98 | −2.28 |
| 79 | +4.54 | 1.0000 | 372 | 4.69 | −3.11 |
| 155 | +4.50 | 1.0000 | 309 | 4.03 | −2.34 |
| 387 | +4.46 | 1.0000 | 325 | 4.16 | −2.48 |
| 347 | +4.05 | 1.0000 | 303 | 4.03 | −2.31 |
| 2 | +3.65 | 0.9999 | 284 | 3.78 | −2.01 |
| 350 | +3.54 | 0.9998 | 280 | 3.76 | −1.97 |
| 489 | +3.47 | 0.9997 | 330 | 4.07 | −2.29 |
| 157 | +3.38 | 0.9996 | 236 | 3.38 | −1.58 |
| 133 | +3.01 | 0.9987 | 262 | 3.62 | −1.76 |

## Headline

- **Nothing in any policy head is dead, stuck at zero gradient, or non-finite.**
  - No NaN/Inf in the 143 checkpoints analyzed in detail, nor in the 4,020 scanned for the trajectory.
  - No pre-block channel is dead (β/|γ| < −3) or mostly off (< −2) in any of the 132 checkpoints that have a pre-block. The lowest β/|γ| anywhere is −1.50 (LMGh, P(on) = 6.7%).
  - No flat channels (|γ| < 5% of median) and no negative γ.
  - No exactly-zero optimizer velocity on any γ, β, pre_conv row or final-conv row in any file that carries velocity.
  - The final conv reads every pre-block channel: no column is under 10% of the median.
- **The policy offset on the long bf16 replay lines is carried by "always-on" pre-block channels.**
  - These channels have β/|γ| > 3: the ReLU never clips them, so they act as a constant input.
  - There are 11 on v5 (0pTW), 20 on Ejp0 @ 1.397M, and 16 on the Ejp0 self-play run-2 champion.
  - Their final-conv columns point entirely along the shared mean row (column shared fraction 1.00). They are also the largest columns: 15–17 on v5 vs a median of 4.0, and 5.8–8.1 on Ejp0 vs 0.85.
  - They are the hottest BN channels: v5 running_var reaches ~6,000, which is 1,373× the median.
  - They carry 38% of v5's weight-borne shared logit level (−115 of −300) and 44% of Ejp0's (−69 of −157). The rest is spread thinly along the same mean row.
- **The shared offset grew only during corpus replay.** Replay reached a static shared logit level of −300 on v5 and −158 on Ejp0 @ 1.397M.
  - On the two Ejp0 self-play runs from the 1.3M replay seed (level −148), it did not grow:
    - Run 2: −148 → −139 over 1.18M steps, with the bias mean constant to 5 digits (−0.92352).
    - Run 1: shrank to −72 in 197k steps.
  - The Phase-2 fine-tune of Ejp0 @ 681k (`oeNy`) keeps its bias mean constant to 1e-5 (−0.5886) and shrinks the level from −55.4 to −43.8.
  - The fresh SE and leaky runs keep their bias mean within 1e-4 of zero.
- **The leaky-FC1 run's policy head is as healthy as its ReLU comparator and nearly the same.**
  - Matched-step summary statistics agree to the third digit. At 12k: static level +0.567 vs +0.611, final-row norm median 1.681 vs 1.686, bias correlation 0.9992.
  - The weight distance between the two runs is 0.23 of the training displacement for the final conv. The zero-β SE change from the same fresh weights moves it further, to 0.37.
- **Underpromotions and 7-square queen moves are the least-trained move types everywhere.** This is expected, not a defect.
  - Their final-conv rows get 50–250× less gradient than the median row (velocity norm ~1e-4 vs ~0.03).
  - They move 4–8× less from init in fresh runs.
  - On the offset-heavy lines their residual (non-shared) row norm is only 0.28–0.36 of the queen-style median.
- **bf16 exposure that remains under the current default tail (`mixed_final_projection`: BN and features in bf16):**
  - Shared-row feature-rounding noise is per-square logit noise, common to all move types and not cancelled by softmax. It is 0.113 nats on v5, 0.043 on Ejp0 and 0.036 on the run-2 champion; every other line is ≤ 0.009.
  - BN-input cancellation (|running_mean|/√running_var) reaches 17.7 on Ejp0 (channels 68, 144, 66), which adds ≈ 0.036σ of noise to those normalized values.

## What this means

- **The policy head has no dead units.** The dead-unit pathology of the SE FC1 bottlenecks does not appear in the policy pre-block on any line at any step.
  - The leaky-FC1 change does not touch the policy head: its pre-block activation is the tower-level `activation_function`, which is still ReLU.
  - The weights confirm the head is unchanged.
- **The opposite failure is real: channels that never turn off.**
  - On the offset-heavy lines, a few pre-block channels drifted to large positive β and became linear pass-throughs.
  - The final conv uses their constant mean as a second bias. Unlike `policy.conv.bias`, it decays with weight decay, but softmax ignores it, so the loss never pulls it back. That makes it an easy place for the offset to accumulate.
  - This also explains the extreme running-var spread: their pre_conv rows grew while the rest shrank.
  - Worth watching on long runs: the always-on count and running-var max/median. Both stayed at 0 and < 5 for the first ~1M Ejp0 replay steps, then rose together.
- **Phase 2 behaves as designed on the policy head.**
  - The bias mean has zero gradient: frozen on `oeNy`, ~0 on the new runs.
  - The weight-borne level stops growing and decays slowly.
  - An existing offset does not go away quickly: `oeNy` is still at −43.8 after 20k steps. Weights inherited from the old replay lines carry their offset, and their always-on channels, for a long time.
- **The fp32 head tail is still needed for the old lineages.** Under `mixed_final_projection`, v5- and Ejp0-era weights get 0.04–0.11 nats of per-square noise from the shared row. That is the plan's "what it gives back" cost, now measured per checkpoint. It is negligible (< 0.01) on every line trained without a large offset.
- **The rarest move types learn slowly by construction.** No action is needed, but they are where a shared-row error matters most, because their own signal is smallest.
- **KbHZ's final conv looks like a random init** (fp32, simple_conv, 532k self-play steps).
  - std 0.1256 vs an init std of 0.1250, biases within ±0.07, uniform families, and 2% row change over its last 37k steps.
  - Its tower does all the work.
  - This is unconfirmed: no KbHZ fresh net survives.
- **The static (weights-only) shared-level estimate is reliable, so it can be tracked from checkpoints alone.**
  - On the 24 surveyed checkpoints whose measured all-logit mean exceeds 5 nats in magnitude, it matches the bf16-head-offset survey's forward pass within 2.4%.
  - LMGh is the exception, and there the survey forward is the broken one: value logits of +43,926 and infinite CE.

## Method

- **Inventory.** Every `.safetensors` and `.dcmmodel` under `~/Library/Application Support/DrewsChessMachine/` (Models/, Sessions/*.dcmsession/, KeptSelfPlayModels/) was read by header.
  - Result: 4,020 unique checkpoints (4,009 safetensors + 11 dcmmodel), after merging 71 byte-identical copies.
  - Copies were merged by (`model_id`, `training_step`, `content_sha256`). `content_sha256` alone is not enough: it hashes tensor content only, so a derived fresh net with unchanged weights shares it (2q0Q and Dmwe vs JZOe).
- **Lineages** are chains of `parent_model_id` links.
  - Cumulative-step bases come from `documentation/dashboards/registry.json` for v5, mini2b, coxw, ykkk, nt8y, qeu8 and qeu8-1blk128. For the Qeu8e epoch branch they are the sums of the earlier segments' final saved steps.
  - The two Ejp0 self-play runs are split at 2026-08-08 17:40 local, when run 2 started. They are separate forks of the 1.3M seed.
  - `oeNy`'s start weights (Ejp0 @ 681,000) come from the `[REPLAY] start-model` line in `dcm_log_20261001-015706.txt`.
  - A lineage is analyzed in detail if any segment's max `training_step` exceeds 75,000, or if it belongs to the SE/leaky experiment.
  - Two supplements: `oeNy`, the only post-fix descendant of an offset-heavy line, and the never-trained full-leaky fresh net `Dmwe`.
- **Detailed set per lineage:**
  - fresh or seed, first trained, median, every segment end, and latest;
  - for self-play, both champion and trainer of the latest session;
  - for SE/leaky, steps 1k, 5k, 10k, 11k, 12k (the leaky latest), 20k and final, plus every leaky checkpoint.
- **Trajectory:** compact metrics for every unique checkpoint, in `results/trajectory.csv`.
- **Channel layout**, confirmed in `PolicyEncoding.swift`:
  - 0–55: queen-style moves, channel `direction*7 + distance-1`. Directions are N, NE, E, SE, S, SW, W, NW in the side-to-move frame.
  - 56–63: knight moves.
  - 64–72: underpromotions, channel `64 + piece*3 + dir`. Pieces are **knight=0, rook=1, bishop=2**; directions are forward, capture-left, capture-right.
  - 73–75: queen promotions.
- **No checkpoint uses `fc_bottleneck`.** Only `intermediate_conv` and `simple_conv` exist. Both 1×1 convs are shared across all 64 squares, so there is no per-square weight structure: any per-square pattern in the logits comes from the tower.
- **Storage.**
  - All stored tensors are F32.
  - Files without optimizer velocity on bf16 lines are 100% bf16-exact.
  - Files with velocity (newer replay saves, session trainers) hold fp32 masters, bf16-exact fraction 0. The exception is the run-2 trainer, which was just reset to the promoted champion.
  - Max |w| in any policy tensor is 5.33 (v5 pre_conv), so overflow headroom is not a concern.
- **Definitions** (weights only):
  - **Post-BN value** of pre-block channel k is modeled as N(β_k, γ_k²). β/|γ| < −3 means dead, < −2 mostly off, > 3 always on.
  - **Post-activation moments** are computed by Gauss–Hermite quadrature.
  - **Static logit level** L_c = b_c + Σ_k W[c,k]·E[a_k]. The mean over c is the static shared level. It is exact under the marginal assumption alone.
  - **Mean row** m = mean_c W[c,:]. **Mean-row ratio** = ‖m‖ / mean ‖W[c]‖. **Residual** = W[c] − m.
  - **Shared-row rounding noise** = 2⁻⁸/√3 · √(Σ_k m_k² E[a_k²]).
  - **BN cancellation** = |running_mean| / √running_var; the normalized-value noise is ≈ (that + 1)·2⁻⁹ σ.
  - **Column shared fraction** = |m_k|·√76 / ‖W[:,k]‖.
- **Velocity.** It is the clipped gradient with momentum. Weight decay is decoupled (`ChessTrainer` ~l.4100), so a channel with no loss gradient has exactly-zero velocity. Layouts were verified against shapes. The pre_conv row velocity is orthogonal to its row (cos −0.04…+0.001), as BN scale invariance predicts.
- **Limits.**
  - The static estimates assume BN running stats match batch stats and that the marginals are Gaussian.
  - The independence-based "logit std" columns are rough.
  - Dead/always-on status is predicted from BN parameters, not measured as a firing rate.

## Findings, ranked

Severity levels:
- **CRITICAL:** numerically broken.
- **HIGH:** a known failure mechanism is active.
- **MEDIUM:** a partial or emerging problem.
- **LOW:** worth knowing.

Per-channel hits are collapsed to one row per checkpoint and rule; every channel is listed in `results/findings.csv`. There are no CRITICAL findings.

<!-- begin:findings_ranked -->

| severity | lineage | checkpoint | rule | tensor | index | value | why | present at lineage-latest | other detailed checkpoints with the same hit |
|---|---|---|---|---|---|---|---|---|---|
| HIGH | Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy@20000 | large shared policy logit level | `policy.conv` |  | -43.82 | static mean logit -43.8: bf16 spacing there is 0.25; any path that rounds logits to bf16 merges near-equal moves | yes | 2 |
| HIGH | Ejp0 self-play run 1 | 20260727-1-Ejp0-10@197340 [trainer] | large shared policy logit level | `policy.conv` |  | -69.64 | static mean logit -69.6: bf16 spacing there is 0.5; any path that rounds logits to bf16 merges near-equal moves | yes | 3 |
| HIGH | Ejp0 self-play run 1 | 20260727-1-Ejp0-10@197340 [trainer] | shared offset carried by always-on channels | `policy.pre_bn + policy.conv.weight` | 20 channels | -27.43 | -27.4 of the -68.7 weight-borne shared logit level comes from channels the activation never clips; their final-conv columns point along the mean row | yes | 2 |
| HIGH | Ejp0 self-play run 2 | 20260727-1-Ejp0-68@1186322 [trainer] | large shared policy logit level | `policy.conv` |  | -138.9 | static mean logit -138.9: bf16 spacing there is 1; any path that rounds logits to bf16 merges near-equal moves | yes | 3 |
| HIGH | Ejp0 self-play run 2 | 20260727-1-Ejp0-68@1186322 [trainer] | shared offset carried by always-on channels | `policy.pre_bn + policy.conv.weight` | 16 channels | -50.09 | -50.1 of the -138.0 weight-borne shared logit level comes from channels the activation never clips; their final-conv columns point along the mean row | yes | 2 |
| HIGH | qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0@1397000 (cum 1572915) | large shared policy logit level | `policy.conv` |  | -157.9 | static mean logit -157.9: bf16 spacing there is 1; any path that rounds logits to bf16 merges near-equal moves | yes | 1 |
| HIGH | qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0@1397000 (cum 1572915) | shared offset carried by always-on channels | `policy.pre_bn + policy.conv.weight` | 20 channels | -68.56 | -68.6 of the -157.0 weight-borne shared logit level comes from channels the activation never clips; their final-conv columns point along the mean row | yes | 0 |
| HIGH | v5 | 20260805-1-0pTW@2000 (cum 859769) | large shared policy logit level | `policy.conv` |  | -300.4 | static mean logit -300.4: bf16 spacing there is 2; any path that rounds logits to bf16 merges near-equal moves | yes | 4 |
| HIGH | v5 | 20260805-1-0pTW@2000 (cum 859769) | shared offset carried by always-on channels | `policy.pre_bn + policy.conv.weight` | 11 channels | -115.1 | -115.1 of the -299.6 weight-borne shared logit level comes from channels the activation never clips; their final-conv columns point along the mean row | yes | 1 |
| HIGH | v5 | 20260805-1-0pTW@2000 (cum 859769) | shared-row feature-rounding noise | `policy.conv.weight` | mean row | 0.1129 | rounding the K policy features to bf16 adds ~0.113 nats of per-square noise common to all move types at that square (does not cancel in softmax) | yes | 4 |
| HIGH | Ejp0 self-play run 1 | 20260727-1-Ejp0-1@2578 [trainer] | shared offset carried by always-on channels | `policy.pre_bn + policy.conv.weight` | 17 channels | -56.12 | -56.1 of the -146.9 weight-borne shared logit level comes from channels the activation never clips; their final-conv columns point along the mean row | no | 0 |
| HIGH | Ejp0 self-play run 2 | 20260727-1-Ejp0-1@20241 | shared offset carried by always-on channels | `policy.pre_bn + policy.conv.weight` | 15 channels | -52.27 | -52.3 of the -146.8 weight-borne shared logit level comes from channels the activation never clips; their final-conv columns point along the mean row | no | 0 |
| HIGH | v5 | 20260729-1-VZ2j@106333 (cum 811769) | shared offset carried by always-on channels | `policy.pre_bn + policy.conv.weight` | 10 channels | -96.68 | -96.7 of the -284.3 weight-borne shared logit level comes from channels the activation never clips; their final-conv columns point along the mean row | no | 0 |
| MEDIUM | Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy@20000 | BN input mean >> std | `policy.pre_bn.running_mean` | 2 ch: k=475, k=317 | 9.341 | most extreme: k=475: |running_mean|/sqrt(running_var)=9.34; bf16 input rounding noise ~0.020 std | yes | 1 |
| MEDIUM | Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy@20000 | hot BN channel | `policy.pre_bn.running_var` | 28 ch: k=414, k=375, k=311, k=14, k=176, k=465, k=482, k=100, k=459, k=140, k=287, k=39 (+16 more) | 4.35 | most extreme: k=414: running_var 4.35 = 50.3x the channel median (beta/|gamma|=+0.33, pre_conv row norm 1.65) | yes | 0 |
| MEDIUM | Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy@20000 | large shared row | `policy.conv.weight` | mean row | 0.7435 | ||mean row|| / mean ||row|| = 0.74 | yes | 2 |
| MEDIUM | Ejp0 self-play run 1 | 20260727-1-Ejp0-9@197340 [champion] | BN input mean >> std | `policy.pre_bn.running_mean` | 5 ch: k=68, k=144, k=108, k=409, k=232 | 14.66 | most extreme: k=68: |running_mean|/sqrt(running_var)=14.7; bf16 input rounding noise ~0.031 std | yes | 0 |
| MEDIUM | Ejp0 self-play run 1 | 20260727-1-Ejp0-10@197340 [trainer] | BN input mean >> std | `policy.pre_bn.running_mean` | 6 ch: k=68, k=144, k=108, k=409, k=232, k=508 | 15.17 | most extreme: k=68: |running_mean|/sqrt(running_var)=15.2; bf16 input rounding noise ~0.032 std | yes | 0 |
| MEDIUM | Ejp0 self-play run 1 | 20260727-1-Ejp0-10@197340 [trainer] | BN running-var spread | `policy.pre_bn.running_var` | k=273 | 521.6 | max/median running_var = 522 (span max/min 5.22e+03); a few pre-conv outputs are orders of magnitude larger than the rest | yes | 3 |
| MEDIUM | Ejp0 self-play run 1 | 20260727-1-Ejp0-10@197340 [trainer] | always-on pre-block channel | `policy.pre_bn` | 13 ch: k=273, k=163, k=230, k=244, k=52, k=332, k=415, k=79, k=155, k=387, k=347, k=2 (+1 more) | 10.66 | most extreme: k=273: beta/|gamma|=+10.7: the ReLU never clips it, so the channel is linear and its mean E[a]=8.7 acts as a constant input; it adds -3.46 to every logit through the final conv's mean row (column shared fraction 1.00) | yes | 0 |
| MEDIUM | Ejp0 self-play run 1 | 20260727-1-Ejp0-9@197340 [champion] | always-on pre-block channel | `policy.pre_bn` | 14 ch: k=273, k=163, k=230, k=244, k=52, k=332, k=415, k=79, k=155, k=387, k=347, k=2 (+2 more) | 10.59 | most extreme: k=273: beta/|gamma|=+10.6: the ReLU never clips it, so the channel is linear and its mean E[a]=8.69 acts as a constant input; it adds -3.59 to every logit through the final conv's mean row (column shared fraction 1.00) | yes | 1 |
| MEDIUM | Ejp0 self-play run 1 | 20260727-1-Ejp0-10@197340 [trainer] | hot BN channel | `policy.pre_bn.running_var` | 106 ch: k=273, k=163, k=230, k=244, k=52, k=332, k=79, k=387, k=489, k=119, k=155, k=415 (+94 more) | 4.482 | most extreme: k=273: running_var 4.482 = 521.2x the channel median (beta/|gamma|=+10.66, pre_conv row norm 0.663) | yes | 0 |
| MEDIUM | Ejp0 self-play run 1 | 20260727-1-Ejp0-9@197340 [champion] | hot BN channel | `policy.pre_bn.running_var` | 107 ch: k=273, k=163, k=230, k=244, k=52, k=332, k=79, k=387, k=489, k=119, k=155, k=415 (+95 more) | 5 | most extreme: k=273: running_var 5 = 502.6x the channel median (beta/|gamma|=+10.59, pre_conv row norm 0.692) | yes | 0 |
| MEDIUM | Ejp0 self-play run 1 | 20260727-1-Ejp0-10@197340 [trainer] | large shared row | `policy.conv.weight` | mean row | 0.9056 | ||mean row|| / mean ||row|| = 0.91 | yes | 3 |
| MEDIUM | Ejp0 self-play run 1 | 20260727-1-Ejp0-10@197340 [trainer] | shared-row feature-rounding noise | `policy.conv.weight` | mean row | 0.01783 | ~0.018 nats per-square noise from bf16 feature rounding | yes | 3 |
| MEDIUM | Ejp0 self-play run 2 | 20260727-1-Ejp0-68@1186322 [trainer] | BN input mean >> std | `policy.pre_bn.running_mean` | 7 ch: k=144, k=68, k=108, k=409, k=66, k=311, k=139 | 15.91 | most extreme: k=144: |running_mean|/sqrt(running_var)=15.9; bf16 input rounding noise ~0.033 std | yes | 1 |
| MEDIUM | Ejp0 self-play run 2 | 20260727-1-Ejp0-68@1186322 [trainer] | BN running-var spread | `policy.pre_bn.running_var` | k=273 | 1696 | max/median running_var = 1696 (span max/min 2.2e+04); a few pre-conv outputs are orders of magnitude larger than the rest | yes | 3 |
| MEDIUM | Ejp0 self-play run 2 | 20260727-1-Ejp0-68@1186322 [trainer] | always-on pre-block channel | `policy.pre_bn` | 16 ch: k=273, k=163, k=230, k=244, k=52, k=332, k=415, k=79, k=155, k=387, k=347, k=2 (+4 more) | 7.785 | most extreme: k=273: beta/|gamma|=+7.79: the ReLU never clips it, so the channel is linear and its mean E[a]=9.06 acts as a constant input; it adds -7.01 to every logit through the final conv's mean row (column shared fraction 1.00) | yes | 2 |
| MEDIUM | Ejp0 self-play run 2 | 20260727-1-Ejp0-68@1186322 [trainer] | hot BN channel | `policy.pre_bn.running_var` | 118 ch: k=273, k=163, k=230, k=244, k=52, k=79, k=332, k=489, k=387, k=155, k=347, k=415 (+106 more) | 33.75 | most extreme: k=273: running_var 33.75 = 1696.2x the channel median (beta/|gamma|=+7.79, pre_conv row norm 1.43) | yes | 1 |
| MEDIUM | Ejp0 self-play run 2 | 20260727-1-Ejp0-68@1186322 [trainer] | large shared row | `policy.conv.weight` | mean row | 0.9354 | ||mean row|| / mean ||row|| = 0.94 | yes | 3 |
| MEDIUM | Ejp0 self-play run 2 | 20260727-1-Ejp0-68@1186322 [trainer] | shared-row feature-rounding noise | `policy.conv.weight` | mean row | 0.03619 | ~0.036 nats per-square noise from bf16 feature rounding | yes | 3 |
| MEDIUM | KbHZ self-play (fp32) | 20260514-1-KbHZ-23@532369 [trainer] | final-conv row ~unchanged from reference | `policy.conv.weight` | 38 ch: 15 Q-E2, 62 Kn-6(left-up), 26 Q-SE6, 23 Q-SE3, 44 Q-W3, 19 Q-E6, 2 Q-N3, 58 Kn-2(right-down), 14 Q-E1, 63 Kn-7(up-left), 43 Q-W2, 6 Q-N7 (+26 more) | 0.01989 | most extreme: 15 Q-E2: relative change 0.0199 vs reference (20260514-1-KbHZ-18@494927); that move type is barely trained | yes | 0 |
| MEDIUM | KbHZ self-play (fp32) | 20260514-1-KbHZ-22@532369 [champion] | final-conv row ~unchanged from reference | `policy.conv.weight` | 42 ch: 35 Q-SW1, 29 Q-S2, 50 Q-NW2, 37 Q-SW3, 62 Kn-6(left-up), 15 Q-E2, 26 Q-SE6, 44 Q-W3, 23 Q-SE3, 28 Q-S1, 19 Q-E6, 2 Q-N3 (+30 more) | 0.01954 | most extreme: 35 Q-SW1: relative change 0.0195 vs reference (20260514-1-KbHZ-18@494927); that move type is barely trained | yes | 0 |
| MEDIUM | bzw3 self-play | 20260601-11-bzw3-32@467099 [trainer] | elevated shared policy logit level | `policy.conv` |  | -14.04 | static mean logit -14.0: bf16 spacing 0.0625 | yes | 0 |
| MEDIUM | bzw3 self-play | 20260601-11-bzw3-32@467099 [trainer] | large shared row | `policy.conv.weight` | mean row | 0.5597 | ||mean row|| / mean ||row|| = 0.56 | yes | 0 |
| MEDIUM | nt8y | 20260708-4-kEiZ@21086 (cum 312748) | BN input mean >> std | `policy.pre_bn.running_mean` | 6 ch: k=255, k=313, k=329, k=391, k=368, k=241 | 10.58 | most extreme: k=255: |running_mean|/sqrt(running_var)=10.6; bf16 input rounding noise ~0.023 std | yes | 0 |
| MEDIUM | nt8y | 20260708-4-kEiZ@21086 (cum 312748) | elevated shared policy logit level | `policy.conv` |  | -14.09 | static mean logit -14.1: bf16 spacing 0.0625 | yes | 1 |
| MEDIUM | nt8y | 20260708-4-kEiZ@21086 (cum 312748) | large shared row | `policy.conv.weight` | mean row | 0.5462 | ||mean row|| / mean ||row|| = 0.55 | yes | 1 |
| MEDIUM | qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0@1397000 (cum 1572915) | BN input mean >> std | `policy.pre_bn.running_mean` | 3 ch: k=68, k=144, k=66 | 17.74 | most extreme: k=68: |running_mean|/sqrt(running_var)=17.7; bf16 input rounding noise ~0.037 std | yes | 0 |
| MEDIUM | qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0@1397000 (cum 1572915) | BN running-var spread | `policy.pre_bn.running_var` | k=273 | 1220 | max/median running_var = 1220 (span max/min 2.27e+04); a few pre-conv outputs are orders of magnitude larger than the rest | yes | 0 |
| MEDIUM | qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0@1397000 (cum 1572915) | always-on pre-block channel | `policy.pre_bn` | 20 ch: k=273, k=163, k=230, k=244, k=52, k=332, k=415, k=79, k=155, k=387, k=347, k=2 (+8 more) | 10.12 | most extreme: k=273: beta/|gamma|=+10.1: the ReLU never clips it, so the channel is linear and its mean E[a]=9.25 acts as a constant input; it adds -8.57 to every logit through the final conv's mean row (column shared fraction 1.00) | yes | 0 |
| MEDIUM | qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0@1397000 (cum 1572915) | hot BN channel | `policy.pre_bn.running_var` | 111 ch: k=273, k=230, k=163, k=244, k=52, k=332, k=79, k=387, k=489, k=415, k=155, k=347 (+99 more) | 60 | most extreme: k=273: running_var 60 = 1216.6x the channel median (beta/|gamma|=+10.12, pre_conv row norm 1.58) | yes | 0 |
| MEDIUM | qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0@1397000 (cum 1572915) | large shared row | `policy.conv.weight` | mean row | 0.9481 | ||mean row|| / mean ||row|| = 0.95 | yes | 1 |
| MEDIUM | qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0@1397000 (cum 1572915) | shared-row feature-rounding noise | `policy.conv.weight` | mean row | 0.0434 | ~0.043 nats per-square noise from bf16 feature rounding | yes | 0 |
| MEDIUM | qeu8init sf100sl100 vs-UCI | 20260722-1-syxR@558000 | elevated shared policy logit level | `policy.conv` |  | -15.98 | static mean logit -16.0: bf16 spacing 0.0625 | yes | 1 |
| MEDIUM | qeu8init sf100sl100 vs-UCI | 20260722-1-syxR@558000 | hot BN channel | `policy.pre_bn.running_var` | 16 ch: k=81, k=203, k=334, k=505, k=174, k=314, k=301, k=63, k=252, k=305, k=404, k=384 (+4 more) | 0.3008 | most extreme: k=81: running_var 0.3008 = 25.8x the channel median (beta/|gamma|=+0.05, pre_conv row norm 0.587) | yes | 0 |
| MEDIUM | qeu8init sf100sl100 vs-UCI | 20260722-1-syxR@558000 | large shared row | `policy.conv.weight` | mean row | 0.664 | ||mean row|| / mean ||row|| = 0.66 | yes | 1 |
| MEDIUM | v5 | 20260805-1-0pTW@2000 (cum 859769) | BN running-var spread | `policy.pre_bn.running_var` | k=111 | 1373 | max/median running_var = 1373 (span max/min 2.45e+04); a few pre-conv outputs are orders of magnitude larger than the rest | yes | 3 |
| MEDIUM | v5 | 20260805-1-0pTW@2000 (cum 859769) | always-on pre-block channel | `policy.pre_bn` | 11 ch: k=34, k=119, k=62, k=111, k=126, k=97, k=42, k=28, k=105, k=65, k=61 | 9.6 | most extreme: k=34: beta/|gamma|=+9.6: the ReLU never clips it, so the channel is linear and its mean E[a]=8.62 acts as a constant input; it adds -17.2 to every logit through the final conv's mean row (column shared fraction 1.00) | yes | 1 |
| MEDIUM | v5 | 20260805-1-0pTW@2000 (cum 859769) | hot BN channel | `policy.pre_bn.running_var` | 35 ch: k=111, k=34, k=97, k=105, k=62, k=42, k=119, k=86, k=96, k=12, k=61, k=8 (+23 more) | 5984 | most extreme: k=111: running_var 5984 = 1358.1x the channel median (beta/|gamma|=+5.49, pre_conv row norm 5.33) | yes | 2 |
| MEDIUM | v5 | 20260805-1-0pTW@2000 (cum 859769) | large shared row | `policy.conv.weight` | mean row | 0.9682 | ||mean row|| / mean ||row|| = 0.97 | yes | 5 |
| MEDIUM | Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy@1000 | cold BN channel | `policy.pre_bn.running_var` | 1 ch: k=276 | 0.0054 | most extreme: k=276: running_var 0.0054 = 0.041x the channel median; BN divides by a tiny std, amplifying rounding noise of its input | no | 0 |
| MEDIUM | Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy@11000 | hot BN channel | `policy.pre_bn.running_var` | 24 ch: k=414, k=375, k=311, k=14, k=176, k=465, k=394, k=39, k=482, k=100, k=459, k=140 (+12 more) | 4.439 | most extreme: k=414: running_var 4.439 = 49.8x the channel median (beta/|gamma|=+0.33, pre_conv row norm 1.73) | no | 0 |
| MEDIUM | Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy@1000 | hot BN channel | `policy.pre_bn.running_var` | 25 ch: k=414, k=273, k=375, k=230, k=163, k=311, k=244, k=14, k=52, k=79, k=39, k=459 (+13 more) | 3.185 | most extreme: k=414: running_var 3.185 = 24.2x the channel median (beta/|gamma|=+0.34, pre_conv row norm 1.89) | no | 0 |
| MEDIUM | Ejp0 self-play run 1 | 20260727-1-Ejp0-4@104903 | BN input mean >> std | `policy.pre_bn.running_mean` | 3 ch: k=68, k=144, k=108 | 13.8 | most extreme: k=68: |running_mean|/sqrt(running_var)=13.8; bf16 input rounding noise ~0.029 std | no | 0 |
| MEDIUM | Ejp0 self-play run 1 | 20260727-1-Ejp0-1@2578 [trainer] | BN input mean >> std | `policy.pre_bn.running_mean` | 6 ch: k=144, k=68, k=311, k=478, k=108, k=66 | 15.17 | most extreme: k=144: |running_mean|/sqrt(running_var)=15.2; bf16 input rounding noise ~0.032 std | no | 0 |
| MEDIUM | Ejp0 self-play run 1 | 20260727-1-Ejp0-1@2578 [trainer] | always-on pre-block channel | `policy.pre_bn` | 17 ch: k=273, k=163, k=230, k=244, k=52, k=332, k=415, k=79, k=155, k=387, k=347, k=2 (+5 more) | 7.703 | most extreme: k=273: beta/|gamma|=+7.7: the ReLU never clips it, so the channel is linear and its mean E[a]=9.07 acts as a constant input; it adds -7.58 to every logit through the final conv's mean row (column shared fraction 1.00) | no | 0 |
| MEDIUM | Ejp0 self-play run 1 | 20260727-1-Ejp0-1@2578 [trainer] | hot BN channel | `policy.pre_bn.running_var` | 105 ch: k=273, k=163, k=230, k=244, k=52, k=79, k=332, k=489, k=387, k=347, k=155, k=415 (+93 more) | 41.21 | most extreme: k=273: running_var 41.21 = 846.1x the channel median (beta/|gamma|=+7.70, pre_conv row norm 1.47) | no | 0 |
| MEDIUM | Ejp0 self-play run 1 | 20260727-1-Ejp0-4@104903 | hot BN channel | `policy.pre_bn.running_var` | 106 ch: k=273, k=163, k=230, k=244, k=52, k=332, k=79, k=387, k=119, k=489, k=155, k=415 (+94 more) | 5.531 | most extreme: k=273: running_var 5.531 = 435.7x the channel median (beta/|gamma|=+10.54, pre_conv row norm 0.739) | no | 0 |
| MEDIUM | Ejp0 self-play run 2 | 20260727-1-Ejp0-1@20241 | BN input mean >> std | `policy.pre_bn.running_mean` | 6 ch: k=144, k=68, k=66, k=108, k=311, k=478 | 15.17 | most extreme: k=144: |running_mean|/sqrt(running_var)=15.2; bf16 input rounding noise ~0.032 std | no | 0 |
| MEDIUM | Ejp0 self-play run 2 | 20260727-1-Ejp0-39@935524 | BN input mean >> std | `policy.pre_bn.running_mean` | 7 ch: k=144, k=68, k=108, k=409, k=311, k=66, k=139 | 15.72 | most extreme: k=144: |running_mean|/sqrt(running_var)=15.7; bf16 input rounding noise ~0.033 std | no | 0 |
| MEDIUM | Ejp0 self-play run 2 | 20260727-1-Ejp0-1@20241 | always-on pre-block channel | `policy.pre_bn` | 15 ch: k=273, k=163, k=230, k=244, k=52, k=332, k=415, k=79, k=155, k=387, k=347, k=2 (+3 more) | 7.632 | most extreme: k=273: beta/|gamma|=+7.63: the ReLU never clips it, so the channel is linear and its mean E[a]=9.06 acts as a constant input; it adds -7.58 to every logit through the final conv's mean row (column shared fraction 1.00) | no | 0 |
| MEDIUM | Ejp0 self-play run 2 | 20260727-1-Ejp0-1@20241 | hot BN channel | `policy.pre_bn.running_var` | 107 ch: k=273, k=163, k=230, k=244, k=52, k=79, k=332, k=489, k=387, k=347, k=155, k=415 (+95 more) | 39 | most extreme: k=273: running_var 39 = 968.1x the channel median (beta/|gamma|=+7.63, pre_conv row norm 1.47) | no | 0 |
| MEDIUM | Ejp0 self-play run 2 | 20260727-1-Ejp0-39@935524 | hot BN channel | `policy.pre_bn.running_var` | 116 ch: k=273, k=163, k=230, k=244, k=52, k=79, k=332, k=489, k=387, k=155, k=347, k=415 (+104 more) | 32.75 | most extreme: k=273: running_var 32.75 = 1541.9x the channel median (beta/|gamma|=+7.79, pre_conv row norm 1.43) | no | 0 |
| MEDIUM | nt8y | 20260707-1-cslu@140000 (cum 291662) | BN input mean >> std | `policy.pre_bn.running_mean` | 5 ch: k=255, k=329, k=313, k=220, k=368 | 9.506 | most extreme: k=255: |running_mean|/sqrt(running_var)=9.51; bf16 input rounding noise ~0.021 std | no | 0 |
| MEDIUM | nt8y | 20260707-1-cslu@107000 | trajectory jump | `rv_max` | from 20260707-1-cslu@106000 | 1.319 | rv_max 0.5508 -> 0.7266 between consecutive saves | no | 0 |
| MEDIUM | qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0@611000 (cum 786915) | BN input mean >> std | `policy.pre_bn.running_mean` | 11 ch: k=491, k=167, k=399, k=206, k=29, k=404, k=103, k=442, k=195, k=436, k=311 | 14.13 | most extreme: k=491: |running_mean|/sqrt(running_var)=14.1; bf16 input rounding noise ~0.030 std | no | 0 |
| MEDIUM | qeu8 (replay main, ends Ejp0) | 20260706-1-PVZp@67000 (cum 175915) | elevated shared policy logit level | `policy.conv` |  | -9.705 | static mean logit -9.7: bf16 spacing 0.0625 | no | 0 |
| MEDIUM | qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0@611000 (cum 786915) | hot BN channel | `policy.pre_bn.running_var` | 40 ch: k=273, k=163, k=230, k=244, k=52, k=79, k=332, k=489, k=155, k=133, k=347, k=387 (+28 more) | 1.031 | most extreme: k=273: running_var 1.031 = 56.7x the channel median (beta/|gamma|=+1.79, pre_conv row norm 0.67) | no | 0 |
| MEDIUM | qeu8init sf100sl100 vs-UCI | 20260714-1-NYAZ@758000 | hot BN channel | `policy.pre_bn.running_var` | 7 ch: k=81, k=505, k=334, k=63, k=301, k=174, k=252 | 0.293 | most extreme: k=81: running_var 0.293 = 16.8x the channel median (beta/|gamma|=+0.15, pre_conv row norm 0.554) | no | 0 |
| MEDIUM | v5 | 20260729-1-VZ2j@106333 (cum 811769) | always-on pre-block channel | `policy.pre_bn` | 10 ch: k=119, k=34, k=62, k=126, k=111, k=28, k=97, k=42, k=65, k=105 | 6.774 | most extreme: k=119: beta/|gamma|=+6.77: the ReLU never clips it, so the channel is linear and its mean E[a]=7.25 acts as a constant input; it adds -10.5 to every logit through the final conv's mean row (column shared fraction 1.00) | no | 0 |
| MEDIUM | v5 | 20260714-1-h7vI@336610 (cum 705436) | always-on pre-block channel | `policy.pre_bn` | 6 ch: k=40, k=119, k=126, k=34, k=28, k=62 | 4.848 | most extreme: k=40: beta/|gamma|=+4.85: the ReLU never clips it, so the channel is linear and its mean E[a]=5 acts as a constant input; it adds -3.82 to every logit through the final conv's mean row (column shared fraction 0.99) | no | 0 |
| MEDIUM | v5 | 20260703-1-Dg5v@268506 (cum 368826) | elevated shared policy logit level | `policy.conv` |  | -16.95 | static mean logit -17.0: bf16 spacing 0.125 | no | 0 |
| MEDIUM | v5 | 20260714-1-h7vI@336610 (cum 705436) | hot BN channel | `policy.pre_bn.running_var` | 28 ch: k=111, k=34, k=97, k=105, k=62, k=42, k=86, k=61, k=96, k=12, k=119, k=8 (+16 more) | 340 | most extreme: k=111: running_var 340 = 337.4x the channel median (beta/|gamma|=+2.63, pre_conv row norm 3.29) | no | 0 |
| LOW | Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy@20000 | bias mean drift | `policy.conv.bias` |  | -0.5886 | softmax-invisible shared part of the bias (init 0); no weight decay on biases | yes | 2 |
| LOW | Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy@20000 | tower channel barely read by the policy head | `policy.pre_conv.weight` | 2 ch: col 18, col 39 |  | most extreme: col 18: pre_conv input column norm < 10% of median | yes | 0 |
| LOW | Ejp0 self-play run 1 | 20260727-1-Ejp0-9@197340 [champion] | always-on pre-block channel | `policy.pre_bn` | 6 ch: k=157, k=133, k=359, k=169, k=312, k=150 | 3.803 | most extreme: k=157: beta/|gamma|=+3.8: the ReLU never clips it, so the channel is linear and its mean E[a]=3.92 acts as a constant input; it adds -0.807 to every logit through the final conv's mean row (column shared fraction 1.00) | yes | 1 |
| LOW | Ejp0 self-play run 1 | 20260727-1-Ejp0-10@197340 [trainer] | always-on pre-block channel | `policy.pre_bn` | 7 ch: k=350, k=157, k=133, k=359, k=169, k=312, k=150 | 4.014 | most extreme: k=350: beta/|gamma|=+4.01: the ReLU never clips it, so the channel is linear and its mean E[a]=4.41 acts as a constant input; it adds -0.98 to every logit through the final conv's mean row (column shared fraction 1.00) | yes | 0 |
| LOW | Ejp0 self-play run 1 | 20260727-1-Ejp0-10@197340 [trainer] | bias mean drift | `policy.conv.bias` |  | -0.9477 | softmax-invisible shared part of the bias (init 0); no weight decay on biases | yes | 3 |
| LOW | Ejp0 self-play run 2 | 20260727-1-Ejp0-68@1186322 [trainer] | bias mean drift | `policy.conv.bias` |  | -0.9235 | softmax-invisible shared part of the bias (init 0); no weight decay on biases | yes | 3 |
| LOW | qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0@1397000 (cum 1572915) | bias mean drift | `policy.conv.bias` |  | -0.9811 | softmax-invisible shared part of the bias (init 0); no weight decay on biases | yes | 1 |
| LOW | v5 | 20260805-1-0pTW@2000 (cum 859769) | bias mean drift | `policy.conv.bias` |  | -0.8046 | softmax-invisible shared part of the bias (init 0); no weight decay on biases | yes | 3 |
| LOW | v5 | 20260805-1-0pTW@2000 (cum 859769) | tower channel barely read by the policy head | `policy.pre_conv.weight` | 9 ch: col 14, col 31, col 32, col 35, col 39, col 56, col 75, col 77, col 87 |  | most extreme: col 14: pre_conv input column norm < 10% of median | yes | 1 |
| LOW | Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy@11000 | tower channel barely read by the policy head | `policy.pre_conv.weight` | 1 ch: col 39 |  | most extreme: col 39: pre_conv input column norm < 10% of median | no | 0 |
| LOW | qeu8init sf100sl100 vs-UCI | 20260714-1-NYAZ@758000 | trajectory jump at segment boundary | `mean_row_norm` | from 20260712-6-lTiK@220000 | 1.861 | mean_row_norm 0.4511 -> 0.8395 between consecutive saves | no | 0 |
| LOW | qeu8init sf100sl100 vs-UCI | 20260714-1-NYAZ@758000 | trajectory jump at segment boundary | `pre_row_norm_median` | from 20260712-6-lTiK@220000 | 0.4253 | pre_row_norm_median 0.5128 -> 0.2181 between consecutive saves | no | 0 |
| LOW | v5 | 20260714-1-h7vI@336610 (cum 705436) | tower channel barely read by the policy head | `policy.pre_conv.weight` | 2 ch: col 31, col 32 |  | most extreme: col 31: pre_conv input column norm < 10% of median | no | 0 |
| LOW | v5 | 20260729-1-VZ2j@106333 (cum 811769) | tower channel barely read by the policy head | `policy.pre_conv.weight` | 6 ch: col 31, col 32, col 39, col 56, col 77, col 87 |  | most extreme: col 31: pre_conv input column norm < 10% of median | no | 0 |
| LOW | v5 | 20260714-1-h7vI@107000 | trajectory jump at segment boundary | `mean_row_norm` | from 20260703-1-Dg5v@268506 | 1.461 | mean_row_norm 1.369 -> 2 between consecutive saves | no | 0 |
| LOW | v5 | 20260714-1-h7vI@107000 | trajectory jump at segment boundary | `rv_max` | from 20260703-1-Dg5v@268506 | 1.27 | rv_max 1.82 -> 2.312 between consecutive saves | no | 0 |

<!-- end:findings_ranked -->

## E. Cross-lineage comparison (one row per lineage-latest; champion and trainer both shown for self-play)

"Underpromo vs Q median: row / residual": the raw row ratio is masked by the shared row on offset-heavy lines (v5 0.98, Ejp0 0.95). The residual ratio shows the real gap (0.46, 0.33).

<!-- begin:cross_lineage -->

| lineage | checkpoint | cum step | tower | policy style | K | compute | dead/mostly-off/always-on | rv max/mean / max/median | max abs(mu)/sigma | mean-row ratio | static shared logit level | bias mean | weakest family (vs Q median) | underpromo vs Q median: row / residual | max abs W (final) | MEDIUM+ findings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SE scale+bias full-leaky fresh (never trained) | 20261001-23-Dmwe @ fresh |  | stem7 3x7x7@128 SE:scale_and_bias | intermediate_conv | 128 | bfloat16 | 0/0/0 | 2.4 / 2.6 | 2.5 | 0.10 | -0.1 | +0.00 | queen-style dir N (0.95) | 0.98 / 0.98 | 0.48 |  |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 12000 | 12000 | stem7 3x7x7@128 SE:scale_and_bias | intermediate_conv | 128 | bfloat16 | 0/0/0 | 3.1 / 3.4 | 1.7 | 0.20 | +0.6 | -0.00 | underpromo dir capL (0.72) | 0.77 / 0.81 | 1.08 |  |
| SE scale+bias s1 | 20260929-22-bWdy @ 33014 | 33014 | stem7 3x7x7@128 SE:scale_and_bias | intermediate_conv | 128 | bfloat16 | 0/0/0 | 3.4 / 3.8 | 2.1 | 0.22 | +0.6 | -0.00 | underpromo dir capL (0.61) | 0.67 / 0.71 | 1.66 |  |
| SE scale+bias s2 | 20260930-4-k98x @ 7282 | 7282 | stem7 3x7x7@128 SE:scale_and_bias | intermediate_conv | 128 | bfloat16 | 0/0/0 | 2.7 / 2.9 | 1.5 | 0.21 | +0.5 | +0.00 | underpromo dir capL (0.73) | 0.77 / 0.82 | 0.80 |  |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 33012 | 33012 | stem7 3x7x7@128 SE:attenuate_only | intermediate_conv | 128 | bfloat16 | 0/0/0 | 3.3 / 3.8 | 2.5 | 0.23 | +0.9 | +0.00 | underpromo dir capR (0.56) | 0.66 / 0.69 | 1.40 |  |
| SE attenuate-only s2 | 20260930-5-5TXu @ 7289 | 7289 | stem7 3x7x7@128 SE:attenuate_only | intermediate_conv | 128 | bfloat16 | 0/0/0 | 2.8 / 3.1 | 1.9 | 0.21 | +0.7 | +0.00 | underpromo dir capL (0.73) | 0.79 / 0.83 | 0.83 |  |
| SE none s1 | 20260929-24-834D @ 32036 | 32036 | stem7 3x7x7@128 SE:none | intermediate_conv | 128 | bfloat16 | 0/0/0 | 3.0 / 3.3 | 2.9 | 0.22 | +0.8 | -0.00 | underpromo dir capR (0.60) | 0.67 / 0.72 | 1.38 |  |
| SE none s2 | 20260930-6-LkS6 @ 7019 | 7019 | stem7 3x7x7@128 SE:none | intermediate_conv | 128 | bfloat16 | 0/0/0 | 2.0 / 2.2 | 2.1 | 0.21 | +0.8 | +0.00 | underpromo dir capR (0.72) | 0.78 / 0.83 | 0.85 |  |
| SE zero-beta s1 | 20260930-9-RrGx @ 5030 | 5030 | stem7 3x7x7@128 SE:scale_and_bias | intermediate_conv | 128 | bfloat16 | 0/0/0 | 3.6 / 3.9 | 1.9 | 0.21 | +0.6 | -0.00 | underpromo dir capL (0.72) | 0.78 / 0.81 | 1.05 |  |
| SE zero-beta s2 | 20260930-10-H51a @ 5004 | 5004 | stem7 3x7x7@128 SE:scale_and_bias | intermediate_conv | 128 | bfloat16 | 0/0/0 | 2.7 / 3.0 | 1.6 | 0.21 | +0.5 | -0.00 | underpromo dir capL (0.75) | 0.79 / 0.83 | 0.81 |  |
| v5 | 20260805-1-0pTW @ 2000 | 859769 | stem7 5x7x7@128 SE:scale_and_bias | intermediate_conv | 128 | bfloat16 | 0/0/11 | 15.2 / 1372.7 | 3.6 | 0.97 | -300.4 | -0.80 | underpromo dir capL (0.98) | 0.98 / 0.46 | 2.73 | BN running-var spread; always-on pre-block channel; hot BN channel; large shared policy logit level; large shared row; shared offset carried by always-on channels; shared-row feature-rounding noise |
| mini2b | 20260705-1-znR7 @ 114000 | 256159 | stem7 1x7x7@128 SE:none + 1x3x3@128 SE:none | intermediate_conv | 128 | bfloat16 | 0/0/0 | 2.2 / 2.3 | 2.5 | 0.49 | -6.8 | -0.22 | underpromo dir capL (0.62) | 0.64 / 0.56 | 1.94 |  |
| coxw | 20260709-1-avoB @ 277000 | 332550 | stem5 1x7x7@128 SE:none | intermediate_conv | 128 | bfloat16 | 0/0/0 | 4.4 / 5.3 | 3.3 | 0.43 | -3.1 | -0.20 | underpromo dir capR (0.48) | 0.52 / 0.47 | 1.48 |  |
| ykkk | 20260710-1-amlg @ 250803 | 453810 | stem3 2x3x3@64 SE:attenuate_only | simple_conv | — | bfloat16 | n/a | n/a | n/a | 0.45 | n/a | -0.26 | underpromo dir capR (0.43) | 0.53 / 0.48 | 1.36 |  |
| nt8y | 20260708-4-kEiZ @ 21086 | 312748 | stem5 3x15x15@32 SE:scale_and_bias | intermediate_conv | 512 | bfloat16 | 0/0/0 | 2.9 / 3.2 | 10.6 | 0.55 | -14.1 | -0.23 | underpromo dir capL (0.65) | 0.68 / 0.53 | 2.20 | BN input mean >> std; elevated shared policy logit level; large shared row |
| qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0 @ 1397000 | 1572915 | stem7 2x15x15@64 SE:scale_and_bias | intermediate_conv | 512 | bfloat16 | 0/0/20 | 57.9 / 1219.7 | 17.7 | 0.95 | -157.9 | -0.98 | underpromo dir capL (0.95) | 0.95 / 0.33 | 1.33 | BN input mean >> std; BN running-var spread; always-on pre-block channel; hot BN channel; large shared policy logit level; large shared row; shared offset carried by always-on channels; shared-row feature-rounding noise |
| qeu8e (epoch branch) | 20260708-6-sFzi @ 88107 | 220837 | stem7 2x15x15@64 SE:scale_and_bias | intermediate_conv | 512 | bfloat16 | 0/0/0 | 3.5 / 3.9 | 3.6 | 0.47 | -7.1 | -0.13 | underpromo dir capL (0.55) | 0.60 / 0.55 | 1.87 |  |
| qeu8-1blk128 | 20260711-17-pycz @ 120000 | 120000 | stem7 1x15x15@128 SE:scale_and_bias | intermediate_conv | 512 | bfloat16 | 0/0/0 | 2.7 / 2.9 | 2.2 | 0.30 | -2.2 | -0.06 | underpromo dir capL (0.61) | 0.65 / 0.69 | 1.65 |  |
| qeu8init sf100sl100 vs-UCI | 20260722-1-syxR @ 558000 |  | stem7 2x15x15@64 SE:scale_and_bias | intermediate_conv | 512 | bfloat16 | 0/0/0 | 12.9 / 25.8 | 3.4 | 0.66 | -16.0 | -0.44 | underpromo bishop (0.66) | 0.67 / 0.30 | 0.88 | elevated shared policy logit level; hot BN channel; large shared row |
| Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy @ 20000 |  | stem7 2x15x15@64 SE:scale_and_bias | intermediate_conv | 512 | bfloat16 | 0/0/0 | 20.6 / 50.6 | 9.3 | 0.74 | -43.8 | -0.59 | underpromo dir capR (0.83) | 0.83 / 0.35 | 1.53 | BN input mean >> std; hot BN channel; large shared policy logit level; large shared row |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-10 @ 197340 [trainer] |  | stem7 2x15x15@64 SE:scale_and_bias | intermediate_conv | 512 | bfloat16 | 0/0/20 | 44.8 / 521.6 | 15.2 | 0.91 | -69.6 | -0.95 | underpromo dir capL (0.92) | 0.92 / 0.33 | 0.80 | BN input mean >> std; BN running-var spread; always-on pre-block channel; hot BN channel; large shared policy logit level; large shared row; shared offset carried by always-on channels; shared-row feature-rounding noise |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-9 @ 197340 [champion] |  | stem7 2x15x15@64 SE:scale_and_bias | intermediate_conv | 512 | bfloat16 | 0/0/20 | 44.5 / 502.6 | 14.7 | 0.91 | -71.9 | -0.95 | underpromo dir capL (0.92) | 0.92 / 0.33 | 0.82 | BN input mean >> std; BN running-var spread; always-on pre-block channel; hot BN channel; large shared policy logit level; large shared row; shared offset carried by always-on channels; shared-row feature-rounding noise |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-68 @ 1186322 [trainer] |  | stem7 2x15x15@64 SE:scale_and_bias | intermediate_conv | 512 | bfloat16 | 0/0/16 | 56.2 / 1696.2 | 15.9 | 0.94 | -138.9 | -0.92 | underpromo dir capL (0.94) | 0.94 / 0.36 | 1.23 | BN input mean >> std; BN running-var spread; always-on pre-block channel; hot BN channel; large shared policy logit level; large shared row; shared offset carried by always-on channels; shared-row feature-rounding noise |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-67 @ 1186322 [champion] |  | stem7 2x15x15@64 SE:scale_and_bias | intermediate_conv | 512 | bfloat16 | 0/0/16 | 56.2 / 1696.2 | 15.9 | 0.94 | -138.9 | -0.92 | underpromo dir capL (0.94) | 0.94 / 0.36 | 1.23 | BN input mean >> std; BN running-var spread; always-on pre-block channel; hot BN channel; large shared policy logit level; large shared row; shared offset carried by always-on channels; shared-row feature-rounding noise |
| bzw3 self-play | 20260601-11-bzw3-32 @ 467099 [trainer] |  | stem7 5x7x7@128 SE:scale_and_bias | intermediate_conv | 128 | bfloat16 | 0/0/0 | 4.4 / 5.2 | 2.2 | 0.56 | -14.0 | -0.28 | underpromo dir capL (0.72) | 0.76 / 0.65 | 2.79 | elevated shared policy logit level; large shared row |
| bzw3 self-play | 20260601-11-bzw3-31 @ 467099 [champion] |  | stem7 5x7x7@128 SE:scale_and_bias | intermediate_conv | 128 | bfloat16 | 0/0/0 | 3.5 / 4.1 | 1.3 | 0.48 | -7.7 | -0.19 | underpromo dir capL (0.76) | 0.80 / 0.75 | 1.85 |  |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-23 @ 532369 [trainer] |  | stem3 8x3x3@128 SE:attenuate_only | simple_conv | — | float32 | n/a | n/a | n/a | 0.16 | n/a | -0.00 | queen-style dist 6 (0.95) | 0.99 / 1.00 | 0.46 | final-conv row ~unchanged from reference |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-22 @ 532369 [champion] |  | stem3 8x3x3@128 SE:attenuate_only | simple_conv | — | float32 | n/a | n/a | n/a | 0.16 | n/a | -0.00 | queen-style dist 6 (0.95) | 0.99 / 1.00 | 0.46 | final-conv row ~unchanged from reference |
| sMe9 self-play (fp32) | 20260525-1-sMe9-33 @ 373416 [trainer] |  | v3_8block_3x3 (8x3x3 @128) | simple_conv | — | float32 | n/a | n/a | n/a | 0.19 | n/a | -0.00 | knight (0.89) | 0.95 / 0.94 | 0.62 |  |
| sMe9 self-play (fp32) | 20260525-1-sMe9-32 @ 373416 [champion] |  | v3_8block_3x3 (8x3x3 @128) | simple_conv | — | float32 | n/a | n/a | n/a | 0.19 | n/a | -0.00 | knight (0.90) | 0.96 / 0.95 | 0.62 |  |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-11 @ 106695 [trainer] |  | v4_12block_3x3 (12x3x3 @128) | intermediate_conv | 128 | bfloat16 | 0/0/0 | 2.1 / 2.1 | 1.2 | 0.19 | -0.2 | -0.02 | underpromo dir fwd (0.84) | 0.88 / 0.89 | 1.57 |  |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-10 @ 106695 [champion] |  | v4_12block_3x3 (12x3x3 @128) | intermediate_conv | 128 | bfloat16 | 0/0/0 | 2.0 / 2.2 | 1.0 | 0.18 | -0.1 | -0.01 | underpromo dir fwd (0.87) | 0.90 / 0.92 | 1.31 |  |
| LMGh self-play | 20260609-12-LMGh-4 @ 79135 [trainer] |  | stem3 50x3x3@32 SE:scale_and_bias | intermediate_conv | 32 | bfloat16 | 0/0/0 | 3.9 / 4.9 | 4.2 | 0.35 | -7.6 | -0.01 | queen-style dist 7 (0.79) | 1.06 / 1.15 | 3.44 |  |
| LMGh self-play | 20260609-12-LMGh-3 @ 79135 [champion] |  | stem3 50x3x3@32 SE:scale_and_bias | intermediate_conv | 32 | bfloat16 | 0/0/0 | 3.9 / 4.9 | 4.2 | 0.35 | -7.6 | -0.01 | queen-style dist 7 (0.79) | 1.06 / 1.15 | 3.44 |  |
| WjRY self-play | 20260609-14-WjRY-8 @ 98974 [trainer] |  | stem3 8x3x3@128 SE:scale_and_bias | intermediate_conv | 128 | bfloat16 | 0/0/0 | 5.1 / 6.5 | 2.3 | 0.40 | -3.7 | -0.02 | underpromo rook (0.90) | 0.95 / 1.00 | 1.88 |  |
| WjRY self-play | 20260609-14-WjRY-7 @ 98974 [champion] |  | stem3 8x3x3@128 SE:scale_and_bias | intermediate_conv | 128 | bfloat16 | 0/0/0 | 3.1 / 3.3 | 1.2 | 0.18 | -0.1 | -0.01 | underpromo rook (0.85) | 0.90 / 0.90 | 0.91 |  |

<!-- end:cross_lineage -->

## Offset carriers (lineage-latest)

<!-- begin:offset_carriers -->

| lineage | checkpoint | static shared level | from bias mean | from mean row · E[a] | always-on channels | from always-on | top 5 contributors (k: m_k·E[a_k]) |
|---|---|---|---|---|---|---|---|
| SE scale+bias full-leaky fresh (never trained) | 20261001-23-Dmwe @ fresh | -0.05 | +0.000 | -0.05 | 0 | +0.00 | k28: -0.01 (β +0.00, γ +1.00, rv 0.73, col 1.10, shared frac 0.30); k38: +0.01 (β +0.00, γ +1.00, rv 0.711, col 1.04, shared frac 0.29); k22: -0.01 (β +0.00, γ +1.00, rv 0.93, col 1.16, shared frac 0.22); k78: -0.01 (β +0.00, γ +1.00, rv 0.57, col 0.95, shared frac 0.27); k83: +0.01 (β +0.00, γ +1.00, rv 0.922, col 0.84, shared frac 0.27) |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 12000 | +0.57 | -0.000 | +0.57 | 0 | +0.00 | k38: +0.05 (β -0.00, γ +1.16, rv 1.41, col 1.57, shared frac 0.64); k52: +0.04 (β -0.01, γ +1.19, rv 0.76, col 1.48, shared frac 0.49); k120: -0.04 (β +0.27, γ +1.47, rv 0.813, col 1.51, shared frac 0.30); k55: +0.04 (β -0.01, γ +1.31, rv 1.22, col 1.73, shared frac 0.36); k99: +0.03 (β -0.05, γ +1.20, rv 0.716, col 1.52, shared frac 0.39) |
| SE scale+bias s1 | 20260929-22-bWdy @ 33014 | +0.59 | -0.000 | +0.59 | 0 | +0.00 | k120: -0.07 (β +0.41, γ +2.03, rv 0.5, col 2.10, shared frac 0.30); k38: +0.05 (β -0.07, γ +1.09, rv 1.16, col 1.56, shared frac 0.65); k52: +0.04 (β -0.04, γ +1.21, rv 0.777, col 1.55, shared frac 0.47); k55: +0.04 (β -0.04, γ +1.26, rv 1.13, col 1.81, shared frac 0.37); k81: +0.03 (β +0.07, γ +1.23, rv 1.22, col 1.68, shared frac 0.31) |
| SE scale+bias s2 | 20260930-4-k98x @ 7282 | +0.52 | +0.000 | +0.52 | 0 | +0.00 | k124: +0.04 (β +0.03, γ +1.21, rv 0.974, col 1.63, shared frac 0.45); k127: +0.04 (β +0.13, γ +1.28, rv 1.11, col 1.80, shared frac 0.34); k54: +0.04 (β +0.04, γ +1.19, rv 0.979, col 1.69, shared frac 0.38); k38: -0.03 (β +0.13, γ +1.09, rv 1.16, col 1.18, shared frac 0.50); k8: +0.03 (β +0.06, γ +1.18, rv 0.886, col 1.66, shared frac 0.34) |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 33012 | +0.86 | +0.000 | +0.86 | 0 | +0.00 | k101: +0.05 (β +0.10, γ +1.12, rv 0.785, col 1.68, shared frac 0.48); k5: +0.04 (β -0.05, γ +1.08, rv 0.656, col 1.59, shared frac 0.59); k93: +0.03 (β +0.18, γ +1.12, rv 1.66, col 1.55, shared frac 0.35); k35: +0.03 (β +0.15, γ +1.05, rv 1.29, col 1.59, shared frac 0.33); k119: +0.03 (β +0.10, γ +1.15, rv 1.04, col 1.51, shared frac 0.34) |
| SE attenuate-only s2 | 20260930-5-5TXu @ 7289 | +0.69 | +0.000 | +0.69 | 0 | +0.00 | k96: +0.05 (β +0.18, γ +1.23, rv 1.44, col 1.53, shared frac 0.45); k80: +0.04 (β -0.06, γ +1.17, rv 0.91, col 1.45, shared frac 0.55); k68: +0.04 (β +0.11, γ +1.32, rv 1.19, col 1.78, shared frac 0.34); k100: +0.04 (β +0.05, γ +1.29, rv 0.756, col 1.51, shared frac 0.38); k73: +0.03 (β +0.07, γ +1.23, rv 0.97, col 1.52, shared frac 0.38) |
| SE none s1 | 20260929-24-834D @ 32036 | +0.79 | -0.000 | +0.79 | 0 | +0.00 | k115: +0.06 (β +0.14, γ +1.41, rv 1.38, col 1.93, shared frac 0.44); k32: +0.04 (β +0.03, γ +1.16, rv 0.652, col 1.57, shared frac 0.50); k79: -0.04 (β +0.10, γ +1.76, rv 0.262, col 1.68, shared frac 0.25); k15: +0.03 (β -0.04, γ +1.23, rv 0.75, col 1.59, shared frac 0.38); k27: +0.03 (β -0.00, γ +1.14, rv 0.566, col 1.44, shared frac 0.42) |
| SE none s2 | 20260930-6-LkS6 @ 7019 | +0.81 | +0.000 | +0.81 | 0 | +0.00 | k99: +0.04 (β -0.04, γ +1.16, rv 0.761, col 1.58, shared frac 0.54); k77: +0.04 (β +0.06, γ +1.23, rv 0.864, col 1.58, shared frac 0.40); k74: +0.04 (β +0.02, γ +1.14, rv 0.85, col 1.43, shared frac 0.48); k91: +0.04 (β +0.16, γ +1.10, rv 1.16, col 1.57, shared frac 0.37); k85: +0.03 (β +0.01, γ +1.17, rv 0.997, col 1.63, shared frac 0.39) |
| SE zero-beta s1 | 20260930-9-RrGx @ 5030 | +0.61 | -0.000 | +0.61 | 0 | +0.00 | k38: +0.05 (β -0.01, γ +1.13, rv 1.33, col 1.56, shared frac 0.65); k52: +0.04 (β +0.00, γ +1.15, rv 0.976, col 1.49, shared frac 0.47); k55: +0.04 (β -0.02, γ +1.30, rv 1.32, col 1.74, shared frac 0.36); k96: +0.03 (β +0.05, γ +1.36, rv 0.759, col 1.59, shared frac 0.33); k120: -0.03 (β +0.26, γ +1.42, rv 0.772, col 1.51, shared frac 0.27) |
| SE zero-beta s2 | 20260930-10-H51a @ 5004 | +0.53 | -0.000 | +0.53 | 0 | +0.00 | k127: +0.04 (β +0.13, γ +1.29, rv 1.29, col 1.80, shared frac 0.34); k124: +0.04 (β +0.01, γ +1.20, rv 1.01, col 1.60, shared frac 0.43); k5: +0.03 (β +0.08, γ +1.18, rv 0.866, col 1.49, shared frac 0.38); k54: +0.03 (β +0.03, γ +1.14, rv 1.08, col 1.64, shared frac 0.37); k76: +0.03 (β -0.06, γ +1.14, rv 1.09, col 1.64, shared frac 0.39) |
| v5 | 20260805-1-0pTW @ 2000 | -300.38 | -0.805 | -299.58 | 11 | -115.08 | k34: -17.20 (β +8.62, γ +0.90, rv 5.86e+03, col 17.40, shared frac 1.00); k111: -15.16 (β +7.94, γ +1.45, rv 5.98e+03, col 16.67, shared frac 1.00); k97: -13.53 (β +7.41, γ +1.77, rv 5.44e+03, col 15.95, shared frac 1.00); k105: -12.13 (β +6.91, γ +1.96, rv 4.9e+03, col 15.33, shared frac 1.00); k62: -11.87 (β +7.09, γ +1.09, rv 3.44e+03, col 14.61, shared frac 1.00) |
| mini2b | 20260705-1-znR7 @ 114000 | -6.82 | -0.217 | -6.60 | 0 | +0.00 | k90: -0.52 (β +1.95, γ +1.88, rv 1.57, col 2.29, shared frac 0.94); k108: -0.41 (β +1.50, γ +2.61, rv 0.781, col 2.75, shared frac 0.66); k123: -0.33 (β +1.54, γ +1.65, rv 1.12, col 1.86, shared frac 0.92); k31: -0.32 (β +1.44, γ +1.73, rv 1.28, col 1.83, shared frac 0.92); k12: -0.30 (β +1.49, γ +1.55, rv 1.51, col 1.85, shared frac 0.86) |
| coxw | 20260709-1-avoB @ 277000 | -3.11 | -0.202 | -2.91 | 0 | +0.00 | k8: -0.32 (β +1.48, γ +2.62, rv 0.11, col 2.14, shared frac 0.68); k104: -0.20 (β +1.11, γ +3.23, rv 0.121, col 2.14, shared frac 0.43); k112: -0.19 (β +1.18, γ +1.64, rv 0.189, col 1.31, shared frac 0.92); k44: -0.18 (β +1.08, γ +1.60, rv 0.124, col 1.26, shared frac 0.92); k17: -0.14 (β +0.85, γ +1.67, rv 0.0879, col 1.20, shared frac 0.86) |
| nt8y | 20260708-4-kEiZ @ 21086 | -14.09 | -0.228 | -13.87 | 0 | +0.00 | k26: -0.26 (β +1.29, γ +3.38, rv 0.153, col 2.91, shared frac 0.38); k123: -0.24 (β +0.79, γ +2.89, rv 0.149, col 2.10, shared frac 0.63); k297: -0.19 (β +1.08, γ +2.23, rv 0.12, col 1.90, shared frac 0.58); k451: -0.19 (β +1.05, γ +2.94, rv 0.149, col 2.57, shared frac 0.37); k387: -0.19 (β +0.83, γ +2.94, rv 0.2, col 2.47, shared frac 0.41) |
| qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0 @ 1397000 | -157.94 | -0.981 | -156.96 | 20 | -68.56 | k273: -8.57 (β +9.25, γ +0.91, rv 60, col 8.08, shared frac 1.00); k163: -6.52 (β +8.00, γ +0.96, rv 40, col 7.11, shared frac 1.00); k230: -6.38 (β +7.91, γ +1.00, rv 40.5, col 7.04, shared frac 1.00); k244: -4.98 (β +6.88, γ +1.08, rv 13.2, col 6.32, shared frac 1.00); k52: -4.15 (β +6.25, γ +1.05, rv 13.1, col 5.79, shared frac 1.00) |
| qeu8e (epoch branch) | 20260708-6-sFzi @ 88107 | -7.07 | -0.133 | -6.93 | 0 | +0.00 | k71: -0.14 (β +0.86, γ +2.97, rv 0.188, col 2.16, shared frac 0.35); k311: -0.11 (β +0.70, γ +1.77, rv 0.246, col 1.16, shared frac 0.73); k170: -0.11 (β +0.73, γ +1.60, rv 0.24, col 1.09, shared frac 0.80); k49: -0.09 (β +0.61, γ +1.77, rv 0.241, col 1.13, shared frac 0.65); k272: -0.09 (β +0.33, γ +2.75, rv 0.108, col 1.89, shared frac 0.32) |
| qeu8-1blk128 | 20260711-17-pycz @ 120000 | -2.18 | -0.063 | -2.12 | 0 | +0.00 | k355: -0.06 (β +0.32, γ +2.34, rv 0.203, col 1.83, shared frac 0.28); k373: -0.04 (β +0.16, γ +1.98, rv 0.123, col 1.53, shared frac 0.28); k354: -0.04 (β +0.12, γ +2.05, rv 0.134, col 1.53, shared frac 0.24); k82: -0.03 (β +0.18, γ +1.33, rv 0.164, col 0.84, shared frac 0.56); k103: -0.03 (β +0.23, γ +1.44, rv 0.175, col 0.98, shared frac 0.41) |
| qeu8init sf100sl100 vs-UCI | 20260722-1-syxR @ 558000 | -15.98 | -0.442 | -15.54 | 0 | +0.00 | k401: -0.22 (β +2.08, γ +2.12, rv 0.0236, col 0.91, shared frac 0.91); k213: -0.20 (β +1.87, γ +2.17, rv 0.00909, col 0.91, shared frac 0.90); k459: -0.19 (β +2.06, γ +2.00, rv 0.0115, col 0.86, shared frac 0.88); k429: -0.18 (β +1.66, γ +1.85, rv 0.0248, col 0.92, shared frac 0.91); k503: -0.18 (β +1.80, γ +2.03, rv 0.0139, col 0.83, shared frac 0.92) |
| Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy @ 20000 | -43.82 | -0.589 | -43.24 | 0 | +0.00 | k273: -0.74 (β +2.76, γ +1.38, rv 0.987, col 2.36, shared frac 0.99); k163: -0.55 (β +2.32, γ +1.41, rv 0.837, col 2.04, shared frac 0.99); k230: -0.53 (β +2.28, γ +1.45, rv 0.899, col 2.03, shared frac 0.99); k501: -0.53 (β +2.61, γ +2.69, rv 0.234, col 1.92, shared frac 0.84); k39: -0.50 (β +3.08, γ +2.51, rv 1.52, col 1.43, shared frac 0.96) |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-10 @ 197340 [trainer] | -69.64 | -0.948 | -68.69 | 20 | -27.43 | k273: -3.46 (β +8.70, γ +0.82, rv 4.48, col 3.46, shared frac 1.00); k163: -2.64 (β +7.57, γ +0.89, rv 3.06, col 3.04, shared frac 1.00); k230: -2.57 (β +7.45, γ +0.93, rv 3.01, col 3.01, shared frac 1.00); k244: -1.98 (β +6.45, γ +1.03, rv 1.36, col 2.68, shared frac 1.00); k52: -1.66 (β +5.85, γ +1.01, rv 1.13, col 2.47, shared frac 1.00) |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-9 @ 197340 [champion] | -71.92 | -0.947 | -70.97 | 20 | -28.49 | k273: -3.59 (β +8.69, γ +0.82, rv 5, col 3.60, shared frac 1.00); k163: -2.74 (β +7.56, γ +0.89, rv 3.42, col 3.16, shared frac 1.00); k230: -2.67 (β +7.44, γ +0.93, rv 3.36, col 3.13, shared frac 1.00); k244: -2.05 (β +6.44, γ +1.03, rv 1.52, col 2.78, shared frac 1.00); k52: -1.72 (β +5.84, γ +1.01, rv 1.27, col 2.57, shared frac 1.00) |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-68 @ 1186322 [trainer] | -138.92 | -0.924 | -138.00 | 16 | -50.09 | k273: -7.01 (β +9.06, γ +1.16, rv 33.8, col 6.75, shared frac 1.00); k163: -5.34 (β +7.84, γ +1.19, rv 26, col 5.94, shared frac 1.00); k230: -5.22 (β +7.72, γ +1.23, rv 21.9, col 5.90, shared frac 1.00); k244: -4.01 (β +6.66, γ +1.30, rv 9.5, col 5.26, shared frac 1.00); k52: -3.38 (β +6.06, γ +1.26, rv 7.59, col 4.87, shared frac 1.00) |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-67 @ 1186322 [champion] | -138.92 | -0.924 | -138.00 | 16 | -50.09 | k273: -7.01 (β +9.06, γ +1.16, rv 33.8, col 6.75, shared frac 1.00); k163: -5.34 (β +7.84, γ +1.19, rv 26, col 5.94, shared frac 1.00); k230: -5.22 (β +7.72, γ +1.23, rv 21.9, col 5.90, shared frac 1.00); k244: -4.01 (β +6.66, γ +1.30, rv 9.5, col 5.26, shared frac 1.00); k52: -3.38 (β +6.06, γ +1.26, rv 7.59, col 4.87, shared frac 1.00) |
| bzw3 self-play | 20260601-11-bzw3-32 @ 467099 [trainer] | -14.04 | -0.276 | -13.77 | 0 | +0.00 | k91: -0.99 (β +2.50, γ +2.42, rv 1.28, col 3.84, shared frac 0.83); k72: -0.97 (β +2.71, γ +2.91, rv 1.3, col 4.14, shared frac 0.68); k13: -0.96 (β +2.38, γ +2.97, rv 1.07, col 3.69, shared frac 0.83); k2: -0.91 (β +2.87, γ +2.14, rv 2.73, col 3.18, shared frac 0.84); k112: -0.72 (β +2.40, γ +1.49, rv 1.76, col 2.92, shared frac 0.88) |
| bzw3 self-play | 20260601-11-bzw3-31 @ 467099 [champion] | -7.69 | -0.189 | -7.50 | 0 | +0.00 | k72: -0.56 (β +1.94, γ +2.66, rv 0.66, col 3.06, shared frac 0.69); k13: -0.52 (β +1.64, γ +2.64, rv 0.699, col 2.83, shared frac 0.78); k2: -0.51 (β +2.00, γ +1.82, rv 2.61, col 2.49, shared frac 0.85); k91: -0.48 (β +1.74, γ +2.22, rv 1.12, col 2.65, shared frac 0.78); k114: -0.38 (β +1.66, γ +1.60, rv 3.25, col 2.00, shared frac 0.93) |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-11 @ 106695 [trainer] | -0.23 | -0.016 | -0.21 | 0 | +0.00 | k59: -0.05 (β +0.28, γ +1.82, rv 0.633, col 1.87, shared frac 0.29); k83: -0.04 (β +0.28, γ +1.19, rv 2.03, col 1.21, shared frac 0.43); k33: +0.04 (β +0.17, γ +1.29, rv 2.74, col 1.73, shared frac 0.30); k9: -0.04 (β +0.20, γ +1.37, rv 0.917, col 1.53, shared frac 0.31); k116: +0.03 (β -0.00, γ +1.11, rv 2.17, col 1.63, shared frac 0.31) |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-10 @ 106695 [champion] | -0.06 | -0.013 | -0.04 | 0 | +0.00 | k33: +0.04 (β +0.15, γ +1.34, rv 2.61, col 1.69, shared frac 0.32); k59: -0.03 (β +0.18, γ +1.57, rv 0.742, col 1.59, shared frac 0.26); k9: -0.03 (β +0.15, γ +1.29, rv 0.953, col 1.45, shared frac 0.30); k83: -0.03 (β +0.20, γ +1.16, rv 1.99, col 1.19, shared frac 0.37); k49: +0.03 (β -0.04, γ +1.05, rv 1.57, col 1.41, shared frac 0.43) |
| LMGh self-play | 20260609-12-LMGh-4 @ 79135 [trainer] | -7.61 | -0.007 | -7.60 | 0 | +0.00 | k23: -0.89 (β +2.16, γ +3.64, rv 7.09, col 4.49, shared frac 0.62); k13: -0.85 (β +4.19, γ +1.62, rv 6.34, col 5.12, shared frac 0.35); k14: -0.83 (β +3.23, γ +1.64, rv 4.06, col 4.73, shared frac 0.47); k17: -0.81 (β +3.48, γ +2.31, rv 8.44, col 4.77, shared frac 0.42); k24: -0.56 (β +1.40, γ +2.66, rv 11.5, col 4.64, shared frac 0.56) |
| LMGh self-play | 20260609-12-LMGh-3 @ 79135 [champion] | -7.61 | -0.007 | -7.60 | 0 | +0.00 | k23: -0.89 (β +2.16, γ +3.64, rv 7.09, col 4.49, shared frac 0.62); k13: -0.85 (β +4.19, γ +1.62, rv 6.34, col 5.12, shared frac 0.35); k14: -0.83 (β +3.23, γ +1.64, rv 4.06, col 4.73, shared frac 0.47); k17: -0.81 (β +3.48, γ +2.31, rv 8.44, col 4.77, shared frac 0.42); k24: -0.56 (β +1.40, γ +2.66, rv 11.5, col 4.64, shared frac 0.56) |
| WjRY self-play | 20260609-14-WjRY-8 @ 98974 [trainer] | -3.67 | -0.020 | -3.65 | 0 | +0.00 | k10: -0.30 (β +1.40, γ +1.71, rv 1.82, col 2.24, shared frac 0.74); k52: -0.30 (β +1.33, γ +1.90, rv 1.19, col 2.35, shared frac 0.70); k70: -0.23 (β +1.08, γ +1.74, rv 0.72, col 2.13, shared frac 0.69); k28: +0.17 (β +1.63, γ +0.67, rv 7.15, col 2.28, shared frac 0.40); k58: -0.16 (β +1.00, γ +1.26, rv 1.04, col 1.77, shared frac 0.68) |
| WjRY self-play | 20260609-14-WjRY-7 @ 98974 [champion] | -0.06 | -0.012 | -0.05 | 0 | +0.00 | k55: -0.03 (β +0.12, γ +1.20, rv 0.535, col 1.25, shared frac 0.41); k38: +0.03 (β +0.01, γ +1.16, rv 1.44, col 1.51, shared frac 0.39); k28: +0.03 (β +0.27, γ +1.07, rv 3.03, col 1.62, shared frac 0.29); k52: -0.03 (β +0.14, γ +1.24, rv 0.84, col 1.35, shared frac 0.34); k8: -0.03 (β +0.08, γ +0.99, rv 1.02, col 1.08, shared frac 0.52) |

<!-- end:offset_carriers -->

## Growth over each lineage (first trained → last; all checkpoints in `results/trajectory.csv`)

<!-- begin:growth -->

| lineage | first trained | last | static_shared_level | bias_mean | mean_row_norm | row_norm_median | conv_w_max_abs | pre_row_norm_median | rv_max | gamma_median | shared_row_rounding_noise |
|---|---|---|---|---|---|---|---|---|---|---|---|
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz@1000 | 20261001-43-NbWz@12000 | 0.422 → 0.567 | 4.12e-06 → -1.51e-05 | 0.279 → 0.331 | 1.55 → 1.68 | 0.633 → 1.08 | 1.42 → 1.37 | 2.18 → 3.01 | 1.04 → 1.06 | 0.000524 → 0.000628 |
| SE scale+bias s1 | 20260929-22-bWdy@1000 | 20260929-22-bWdy@33014 | 0.411 → 0.591 | -1.66e-05 → -9.69e-06 | 0.277 → 0.36 | 1.54 → 1.69 | 0.629 → 1.66 | 1.42 → 1.28 | 2.25 → 2.94 | 1.04 → 1.03 | 0.000518 → 0.000687 |
| SE scale+bias s2 | 20260930-4-k98x@1000 | 20260930-4-k98x@7282 | 0.388 → 0.524 | 2.09e-05 → 6.03e-05 | 0.292 → 0.341 | 1.55 → 1.69 | 0.622 → 0.805 | 1.41 → 1.36 | 1.99 → 2.47 | 1.04 → 1.08 | 0.000536 → 0.000636 |
| SE attenuate-only s1 | 20260929-23-L6Qm@1000 | 20260929-23-L6Qm@33012 | 0.544 → 0.855 | 2.72e-05 → 4.43e-05 | 0.293 → 0.369 | 1.54 → 1.71 | 0.602 → 1.4 | 1.4 → 1.27 | 2.12 → 2.7 | 1.05 → 1.06 | 0.000541 → 0.000665 |
| SE attenuate-only s2 | 20260930-5-5TXu@1000 | 20260930-5-5TXu@7289 | 0.535 → 0.69 | 7.7e-06 → 3.96e-05 | 0.292 → 0.348 | 1.57 → 1.69 | 0.638 → 0.826 | 1.42 → 1.37 | 1.71 → 2.54 | 1.04 → 1.07 | 0.000538 → 0.000648 |
| SE none s1 | 20260929-24-834D@1000 | 20260929-24-834D@32036 | 0.545 → 0.786 | -9.84e-06 → -6.29e-05 | 0.271 → 0.369 | 1.55 → 1.72 | 0.652 → 1.38 | 1.41 → 1.27 | 2.27 → 2.19 | 1.05 → 1.09 | 0.0005 → 0.000695 |
| SE none s2 | 20260930-6-LkS6@1000 | 20260930-6-LkS6@7019 | 0.593 → 0.812 | 1.66e-05 → 5.72e-05 | 0.275 → 0.343 | 1.52 → 1.65 | 0.582 → 0.852 | 1.4 → 1.36 | 1.63 → 1.65 | 1.05 → 1.05 | 0.000494 → 0.000616 |
| SE zero-beta s1 | 20260930-9-RrGx@1000 | 20260930-9-RrGx@5030 | 0.454 → 0.609 | 3.72e-06 → -3.17e-05 | 0.281 → 0.338 | 1.56 → 1.67 | 0.668 → 1.05 | 1.42 → 1.38 | 2.35 → 3.51 | 1.05 → 1.04 | 0.00053 → 0.000624 |
| SE zero-beta s2 | 20260930-10-H51a@1000 | 20260930-10-H51a@5004 | 0.402 → 0.534 | -9.74e-06 → -3.01e-05 | 0.283 → 0.339 | 1.56 → 1.68 | 0.636 → 0.805 | 1.4 → 1.37 | 1.96 → 2.49 | 1.05 → 1.07 | 0.00053 → 0.000624 |
| v5 | 20260628-2-a5fc@10000 | 20260805-1-0pTW@2000 | 0.229 → -300 | -0.00298 → -0.805 | 0.278 → 7.97 | 1.6 → 8.2 | 0.629 → 2.73 | 1.42 → 1.35 | 1.59 → 5.98e+03 | 1.12 → 1.96 | 0.000561 → 0.113 |
| mini2b | 20260629-4-y5u7@10000 | 20260705-1-znR7@114000 | 0.438 → -6.82 | -0.00414 → -0.217 | 0.273 → 0.897 | 1.61 → 1.87 | 0.715 → 1.94 | 1.41 → 1.14 | 1.79 → 2.05 | 1.14 → 1.09 | 0.000552 → 0.00375 |
| coxw | 20260629-6-yqMI@10000 | 20260709-1-avoB@277000 | 0.45 → -3.11 | -0.00433 → -0.202 | 0.267 → 0.555 | 1.62 → 1.35 | 0.629 → 1.48 | 1.4 → 0.794 | 2.16 → 0.871 | 1.16 → 1.38 | 0.000543 → 0.00212 |
| ykkk | 20260630-2-6y0s@40677 | 20260710-1-amlg@250803 |  | -0.0189 → -0.259 | 0.391 → 0.625 | 1.75 → 1.46 | 1.19 → 1.36 |  |  |  |  |
| nt8y | 20260701-4-CIvL@1000 | 20260708-4-kEiZ@21086 | 0.605 → -14.1 | -0.000138 → -0.228 | 0.24 → 1.1 | 1.48 → 2 | 0.309 → 2.2 | 1.39 → 0.661 | 1.93 → 0.412 | 1.01 → 0.992 | 0.000394 → 0.0035 |
| qeu8 (replay main, ends Ejp0) | 20260702-9-GLu5@1000 | 20260727-1-Ejp0@1397000 | 0.441 → -158 | -0.000142 → -0.981 | 0.239 → 3.8 | 1.48 → 4 | 0.328 → 1.33 | 1.42 → 0.292 | 2.22 → 60 | 1.01 → 1.41 | 0.000394 → 0.0434 |
| qeu8e (epoch branch) | 20260704-1-X79T@1000 | 20260708-6-sFzi@88107 | 0.476 → -7.07 | -0.000178 → -0.133 | 0.24 → 0.75 | 1.48 → 1.62 | 0.322 → 1.87 | 1.42 → 0.661 | 2.03 → 0.766 | 1.01 → 1.05 | 0.000394 → 0.00188 |
| qeu8-1blk128 | 20260711-17-pycz@1000 | 20260711-17-pycz@120000 | 0.477 → -2.18 | -0.000108 → -0.0631 | 0.232 → 0.416 | 1.46 → 1.37 | 0.297 → 1.65 | 1.4 → 0.807 | 1.86 → 0.766 | 1.01 → 1.02 | 0.000384 → 0.000797 |
| qeu8init sf100sl100 vs-UCI | 20260712-6-lTiK@220000 | 20260722-1-syxR@558000 | -3.33 → -16 | -0.0747 → -0.442 | 0.451 → 0.786 | 1.05 → 1.2 | 0.914 → 0.879 | 0.513 → 0.203 | 0.252 → 0.301 | 1.07 → 1.38 | 0.000869 → 0.00346 |
| Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy@1000 | 20261001-18-oeNy@20000 | -49 → -43.8 | -0.589 → -0.589 | 1.92 → 1.7 | 2.37 → 2.3 | 1.36 → 1.53 | 0.449 → 0.461 | 3.19 → 4.35 | 1.3 → 1.3 | 0.00985 → 0.00872 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0@2578 | 20260727-1-Ejp0-9@197340 | -148 → -71.9 | -0.924 → -0.947 | 3.52 → 1.73 | 3.74 → 1.88 | 1.3 → 0.82 | 0.294 → 0.172 | 42.5 → 5 | 1.42 → 1.42 | 0.039 → 0.0185 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-1@20241 | 20260727-1-Ejp0-67@1186322 | -148 → -139 | -0.924 → -0.924 | 3.52 → 3.28 | 3.73 → 3.48 | 1.3 → 1.23 | 0.293 → 0.26 | 39 → 33.8 | 1.42 → 1.42 | 0.0389 → 0.0362 |
| bzw3 self-play | 20260601-11-bzw3-31@467065 | 20260601-11-bzw3-31@467099 | -7.69 → -7.69 | -0.189 → -0.189 | 0.935 → 0.935 | 1.95 → 1.95 | 1.85 → 1.85 | 1.23 → 1.23 | 5.44 → 5.44 | 0.973 → 0.973 | 0.00433 → 0.00433 |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-17@494927 | 20260514-1-KbHZ-22@532369 |  | 8e-09 → -6.29e-09 | 0.215 → 0.22 | 1.4 → 1.39 | 0.469 → 0.464 |  |  |  |  |
| sMe9 self-play (fp32) | 20260525-1-sMe9-26@197269 | 20260525-1-sMe9-32@373416 |  | -7.88e-09 → -1.42e-08 | 0.254 → 0.234 | 1.37 → 1.25 | 0.631 → 0.624 |  |  |  |  |

<!-- end:growth -->

## A. Pre-block (BN + activation), every detailed checkpoint

<!-- begin:evolution_pre -->

#### SE scale+bias full-leaky fresh (never trained)

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261001-23-Dmwe @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.391 / 0.664 / 1.7 | 2.6 | 2.5 | 1.24 / 1.42 / 1.64 | 0.52 | 0 |  |

#### leaky-FC1 (SE scale+bias, FC1 leaky)

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261001-42-2q0Q @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.391 / 0.664 / 1.7 | 2.6 | 2.5 | 1.24 / 1.42 / 1.64 | 0.52 | 0 |  |
| 20261001-43-NbWz @ 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 1.04 | -0.07..+0.18 | -0.07 | 0.51 | 0.459 / 0.79 / 2.18 | 2.8 | 1.9 | 1.26 / 1.42 / 1.62 | 0.476 | 0 | 0/0 |
| 20261001-43-NbWz @ 2000 | 2000 | 0 | 0 | 0 | 0 | 0 | 1.03 | -0.11..+0.22 | -0.09 | 0.51 | 0.488 / 0.923 / 2.87 | 3.1 | 1.8 | 1.26 / 1.41 / 1.6 | 0.492 | 0 | 0/0 |
| 20261001-43-NbWz @ 3000 | 3000 | 0 | 0 | 0 | 0 | 0 | 1.03 | -0.13..+0.26 | -0.10 | 0.51 | 0.475 / 0.943 / 3.02 | 3.2 | 1.7 | 1.23 / 1.4 / 1.59 | 0.487 | 0 | 0/0 |
| 20261001-43-NbWz @ 4000 | 4000 | 0 | 0 | 0 | 0 | 0 | 1.03 | -0.13..+0.28 | -0.11 | 0.51 | 0.479 / 0.931 / 3.1 | 3.3 | 1.7 | 1.21 / 1.39 / 1.58 | 0.485 | 0 | 0/0 |
| 20261001-43-NbWz @ 5000 | 5000 | 0 | 0 | 0 | 0 | 0 | 1.04 | -0.12..+0.30 | -0.10 | 0.51 | 0.452 / 0.918 / 3.07 | 3.3 | 1.7 | 1.2 / 1.38 / 1.57 | 0.485 | 0 | 0/0 |
| 20261001-43-NbWz @ 6000 | 6000 | 0 | 0 | 0 | 0 | 0 | 1.05 | -0.12..+0.31 | -0.10 | 0.51 | 0.451 / 0.898 / 3.02 | 3.4 | 1.7 | 1.2 / 1.38 / 1.57 | 0.484 | 0 | 0/0 |
| 20261001-43-NbWz @ 7000 | 7000 | 0 | 0 | 0 | 0 | 0 | 1.05 | -0.12..+0.31 | -0.10 | 0.51 | 0.449 / 0.899 / 3.03 | 3.4 | 1.7 | 1.19 / 1.37 / 1.56 | 0.484 | 0 | 0/0 |
| 20261001-43-NbWz @ 8000 | 8000 | 0 | 0 | 0 | 0 | 0 | 1.06 | -0.12..+0.31 | -0.10 | 0.51 | 0.442 / 0.892 / 3.02 | 3.4 | 1.7 | 1.19 / 1.37 / 1.56 | 0.485 | 0 | 0/0 |
| 20261001-43-NbWz @ 9000 | 9000 | 0 | 0 | 0 | 0 | 0 | 1.06 | -0.12..+0.31 | -0.10 | 0.51 | 0.437 / 0.89 / 3.01 | 3.4 | 1.7 | 1.19 / 1.37 / 1.56 | 0.485 | 0 | 0/0 |
| 20261001-43-NbWz @ 10000 | 10000 | 0 | 0 | 0 | 0 | 0 | 1.06 | -0.12..+0.32 | -0.10 | 0.51 | 0.438 / 0.885 / 3 | 3.4 | 1.7 | 1.19 / 1.37 / 1.56 | 0.485 | 0 | 0/0 |
| 20261001-43-NbWz @ 11000 | 11000 | 0 | 0 | 0 | 0 | 0 | 1.06 | -0.12..+0.32 | -0.10 | 0.51 | 0.436 / 0.883 / 3.01 | 3.4 | 1.7 | 1.19 / 1.37 / 1.56 | 0.485 | 0 | 0/0 |
| 20261001-43-NbWz @ 12000 | 12000 | 0 | 0 | 0 | 0 | 0 | 1.06 | -0.12..+0.32 | -0.10 | 0.51 | 0.436 / 0.88 / 3.01 | 3.4 | 1.7 | 1.19 / 1.37 / 1.56 | 0.484 | 0 | 0/0 |

#### SE scale+bias s1

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260929-12-JZOe @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.391 / 0.664 / 1.7 | 2.6 | 2.5 | 1.24 / 1.42 / 1.64 | 0.52 | 0 |  |
| 20260929-22-bWdy @ 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 1.04 | -0.07..+0.18 | -0.07 | 0.51 | 0.449 / 0.795 / 2.25 | 2.8 | 1.9 | 1.26 / 1.42 / 1.62 | 0.482 | 0 |  |
| 20260929-22-bWdy @ 5000 | 5000 | 0 | 0 | 0 | 0 | 0 | 1.05 | -0.10..+0.30 | -0.09 | 0.51 | 0.432 / 0.895 / 3.11 | 3.5 | 1.6 | 1.2 / 1.38 / 1.57 | 0.486 | 0 |  |
| 20260929-22-bWdy @ 10000 | 10000 | 0 | 0 | 0 | 0 | 0 | 1.06 | -0.10..+0.32 | -0.09 | 0.51 | 0.422 / 0.855 / 3.08 | 3.6 | 1.6 | 1.19 / 1.37 / 1.56 | 0.492 | 0 |  |
| 20260929-22-bWdy @ 11000 | 11000 | 0 | 0 | 0 | 0 | 0 | 1.06 | -0.10..+0.32 | -0.09 | 0.51 | 0.42 / 0.859 / 3.08 | 3.6 | 1.6 | 1.19 / 1.37 / 1.56 | 0.492 | 0 |  |
| 20260929-22-bWdy @ 12000 | 12000 | 0 | 0 | 0 | 0 | 0 | 1.06 | -0.10..+0.32 | -0.09 | 0.51 | 0.42 / 0.854 / 3.06 | 3.6 | 1.6 | 1.19 / 1.37 / 1.56 | 0.492 | 0 |  |
| 20260929-22-bWdy @ 20000 | 20000 | 0 | 0 | 0 | 0 | 0 | 1.04 | -0.13..+0.35 | -0.11 | 0.51 | 0.418 / 0.848 / 2.88 | 3.4 | 1.6 | 1.13 / 1.34 / 1.53 | 0.484 | 0 |  |
| 20260929-22-bWdy @ 33014 | 33014 | 0 | 0 | 0 | 0 | 0 | 1.03 | -0.17..+0.43 | -0.15 | 0.51 | 0.389 / 0.777 / 2.94 | 3.8 | 2.1 | 1.03 / 1.28 / 1.51 | 0.498 | 0 |  |

#### SE scale+bias s2

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260930-1-H1Oq @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.35 / 0.615 / 1.53 | 2.5 | 2.0 | 1.25 / 1.4 / 1.69 | 0.504 | 0 |  |
| 20260930-4-k98x @ 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 1.04 | -0.05..+0.13 | -0.05 | 0.51 | 0.447 / 0.739 / 1.99 | 2.7 | 1.6 | 1.25 / 1.41 / 1.68 | 0.501 | 0 | 0/0 |
| 20260930-4-k98x @ 5000 | 5000 | 0 | 0 | 0 | 0 | 0 | 1.06 | -0.09..+0.22 | -0.08 | 0.51 | 0.461 / 0.877 / 2.5 | 2.9 | 1.6 | 1.16 / 1.37 / 1.61 | 0.556 | 0 | 0/0 |
| 20260930-4-k98x @ 7282 | 7282 | 0 | 0 | 0 | 0 | 0 | 1.08 | -0.09..+0.23 | -0.08 | 0.51 | 0.429 / 0.851 / 2.47 | 2.9 | 1.5 | 1.15 / 1.36 / 1.61 | 0.566 | 0 | 0/0 |

#### SE attenuate-only s1

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260929-13-06yp @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.357 / 0.6 / 1.11 | 1.9 | 2.0 | 1.23 / 1.4 / 1.69 | 0.531 | 0 |  |
| 20260929-23-L6Qm @ 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 1.05 | -0.03..+0.18 | -0.03 | 0.51 | 0.367 / 0.742 / 2.12 | 2.9 | 1.6 | 1.24 / 1.4 / 1.68 | 0.516 | 0 |  |
| 20260929-23-L6Qm @ 5000 | 5000 | 0 | 0 | 0 | 0 | 0 | 1.06 | -0.08..+0.29 | -0.07 | 0.51 | 0.414 / 0.844 / 2.83 | 3.4 | 1.6 | 1.19 / 1.37 / 1.62 | 0.555 | 0 |  |
| 20260929-23-L6Qm @ 10000 | 10000 | 0 | 0 | 0 | 0 | 0 | 1.07 | -0.08..+0.30 | -0.07 | 0.51 | 0.396 / 0.828 / 2.81 | 3.4 | 1.6 | 1.19 / 1.37 / 1.61 | 0.555 | 0 |  |
| 20260929-23-L6Qm @ 11000 | 11000 | 0 | 0 | 0 | 0 | 0 | 1.07 | -0.08..+0.30 | -0.07 | 0.51 | 0.398 / 0.828 / 2.81 | 3.4 | 1.6 | 1.19 / 1.37 / 1.61 | 0.555 | 0 |  |
| 20260929-23-L6Qm @ 12000 | 12000 | 0 | 0 | 0 | 0 | 0 | 1.07 | -0.08..+0.30 | -0.07 | 0.51 | 0.4 / 0.83 / 2.81 | 3.4 | 1.6 | 1.19 / 1.36 / 1.61 | 0.559 | 0 |  |
| 20260929-23-L6Qm @ 20000 | 20000 | 0 | 0 | 0 | 0 | 0 | 1.06 | -0.10..+0.31 | -0.08 | 0.51 | 0.424 / 0.809 / 2.72 | 3.4 | 1.7 | 1.14 / 1.33 / 1.56 | 0.566 | 0 |  |
| 20260929-23-L6Qm @ 33012 | 33012 | 0 | 0 | 0 | 0 | 0 | 1.06 | -0.13..+0.35 | -0.11 | 0.51 | 0.406 / 0.705 / 2.7 | 3.8 | 2.5 | 1.05 / 1.27 / 1.53 | 0.574 | 0 |  |

#### SE attenuate-only s2

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260930-2-Gf9P @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.352 / 0.637 / 1.42 | 2.2 | 1.9 | 1.16 / 1.41 / 1.63 | 0.543 | 0 |  |
| 20260930-5-5TXu @ 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 1.04 | -0.06..+0.14 | -0.05 | 0.51 | 0.395 / 0.769 / 1.71 | 2.2 | 1.9 | 1.18 / 1.42 / 1.62 | 0.563 | 0 | 0/0 |
| 20260930-5-5TXu @ 5000 | 5000 | 0 | 0 | 0 | 0 | 0 | 1.06 | -0.14..+0.19 | -0.11 | 0.51 | 0.437 / 0.833 / 2.52 | 3.0 | 1.8 | 1.12 / 1.38 / 1.57 | 0.706 | 0 | 0/0 |
| 20260930-5-5TXu @ 7289 | 7289 | 0 | 0 | 0 | 0 | 0 | 1.07 | -0.13..+0.20 | -0.11 | 0.51 | 0.429 / 0.818 / 2.54 | 3.1 | 1.9 | 1.12 / 1.37 / 1.56 | 0.702 | 0 | 0/0 |

#### SE none s1

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260929-18-D9is @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.314 / 0.641 / 1.42 | 2.2 | 3.1 | 1.21 / 1.4 / 1.67 | 0.527 | 0 |  |
| 20260929-24-834D @ 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 1.05 | -0.05..+0.17 | -0.04 | 0.51 | 0.504 / 0.764 / 2.27 | 3.0 | 1.6 | 1.22 / 1.41 / 1.66 | 0.535 | 0 |  |
| 20260929-24-834D @ 5000 | 5000 | 0 | 0 | 0 | 0 | 0 | 1.07 | -0.12..+0.21 | -0.10 | 0.51 | 0.508 / 0.793 / 2.33 | 2.9 | 2.4 | 1.13 / 1.37 / 1.64 | 0.621 | 0 |  |
| 20260929-24-834D @ 10000 | 10000 | 0 | 0 | 0 | 0 | 0 | 1.09 | -0.12..+0.22 | -0.10 | 0.51 | 0.467 / 0.773 / 2.23 | 2.9 | 2.5 | 1.12 / 1.36 / 1.63 | 0.633 | 0 |  |
| 20260929-24-834D @ 11000 | 11000 | 0 | 0 | 0 | 0 | 0 | 1.09 | -0.12..+0.22 | -0.10 | 0.51 | 0.465 / 0.775 / 2.23 | 2.9 | 2.5 | 1.12 / 1.36 / 1.63 | 0.637 | 0 |  |
| 20260929-24-834D @ 12000 | 12000 | 0 | 0 | 0 | 0 | 0 | 1.09 | -0.12..+0.22 | -0.10 | 0.51 | 0.461 / 0.773 / 2.22 | 2.9 | 2.6 | 1.12 / 1.36 / 1.62 | 0.637 | 0 |  |
| 20260929-24-834D @ 20000 | 20000 | 0 | 0 | 0 | 0 | 0 | 1.08 | -0.14..+0.24 | -0.12 | 0.51 | 0.383 / 0.75 / 2.25 | 3.0 | 2.5 | 1.07 / 1.33 / 1.62 | 0.688 | 0 |  |
| 20260929-24-834D @ 32036 | 32036 | 0 | 0 | 0 | 0 | 0 | 1.09 | -0.18..+0.26 | -0.16 | 0.51 | 0.262 / 0.668 / 2.19 | 3.3 | 2.9 | 0.973 / 1.27 / 1.68 | 0.707 | 0 |  |

#### SE none s2

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260930-3-V9zk @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.357 / 0.648 / 1.15 | 1.8 | 2.0 | 1.18 / 1.4 / 1.61 | 0.598 | 0 |  |
| 20260930-6-LkS6 @ 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 1.05 | -0.07..+0.12 | -0.07 | 0.51 | 0.417 / 0.758 / 1.63 | 2.2 | 1.8 | 1.19 / 1.4 / 1.6 | 0.536 | 0 | 0/0 |
| 20260930-6-LkS6 @ 5000 | 5000 | 0 | 0 | 0 | 0 | 0 | 1.04 | -0.09..+0.20 | -0.09 | 0.51 | 0.336 / 0.782 / 1.65 | 2.1 | 2.1 | 1.14 / 1.37 / 1.54 | 0.595 | 0 | 0/0 |
| 20260930-6-LkS6 @ 7019 | 7019 | 0 | 0 | 0 | 0 | 0 | 1.05 | -0.09..+0.21 | -0.09 | 0.51 | 0.324 / 0.764 / 1.65 | 2.2 | 2.1 | 1.13 / 1.36 / 1.53 | 0.605 | 0 | 0/0 |

#### SE zero-beta s1

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260930-7-crxN @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.391 / 0.664 / 1.7 | 2.6 | 2.5 | 1.24 / 1.42 / 1.64 | 0.52 | 0 |  |
| 20260930-9-RrGx @ 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 1.05 | -0.05..+0.19 | -0.06 | 0.51 | 0.463 / 0.785 / 2.35 | 3.0 | 2.0 | 1.26 / 1.42 / 1.62 | 0.517 | 0 | 0/0 |
| 20260930-9-RrGx @ 5000 | 5000 | 0 | 0 | 0 | 0 | 0 | 1.04 | -0.11..+0.30 | -0.09 | 0.51 | 0.431 / 0.904 / 3.51 | 3.9 | 1.9 | 1.21 / 1.38 / 1.57 | 0.516 | 0 | 0/0 |
| 20260930-9-RrGx @ 5030 | 5030 | 0 | 0 | 0 | 0 | 0 | 1.04 | -0.11..+0.30 | -0.09 | 0.51 | 0.43 / 0.905 / 3.51 | 3.9 | 1.9 | 1.21 / 1.38 / 1.57 | 0.517 | 0 | 0/0 |

#### SE zero-beta s2

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260930-8-8qyR @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.35 / 0.615 / 1.53 | 2.5 | 2.0 | 1.25 / 1.4 / 1.69 | 0.504 | 0 |  |
| 20260930-10-H51a @ 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 1.05 | -0.04..+0.13 | -0.04 | 0.51 | 0.417 / 0.702 / 1.96 | 2.8 | 1.7 | 1.25 / 1.4 / 1.67 | 0.507 | 0 | 0/0 |
| 20260930-10-H51a @ 5000 | 5000 | 0 | 0 | 0 | 0 | 0 | 1.07 | -0.07..+0.22 | -0.07 | 0.51 | 0.429 / 0.834 / 2.49 | 3.0 | 1.6 | 1.16 / 1.37 / 1.6 | 0.562 | 0 | 0/0 |
| 20260930-10-H51a @ 5004 | 5004 | 0 | 0 | 0 | 0 | 0 | 1.07 | -0.07..+0.22 | -0.07 | 0.51 | 0.428 / 0.834 / 2.49 | 3.0 | 1.6 | 1.16 / 1.37 / 1.6 | 0.562 | 0 | 0/0 |

#### v5

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260628-1-tWtk @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.398 / 0.656 / 1.18 | 1.8 | 2.0 | 1.19 / 1.42 / 1.69 | 0.523 | 0 |  |
| 20260628-2-a5fc @ 10000 | 10000 | 0 | 0 | 0 | 0 | 0 | 1.12 | -0.07..+0.19 | -0.05 | 0.52 | 0.4 / 0.822 / 1.59 | 1.9 | 1.5 | 1.2 / 1.42 / 1.68 | 0.508 | 0 |  |
| 20260628-2-a5fc @ 45441 | 45441 | 0 | 0 | 0 | 0 | 0 | 1.16 | -0.16..+0.25 | -0.11 | 0.52 | 0.449 / 1.03 / 2.12 | 2.1 | 1.6 | 1.19 / 1.42 / 1.65 | 0.494 | 0 |  |
| 20260628-9-OdUt @ 15460 | 60901 | 0 | 0 | 0 | 0 | 0 | 1.16 | -0.18..+0.26 | -0.13 | 0.52 | 0.439 / 0.961 / 1.95 | 2.0 | 1.7 | 1.12 / 1.33 / 1.54 | 0.488 | 0 |  |
| 20260629-1-Uf4p @ 39419 | 100320 | 0 | 0 | 0 | 0 | 0 | 1.16 | -0.29..+0.32 | -0.21 | 0.52 | 0.406 / 0.994 / 2.34 | 2.4 | 1.9 | 1.02 / 1.28 / 1.5 | 0.535 | 0 |  |
| 20260703-1-Dg5v @ 268506 | 368826 | 0 | 0 | 0 | 0 | 0 | 1.21 | -0.86..+3.66 | -0.80 | 0.58 | 0.143 / 0.451 / 1.82 | 4.0 | 3.0 | 0.624 / 1.12 / 1.5 | 1.16 | 0 |  |
| 20260714-1-h7vI @ 115000 | 483826 | 0 | 0 | 0 | 0 | 0 | 1.36 | -0.98..+5.56 | -1.07 | 0.66 | 0.112 / 0.453 / 2.7 | 6.0 | 4.3 | 0.473 / 1.07 / 1.49 | 1.09 | 0 |  |
| 20260714-1-h7vI @ 336610 | 705436 | 0 | 0 | 6 | 0 | 0 | 1.81 | -1.12..+10.12 | -1.15 | 0.81 | 0.129 / 1 / 340 | 338.7 | 4.3 | 0.314 / 1.16 / 3.29 | 2.84 | 2 |  |
| 20260729-1-VZ2j @ 106333 | 811769 | 0 | 0 | 10 | 0 | 0 | 1.96 | -1.14..+10.38 | -1.15 | 0.83 | 0.204 / 2.8 / 2.98e+03 | 1061.1 | 4.2 | 0.262 / 1.29 / 4.63 | 4.22 | 6 |  |
| 20260802-2-Xuub @ 49374 | 861143 | 0 | 0 | 11 | 0 | 0 | 1.95 | -1.15..+10.19 | -1.18 | 0.83 | 0.245 / 4.39 / 5.98e+03 | 1362.9 | 3.8 | 0.255 / 1.35 / 5.33 | 4.94 | 9 |  |
| 20260805-1-0pTW @ 2000 | 859769 | 0 | 0 | 11 | 0 | 0 | 1.96 | -1.15..+10.25 | -1.19 | 0.83 | 0.244 / 4.36 / 5.98e+03 | 1372.7 | 3.6 | 0.255 / 1.35 / 5.33 | 4.94 | 9 |  |

#### mini2b

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260629-3-3MIV @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.371 / 0.629 / 1.73 | 2.8 | 2.3 | 1.13 / 1.42 / 1.59 | 0.52 | 0 |  |
| 20260629-4-y5u7 @ 10000 | 10000 | 0 | 0 | 0 | 0 | 0 | 1.14 | -0.05..+0.16 | -0.05 | 0.52 | 0.428 / 0.793 / 1.79 | 2.3 | 2.1 | 1.15 / 1.41 / 1.58 | 0.531 | 0 |  |
| 20260629-4-y5u7 @ 13464 | 13464 | 0 | 0 | 0 | 0 | 0 | 1.16 | -0.06..+0.17 | -0.06 | 0.52 | 0.443 / 0.83 / 1.8 | 2.2 | 2.0 | 1.16 / 1.4 / 1.58 | 0.555 | 0 |  |
| 20260630-5-BEKK @ 120695 | 134159 | 0 | 0 | 0 | 0 | 0 | 1.12 | -0.38..+0.53 | -0.32 | 0.50 | 0.285 / 0.922 / 1.97 | 2.1 | 2.6 | 0.946 / 1.26 / 1.5 | 0.73 | 0 |  |
| 20260701-2-SvRu @ 8000 | 142159 | 0 | 0 | 0 | 0 | 0 | 1.12 | -0.40..+0.57 | -0.34 | 0.50 | 0.273 / 0.918 / 2.02 | 2.2 | 2.6 | 0.928 / 1.25 / 1.5 | 0.73 | 0 |  |
| 20260705-1-znR7 @ 53000 | 195159 | 0 | 0 | 0 | 0 | 0 | 1.11 | -0.50..+1.11 | -0.44 | 0.50 | 0.252 / 0.904 / 2.45 | 2.7 | 2.5 | 0.814 / 1.19 / 1.5 | 0.738 | 0 |  |
| 20260705-1-znR7 @ 114000 | 256159 | 0 | 0 | 0 | 0 | 0 | 1.09 | -0.61..+1.95 | -0.57 | 0.50 | 0.217 / 0.895 / 2.05 | 2.3 | 2.5 | 0.701 / 1.14 / 1.46 | 0.738 | 0 |  |

#### coxw

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260629-5-Coxw @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.387 / 0.652 / 1.63 | 2.5 | 2.0 | 1.22 / 1.42 / 1.62 | 0.508 | 0 |  |
| 20260629-6-yqMI @ 10000 | 10000 | 0 | 0 | 0 | 0 | 0 | 1.16 | -0.05..+0.20 | -0.05 | 0.52 | 0.453 / 0.811 / 2.16 | 2.7 | 1.6 | 1.21 / 1.4 / 1.58 | 0.508 | 0 |  |
| 20260629-6-yqMI @ 55550 | 55550 | 0 | 0 | 0 | 0 | 0 | 1.21 | -0.14..+0.26 | -0.11 | 0.51 | 0.41 / 0.814 / 2.31 | 2.8 | 2.2 | 1.13 / 1.36 / 1.58 | 0.648 | 0 |  |
| 20260709-1-avoB @ 136000 | 191550 | 0 | 0 | 0 | 0 | 0 | 1.29 | -0.54..+0.60 | -0.43 | 0.51 | 0.0728 / 0.288 / 1.05 | 3.6 | 4.5 | 0.588 / 0.915 / 1.31 | 0.852 | 0 |  |
| 20260709-1-avoB @ 277000 | 332550 | 0 | 0 | 0 | 0 | 0 | 1.38 | -0.85..+1.48 | -0.75 | 0.51 | 0.0364 / 0.164 / 0.871 | 5.3 | 3.3 | 0.292 / 0.794 / 1.18 | 0.715 | 0 |  |

#### nt8y

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260701-3-nT8Y @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.185 / 0.627 / 1.63 | 2.6 | 2.4 | 0.956 / 1.39 / 1.9 | 1.03 | 0 |  |
| 20260701-4-CIvL @ 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 1.01 | -0.02..+0.04 | -0.02 | 0.50 | 0.178 / 0.617 / 1.93 | 3.1 | 2.9 | 0.954 / 1.39 / 1.9 | 1.03 | 0 |  |
| 20260701-4-CIvL @ 65883 | 65883 | 0 | 0 | 0 | 0 | 0 | 1 | -0.20..+0.46 | -0.20 | 0.50 | 0.137 / 0.527 / 1.8 | 3.4 | 5.3 | 0.834 / 1.2 / 1.63 | 0.914 | 0 |  |
| 20260701-5-bOYQ @ 70000 | 135883 | 0 | 0 | 0 | 0 | 0 | 0.984 | -0.39..+0.77 | -0.40 | 0.50 | 0.0869 / 0.354 / 0.973 | 2.8 | 7.2 | 0.716 / 1.02 / 1.38 | 0.797 | 0 |  |
| 20260701-5-bOYQ @ 70779 | 136662 | 0 | 0 | 0 | 0 | 0 | 0.984 | -0.39..+0.77 | -0.40 | 0.50 | 0.0845 / 0.354 / 1.01 | 2.9 | 7.4 | 0.715 / 1.02 / 1.38 | 0.797 | 0 |  |
| 20260706-2-3CZF @ 15000 | 151662 | 0 | 0 | 0 | 0 | 0 | 0.982 | -0.42..+0.82 | -0.45 | 0.50 | 0.0825 / 0.324 / 0.93 | 2.9 | 7.7 | 0.693 / 0.986 / 1.33 | 0.773 | 0 |  |
| 20260707-1-cslu @ 140000 | 291662 | 0 | 0 | 0 | 0 | 0 | 0.984 | -0.62..+1.24 | -1.07 | 0.53 | 0.0393 / 0.151 / 0.447 | 3.0 | 9.5 | 0.511 / 0.73 / 1.2 | 0.688 | 0 |  |
| 20260708-4-kEiZ @ 21086 | 312748 | 0 | 0 | 0 | 0 | 0 | 0.992 | -0.64..+1.29 | -1.08 | 0.53 | 0.0304 / 0.127 / 0.412 | 3.2 | 10.6 | 0.464 / 0.661 / 1.1 | 0.629 | 0 |  |

#### qeu8 (replay main, ends Ejp0)

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260702-7-Qeu8 @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.299 / 0.623 / 1.9 | 3.0 | 2.4 | 1.03 / 1.42 / 1.89 | 0.703 | 0 |  |
| 20260702-9-GLu5 @ 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 1.01 | -0.01..+0.03 | -0.01 | 0.50 | 0.21 / 0.623 / 2.22 | 3.6 | 1.9 | 1.03 / 1.42 / 1.88 | 0.707 | 0 |  |
| 20260702-9-GLu5 @ 41000 | 41000 | 0 | 0 | 0 | 0 | 0 | 1.02 | -0.08..+0.11 | -0.08 | 0.50 | 0.199 / 0.689 / 3.23 | 4.7 | 2.5 | 0.975 / 1.29 / 1.71 | 0.641 | 0 |  |
| 20260703-1-Lnji @ 67508 | 108915 | 0 | 0 | 0 | 0 | 0 | 1.02 | -0.23..+0.40 | -0.21 | 0.51 | 0.142 / 0.489 / 2.14 | 4.4 | 3.3 | 0.849 / 1.11 / 1.44 | 0.609 | 0 |  |
| 20260706-1-PVZp @ 67000 | 175915 | 0 | 0 | 0 | 0 | 0 | 1.04 | -0.36..+0.62 | -0.39 | 0.52 | 0.108 / 0.347 / 1.48 | 4.3 | 3.9 | 0.72 / 0.964 / 1.23 | 0.68 | 0 |  |
| 20260727-1-Ejp0 @ 611000 | 786915 | 0 | 0 | 0 | 0 | 0 | 1.3 | -0.75..+2.97 | -0.69 | 0.65 | 0.00178 / 0.0181 / 1.03 | 56.9 | 14.1 | 0.0727 / 0.29 / 0.882 | 0.633 | 0 |  |
| 20260727-1-Ejp0 @ 1397000 | 1572915 | 0 | 0 | 20 | 0 | 0 | 1.41 | -1.24..+9.25 | -1.07 | 0.74 | 0.00264 / 0.0492 / 60 | 1219.7 | 17.7 | 0.052 / 0.292 / 1.58 | 1.55 | 0 |  |

#### qeu8e (epoch branch)

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260702-7-Qeu8 @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.299 / 0.623 / 1.9 | 3.0 | 2.4 | 1.03 / 1.42 / 1.89 | 0.703 | 0 |  |
| 20260704-1-X79T @ 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 1.01 | -0.02..+0.03 | -0.02 | 0.50 | 0.217 / 0.648 / 2.03 | 3.1 | 1.9 | 1.03 / 1.42 / 1.88 | 0.707 | 0 |  |
| 20260704-1-X79T @ 21224 | 21224 | 0 | 0 | 0 | 0 | 0 | 1.01 | -0.04..+0.08 | -0.04 | 0.50 | 0.334 / 0.773 / 3.25 | 4.2 | 2.2 | 0.993 / 1.35 / 1.79 | 0.668 | 0 |  |
| 20260704-2-jSjr @ 42507 | 63731 | 0 | 0 | 0 | 0 | 0 | 1.01 | -0.12..+0.27 | -0.14 | 0.50 | 0.281 / 0.695 / 2.92 | 4.2 | 2.8 | 0.928 / 1.23 / 1.61 | 0.641 | 0 |  |
| 20260704-3-h7Pp @ 42507 | 106238 | 0 | 0 | 0 | 0 | 0 | 1.01 | -0.21..+0.47 | -0.23 | 0.51 | 0.26 / 0.633 / 2.73 | 4.3 | 2.7 | 0.86 / 1.12 / 1.45 | 0.637 | 0 |  |
| 20260708-5-0YQL @ 5000 | 111238 | 0 | 0 | 0 | 0 | 0 | 1.01 | -0.22..+0.49 | -0.23 | 0.51 | 0.249 / 0.594 / 2.58 | 4.3 | 2.7 | 0.84 / 1.09 / 1.42 | 0.633 | 0 |  |
| 20260708-5-0YQL @ 26492 | 132730 | 0 | 0 | 0 | 0 | 0 | 1.02 | -0.25..+0.56 | -0.25 | 0.51 | 0.2 / 0.486 / 1.89 | 3.9 | 2.9 | 0.756 / 0.986 / 1.27 | 0.613 | 0 |  |
| 20260708-6-sFzi @ 88107 | 220837 | 0 | 0 | 0 | 0 | 0 | 1.05 | -0.37..+0.86 | -0.36 | 0.52 | 0.0913 / 0.198 / 0.766 | 3.9 | 3.6 | 0.496 / 0.661 / 0.952 | 0.498 | 0 |  |

#### qeu8-1blk128

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260711-16-VRR4 @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.334 / 0.633 / 1.86 | 2.9 | 2.5 | 1.15 / 1.41 / 1.65 | 0.609 | 0 |  |
| 20260711-17-pycz @ 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 1.01 | -0.02..+0.04 | -0.02 | 0.50 | 0.247 / 0.555 / 1.86 | 3.4 | 2.2 | 1.15 / 1.4 / 1.65 | 0.602 | 0 |  |
| 20260711-17-pycz @ 61000 | 61000 | 0 | 0 | 0 | 0 | 0 | 1.02 | -0.16..+0.15 | -0.12 | 0.50 | 0.216 / 0.443 / 1.31 | 3.0 | 2.3 | 0.871 / 1.05 / 1.23 | 0.484 | 0 |  |
| 20260711-17-pycz @ 120000 | 120000 | 0 | 0 | 0 | 0 | 0 | 1.02 | -0.24..+0.32 | -0.19 | 0.50 | 0.123 / 0.268 / 0.766 | 2.9 | 2.2 | 0.665 / 0.807 / 1.02 | 0.457 | 0 |  |

#### qeu8init sf100sl100 vs-UCI

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260702-7-Qeu8 @ fresh |  | 0 | 0 | 0 | 0 | 0 | 1 | +0.00..+0.00 | +0.00 | 0.50 | 0.299 / 0.623 / 1.9 | 3.0 | 2.4 | 1.03 / 1.42 / 1.89 | 0.703 | 0 |  |
| 20260712-6-lTiK @ 220000 |  | 0 | 0 | 0 | 0 | 0 | 1.07 | -0.11..+0.26 | -0.11 | 0.52 | 0.0366 / 0.092 / 0.252 | 2.7 | 3.6 | 0.388 / 0.513 / 0.744 | 0.33 | 0 |  |
| 20260714-1-NYAZ @ 758000 |  | 0 | 0 | 0 | 0 | 0 | 1.29 | -0.33..+1.62 | -0.36 | 0.57 | 0.00333 / 0.0174 / 0.293 | 16.8 | 3.6 | 0.107 / 0.218 / 0.762 | 0.516 | 0 |  |
| 20260722-1-syxR @ 558000 |  | 0 | 0 | 0 | 0 | 0 | 1.38 | -0.36..+2.14 | -0.36 | 0.59 | 0.00255 / 0.0117 / 0.301 | 25.8 | 3.4 | 0.106 / 0.203 / 0.684 | 0.512 | 0 |  |

#### Ejp0 headfix-phase2 (from Ejp0 @681k)

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261001-18-oeNy @ 1000 |  | 0 | 0 | 0 | 0 | 0 | 1.3 | -0.76..+3.05 | -0.73 | 0.65 | 0.0054 / 0.132 / 3.19 | 24.2 | 6.2 | 0.228 / 0.449 / 1.89 | 1.35 | 0 | 0/0 |
| 20261001-18-oeNy @ 11000 |  | 0 | 0 | 0 | 0 | 0 | 1.3 | -0.77..+3.08 | -0.76 | 0.65 | 0.0114 / 0.0883 / 4.44 | 50.2 | 8.8 | 0.248 / 0.47 / 1.73 | 1.23 | 1 | 0/0 |
| 20261001-18-oeNy @ 20000 |  | 0 | 0 | 0 | 0 | 0 | 1.3 | -0.77..+3.08 | -0.75 | 0.65 | 0.0118 / 0.086 / 4.35 | 50.6 | 9.3 | 0.244 / 0.461 / 1.65 | 1.17 | 2 | 0/0 |

#### Ejp0 self-play run 1

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260727-1-Ejp0-1 @ 2578 [trainer] |  | 0 | 0 | 17 | 0 | 0 | 1.42 | -1.16..+9.07 | -0.99 | 0.74 | 0.00274 / 0.0487 / 41.2 | 846.7 | 15.2 | 0.0597 / 0.294 / 1.47 | 1.43 | 0 | 0/0 |
| 20260727-1-Ejp0-4 @ 104903 |  | 0 | 0 | 20 | 0 | 0 | 1.42 | -1.29..+8.69 | -0.99 | 0.74 | 0.0011 / 0.0126 / 5.53 | 437.8 | 13.8 | 0.0656 / 0.183 / 0.739 | 0.727 | 0 |  |
| 20260727-1-Ejp0-10 @ 197340 [trainer] |  | 0 | 0 | 20 | 0 | 0 | 1.42 | -1.29..+8.70 | -0.98 | 0.74 | 0.000859 / 0.00859 / 4.48 | 521.6 | 15.2 | 0.0588 / 0.164 / 0.663 | 0.651 | 0 | 0/0 |
| 20260727-1-Ejp0-9 @ 197340 [champion] |  | 0 | 0 | 20 | 0 | 0 | 1.42 | -1.29..+8.69 | -0.98 | 0.74 | 0.00095 / 0.00995 / 5 | 502.6 | 14.7 | 0.0613 / 0.172 / 0.692 | 0.68 | 0 |  |

#### Ejp0 self-play run 2

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260727-1-Ejp0-1 @ 20241 |  | 0 | 0 | 15 | 0 | 0 | 1.42 | -1.15..+9.06 | -0.95 | 0.74 | 0.00244 / 0.0397 / 39 | 983.0 | 15.2 | 0.0593 / 0.293 / 1.47 | 1.43 | 0 |  |
| 20260727-1-Ejp0-39 @ 935524 |  | 0 | 0 | 16 | 0 | 0 | 1.42 | -1.16..+9.06 | -0.92 | 0.74 | 0.00184 / 0.0212 / 32.8 | 1546.3 | 15.7 | 0.0542 / 0.265 / 1.43 | 1.4 | 0 |  |
| 20260727-1-Ejp0-68 @ 1186322 [trainer] |  | 0 | 0 | 16 | 0 | 0 | 1.42 | -1.16..+9.06 | -0.92 | 0.74 | 0.00153 / 0.0199 / 33.8 | 1696.2 | 15.9 | 0.0529 / 0.26 / 1.43 | 1.4 | 0 | 0/0 |
| 20260727-1-Ejp0-67 @ 1186322 [champion] |  | 0 | 0 | 16 | 0 | 0 | 1.42 | -1.16..+9.06 | -0.92 | 0.74 | 0.00153 / 0.0199 / 33.8 | 1696.2 | 15.9 | 0.0529 / 0.26 / 1.43 | 1.4 | 0 |  |

#### bzw3 self-play

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260601-11-bzw3-31 @ 467065 |  | 0 | 0 | 0 | 0 | 0 | 0.973 | -0.55..+2.00 | -0.58 | 0.55 | 0.316 / 1.32 / 5.44 | 4.1 | 1.3 | 0.954 / 1.23 / 1.61 | 1.01 | 0 |  |
| 20260601-11-bzw3-32 @ 467099 [trainer] |  | 0 | 0 | 0 | 0 | 0 | 0.64 | -0.67..+2.87 | -0.99 | 0.60 | 0.452 / 1.41 / 7.4 | 5.2 | 2.2 | 0.801 / 1.3 / 1.91 | 1.18 | 0 | 0/0 |
| 20260601-11-bzw3-31 @ 467099 [champion] |  | 0 | 0 | 0 | 0 | 0 | 0.973 | -0.55..+2.00 | -0.58 | 0.55 | 0.316 / 1.32 / 5.44 | 4.1 | 1.3 | 0.954 / 1.23 / 1.61 | 1.01 | 0 |  |

#### LWKa self-play (v4 12-block)

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260531-9-LWKa-11 @ 106695 [trainer] |  | 0 | 0 | 0 | 0 | 0 | 1.01 | -0.13..+0.40 | -0.12 | 0.52 | 0.446 / 1.28 / 2.74 | 2.1 | 1.2 | 1.15 / 1.37 / 1.58 | 0.603 | 0 | 0/0 |
| 20260531-9-LWKa-10 @ 106695 [champion] |  | 0 | 0 | 0 | 0 | 0 | 1.04 | -0.10..+0.31 | -0.09 | 0.52 | 0.5 / 1.2 / 2.61 | 2.2 | 1.0 | 1.16 / 1.38 / 1.55 | 0.582 | 0 |  |

#### LMGh self-play

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260609-12-LMGh-4 @ 79135 [trainer] |  | 0 | 0 | 0 | 0 | 0 | 1.42 | -2.08..+4.19 | -1.50 | 0.80 | 1.52 / 4.64 / 22.9 | 4.9 | 4.2 | 1.56 / 2.05 / 3.07 | 2.48 | 0 | 0/0 |
| 20260609-12-LMGh-3 @ 79135 [champion] |  | 0 | 0 | 0 | 0 | 0 | 1.42 | -2.08..+4.19 | -1.50 | 0.80 | 1.52 / 4.64 / 22.9 | 4.9 | 4.2 | 1.56 / 2.05 / 3.07 | 2.48 | 0 |  |

#### WjRY self-play

| checkpoint | cum | dead | mostly-off | always-on | flat | gamma<0 | median abs(gamma) | beta range | min beta/abs(gamma) | median P(on) | running var min / median / max | rv max/median | max abs(mu)/sigma | pre_conv row norm min / median / max | max abs pre_conv | weak tower cols | zero-velocity gamma+beta / pre rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260609-14-WjRY-8 @ 98974 [trainer] |  | 0 | 0 | 0 | 0 | 0 | 1.09 | -0.60..+1.63 | -0.75 | 0.60 | 0.406 / 1.1 / 7.15 | 6.5 | 2.3 | 1.19 / 1.41 / 1.69 | 0.751 | 0 | 0/0 |
| 20260609-14-WjRY-7 @ 98974 [champion] |  | 0 | 0 | 0 | 0 | 0 | 1.02 | -0.10..+0.29 | -0.10 | 0.51 | 0.535 / 1.21 / 4 | 3.3 | 1.2 | 1.22 / 1.39 / 1.58 | 0.582 | 0 |  |

<!-- end:evolution_pre -->

## B. Final projection, every detailed checkpoint

<!-- begin:evolution_final -->

#### SE scale+bias full-leaky fresh (never trained)

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261001-23-Dmwe @ fresh |  | 1.18 / 1.41 / 1.71 | 0.148 | 0.10 | 1.4 | +0.000 / 0.000 | +0.00..+0.00 | -0.1 | 0.0002 | 0.98 | 1.03 | 1.00 |  /  | 0.48 | 0 |  |

#### leaky-FC1 (SE scale+bias, FC1 leaky)

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261001-42-2q0Q @ fresh |  | 1.18 / 1.41 / 1.71 | 0.148 | 0.10 | 1.4 | +0.000 / 0.000 | +0.00..+0.00 | -0.1 | 0.0002 | 0.98 | 1.03 | 1.00 |  /  | 0.48 | 0 |  |
| 20261001-43-NbWz @ 1000 | 1000 | 1.28 / 1.55 / 1.8 | 0.279 | 0.18 | 1.53 | +0.000 / 0.051 | -0.06..+0.31 | +0.4 | 0.0005 | 0.87 | 1.00 | 0.94 | 0.047 / 0.408 | 0.63 | 0 | 0.000195 / 0.0157 |
| 20261001-43-NbWz @ 2000 | 2000 | 1.24 / 1.62 / 1.92 | 0.319 | 0.20 | 1.58 | -0.000 / 0.071 | -0.06..+0.48 | +0.5 | 0.0006 | 0.83 | 0.98 | 0.93 | 0.075 / 0.542 | 0.74 | 0 | 6.73e-05 / 0.0197 |
| 20261001-43-NbWz @ 3000 | 3000 | 1.23 / 1.65 / 2.05 | 0.328 | 0.20 | 1.6 | -0.000 / 0.080 | -0.07..+0.55 | +0.5 | 0.0006 | 0.80 | 0.97 | 0.92 | 0.090 / 0.599 | 0.80 | 0 | 0.000154 / 0.019 |
| 20261001-43-NbWz @ 4000 | 4000 | 1.21 / 1.66 / 2.12 | 0.332 | 0.20 | 1.62 | -0.000 / 0.084 | -0.07..+0.59 | +0.5 | 0.0006 | 0.79 | 0.97 | 0.92 | 0.103 / 0.620 | 0.93 | 0 | 0.000221 / 0.0187 |
| 20261001-43-NbWz @ 5000 | 5000 | 1.2 / 1.67 / 2.16 | 0.332 | 0.20 | 1.63 | -0.000 / 0.087 | -0.07..+0.61 | +0.5 | 0.0006 | 0.78 | 0.97 | 0.92 | 0.110 / 0.631 | 1.00 | 0 | 8.51e-05 / 0.0196 |
| 20261001-43-NbWz @ 6000 | 6000 | 1.2 / 1.67 / 2.19 | 0.332 | 0.20 | 1.64 | -0.000 / 0.088 | -0.07..+0.62 | +0.6 | 0.0006 | 0.77 | 0.97 | 0.92 | 0.114 / 0.638 | 1.04 | 0 | 4.08e-05 / 0.0248 |
| 20261001-43-NbWz @ 7000 | 7000 | 1.19 / 1.67 / 2.21 | 0.331 | 0.20 | 1.64 | -0.000 / 0.088 | -0.07..+0.62 | +0.6 | 0.0006 | 0.77 | 0.97 | 0.92 | 0.117 / 0.642 | 1.05 | 0 | 0.000158 / 0.0203 |
| 20261001-43-NbWz @ 8000 | 8000 | 1.19 / 1.68 / 2.22 | 0.331 | 0.20 | 1.64 | -0.000 / 0.088 | -0.07..+0.63 | +0.6 | 0.0006 | 0.77 | 0.97 | 0.92 | 0.118 / 0.643 | 1.06 | 0 | 0.000238 / 0.0226 |
| 20261001-43-NbWz @ 9000 | 9000 | 1.19 / 1.68 / 2.23 | 0.331 | 0.20 | 1.64 | -0.000 / 0.089 | -0.07..+0.63 | +0.6 | 0.0006 | 0.77 | 0.97 | 0.92 | 0.119 / 0.645 | 1.07 | 0 | 0.000507 / 0.0281 |
| 20261001-43-NbWz @ 10000 | 10000 | 1.19 / 1.68 / 2.23 | 0.331 | 0.20 | 1.64 | -0.000 / 0.089 | -0.07..+0.63 | +0.6 | 0.0006 | 0.77 | 0.97 | 0.92 | 0.119 / 0.646 | 1.08 | 0 | 0.000457 / 0.0338 |
| 20261001-43-NbWz @ 11000 | 11000 | 1.19 / 1.68 / 2.23 | 0.331 | 0.20 | 1.64 | -0.000 / 0.089 | -0.07..+0.63 | +0.6 | 0.0006 | 0.77 | 0.97 | 0.92 | 0.119 / 0.648 | 1.08 | 0 | 0.000465 / 0.0431 |
| 20261001-43-NbWz @ 12000 | 12000 | 1.19 / 1.68 / 2.24 | 0.331 | 0.20 | 1.65 | -0.000 / 0.089 | -0.07..+0.63 | +0.6 | 0.0006 | 0.77 | 0.97 | 0.92 | 0.120 / 0.649 | 1.08 | 0 | 0.000394 / 0.028 |

#### SE scale+bias s1

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260929-12-JZOe @ fresh |  | 1.18 / 1.41 / 1.71 | 0.148 | 0.10 | 1.4 | +0.000 / 0.000 | +0.00..+0.00 | -0.1 | 0.0002 | 0.98 | 1.03 | 1.00 |  /  | 0.48 | 0 |  |
| 20260929-22-bWdy @ 1000 | 1000 | 1.28 / 1.54 / 1.78 | 0.277 | 0.18 | 1.53 | -0.000 / 0.052 | -0.06..+0.32 | +0.4 | 0.0005 | 0.88 | 1.01 | 0.94 | 0.046 / 0.413 | 0.63 | 0 |  |
| 20260929-22-bWdy @ 5000 | 5000 | 1.21 / 1.67 / 2.19 | 0.333 | 0.20 | 1.63 | -0.000 / 0.085 | -0.07..+0.60 | +0.6 | 0.0006 | 0.78 | 0.97 | 0.92 | 0.113 / 0.634 | 1.02 | 0 |  |
| 20260929-22-bWdy @ 10000 | 10000 | 1.19 / 1.68 / 2.27 | 0.332 | 0.20 | 1.63 | -0.000 / 0.087 | -0.08..+0.61 | +0.6 | 0.0006 | 0.77 | 0.97 | 0.92 | 0.123 / 0.647 | 1.09 | 0 |  |
| 20260929-22-bWdy @ 11000 | 11000 | 1.19 / 1.69 / 2.27 | 0.332 | 0.20 | 1.63 | -0.000 / 0.087 | -0.08..+0.62 | +0.6 | 0.0006 | 0.77 | 0.97 | 0.92 | 0.123 / 0.648 | 1.09 | 0 |  |
| 20260929-22-bWdy @ 12000 | 12000 | 1.19 / 1.69 / 2.27 | 0.332 | 0.20 | 1.63 | -0.000 / 0.087 | -0.08..+0.62 | +0.6 | 0.0006 | 0.77 | 0.97 | 0.92 | 0.124 / 0.649 | 1.10 | 0 |  |
| 20260929-22-bWdy @ 20000 | 20000 | 1.13 / 1.67 / 2.36 | 0.339 | 0.21 | 1.62 | -0.000 / 0.097 | -0.08..+0.70 | +0.6 | 0.0006 | 0.74 | 0.95 | 0.91 | 0.159 / 0.683 | 1.37 | 0 |  |
| 20260929-22-bWdy @ 33014 | 33014 | 1.03 / 1.69 / 2.64 | 0.36 | 0.22 | 1.65 | -0.000 / 0.121 | -0.08..+0.89 | +0.6 | 0.0007 | 0.67 | 0.95 | 0.88 | 0.238 / 0.757 | 1.66 | 0 |  |

#### SE scale+bias s2

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260930-1-H1Oq @ fresh |  | 1.22 / 1.42 / 1.63 | 0.164 | 0.12 | 1.41 | +0.000 / 0.000 | +0.00..+0.00 | -0.1 | 0.0003 | 0.99 | 0.97 | 1.00 |  /  | 0.48 | 0 |  |
| 20260930-4-k98x @ 1000 | 1000 | 1.23 / 1.55 / 1.82 | 0.292 | 0.19 | 1.51 | +0.000 / 0.053 | -0.06..+0.37 | +0.4 | 0.0005 | 0.89 | 1.01 | 0.96 | 0.030 / 0.416 | 0.62 | 0 | 0.000425 / 0.0209 |
| 20260930-4-k98x @ 5000 | 5000 | 1.16 / 1.68 / 2.1 | 0.341 | 0.21 | 1.63 | +0.000 / 0.078 | -0.07..+0.55 | +0.5 | 0.0006 | 0.78 | 0.99 | 0.92 | 0.101 / 0.655 | 0.79 | 0 | 0.000108 / 0.0186 |
| 20260930-4-k98x @ 7282 | 7282 | 1.15 / 1.69 / 2.13 | 0.341 | 0.21 | 1.64 | +0.000 / 0.079 | -0.07..+0.56 | +0.5 | 0.0006 | 0.77 | 0.99 | 0.92 | 0.107 / 0.663 | 0.80 | 0 | 0.000115 / 0.0258 |

#### SE attenuate-only s1

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260929-13-06yp @ fresh |  | 1.06 / 1.39 / 1.62 | 0.17 | 0.12 | 1.38 | +0.000 / 0.000 | +0.00..+0.00 | +0.0 | 0.0003 | 0.97 | 0.99 | 1.01 |  /  | 0.50 | 0 |  |
| 20260929-23-L6Qm @ 1000 | 1000 | 1.05 / 1.54 / 1.79 | 0.293 | 0.19 | 1.51 | +0.000 / 0.049 | -0.06..+0.31 | +0.5 | 0.0005 | 0.87 | 1.02 | 0.97 | 0.073 / 0.405 | 0.60 | 0 |  |
| 20260929-23-L6Qm @ 5000 | 5000 | 0.973 / 1.68 / 2.19 | 0.343 | 0.21 | 1.63 | +0.000 / 0.076 | -0.07..+0.54 | +0.8 | 0.0006 | 0.76 | 0.98 | 0.91 | 0.114 / 0.641 | 0.90 | 0 |  |
| 20260929-23-L6Qm @ 10000 | 10000 | 0.964 / 1.69 / 2.23 | 0.342 | 0.21 | 1.64 | +0.000 / 0.078 | -0.07..+0.55 | +0.8 | 0.0006 | 0.75 | 0.99 | 0.91 | 0.122 / 0.657 | 0.93 | 0 |  |
| 20260929-23-L6Qm @ 11000 | 11000 | 0.964 / 1.69 / 2.23 | 0.342 | 0.21 | 1.64 | +0.000 / 0.078 | -0.07..+0.55 | +0.8 | 0.0006 | 0.75 | 0.99 | 0.91 | 0.122 / 0.658 | 0.94 | 0 |  |
| 20260929-23-L6Qm @ 12000 | 12000 | 0.963 / 1.69 / 2.24 | 0.342 | 0.21 | 1.64 | +0.000 / 0.078 | -0.07..+0.55 | +0.8 | 0.0006 | 0.75 | 0.99 | 0.91 | 0.122 / 0.658 | 0.94 | 0 |  |
| 20260929-23-L6Qm @ 20000 | 20000 | 0.916 / 1.67 / 2.27 | 0.351 | 0.22 | 1.62 | +0.000 / 0.084 | -0.08..+0.59 | +0.8 | 0.0006 | 0.72 | 0.97 | 0.89 | 0.163 / 0.693 | 1.11 | 0 |  |
| 20260929-23-L6Qm @ 33012 | 33012 | 0.836 / 1.71 / 2.53 | 0.369 | 0.23 | 1.65 | +0.000 / 0.099 | -0.09..+0.70 | +0.9 | 0.0007 | 0.66 | 0.99 | 0.85 | 0.239 / 0.790 | 1.40 | 0 |  |

#### SE attenuate-only s2

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260930-2-Gf9P @ fresh |  | 1.21 / 1.43 / 1.57 | 0.176 | 0.12 | 1.42 | +0.000 / 0.000 | +0.00..+0.00 | +0.1 | 0.0003 | 1.00 | 1.01 | 1.05 |  /  | 0.49 | 0 |  |
| 20260930-5-5TXu @ 1000 | 1000 | 1.18 / 1.57 / 1.8 | 0.292 | 0.19 | 1.53 | +0.000 / 0.047 | -0.06..+0.26 | +0.5 | 0.0005 | 0.89 | 1.03 | 0.99 | 0.047 / 0.427 | 0.64 | 0 | 0.000166 / 0.0193 |
| 20260930-5-5TXu @ 5000 | 5000 | 1.13 / 1.68 / 2.23 | 0.348 | 0.21 | 1.63 | +0.000 / 0.072 | -0.07..+0.46 | +0.7 | 0.0006 | 0.79 | 1.01 | 0.95 | 0.097 / 0.641 | 0.80 | 0 | 0.000166 / 0.0195 |
| 20260930-5-5TXu @ 7289 | 7289 | 1.13 / 1.69 / 2.26 | 0.348 | 0.21 | 1.64 | +0.000 / 0.074 | -0.07..+0.47 | +0.7 | 0.0006 | 0.79 | 1.01 | 0.95 | 0.103 / 0.650 | 0.83 | 0 | 9.93e-05 / 0.0233 |

#### SE none s1

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260929-18-D9is @ fresh |  | 1.24 / 1.4 / 1.7 | 0.152 | 0.11 | 1.39 | +0.000 / 0.000 | +0.00..+0.00 | +0.1 | 0.0002 | 1.01 | 1.01 | 1.01 |  /  | 0.52 | 0 |  |
| 20260929-24-834D @ 1000 | 1000 | 1.24 / 1.55 / 1.79 | 0.271 | 0.18 | 1.53 | -0.000 / 0.041 | -0.07..+0.20 | +0.5 | 0.0005 | 0.90 | 1.03 | 0.95 | 0.045 / 0.388 | 0.65 | 0 |  |
| 20260929-24-834D @ 5000 | 5000 | 1.15 / 1.68 / 2.16 | 0.333 | 0.20 | 1.62 | -0.000 / 0.066 | -0.09..+0.37 | +0.7 | 0.0006 | 0.79 | 1.01 | 0.91 | 0.109 / 0.623 | 0.81 | 0 |  |
| 20260929-24-834D @ 10000 | 10000 | 1.14 / 1.69 / 2.19 | 0.333 | 0.20 | 1.64 | -0.000 / 0.067 | -0.09..+0.38 | +0.7 | 0.0006 | 0.78 | 1.01 | 0.91 | 0.115 / 0.640 | 0.82 | 0 |  |
| 20260929-24-834D @ 11000 | 11000 | 1.14 / 1.69 / 2.19 | 0.333 | 0.20 | 1.64 | -0.000 / 0.067 | -0.09..+0.38 | +0.7 | 0.0006 | 0.78 | 1.01 | 0.91 | 0.115 / 0.641 | 0.82 | 0 |  |
| 20260929-24-834D @ 12000 | 12000 | 1.14 / 1.69 / 2.19 | 0.333 | 0.20 | 1.64 | +0.000 / 0.067 | -0.09..+0.38 | +0.8 | 0.0006 | 0.78 | 1.01 | 0.91 | 0.115 / 0.642 | 0.82 | 0 |  |
| 20260929-24-834D @ 20000 | 20000 | 1.08 / 1.68 / 2.22 | 0.346 | 0.21 | 1.62 | -0.000 / 0.071 | -0.10..+0.39 | +0.8 | 0.0006 | 0.74 | 1.01 | 0.89 | 0.158 / 0.686 | 1.05 | 0 |  |
| 20260929-24-834D @ 32036 | 32036 | 0.99 / 1.72 / 2.46 | 0.369 | 0.22 | 1.66 | -0.000 / 0.082 | -0.10..+0.45 | +0.8 | 0.0007 | 0.67 | 1.01 | 0.85 | 0.241 / 0.765 | 1.38 | 0 |  |

#### SE none s2

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260930-3-V9zk @ fresh |  | 1.14 / 1.41 / 1.64 | 0.146 | 0.10 | 1.4 | +0.000 / 0.000 | +0.00..+0.00 | +0.0 | 0.0002 | 0.99 | 1.06 | 0.97 |  /  | 0.47 | 0 |  |
| 20260930-6-LkS6 @ 1000 | 1000 | 1.13 / 1.52 / 1.79 | 0.275 | 0.18 | 1.49 | +0.000 / 0.041 | -0.05..+0.23 | +0.6 | 0.0005 | 0.90 | 1.06 | 0.93 | 0.038 / 0.385 | 0.58 | 0 | 0.000147 / 0.0186 |
| 20260930-6-LkS6 @ 5000 | 5000 | 1.04 / 1.64 / 2.15 | 0.344 | 0.21 | 1.6 | +0.000 / 0.065 | -0.06..+0.43 | +0.8 | 0.0006 | 0.79 | 1.03 | 0.90 | 0.096 / 0.618 | 0.83 | 0 | 0.000221 / 0.0183 |
| 20260930-6-LkS6 @ 7019 | 7019 | 1.03 / 1.65 / 2.18 | 0.343 | 0.21 | 1.61 | +0.000 / 0.066 | -0.07..+0.44 | +0.8 | 0.0006 | 0.78 | 1.03 | 0.90 | 0.103 / 0.628 | 0.85 | 0 | 0.000218 / 0.0276 |

#### SE zero-beta s1

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260930-7-crxN @ fresh |  | 1.18 / 1.41 / 1.71 | 0.148 | 0.10 | 1.4 | +0.000 / 0.000 | +0.00..+0.00 | -0.1 | 0.0002 | 0.98 | 1.03 | 1.00 |  /  | 0.48 | 0 |  |
| 20260930-9-RrGx @ 1000 | 1000 | 1.28 / 1.56 / 1.81 | 0.281 | 0.18 | 1.53 | +0.000 / 0.050 | -0.06..+0.30 | +0.5 | 0.0005 | 0.87 | 1.01 | 0.94 | 0.050 / 0.422 | 0.67 | 0 | 0.000146 / 0.0184 |
| 20260930-9-RrGx @ 5000 | 5000 | 1.21 / 1.67 / 2.19 | 0.337 | 0.21 | 1.62 | -0.000 / 0.084 | -0.07..+0.60 | +0.6 | 0.0006 | 0.78 | 0.97 | 0.92 | 0.109 / 0.646 | 1.05 | 0 | 0.000164 / 0.0235 |
| 20260930-9-RrGx @ 5030 | 5030 | 1.21 / 1.67 / 2.19 | 0.338 | 0.21 | 1.62 | -0.000 / 0.084 | -0.07..+0.60 | +0.6 | 0.0006 | 0.78 | 0.97 | 0.92 | 0.109 / 0.645 | 1.05 | 0 | 0.000147 / 0.0223 |

#### SE zero-beta s2

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260930-8-8qyR @ fresh |  | 1.22 / 1.42 / 1.63 | 0.164 | 0.12 | 1.41 | +0.000 / 0.000 | +0.00..+0.00 | -0.1 | 0.0003 | 0.99 | 0.97 | 1.00 |  /  | 0.48 | 0 |  |
| 20260930-10-H51a @ 1000 | 1000 | 1.24 / 1.56 / 1.8 | 0.283 | 0.18 | 1.52 | -0.000 / 0.047 | -0.06..+0.31 | +0.4 | 0.0005 | 0.89 | 1.01 | 0.97 | 0.044 / 0.417 | 0.64 | 0 | 0.000114 / 0.0192 |
| 20260930-10-H51a @ 5000 | 5000 | 1.16 / 1.68 / 2.07 | 0.339 | 0.21 | 1.61 | -0.000 / 0.076 | -0.07..+0.54 | +0.5 | 0.0006 | 0.79 | 0.99 | 0.93 | 0.096 / 0.650 | 0.80 | 0 | 5.86e-05 / 0.0195 |
| 20260930-10-H51a @ 5004 | 5004 | 1.16 / 1.68 / 2.07 | 0.339 | 0.21 | 1.61 | -0.000 / 0.076 | -0.07..+0.54 | +0.5 | 0.0006 | 0.79 | 0.99 | 0.93 | 0.096 / 0.649 | 0.81 | 0 | 6.01e-05 / 0.0235 |

#### v5

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260628-1-tWtk @ fresh |  | 1.18 / 1.41 / 1.65 | 0.178 | 0.13 | 1.39 | +0.000 / 0.000 | +0.00..+0.00 | -0.1 | 0.0003 | 1.03 | 1.00 | 1.07 |  /  | 0.53 | 0 |  |
| 20260628-2-a5fc @ 10000 | 10000 | 1.23 / 1.6 / 1.84 | 0.278 | 0.17 | 1.57 | -0.003 / 0.044 | -0.06..+0.24 | +0.2 | 0.0006 | 0.90 | 0.99 | 0.98 | 0.031 / 0.471 | 0.63 | 0 |  |
| 20260628-2-a5fc @ 45441 | 45441 | 1.19 / 1.67 / 1.99 | 0.322 | 0.19 | 1.64 | -0.019 / 0.057 | -0.09..+0.31 | -0.2 | 0.0006 | 0.82 | 0.95 | 0.96 | 0.089 / 0.619 | 0.82 | 0 |  |
| 20260628-9-OdUt @ 15460 | 60901 | 1.11 / 1.61 / 1.92 | 0.327 | 0.20 | 1.58 | -0.027 / 0.060 | -0.11..+0.32 | -0.3 | 0.0007 | 0.79 | 0.95 | 0.94 | 0.158 / 0.634 | 0.88 | 0 |  |
| 20260629-1-Uf4p @ 39419 | 100320 | 1.05 / 1.67 / 2.23 | 0.417 | 0.25 | 1.61 | -0.061 / 0.069 | -0.15..+0.34 | -0.9 | 0.0008 | 0.73 | 0.96 | 0.90 | 0.268 / 0.754 | 1.27 | 0 |  |
| 20260703-1-Dg5v @ 268506 | 368826 | 1.49 / 2.25 / 3.12 | 1.37 | 0.62 | 1.8 | -0.308 / 0.059 | -0.45..-0.06 | -17.0 | 0.0081 | 0.68 | 0.96 | 0.80 | 0.929 / 1.411 | 1.90 | 0 |  |
| 20260714-1-h7vI @ 115000 | 483826 | 2.13 / 2.65 / 3.39 | 2.07 | 0.79 | 1.64 | -0.421 / 0.051 | -0.61..-0.28 | -37.2 | 0.0161 | 0.81 | 1.00 | 0.86 | 1.350 / 1.781 | 1.83 | 0 |  |
| 20260714-1-h7vI @ 336610 | 705436 | 5.28 / 5.54 / 6.4 | 5.27 | 0.94 | 1.74 | -0.646 / 0.087 | -1.04..-0.55 | -191.1 | 0.0697 | 0.95 | 1.03 | 0.96 | 3.244 / 3.894 | 2.44 | 0 |  |
| 20260729-1-VZ2j @ 106333 | 811769 | 7.32 / 7.52 / 8.24 | 7.29 | 0.96 | 1.85 | -0.756 / 0.103 | -1.21..-0.65 | -285.1 | 0.1061 | 0.97 | 1.02 | 0.98 | 4.467 / 5.324 | 2.70 | 0 |  |
| 20260802-2-Xuub @ 49374 | 861143 | 8.02 / 8.22 / 8.83 | 7.99 | 0.97 | 1.93 | -0.807 / 0.108 | -1.28..-0.70 | -299.6 | 0.1126 | 0.98 | 1.01 | 0.98 | 4.906 / 5.828 | 2.73 | 0 |  |
| 20260805-1-0pTW @ 2000 | 859769 | 8.01 / 8.2 / 8.84 | 7.97 | 0.97 | 1.93 | -0.805 / 0.109 | -1.28..-0.69 | -300.4 | 0.1129 | 0.98 | 1.01 | 0.98 | 4.892 / 5.816 | 2.73 | 0 |  |

#### mini2b

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260629-3-3MIV @ fresh |  | 1.15 / 1.4 / 1.6 | 0.166 | 0.12 | 1.39 | +0.000 / 0.000 | +0.00..+0.00 | +0.1 | 0.0003 | 0.97 | 1.05 | 0.94 |  /  | 0.46 | 0 |  |
| 20260629-4-y5u7 @ 10000 | 10000 | 1.23 / 1.61 / 1.89 | 0.273 | 0.17 | 1.58 | -0.004 / 0.043 | -0.06..+0.23 | +0.4 | 0.0006 | 0.84 | 1.04 | 0.91 | 0.062 / 0.520 | 0.71 | 0 |  |
| 20260629-4-y5u7 @ 13464 | 13464 | 1.21 / 1.63 / 1.91 | 0.275 | 0.17 | 1.61 | -0.006 / 0.044 | -0.06..+0.24 | +0.4 | 0.0006 | 0.82 | 1.02 | 0.91 | 0.065 / 0.549 | 0.74 | 0 |  |
| 20260630-5-BEKK @ 120695 | 134159 | 1.01 / 1.75 / 2.63 | 0.493 | 0.29 | 1.66 | -0.099 / 0.069 | -0.19..+0.33 | -1.5 | 0.0012 | 0.63 | 0.94 | 0.79 | 0.450 / 0.874 | 1.77 | 0 |  |
| 20260701-2-SvRu @ 8000 | 142159 | 1.01 / 1.76 / 2.65 | 0.513 | 0.30 | 1.66 | -0.106 / 0.070 | -0.20..+0.33 | -1.7 | 0.0013 | 0.62 | 0.94 | 0.78 | 0.475 / 0.885 | 1.78 | 0 |  |
| 20260705-1-znR7 @ 53000 | 195159 | 1.05 / 1.79 / 2.72 | 0.665 | 0.38 | 1.66 | -0.157 / 0.077 | -0.26..+0.31 | -3.3 | 0.0021 | 0.61 | 0.94 | 0.76 | 0.643 / 0.957 | 1.88 | 0 |  |
| 20260705-1-znR7 @ 114000 | 256159 | 1.17 / 1.87 / 2.78 | 0.897 | 0.49 | 1.63 | -0.217 / 0.079 | -0.35..+0.24 | -6.8 | 0.0038 | 0.64 | 0.95 | 0.76 | 0.799 / 1.065 | 1.94 | 0 |  |

#### coxw

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260629-5-Coxw @ fresh |  | 1.24 / 1.4 / 1.57 | 0.165 | 0.12 | 1.39 | +0.000 / 0.000 | +0.00..+0.00 | +0.0 | 0.0003 | 0.96 | 0.99 | 0.99 |  /  | 0.47 | 0 |  |
| 20260629-6-yqMI @ 10000 | 10000 | 1.21 / 1.62 / 1.99 | 0.267 | 0.17 | 1.59 | -0.004 / 0.039 | -0.07..+0.20 | +0.5 | 0.0005 | 0.84 | 1.01 | 0.96 | 0.048 / 0.512 | 0.63 | 0 |  |
| 20260629-6-yqMI @ 55550 | 55550 | 1.09 / 1.73 / 2.22 | 0.329 | 0.20 | 1.69 | -0.031 / 0.049 | -0.11..+0.25 | +0.0 | 0.0007 | 0.70 | 0.96 | 0.89 | 0.199 / 0.740 | 0.89 | 0 |  |
| 20260709-1-avoB @ 136000 | 191550 | 0.66 / 1.39 / 2.21 | 0.384 | 0.29 | 1.33 | -0.111 / 0.070 | -0.23..+0.28 | -0.9 | 0.0010 | 0.53 | 0.95 | 0.75 | 0.606 / 0.802 | 1.54 | 0 |  |
| 20260709-1-avoB @ 277000 | 332550 | 0.64 / 1.35 / 2.21 | 0.555 | 0.43 | 1.21 | -0.202 / 0.090 | -0.34..+0.30 | -3.1 | 0.0021 | 0.52 | 0.99 | 0.72 | 0.807 / 0.941 | 1.48 | 0 |  |

#### ykkk

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260630-1-YkKk @ fresh |  | 1.12 / 1.39 / 1.82 | 0.182 | 0.13 | 1.37 | +0.000 / 0.000 | +0.00..+0.00 |  |  | 0.94 | 1.04 | 1.11 |  /  | 0.68 | 0 |  |
| 20260630-2-6y0s @ 40677 | 40677 | 0.968 / 1.75 / 2.3 | 0.391 | 0.22 | 1.7 | -0.019 / 0.059 | -0.18..+0.16 |  |  | 0.71 | 1.01 | 0.98 | 0.141 / 0.947 | 1.19 | 0 |  |
| 20260701-1-0Iwe @ 162330 | 203007 | 0.72 / 1.9 / 2.79 | 0.637 | 0.35 | 1.79 | -0.126 / 0.068 | -0.32..+0.07 |  |  | 0.54 | 0.95 | 0.88 | 0.562 / 1.258 | 1.84 | 0 |  |
| 20260710-1-amlg @ 118000 | 321007 | 0.562 / 1.57 / 2.34 | 0.592 | 0.40 | 1.45 | -0.186 / 0.077 | -0.43..+0.04 |  |  | 0.51 | 0.99 | 0.86 | 0.768 / 1.166 | 1.45 | 0 |  |
| 20260710-1-amlg @ 250803 | 453810 | 0.58 / 1.46 / 2.2 | 0.625 | 0.45 | 1.32 | -0.259 / 0.089 | -0.57..+0.02 |  |  | 0.53 | 1.04 | 0.82 | 0.905 / 1.179 | 1.36 | 0 |  |

#### nt8y

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260701-3-nT8Y @ fresh |  | 1.31 / 1.42 / 1.5 | 0.151 | 0.11 | 1.41 | +0.000 / 0.000 | +0.00..+0.00 | +0.1 | 0.0002 | 1.01 | 1.01 | 0.97 |  /  | 0.25 | 0 |  |
| 20260701-4-CIvL @ 1000 | 1000 | 1.3 / 1.48 / 1.68 | 0.24 | 0.16 | 1.47 | -0.000 / 0.013 | -0.02..+0.05 | +0.6 | 0.0004 | 0.95 | 1.04 | 0.95 | 0.052 / 0.335 | 0.31 | 0 |  |
| 20260701-4-CIvL @ 65883 | 65883 | 1.2 / 1.75 / 2.68 | 0.459 | 0.27 | 1.66 | -0.036 / 0.070 | -0.12..+0.44 | -2.4 | 0.0008 | 0.72 | 0.94 | 0.88 | 0.243 / 1.012 | 1.64 | 0 |  |
| 20260701-5-bOYQ @ 70000 | 135883 | 1.15 / 1.81 / 3.22 | 0.675 | 0.37 | 1.67 | -0.086 / 0.083 | -0.16..+0.50 | -5.2 | 0.0015 | 0.65 | 0.96 | 0.82 | 0.493 / 1.198 | 2.12 | 0 |  |
| 20260701-5-bOYQ @ 70779 | 136662 | 1.15 / 1.81 / 3.23 | 0.678 | 0.37 | 1.67 | -0.087 / 0.083 | -0.16..+0.50 | -5.2 | 0.0015 | 0.65 | 0.96 | 0.82 | 0.496 / 1.199 | 2.12 | 0 |  |
| 20260706-2-3CZF @ 15000 | 151662 | 1.16 / 1.84 / 3.29 | 0.724 | 0.39 | 1.67 | -0.098 / 0.084 | -0.17..+0.50 | -6.0 | 0.0017 | 0.65 | 0.97 | 0.82 | 0.549 / 1.236 | 2.16 | 0 |  |
| 20260707-1-cslu @ 140000 | 291662 | 1.36 / 2.09 / 3.63 | 1.14 | 0.54 | 1.74 | -0.216 / 0.080 | -0.38..+0.29 | -14.1 | 0.0035 | 0.68 | 1.08 | 0.77 | 0.969 / 1.522 | 2.34 | 0 |  |
| 20260708-4-kEiZ @ 21086 | 312748 | 1.3 / 2 / 3.45 | 1.1 | 0.55 | 1.65 | -0.228 / 0.079 | -0.40..+0.25 | -14.1 | 0.0035 | 0.68 | 1.09 | 0.77 | 0.975 / 1.480 | 2.20 | 0 |  |

#### qeu8 (replay main, ends Ejp0)

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260702-7-Qeu8 @ fresh |  | 1.31 / 1.41 / 1.52 | 0.169 | 0.12 | 1.41 | +0.000 / 0.000 | +0.00..+0.00 | +0.0 | 0.0003 | 1.00 | 1.01 | 1.02 |  /  | 0.25 | 0 |  |
| 20260702-9-GLu5 @ 1000 | 1000 | 1.31 / 1.48 / 1.72 | 0.239 | 0.16 | 1.46 | -0.000 / 0.015 | -0.03..+0.08 | +0.4 | 0.0004 | 0.95 | 1.03 | 1.00 | 0.054 / 0.328 | 0.33 | 0 |  |
| 20260702-9-GLu5 @ 41000 | 41000 | 1.22 / 1.71 / 2.19 | 0.393 | 0.23 | 1.66 | -0.024 / 0.049 | -0.07..+0.29 | -1.3 | 0.0007 | 0.76 | 0.95 | 0.94 | 0.181 / 0.836 | 0.71 | 0 |  |
| 20260703-1-Lnji @ 67508 | 108915 | 1.13 / 1.86 / 2.78 | 0.67 | 0.37 | 1.72 | -0.082 / 0.060 | -0.15..+0.29 | -4.8 | 0.0013 | 0.66 | 0.95 | 0.87 | 0.461 / 1.096 | 1.50 | 0 |  |
| 20260706-1-PVZp @ 67000 | 175915 | 1.25 / 2.02 / 3.09 | 0.981 | 0.50 | 1.75 | -0.145 / 0.054 | -0.23..+0.12 | -9.7 | 0.0022 | 0.66 | 0.97 | 0.84 | 0.736 / 1.281 | 1.71 | 0 |  |
| 20260727-1-Ejp0 @ 611000 | 786915 | 1.86 / 2.22 / 2.87 | 1.86 | 0.84 | 1.18 | -0.548 / 0.126 | -1.19..-0.40 | -47.6 | 0.0095 | 0.85 | 1.10 | 0.88 | 1.554 / 1.772 | 1.34 | 0 |  |
| 20260727-1-Ejp0 @ 1397000 | 1572915 | 3.81 / 4 / 4.39 | 3.8 | 0.95 | 1.24 | -0.981 / 0.197 | -1.95..-0.77 | -157.9 | 0.0434 | 0.95 | 1.03 | 0.96 | 2.742 / 2.950 | 1.33 | 0 |  |

#### qeu8e (epoch branch)

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260702-7-Qeu8 @ fresh |  | 1.31 / 1.41 / 1.52 | 0.169 | 0.12 | 1.41 | +0.000 / 0.000 | +0.00..+0.00 | +0.0 | 0.0003 | 1.00 | 1.01 | 1.02 |  /  | 0.25 | 0 |  |
| 20260704-1-X79T @ 1000 | 1000 | 1.31 / 1.48 / 1.71 | 0.24 | 0.16 | 1.46 | -0.000 / 0.014 | -0.03..+0.07 | +0.5 | 0.0004 | 0.95 | 1.03 | 1.00 | 0.044 / 0.327 | 0.32 | 0 |  |
| 20260704-1-X79T @ 21224 | 21224 | 1.27 / 1.64 / 1.96 | 0.312 | 0.19 | 1.59 | -0.010 / 0.038 | -0.05..+0.26 | -0.3 | 0.0005 | 0.82 | 0.95 | 0.94 | 0.104 / 0.729 | 0.48 | 0 |  |
| 20260704-2-jSjr @ 42507 | 63731 | 1.16 / 1.74 / 2.37 | 0.447 | 0.26 | 1.66 | -0.037 / 0.059 | -0.09..+0.41 | -1.8 | 0.0008 | 0.73 | 0.95 | 0.91 | 0.260 / 0.954 | 1.33 | 0 |  |
| 20260704-3-h7Pp @ 42507 | 106238 | 1.09 / 1.82 / 2.91 | 0.597 | 0.33 | 1.7 | -0.070 / 0.077 | -0.12..+0.51 | -3.8 | 0.0011 | 0.66 | 0.93 | 0.85 | 0.413 / 1.088 | 1.86 | 0 |  |
| 20260708-5-0YQL @ 5000 | 111238 | 1.07 / 1.8 / 2.9 | 0.598 | 0.34 | 1.68 | -0.072 / 0.078 | -0.13..+0.51 | -3.9 | 0.0011 | 0.66 | 0.93 | 0.85 | 0.428 / 1.083 | 1.85 | 0 |  |
| 20260708-5-0YQL @ 26492 | 132730 | 1 / 1.73 / 2.89 | 0.62 | 0.37 | 1.61 | -0.085 / 0.083 | -0.15..+0.53 | -4.3 | 0.0013 | 0.64 | 0.94 | 0.83 | 0.497 / 1.076 | 1.85 | 0 |  |
| 20260708-6-sFzi @ 88107 | 220837 | 0.861 / 1.62 / 2.88 | 0.75 | 0.47 | 1.43 | -0.133 / 0.110 | -0.25..+0.62 | -7.1 | 0.0019 | 0.60 | 0.96 | 0.79 | 0.743 / 1.147 | 1.87 | 0 |  |

#### qeu8-1blk128

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260711-16-VRR4 @ fresh |  | 1.27 / 1.41 / 1.5 | 0.165 | 0.12 | 1.4 | +0.000 / 0.000 | +0.00..+0.00 | -0.0 | 0.0003 | 1.00 | 0.99 | 1.00 |  /  | 0.25 | 0 |  |
| 20260711-17-pycz @ 1000 | 1000 | 1.32 / 1.46 / 1.68 | 0.232 | 0.16 | 1.45 | -0.000 / 0.013 | -0.02..+0.05 | +0.5 | 0.0004 | 0.96 | 1.02 | 0.98 | 0.049 / 0.263 | 0.30 | 0 |  |
| 20260711-17-pycz @ 61000 | 61000 | 1.03 / 1.44 / 2.09 | 0.302 | 0.21 | 1.4 | -0.027 / 0.031 | -0.06..+0.20 | -0.7 | 0.0005 | 0.76 | 0.94 | 0.91 | 0.283 / 0.702 | 1.31 | 0 |  |
| 20260711-17-pycz @ 120000 | 120000 | 0.834 / 1.37 / 2.36 | 0.416 | 0.30 | 1.31 | -0.063 / 0.041 | -0.11..+0.22 | -2.2 | 0.0008 | 0.65 | 0.97 | 0.83 | 0.492 / 0.835 | 1.65 | 0 |  |

#### qeu8init sf100sl100 vs-UCI

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260702-7-Qeu8 @ fresh |  | 1.31 / 1.41 / 1.52 | 0.169 | 0.12 | 1.41 | +0.000 / 0.000 | +0.00..+0.00 | +0.0 | 0.0003 | 1.00 | 1.01 | 1.02 |  /  | 0.25 | 0 |  |
| 20260712-6-lTiK @ 220000 |  | 0.559 / 1.05 / 1.99 | 0.451 | 0.42 | 0.946 | -0.075 / 0.051 | -0.13..+0.11 | -3.3 | 0.0009 | 0.64 | 1.23 | 0.81 | 0.716 / 0.868 | 0.91 | 0 |  |
| 20260714-1-NYAZ @ 758000 |  | 0.815 / 1.21 / 2.12 | 0.839 | 0.68 | 0.886 | -0.296 / 0.148 | -0.51..+0.30 | -14.2 | 0.0031 | 0.71 | 1.25 | 0.78 | 1.108 / 1.253 | 0.96 | 0 |  |
| 20260722-1-syxR @ 558000 |  | 0.776 / 1.2 / 1.98 | 0.786 | 0.66 | 0.896 | -0.442 / 0.176 | -0.68..+0.39 | -16.0 | 0.0035 | 0.67 | 1.17 | 0.72 | 1.096 / 1.255 | 0.88 | 0 |  |

#### Ejp0 headfix-phase2 (from Ejp0 @681k)

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261001-18-oeNy @ 1000 |  | 2.12 / 2.37 / 3.15 | 1.92 | 0.80 | 1.38 | -0.589 / 0.117 | -1.16..-0.46 | -49.0 | 0.0098 | 0.93 | 1.08 | 0.92 | 0.079 / 0.138 | 1.36 | 0 | 0.0013 / 0.0327 |
| 20261001-18-oeNy @ 11000 |  | 2.02 / 2.36 / 3.2 | 1.77 | 0.76 | 1.51 | -0.589 / 0.119 | -1.14..-0.43 | -45.7 | 0.0091 | 0.85 | 1.07 | 0.88 | 0.071 / 0.228 | 1.51 | 0 | 0.000423 / 0.0408 |
| 20261001-18-oeNy @ 20000 |  | 1.92 / 2.3 / 3.16 | 1.7 | 0.74 | 1.5 | -0.589 / 0.118 | -1.13..-0.43 | -43.8 | 0.0087 | 0.83 | 1.07 | 0.87 | 0.083 / 0.244 | 1.53 | 0 | 0.00013 / 0.0195 |

#### Ejp0 self-play run 1

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260727-1-Ejp0-1 @ 2578 [trainer] |  | 3.52 / 3.74 / 4.15 | 3.52 | 0.94 | 1.23 | -0.925 / 0.190 | -1.85..-0.72 | -147.8 | 0.0390 | 0.94 | 1.03 | 0.95 | 0.002 / 0.007 | 1.31 | 0 | 0.00162 / 0.0478 |
| 20260727-1-Ejp0-4 @ 104903 |  | 1.85 / 1.99 / 2.38 | 1.83 | 0.91 | 0.777 | -0.947 / 0.191 | -1.88..-0.74 | -75.8 | 0.0196 | 0.93 | 1.06 | 0.94 | 0.458 / 0.470 | 0.86 | 0 |  |
| 20260727-1-Ejp0-10 @ 197340 [trainer] |  | 1.68 / 1.81 / 2.26 | 1.66 | 0.91 | 0.714 | -0.948 / 0.191 | -1.88..-0.74 | -69.6 | 0.0178 | 0.92 | 1.07 | 0.94 | 0.503 / 0.519 | 0.80 | 0 | 0.00256 / 0.0605 |
| 20260727-1-Ejp0-9 @ 197340 [champion] |  | 1.75 / 1.88 / 2.31 | 1.73 | 0.91 | 0.737 | -0.947 / 0.191 | -1.88..-0.74 | -71.9 | 0.0185 | 0.92 | 1.06 | 0.94 | 0.486 / 0.501 | 0.82 | 0 |  |

#### Ejp0 self-play run 2

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260727-1-Ejp0-1 @ 20241 |  | 3.53 / 3.73 / 4.16 | 3.52 | 0.94 | 1.23 | -0.924 / 0.190 | -1.85..-0.72 | -147.8 | 0.0389 | 0.94 | 1.03 | 0.95 | 0.000 / 0.003 | 1.30 | 0 |  |
| 20260727-1-Ejp0-39 @ 935524 |  | 3.29 / 3.49 / 4.09 | 3.29 | 0.94 | 1.16 | -0.924 / 0.190 | -1.85..-0.72 | -139.5 | 0.0363 | 0.94 | 1.04 | 0.95 | 0.068 / 0.071 | 1.23 | 0 |  |
| 20260727-1-Ejp0-68 @ 1186322 [trainer] |  | 3.28 / 3.48 / 4.08 | 3.28 | 0.94 | 1.15 | -0.924 / 0.190 | -1.85..-0.72 | -138.9 | 0.0362 | 0.94 | 1.04 | 0.95 | 0.072 / 0.075 | 1.23 | 0 | 0.000512 / 0.0384 |
| 20260727-1-Ejp0-67 @ 1186322 [champion] |  | 3.28 / 3.48 / 4.08 | 3.28 | 0.94 | 1.15 | -0.924 / 0.190 | -1.85..-0.72 | -138.9 | 0.0362 | 0.94 | 1.04 | 0.95 | 0.072 / 0.075 | 1.23 | 0 |  |

#### bzw3 self-play

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260601-11-bzw3-31 @ 467065 |  | 1.47 / 1.95 / 2.95 | 0.935 | 0.48 | 1.7 | -0.189 / 0.159 | -0.33..+0.74 | -7.7 | 0.0043 | 0.80 | 1.01 | 0.82 |  /  | 1.85 | 0 |  |
| 20260601-11-bzw3-32 @ 467099 [trainer] |  | 1.78 / 2.47 / 4.05 | 1.41 | 0.56 | 2.03 | -0.276 / 0.243 | -0.46..+0.78 | -14.0 | 0.0074 | 0.76 | 1.07 | 0.79 | 0.309 / 0.451 | 2.79 | 0 | 0.000496 / 0.0165 |
| 20260601-11-bzw3-31 @ 467099 [champion] |  | 1.47 / 1.95 / 2.95 | 0.935 | 0.48 | 1.7 | -0.189 / 0.159 | -0.33..+0.74 | -7.7 | 0.0043 | 0.80 | 1.01 | 0.82 | 0.000 / 0.000 | 1.85 | 0 |  |

#### KbHZ self-play (fp32)

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260514-1-KbHZ-18 @ 494927 [trainer] |  | 1.12 / 1.4 / 1.79 | 0.216 | 0.15 | 1.39 | -0.000 / 0.013 | -0.02..+0.07 |  |  | 0.98 | 1.08 | 0.99 |  /  | 0.47 | 0 | 8.12e-06 / 0.00205 |
| 20260514-1-KbHZ-23 @ 532369 [trainer] |  | 1.12 / 1.39 / 1.78 | 0.22 | 0.16 | 1.39 | -0.000 / 0.013 | -0.03..+0.07 |  |  | 0.99 | 1.07 | 1.00 | 0.004 / 0.020 | 0.46 | 0 | 7.45e-06 / 0.00376 |
| 20260514-1-KbHZ-22 @ 532369 [champion] |  | 1.12 / 1.39 / 1.78 | 0.22 | 0.16 | 1.39 | -0.000 / 0.013 | -0.03..+0.07 |  |  | 0.99 | 1.07 | 1.00 | 0.004 / 0.019 | 0.46 | 0 |  |

#### sMe9 self-play (fp32)

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260525-1-sMe9-26 @ 197269 |  | 1.15 / 1.37 / 1.74 | 0.254 | 0.18 | 1.35 | -0.000 / 0.017 | -0.02..+0.08 |  |  | 1.00 | 0.92 | 1.00 |  /  | 0.63 | 0 |  |
| 20260525-1-sMe9-33 @ 373416 [trainer] |  | 1 / 1.22 / 1.56 | 0.229 | 0.19 | 1.19 | -0.000 / 0.020 | -0.03..+0.10 |  |  | 0.95 | 0.89 | 0.98 | 0.117 / 0.156 | 0.62 | 0 | 0.000646 / 0.0351 |
| 20260525-1-sMe9-32 @ 373416 [champion] |  | 1.03 / 1.25 / 1.6 | 0.234 | 0.19 | 1.22 | -0.000 / 0.020 | -0.03..+0.10 |  |  | 0.96 | 0.90 | 0.99 | 0.097 / 0.131 | 0.62 | 0 |  |

#### LWKa self-play (v4 12-block)

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260531-9-LWKa-11 @ 106695 [trainer] |  | 1.38 / 1.58 / 2.43 | 0.311 | 0.19 | 1.54 | -0.016 / 0.106 | -0.11..+0.74 | -0.2 | 0.0006 | 0.88 | 0.90 | 0.97 |  /  | 1.57 | 0 | 0.000547 / 0.029 |
| 20260531-9-LWKa-10 @ 106695 [champion] |  | 1.36 / 1.57 / 2.17 | 0.291 | 0.18 | 1.54 | -0.013 / 0.083 | -0.10..+0.55 | -0.1 | 0.0005 | 0.90 | 0.91 | 0.99 | 0.041 / 0.087 | 1.31 | 0 |  |

#### LMGh self-play

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260609-12-LMGh-4 @ 79135 [trainer] |  | 1.21 / 2.03 / 5.49 | 0.826 | 0.35 | 1.85 | -0.007 / 0.284 | -0.64..+0.87 | -7.6 | 0.0059 | 1.06 | 1.57 | 1.10 |  /  | 3.44 | 0 | 0.000627 / 0.016 |
| 20260609-12-LMGh-3 @ 79135 [champion] |  | 1.21 / 2.03 / 5.49 | 0.826 | 0.35 | 1.85 | -0.007 / 0.284 | -0.64..+0.87 | -7.6 | 0.0059 | 1.06 | 1.57 | 1.10 | 0.000 / 0.000 | 3.44 | 0 |  |

#### WjRY self-play

| checkpoint | cum | row norm min / median / max | mean-row norm | mean-row ratio | residual norm median | bias mean / std | bias range | static shared level | shared-row rounding noise (nats) | underpromo/Q | knight/Q | Q-promo/Q | row rel. change vs reference min / median | max abs W | weak final cols | row velocity norm min / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260609-14-WjRY-8 @ 98974 [trainer] |  | 1.5 / 1.82 / 3.6 | 0.766 | 0.40 | 1.69 | -0.020 / 0.246 | -0.22..+1.53 | -3.7 | 0.0024 | 0.95 | 1.08 | 1.02 |  /  | 1.88 | 0 | 0.00184 / 0.0235 |
| 20260609-14-WjRY-7 @ 98974 [champion] |  | 1.35 / 1.62 / 2.12 | 0.298 | 0.18 | 1.59 | -0.012 / 0.078 | -0.11..+0.49 | -0.1 | 0.0005 | 0.90 | 0.94 | 0.94 | 0.231 / 0.490 | 0.91 | 0 |  |

<!-- end:evolution_final -->

### Move-type families at lineage-latest

Q1–Q7 = queen-style moves by distance; Kn = knight; UP-N/R/B = underpromotion by piece; UP-fwd/capL/capR = underpromotion by direction; QP = queen promotion.

<!-- begin:families_latest -->

**Row norm, family mean / median queen-style row norm**

| lineage | checkpoint | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Kn | UP-N | UP-R | UP-B | UP-fwd | UP-capL | UP-capR | QP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SE scale+bias full-leaky fresh (never trained) | 20261001-23-Dmwe @ fresh | 0.99 | 0.96 | 1.06 | 0.99 | 1.02 | 1.01 | 0.99 | 1.03 | 0.98 | 1.00 | 0.97 | 0.97 | 0.97 | 1.01 | 1.00 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 12000 | 1.07 | 1.02 | 1.04 | 1.04 | 1.04 | 0.94 | 0.87 | 0.97 | 0.77 | 0.78 | 0.77 | 0.84 | 0.72 | 0.75 | 0.92 |
| SE scale+bias s1 | 20260929-22-bWdy @ 33014 | 1.09 | 1.03 | 1.01 | 1.03 | 1.03 | 0.93 | 0.84 | 0.95 | 0.67 | 0.67 | 0.67 | 0.76 | 0.61 | 0.64 | 0.88 |
| SE scale+bias s2 | 20260930-4-k98x @ 7282 | 1.05 | 1.04 | 1.02 | 1.02 | 1.03 | 0.94 | 0.89 | 0.99 | 0.77 | 0.79 | 0.77 | 0.84 | 0.73 | 0.75 | 0.92 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 33012 | 1.06 | 1.06 | 1.00 | 1.02 | 1.01 | 0.90 | 0.84 | 0.99 | 0.61 | 0.69 | 0.67 | 0.76 | 0.64 | 0.56 | 0.85 |
| SE attenuate-only s2 | 20260930-5-5TXu @ 7289 | 1.03 | 1.04 | 1.03 | 1.02 | 1.02 | 0.95 | 0.86 | 1.01 | 0.78 | 0.79 | 0.79 | 0.87 | 0.73 | 0.76 | 0.95 |
| SE none s1 | 20260929-24-834D @ 32036 | 1.11 | 1.07 | 1.00 | 1.01 | 1.03 | 0.93 | 0.83 | 1.01 | 0.67 | 0.68 | 0.67 | 0.77 | 0.63 | 0.60 | 0.85 |
| SE none s2 | 20260930-6-LkS6 @ 7019 | 1.06 | 1.05 | 1.03 | 1.06 | 1.04 | 0.96 | 0.89 | 1.03 | 0.81 | 0.79 | 0.75 | 0.88 | 0.76 | 0.72 | 0.90 |
| SE zero-beta s1 | 20260930-9-RrGx @ 5030 | 1.05 | 1.01 | 1.03 | 1.03 | 1.03 | 0.94 | 0.87 | 0.97 | 0.77 | 0.79 | 0.77 | 0.84 | 0.72 | 0.76 | 0.92 |
| SE zero-beta s2 | 20260930-10-H51a @ 5004 | 1.04 | 1.04 | 1.02 | 1.02 | 1.03 | 0.94 | 0.90 | 0.99 | 0.78 | 0.80 | 0.78 | 0.85 | 0.75 | 0.77 | 0.93 |
| v5 | 20260805-1-0pTW @ 2000 | 1.05 | 1.02 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 1.01 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 |
| mini2b | 20260705-1-znR7 @ 114000 | 1.19 | 1.04 | 1.01 | 0.98 | 0.97 | 0.95 | 0.84 | 0.95 | 0.65 | 0.66 | 0.63 | 0.68 | 0.62 | 0.63 | 0.76 |
| coxw | 20260709-1-avoB @ 277000 | 1.33 | 1.09 | 1.00 | 0.99 | 0.98 | 0.92 | 0.79 | 0.99 | 0.52 | 0.52 | 0.51 | 0.58 | 0.49 | 0.48 | 0.72 |
| ykkk | 20260710-1-amlg @ 250803 | 1.20 | 1.13 | 0.97 | 0.92 | 1.00 | 0.91 | 0.76 | 1.04 | 0.54 | 0.54 | 0.50 | 0.71 | 0.44 | 0.43 | 0.82 |
| nt8y | 20260708-4-kEiZ @ 21086 | 1.43 | 1.23 | 1.05 | 1.02 | 0.95 | 0.89 | 0.82 | 1.09 | 0.68 | 0.68 | 0.67 | 0.72 | 0.65 | 0.65 | 0.77 |
| qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0 @ 1397000 | 1.06 | 1.03 | 1.01 | 1.00 | 0.99 | 0.97 | 0.97 | 1.03 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.96 |
| qeu8e (epoch branch) | 20260708-6-sFzi @ 88107 | 1.34 | 1.11 | 1.00 | 0.98 | 0.96 | 0.92 | 0.80 | 0.96 | 0.59 | 0.61 | 0.58 | 0.68 | 0.55 | 0.56 | 0.79 |
| qeu8-1blk128 | 20260711-17-pycz @ 120000 | 1.34 | 1.07 | 0.98 | 1.00 | 1.00 | 0.96 | 0.85 | 0.97 | 0.66 | 0.64 | 0.65 | 0.72 | 0.61 | 0.62 | 0.83 |
| qeu8init sf100sl100 vs-UCI | 20260722-1-syxR @ 558000 | 1.33 | 1.21 | 1.07 | 0.99 | 0.90 | 0.79 | 0.73 | 1.17 | 0.66 | 0.68 | 0.66 | 0.67 | 0.66 | 0.66 | 0.72 |
| Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy @ 20000 | 1.13 | 1.09 | 1.01 | 1.00 | 0.97 | 0.94 | 0.91 | 1.07 | 0.83 | 0.83 | 0.83 | 0.84 | 0.83 | 0.83 | 0.87 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-10 @ 197340 [trainer] | 1.13 | 1.06 | 1.02 | 1.00 | 0.98 | 0.96 | 0.95 | 1.07 | 0.92 | 0.92 | 0.92 | 0.93 | 0.92 | 0.92 | 0.94 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-9 @ 197340 [champion] | 1.12 | 1.06 | 1.02 | 1.00 | 0.98 | 0.96 | 0.95 | 1.06 | 0.92 | 0.92 | 0.92 | 0.93 | 0.92 | 0.92 | 0.94 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-68 @ 1186322 [trainer] | 1.09 | 1.04 | 1.01 | 1.00 | 0.99 | 0.97 | 0.96 | 1.04 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.95 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-67 @ 1186322 [champion] | 1.09 | 1.04 | 1.01 | 1.00 | 0.99 | 0.97 | 0.96 | 1.04 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.95 |
| bzw3 self-play | 20260601-11-bzw3-32 @ 467099 [trainer] | 1.40 | 1.15 | 1.01 | 0.98 | 0.97 | 0.95 | 0.93 | 1.07 | 0.76 | 0.77 | 0.76 | 0.83 | 0.72 | 0.74 | 0.79 |
| bzw3 self-play | 20260601-11-bzw3-31 @ 467099 [champion] | 1.26 | 1.07 | 0.99 | 0.99 | 1.00 | 1.01 | 0.98 | 1.01 | 0.79 | 0.82 | 0.79 | 0.86 | 0.76 | 0.77 | 0.82 |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-23 @ 532369 [trainer] | 1.12 | 1.06 | 1.00 | 1.01 | 0.96 | 0.95 | 0.98 | 1.07 | 0.96 | 1.00 | 1.00 | 0.97 | 1.02 | 0.96 | 1.00 |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-22 @ 532369 [champion] | 1.12 | 1.06 | 1.00 | 1.01 | 0.96 | 0.95 | 0.98 | 1.07 | 0.96 | 1.00 | 1.00 | 0.97 | 1.02 | 0.96 | 1.00 |
| sMe9 self-play (fp32) | 20260525-1-sMe9-33 @ 373416 [trainer] | 0.91 | 0.95 | 1.03 | 1.02 | 1.08 | 1.02 | 0.99 | 0.89 | 0.96 | 0.95 | 0.93 | 0.93 | 0.97 | 0.95 | 0.98 |
| sMe9 self-play (fp32) | 20260525-1-sMe9-32 @ 373416 [champion] | 0.91 | 0.95 | 1.03 | 1.02 | 1.09 | 1.03 | 1.00 | 0.90 | 0.97 | 0.96 | 0.95 | 0.94 | 0.99 | 0.96 | 0.99 |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-11 @ 106695 [trainer] | 0.98 | 1.06 | 0.98 | 0.96 | 1.02 | 1.04 | 0.97 | 0.90 | 0.88 | 0.87 | 0.89 | 0.84 | 0.89 | 0.90 | 0.97 |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-10 @ 106695 [champion] | 0.96 | 1.04 | 0.99 | 0.96 | 1.02 | 1.03 | 0.97 | 0.91 | 0.90 | 0.89 | 0.91 | 0.87 | 0.91 | 0.93 | 0.99 |
| LMGh self-play | 20260609-12-LMGh-4 @ 79135 [trainer] | 1.86 | 1.47 | 1.06 | 0.97 | 0.93 | 0.82 | 0.79 | 1.57 | 1.10 | 1.06 | 1.01 | 1.50 | 0.80 | 0.87 | 1.10 |
| LMGh self-play | 20260609-12-LMGh-3 @ 79135 [champion] | 1.86 | 1.47 | 1.06 | 0.97 | 0.93 | 0.82 | 0.79 | 1.57 | 1.10 | 1.06 | 1.01 | 1.50 | 0.80 | 0.87 | 1.10 |
| WjRY self-play | 20260609-14-WjRY-8 @ 98974 [trainer] | 1.55 | 1.14 | 0.99 | 0.96 | 0.98 | 1.00 | 0.97 | 1.08 | 1.03 | 0.90 | 0.93 | 1.01 | 0.94 | 0.91 | 1.02 |
| WjRY self-play | 20260609-14-WjRY-7 @ 98974 [champion] | 1.03 | 1.01 | 0.99 | 1.00 | 1.02 | 1.04 | 1.01 | 0.94 | 0.96 | 0.85 | 0.90 | 0.91 | 0.91 | 0.89 | 0.94 |

**Bias, family mean (nats; init 0)**

| lineage | checkpoint | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Kn | UP-N | UP-R | UP-B | UP-fwd | UP-capL | UP-capR | QP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SE scale+bias full-leaky fresh (never trained) | 20261001-23-Dmwe @ fresh | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 12000 | +0.15 | +0.02 | -0.03 | -0.04 | -0.04 | -0.04 | -0.03 | +0.04 | -0.01 | -0.03 | -0.04 | -0.00 | -0.04 | -0.03 | -0.02 |
| SE scale+bias s1 | 20260929-22-bWdy @ 33014 | +0.19 | +0.03 | -0.04 | -0.04 | -0.05 | -0.05 | -0.04 | +0.05 | -0.01 | -0.03 | -0.04 | -0.01 | -0.04 | -0.02 | -0.03 |
| SE scale+bias s2 | 20260930-4-k98x @ 7282 | +0.13 | +0.01 | -0.02 | -0.03 | -0.03 | -0.03 | -0.04 | +0.05 | -0.02 | -0.02 | -0.02 | -0.01 | -0.03 | -0.02 | -0.02 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 33012 | +0.16 | +0.00 | -0.03 | -0.04 | -0.05 | -0.03 | -0.03 | +0.04 | -0.02 | -0.01 | -0.03 | +0.00 | -0.05 | -0.02 | -0.04 |
| SE attenuate-only s2 | 20260930-5-5TXu @ 7289 | +0.12 | +0.02 | -0.02 | -0.04 | -0.03 | -0.03 | -0.04 | +0.05 | -0.03 | -0.02 | -0.02 | +0.00 | -0.03 | -0.04 | -0.03 |
| SE none s1 | 20260929-24-834D @ 32036 | +0.14 | +0.01 | -0.03 | -0.04 | -0.03 | -0.03 | -0.03 | +0.07 | -0.02 | -0.03 | -0.02 | -0.00 | -0.03 | -0.04 | -0.02 |
| SE none s2 | 20260930-6-LkS6 @ 7019 | +0.11 | +0.02 | -0.01 | -0.04 | -0.03 | -0.03 | -0.02 | +0.05 | -0.02 | -0.04 | -0.04 | -0.04 | -0.03 | -0.03 | -0.01 |
| SE zero-beta s1 | 20260930-9-RrGx @ 5030 | +0.13 | +0.02 | -0.03 | -0.04 | -0.04 | -0.04 | -0.03 | +0.04 | -0.01 | -0.03 | -0.03 | -0.00 | -0.04 | -0.03 | -0.02 |
| SE zero-beta s2 | 20260930-10-H51a @ 5004 | +0.12 | +0.01 | -0.02 | -0.03 | -0.03 | -0.03 | -0.04 | +0.05 | -0.02 | -0.02 | -0.02 | -0.01 | -0.03 | -0.02 | -0.02 |
| v5 | 20260805-1-0pTW @ 2000 | -0.93 | -0.90 | -0.81 | -0.80 | -0.78 | -0.73 | -0.72 | -0.90 | -0.72 | -0.73 | -0.71 | -0.72 | -0.73 | -0.71 | -0.71 |
| mini2b | 20260705-1-znR7 @ 114000 | -0.11 | -0.21 | -0.24 | -0.24 | -0.23 | -0.23 | -0.24 | -0.23 | -0.24 | -0.24 | -0.23 | -0.23 | -0.24 | -0.24 | -0.20 |
| coxw | 20260709-1-avoB @ 277000 | -0.15 | -0.19 | -0.25 | -0.25 | -0.23 | -0.20 | -0.21 | -0.16 | -0.18 | -0.18 | -0.18 | -0.17 | -0.19 | -0.18 | -0.21 |
| ykkk | 20260710-1-amlg @ 250803 | -0.29 | -0.26 | -0.31 | -0.31 | -0.28 | -0.24 | -0.24 | -0.17 | -0.22 | -0.19 | -0.25 | -0.25 | -0.21 | -0.21 | -0.27 |
| nt8y | 20260708-4-kEiZ @ 21086 | -0.16 | -0.24 | -0.24 | -0.24 | -0.24 | -0.22 | -0.23 | -0.32 | -0.19 | -0.19 | -0.18 | -0.19 | -0.18 | -0.19 | -0.17 |
| qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0 @ 1397000 | -1.30 | -1.14 | -1.02 | -0.95 | -0.90 | -0.85 | -0.82 | -1.14 | -0.79 | -0.78 | -0.80 | -0.80 | -0.78 | -0.79 | -0.80 |
| qeu8e (epoch branch) | 20260708-6-sFzi @ 88107 | -0.01 | -0.12 | -0.18 | -0.17 | -0.17 | -0.15 | -0.14 | -0.14 | -0.11 | -0.11 | -0.13 | -0.11 | -0.13 | -0.12 | -0.14 |
| qeu8-1blk128 | 20260711-17-pycz @ 120000 | -0.00 | -0.06 | -0.08 | -0.08 | -0.08 | -0.08 | -0.07 | -0.06 | -0.06 | -0.06 | -0.05 | -0.06 | -0.05 | -0.06 | -0.08 |
| qeu8init sf100sl100 vs-UCI | 20260722-1-syxR @ 558000 | -0.24 | -0.42 | -0.50 | -0.48 | -0.49 | -0.52 | -0.49 | -0.33 | -0.49 | -0.50 | -0.50 | -0.49 | -0.50 | -0.49 | -0.48 |
| Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy @ 20000 | -0.76 | -0.70 | -0.64 | -0.59 | -0.55 | -0.51 | -0.49 | -0.68 | -0.46 | -0.45 | -0.47 | -0.46 | -0.46 | -0.46 | -0.45 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-10 @ 197340 [trainer] | -1.26 | -1.11 | -0.99 | -0.92 | -0.87 | -0.82 | -0.79 | -1.11 | -0.76 | -0.75 | -0.77 | -0.77 | -0.75 | -0.76 | -0.77 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-9 @ 197340 [champion] | -1.26 | -1.11 | -0.99 | -0.92 | -0.87 | -0.82 | -0.79 | -1.11 | -0.76 | -0.75 | -0.77 | -0.77 | -0.75 | -0.76 | -0.77 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-68 @ 1186322 [trainer] | -1.23 | -1.08 | -0.97 | -0.90 | -0.84 | -0.80 | -0.77 | -1.08 | -0.73 | -0.73 | -0.75 | -0.75 | -0.73 | -0.73 | -0.75 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-67 @ 1186322 [champion] | -1.23 | -1.08 | -0.97 | -0.90 | -0.84 | -0.80 | -0.77 | -1.08 | -0.73 | -0.73 | -0.75 | -0.75 | -0.73 | -0.73 | -0.75 |
| bzw3 self-play | 20260601-11-bzw3-32 @ 467099 [trainer] | +0.34 | -0.21 | -0.36 | -0.39 | -0.40 | -0.38 | -0.37 | -0.28 | -0.36 | -0.38 | -0.38 | -0.39 | -0.35 | -0.38 | -0.39 |
| bzw3 self-play | 20260601-11-bzw3-31 @ 467099 [champion] | +0.19 | -0.18 | -0.26 | -0.26 | -0.25 | -0.25 | -0.24 | -0.18 | -0.23 | -0.25 | -0.25 | -0.25 | -0.22 | -0.25 | -0.25 |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-23 @ 532369 [trainer] | +0.01 | +0.00 | -0.00 | -0.01 | +0.00 | -0.00 | -0.01 | +0.00 | -0.01 | +0.00 | -0.00 | +0.00 | -0.01 | -0.01 | +0.00 |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-22 @ 532369 [champion] | +0.01 | +0.00 | -0.00 | -0.01 | +0.00 | -0.00 | -0.01 | +0.00 | -0.01 | +0.00 | -0.00 | +0.00 | -0.01 | -0.01 | +0.00 |
| sMe9 self-play (fp32) | 20260525-1-sMe9-33 @ 373416 [trainer] | +0.03 | -0.00 | -0.01 | -0.01 | -0.01 | -0.01 | -0.01 | +0.03 | +0.01 | +0.01 | -0.01 | +0.00 | +0.01 | +0.00 | -0.00 |
| sMe9 self-play (fp32) | 20260525-1-sMe9-32 @ 373416 [champion] | +0.03 | -0.00 | -0.01 | -0.01 | -0.01 | -0.01 | -0.01 | +0.03 | +0.01 | +0.01 | -0.01 | +0.00 | +0.01 | +0.00 | -0.00 |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-11 @ 106695 [trainer] | +0.17 | +0.00 | -0.05 | -0.06 | -0.06 | -0.06 | -0.05 | +0.03 | -0.04 | -0.05 | -0.06 | -0.04 | -0.05 | -0.07 | -0.05 |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-10 @ 106695 [champion] | +0.14 | +0.00 | -0.04 | -0.05 | -0.05 | -0.05 | -0.04 | +0.03 | -0.03 | -0.04 | -0.05 | -0.03 | -0.04 | -0.06 | -0.04 |
| LMGh self-play | 20260609-12-LMGh-4 @ 79135 [trainer] | +0.47 | +0.13 | -0.13 | -0.13 | -0.15 | -0.14 | -0.16 | +0.20 | -0.15 | -0.08 | -0.07 | +0.11 | -0.18 | -0.22 | -0.10 |
| LMGh self-play | 20260609-12-LMGh-3 @ 79135 [champion] | +0.47 | +0.13 | -0.13 | -0.13 | -0.15 | -0.14 | -0.16 | +0.20 | -0.15 | -0.08 | -0.07 | +0.11 | -0.18 | -0.22 | -0.10 |
| WjRY self-play | 20260609-14-WjRY-8 @ 98974 [trainer] | +0.53 | +0.00 | -0.11 | -0.12 | -0.13 | -0.12 | -0.12 | +0.10 | -0.18 | -0.14 | -0.15 | -0.16 | -0.13 | -0.17 | -0.12 |
| WjRY self-play | 20260609-14-WjRY-7 @ 98974 [champion] | +0.15 | -0.01 | -0.03 | -0.04 | -0.05 | -0.05 | -0.04 | +0.03 | -0.07 | -0.03 | -0.04 | -0.04 | -0.04 | -0.06 | -0.03 |

**Residual row norm (row minus the shared mean row), family mean / median queen-style residual** — the family comparison that is not masked by the shared row

| lineage | checkpoint | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Kn | UP-N | UP-R | UP-B | UP-fwd | UP-capL | UP-capR | QP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SE scale+bias full-leaky fresh (never trained) | 20261001-23-Dmwe @ fresh | 0.99 | 0.96 | 1.05 | 0.99 | 1.01 | 1.01 | 0.98 | 1.02 | 0.97 | 0.99 | 0.97 | 0.96 | 0.97 | 1.00 | 1.00 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 12000 | 1.07 | 1.02 | 1.03 | 1.04 | 1.06 | 0.96 | 0.88 | 0.99 | 0.80 | 0.81 | 0.81 | 0.87 | 0.75 | 0.80 | 0.96 |
| SE scale+bias s1 | 20260929-22-bWdy @ 33014 | 1.09 | 1.02 | 1.00 | 1.02 | 1.04 | 0.94 | 0.85 | 0.97 | 0.70 | 0.71 | 0.71 | 0.79 | 0.65 | 0.68 | 0.91 |
| SE scale+bias s2 | 20260930-4-k98x @ 7282 | 1.04 | 1.03 | 1.02 | 1.02 | 1.04 | 0.95 | 0.91 | 1.01 | 0.79 | 0.84 | 0.82 | 0.87 | 0.79 | 0.80 | 0.95 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 33012 | 1.05 | 1.05 | 1.00 | 1.02 | 1.02 | 0.92 | 0.85 | 1.01 | 0.64 | 0.73 | 0.71 | 0.79 | 0.69 | 0.60 | 0.88 |
| SE attenuate-only s2 | 20260930-5-5TXu @ 7289 | 1.03 | 1.04 | 1.02 | 1.03 | 1.04 | 0.97 | 0.88 | 1.03 | 0.82 | 0.85 | 0.83 | 0.91 | 0.78 | 0.81 | 0.98 |
| SE none s1 | 20260929-24-834D @ 32036 | 1.10 | 1.06 | 1.00 | 1.01 | 1.05 | 0.94 | 0.85 | 1.03 | 0.71 | 0.71 | 0.73 | 0.81 | 0.68 | 0.65 | 0.89 |
| SE none s2 | 20260930-6-LkS6 @ 7019 | 1.05 | 1.04 | 1.03 | 1.06 | 1.05 | 0.97 | 0.91 | 1.05 | 0.85 | 0.83 | 0.80 | 0.90 | 0.81 | 0.77 | 0.91 |
| SE zero-beta s1 | 20260930-9-RrGx @ 5030 | 1.05 | 1.01 | 1.03 | 1.03 | 1.04 | 0.96 | 0.88 | 0.99 | 0.81 | 0.82 | 0.81 | 0.87 | 0.77 | 0.81 | 0.95 |
| SE zero-beta s2 | 20260930-10-H51a @ 5004 | 1.02 | 1.03 | 1.02 | 1.02 | 1.04 | 0.95 | 0.91 | 1.01 | 0.81 | 0.85 | 0.83 | 0.88 | 0.80 | 0.81 | 0.96 |
| v5 | 20260805-1-0pTW @ 2000 | 1.62 | 1.32 | 1.07 | 1.01 | 0.90 | 0.81 | 0.70 | 1.19 | 0.46 | 0.46 | 0.46 | 0.52 | 0.42 | 0.44 | 0.57 |
| mini2b | 20260705-1-znR7 @ 114000 | 1.24 | 1.06 | 1.01 | 0.99 | 0.98 | 0.96 | 0.81 | 0.98 | 0.57 | 0.58 | 0.54 | 0.62 | 0.52 | 0.54 | 0.71 |
| coxw | 20260709-1-avoB @ 277000 | 1.36 | 1.10 | 1.00 | 0.99 | 0.98 | 0.92 | 0.78 | 1.01 | 0.47 | 0.47 | 0.46 | 0.53 | 0.44 | 0.44 | 0.68 |
| ykkk | 20260710-1-amlg @ 250803 | 1.20 | 1.12 | 0.97 | 0.92 | 1.02 | 0.92 | 0.76 | 1.08 | 0.49 | 0.50 | 0.46 | 0.70 | 0.37 | 0.37 | 0.80 |
| nt8y | 20260708-4-kEiZ @ 21086 | 1.56 | 1.29 | 1.06 | 1.02 | 0.94 | 0.86 | 0.75 | 1.14 | 0.53 | 0.54 | 0.53 | 0.61 | 0.50 | 0.50 | 0.68 |
| qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0 @ 1397000 | 1.47 | 1.24 | 1.07 | 0.99 | 0.85 | 0.68 | 0.56 | 1.20 | 0.33 | 0.33 | 0.32 | 0.38 | 0.30 | 0.30 | 0.46 |
| qeu8e (epoch branch) | 20260708-6-sFzi @ 88107 | 1.40 | 1.12 | 0.99 | 0.98 | 0.97 | 0.92 | 0.77 | 0.99 | 0.55 | 0.58 | 0.54 | 0.64 | 0.50 | 0.52 | 0.78 |
| qeu8-1blk128 | 20260711-17-pycz @ 120000 | 1.36 | 1.06 | 0.98 | 1.00 | 1.00 | 0.97 | 0.86 | 1.00 | 0.71 | 0.69 | 0.67 | 0.75 | 0.65 | 0.67 | 0.85 |
| qeu8init sf100sl100 vs-UCI | 20260722-1-syxR @ 558000 | 1.51 | 1.33 | 1.12 | 0.97 | 0.82 | 0.60 | 0.44 | 1.30 | 0.28 | 0.35 | 0.27 | 0.33 | 0.28 | 0.29 | 0.44 |
| Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy @ 20000 | 1.50 | 1.29 | 1.07 | 0.98 | 0.88 | 0.75 | 0.61 | 1.22 | 0.35 | 0.36 | 0.35 | 0.48 | 0.28 | 0.29 | 0.57 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-10 @ 197340 [trainer] | 1.63 | 1.32 | 1.09 | 0.98 | 0.85 | 0.69 | 0.58 | 1.33 | 0.33 | 0.33 | 0.33 | 0.39 | 0.30 | 0.30 | 0.49 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-9 @ 197340 [champion] | 1.61 | 1.31 | 1.08 | 0.98 | 0.85 | 0.69 | 0.58 | 1.32 | 0.33 | 0.33 | 0.33 | 0.39 | 0.30 | 0.30 | 0.49 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-68 @ 1186322 [trainer] | 1.58 | 1.28 | 1.08 | 0.99 | 0.86 | 0.70 | 0.59 | 1.29 | 0.36 | 0.36 | 0.36 | 0.41 | 0.34 | 0.33 | 0.49 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-67 @ 1186322 [champion] | 1.58 | 1.28 | 1.08 | 0.99 | 0.86 | 0.70 | 0.59 | 1.29 | 0.36 | 0.36 | 0.36 | 0.41 | 0.34 | 0.33 | 0.49 |
| bzw3 self-play | 20260601-11-bzw3-32 @ 467099 [trainer] | 1.55 | 1.19 | 1.02 | 0.97 | 0.96 | 0.92 | 0.89 | 1.13 | 0.64 | 0.66 | 0.65 | 0.75 | 0.59 | 0.60 | 0.68 |
| bzw3 self-play | 20260601-11-bzw3-31 @ 467099 [champion] | 1.29 | 1.07 | 0.98 | 0.99 | 0.99 | 1.00 | 0.97 | 1.04 | 0.73 | 0.77 | 0.74 | 0.83 | 0.71 | 0.71 | 0.78 |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-23 @ 532369 [trainer] | 1.11 | 1.05 | 1.00 | 1.00 | 0.96 | 0.96 | 0.98 | 1.09 | 0.97 | 1.01 | 1.01 | 0.99 | 1.03 | 0.97 | 1.02 |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-22 @ 532369 [champion] | 1.11 | 1.05 | 1.00 | 1.00 | 0.96 | 0.96 | 0.98 | 1.09 | 0.97 | 1.01 | 1.01 | 0.99 | 1.03 | 0.97 | 1.02 |
| sMe9 self-play (fp32) | 20260525-1-sMe9-33 @ 373416 [trainer] | 0.92 | 0.95 | 1.03 | 1.02 | 1.08 | 1.02 | 1.00 | 0.91 | 0.95 | 0.94 | 0.93 | 0.92 | 0.96 | 0.95 | 0.98 |
| sMe9 self-play (fp32) | 20260525-1-sMe9-32 @ 373416 [champion] | 0.92 | 0.95 | 1.03 | 1.02 | 1.08 | 1.02 | 1.00 | 0.91 | 0.97 | 0.95 | 0.94 | 0.93 | 0.97 | 0.96 | 0.98 |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-11 @ 106695 [trainer] | 1.00 | 1.06 | 0.98 | 0.96 | 1.02 | 1.05 | 0.99 | 0.94 | 0.89 | 0.88 | 0.91 | 0.86 | 0.89 | 0.92 | 0.98 |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-10 @ 106695 [champion] | 0.98 | 1.04 | 0.98 | 0.96 | 1.02 | 1.04 | 0.98 | 0.95 | 0.91 | 0.91 | 0.94 | 0.88 | 0.92 | 0.95 | 1.00 |
| LMGh self-play | 20260609-12-LMGh-4 @ 79135 [trainer] | 1.89 | 1.46 | 1.05 | 0.97 | 0.94 | 0.84 | 0.82 | 1.67 | 1.21 | 1.13 | 1.11 | 1.59 | 0.88 | 0.98 | 1.19 |
| LMGh self-play | 20260609-12-LMGh-3 @ 79135 [champion] | 1.89 | 1.46 | 1.05 | 0.97 | 0.94 | 0.84 | 0.82 | 1.67 | 1.21 | 1.13 | 1.11 | 1.59 | 0.88 | 0.98 | 1.19 |
| WjRY self-play | 20260609-14-WjRY-8 @ 98974 [trainer] | 1.49 | 1.08 | 0.93 | 0.92 | 0.98 | 1.01 | 1.01 | 1.05 | 1.07 | 0.95 | 0.97 | 1.04 | 1.00 | 0.96 | 1.07 |
| WjRY self-play | 20260609-14-WjRY-7 @ 98974 [champion] | 1.03 | 1.01 | 0.98 | 0.99 | 1.02 | 1.04 | 1.02 | 0.96 | 0.95 | 0.86 | 0.90 | 0.91 | 0.92 | 0.89 | 0.95 |

**Row relative change vs lineage reference, family mean (‖W−W_ref‖/‖W_ref‖)** — meaningful only where the reference is the lineage's fresh net or seed; for self-play rows the reference is simply the earliest surviving file of that run

| lineage | checkpoint | reference | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Kn | UP-N | UP-R | UP-B | UP-fwd | UP-capL | UP-capR | QP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SE scale+bias full-leaky fresh (never trained) | 20261001-23-Dmwe @ fresh |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 12000 | 20261001-42-2q0Q @ fresh | 0.87 | 0.82 | 0.69 | 0.75 | 0.68 | 0.61 | 0.45 | 0.66 | 0.27 | 0.27 | 0.27 | 0.52 | 0.17 | 0.12 | 0.51 |
| SE scale+bias s1 | 20260929-22-bWdy @ 33014 | 20260929-12-JZOe @ fresh | 1.04 | 0.97 | 0.79 | 0.85 | 0.79 | 0.71 | 0.59 | 0.77 | 0.38 | 0.39 | 0.37 | 0.61 | 0.27 | 0.25 | 0.63 |
| SE scale+bias s2 | 20260930-4-k98x @ 7282 | 20260930-1-H1Oq @ fresh | 0.91 | 0.78 | 0.67 | 0.74 | 0.66 | 0.59 | 0.44 | 0.70 | 0.26 | 0.21 | 0.27 | 0.47 | 0.15 | 0.11 | 0.47 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 33012 | 20260929-13-06yp @ fresh | 0.92 | 0.93 | 0.79 | 0.80 | 0.78 | 0.78 | 0.64 | 0.80 | 0.44 | 0.36 | 0.35 | 0.61 | 0.27 | 0.26 | 0.61 |
| SE attenuate-only s2 | 20260930-5-5TXu @ 7289 | 20260930-2-Gf9P @ fresh | 0.81 | 0.79 | 0.68 | 0.68 | 0.68 | 0.55 | 0.43 | 0.65 | 0.29 | 0.25 | 0.27 | 0.50 | 0.15 | 0.16 | 0.46 |
| SE none s1 | 20260929-24-834D @ 32036 | 20260929-18-D9is @ fresh | 0.96 | 0.88 | 0.83 | 0.83 | 0.76 | 0.71 | 0.59 | 0.83 | 0.32 | 0.38 | 0.32 | 0.50 | 0.26 | 0.27 | 0.59 |
| SE none s2 | 20260930-6-LkS6 @ 7019 | 20260930-3-V9zk @ fresh | 0.83 | 0.74 | 0.67 | 0.67 | 0.64 | 0.55 | 0.40 | 0.61 | 0.22 | 0.27 | 0.22 | 0.46 | 0.13 | 0.13 | 0.53 |
| SE zero-beta s1 | 20260930-9-RrGx @ 5030 | 20260930-7-crxN @ fresh | 0.87 | 0.81 | 0.68 | 0.73 | 0.65 | 0.59 | 0.43 | 0.65 | 0.26 | 0.26 | 0.26 | 0.51 | 0.16 | 0.12 | 0.49 |
| SE zero-beta s2 | 20260930-10-H51a @ 5004 | 20260930-8-8qyR @ fresh | 0.88 | 0.75 | 0.66 | 0.73 | 0.64 | 0.58 | 0.42 | 0.68 | 0.26 | 0.19 | 0.26 | 0.46 | 0.14 | 0.11 | 0.46 |
| v5 | 20260805-1-0pTW @ 2000 | 20260628-1-tWtk @ fresh | 6.12 | 5.90 | 5.80 | 5.86 | 5.82 | 5.85 | 5.74 | 5.92 | 5.42 | 5.84 | 5.66 | 5.64 | 5.27 | 6.01 | 5.43 |
| mini2b | 20260705-1-znR7 @ 114000 | 20260629-3-3MIV @ fresh | 1.44 | 1.28 | 1.10 | 1.13 | 1.14 | 1.12 | 0.99 | 1.01 | 0.88 | 0.88 | 0.93 | 0.98 | 0.87 | 0.84 | 0.92 |
| coxw | 20260709-1-avoB @ 277000 | 20260629-5-Coxw @ fresh | 1.21 | 0.98 | 0.97 | 0.95 | 0.98 | 0.93 | 0.95 | 0.90 | 0.88 | 0.90 | 0.90 | 0.90 | 0.88 | 0.89 | 0.89 |
| ykkk | 20260710-1-amlg @ 250803 | 20260630-1-YkKk @ fresh | 1.24 | 1.24 | 1.17 | 1.20 | 1.43 | 1.21 | 1.15 | 1.17 | 1.00 | 0.99 | 1.04 | 1.11 | 0.96 | 0.96 | 1.08 |
| nt8y | 20260708-4-kEiZ @ 21086 | 20260701-3-nT8Y @ fresh | 2.14 | 1.85 | 1.59 | 1.57 | 1.44 | 1.32 | 1.24 | 1.62 | 1.05 | 1.09 | 0.99 | 1.09 | 1.01 | 1.03 | 1.17 |
| qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0 @ 1397000 | 20260702-7-Qeu8 @ fresh | 3.10 | 3.06 | 2.98 | 2.99 | 2.99 | 2.85 | 2.93 | 3.03 | 2.91 | 2.84 | 2.92 | 2.90 | 2.93 | 2.85 | 2.83 |
| qeu8e (epoch branch) | 20260708-6-sFzi @ 88107 | 20260702-7-Qeu8 @ fresh | 1.64 | 1.38 | 1.24 | 1.17 | 1.16 | 1.06 | 0.97 | 1.16 | 0.79 | 0.78 | 0.82 | 0.85 | 0.79 | 0.75 | 0.92 |
| qeu8-1blk128 | 20260711-17-pycz @ 120000 | 20260711-16-VRR4 @ fresh | 1.26 | 1.00 | 0.88 | 0.85 | 0.82 | 0.79 | 0.69 | 0.89 | 0.56 | 0.60 | 0.53 | 0.65 | 0.52 | 0.52 | 0.67 |
| qeu8init sf100sl100 vs-UCI | 20260722-1-syxR @ 558000 | 20260702-7-Qeu8 @ fresh | 1.40 | 1.40 | 1.30 | 1.29 | 1.24 | 1.19 | 1.17 | 1.31 | 1.13 | 1.14 | 1.17 | 1.13 | 1.16 | 1.15 | 1.13 |
| Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy @ 20000 | 20260727-1-Ejp0 @ 681000 | 0.47 | 0.37 | 0.27 | 0.24 | 0.22 | 0.21 | 0.18 | 0.34 | 0.12 | 0.12 | 0.12 | 0.19 | 0.09 | 0.09 | 0.23 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-10 @ 197340 [trainer] | 20260727-1-Ejp0 @ 1300000 | 0.51 | 0.52 | 0.52 | 0.52 | 0.52 | 0.52 | 0.52 | 0.51 | 0.52 | 0.52 | 0.52 | 0.52 | 0.53 | 0.53 | 0.52 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-9 @ 197340 [champion] | 20260727-1-Ejp0 @ 1300000 | 0.49 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.51 | 0.51 | 0.51 | 0.50 | 0.51 | 0.51 | 0.51 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-68 @ 1186322 [trainer] | 20260727-1-Ejp0 @ 1300000 | 0.11 | 0.09 | 0.08 | 0.07 | 0.07 | 0.07 | 0.07 | 0.09 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.08 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-67 @ 1186322 [champion] | 20260727-1-Ejp0 @ 1300000 | 0.11 | 0.09 | 0.08 | 0.07 | 0.07 | 0.07 | 0.07 | 0.09 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.08 |
| bzw3 self-play | 20260601-11-bzw3-32 @ 467099 [trainer] | 20260601-11-bzw3-31 @ 467065 | 0.58 | 0.58 | 0.50 | 0.46 | 0.42 | 0.39 | 0.38 | 0.59 | 0.46 | 0.44 | 0.46 | 0.46 | 0.44 | 0.46 | 0.46 |
| bzw3 self-play | 20260601-11-bzw3-31 @ 467099 [champion] | 20260601-11-bzw3-31 @ 467065 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-23 @ 532369 [trainer] | 20260514-1-KbHZ-18 @ 494927 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.02 | 0.01 | 0.02 | 0.02 | 0.03 | 0.03 | 0.06 | 0.01 | 0.01 | 0.02 |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-22 @ 532369 [champion] | 20260514-1-KbHZ-18 @ 494927 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.02 | 0.01 | 0.02 | 0.02 | 0.03 | 0.03 | 0.06 | 0.01 | 0.01 | 0.02 |
| sMe9 self-play (fp32) | 20260525-1-sMe9-33 @ 373416 [trainer] | 20260525-1-sMe9-26 @ 197269 | 0.15 | 0.15 | 0.14 | 0.15 | 0.15 | 0.16 | 0.16 | 0.16 | 0.16 | 0.16 | 0.16 | 0.17 | 0.16 | 0.16 | 0.15 |
| sMe9 self-play (fp32) | 20260525-1-sMe9-32 @ 373416 [champion] | 20260525-1-sMe9-26 @ 197269 | 0.13 | 0.13 | 0.12 | 0.13 | 0.13 | 0.14 | 0.14 | 0.14 | 0.13 | 0.13 | 0.14 | 0.14 | 0.13 | 0.13 | 0.12 |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-11 @ 106695 [trainer] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-10 @ 106695 [champion] | 20260531-9-LWKa-11 @ 106695 | 0.14 | 0.11 | 0.09 | 0.10 | 0.09 | 0.08 | 0.08 | 0.09 | 0.06 | 0.05 | 0.06 | 0.07 | 0.05 | 0.05 | 0.05 |
| LMGh self-play | 20260609-12-LMGh-4 @ 79135 [trainer] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LMGh self-play | 20260609-12-LMGh-3 @ 79135 [champion] | 20260609-12-LMGh-4 @ 79135 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| WjRY self-play | 20260609-14-WjRY-8 @ 98974 [trainer] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| WjRY self-play | 20260609-14-WjRY-7 @ 98974 [champion] | 20260609-14-WjRY-8 @ 98974 | 0.77 | 0.70 | 0.57 | 0.48 | 0.43 | 0.38 | 0.30 | 0.78 | 0.40 | 0.39 | 0.38 | 0.44 | 0.35 | 0.39 | 0.42 |

<!-- end:families_latest -->

### Queen-style directions at lineage-latest (row norm / queen median, bias)

<!-- begin:directions_latest -->

| lineage | checkpoint | N norm/Q-med / bias | NE norm/Q-med / bias | E norm/Q-med / bias | SE norm/Q-med / bias | S norm/Q-med / bias | SW norm/Q-med / bias | W norm/Q-med / bias | NW norm/Q-med / bias |
|---|---|---|---|---|---|---|---|---|---|
| SE scale+bias full-leaky fresh (never trained) | 20261001-23-Dmwe @ fresh | 0.95 / +0.00 | 1.00 / +0.00 | 1.01 / +0.00 | 1.02 / +0.00 | 1.04 / +0.00 | 1.02 / +0.00 | 0.98 / +0.00 | 1.01 / +0.00 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 12000 | 1.14 / +0.09 | 0.99 / -0.01 | 1.03 / -0.02 | 0.94 / -0.01 | 0.97 / -0.03 | 0.92 / -0.02 | 1.01 / -0.02 | 1.01 / +0.02 |
| SE scale+bias s1 | 20260929-22-bWdy @ 33014 | 1.18 / +0.13 | 1.00 / -0.01 | 1.02 / -0.03 | 0.91 / -0.02 | 0.95 / -0.04 | 0.89 / -0.03 | 1.02 / -0.03 | 1.00 / +0.02 |
| SE scale+bias s2 | 20260930-4-k98x @ 7282 | 1.14 / +0.08 | 1.01 / +0.00 | 1.02 / +0.00 | 0.93 / -0.03 | 0.96 / -0.03 | 0.91 / -0.03 | 1.01 / -0.01 | 0.99 / -0.00 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 33012 | 1.15 / +0.09 | 0.98 / +0.01 | 1.01 / +0.00 | 0.90 / -0.04 | 0.95 / -0.03 | 0.88 / -0.03 | 1.00 / -0.01 | 1.01 / +0.01 |
| SE attenuate-only s2 | 20260930-5-5TXu @ 7289 | 1.15 / +0.07 | 0.99 / +0.00 | 1.01 / -0.03 | 0.93 / -0.01 | 0.96 / -0.03 | 0.90 / -0.02 | 1.02 / -0.01 | 1.00 / +0.01 |
| SE none s1 | 20260929-24-834D @ 32036 | 1.13 / +0.07 | 0.99 / +0.00 | 1.02 / -0.01 | 0.90 / -0.04 | 0.94 / -0.03 | 0.90 / -0.03 | 1.04 / -0.02 | 1.04 / +0.02 |
| SE none s2 | 20260930-6-LkS6 @ 7019 | 1.18 / +0.06 | 1.03 / +0.01 | 1.04 / -0.02 | 0.94 / -0.02 | 0.96 / -0.03 | 0.93 / -0.02 | 1.02 / -0.00 | 1.01 / +0.01 |
| SE zero-beta s1 | 20260930-9-RrGx @ 5030 | 1.13 / +0.09 | 0.99 / -0.01 | 1.02 / -0.02 | 0.94 / -0.01 | 0.97 / -0.03 | 0.91 / -0.02 | 1.00 / -0.02 | 1.00 / +0.01 |
| SE zero-beta s2 | 20260930-10-H51a @ 5004 | 1.13 / +0.08 | 1.01 / +0.00 | 1.02 / +0.00 | 0.94 / -0.03 | 0.96 / -0.03 | 0.91 / -0.02 | 1.02 / -0.01 | 0.99 / -0.01 |
| v5 | 20260805-1-0pTW @ 2000 | 1.02 / -0.94 | 1.01 / -0.84 | 1.01 / -0.79 | 1.00 / -0.76 | 1.00 / -0.78 | 1.00 / -0.77 | 1.01 / -0.80 | 1.01 / -0.81 |
| mini2b | 20260705-1-znR7 @ 114000 | 1.05 / -0.17 | 1.06 / -0.18 | 1.01 / -0.22 | 0.93 / -0.24 | 0.93 / -0.25 | 0.91 / -0.25 | 1.03 / -0.25 | 1.05 / -0.17 |
| coxw | 20260709-1-avoB @ 277000 | 1.21 / -0.10 | 1.03 / -0.22 | 1.06 / -0.20 | 0.92 / -0.24 | 0.98 / -0.24 | 0.90 / -0.23 | 1.02 / -0.23 | 0.99 / -0.23 |
| ykkk | 20260710-1-amlg @ 250803 | 1.14 / -0.31 | 1.04 / -0.27 | 1.04 / -0.25 | 0.90 / -0.32 | 0.86 / -0.28 | 0.91 / -0.30 | 1.04 / -0.21 | 0.96 / -0.28 |
| nt8y | 20260708-4-kEiZ @ 21086 | 1.25 / -0.15 | 1.12 / -0.23 | 1.08 / -0.22 | 0.96 / -0.24 | 0.95 / -0.27 | 0.94 / -0.25 | 1.05 / -0.23 | 1.10 / -0.21 |
| qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0 @ 1397000 | 1.02 / -1.15 | 1.01 / -1.00 | 1.01 / -1.01 | 0.99 / -0.95 | 0.99 / -0.93 | 0.99 / -0.94 | 1.01 / -0.99 | 1.01 / -1.01 |
| qeu8e (epoch branch) | 20260708-6-sFzi @ 88107 | 1.20 / +0.03 | 1.04 / -0.14 | 1.06 / -0.14 | 0.90 / -0.18 | 0.94 / -0.15 | 0.90 / -0.16 | 1.06 / -0.15 | 1.03 / -0.17 |
| qeu8-1blk128 | 20260711-17-pycz @ 120000 | 1.20 / -0.03 | 1.05 / -0.07 | 1.06 / -0.08 | 0.92 / -0.07 | 0.95 / -0.07 | 0.91 / -0.08 | 1.06 / -0.08 | 1.08 / -0.05 |
| qeu8init sf100sl100 vs-UCI | 20260722-1-syxR @ 558000 | 1.16 / -0.43 | 1.09 / -0.35 | 1.01 / -0.44 | 0.91 / -0.49 | 0.87 / -0.55 | 0.90 / -0.49 | 1.01 / -0.52 | 1.08 / -0.32 |
| Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy @ 20000 | 1.09 / -0.66 | 1.01 / -0.61 | 1.03 / -0.61 | 0.96 / -0.59 | 0.98 / -0.57 | 0.95 / -0.59 | 1.01 / -0.58 | 1.02 / -0.62 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-10 @ 197340 [trainer] | 1.06 / -1.11 | 1.02 / -0.97 | 1.02 / -0.97 | 0.99 / -0.92 | 0.99 / -0.90 | 0.99 / -0.91 | 1.02 / -0.95 | 1.03 / -0.98 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-9 @ 197340 [champion] | 1.06 / -1.11 | 1.02 / -0.97 | 1.02 / -0.97 | 0.99 / -0.92 | 0.99 / -0.90 | 0.99 / -0.91 | 1.02 / -0.95 | 1.03 / -0.98 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-68 @ 1186322 [trainer] | 1.04 / -1.09 | 1.01 / -0.94 | 1.01 / -0.95 | 0.99 / -0.90 | 0.99 / -0.88 | 0.99 / -0.89 | 1.01 / -0.93 | 1.02 / -0.96 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-67 @ 1186322 [champion] | 1.04 / -1.09 | 1.01 / -0.94 | 1.01 / -0.95 | 0.99 / -0.90 | 0.99 / -0.88 | 0.99 / -0.89 | 1.01 / -0.93 | 1.02 / -0.96 |
| bzw3 self-play | 20260601-11-bzw3-32 @ 467099 [trainer] | 1.15 / -0.16 | 1.10 / -0.21 | 1.05 / -0.26 | 0.98 / -0.32 | 1.02 / -0.31 | 1.02 / -0.25 | 1.07 / -0.21 | 1.03 / -0.30 |
| bzw3 self-play | 20260601-11-bzw3-31 @ 467099 [champion] | 1.18 / -0.08 | 1.07 / -0.17 | 1.02 / -0.21 | 0.97 / -0.21 | 1.00 / -0.21 | 1.01 / -0.18 | 1.05 / -0.17 | 1.02 / -0.21 |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-23 @ 532369 [trainer] | 1.05 / +0.01 | 1.01 / +0.01 | 1.03 / -0.01 | 1.03 / +0.00 | 0.97 / -0.01 | 1.00 / +0.00 | 1.00 / -0.01 | 1.01 / +0.00 |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-22 @ 532369 [champion] | 1.05 / +0.01 | 1.01 / +0.01 | 1.03 / -0.01 | 1.03 / +0.00 | 0.97 / -0.01 | 1.00 / +0.00 | 1.00 / -0.01 | 1.01 / +0.00 |
| sMe9 self-play (fp32) | 20260525-1-sMe9-33 @ 373416 [trainer] | 1.03 / -0.00 | 1.03 / -0.00 | 1.03 / -0.01 | 0.98 / -0.00 | 0.96 / -0.01 | 0.99 / -0.01 | 0.97 / +0.00 | 1.02 / -0.00 |
| sMe9 self-play (fp32) | 20260525-1-sMe9-32 @ 373416 [champion] | 1.03 / -0.00 | 1.03 / -0.00 | 1.03 / -0.01 | 0.99 / -0.00 | 0.96 / -0.00 | 0.99 / -0.01 | 0.97 / +0.00 | 1.02 / -0.00 |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-11 @ 106695 [trainer] | 1.12 / +0.08 | 0.99 / -0.02 | 0.99 / -0.03 | 0.96 / -0.04 | 0.99 / -0.03 | 0.96 / -0.05 | 0.99 / -0.03 | 1.02 / -0.00 |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-10 @ 106695 [champion] | 1.08 / +0.06 | 0.99 / -0.01 | 0.99 / -0.02 | 0.96 / -0.04 | 0.99 / -0.02 | 0.96 / -0.04 | 0.98 / -0.02 | 1.01 / -0.00 |
| LMGh self-play | 20260609-12-LMGh-4 @ 79135 [trainer] | 1.28 / +0.10 | 1.13 / -0.32 | 1.12 / -0.02 | 0.91 / -0.04 | 1.15 / -0.05 | 1.06 / +0.03 | 1.17 / +0.08 | 1.21 / +0.09 |
| LMGh self-play | 20260609-12-LMGh-3 @ 79135 [champion] | 1.28 / +0.10 | 1.13 / -0.32 | 1.12 / -0.02 | 0.91 / -0.04 | 1.15 / -0.05 | 1.06 / +0.03 | 1.17 / +0.08 | 1.21 / +0.09 |
| WjRY self-play | 20260609-14-WjRY-8 @ 98974 [trainer] | 1.24 / +0.12 | 1.08 / -0.06 | 1.08 / -0.05 | 1.00 / -0.06 | 1.05 / -0.03 | 1.03 / -0.03 | 1.10 / +0.02 | 1.08 / +0.01 |
| WjRY self-play | 20260609-14-WjRY-7 @ 98974 [champion] | 1.10 / +0.05 | 1.01 / +0.00 | 1.04 / -0.03 | 0.98 / -0.03 | 0.95 / -0.02 | 1.00 / -0.02 | 1.01 / -0.02 | 1.02 / -0.02 |

<!-- end:directions_latest -->

The full 76-row final-conv table for every lineage-latest is in `results/tables/final_conv_channels_latest.md`. Every pre-block channel and final-conv row for every detailed checkpoint is in `results/channels/<lineage>/*-pre.csv` and `*-final.csv`.

## C. Input usage of the final conv (and of the pre_conv)

<!-- begin:input_usage -->

| lineage | checkpoint | K | final-conv column norm min / median / max | min/median | cols < 10% median | weak AND dead/mostly-off | effective contribution (col norm × act std) min / median / max | tower channels barely read (pre_conv col < 10% median) | pre_conv col min/median |
|---|---|---|---|---|---|---|---|---|---|
| SE scale+bias full-leaky fresh (never trained) | 20261001-23-Dmwe @ fresh | 128 | 0.841 / 1.09 / 1.29 | 0.77 | 0 | 0 | 0.492 / 0.638 / 0.753 | 0 of 128 | 0.86 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 12000 | 128 | 0.857 / 1.27 / 1.73 | 0.67 | 0 | 0 | 0.475 / 0.807 / 1.44 | 0 of 128 | 0.84 |
| SE scale+bias s1 | 20260929-22-bWdy @ 33014 | 128 | 0.737 / 1.25 / 2.1 | 0.59 | 0 | 0 | 0.403 / 0.755 / 2.78 | 0 of 128 | 0.76 |
| SE scale+bias s2 | 20260930-4-k98x @ 7282 | 128 | 0.775 / 1.25 / 1.8 | 0.62 | 0 | 0 | 0.429 / 0.806 / 1.42 | 0 of 128 | 0.76 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 33012 | 128 | 0.741 / 1.29 / 1.78 | 0.57 | 0 | 0 | 0.423 / 0.789 / 1.55 | 0 of 128 | 0.76 |
| SE attenuate-only s2 | 20260930-5-5TXu @ 7289 | 128 | 0.835 / 1.25 / 1.78 | 0.67 | 0 | 0 | 0.457 / 0.808 / 1.43 | 0 of 128 | 0.85 |
| SE none s1 | 20260929-24-834D @ 32036 | 128 | 0.744 / 1.3 / 1.93 | 0.57 | 0 | 0 | 0.411 / 0.831 / 1.78 | 0 of 128 | 0.79 |
| SE none s2 | 20260930-6-LkS6 @ 7019 | 128 | 0.798 / 1.28 / 1.7 | 0.63 | 0 | 0 | 0.457 / 0.794 / 1.29 | 0 of 128 | 0.81 |
| SE zero-beta s1 | 20260930-9-RrGx @ 5030 | 128 | 0.857 / 1.26 / 1.74 | 0.68 | 0 | 0 | 0.474 / 0.775 / 1.39 | 0 of 128 | 0.82 |
| SE zero-beta s2 | 20260930-10-H51a @ 5004 | 128 | 0.755 / 1.25 / 1.8 | 0.61 | 0 | 0 | 0.409 / 0.792 / 1.44 | 0 of 128 | 0.77 |
| v5 | 20260805-1-0pTW @ 2000 | 128 | 0.876 / 4.03 / 17.4 | 0.22 | 0 | 0 | 0.189 / 8.09 / 50.2 | 9 of 128 | 0.05 |
| mini2b | 20260705-1-znR7 @ 114000 | 128 | 0.817 / 1.32 / 2.75 | 0.62 | 0 | 0 | 0.369 / 0.821 / 5.51 | 0 of 128 | 0.44 |
| coxw | 20260709-1-avoB @ 277000 | 128 | 0.495 / 0.968 / 2.14 | 0.51 | 0 | 0 | 0.286 / 0.752 / 4.82 | 0 of 128 | 0.31 |
| ykkk | 20260710-1-amlg @ 250803 | 64 | 0.438 / 1.57 / 2.59 | 0.28 | 0 | n/a | n/a | n/a | n/a |
| nt8y | 20260708-4-kEiZ @ 21086 | 512 | 0.298 / 0.633 / 2.91 | 0.47 | 0 | 0 | 0.0695 / 0.351 / 6.94 | 0 of 32 | 0.84 |
| qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0 @ 1397000 | 512 | 0.13 / 0.853 / 8.08 | 0.15 | 0 | 0 | 0.0372 / 1.17 / 7.39 | 0 of 64 | 0.12 |
| qeu8e (epoch branch) | 20260708-6-sFzi @ 88107 | 512 | 0.335 / 0.574 / 2.16 | 0.58 | 0 | 0 | 0.181 / 0.344 / 4.35 | 0 of 64 | 0.77 |
| qeu8-1blk128 | 20260711-17-pycz @ 120000 | 512 | 0.302 / 0.505 / 1.83 | 0.60 | 0 | 0 | 0.168 / 0.3 / 2.72 | 0 of 128 | 0.63 |
| qeu8init sf100sl100 vs-UCI | 20260722-1-syxR @ 558000 | 512 | 0.143 / 0.368 / 1.14 | 0.39 | 0 | 0 | 0.0595 / 0.332 / 2.49 | 0 of 64 | 0.53 |
| Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy @ 20000 | 512 | 0.212 / 0.739 / 2.36 | 0.29 | 0 | 0 | 0.0528 / 0.697 / 4.43 | 2 of 64 | 0.09 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-10 @ 197340 [trainer] | 512 | 0.0865 / 0.461 / 3.46 | 0.19 | 0 | 0 | 0.0237 / 0.61 / 3.28 | 0 of 64 | 0.14 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-9 @ 197340 [champion] | 512 | 0.0885 / 0.468 / 3.6 | 0.19 | 0 | 0 | 0.0242 / 0.625 / 3.37 | 0 of 64 | 0.14 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-68 @ 1186322 [trainer] | 512 | 0.148 / 0.833 / 6.75 | 0.18 | 0 | 0 | 0.0398 / 1.11 / 7.85 | 0 of 64 | 0.12 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-67 @ 1186322 [champion] | 512 | 0.148 / 0.833 / 6.75 | 0.18 | 0 | 0 | 0.0398 / 1.11 / 7.85 | 0 of 64 | 0.12 |
| bzw3 self-play | 20260601-11-bzw3-32 @ 467099 [trainer] | 128 | 0.914 / 1.72 / 4.14 | 0.53 | 0 | 0 | 0.191 / 0.61 / 10.3 | 0 of 128 | 0.19 |
| bzw3 self-play | 20260601-11-bzw3-31 @ 467099 [champion] | 128 | 0.892 / 1.44 / 3.06 | 0.62 | 0 | 0 | 0.388 / 0.807 / 6.53 | 0 of 128 | 0.25 |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-23 @ 532369 [trainer] | 128 | 0.836 / 1.1 / 1.27 | 0.76 | 0 | n/a | n/a | n/a | n/a |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-22 @ 532369 [champion] | 128 | 0.837 / 1.1 / 1.27 | 0.76 | 0 | n/a | n/a | n/a | n/a |
| sMe9 self-play (fp32) | 20260525-1-sMe9-33 @ 373416 [trainer] | 128 | 0.743 / 0.916 / 1.42 | 0.81 | 0 | n/a | n/a | n/a | n/a |
| sMe9 self-play (fp32) | 20260525-1-sMe9-32 @ 373416 [champion] | 128 | 0.763 / 0.937 / 1.44 | 0.81 | 0 | n/a | n/a | n/a | n/a |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-11 @ 106695 [trainer] | 128 | 0.875 / 1.23 / 1.87 | 0.71 | 0 | 0 | 0.457 / 0.77 / 2.17 | 0 of 128 | 0.70 |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-10 @ 106695 [champion] | 128 | 0.85 / 1.23 / 1.69 | 0.69 | 0 | 0 | 0.454 / 0.758 / 1.56 | 0 of 128 | 0.73 |
| LMGh self-play | 20260609-12-LMGh-4 @ 79135 [trainer] | 32 | 2.14 / 3.75 / 5.12 | 0.57 | 0 | 0 | 0.557 / 4.68 / 12.6 | 0 of 32 | 0.45 |
| LMGh self-play | 20260609-12-LMGh-3 @ 79135 [champion] | 32 | 2.14 / 3.75 / 5.12 | 0.57 | 0 | 0 | 0.557 / 4.68 / 12.6 | 0 of 32 | 0.45 |
| WjRY self-play | 20260609-14-WjRY-8 @ 98974 [trainer] | 128 | 0.917 / 1.46 / 2.35 | 0.63 | 0 | 0 | 0.346 / 0.998 / 3.56 | 0 of 128 | 0.68 |
| WjRY self-play | 20260609-14-WjRY-7 @ 98974 [champion] | 128 | 0.843 / 1.24 / 1.74 | 0.68 | 0 | 0 | 0.469 / 0.735 / 1.52 | 0 of 128 | 0.69 |

<!-- end:input_usage -->

- No final-conv column falls under 10% of the median, and no channel is dead, so there is nothing to cross-check.
- The lowest min/median column ratios are on the offset lines (Ejp0 0.15, run 1 0.19, v5 0.22). The hot always-on columns inflate the median there.
- On v5, the pre_conv barely reads 9 of the 128 tower channels (cols 14, 31, 32, 35, 39, 56, 75, 77, 87), up from 2 at h7vI @ 336,610. This points at quiet tower outputs, outside the head.

## D. Leaky-FC1 vs its ReLU comparator (2q0Q is JZOe re-stamped; all 7 policy tensors bit-identical)

<!-- begin:leaky_vs_relu -->

| step | run | checkpoint | dead | mostly-off | always-on | median abs(gamma) | beta range | running var min / median / max | pre_conv row norm median | final row norm median | mean-row ratio | bias mean | bias std | static shared level | underpromo/Q | row rel. change vs fresh (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fresh | leaky-FC1 | 20261001-42-2q0Q @ fresh | 0 | 0 | 0 | 1.000 | +0.000..+0.000 | 0.391 / 0.664 / 1.7 | 1.423 | 1.408 | 0.105 | +0.0000 | 0.000 | -0.053 | 0.984 |  |
| fresh | ReLU s1 | 20260929-12-JZOe @ fresh | 0 | 0 | 0 | 1.000 | +0.000..+0.000 | 0.391 / 0.664 / 1.7 | 1.423 | 1.408 | 0.105 | +0.0000 | 0.000 | -0.053 | 0.984 |  |
| 1000 | leaky-FC1 | 20261001-43-NbWz @ 1000 | 0 | 0 | 0 | 1.042 | -0.065..+0.185 | 0.459 / 0.79 / 2.18 | 1.417 | 1.546 | 0.181 | +0.0000 | 0.051 | +0.422 | 0.872 | 0.408 |
| 1000 | ReLU s1 | 20260929-22-bWdy @ 1000 | 0 | 0 | 0 | 1.043 | -0.068..+0.184 | 0.449 / 0.795 / 2.25 | 1.417 | 1.543 | 0.180 | -0.0000 | 0.052 | +0.411 | 0.876 | 0.413 |
| 5000 | leaky-FC1 | 20261001-43-NbWz @ 5000 | 0 | 0 | 0 | 1.040 | -0.124..+0.298 | 0.452 / 0.918 / 3.07 | 1.381 | 1.668 | 0.202 | -0.0000 | 0.087 | +0.549 | 0.779 | 0.631 |
| 5000 | ReLU s1 | 20260929-22-bWdy @ 5000 | 0 | 0 | 0 | 1.047 | -0.104..+0.299 | 0.432 / 0.895 / 3.11 | 1.378 | 1.668 | 0.202 | -0.0000 | 0.085 | +0.591 | 0.780 | 0.634 |
| 10000 | leaky-FC1 | 20261001-43-NbWz @ 10000 | 0 | 0 | 0 | 1.058 | -0.119..+0.315 | 0.438 / 0.885 / 3 | 1.371 | 1.680 | 0.200 | -0.0000 | 0.089 | +0.565 | 0.770 | 0.646 |
| 10000 | ReLU s1 | 20260929-22-bWdy @ 10000 | 0 | 0 | 0 | 1.059 | -0.104..+0.318 | 0.422 / 0.855 / 3.08 | 1.368 | 1.684 | 0.201 | -0.0001 | 0.087 | +0.608 | 0.773 | 0.647 |
| 11000 | leaky-FC1 | 20261001-43-NbWz @ 11000 | 0 | 0 | 0 | 1.059 | -0.119..+0.316 | 0.436 / 0.883 / 3.01 | 1.371 | 1.680 | 0.200 | -0.0000 | 0.089 | +0.566 | 0.770 | 0.648 |
| 11000 | ReLU s1 | 20260929-22-bWdy @ 11000 | 0 | 0 | 0 | 1.059 | -0.104..+0.318 | 0.42 / 0.859 / 3.08 | 1.368 | 1.686 | 0.201 | -0.0000 | 0.087 | +0.610 | 0.774 | 0.648 |
| 12000 | leaky-FC1 | 20261001-43-NbWz @ 12000 | 0 | 0 | 0 | 1.060 | -0.119..+0.317 | 0.436 / 0.88 / 3.01 | 1.370 | 1.681 | 0.200 | -0.0000 | 0.089 | +0.567 | 0.769 | 0.649 |
| 12000 | ReLU s1 | 20260929-22-bWdy @ 12000 | 0 | 0 | 0 | 1.059 | -0.103..+0.320 | 0.42 / 0.854 / 3.06 | 1.367 | 1.686 | 0.201 | -0.0000 | 0.087 | +0.611 | 0.773 | 0.649 |

**Distance between the two runs relative to how far the ReLU run moved from the shared fresh net** (‖leaky − ReLU‖ / ‖ReLU − fresh‖ per tensor; 0 = identical, 1 = as different as the training displacement itself), plus per-row cosine of the final conv and bias correlation:

| step | policy.pre_conv.weight | policy.pre_bn.weight | policy.pre_bn.bias | policy.pre_bn.running_mean | policy.pre_bn.running_var | policy.conv.weight | policy.conv.bias | final-conv row cosine min / median | bias corr |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | 0.29 | 0.13 | 0.11 | 0.24 | 0.14 | 0.16 | 0.04 | 0.992 / 0.999 | 0.9994 |
| 5000 | 0.36 | 0.17 | 0.15 | 0.31 | 0.16 | 0.22 | 0.04 | 0.976 / 0.994 | 0.9993 |
| 10000 | 0.36 | 0.17 | 0.15 | 0.30 | 0.16 | 0.23 | 0.04 | 0.974 / 0.993 | 0.9992 |
| 11000 | 0.36 | 0.17 | 0.15 | 0.30 | 0.17 | 0.23 | 0.04 | 0.975 / 0.993 | 0.9992 |
| 12000 | 0.37 | 0.16 | 0.15 | 0.30 | 0.16 | 0.23 | 0.04 | 0.974 / 0.993 | 0.9992 |

**Context: the same distance for SE zero-beta s1 (fresh net derived from the same JZOe weights, SE β init changed) vs ReLU s1** — how far a different single change moves the policy head:

| step | policy.pre_conv.weight | policy.pre_bn.weight | policy.pre_bn.bias | policy.pre_bn.running_mean | policy.pre_bn.running_var | policy.conv.weight | policy.conv.bias |
|---|---|---|---|---|---|---|---|
| 1000 | 0.48 | 0.24 | 0.16 | 0.43 | 0.27 | 0.28 | 0.08 |
| 5000 | 0.59 | 0.31 | 0.26 | 0.46 | 0.30 | 0.37 | 0.07 |

**Optimizer velocity, leaky-FC1 vs the ReLU seed-2 run (the only ReLU scale+bias run whose checkpoints carry velocity)**

| step | run | checkpoint | zero-velocity gamma+beta channels | zero-velocity pre rows | zero-velocity final rows | pre row velocity norm min / median / max | median abs velocity gamma / beta | final row velocity norm min / median / max | cos(velocity, weight) range, pre rows |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | leaky-FC1 | 20261001-43-NbWz @ 1000 | 0 | 0 | 0 | 0.0034 / 0.023 / 0.0508 | 0.00307 / 0.00157 | 0.000195 / 0.0157 / 0.257 | -0.0412..-0.0013 |
| 1000 | ReLU s2 | 20260930-4-k98x @ 1000 | 0 | 0 | 0 | 0.0053 / 0.0321 / 0.0649 | 0.00393 / 0.00209 | 0.000425 / 0.0209 / 0.206 | -0.0728..-0.0039 |
| 2000 | leaky-FC1 | 20261001-43-NbWz @ 2000 | 0 | 0 | 0 | 0.00297 / 0.0235 / 0.0822 | 0.00352 / 0.00138 | 6.73e-05 / 0.0197 / 0.19 | -0.0235..-0.0010 |
| 3000 | leaky-FC1 | 20261001-43-NbWz @ 3000 | 0 | 0 | 0 | 0.00175 / 0.0226 / 0.0423 | 0.00249 / 0.00105 | 0.000154 / 0.019 / 0.165 | -0.0211..-0.0008 |
| 4000 | leaky-FC1 | 20261001-43-NbWz @ 4000 | 0 | 0 | 0 | 0.00204 / 0.0272 / 0.0656 | 0.00383 / 0.00116 | 0.000221 / 0.0187 / 0.231 | -0.0113..-0.0001 |
| 5000 | leaky-FC1 | 20261001-43-NbWz @ 5000 | 0 | 0 | 0 | 0.00243 / 0.0302 / 0.0886 | 0.00403 / 0.00196 | 8.51e-05 / 0.0196 / 0.191 | -0.0088..-0.0003 |
| 5000 | ReLU s2 | 20260930-4-k98x @ 5000 | 0 | 0 | 0 | 0.00251 / 0.025 / 0.0553 | 0.00367 / 0.00128 | 0.000108 / 0.0186 / 0.127 | -0.0082..+0.0002 |
| 6000 | leaky-FC1 | 20261001-43-NbWz @ 6000 | 0 | 0 | 0 | 0.00158 / 0.0259 / 0.0521 | 0.00376 / 0.00139 | 4.08e-05 / 0.0248 / 0.17 | -0.0062..+0.0007 |
| 7000 | leaky-FC1 | 20261001-43-NbWz @ 7000 | 0 | 0 | 0 | 0.00211 / 0.0319 / 0.0651 | 0.00455 / 0.00173 | 0.000158 / 0.0203 / 0.217 | -0.0036..+0.0004 |
| 8000 | leaky-FC1 | 20261001-43-NbWz @ 8000 | 0 | 0 | 0 | 0.00248 / 0.0357 / 0.0814 | 0.00462 / 0.00184 | 0.000238 / 0.0226 / 0.266 | -0.0025..+0.0013 |
| 9000 | leaky-FC1 | 20261001-43-NbWz @ 9000 | 0 | 0 | 0 | 0.0023 / 0.0337 / 0.0761 | 0.00527 / 0.00181 | 0.000507 / 0.0281 / 0.272 | -0.0028..+0.0012 |
| 10000 | leaky-FC1 | 20261001-43-NbWz @ 10000 | 0 | 0 | 0 | 0.00233 / 0.0473 / 0.126 | 0.0068 / 0.00283 | 0.000457 / 0.0338 / 0.342 | -0.0023..+0.0004 |
| 11000 | leaky-FC1 | 20261001-43-NbWz @ 11000 | 0 | 0 | 0 | 0.00241 / 0.0414 / 0.0944 | 0.00571 / 0.002 | 0.000465 / 0.0431 / 0.219 | -0.0037..+0.0006 |
| 12000 | leaky-FC1 | 20261001-43-NbWz @ 12000 | 0 | 0 | 0 | 0.00219 / 0.0404 / 0.0905 | 0.00689 / 0.00238 | 0.000394 / 0.028 / 0.311 | -0.0022..+0.0006 |

<!-- end:leaky_vs_relu -->

- Every summary statistic is within run-to-run noise at every matched step.
- Neither run has any dead, mostly-off or always-on channels, and neither has zero velocity anywhere.
- The leaky run's static level runs ~0.04 lower (+0.567 vs +0.611 at 12k), which is immaterial.
- In both runs, underpromotion captures and long diagonals have the smallest final-row velocity: ≈ 4e-4 vs a median of 0.028.

## SE-style experiment, all arms and seeds

<!-- begin:se_arms -->

| run | checkpoint | dead | mostly-off | always-on | flat | median abs(gamma) | beta range | running var min / median / max | max abs(mu)/sigma | final row norm median | mean-row ratio | bias mean | static shared level | underpromo/Q | knight/Q | min row rel. change vs fresh | max abs W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SE scale+bias full-leaky fresh (never trained) | 20261001-23-Dmwe @ fresh | 0 | 0 | 0 | 0 | 1.000 | +0.000..+0.000 | 0.391 / 0.664 / 1.7 | 2.50 | 1.408 | 0.105 | +0.0000 | -0.052 | 0.984 | 1.029 |  | 0.477 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-42-2q0Q @ fresh | 0 | 0 | 0 | 0 | 1.000 | +0.000..+0.000 | 0.391 / 0.664 / 1.7 | 2.50 | 1.408 | 0.105 | +0.0000 | -0.053 | 0.984 | 1.029 |  | 0.477 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 1000 | 0 | 0 | 0 | 0 | 1.042 | -0.065..+0.185 | 0.459 / 0.79 / 2.18 | 1.88 | 1.546 | 0.181 | +0.0000 | +0.422 | 0.872 | 1.004 | 0.047 | 0.633 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 2000 | 0 | 0 | 0 | 0 | 1.032 | -0.106..+0.220 | 0.488 / 0.923 / 2.87 | 1.79 | 1.618 | 0.198 | -0.0000 | +0.514 | 0.827 | 0.980 | 0.075 | 0.737 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 3000 | 0 | 0 | 0 | 0 | 1.028 | -0.125..+0.258 | 0.475 / 0.943 / 3.02 | 1.74 | 1.650 | 0.201 | -0.0000 | +0.530 | 0.802 | 0.971 | 0.090 | 0.799 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 4000 | 0 | 0 | 0 | 0 | 1.034 | -0.128..+0.283 | 0.479 / 0.931 / 3.1 | 1.69 | 1.660 | 0.202 | -0.0000 | +0.541 | 0.786 | 0.966 | 0.103 | 0.934 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 5000 | 0 | 0 | 0 | 0 | 1.040 | -0.124..+0.298 | 0.452 / 0.918 / 3.07 | 1.67 | 1.668 | 0.202 | -0.0000 | +0.549 | 0.779 | 0.966 | 0.110 | 1.002 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 6000 | 0 | 0 | 0 | 0 | 1.048 | -0.123..+0.306 | 0.451 / 0.898 / 3.02 | 1.68 | 1.672 | 0.201 | -0.0000 | +0.556 | 0.774 | 0.965 | 0.114 | 1.036 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 7000 | 0 | 0 | 0 | 0 | 1.052 | -0.121..+0.310 | 0.449 / 0.899 / 3.03 | 1.66 | 1.675 | 0.201 | -0.0000 | +0.560 | 0.772 | 0.966 | 0.117 | 1.053 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 8000 | 0 | 0 | 0 | 0 | 1.056 | -0.121..+0.313 | 0.442 / 0.892 / 3.02 | 1.66 | 1.677 | 0.200 | -0.0000 | +0.562 | 0.771 | 0.967 | 0.118 | 1.063 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 9000 | 0 | 0 | 0 | 0 | 1.057 | -0.120..+0.314 | 0.437 / 0.89 / 3.01 | 1.67 | 1.679 | 0.200 | -0.0000 | +0.564 | 0.771 | 0.967 | 0.119 | 1.070 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 10000 | 0 | 0 | 0 | 0 | 1.058 | -0.119..+0.315 | 0.438 / 0.885 / 3 | 1.67 | 1.680 | 0.200 | -0.0000 | +0.565 | 0.770 | 0.967 | 0.119 | 1.075 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 11000 | 0 | 0 | 0 | 0 | 1.059 | -0.119..+0.316 | 0.436 / 0.883 / 3.01 | 1.67 | 1.680 | 0.200 | -0.0000 | +0.566 | 0.770 | 0.967 | 0.119 | 1.080 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 12000 | 0 | 0 | 0 | 0 | 1.060 | -0.119..+0.317 | 0.436 / 0.88 / 3.01 | 1.67 | 1.681 | 0.200 | -0.0000 | +0.567 | 0.769 | 0.967 | 0.120 | 1.084 |
| SE scale+bias s1 | 20260929-12-JZOe @ fresh | 0 | 0 | 0 | 0 | 1.000 | +0.000..+0.000 | 0.391 / 0.664 / 1.7 | 2.50 | 1.408 | 0.105 | +0.0000 | -0.053 | 0.984 | 1.029 |  | 0.477 |
| SE scale+bias s1 | 20260929-22-bWdy @ 1000 | 0 | 0 | 0 | 0 | 1.043 | -0.068..+0.184 | 0.449 / 0.795 / 2.25 | 1.87 | 1.543 | 0.180 | -0.0000 | +0.411 | 0.876 | 1.008 | 0.046 | 0.629 |
| SE scale+bias s1 | 20260929-22-bWdy @ 5000 | 0 | 0 | 0 | 0 | 1.047 | -0.104..+0.299 | 0.432 / 0.895 / 3.11 | 1.64 | 1.668 | 0.202 | -0.0000 | +0.591 | 0.780 | 0.965 | 0.113 | 1.023 |
| SE scale+bias s1 | 20260929-22-bWdy @ 10000 | 0 | 0 | 0 | 0 | 1.059 | -0.104..+0.318 | 0.422 / 0.855 / 3.08 | 1.60 | 1.684 | 0.201 | -0.0001 | +0.608 | 0.773 | 0.969 | 0.123 | 1.094 |
| SE scale+bias s1 | 20260929-22-bWdy @ 11000 | 0 | 0 | 0 | 0 | 1.059 | -0.104..+0.318 | 0.42 / 0.859 / 3.08 | 1.61 | 1.686 | 0.201 | -0.0000 | +0.610 | 0.774 | 0.970 | 0.123 | 1.094 |
| SE scale+bias s1 | 20260929-22-bWdy @ 12000 | 0 | 0 | 0 | 0 | 1.059 | -0.103..+0.320 | 0.42 / 0.854 / 3.06 | 1.61 | 1.686 | 0.201 | -0.0000 | +0.611 | 0.773 | 0.969 | 0.124 | 1.102 |
| SE scale+bias s1 | 20260929-22-bWdy @ 20000 | 0 | 0 | 0 | 0 | 1.039 | -0.129..+0.350 | 0.418 / 0.848 / 2.88 | 1.57 | 1.671 | 0.208 | -0.0000 | +0.578 | 0.740 | 0.954 | 0.159 | 1.367 |
| SE scale+bias s1 | 20260929-22-bWdy @ 33014 | 0 | 0 | 0 | 0 | 1.031 | -0.170..+0.428 | 0.389 / 0.777 / 2.94 | 2.07 | 1.688 | 0.219 | -0.0000 | +0.591 | 0.670 | 0.952 | 0.238 | 1.656 |
| SE scale+bias s2 | 20260930-1-H1Oq @ fresh | 0 | 0 | 0 | 0 | 1.000 | +0.000..+0.000 | 0.35 / 0.615 / 1.53 | 1.97 | 1.416 | 0.116 | +0.0000 | -0.073 | 0.985 | 0.969 |  | 0.479 |
| SE scale+bias s2 | 20260930-4-k98x @ 1000 | 0 | 0 | 0 | 0 | 1.035 | -0.045..+0.131 | 0.447 / 0.739 / 1.99 | 1.60 | 1.549 | 0.189 | +0.0000 | +0.388 | 0.885 | 1.009 | 0.030 | 0.622 |
| SE scale+bias s2 | 20260930-4-k98x @ 5000 | 0 | 0 | 0 | 0 | 1.063 | -0.093..+0.219 | 0.461 / 0.877 / 2.5 | 1.57 | 1.685 | 0.207 | +0.0001 | +0.509 | 0.783 | 0.992 | 0.101 | 0.786 |
| SE scale+bias s2 | 20260930-4-k98x @ 7282 | 0 | 0 | 0 | 0 | 1.077 | -0.095..+0.229 | 0.429 / 0.851 / 2.47 | 1.53 | 1.690 | 0.206 | +0.0001 | +0.524 | 0.774 | 0.990 | 0.107 | 0.805 |
| SE attenuate-only s1 | 20260929-13-06yp @ fresh | 0 | 0 | 0 | 0 | 1.000 | +0.000..+0.000 | 0.357 / 0.6 / 1.11 | 1.96 | 1.393 | 0.121 | +0.0000 | +0.024 | 0.972 | 0.995 |  | 0.500 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 1000 | 0 | 0 | 0 | 0 | 1.051 | -0.034..+0.181 | 0.367 / 0.742 / 2.12 | 1.65 | 1.542 | 0.191 | +0.0000 | +0.544 | 0.868 | 1.023 | 0.073 | 0.602 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 5000 | 0 | 0 | 0 | 0 | 1.059 | -0.082..+0.287 | 0.414 / 0.844 / 2.83 | 1.62 | 1.680 | 0.210 | +0.0001 | +0.765 | 0.759 | 0.984 | 0.114 | 0.898 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 10000 | 0 | 0 | 0 | 0 | 1.070 | -0.082..+0.301 | 0.396 / 0.828 / 2.81 | 1.61 | 1.686 | 0.208 | +0.0001 | +0.791 | 0.752 | 0.987 | 0.122 | 0.934 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 11000 | 0 | 0 | 0 | 0 | 1.074 | -0.082..+0.301 | 0.398 / 0.828 / 2.81 | 1.61 | 1.687 | 0.208 | +0.0001 | +0.793 | 0.751 | 0.987 | 0.122 | 0.938 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 12000 | 0 | 0 | 0 | 0 | 1.074 | -0.082..+0.303 | 0.4 / 0.83 / 2.81 | 1.62 | 1.688 | 0.208 | +0.0001 | +0.795 | 0.751 | 0.986 | 0.122 | 0.938 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 20000 | 0 | 0 | 0 | 0 | 1.059 | -0.099..+0.314 | 0.424 / 0.809 / 2.72 | 1.73 | 1.672 | 0.215 | +0.0001 | +0.812 | 0.716 | 0.972 | 0.163 | 1.109 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 33012 | 0 | 0 | 0 | 0 | 1.062 | -0.131..+0.352 | 0.406 / 0.705 / 2.7 | 2.48 | 1.709 | 0.225 | +0.0000 | +0.855 | 0.656 | 0.988 | 0.239 | 1.398 |
| SE attenuate-only s2 | 20260930-2-Gf9P @ fresh | 0 | 0 | 0 | 0 | 1.000 | +0.000..+0.000 | 0.352 / 0.637 / 1.42 | 1.92 | 1.427 | 0.123 | +0.0000 | +0.064 | 1.005 | 1.013 |  | 0.494 |
| SE attenuate-only s2 | 20260930-5-5TXu @ 1000 | 0 | 0 | 0 | 0 | 1.043 | -0.063..+0.144 | 0.395 / 0.769 / 1.71 | 1.93 | 1.574 | 0.189 | +0.0000 | +0.535 | 0.892 | 1.034 | 0.047 | 0.638 |
| SE attenuate-only s2 | 20260930-5-5TXu @ 5000 | 0 | 0 | 0 | 0 | 1.059 | -0.135..+0.192 | 0.437 / 0.833 / 2.52 | 1.81 | 1.683 | 0.211 | +0.0000 | +0.679 | 0.794 | 1.012 | 0.097 | 0.798 |
| SE attenuate-only s2 | 20260930-5-5TXu @ 7289 | 0 | 0 | 0 | 0 | 1.072 | -0.133..+0.202 | 0.429 / 0.818 / 2.54 | 1.85 | 1.685 | 0.211 | +0.0000 | +0.690 | 0.787 | 1.011 | 0.103 | 0.826 |
| SE none s1 | 20260929-18-D9is @ fresh | 0 | 0 | 0 | 0 | 1.000 | +0.000..+0.000 | 0.314 / 0.641 / 1.42 | 3.07 | 1.398 | 0.108 | +0.0000 | +0.064 | 1.011 | 1.012 |  | 0.520 |
| SE none s1 | 20260929-24-834D @ 1000 | 0 | 0 | 0 | 0 | 1.055 | -0.047..+0.169 | 0.504 / 0.764 / 2.27 | 1.65 | 1.550 | 0.177 | -0.0000 | +0.545 | 0.895 | 1.027 | 0.045 | 0.652 |
| SE none s1 | 20260929-24-834D @ 5000 | 0 | 0 | 0 | 0 | 1.070 | -0.119..+0.207 | 0.508 / 0.793 / 2.33 | 2.38 | 1.677 | 0.203 | -0.0000 | +0.721 | 0.787 | 1.013 | 0.109 | 0.809 |
| SE none s1 | 20260929-24-834D @ 10000 | 0 | 0 | 0 | 0 | 1.086 | -0.120..+0.223 | 0.467 / 0.773 / 2.23 | 2.53 | 1.690 | 0.202 | -0.0000 | +0.748 | 0.776 | 1.013 | 0.115 | 0.816 |
| SE none s1 | 20260929-24-834D @ 11000 | 0 | 0 | 0 | 0 | 1.086 | -0.120..+0.224 | 0.465 / 0.775 / 2.23 | 2.54 | 1.692 | 0.202 | -0.0000 | +0.750 | 0.776 | 1.014 | 0.115 | 0.816 |
| SE none s1 | 20260929-24-834D @ 12000 | 0 | 0 | 0 | 0 | 1.094 | -0.120..+0.224 | 0.461 / 0.773 / 2.22 | 2.55 | 1.692 | 0.202 | +0.0000 | +0.750 | 0.775 | 1.014 | 0.115 | 0.816 |
| SE none s1 | 20260929-24-834D @ 20000 | 0 | 0 | 0 | 0 | 1.078 | -0.140..+0.238 | 0.383 / 0.75 / 2.25 | 2.55 | 1.676 | 0.211 | -0.0000 | +0.759 | 0.743 | 1.005 | 0.158 | 1.047 |
| SE none s1 | 20260929-24-834D @ 32036 | 0 | 0 | 0 | 0 | 1.086 | -0.181..+0.264 | 0.262 / 0.668 / 2.19 | 2.86 | 1.723 | 0.223 | -0.0001 | +0.786 | 0.670 | 1.009 | 0.241 | 1.375 |
| SE none s2 | 20260930-3-V9zk @ fresh | 0 | 0 | 0 | 0 | 1.000 | +0.000..+0.000 | 0.357 / 0.648 / 1.15 | 2.02 | 1.405 | 0.104 | +0.0000 | +0.010 | 0.987 | 1.058 |  | 0.473 |
| SE none s2 | 20260930-6-LkS6 @ 1000 | 0 | 0 | 0 | 0 | 1.046 | -0.068..+0.120 | 0.417 / 0.758 / 1.63 | 1.78 | 1.518 | 0.181 | +0.0000 | +0.593 | 0.897 | 1.060 | 0.038 | 0.582 |
| SE none s2 | 20260930-6-LkS6 @ 5000 | 0 | 0 | 0 | 0 | 1.042 | -0.086..+0.196 | 0.336 / 0.782 / 1.65 | 2.14 | 1.645 | 0.211 | +0.0001 | +0.797 | 0.789 | 1.029 | 0.096 | 0.829 |
| SE none s2 | 20260930-6-LkS6 @ 7019 | 0 | 0 | 0 | 0 | 1.053 | -0.087..+0.207 | 0.324 / 0.764 / 1.65 | 2.10 | 1.653 | 0.210 | +0.0001 | +0.812 | 0.784 | 1.032 | 0.103 | 0.852 |
| SE zero-beta s1 | 20260930-7-crxN @ fresh | 0 | 0 | 0 | 0 | 1.000 | +0.000..+0.000 | 0.391 / 0.664 / 1.7 | 2.50 | 1.408 | 0.105 | +0.0000 | -0.053 | 0.984 | 1.029 |  | 0.477 |
| SE zero-beta s1 | 20260930-9-RrGx @ 1000 | 0 | 0 | 0 | 0 | 1.049 | -0.052..+0.186 | 0.463 / 0.785 / 2.35 | 1.98 | 1.558 | 0.182 | +0.0000 | +0.454 | 0.870 | 1.014 | 0.050 | 0.668 |
| SE zero-beta s1 | 20260930-9-RrGx @ 5000 | 0 | 0 | 0 | 0 | 1.043 | -0.110..+0.300 | 0.431 / 0.904 / 3.51 | 1.90 | 1.671 | 0.205 | -0.0000 | +0.608 | 0.776 | 0.975 | 0.109 | 1.048 |
| SE zero-beta s1 | 20260930-9-RrGx @ 5030 | 0 | 0 | 0 | 0 | 1.043 | -0.111..+0.301 | 0.43 / 0.905 / 3.51 | 1.90 | 1.672 | 0.205 | -0.0000 | +0.609 | 0.776 | 0.974 | 0.109 | 1.049 |
| SE zero-beta s2 | 20260930-8-8qyR @ fresh | 0 | 0 | 0 | 0 | 1.000 | +0.000..+0.000 | 0.35 / 0.615 / 1.53 | 1.97 | 1.416 | 0.116 | +0.0000 | -0.073 | 0.985 | 0.969 |  | 0.479 |
| SE zero-beta s2 | 20260930-10-H51a @ 1000 | 0 | 0 | 0 | 0 | 1.050 | -0.035..+0.131 | 0.417 / 0.702 / 1.96 | 1.67 | 1.557 | 0.183 | -0.0000 | +0.402 | 0.889 | 1.012 | 0.044 | 0.636 |
| SE zero-beta s2 | 20260930-10-H51a @ 5000 | 0 | 0 | 0 | 0 | 1.066 | -0.074..+0.218 | 0.429 / 0.834 / 2.49 | 1.64 | 1.676 | 0.206 | -0.0000 | +0.534 | 0.788 | 0.992 | 0.096 | 0.803 |
| SE zero-beta s2 | 20260930-10-H51a @ 5004 | 0 | 0 | 0 | 0 | 1.066 | -0.074..+0.219 | 0.428 / 0.834 / 2.49 | 1.64 | 1.676 | 0.206 | -0.0000 | +0.534 | 0.788 | 0.992 | 0.096 | 0.805 |

<!-- end:se_arms -->

- All eight runs (four arms × two seeds) have the same policy-head profile at matched steps: no dead or always-on channels, mean-row ratio 0.20–0.23, static level +0.5…+0.9, bias mean |≤ 1e-4|.
- The SE style changes nothing visible in the policy head.

## Validation: static shared level vs the bf16-head-offset survey's forward pass

- The large mismatches are the June q2Bb experiments (Juoc, eUsW, Pa83, I78x) and LMGh. The survey itself flagged all of them as "fp64 CE implausible", meaning its forward pass did not model them correctly.
- wTp3 was not measured.

<!-- begin:validation_static_level -->

| model_id | step | static shared level (this study, weights only) | measured all-logit mean, median over positions (bf16-head-offset survey) | measured legal-logit mean (survey) |
|---|---|---|---|---|
| 20260628-1-tWtk | None | -0.12 | -0.06 | -0.08 |
| 20260628-2-a5fc | 45441 | -0.18 | -0.18 | +13.53 |
| 20260628-9-OdUt | 15460 | -0.33 | -0.31 | +13.15 |
| 20260629-1-Uf4p | 39419 | -0.93 | -0.92 | +12.29 |
| 20260703-1-Dg5v | 268506 | -16.95 | -17.30 | -2.23 |
| 20260714-1-h7vI | 336610 | -191.10 | -195.02 | -177.64 |
| 20260729-1-VZ2j | 106333 | -285.06 | -287.93 | -257.44 |
| 20260802-2-Xuub | 49374 | -299.60 | -303.89 | -265.26 |
| 20260805-1-0pTW | 2000 | -300.38 | -304.20 | -266.25 |
| 20260629-3-3MIV | None | +0.07 | +0.07 | +0.10 |
| 20260629-4-y5u7 | 13464 | +0.39 | +0.37 | +14.44 |
| 20260630-5-BEKK | 120695 | -1.50 | -1.50 | +11.82 |
| 20260701-2-SvRu | 8000 | -1.68 | -1.67 | +11.69 |
| 20260705-1-znR7 | 114000 | -6.82 | -6.94 | +6.58 |
| 20260629-5-Coxw | None | +0.00 | -0.01 | +0.25 |
| 20260629-6-yqMI | 55550 | +0.01 | -0.02 | +13.95 |
| 20260709-1-avoB | 277000 | -3.11 | -3.15 | +10.01 |
| 20260701-3-nT8Y | None | +0.09 | +0.03 | +0.38 |
| 20260701-4-CIvL | 65883 | -2.36 | -2.27 | +12.19 |
| 20260701-5-bOYQ | 70779 | -5.22 | -5.13 | +9.41 |
| 20260706-2-3CZF | 15000 | -5.96 | -5.82 | +8.83 |
| 20260707-1-cslu | 140000 | -14.06 | -14.06 | +1.42 |
| 20260708-4-kEiZ | 21086 | -14.09 | -14.10 | +1.12 |
| 20260702-7-Qeu8 | None | +0.00 | +0.01 | +0.18 |
| 20260702-9-GLu5 | 41000 | -1.35 | -1.33 | +13.01 |
| 20260703-1-Lnji | 67508 | -4.79 | -4.91 | +9.40 |
| 20260706-1-PVZp | 67000 | -9.71 | -9.69 | +4.44 |
| 20260727-1-Ejp0 | 1397000 | -157.94 | -159.36 | -138.17 |
| 20260704-1-X79T | 21224 | -0.33 | -0.32 | +13.78 |
| 20260704-2-jSjr | 42507 | -1.84 | -1.83 | +12.58 |
| 20260704-3-h7Pp | 42507 | -3.80 | -3.81 | +10.21 |
| 20260708-5-0YQL | 26492 | -4.33 | -4.20 | +9.91 |
| 20260708-6-sFzi | 88107 | -7.07 | -6.97 | +7.02 |
| 20260711-16-VRR4 | None | -0.00 | -0.03 | -0.05 |
| 20260711-17-pycz | 120000 | -2.18 | -2.17 | +11.29 |
| 20260712-6-lTiK | 220000 | -3.33 | -3.19 | +10.21 |
| 20260714-1-NYAZ | 758000 | -14.25 | -14.46 | +0.21 |
| 20260722-1-syxR | 558000 | -15.98 | -16.00 | -0.86 |
| 20260727-1-Ejp0 | 2578 | -147.78 | -147.76 | -128.31 |
| 20260727-1-Ejp0 | 21009 | -147.78 | -147.76 | -128.31 |
| 20260727-1-Ejp0-9 | 197340 | -71.92 | -72.15 | -57.16 |
| 20260727-1-Ejp0-3 | 82033 | -147.77 | -148.29 | -130.38 |
| 20260727-1-Ejp0-59 | 1118017 | -139.17 | -140.13 | -122.86 |
| 20260727-1-Ejp0-66 | 1183795 | -138.92 | -139.86 | -122.46 |
| 20260727-1-Ejp0-67 | 1186322 | -138.92 | -139.97 | -122.44 |
| 20260601-11-bzw3-31 | 467065 | -7.69 | -7.66 | -2.95 |
| 20260601-11-bzw3-31 | 467099 | -7.69 | -7.66 | -2.95 |
| 20260609-12-LMGh-3 | 79135 | -7.61 | -174.05 | -212.72 |
| 20260612-21-eBNC-10 | 49836 | +0.22 | +0.17 | +12.08 |
| 20260614-4-wTp3-3 | 35496 | +0.42 |  |  |
| 20260626-2-q2Bb | None | +0.02 | +0.02 | -0.12 |
| 20260626-4-Juoc | 9580 | -2.05 | -55.85 | -9.46 |
| 20260627-2-eUsW | 26000 | -7.28 | +0.25 | +3.00 |
| 20260627-3-Jbl3 | 7000 | +0.89 | +0.24 | +9.88 |
| 20260627-4-Pa83 | 10000 | +0.34 | +19.29 | +127.38 |
| 20260627-5-I78x | 4000 | +0.25 | +73.03 | +163.19 |
| 20260630-3-T97X | None | -0.00 | -0.04 | -0.06 |
| 20260630-4-ASdQ | 4433 | +0.86 | +0.73 | +18.60 |
| 20260702-1-CPZm | 1125 | +0.27 | +0.34 | +13.02 |
| 20260702-4-DEQi | 74225 | -3.38 | -3.19 | +11.80 |
| 20260711-1-wXIL | None | +0.01 | +0.04 | +0.17 |
| 20260711-10-FxUc | 5000 | +0.30 | +0.30 | +14.39 |
| 20260711-11-xE1v | None | +0.01 | +0.02 | -0.30 |
| 20260711-13-9mEU | 49000 | -0.91 | -0.89 | +13.49 |
| 20260711-2-b5TY | 23000 | -0.96 | -0.88 | +13.19 |
| 20260711-3-pm4J | None | -0.12 | +0.12 | -0.12 |
| 20260711-4-1mjX | 20000 | -0.24 | -0.23 | +14.41 |
| 20260711-9-dOjG | None | -0.05 | +0.00 | +0.06 |
| 20260712-2-utcG | 1000 | +0.30 | +0.25 | +8.39 |
| 20260712-3-7tGs | 3000 | +0.26 | +0.25 | +12.02 |
| 20260712-4-s3bg | 15000 | -0.11 | -0.10 | +13.14 |
| 20260712-5-iOfG | 4000 | +0.24 | +0.22 | +10.58 |
| 20260921-2-Mh5n-3 | 15901 | +0.59 | +0.54 | +12.22 |

<!-- end:validation_static_level -->

## Lineages and exclusions

<!-- begin:lineages -->

| lineage | group | analyzed in detail | max segment step | unique checkpoints | reference (init comparison) | segments (model_id suffix) |
|---|---|---|---|---|---|---|
| SE scale+bias full-leaky fresh (never trained) | supplement | yes | -1 | 1 | 20261001-23-Dmwe @ fresh | Dmwe |
| leaky-FC1 (SE scale+bias, FC1 leaky) | experiment | yes | 12000 | 13 | 20261001-42-2q0Q @ fresh | 2q0Q, NbWz |
| SE scale+bias s1 | experiment | yes | 33014 | 35 | 20260929-12-JZOe @ fresh | JZOe, bWdy |
| SE scale+bias s2 | experiment | yes | 7282 | 9 | 20260930-1-H1Oq @ fresh | H1Oq, k98x |
| SE attenuate-only s1 | experiment | yes | 33012 | 35 | 20260929-13-06yp @ fresh | 06yp, L6Qm |
| SE attenuate-only s2 | experiment | yes | 7289 | 9 | 20260930-2-Gf9P @ fresh | Gf9P, 5TXu |
| SE none s1 | experiment | yes | 32036 | 34 | 20260929-18-D9is @ fresh | D9is, 834D |
| SE none s2 | experiment | yes | 7019 | 9 | 20260930-3-V9zk @ fresh | V9zk, LkS6 |
| SE zero-beta s1 | experiment | yes | 5030 | 7 | 20260930-7-crxN @ fresh | crxN, RrGx |
| SE zero-beta s2 | experiment | yes | 5004 | 7 | 20260930-8-8qyR @ fresh | 8qyR, H51a |
| v5 | lineage | yes | 336610 | 678 | 20260628-1-tWtk @ fresh | tWtk, a5fc, OdUt, Uf4p, Dg5v, h7vI, VZ2j, Xuub, 0pTW |
| mini2b | lineage | yes | 120695 | 125 | 20260629-3-3MIV @ fresh | 3MIV, y5u7, BEKK, SvRu, znR7 |
| coxw | lineage | yes | 277000 | 285 | 20260629-5-Coxw @ fresh | Coxw, yqMI, avoB |
| ykkk | lineage | yes | 250803 | 269 | 20260630-1-YkKk @ fresh | YkKk, 6y0s, 0Iwe, amlg |
| nt8y | lineage | yes | 140000 | 203 | 20260701-3-nT8Y @ fresh | nT8Y, CIvL, bOYQ, 3CZF, cslu, kEiZ |
| qeu8 (replay main, ends Ejp0) | lineage | yes | 1397000 | 1572 | 20260702-7-Qeu8 @ fresh | Qeu8, GLu5, Lnji, PVZp, Ejp0 |
| qeu8e (epoch branch) | lineage | yes | 88107 | 225 | 20260702-7-Qeu8 @ fresh | Qeu8, X79T, jSjr, h7Pp, 0YQL, sFzi |
| qeu8-1blk128 | lineage | yes | 120000 | 121 | 20260711-16-VRR4 @ fresh | VRR4, pycz |
| qeu8init sf100sl100 vs-UCI | lineage | yes | 758000 | 4 | 20260702-7-Qeu8 @ fresh | Qeu8, lTiK, NYAZ, syxR |
| Ejp0 headfix-phase2 (from Ejp0 @681k) | supplement | yes | 20000 | 20 | 20260727-1-Ejp0 @ 681000 | oeNy |
| Ejp0 self-play run 1 | lineage | yes | 197340 | 15 | 20260727-1-Ejp0 @ 1300000 | Ejp0:selfplay |
| Ejp0 self-play run 2 | lineage | yes | 1186322 | 70 | 20260727-1-Ejp0 @ 1300000 | Ejp0:selfplay |
| bzw3 self-play | lineage | yes | 467099 | 3 | 20260601-11-bzw3-31 @ 467065 | bzw3:selfplay |
| KbHZ self-play (fp32) | lineage | yes | 532369 | 4 | 20260514-1-KbHZ-18 @ 494927 | KbHZ:selfplay |
| sMe9 self-play (fp32) | lineage | yes | 373416 | 3 | 20260525-1-sMe9-26 @ 197269 | sMe9:selfplay |
| LWKa self-play (v4 12-block) | lineage | yes | 106695 | 2 | 20260531-9-LWKa-11 @ 106695 | LWKa:selfplay |
| LMGh self-play | lineage | yes | 79135 | 2 | 20260609-12-LMGh-4 @ 79135 | LMGh:selfplay |
| WjRY self-play | lineage | yes | 98974 | 2 | 20260609-14-WjRY-8 @ 98974 | WjRY:selfplay |

<!-- end:lineages -->

Segments not in any analyzed lineage (max step ≤ 75,000; trajectory only):

<!-- begin:excluded_segments -->

| segment (model_id base : kind) | unique checkpoints | max training_step | policy style | K | latest file |
|---|---|---|---|---|---|
| 20260529-10-ysdg:selfplay:run1 | 2 | 71604 | intermediate_conv | 128 | Sessions/20260531-024912-20260529-11-G5w2-promote-keep.dcmsession/trainer.dcmmodel |
| 20260531-3-KXvb:selfplay:run1 | 2 | 49686 | intermediate_conv | 128 | KeptSelfPlayModels/KXvb/20260531-184026-20260531-4-C2UF-periodic.dcmsession__trainer.dcmmodel |
| 20260607-4-2Gd1:selfplay:run1 | 2 | 54495 | intermediate_conv | 128 | Sessions/20260607-195042-20260607-5-oItC-manual.dcmsession/trainer.safetensors |
| 20260607-6-eaRt:selfplay:run1 | 2 | 48825 | intermediate_conv | 128 | Sessions/20260608-140104-20260607-7-KnCx-promote-keep.dcmsession/trainer.safetensors |
| 20260608-4-jaq1:selfplay:run1 | 2 | 13861 | intermediate_conv | 256 | Sessions/20260609-025811-20260608-5-cwkO-manual.dcmsession/trainer.safetensors |
| 20260610-1-JhJQ:selfplay:run1 | 2 | 58647 | intermediate_conv | 128 | KeptSelfPlayModels/JhJQ/20260611-220045-20260610-2-3p0G-periodic.dcmsession__trainer.safetensors |
| 20260612-21-eBNC:selfplay:run1 | 2 | 49836 | intermediate_conv | 128 | Sessions/20260614-002022-20260612-22-gFlw-manual.dcmsession/trainer.safetensors |
| 20260614-4-wTp3:selfplay:run1 | 2 | 35496 | intermediate_conv | 128 | Sessions/20260614-133904-20260614-5-t9sX-promote-keep.dcmsession/trainer.safetensors |
| 20260626-2-q2Bb:replay | 1 | fresh only | intermediate_conv | 128 | Models/20260626-202904-20260626-2-q2Bb-manual.safetensors |
| 20260626-4-Juoc:replay | 1 | 9580 | intermediate_conv | 128 | Models/20260626-202904-20260626-2-q2Bb-manual-replay-latest.safetensors |
| 20260627-2-eUsW:replay | 1 | 26000 | intermediate_conv | 128 | Models/20260626-q2Bb-clamp-replay-latest.safetensors |
| 20260627-3-Jbl3:replay | 2 | 7000 | intermediate_conv | 128 | Models/20260626-q2Bb-tanh-replay-latest.safetensors |
| 20260627-4-Pa83:replay | 2 | 10000 | intermediate_conv | 128 | Models/20260626-q2Bb-tanhSqrtN-replay-latest.safetensors |
| 20260627-5-I78x:replay | 4 | 4000 | intermediate_conv | 128 | Models/20260627-freshNet-replay-latest.safetensors |
| 20260627-7-mUF5:replay | 25 | 28797 | simple_conv | 128 | Models/20260627-v3_8block_3x3-replay-latest.safetensors |
| 20260630-3-T97X:replay | 1 | fresh only | intermediate_conv | 5 | Models/20260630-002027-20260630-3-T97X-manual.safetensors |
| 20260630-4-ASdQ:replay | 3 | 4433 | intermediate_conv | 5 | Models/20260630-T97X-replay-latest.safetensors |
| 20260701-1-S916:replay | 1 | fresh only | intermediate_conv | 512 | Models/20260701-uni16-9x9stem-seed.safetensors |
| 20260702-1-B9SE:replay | 1 | fresh only | intermediate_conv | 512 | Models/20260702-9blk16se-9x9stem-seed.safetensors |
| 20260702-1-CPZm:replay | 1 | 1125 | intermediate_conv | 512 | Models/20260701-uni16-9x9stem-replay-latest.safetensors |
| 20260702-4-DEQi:replay | 75 | 74225 | intermediate_conv | 512 | Models/20260702-9blk16se-9x9stem-replay-latest.safetensors |
| 20260711-1-wXIL:replay | 1 | fresh only | intermediate_conv | 512 | Models/20260711-014615-20260711-1-wXIL-manual.safetensors |
| 20260711-10-FxUc:replay | 5 | 5000 | intermediate_conv | 512 | Models/20260711-nt8y3x3stem-seed2-std-replay-latest.safetensors |
| 20260711-11-xE1v:replay | 1 | fresh only | intermediate_conv | 512 | Models/20260711-nt8y15x15stem-fresh.safetensors |
| 20260711-13-9mEU:replay | 49 | 49000 | intermediate_conv | 512 | Models/20260711-nt8y15x15stem-std-replay-latest.safetensors |
| 20260711-2-b5TY:replay | 23 | 23000 | intermediate_conv | 512 | Models/20260711-wXIL-std2026_05-replay-latest.safetensors |
| 20260711-3-pm4J:replay | 1 | fresh only | intermediate_conv | 512 | Models/20260711-nt8y3x3stem-fresh.safetensors |
| 20260711-4-1mjX:replay | 20 | 20000 | intermediate_conv | 512 | Models/20260711-nt8y3x3stem-std-replay-latest.safetensors |
| 20260711-9-dOjG:replay | 1 | fresh only | intermediate_conv | 512 | Models/20260711-nt8y3x3stem-seed2-fresh.safetensors |
| 20260712-2-utcG:replay | 1 | 1000 | intermediate_conv | 512 | Models/20260712-qeu8init-sf10-vsuci-latest.safetensors |
| 20260712-3-7tGs:replay | 3 | 3000 | intermediate_conv | 512 | Models/20260712-qeu8init-sf100-vsuci-latest.safetensors |
| 20260712-4-s3bg:replay | 15 | 15000 | intermediate_conv | 512 | Models/20260712-qeu8init-sf200-vsuci-latest.safetensors |
| 20260712-5-iOfG:replay | 4 | 4000 | intermediate_conv | 512 | Models/20260712-qeu8init-sloppy20-vsuci-latest.safetensors |
| 20260921-2-Mh5n:selfplay:run1 | 2 | 15901 | intermediate_conv | 128 | Sessions/20260921-224358-20260921-3-MNTv-sigusr2.dcmsession/trainer.safetensors |

<!-- end:excluded_segments -->

### Skipped files

- **Two torn exports were not read:** `Models/20260702-Qeu8-resume3-replay-step428000.safetensors.CORRUPT-torn-export` and `…-step632000…`. They are already labelled torn, and their extension is not `.safetensors`.
- **No `.dcmmodel` file was skipped.** All 21 were readable (SHA-256 trailer verified, archHash resolved to a preset, positional layout mapped and size-checked); 11 are unique.
  - sMe9 (197k, 373k), KbHZ (495k) and LWKa (107k) are in the detailed set.
  - ysdg (72k) and KXvb (50k) are trajectory-only.
- **v5 Xuub @ 49,374 is on a dead-end branch.** It is the `-DO-NOT-RESUME` file. Run 5 (0pTW) resumed from Xuub @ 46,000 (v5-lineage.md §5), so Xuub's cum step 861,143 lies past 0pTW's base.

## Reproduce

CPU-only and read-only on checkpoints (Python 3.14, numpy only). From `documentation/research/policy-head-2026-10-01/scripts/`:

```bash
python3 scan_headers.py > inventory_raw.jsonl   # optional raw header dump of every checkpoint
python3 run_all.py          # inventory.csv, trajectory.csv, lineages.json, detailed.jsonl.gz, channels/
python3 findings.py         # findings.csv, findings_aggregated.csv, growth.json
python3 render_tables.py    # results/tables/*.md
python3 assemble_report.py  # fills the tables in this REPORT.md
```

- `run_all.py` reads the policy tensors of ~4,000 checkpoints (a few hundred KB each), plus each detailed file in full for its SHA-256. It takes a few minutes.
- The leaky-FC1 run was live, so a re-run picks up newer `NbWz` checkpoints. The matched step is chosen dynamically as the leaky run's latest.
- Outputs (base-2 sizes):
  - `results/`: 22 MB total.
  - `channels/`: 9.0 MB.
  - `detailed.jsonl.gz`: 7.8 MB.
  - `trajectory.csv`: 3.1 MB.
  - `tables/`: 504 KB.

Checkpoint inventory of the detailed set (model_id / step from `__metadata__`; sha256 of the file):

<!-- begin:inventory -->

| lineage | model_id @ step | role | cum step | creator | file (under DrewsChessMachine/) | sha256[:12] | velocity | bf16-exact fraction (policy.conv.weight) |
|---|---|---|---|---|---|---|---|---|
| SE scale+bias full-leaky fresh (never trained) | 20261001-23-Dmwe @ fresh | M |  | derive-model | `Models/20261001-test_SE_scale+bias-leaky-fresh.safetensors` | 05a96c973c0c | no | 1.000 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-42-2q0Q @ fresh | M |  | derive-model | `Models/20261001-test_SE_scale+bias-fc1leaky-fresh.safetensors` | 37648d6de0ff | no | 1.000 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 1000 | M | 1000 | replay | `Models/20261001-test_SE_scale+bias-fc1leaky-replay-step1000.safetensors` | c7c5ec5ae9b4 | yes | 0.000 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 2000 | M | 2000 | replay | `Models/20261001-test_SE_scale+bias-fc1leaky-replay-step2000.safetensors` | b2c7615da2f0 | yes | 0.000 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 3000 | M | 3000 | replay | `Models/20261001-test_SE_scale+bias-fc1leaky-replay-step3000.safetensors` | af5933770887 | yes | 0.000 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 4000 | M | 4000 | replay | `Models/20261001-test_SE_scale+bias-fc1leaky-replay-step4000.safetensors` | 3e6db66dd5fc | yes | 0.000 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 5000 | M | 5000 | replay | `Models/20261001-test_SE_scale+bias-fc1leaky-replay-step5000.safetensors` | 20d9a19c0829 | yes | 0.000 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 6000 | M | 6000 | replay | `Models/20261001-test_SE_scale+bias-fc1leaky-replay-step6000.safetensors` | a78c394b9595 | yes | 0.000 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 7000 | M | 7000 | replay | `Models/20261001-test_SE_scale+bias-fc1leaky-replay-step7000.safetensors` | e80ee54cab9a | yes | 0.000 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 8000 | M | 8000 | replay | `Models/20261001-test_SE_scale+bias-fc1leaky-replay-step8000.safetensors` | 7980790ef0b7 | yes | 0.000 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 9000 | M | 9000 | replay | `Models/20261001-test_SE_scale+bias-fc1leaky-replay-step9000.safetensors` | 360fc69248d8 | yes | 0.000 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 10000 | M | 10000 | replay | `Models/20261001-test_SE_scale+bias-fc1leaky-replay-step10000.safetensors` | 3e40abd602b8 | yes | 0.000 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 11000 | M | 11000 | replay | `Models/20261001-test_SE_scale+bias-fc1leaky-replay-step11000.safetensors` | c57048bc289a | yes | 0.000 |
| leaky-FC1 (SE scale+bias, FC1 leaky) | 20261001-43-NbWz @ 12000 | M | 12000 | replay | `Models/20261001-test_SE_scale+bias-fc1leaky-replay-latest.safetensors` | 36746e82fc95 | yes | 0.000 |
| SE scale+bias s1 | 20260929-12-JZOe @ fresh | M |  | new-model | `Models/20260929-test_SE_scale+bias-fresh.safetensors` | 2c4b779b0be4 | no | 1.000 |
| SE scale+bias s1 | 20260929-22-bWdy @ 1000 | M | 1000 | replay | `Models/20260929-test_SE_scale+bias-replay-step1000.safetensors` | 0b6f727d9332 | no | 1.000 |
| SE scale+bias s1 | 20260929-22-bWdy @ 5000 | M | 5000 | replay | `Models/20260929-test_SE_scale+bias-replay-step5000.safetensors` | be899f2fbda6 | no | 1.000 |
| SE scale+bias s1 | 20260929-22-bWdy @ 10000 | M | 10000 | replay | `Models/20260929-test_SE_scale+bias-replay-step10000.safetensors` | af021cef39de | no | 1.000 |
| SE scale+bias s1 | 20260929-22-bWdy @ 11000 | M | 11000 | replay | `Models/20260929-test_SE_scale+bias-replay-step11000.safetensors` | f69664c977e9 | no | 1.000 |
| SE scale+bias s1 | 20260929-22-bWdy @ 12000 | M | 12000 | replay | `Models/20260929-test_SE_scale+bias-replay-step12000.safetensors` | 466f234186ce | no | 1.000 |
| SE scale+bias s1 | 20260929-22-bWdy @ 20000 | M | 20000 | replay | `Models/20260929-test_SE_scale+bias-replay-step20000.safetensors` | 210f43a52aa6 | no | 1.000 |
| SE scale+bias s1 | 20260929-22-bWdy @ 33014 | M | 33014 | replay | `Models/20260929-test_SE_scale+bias-replay-latest.safetensors` | dfc9ef9290f1 | no | 1.000 |
| SE scale+bias s2 | 20260930-1-H1Oq @ fresh | M |  | new-model | `Models/20260929-test_SE_scale+bias-seed2-fresh.safetensors` | 649161860eae | no | 1.000 |
| SE scale+bias s2 | 20260930-4-k98x @ 1000 | M | 1000 | replay | `Models/20260929-test_SE_scale+bias-seed2-replay-step1000.safetensors` | f45d7eccf8f1 | yes | 0.000 |
| SE scale+bias s2 | 20260930-4-k98x @ 5000 | M | 5000 | replay | `Models/20260929-test_SE_scale+bias-seed2-replay-step5000.safetensors` | 762d4b93ddd3 | yes | 0.000 |
| SE scale+bias s2 | 20260930-4-k98x @ 7282 | M | 7282 | replay | `Models/20260929-test_SE_scale+bias-seed2-replay-latest.safetensors` | 23a23687a901 | yes | 0.000 |
| SE attenuate-only s1 | 20260929-13-06yp @ fresh | M |  | new-model | `Models/20260929-test_SE_attenuate-only-fresh.safetensors` | dcd7caf29e56 | no | 1.000 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 1000 | M | 1000 | replay | `Models/20260929-test_SE_attenuate-only-replay-step1000.safetensors` | b4c6f281c112 | no | 1.000 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 5000 | M | 5000 | replay | `Models/20260929-test_SE_attenuate-only-replay-step5000.safetensors` | c87c488cd1e7 | no | 1.000 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 10000 | M | 10000 | replay | `Models/20260929-test_SE_attenuate-only-replay-step10000.safetensors` | de02da41ffe8 | no | 1.000 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 11000 | M | 11000 | replay | `Models/20260929-test_SE_attenuate-only-replay-step11000.safetensors` | 5eec6ba87661 | no | 1.000 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 12000 | M | 12000 | replay | `Models/20260929-test_SE_attenuate-only-replay-step12000.safetensors` | 552100707efe | no | 1.000 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 20000 | M | 20000 | replay | `Models/20260929-test_SE_attenuate-only-replay-step20000.safetensors` | df506ce46236 | no | 1.000 |
| SE attenuate-only s1 | 20260929-23-L6Qm @ 33012 | M | 33012 | replay | `Models/20260929-test_SE_attenuate-only-replay-latest.safetensors` | 36887d785443 | no | 1.000 |
| SE attenuate-only s2 | 20260930-2-Gf9P @ fresh | M |  | new-model | `Models/20260929-test_SE_attenuate-only-seed2-fresh.safetensors` | 6f906ea13cdf | no | 1.000 |
| SE attenuate-only s2 | 20260930-5-5TXu @ 1000 | M | 1000 | replay | `Models/20260929-test_SE_attenuate-only-seed2-replay-step1000.safetensors` | 0ababa4386e8 | yes | 0.000 |
| SE attenuate-only s2 | 20260930-5-5TXu @ 5000 | M | 5000 | replay | `Models/20260929-test_SE_attenuate-only-seed2-replay-step5000.safetensors` | 5ca8c5bd2125 | yes | 0.000 |
| SE attenuate-only s2 | 20260930-5-5TXu @ 7289 | M | 7289 | replay | `Models/20260929-test_SE_attenuate-only-seed2-replay-latest.safetensors` | 63f1e3ea5d75 | yes | 0.000 |
| SE none s1 | 20260929-18-D9is @ fresh | M |  | new-model | `Models/20260929-test_SE_none-fresh.safetensors` | f4424eb972b9 | no | 1.000 |
| SE none s1 | 20260929-24-834D @ 1000 | M | 1000 | replay | `Models/20260929-test_SE_none-replay-step1000.safetensors` | 2fefad37808e | no | 1.000 |
| SE none s1 | 20260929-24-834D @ 5000 | M | 5000 | replay | `Models/20260929-test_SE_none-replay-step5000.safetensors` | d17c14a3c7b9 | no | 1.000 |
| SE none s1 | 20260929-24-834D @ 10000 | M | 10000 | replay | `Models/20260929-test_SE_none-replay-step10000.safetensors` | 0479a42ad9b4 | no | 1.000 |
| SE none s1 | 20260929-24-834D @ 11000 | M | 11000 | replay | `Models/20260929-test_SE_none-replay-step11000.safetensors` | 81ed6acc99d9 | no | 1.000 |
| SE none s1 | 20260929-24-834D @ 12000 | M | 12000 | replay | `Models/20260929-test_SE_none-replay-step12000.safetensors` | 607310c9d161 | no | 1.000 |
| SE none s1 | 20260929-24-834D @ 20000 | M | 20000 | replay | `Models/20260929-test_SE_none-replay-step20000.safetensors` | 504dc7c6a3a6 | no | 1.000 |
| SE none s1 | 20260929-24-834D @ 32036 | M | 32036 | replay | `Models/20260929-test_SE_none-replay-latest.safetensors` | 490784e74250 | no | 1.000 |
| SE none s2 | 20260930-3-V9zk @ fresh | M |  | new-model | `Models/20260929-test_SE_none-seed2-fresh.safetensors` | ae555c8cdc60 | no | 1.000 |
| SE none s2 | 20260930-6-LkS6 @ 1000 | M | 1000 | replay | `Models/20260929-test_SE_none-seed2-replay-step1000.safetensors` | caf35df3da77 | yes | 0.000 |
| SE none s2 | 20260930-6-LkS6 @ 5000 | M | 5000 | replay | `Models/20260929-test_SE_none-seed2-replay-step5000.safetensors` | 77d6ffc22279 | yes | 0.000 |
| SE none s2 | 20260930-6-LkS6 @ 7019 | M | 7019 | replay | `Models/20260929-test_SE_none-seed2-replay-latest.safetensors` | 4b2cb87b93f4 | yes | 0.000 |
| SE zero-beta s1 | 20260930-7-crxN @ fresh | M |  | derive-model | `Models/20260929-test_SE_zerobeta-seed1-fresh.safetensors` | d31dcd58c874 | no | 1.000 |
| SE zero-beta s1 | 20260930-9-RrGx @ 1000 | M | 1000 | replay | `Models/20260929-test_SE_zerobeta-seed1-replay-step1000.safetensors` | 13530fd55005 | yes | 0.000 |
| SE zero-beta s1 | 20260930-9-RrGx @ 5000 | M | 5000 | replay | `Models/20260929-test_SE_zerobeta-seed1-replay-step5000.safetensors` | 90a3d00f9303 | yes | 0.000 |
| SE zero-beta s1 | 20260930-9-RrGx @ 5030 | M | 5030 | replay | `Models/20260929-test_SE_zerobeta-seed1-replay-latest.safetensors` | 2597047748ed | yes | 0.000 |
| SE zero-beta s2 | 20260930-8-8qyR @ fresh | M |  | derive-model | `Models/20260929-test_SE_zerobeta-seed2-fresh.safetensors` | ab2647efa6b3 | no | 1.000 |
| SE zero-beta s2 | 20260930-10-H51a @ 1000 | M | 1000 | replay | `Models/20260929-test_SE_zerobeta-seed2-replay-step1000.safetensors` | a1d387d2a83a | yes | 0.000 |
| SE zero-beta s2 | 20260930-10-H51a @ 5000 | M | 5000 | replay | `Models/20260929-test_SE_zerobeta-seed2-replay-step5000.safetensors` | d5d9fc64d78b | yes | 0.000 |
| SE zero-beta s2 | 20260930-10-H51a @ 5004 | M | 5004 | replay | `Models/20260929-test_SE_zerobeta-seed2-replay-latest.safetensors` | a6df9b146608 | yes | 0.000 |
| v5 | 20260628-1-tWtk @ fresh | M |  | new-model | `Models/20260627-v5_5block_7x7_lnout-fresh.safetensors` | 6886f9b39b0f | no | 1.000 |
| v5 | 20260628-2-a5fc @ 10000 | M | 10000 | replay | `Models/20260628-v5_5block_7x7_lnout-step10000-frozen.safetensors` | a36f41a4fde4 | no | 1.000 |
| v5 | 20260628-2-a5fc @ 45441 | M | 45441 | replay | `Models/20260628-v5_5block_7x7_lnout-replay-latest.safetensors` | 1a5a543d1727 | no | 1.000 |
| v5 | 20260628-9-OdUt @ 15460 | M | 60901 | replay | `Models/20260628-v5_5block_7x7_lnout-wd5e4-replay-latest.safetensors` | 74179b73aa43 | no | 1.000 |
| v5 | 20260629-1-Uf4p @ 39419 | M | 100320 | replay | `Models/20260628-v5_5block_7x7_lnout-wd2.5e4-m93-replay-latest.safetensors` | 1de5353c11d7 | no | 1.000 |
| v5 | 20260703-1-Dg5v @ 268506 | M | 368826 | replay | `Models/20260702-v5cont-replay-step268506.safetensors` | 7543ca90e03c | no | 1.000 |
| v5 | 20260714-1-h7vI @ 115000 | M | 483826 | replay | `Models/20260713-v5cont-resume-replay-step115000.safetensors` | ae562a9ce2f5 | no | 1.000 |
| v5 | 20260714-1-h7vI @ 336610 | M | 705436 | replay | `Models/20260713-v5cont-resume-replay-step336610.safetensors` | f21eee621c9c | no | 1.000 |
| v5 | 20260729-1-VZ2j @ 106333 | M | 811769 | replay | `Models/20260728-v5cont-resume2-replay-step106333.safetensors` | 0500e1ad87b1 | no | 1.000 |
| v5 | 20260802-2-Xuub @ 49374 | M | 861143 | replay | `Models/20260802-v5cont-resume3-replay-step49374-DO-NOT-RESUME.safetensors` | a097bd1b9745 | no | 1.000 |
| v5 | 20260805-1-0pTW @ 2000 | M | 859769 | replay | `Models/20260804-v5cont-resume4-replay-latest.safetensors` | 1ea3c2c2c0fa | no | 1.000 |
| mini2b | 20260629-3-3MIV @ fresh | M |  | manual | `Models/20260629-161538-20260629-3-3MIV-manual.safetensors` | d4de3d2e6f8a | no | 1.000 |
| mini2b | 20260629-4-y5u7 @ 10000 | M | 10000 | replay | `Models/20260629-mini2b-3MIV-step10000-frozen.safetensors` | 6ff317368e02 | no | 1.000 |
| mini2b | 20260629-4-y5u7 @ 13464 | M | 13464 | replay | `Models/20260629-mini2b-3MIV-replay-latest.safetensors` | 8592d53a0852 | no | 1.000 |
| mini2b | 20260630-5-BEKK @ 120695 | M | 134159 | replay | `Models/20260629-mini2b-3MIV-resume-replay-latest.safetensors` | 7720afebb228 | no | 1.000 |
| mini2b | 20260701-2-SvRu @ 8000 | M | 142159 | replay | `Models/20260629-mini2b-3MIV-resume2-replay-latest.safetensors` | 6f4ce6a5b1b3 | no | 1.000 |
| mini2b | 20260705-1-znR7 @ 53000 | M | 195159 | replay | `Models/20260629-mini2b-3MIV-resume3-replay-step53000.safetensors` | f6f022a195cb | no | 1.000 |
| mini2b | 20260705-1-znR7 @ 114000 | M | 256159 | replay | `Models/20260629-mini2b-3MIV-resume3-replay-latest.safetensors` | 9e86b3d6cef8 | no | 1.000 |
| coxw | 20260629-5-Coxw @ fresh | M |  | manual | `Models/20260629-182512-20260629-5-Coxw-manual.safetensors` | 688af999bc7b | no | 1.000 |
| coxw | 20260629-6-yqMI @ 10000 | M | 10000 | replay | `Models/20260629-mini1b-Coxw-step10000-frozen.safetensors` | 928d5f6f14bf | no | 1.000 |
| coxw | 20260629-6-yqMI @ 55550 | M | 55550 | replay | `Models/20260629-mini1b-Coxw-replay-latest.safetensors` | d87836ba1794 | no | 1.000 |
| coxw | 20260709-1-avoB @ 136000 | M | 191550 | replay | `Models/20260629-mini1b-Coxw-resume-replay-step136000.safetensors` | 43162f0e2ee8 | no | 1.000 |
| coxw | 20260709-1-avoB @ 277000 | M | 332550 | replay | `Models/20260629-mini1b-Coxw-resume-replay-latest.safetensors` | 9d91f6991035 | no | 1.000 |
| ykkk | 20260630-1-YkKk @ fresh | M |  | manual | `Models/20260630-000533-20260630-1-YkKk-manual.safetensors` | c52c64da522c | no | 1.000 |
| ykkk | 20260630-2-6y0s @ 40677 | M | 40677 | replay | `Models/20260630-mini-YkKk-replay-latest.safetensors` | 6858e2554a27 | no | 1.000 |
| ykkk | 20260701-1-0Iwe @ 162330 | M | 203007 | replay | `Models/20260630-mini-YkKk-resume-replay-latest.safetensors` | 50849c9a71c3 | no | 1.000 |
| ykkk | 20260710-1-amlg @ 118000 | M | 321007 | replay | `Models/20260630-mini-YkKk-resume2-replay-step118000.safetensors` | 22717e13a253 | no | 1.000 |
| ykkk | 20260710-1-amlg @ 250803 | M | 453810 | replay | `Models/20260630-mini-YkKk-resume2-replay-latest.safetensors` | 375bf9ebb78d | no | 1.000 |
| nt8y | 20260701-3-nT8Y @ fresh | M |  | manual | `Models/20260701-161012-20260701-3-nT8Y-manual.safetensors` | 77c3baee597c | no | 1.000 |
| nt8y | 20260701-4-CIvL @ 1000 | M | 1000 | replay | `Models/20260701-nT8Y-fatconv-step1000-frozen.safetensors` | b35e43eee91b | no | 1.000 |
| nt8y | 20260701-4-CIvL @ 65883 | M | 65883 | replay | `Models/20260701-nT8Y-fatconv-replay-latest.safetensors` | c715e208b212 | no | 1.000 |
| nt8y | 20260701-5-bOYQ @ 70000 | M | 135883 | replay | `Models/20260701-nT8Y-fatconv-step135883-frozen.safetensors` | 8e37dfe4a599 | no | 1.000 |
| nt8y | 20260701-5-bOYQ @ 70779 | M | 136662 | replay | `Models/20260701-nT8Y-fatconv-resume-replay-latest.safetensors` | 5f2c7afb23e8 | no | 1.000 |
| nt8y | 20260706-2-3CZF @ 15000 | M | 151662 | replay | `Models/20260701-nT8Y-resume2-replay-latest.safetensors` | 6839a59cb79e | no | 1.000 |
| nt8y | 20260707-1-cslu @ 140000 | M | 291662 | replay | `Models/20260701-nT8Y-resume3-replay-latest.safetensors` | 4810b2188cd0 | no | 1.000 |
| nt8y | 20260708-4-kEiZ @ 21086 | M | 312748 | replay | `Models/20260701-nT8Y-resume4-replay-latest.safetensors` | a362b1efcdf5 | no | 1.000 |
| qeu8 (replay main, ends Ejp0) | 20260702-7-Qeu8 @ fresh | M |  | manual | `Models/20260702-164826-20260702-7-Qeu8-manual.safetensors` | 50f4aa23ae70 | no | 1.000 |
| qeu8 (replay main, ends Ejp0) | 20260702-9-GLu5 @ 1000 | M | 1000 | replay | `Models/20260702-Qeu8-step1000-frozen.safetensors` | 202628a58b2a | no | 1.000 |
| qeu8 (replay main, ends Ejp0) | 20260702-9-GLu5 @ 41000 | M | 41000 | replay | `Models/20260702-Qeu8-step41000-frozen.safetensors` | 648b5f0743f1 | no | 1.000 |
| qeu8 (replay main, ends Ejp0) | 20260703-1-Lnji @ 67508 | M | 108915 | replay | `Models/20260702-Qeu8-replay-latest.safetensors` | 31900c0502cc | no | 1.000 |
| qeu8 (replay main, ends Ejp0) | 20260706-1-PVZp @ 67000 | M | 175915 | replay | `Models/20260702-Qeu8-resume2-replay-latest.safetensors` | 4c896603f8e9 | no | 1.000 |
| qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0 @ 611000 | M | 786915 | replay | `Models/20260702-Qeu8-resume3-replay-step611000.safetensors` | fae870ae989e | no | 1.000 |
| qeu8 (replay main, ends Ejp0) | 20260727-1-Ejp0 @ 1397000 | M | 1572915 | replay | `Models/20260702-Qeu8-resume3-replay-latest.safetensors` | b4a131003a0b | no | 1.000 |
| qeu8e (epoch branch) | 20260702-7-Qeu8 @ fresh | M |  | manual | `Models/20260702-164826-20260702-7-Qeu8-manual.safetensors` | 50f4aa23ae70 | no | 1.000 |
| qeu8e (epoch branch) | 20260704-1-X79T @ 1000 | M | 1000 | replay | `Models/20260704-Qeu8e-replay-step1000.safetensors` | 282a011e84d8 | no | 1.000 |
| qeu8e (epoch branch) | 20260704-1-X79T @ 21224 | M | 21224 | replay | `Models/20260704-Qeu8e-replay-latest.safetensors` | 59dda07cd2e6 | no | 1.000 |
| qeu8e (epoch branch) | 20260704-2-jSjr @ 42507 | M | 63731 | replay | `Models/20260704-Qeu8e2-replay-latest.safetensors` | f79485eee6f4 | no | 1.000 |
| qeu8e (epoch branch) | 20260704-3-h7Pp @ 42507 | M | 106238 | replay | `Models/20260704-Qeu8e3-replay-latest.safetensors` | 76c114df1e22 | no | 1.000 |
| qeu8e (epoch branch) | 20260708-5-0YQL @ 5000 | M | 111238 | replay | `Models/20260704-Qeu8e4-replay-step5000.safetensors` | 1635252e8766 | no | 1.000 |
| qeu8e (epoch branch) | 20260708-5-0YQL @ 26492 | M | 132730 | replay | `Models/20260704-Qeu8e4-replay-latest.safetensors` | 7bfb2416608c | no | 1.000 |
| qeu8e (epoch branch) | 20260708-6-sFzi @ 88107 | M | 220837 | replay | `Models/20260704-Qeu8e5-replay-latest.safetensors` | 072873980516 | no | 1.000 |
| qeu8-1blk128 | 20260711-16-VRR4 @ fresh | M |  | new-model | `Models/20260711-qeu8-1blk128-fresh.safetensors` | ac4cae565f63 | no | 1.000 |
| qeu8-1blk128 | 20260711-17-pycz @ 1000 | M | 1000 | replay | `Models/20260711-qeu8-1blk128-std-replay-step1000.safetensors` | 8aa73c6dacf8 | no | 1.000 |
| qeu8-1blk128 | 20260711-17-pycz @ 61000 | M | 61000 | replay | `Models/20260711-qeu8-1blk128-std-replay-step61000.safetensors` | d775fa2461fb | no | 1.000 |
| qeu8-1blk128 | 20260711-17-pycz @ 120000 | M | 120000 | replay | `Models/20260711-qeu8-1blk128-std-replay-latest.safetensors` | af576673d1f0 | no | 1.000 |
| qeu8init sf100sl100 vs-UCI | 20260702-7-Qeu8 @ fresh | M |  | manual | `Models/20260702-164826-20260702-7-Qeu8-manual.safetensors` | 50f4aa23ae70 | no | 1.000 |
| qeu8init sf100sl100 vs-UCI | 20260712-6-lTiK @ 220000 | M |  | train-vs-uci | `Models/20260712-qeu8init-sf100sl100-vsuci-step220000-STOP.safetensors` | b86e2b161b01 | no | 1.000 |
| qeu8init sf100sl100 vs-UCI | 20260714-1-NYAZ @ 758000 | M |  | train-vs-uci | `Models/20260712-qeu8init-sf100sl100-vsuci-step978000-STOP.safetensors` | ae80ec37b377 | no | 1.000 |
| qeu8init sf100sl100 vs-UCI | 20260722-1-syxR @ 558000 | M |  | train-vs-uci | `Models/20260712-qeu8init-sf100sl100-vsuci-latest.safetensors` | a1305fd5447c | no | 1.000 |
| Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy @ 1000 | M |  | replay | `Models/20261001-headfix-phase2-Ejp0-replay-step1000.safetensors` | 849745bbf6af | yes | 0.000 |
| Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy @ 11000 | M |  | replay | `Models/20261001-headfix-phase2-Ejp0-replay-step11000.safetensors` | e4df2c1cc86f | yes | 0.000 |
| Ejp0 headfix-phase2 (from Ejp0 @681k) | 20261001-18-oeNy @ 20000 | M |  | replay | `Models/20261001-headfix-phase2-Ejp0-replay-latest.safetensors` | c3f9ae7ecf2e | yes | 0.000 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-1 @ 2578 | T |  | sigusr2 | `Sessions/20260807-071405-20260807-4-FeUB-sigusr2.dcmsession/trainer.safetensors` | 023557f5f7f9 | yes | 0.000 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-4 @ 104903 | M |  | promote | `KeptSelfPlayModels/Ejp0r1/20260727-1-Ejp0-4-step104903-promote-20260808-051503.safetensors` | 807b4a8cea34 | no | 1.000 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-10 @ 197340 | T |  | sigusr2 | `Sessions/20260808-222418-20260807-6-orSA-sigusr2.dcmsession/trainer.safetensors` | cf85bc8b554c | yes | 0.000 |
| Ejp0 self-play run 1 | 20260727-1-Ejp0-9 @ 197340 | C |  | sigusr2 | `Sessions/20260808-222418-20260807-6-orSA-sigusr2.dcmsession/champion.safetensors` | 40e7900e62b8 | no | 1.000 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-1 @ 20241 | M |  | promote | `KeptSelfPlayModels/Ejp0/20260727-1-Ejp0-1-step20241-promote-20260809-021726.safetensors` | 894ded5fa0d0 | no | 1.000 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-39 @ 935524 | M |  | promote | `KeptSelfPlayModels/Ejp0/20260727-1-Ejp0-39-step935524-promote-20260828-191834.safetensors` | da93ffd54b76 | no | 1.000 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-68 @ 1186322 | T |  | promote | `Sessions/20260917-223317-20260808-2-sjIy-promote-keep.dcmsession/trainer.safetensors` | 5ff067fa4708 | yes | 1.000 |
| Ejp0 self-play run 2 | 20260727-1-Ejp0-67 @ 1186322 | C |  | promote | `Sessions/20260917-223317-20260808-2-sjIy-promote-keep.dcmsession/champion.safetensors` | 4288ad713f59 | no | 1.000 |
| bzw3 self-play | 20260601-11-bzw3-31 @ 467065 | M |  | manual | `Models/20260607-015745-20260601-11-bzw3-31-manual.safetensors` | b6a47aa48974 | no | 1.000 |
| bzw3 self-play | 20260601-11-bzw3-32 @ 467099 | T |  | manual | `Sessions/20260607-015807-20260601-12-5K7Z-manual.dcmsession/trainer.safetensors` | 8a6140a92437 | yes | 0.000 |
| bzw3 self-play | 20260601-11-bzw3-31 @ 467099 | C |  | manual | `Sessions/20260607-015807-20260601-12-5K7Z-manual.dcmsession/champion.safetensors` | 26a3263567ff | no | 1.000 |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-18 @ 494927 | T |  | manual | `Sessions/last-before-big-changeup-20260524-175426-20260514-2-Ko63-manual.dcmsession/trainer.dcmmodel` | 660f694e9505 | yes |  |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-23 @ 532369 | T |  | manual | `Sessions/20260611-212501-20260514-2-Ko63-manual.dcmsession/trainer.safetensors` | 82611859134e | yes | 0.000 |
| KbHZ self-play (fp32) | 20260514-1-KbHZ-22 @ 532369 | C |  | manual | `Sessions/20260611-212501-20260514-2-Ko63-manual.dcmsession/champion.safetensors` | 413a90e99572 | no | 0.000 |
| sMe9 self-play (fp32) | 20260525-1-sMe9-26 @ 197269 | M |  | manual | `Models/20260527-143402-20260525-1-sMe9-26-manual.dcmmodel` | 54f07d90ab5f | no |  |
| sMe9 self-play (fp32) | 20260525-1-sMe9-33 @ 373416 | T |  | manual | `Sessions/20260529-182349-20260525-2-IWkd-manual.dcmsession/trainer.dcmmodel` | 2988aba2f23c | yes |  |
| sMe9 self-play (fp32) | 20260525-1-sMe9-32 @ 373416 | C |  | manual | `Sessions/20260529-182349-20260525-2-IWkd-manual.dcmsession/champion.dcmmodel` | 050f586ab97e | no |  |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-11 @ 106695 | T |  | periodic | `KeptSelfPlayModels/LWKa/20260602-005840-20260531-10-WcRm-periodic.dcmsession__trainer.dcmmodel` | 1b7400b3e84a | yes |  |
| LWKa self-play (v4 12-block) | 20260531-9-LWKa-10 @ 106695 | C |  | periodic | `KeptSelfPlayModels/LWKa/20260602-005840-20260531-10-WcRm-periodic.dcmsession__champion.dcmmodel` | 46dcfdc57712 | no |  |
| LMGh self-play | 20260609-12-LMGh-4 @ 79135 | T |  | promote | `Sessions/20260609-144428-20260609-13-vJnd-promote-keep.dcmsession/trainer.safetensors` | 4841e2201bf1 | yes | 1.000 |
| LMGh self-play | 20260609-12-LMGh-3 @ 79135 | C |  | promote | `Sessions/20260609-144428-20260609-13-vJnd-promote-keep.dcmsession/champion.safetensors` | 58bd9ca48c88 | no | 1.000 |
| WjRY self-play | 20260609-14-WjRY-8 @ 98974 | T |  | periodic | `KeptSelfPlayModels/WjRY/20260610-130641-20260609-15-tGOH-periodic.dcmsession__trainer.safetensors` | fd152b40ab5d | yes | 0.000 |
| WjRY self-play | 20260609-14-WjRY-7 @ 98974 | C |  | periodic | `KeptSelfPlayModels/WjRY/20260610-130641-20260609-15-tGOH-periodic.dcmsession__champion.safetensors` | a63cfddf4e9f | no | 1.000 |

<!-- end:inventory -->
