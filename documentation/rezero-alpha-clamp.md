# ReZero α soft-bound (tanh)

## TL;DR

The per-block ReZero scalar `α` (`*_res_scale`, `ChessNetwork.residualBlock`) is a
free, trainable, **un-decayed** scalar with no transform and nothing opposing its
growth. On a from-scratch corpus-replay run it ran away — from its `1/√N` init to
**~30+** over ~4–8k steps — which drowned the identity skip, degenerated the tower,
and destroyed inference. The fix is a smooth **forward soft-bound** of `α` through
`C·tanh(α/C)`, with `C` the asymptotic ceiling:

```
effective_α = C · tanh(α / C)     C = rezeroTanhCeilingMultiple · α₀ = 1/√N
```

with `C = α₀ = 1/√N` (`rezeroTanhCeilingMultiple · α₀`, multiplier 1.0). For small
`α` this is near-identity, and it saturates smoothly to `±C` so `α` can never enter
the runaway regime. Standard for every build (bf16 and fp32).

**Why `C = 1/√N`, not a bigger constant.** The raw `α` ratchets up regardless of the
bound (its gradient is one-way — see below), so the *effective* α saturates **at the
cap** across all blocks. The cap must therefore be set so that the fully-saturated
state is itself safe: with all N blocks at effective α = C, the residual-stream
variance scales as `Σα² = N·C²`. Setting `C = 1/√N` gives `Σα² = 1` (variance-
preserving) even when every block has saturated. An earlier `C = 1.0` (absolute)
was tried and **failed**: effective α saturated at ~0.95 across all 5 blocks
(`Σα² ≈ 4.5`), the residual-stream *mean* still compounded (`bn1Mean` 43→1384 over
1k steps), and the run broke ~step 5800 — the same failure as the hard clamp, just
delayed ~1k steps. See "The C=1.0 run" below.

**Why tanh and not a hard clamp:** the first fix here was a hard `[0, C]` clamp
with `C = 4.5·α₀ ≈ 2.0` at 5 blocks. It bounded the magnitude but had **two**
problems (see "The hard-clamp run" below): (1) zero gradient past `C`, so the
stored `α` drifted freely above the wall (2.0→2.27) while pinned and the parameter
stopped being meaningful; (2) `C≈2.0` is *already in the degraded regime* —
degradation set in at `α≈1.3–1.5`, below the cap — so the tower spent ~17k steps
broken before slowly, noisily clawing back to a mediocre ~650 pElo (vs the same
start model's ~900 at step 2k). tanh fixes the dead-zone (live gradient
everywhere); the ceiling must additionally be set to `1/√N` so the saturated state
is variance-safe (a `C = 1.0` tanh still broke — see below).

## What happened (the run that motivated this)

A `--replay-corpus` run (lichess `2026-05`, fresh `q2Bb` start model, 5×128 7×7
v4 tower, bf16) trained cleanly through ~3000 steps (probe pElo ~895), then the
inference probe collapsed and stayed collapsed:

| step | probe pElo | probe nll | block-0 α | bn1 running_mean |
|------|-----------|-----------|-----------|------------------|
| 3000 | 895 | 3.10 | (healthy) | small |
| 4000 | 448 | 15.34 | 1.37 | ~46 |
| 5000 | 463 | 15.86 | — | ~70,000 |
| 6000 | 455 | 16.11 | — | ~750,000 |
| 7000 | 409 | 16.51 | — | ~2,000,000 |
| 8000 | 524 | 12.40 | 28.9 | ~3,900,000 |
| 9000 | 422 | 16.65 | 31.8 | ~7,100,000 |

`nll` worse than uniform (`ln 4864 ≈ 8.49`) = the saved model evaluates to noise.
Meanwhile the *training-mode* stats stayed merely degraded (legal-mass ~0.6), not
random — the collapse was specific to the inference path.

## Root cause: an unbounded free scalar

`out = skip + α · F(x)` with `α` a single learnable scalar per block. It is:

- **trainable** (gets gradient every step),
- **un-decayed** (`shouldDecay = false` — nothing pulls it back toward init),
- **un-transformed / un-clamped** (used raw, no `tanh`/`sigmoid`/clip),

so it is the one parameter in the network with a one-way ratchet and no restoring
force. Two facts make the gradient push it up indefinitely:

1. **No competing brake.** Conv weights are weight-decayed *and* their scale is
   renormalized away by the next BN, so growing them is pointless and penalized.
   BN γ/β and biases sit before a BN (mean-subtracted) or a sigmoid (saturating),
   so they self-limit. `α` alone is an un-renormalized gain on the residual
   highway with decay off.
2. **The training loss can't see the cost.** The next block's BN renormalizes the
   (now huge) stream back to ~unit scale before it reaches the loss, so in the
   batch-stat *training* forward, growing `α` is nearly free — there is no finite
   optimum the optimizer settles into, only a monotone "bigger helps (a little)".
   Growing `α` makes the block favor its learned branch over the identity skip;
   that lowers training loss with diminishing-but-never-reversing gradient, so
   `α` drifts up without bound.

As `α` grows it **drowns the skip** (`α·F ≫ skip`), turning the residual block into
a plain conv block and dismantling the identity highway. The forward activations
(the residual *stream*) compound down the tower to ~10⁶, and the EMA running stats
track that. Inference, which normalizes against the (stale, exploded ~10⁹-variance)
running stats, produces garbage; training, using fresh per-batch stats, stays
merely degraded. That train-vs-inference split is a **statistics** mismatch, not a
precision one.

## Precision (bf16) was investigated and ruled out

An earlier hypothesis blamed bf16's 8-bit mantissa in the BN `(x − mean)` subtract
at large magnitudes (catastrophic cancellation). It was tested directly and is
**not** the cause:

- Running the BN normalize in fp32 (cast input + fp32 stats + fp32 normalize)
  changed the broken probe's `nll` by ~5×10⁻⁵ — i.e. not at all.
- Forcing **full fp32 compute** on the same weights (via the resume path's
  `forceFloat32`, `SessionController+Checkpoint.swift`) still gave pElo ~477 /
  nll ~13.8 — still broken.

With zero precision loss anywhere the model still evaluates to noise, so the
damage is in the *weights/function*, not the arithmetic. No inference-side numeric
change recovers an α-runaway checkpoint; only retraining with the clamp does.
(Carrying the residual stream in fp32 would be a separate, larger change and is
**not** part of this fix — it was shown unnecessary here.)

## The hard-clamp run (why the clamp was replaced)

The first fix was a hard forward clamp `α ← max(min(α, C), 0)` with
`C = 4.5·α₀ ≈ 2.0` at 5 blocks. A clean A/B retrain (same `q2Bb` start model, same
lichess corpus, same hyperparameters as the broken run, *only* the clamp added)
showed the clamp **engaged and bounded** `α` — but did not produce a healthy model:

| step | probe pElo | probe nll | block-0 α (stored) | bn1 running_mean | note |
|------|-----------|-----------|--------------------|------------------|------|
| 3000 | 794 | 3.48 | ~1.0 | ~20 | still healthy, clamp inactive |
| 5000 | 577 | 6.06 | 2.0+ (pinned) | ~6,200 | clamp engaged; degrading |
| 9000 | 435 | 11.5 | 2.0+ | ~1,150,000 | broken regime |
| 17000 | 373 | 14.6 | 2.0+ | ~1,780,000 | worst; train-side still improving |
| 22000 | 677 | 4.31 | 2.0+ | ~2,130,000 | best; slow noisy recovery |
| 26000 | 596 | 5.14 | 2.0+ | ~2,310,000 | oscillating ~400–680 |

Two findings. **(1) C≈2.0 is too loose.** Degradation onset was `α≈1.3–1.5` —
*below* the cap — so the clamp never engaged until the tower was already in the
bad regime; once pinned at 2.0, the residual stream still compounded (BN
running-mean climbed monotonically past 2×10⁶) because the convs re-amplify a
stream that's already large. **(2) The hard clamp has a dead zone.** Gradient is
exactly 0 past `C`, so the stored `α` drifted to 2.0→2.27 (momentum pushing a
parameter that contributes nothing new) and the block was simply slammed against
the wall. The model spent ~17k steps broken, then slowly and noisily recovered to
a *mediocre* ~650 — still ~250 Elo below where the same start model sat at step
2000 (~900). Recoverable, but not a fix.

## The C=1.0 tanh run (why the ceiling must be 1/√N)

Replacing the hard clamp with `C·tanh(α/C)` at `C = 1.0` (absolute) fixed the
dead-zone but **still broke** — and the break exposed the real constraint. Clean
A/B again (same `q2Bb` start, same corpus/hparams, only the bound changed):

| step | probe pElo / nll | train pLoss / legalMass | α effective (max) | bn1Mean |
|------|------------------|-------------------------|-------------------|---------|
| 3000 | 985 / 2.94 | 2.98 / 0.948 | 0.97 | 10 |
| 4000 | **1015 / 2.88** | 2.94 / 0.959 | 0.78 (raw 1.05) | 17.5 |
| 5000 | 471 / 8.97 | 2.91 / 0.965 | 0.84 (raw 1.21) | 43 |
| 6000 | 442 / 6.20 | 6.09 / 0.18 | **0.95 (raw 2.4)** | **1384** |

It trained beautifully — pElo **1015 at 4k, the best of any run** — then broke
~step 5800. Mechanism, confirmed:

- **The raw `α` keeps ratcheting up regardless of the bound** (its loss-gradient is
  one-way; tanh caps the *effective* value, not the raw). Raw α went 1.2→2.4 in
  1000 steps; effective saturated toward the `C = 1.0` wall on **every** block
  (0.92–0.98 at 6k).
- **All blocks saturating near 1.0 is too much.** Variance-preserving residual
  scaling needs `Σα² ≈ 1`, i.e. α ≈ `1/√N ≈ 0.45` for 5 blocks. With all five at
  ~0.95, `Σα² ≈ 4.5` — 4.5× too large.
- **What actually explodes is the stream *mean*.** Each block's branch ends in ReLU
  (mean > 0), so `out = skip + α·F` adds a positive increment onto the skip highway
  every block, scaled by α, accumulating down the tower and over steps. `bn1Mean`
  crept 3.5→43 while α was ~0.8, then jumped 43→1384 as α→0.95. The blown-up running
  mean wrecks the BN eval path (train uses batch stats and stays fine — note train
  pLoss/legalMass were still pristine at 5k while the probe had already dropped).

Conclusion: capping the *effective* value isn't enough; the cap value itself must be
variance-safe **when saturated**, because saturation is where α ends up. Hence
`C = 1/√N`.

In `ChessNetwork.residualBlock`, soft-bound `α` through `C·tanh(α/C)` in the
forward before it scales the branch:

```swift
// C = α₀ · rezeroTanhCeilingMultiple  (multiplier 1.0 → C = α₀ = 1/√N)
let cConst = graph.constant(Double(spec.rezeroAlphaInit) * NetworkArchitecture.rezeroTanhCeilingMultiple, dataType: alpha.dataType)
let alphaBounded = graph.multiplication(
    cConst,
    graph.tanh(with: graph.division(alpha, cConst, name: nil), name: nil),
    name: "\(prefix)_res_scale_tanh")
branch = graph.multiplication(seOut, castInForward(alphaBounded), name: "...")
```

Design choices:

- **tanh, not a hard clamp.** A hard `[0, C]` projection has zero gradient past
  `C` (the dead-zone / parameter-drift problem above). `C·tanh(α/C)` is a smooth
  saturating bound with a gradient that is **alive everywhere** — it shrinks as
  `α` approaches `C`, giving a soft restoring pressure so `α` settles in the
  interior instead of pinning against a wall. It is near-identity for small `α`
  (`≈ α` when `α ≪ C`), so it starts at the `1/√N` init and behaves normally in
  the healthy range. (sigmoid was rejected earlier for throttling movement near
  0; tanh has unit slope at 0, so no slow-ignition.)
- **In the forward, not a post-step projection.** A saved `α` is still bounded on
  reload, and the branch weights keep full gradient.
- **`C = α₀ = 1/√N`** (`rezeroTanhCeilingMultiple · α₀`, multiplier 1.0). The raw
  `α` ratchets up no matter the bound (one-way gradient), so the *effective* α
  saturates **at** `C` across all blocks — the cap must therefore be variance-safe
  *when saturated*. With all N blocks at α = C, residual-stream variance scales as
  `Σα² = N·C²`; `C = 1/√N` gives `Σα² = 1`. This is depth-aware (tracks the `1/√N`
  init automatically) and ties to whatever per-group init the user sets in
  Build-New-Model. **`C = 1.0` was tried and failed** (see below): it let effective
  α saturate at ~0.95 (`Σα² ≈ 4.5`), the stream mean compounded, and the run broke
  ~step 5800.

### What this does **not** do

- It does **not** fix already-broken checkpoints. Bounding a model that trained
  *with* α≈30 (or pinned at 2.0) mismatches its convs (they adapted to the large
  α) — probing such a checkpoint under any inference-side bound leaves it broken.
  Only a **from-scratch retrain** with the soft-bound produces a healthy model.
- It does **not** add weight decay to `α`; the tanh soft-bound is the sole
  mechanism. (Adding `α` to the weight-decay set is an alternative — see below.)

## Background / prior art

- ReZero (Bachlechner et al. 2020, *ReZero is All You Need*) — the per-layer
  learnable residual scalar, init **0** (exact identity). Leaves `α` unconstrained
  and reports it staying small in their (LayerNorm-Transformer / fp32) regime, so
  the runaway never surfaced for them. This project uses `1/√N` init (a
  Fixup/SkipInit-flavored variant) for a faster start with no dead blocks — which
  starts `α` already off the floor and, unguarded, opened the runaway.
- LayerScale (Touvron et al. 2021, CaiT) — per-**channel** learnable scale, init a
  *small nonzero* constant (1e-4…1e-6, smaller for deeper). Fixes both the
  dead-block and fast-ignition problems but is still unbounded; the clamp here is
  complementary.
- DeepNorm (Wang et al. 2022) — fixed (non-learned) depth-scaled residual
  constant; sidesteps a learnable runaway entirely.
