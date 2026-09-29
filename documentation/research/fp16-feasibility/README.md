# fp16 feasibility: inference and training (2026-09-28)

Read-only research. No project source was changed, and nothing was built, run or tested in the app. Everything below comes from reading the code and from numpy emulation of real checkpoints. **Measured** means computed here on real weights and positions. **Inferred** means reasoned from code or numbers and not tested on the GPU.

Line numbers are for the working tree on 2026-09-28. `Network/ChessNetwork.swift` was being edited at the time, so its line numbers may drift; function names are given too. Paths are relative to `DrewsChessMachine/DrewsChessMachine/` unless they start with `documentation/`.

## Recommendation

- **Training: stay on bf16 and implement `documentation/plans-active/HEAD_NUMERICS_PLAN.md`. Don't invest in fp16 training now.**
  - With fp32 head tails, the bf16 body costs little:
    - value KL ≤ 8.5e-5 under pessimistic per-op rounding, and ~3e-11 under the emulation calibrated against the bot's recorded outputs;
    - value ties 0.
  - fp16 training needs a whole dynamic loss-scaling system plus several fp32 islands. The measured loss-scale window is narrow: 2^10 to 2^14 (details below).
  - Nothing in the repo shows fp16 is faster. bf16 has measured speedups on M5; fp16 has never been timed.
  - Plan Phase 2 (fp32 targets and mean-centering before softmax) zeroes the shared-offset gradient in *either* format, so fp16's slower drift is no reason to switch.
- **One addition to the head plan (measured).** On the lines with large policy offsets, fp32 tails that start at the final conv still leave real policy damage:
  - h7vI and Xuub: policy KL 1.5–3.3e-3, and 3.5–5.3% of positions lose their fp64 top move.
  - Cause: bf16 rounding of the policy features, multiplied by the large mean row of `policy.conv`.
  - Starting the fp32 policy tail at the pre-BN normalize (BN + ReLU + final conv in fp32) cuts that 7–14× (table 4b).
- **fp16 inference works today and is sound.** On a bf16-trained checkpoint it is ~50× better at the heads than bf16. But fp32 tails fix the heads in bf16 too.
  - For single-position consumers (UCI, the Lichess bot), fp32 compute is the simplest exact choice. Its cost wasn't measured, but it's one forward per move.
  - There's no dtype-override flag outside session resume.

## 1. Current state of fp16 support

| Path | fp16 status | Where |
|---|---|---|
| `ComputeDataType.float16` | Defined, per-model (`compute_data_type` in the embedded architecture). `validate()` never rejects it. The doc comment is garbled: "Same 10-bit mantissa precision as bf16's 7-bit". Its "ANE-native, so inference may run faster" is unmeasured. | `Network/NetworkArchitecture.swift:320-330` |
| Inference graph | Works. Input placeholder is fp32 with an in-graph `board_input_cast`. Weights are stored as fp16 graph variables. Tower, BN/LN, SE, ReZero and heads all run in fp16. Policy is widened to fp32 after the head (`policy_output_f32`). Value probs and scalar are read back as fp16 and widened on the host (vImage). | `ChessNetwork.swift` ~:675, ~:958, `mpsDataType(for:)` ~:2353 |
| Batched evaluator (self-play, arena) | Same builder, so fp16 works. | `evaluateBatched` |
| Config D (`bf16CastInForward`) | bf16 only. fp16 is excluded by `== .bFloat16`. | `ChessNetwork.swift` ~:571; `App/SessionController.swift:963` |
| Checkpoints | Disk is **F32 only**, and the dtype lives only in the architecture JSON. fp16 models save and load; the round-trip test passes. `Models/` files hold the working (fp16-exact) weights, so the fp32 masters are discarded there (same gap as bf16). | `Persistence/SafetensorsFile.swift:22, :109, :178` |
| Trainer | Accepts fp16. The dtype-generic fp32-master path is used: `useMaster = dtype != .float32 && !bf16CastActive`. fp16-only fixes: masked-logit bias −3e4 instead of −1e9, and the entropy log-path in fp32. **No loss scaling, no skip-step.** A non-finite loss or gNorm throws `nonFiniteLoss` *after* the weights were updated, and the session suspends. | `Training/ChessTrainer.swift:4034`, `:2608`, `:3351`, `:6845-6860` |
| gNorm readback | The fp32 norm is narrowed to the compute dtype for readback. In fp16, any norm > 65504 reads back as +inf, even though clipping uses the fp32 value. | `ChessTrainer.swift:3911-3915` |
| UI / CLI | Build-New-Model offers fp16 with no warning. There's no `--precision` flag; precision comes from the architecture. Play-and-Train does not refuse fp16. UCI, the bot and human play use the file's dtype. The only override is `forceFloat32` on session resume. | `App/UpperContentView/BuildNewModelView.swift:123`; `App/SessionController+Checkpoint.swift:648-653` |
| Tests | `FP16ConversionTests` pass. In `FP16ComputePathTests`, forward and safetensors round-trip pass. Its four trainStep cells fail by design as tripwires. | `DrewsChessMachineTests/FP16ComputePathTests.swift` |

**Known failures and how to classify them**

- **Universal fp16 bugs, fixed in `98f1a4d`:**
  - the −1e9 mask overflowed to −inf, so 0·(−inf) = NaN;
  - the entropy ε = 1e-7 is an fp16 subnormal and was lost, so log(0) = −inf gave NaN.
  - These are why the trainer's comment says fp16 denormals flush on MPS. That claim is inferred from the NaN, not tested directly.
- **The remaining tripwire failure:** a "gradient norm overflows to ∞ within a step or two" (`FP16ComputePathTests.swift:24-37`). It was seen on the macOS-27 beta. Re-baselining on a non-beta OS is still pending. The findings doc §5, ROADMAP :1558 and CHANGELOG `b25f37e` still say "NaN on the first step", which predates `98f1a4d`.
- **Those cells train on synthetic data, not replay data** (`ChessTrainer.swift:4449-4491`):
  - uniform [0,1) in all 30 planes;
  - an all-ones legal mask;
  - random move indices.
  - On that data, **fp32** plateaus at gNorm ~1e7 (`MacOS27NaNIsolationTests.swift:462`), which is >65504.
  - Inferred: the tripwire cannot pass in fp16 without loss scaling, whatever the OS, and it says little about real-data fp16 training. Its readback would also show inf from the narrowing alone.
- **Real data is far from that:** the Ejp0 self-play log `dcm_log_20260806-215333.txt` (first 734 `[STATS]` lines) has gNorm 2.1–3.6, median 2.78.
- **Batch-1:** bf16 batch 1 is finite. fp16 fails at batch 1 and 64 (findings doc §5).
- **`MacOS27NaNIsolationTests` has no fp16 cells.** Its bf16/fp32 matrix and the §1–§4 findings (dual-write stomp, layout conversion, reduced-precision fast-math, ANE fp32 noise) are beta-OS issues. By the standing rule, they are not to be coded around.

## 2. Range analysis on real checkpoints (measured)

Identity comes from `__metadata__`, not filenames.

| model_id | training_step | native dtype | architecture | source |
|---|---|---|---|---|
| 20260727-1-Ejp0 | 681000 | bf16 | v5: 7×7 stem, 2×[15×15 @64, SE scale_and_bias, pre-act, ReZero, LN], intermediate_conv 512, WDL | `Models/20260702-Qeu8-resume3-replay-step681000.safetensors` |
| 20260714-1-h7vI | 336610 | bf16 | v5: 5×[7×7 @128 …], intermediate_conv 128, WDL fc 128 | `Models/20260713-v5cont-resume-replay-step336610.safetensors` |
| 20260802-2-Xuub | 49374 | bf16 | same as h7vI | `Models/20260802-v5cont-resume3-replay-step49374-DO-NOT-RESUME.safetensors` |
| 20260627-7-mUF5 | 28797 | fp32 | v3: 8×[3×3 @128, post-act, attenuate SE, gated add], simple_conv | `Models/20260627-v3_8block_3x3-step28797-FINAL-frozen.safetensors` |
| 20260514-1-KbHZ-22 | 532369 | fp32 | same as mUF5 | champion of session `20260611-212501-…-Ko63-manual` |

**How it was measured**

- **Positions:** 1,180 positions: 900 from corpus w3aA5b shard 45, plus 280 from the first 4 Lichess bot games. This is the survey's `posset.pkl`.
- **Forward:** `scripts/fwd16.py`, which generalizes the survey's validated forward. It matches `bf16-head-offset/scripts/fwd3.py` to within 0.0 on all three architectures.
- **Caveat on KbHZ:** its fp64 CE under current code is implausible (value 1.63, policy 4.49). Its rounding numbers are valid as range measures, but not as CE levels. mUF5 is the clean fp32 reference (CE 0.745 / 2.30).

### 2a. Weights

| model | trainable elements | max \|w\| | fraction fp16-subnormal (0 < \|w\| < 2^-14) | fraction → 0 in fp16 | worst tensor (subnormal fraction) | BN running_var min / max |
|---|---|---|---|---|---|---|
| Ejp0 681k | 3,927,569 | 14.56 | 0.684% | 4.1e-6 | value.fc1.weight 2.1% | 1.9e-4 / 4.81 |
| h7vI 336610 | 8,443,668 | 24.25 | 0.417% | 2.0e-6 | blocks.2 SE fc1.bias 3.1% | 1.29e-3 / 340 |
| Xuub 49374 | 8,443,668 | 22.75 | 0.441% | 3.3e-6 | value.fc1.bias 0.8% | 5.9e-4 / 5,984 |
| mUF5 28797 | 2,479,313 | 1.78 | 0.115% | 4.0e-7 | blocks.4 SE fc1.bias 3.1% | 0.024 / 11.8 |
| KbHZ-22 532369 | 2,479,313 | 1.72 | 0.117% | 0 | blocks.2 SE fc2.bias 2.3% | 0.012 / 10.9 |

- **No weight comes near 65504.**
- **bf16-exact weights convert to fp16 exactly** whenever they're in the normal range: fp16 has 11 significant bits, bf16 has 8. Only the 0.1–0.7% below 2^-14 lose bits.
- **Flushing those to zero changes nothing measurable** (the "FTZ" rows below).

### 2b. Activations (inference forward, every op)

| model | max \|x\| (op) | max L1 accumulation bound Σ\|x\|\|w\| (op) | max x² at a BN/LN input (op) | min BN batch variance (op) | min LN variance | worst fraction of fp16-subnormal activations (op) |
|---|---|---|---|---|---|---|
| Ejp0 681k | 987 (value fc2) | 1,002 (value fc2) | 152 (tower BN) | 1.99e-4 (b1.bn1) | 0.080 | 0.15% (b1.bn1 x−μ) |
| h7vI 336610 | 1,055 (value fc2) | 1,067 (value fc2) | 4,973 (policy pre-BN, \|x\| 70.5) | 1.24e-3 (b2.bn1) | 1.38 | 2.2% (b1 SE fc2 matmul) |
| Xuub 49374 | 968 (value fc2) | 984 (value fc2) | **142,429 (policy pre-BN, \|x\| 377)** | 6.1e-4 (b2.bn1) | 1.69 | 11.4% (b1 SE fc2 matmul) |
| mUF5 28797 | 34 (b7 add) | 104 (b7 conv1) | 489 (b7.bn1) | 0.025 (stem BN) | — | 0.07% |
| KbHZ-22 532369 | 34 (policy conv) | 75 (policy conv) | 354 (value BN) | 0.012 (stem BN) | — | 0.09% |

- **Inference: nothing overflows.** Even the L1 bound, which caps every partial sum in any accumulation order, is ≤ 1,067. So an fp16 accumulator cannot overflow inside any conv or matmul.
- **Training forward: Xuub's policy pre-BN input reaches |x| = 377.**
  - Its square is 142,429 > 65504. A batch-variance op that materializes x² or (x−μ)² in fp16 overflows there (inferred; MPSGraph's `variance` internals are not documented).
  - The batch variance itself (6,846) and the running variance (5,984) fit.
- **Smallest variances:** BN batch variances go down to 1.99e-4, only 3.3× above fp16's smallest normal.
  - `normalize` uses ε = 1e-5, which is an fp16 subnormal (`ChessNetwork.swift` batchNorm ~:2550, layerNorm ~:2598).
  - If MPS flushes it, a channel whose variance falls below 2^-14 divides by √0 (inferred risk). No such channel was measured.

### 2c. Head outputs and whole network against fp64

- **Head levels (fp64):**
  - Value shared logit median: +510.5 (Ejp0), +515.2 (h7vI), +513.2 (Xuub).
  - Policy legal mean: −42.2, −177.6, −265.3.
- **Rounding steps there:**
  - value: bf16 2 / 4 / 4 against fp16 0.25 / 0.5 / 0.5;
  - policy legal max: bf16 0.25 / 1 / 2 against fp16 0.031 / 0.125 / 0.25.
- **The emulations in the table:**
  - "calibrated" is the survey's fit to the bot's records: float64 internals, the policy pre-BN output rounded, and the fused head output rounded.
  - "per-op" rounds every op (pessimistic).
  - "+ fp32 tails" leaves the final conv / fc2, their bias and the softmax in fp32.
- **top-1 lost** counts positions where the fp64 best move is no longer the strict maximum. For scale, fp64 has a top-2 gap < 0.01 in 0.8–1.2% of positions.

| model | emulation | value ties | value KL | value ΔCE | \|Δv\| mean | policy KL | policy top-2 ties | top-1 lost | policy ΔCE |
|---|---|---|---|---|---|---|---|---|---|
| Ejp0 681k | bf16 calibrated | 54.6% | 0.108 | +0.145 | 0.284 | 2.36e-3 | 11.3% | 0.1% | +0.0052 |
| | fp16 calibrated | 4.8% | 1.96e-3 | +0.0012 | 0.037 | 3.64e-5 | 1.1% | 0 | −0.0003 |
| | bf16 per-op | 58.3% | 0.110 | +0.143 | 0.289 | 2.48e-3 | 11.9% | 0.1% | +0.0053 |
| | fp16 per-op | 4.5% | 1.97e-3 | +0.0024 | 0.037 | 3.89e-5 | 1.4% | 0 | −0.0001 |
| | fp16 per-op, subnormals flushed | 4.5% | 2.03e-3 | +0.0032 | 0.038 | 3.89e-5 | 1.4% | 0 | −0.0001 |
| | bf16 calibrated + fp32 tails | 0 | 2.9e-11 | ~0 | — | 3.28e-5 | 0 | 0.59% | +0.0005 |
| | fp16 calibrated + fp32 tails | 0 | 2.9e-11 | ~0 | — | 5.4e-7 | 0 | 0 | −0.00003 |
| | bf16 per-op + fp32 tails | 0 | 1.91e-5 | +0.0008 | 0.0033 | 1.65e-4 | 0 | 1.0% | −0.0005 |
| | fp16 per-op + fp32 tails | 0 | 3.5e-7 | +0.00003 | 0.0004 | 2.87e-6 | 0 | 0.08% | −0.0002 |
| h7vI 336610 | bf16 calibrated | 97.7% | 0.272 | +0.230 | 0.310 | 3.64e-2 | 43.6% | 0.1% | +0.056 |
| | fp16 calibrated | 26.5% | 4.48e-3 | +0.0054 | 0.054 | 5.84e-4 | 7.3% | 0 | −0.0005 |
| | bf16 per-op | 98.1% | 0.254 | +0.213 | 0.299 | 3.67e-2 | 43.8% | 0.2% | +0.056 |
| | fp16 per-op | 26.9% | 4.53e-3 | +0.0059 | 0.054 | 5.93e-4 | 7.1% | 0 | +0.0007 |
| | bf16 calibrated + fp32 tails | 0 | 7.1e-11 | ~0 | — | **1.53e-3** | 0 | **3.47%** | +0.0036 |
| | fp16 calibrated + fp32 tails | 0 | 7.0e-11 | ~0 | — | 2.32e-5 | 0 | 0.42% | −0.0003 |
| | bf16 per-op + fp32 tails | 0 | 7.37e-5 | +0.0002 | 0.0066 | **1.99e-3** | 0 | **4.3%** | +0.0048 |
| | fp16 per-op + fp32 tails | 0 | 1.22e-6 | +0.00007 | 0.0009 | 3.07e-5 | 0 | 0.5% | −0.0002 |
| Xuub 49374 | bf16 calibrated | 96.8% | 0.267 | +0.332 | 0.306 | 0.166 | 62.4% | 0 | +0.0007 |
| | fp16 calibrated | 18.0% | 4.33e-3 | +0.0032 | 0.055 | 2.39e-3 | 13.4% | 0 | +0.0069 |
| | bf16 per-op | 97.5% | 0.290 | +0.324 | 0.333 | 0.168 | 62.8% | 0 | +0.015 |
| | fp16 per-op | 16.2% | 4.26e-3 | +0.0070 | 0.054 | 2.40e-3 | 13.1% | 0 | +0.0077 |
| | bf16 calibrated + fp32 tails | 0 | 6.2e-11 | ~0 | — | **2.98e-3** | 0 | **4.32%** | +0.0080 |
| | fp16 calibrated + fp32 tails | 0 | 6.2e-11 | ~0 | — | 4.67e-5 | 0 | 0.34% | −0.0003 |
| | bf16 per-op + fp32 tails | 0 | 8.52e-5 | −0.0015 | 0.0068 | **3.27e-3** | 0 | **5.25%** | +0.0057 |
| | fp16 per-op + fp32 tails | 0 | 2.57e-6 | −0.0002 | 0.0008 | 5.10e-5 | 0 | 0.34% | −0.0001 |
| mUF5 28797 (fp32 line, hypothetical) | bf16 per-op | 1.4% | 2.1e-5 | +0.0003 | 0.0039 | 1.02e-3 | 9.2% | 0.8% | −0.0008 |
| | fp16 per-op | 0.2% | 1.6e-6 | −0.0001 | 0.0012 | 1.56e-5 | 1.4% | 0.1% | +0.0002 |
| KbHZ-22 (fp32 line, hypothetical) | bf16 per-op | 0 | 2.2e-5 | — | 0.0023 | 1.03e-3 | 12.8% | 1.7% | — |
| | fp16 per-op | 0 | 5.0e-7 | — | 0.0003 | 1.78e-5 | 1.9% | 0.3% | — |

- **fp16 heads at the current offsets:**
  - value ties 5–27% and ΔCE +0.001 to +0.007: "degraded" by the survey's rubric, not "BAD";
  - policy KL 4e-5 to 2.4e-3.
- **fp16 per-op vs bf16 per-op:** everywhere, fp16 is 13–70× lower in KL, for both value and policy.
- **Where the fp16 error comes from:** almost all of it is the heads. Flushing fp16 subnormals changes nothing material.

## 3. Training in fp16

### 3a. Gradient magnitudes (measured)

**Method.** An exact numpy backward of the trainer's loss through the heads:
- the loss: signed-advantage CE + complement CE, illegal-mass penalty, and value CE with ε = 0.1 / 0.013;
- batch B = 4096, resampled from the 900 corpus positions, with the batch mean giving the 1/B factor;
- the activation gradient taken back to the tower output, using inference-mode BN as the linear map through the heads' BN.

"energy" means the fraction of Σg² carried by the elements in question.

| tensor (Ejp0 681k) | max \|g\| | median \|g\| | subnormal at S = 1 (elements / energy) | → 0 at S = 1 | subnormal at S = 2^10 | subnormal at S = 2^14 | 65504 / max (largest safe S) |
|---|---|---|---|---|---|---|---|
| policy dlogit, legal cells | 3.9e-4 | 2.8e-6 | 97.8% / 5.9% | 0.5% | 1.4% | 0.1% | 1.7e8 |
| policy dlogit, illegal cells | 1.5e-5 | 2.1e-12 | 100% / — | 100% | 100% | 99.8% | 4.4e9 |
| value dlogit | 2.4e-4 | 5.5e-5 | 52.6% / 2.5% | 0.1% | 0.3% | 0 | 2.7e8 |
| dW policy.conv | 0.125 | 8.2e-5 | 48.3% | 0 | 0 | 0 | 5.2e5 |
| dW value.wdl_fc2 | **2.79** | 6.5e-3 | 2.2% | 0 | 0 | 0 | **2.35e4** |
| dW value.fc1 | 0.078 | 1.1e-4 | 41.6% | 6.2% (near-zero rows, likely dead ReLU units) | 6.6% | 5.7% | 8.4e5 |
| dW policy.pre_conv | 0.115 | 2.6e-3 | 4.5% | 0 | 0 | 0 | 5.7e5 |
| activation gradient at tower output | 3.8e-3 | 1.2e-7 | 99.7% / 1.2% | 29.9% | 38.5% (energy 2.7e-7) | 15.7% | 1.7e7 |

Same measurements on the other checkpoints:

| checkpoint | tower-output activation gradient: median, subnormal at S = 1 (elements / energy), at 2^14 | policy legal dlogit subnormal at S = 1 | value dlogit subnormal | largest weight-gradient max (tensor) |
|---|---|---|---|---|
| Ejp0-60 trainer @ 1,118,017 (bf16) | 1.6e-7; 99.8% / 2.4%; 10.0% | 96.8% | 43.2% | 0.65 (policy.pre_conv) |
| h7vI 336610 | 7.8e-9; 99.9% / 3.1%; 47.2% | 98.0% | 50.5% | 0.91 (policy.conv) |
| KbHZ-23 trainer @ 532,369 (fp32) | 1.9e-7; 99.9% / **74.5%**; 32.0% | 96.5% | 29.9% | 1.60 (policy.conv) |

**Tower weight gradients, from the saved optimizer velocity** (`opt.*.velocity` in `trainer.safetensors`; v = μv + g):
- Implied per-step \|g\| ≈ \|v\|(1−μ) for the persistent part and \|v\|√(1−μ²) for the noise part. This is an estimate, not a direct measurement.
- **Ejp0-60 @ 1,118,017** (μ 0.9): median \|v\| 5.2e-4. Estimated \|g\| < 2^-14 for 55% of elements (persistent) / 20% (noise). Tower convs: 55% / 20%.
- **KbHZ-23 @ 532,369** (μ 0.65): median \|v\| 4.1e-4. 24% / 12%.

**Conclusions**
- **Without loss scaling**, fp16 keeps the large gradients but makes the bulk subnormal: 97–98% of legal-cell policy logit gradients and ~99.8% of the tower-entry activation gradients.
  - If subnormals survive, the energy lost is modest: 1–11%, but **74.5% at KbHZ's tower entry**.
  - If MPS flushes fp16 subnormals, as the trainer's own comment says (`ChessTrainer.swift:3337-3339`; inferred from the entropy NaN), all of it is lost.
  - So loss scaling is required (inferred from measured magnitudes).
- **The window for a static scale:**
  - Floor: S ≥ 2^10 brings the subnormal energy to ≤ 6.5e-6 on every tensor measured.
  - Ceiling: S ≤ 2^14 is set by `dW value.wdl_fc2` on Ejp0 (max 2.79, so 65504/2.79 = 2.35e4). At S = 2^16, 10% of that tensor's elements overflow.
  - That leaves a factor of 16, and tower-internal maxima (BN backward divides by σ; min σ ≈ 0.014) were not measured.
  - So only **dynamic** scaling is safe (inferred).
- **Illegal-cell logit gradients stay subnormal at any usable S** (median 2e-12). They only matter if the logit gradient lives in fp16. With the plan's fp32 tails and fp32 CE, it never does: the tail's fp32 conv backward sums them before anything is narrowed.

### 3b. What dynamic loss scaling would look like here (MPSGraph has no AMP)

1. **Keep the loss path fp32.** Don't narrow `policyLoss`, `valueLoss`, `policyEntropy` or `illegalMassPenalty` back to the compute dtype (`narrowReductionResult` at `ChessTrainer.swift:3050-3057, :3211-3214, :3381-3388, :3413-3420`). Make the loss-weight placeholders fp32 (`:3782-3796`) and form `total` in fp32 (`:3811-3845`).
   - Today `total` is in the compute dtype, so a scaled total would overflow fp16 at S·loss > 65504. Loss ≈ 3 in steady state, and 120–166 in the synthetic tests.
2. **`scaledLoss = total × S`**, with S an fp32 scalar placeholder, and `graph.gradients(of: scaledLoss, …)` (`:3849`). The gradient enters fp16 at each `widenForReduction` cast as S/B per position.
3. **Unscale in fp32:** `gradF = cast(grad, fp32) / S` (`:4166`), before the norm (`:3864-3887`) and the clip (`:3972-3981`).
4. **Skip in-graph.** Make `ok = isFinite(gradSumOfSquares)` (fp32). Wrap every state write in `select(ok, new, old)`:
   - velocity (`:4171-4173`);
   - master (`:4192-4207`);
   - working-copy sync, fused (`:4213`) and split (`buildWorkingSyncOps` `:2412-2431`);
   - BN running-stat masters and their working syncs (`:4224-4255`).
   - An overflow step then leaves all state bit-identical. Read `ok` back.
5. **Host-side scale controller:** halve S on a skipped step. Double it after N consecutive clean steps (PyTorch's default is 2000). Clamp it to a range. Start at 2^14. Stop with an error if S hits the floor, or after K consecutive skips.
6. **`nonFiniteLoss` handling** (`:6845-6860`): a skipped step is not a divergence. Only a non-finite *unscaled loss*, or a scale collapse, should suspend. Decide whether skipped steps advance `completedTrainSteps` and the warmup.
7. **Read gNorm back in fp32** (`:3911-3915`). This is needed anyway: fp16 turns anything > 65504 into inf.
8. **Persist S and the clean-step counter in the session**, and log them. Follow the CLAUDE.md parameter checklist if the initial scale or growth interval become parameters: `SessionCheckpointState`, a `[RESUME-PARAM]` block, `[STATS]` fields such as `lossScale=` and `skipped=`, and `results.json`.

### 3c. fp32 masters, optimizer state, weight decay, momentum: already done

- `useMaster` is true for fp16 (`ChessTrainer.swift:4034`).
- The masters and BN running-stat masters are fp32 (`:4090-4136`).
- Velocity is fp32 (`:4063-4068`).
- lr, weight decay, clip and μ placeholders are fp32 (`:3751-3810`).
- Decoupled weight decay is applied to the fp32 master (`:4195-4202`).
- The BN EMA runs on fp32 masters (`:4224-4255`).
- Nothing here is dtype-specific. It only needs the skip gating above.
- **Why the masters matter more in fp16 (inferred from numbers):** a per-step update of lr·v ≈ 1e-3 × 5e-4 = 5e-7 is far below the fp16 step at a typical weight (median \|w\| 6e-3 has an fp16 step of 3.8e-6). Without masters, every such update would round away.

### 3d. What must be an fp32 island under fp16

| Op | Why (measured unless marked) | Where |
|---|---|---|
| CE targets + `softMaxCrossEntropy` (policy, complement, value) | Targets built in fp16 don't sum to 1 (table 4c). The dlogit gradients are 97–98% subnormal. The plan's Phase 2 does this for bf16 as well. | `ChessTrainer.swift:2704-2834, :3140-3206` |
| Illegal-mass softmax | Illegal-cell gradients are 100% subnormal at S = 1 and 99.8% at 2^14. | `:3395-3420` |
| Entropy log-path | ε = 1e-7 is subnormal. Already fp32. | `:3351` |
| BN and LN batch statistics (`graph.mean`/`graph.variance`) and the ε add | x² up to 142,429 (Xuub policy pre-BN); ε = 1e-5 is subnormal; batch variance down to 2e-4 | `ChessNetwork.swift` batchNorm ~:2491-2492, ~:2550; layerNorm ~:2593-2598 |
| Head tails (final conv / fc2, bias, value softmax) | value ties 5–27% in fp16 at today's offsets. Plan Phase 1a. | `ChessNetwork.policyHead` / `valueHead` |
| Advantage RMS normalization | Values ≤ 4 are fine; the ε 1e-6 is subnormal but sits under the 0.04 floor. Keep it fp32 for cleanliness (inferred). | `ChessTrainer.swift:2944-2972` |
| Value baseline feed | Plan Phase 2 item 8 | `:2582-2587` |
| gNorm readback | 65504 cap | `:3915` |

**Not needed in fp32**
- **SE sigmoid:** in fp16, 0–4.6% of gates round to exactly 1, against 0–22% in bf16 (h7vI block 0: 22.4% bf16, 4.6% fp16).
- **ReZero tanh:** see 3e.
- **Tower convs:** see the accumulation bound in 2b.

### 3e. ReZero tanh (measured)

- **Ejp0 (both blocks):** α = 1.734375, C = 0.5, so tanh(α/C) = 0.998060.
  - bf16 rounds it to **1.0**, and the derivative computed from the output is **0**.
  - fp16 gives 0.998047, with derivative 0.00390 (fp64 value 0.00388).
- **Confirmed on a real trainer:** the Ejp0-60 `trainer.safetensors` at 1,118,017 has α master = 1.734375 and **velocity exactly 0.0 in both blocks**. α is frozen under bf16.
- **h7vI:** blocks 2–3 saturate to 1.0 in bf16. Blocks 0, 1 and 4 get 0.99609, which gives derivative 0.0078 against the true 0.0058/0.0047.
- **fp16 tracks within 3–4% in every block.** fp16 is strictly better here.

## 4. bf16 + fp32 head tails (current plan) against fp16

### 4a. Summary comparison

| | bf16 + fp32 tails (plan) | fp16 (+ fp32 tails) |
|---|---|---|
| Exponent range | fp32's; no loss scaling needed | max 65504, min normal 6.1e-5; needs dynamic loss scaling (3b) |
| Mantissa | 8 bits (unit roundoff 2^-9) | 11 bits (2^-12), 8× finer |
| Value head at today's offsets | exact with tails (KL ~3e-11 calibrated) | same with tails; 5–27% ties without |
| Policy on large-offset lines (h7vI, Xuub) | KL 1.5–3.3e-3, top-1 lost 3.5–5.3% with tails from the final conv; KL 2.4–2.9e-4, 1.8% with tails from pre-BN (4b) | KL 2.3–5.1e-5, top-1 lost 0.3–0.5% |
| Body (everything but the heads) | value KL ≤ 8.5e-5 per-op; the survey's calibrated body audit on Ejp0 gave KL 2e-5 | 13–70× lower KL |
| Training machinery | exists and works (fp32 masters, EMA, velocity) | same, plus loss scaling, skip-step, fp32 BN/LN stats, fp32 loss path |
| Speed on Apple Silicon | measured bf16 ≈ 7.6× / 1.5× vs fp32 on M5 (apple10), 1.08× on M4 (`documentation/plans-active/RUNTIME_ARCHITECTURE_CONFIG_PLAN.md` §9, `tools/bf16-probe.swift`). This machine is an Apple M5 Max. | **never measured in the repo.** `bf16-probe.swift` doesn't time fp16. |
| Neural Engine | not usable for bf16/fp32. The console's "ANE cannot handle intermediate tensor type fp32" is MPSGraph trying and falling back (findings §4). | The ANE is fp16-only, so an fp16 *inference* graph might be placed there. Unmeasured, and it could be faster or slower. Inferred to be irrelevant for training, whose fp32 masters and casts keep it off the ANE. |
| Memory | 2 bytes per working element | 2 bytes (`bytesPerWeightElement`) |
| ReZero | α frozen (derivative 0) once tanh(α/C) rounds to 1 | live |

### 4b. Where the policy fp32 tail should start (measured, per-op body)

| model | tail from the final conv (plan) | tail from the pre-BN normalize | tail from the pre-conv | fp16 body, tail from the final conv |
|---|---|---|---|---|
| Ejp0 681k | KL 1.65e-4, top-1 lost 1.0% | 1.24e-4, 1.0% | 1.13e-4, 0.85% | 2.9e-6, 0.08% |
| h7vI 336610 | 1.99e-3, 4.3% | 2.93e-4, 1.8% | 2.61e-4, 1.6% | 3.1e-5, 0.5% |
| Xuub 49374 | 3.27e-3, 5.3% | 2.35e-4, 1.9% | 2.10e-4, 2.0% | 5.1e-5, 0.3% |

- **Why:** the policy features (the pre-BN output, stored in the compute dtype) are multiplied by `policy.conv`'s large mean row. That mean row is itself part of the offset, with norm 3–8 on BAD lines. The product is a per-square error, which softmax does not cancel.
- **Moving the start of the fp32 region one BN earlier removes most of it** for `intermediate_conv` heads.

### 4c. Would the shared-offset drift still happen in fp16?

**Yes, if the targets are built in fp16 and nothing pulls the offset back.**

Measured per-position sum of the targets minus 1 (`scripts/shared16.py`, `grad16.py`):

| target | fp64 | bf16 | fp16 |
|---|---|---|---|
| value (ε = 0.013), every class | 0 | **+8.54e-4** | −1.22e-4 (7.0× smaller, opposite sign) |
| policy positive target (ε = 0.1), batch mean | 0 | −7.96e-4 | −1.12e-4 (7.1× smaller) |

Measured gradient along the shared direction, as a batch mean (Ejp0 681k batch):

| | fp64 | bf16 | fp16 |
|---|---|---|---|
| policy, positive branch | 0 | 4.80e-4 | 0.77e-4 |
| policy, complement branch | **6.71e-4** | 9.21e-4 | 7.23e-4 |
| policy, total | 6.71e-4 | 1.40e-3 | 8.00e-4 |

- **New finding (measured).** About half the bf16 policy push is not a rounding effect.
  - When \|legal\| = 1, the complement target sums to ε = 0.1 instead of 1, in exact math. That's 1.5% of positions.
  - It gives 6.71e-4 of shared push per step in fp64 as well. That is 48% of the bf16 total on this batch.
  - It exists in fp32 and fp16 alike, whenever `signedAdvantageComplementCE` is on. It was on in the Ejp0 self-play session and off in the fp32 KbHZ session; the setting for the replay runs wasn't checked.
  - The head plan's Phase 2 item 1 already covers this case ("give that position zero complement weight"). These numbers show it matters as much as the rounding.
- **Damage threshold (measured).** fp16's step is 8× finer at the same magnitude.
  - fp16 at Xuub's policy level (−265) does the same damage as bf16 at Ejp0-681k's level (−42): KL 2.4e-3 / top-2 ties 13.4%, against 2.4e-3 / 11.3%. That is a 6.3× larger offset for the same harm.
  - fp16 value heads at +510 are "degraded", while bf16 is "BAD". For bf16's value damage at 512, fp16 would need \|logit\| ≈ 4096.
- **How much later it would bite (inferred; assumes roughly linear drift, which the value offset doesn't strictly follow):**
  - value: 8× threshold ÷ (1/7.0 drift rate) ≈ **56× later**;
  - policy: 6.3× ÷ 0.57 ≈ **11× later** with the complement case as it is, or 6.3× ÷ 0.16 ≈ **39× later** with it fixed.
  - For scale: the Ejp0 lineage's value went BAD about 30–75k steps into Ejp0. Its policy reached −138 at 1.397M.
- **Plan Phase 2's mean-centering makes the shared gradient exactly zero in either format.** So this difference disappears once the plan lands.

## 5. Checklist

### 5a. fp16 inference only (small)

1. **Nothing is needed for correctness.** The fp16 forward, safetensors (F32 on disk) and readback all work and are tested.
2. **A compute-dtype override to run an existing bf16 model in fp16.** Today there's only `forceFloat32` at session resume (`App/SessionController+Checkpoint.swift:648-653`). UCI, the bot and human play take the file's dtype. Or use fp32 for those single-position paths.
3. **Apply the head plan's fp32 tails (dtype-generic)**, preferably starting the policy tail at the pre-BN (4b). At today's offsets, fp16 heads without tails still tie 5–27% of value outputs.
4. **Phase 0 audit:** flag any BN running_var < 2^-14 (ε = 1e-5 is subnormal). The lowest measured is 1.9e-4.
5. **Benchmark fp16 vs bf16 plies/hour** on this M5 Max, including whether MPSGraph uses the ANE, before choosing fp16 for self-play. Extending `tools/bf16-probe.swift` is the cheapest route. Not run here.
6. **Add a real-checkpoint fp16-vs-fp32 numerics test.** The existing forward test uses random weights, batch 4, and checks row sums to ±0.02.
7. **Fix the `ComputeDataType.float16` doc comment** (`NetworkArchitecture.swift:323-328`).

### 5b. fp16 training, fully and safely

1. **Everything in 5a.**
2. **Loss path in fp32, with dynamic loss scaling and an in-graph skip-step** (3b, items 1–8).
3. **fp32 islands** (3d): the CE targets and CE, illegal mass, BN/LN batch statistics and ε, the value baseline, and the gNorm readback.
4. **Head plan Phase 2** (fp32 targets renormalized, the \|legal\| = 1 complement case, mean-centering). Needed for bf16 anyway.
5. **Checkpoints:** no format change (F32 on disk). Persist the loss scale and clean-step count in `.dcmsession`. Full-precision `Models/` checkpoints (plan Phase 3) matter as much for fp16 as for bf16.
6. **Tests.** Existing tests need owner approval to change.
   - New real-replay fp16 train cells (`trainStepFromReplay`) with loss scaling.
   - A skip-step test: masters, velocity and BN stats stay bit-identical.
   - Scale-controller halve/double.
   - gNorm > 65504 reads back finite.
   - BN/LN statistics stay finite with inputs \|x\| > 256 under fp16.
   - Keep the synthetic-data tripwires. They can't pass without loss scaling and aren't representative of real data.
7. **Re-baseline on a non-beta macOS** before attributing any remaining fp16 failure to fp16 itself.
8. **UI:** decide whether Build-New-Model / Play-and-Train should warn about or gate fp16 training until this lands (`BuildNewModelView.swift:123`).
9. **Validation run:** an fp16 vs bf16 A/B from the same seed on corpus replay. Compare the loss curves, skip rate, S trajectory, `[STATS]` gNorm, and the Phase 0 audit at the end.

## Files

- `scripts/fwd16.py`: the instrumented forward for all saved architectures. It has named rounding points, capture, L1 accumulation bounds and normalization inputs.
- `scripts/range16.py`: sections 2a–2c for one checkpoint → `results/range_*.json|log`.
- `scripts/grad16.py`: 3a (head backward, loss-scale fractions, velocity) → `results/grad_*.json|log`.
- `scripts/tails16.py`: calibrated emulation with and without fp32 tails → `results/tails16.json`.
- `scripts/tailstart16.py`: 4b → `results/tailstart16.json|log`.
- `scripts/sat16.py`: ReZero and SE saturation → `results/sat16.json`.
- `scripts/shared16.py`: 4c shared-direction gradients → `results/shared16.json`.

**Reproducing**
- The scripts need numpy.
- They read checkpoints under `~/Library/Application Support/DrewsChessMachine/`.
- They read the position set from `posset.pkl`, built by `../bf16-head-offset/scripts/posset.py`. The path can be overridden with `DCM_POSSET`.
- Reproduction was checked: `range16.py` (mUF5) and `shared16.py` from this folder reproduce the saved JSON exactly.
