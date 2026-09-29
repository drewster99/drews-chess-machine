# Ejp0 @ step 681,000 — full bf16 audit of the Lichess bot's model (2026-09-28)

Read-only. No project source, build, app or test run was touched.

- **Model:** `Models/20260702-Qeu8-resume3-replay-step681000.safetensors` (15.0 MB). Identity from `__metadata__`: `model_id` 20260727-1-Ejp0, `training_step` 681000, creator `replay`, corpus w3aA5b, parent 20260706-1-PVZp. File SHA-256 `d8cfc674…decc44`. Arch v5: stem 7×7 → 2 × [15×15 + 15×15 @64, SE scale_and_bias /4, pre-act, clean_add, ReZero 0.5·tanh, output LayerNorm] → tower-end BN → intermediate_conv policy (512) + WDL value (16 → FC64), `bfloat16`.
- **Positions (4,958):** every ply of all 35 bot games this model played (2,262, of which 1,127 are our moves with a recorded decision) plus 2,696 corpus positions from w3aA5b shard 45. The first 900 corpus positions are exactly the prior survey's set (`c900`).
- **Ground truth:** the bot's recorded W/D/L (bf16 read-back) and top-5 policy probabilities (CPU fp32 softmax of the bf16 logits) for the 1,127 moves.

## What the real graph computes (ChessNetwork.swift) and the emulation used

- **Inference graph (bot, `ChessMPSNetwork(.randomWeights, arch:)`, config D off):** the input is cast to bf16 on the GPU, weights are stored bf16, and every activation, conv/matmul output, BN/LN `normalize`, SE, ReZero and head output is a bf16 tensor. LayerNorm's mean and variance are separate bf16 graph ops. The policy output is widened to fp32 only after it has been produced in bf16 (`policyOutputReadback`). Value probabilities are read back as bf16.
- **Calibration against the records** (`calib*.log`). The best fit is: float64 internals, bf16 rounding of the policy pre-BN output, and the head matmul+bias rounded **once** (fused). It matches 724/1127 recorded W/D/L triples exactly, 464/1127 top-5 policy vectors exactly, and 149/159 recorded top-2 ties.
  - Rounding the head matmul before the bias add drops that to 474 / 78.
  - Per-op rounding everywhere drops it to 417–626 / 73–330.
  - So the engine behaves as if intermediates are held more precisely than per-op bf16. Every "real" number below uses this fit. The per-op variant is a pessimistic bound.
- **Training graph (ChessTrainer.swift):** the CE losses run `softMaxCrossEntropy` directly on the bf16 `network.policyOutput` / `network.valueLogits`. The targets are built in the compute dtype (bf16). Only the scalar loss reductions are widened to fp32 (`widenForReduction`). Weights have fp32 masters, but the gradients arrive in bf16.

## 1. Value head — BAD

| quantity | value |
|---|---|
| fc2 mean-row norm / per-class residual norms | 28.95 / 0.62, 0.63, 0.66 |
| fc2 bias (W, D, L) / mean (init ln6/3 = 0.597) | 14.25, 12.375, 14.5625 / **13.73** |
| shared logit (mean of 3) percentiles 1 / 5 / 50 / 75 / 95 / 99 | 14.5 / 509.1 / **510.6** / 535.4 / 968.1 / 990.8 |
| positions with shared logit in [256,512) (bf16 spacing 2) / [512,1024) (spacing 4) | 62.1% / 36.8% |
| functional spread (max − min) median | 3.71 |

| emulation / fix | logit ties | KL(fp64‖q) mean / max | argmax change | corpus CE (fp64 0.9051) | ΔCE | mean \|Δv\| |
|---|---|---|---|---|---|---|
| real bf16 | **55.3%** | 0.108 / 1.18 | 27.3% | 1.0382 | **+0.1330** | 0.281 |
| per-op bf16 | 59.0% | 0.110 / 1.25 | 30.7% | 1.0266 | +0.1214 | 0.286 |
| (a) recentered fc2 (bf16-stored) | 0.06% | 6.3e-5 / 2.9e-3 | 0 | 0.9055 | +0.00036 | 0.0013 |
| (b) fp32 fc2 projection | 0 | 4e-10 | 0 | 0.9051 | ~0 | 4e-6 |
| (c) fp32 per-position mean-subtract, then bf16 | 0.04% | 6.7e-5 / 2.9e-3 | 0 | 0.9052 | +0.00004 | 0.0012 |
| (d) whole network fp32 | 0 | 1e-10 | 0 | 0.9051 | ~0 | 5e-6 |

- **Recorded (real engine), our 1,127 moves:**
  - 62.6% of the recorded W/D/L triples contain a tie.
  - The recorded v = W − L differs from fp64 by 0.274 on average (p90 0.54, max 0.96).
  - The recorded argmax differs from fp64 in 29.2% of moves.
- **Start position (ply 0 as White, 18 games, all identical):**
  - fp64 logits [510.55, 507.86, 511.40] → W/D/L 0.295 / 0.020 / 0.685.
  - bf16 logits [510, 508, 512] → 0.117 / 0.016 / 0.867. This is exactly what the bot recorded in all 18 games.
  - Recentered bf16 → 0.295 / 0.020 / 0.684.
- **Against Stockfish depth 14** (976 of our moves, expected score; `sfcompare.json`):
  - correlation 0.853 (fp64), **0.647 (recorded bf16)**, 0.853 (recentered)
  - MAE 0.225 / 0.260 / 0.226
- **Verdict:** the exact recentering of `value.wdl_fc2` restores the value head on this checkpoint.

## 2. Policy head — degraded (not yet BAD at 681k)

| quantity | value |
|---|---|
| policy.conv mean-row norm / residual norm median | 2.01 / 1.19 |
| policy.conv bias mean / std | −0.589 / 0.135 |
| legal-logit mean percentiles 1 / 50 / 99 | −42.5 / **−42.2** / −37.4 |
| legal std (median) / all-move mean (median) | **0.578** / −57.6 |
| bf16 spacing at the legal max | 0.25 (100% of positions have the legal max in [32,64)) |
| within-position per-square shared-component std / residual std | 1.63 / 2.13 |

| emulation / fix | KL mean / p90 / max | top-1 lost | top-2 ties | top-5 ties | corpus CE (fp64 2.1130) ΔCE |
|---|---|---|---|---|---|
| real bf16 | **2.32e-3** / 3.2e-3 / 7.1e-3 | 0 | **12.7%** (ours 14.0%; recorded 14.1%) | 76.7% | **+0.0045** |
| per-op bf16 | 2.44e-3 | 0.16% | 13.5% | 79.0% | +0.0047 |
| bias-mean recenter (b − mean b), bf16 | 2.24e-3 | 0.04% | 13.0% | 78.9% | +0.0022 |
| constant shift of the bias to the legal level (b + 42), bf16 | 1.62e-3 | **3.3%** | 0.4% | 2.2% | +0.0035 |
| mean-row recenter (W − mean row) — exact math, **not softmax-invariant** | 0.98 | 71% | — | — | +1.159 |
| (b) fp32 final projection | 3.2e-5 | 0.63%* | 0 | 0 | +0.00013 |
| (c1) fp32 all-move mean-subtract, then bf16 | 3.7e-4 | 0 | 6.9% | 46.0% | +0.00048 |
| (c2) fp32 max-subtract, then bf16 | 3.6e-5 | 0.63%* | 0 | 5.4% | +0.00018 |
| (d) whole network fp32 | 5.6e-13 | 0 | 0 | 0 | ~0 |

\* This residual is from rounding the internal pre-BN output on positions whose fp64 top two are nearly tied. It is not from the head.

- **Why no weight edit is exact for the policy:**
  - The offset is `mean_row · feat[square] + mean(b)`, so it varies by square and by position. A 1×1 conv can't subtract a per-position mean.
  - A constant bias shift is softmax-invariant, but the shifted biases (≈ 41) have a bf16 spacing of 0.25. The per-channel rounding (up to 0.125) then distorts the policy.

## 3. Rest of the network — no second problem site

- **Dynamic range:**
  - The residual stream is re-normalized by LayerNorm every block, so |x| ≤ 10 at both block adds.
  - LN inputs have a per-square |mean_C|/std_C of median 0.15–0.22 and max 0.70, so there's no LN cancellation.
  - The largest internal offset-to-spread ratio is `b0.ln` → `blocks.1.bn1`: one channel has |running_mean|/√running_var = 30.9. A bf16-rounded input there carries an error of 4.3% of the normalizer scale (median over channels 1.2%).
  - Next are `stem.bn` → `blocks.0.bn1` (11.4, 2.2%) and `policy.pre_conv` → `policy.pre_bn` (14.0, 2.2%).
  - Everything else is ≤ 9.
- **Single-point bf16 rounding with the heads unrounded** (`net681_scan.json`):
  - The largest single internal contributors are `b0.ln` (policy KL 3.9e-5, value KL 6.9e-6), `p.pre_bn` (3.2e-5) and `tower.bn` (1.4e-5).
  - All internal points rounded together: value KL 2.0e-5, ΔCE +0.0007, mean |Δv| 0.0033; policy KL 1.6e-4, ΔCE +0.0001.
  - For comparison, the head outputs alone give value KL 0.108 and policy KL 2.3e-3. That's 5,400× (value) and 14× (policy) more than the whole body.
- **SE gates:** 0% of gate values round to exactly 0 or 1 in bf16.
- **ReZero:**
  - Raw α = 1.734375 in both blocks, unchanged in every checkpoint from 150k to 1,397k. Effective α = 0.499.
  - tanh(α/C) = 0.99806 rounds to exactly 1.0 in bf16. The fp64 derivative is 0.0039, and it is exactly 0 if the derivative is computed from the bf16 tanh output (inferred).
  - α sits at its designed cap, but it is no longer a live parameter.

## 4. Training-time gradients (`gradcheck.json`, corpus positions, the trainer's own ε)

- **Value CE gradient (functional part):**
  - Per-position relative error from bf16 logits: median 35%, p90 116%.
  - The batch-summed fc2 weight gradient keeps cosine 0.999, so the corruption is per-sample noise, which is what propagates into fc1 and the tower.
- **Policy CE gradient:** per-position relative error median 2.1%, p90 4.1%; conv-weight gradient cosine 0.9997.
- **Shared-direction gradient (the drift driver), measured:**
  - bf16-built targets sum to 1 + 8.5e-4 (value, ε = 0.013) and 1 − 7.8e-4 (policy, ε = 0.1).
  - That gives a constant per-position push along the softmax-invariant direction of −8.5e-4 / +7.8e-4, identical under fp64 and bf16 logits.
  - The predicted value-bias-mean drift, lr·(1−μ)⁻¹·8.5e-4/3 = 2.8e-5 per step, has the sign and 74% of the magnitude of the observed linear drift of +2.1e-5 per step from 50k to 1,397k.
  - The policy bias mean drifts −5.5e-7 per step, as predicted in the lineage survey.
- **Trainer-reported vLoss** (bf16 graph, `trainer_vloss_binned.txt`):
  - median 0.879 over 0–50k, then 1.03–1.05 from 50k to 700k and 1.09–1.10 by 1.3M
  - over the same span the fp64 c900 CE of the sampled checkpoints is 0.72–0.94, except 1.15 (150k), 1.37 (200k), 1.01 (400k) and 1.20 (1,397k)

## 5. Trend along the line (`trend_struct.csv` = every checkpoint; `trend_fwd.json` = forwards on c900 + 376 bot positions)

| checkpoint | fc2 mean-row | fc2 bias mean | shared logit med | value ties | value ΔCE (c900) | policy bias mean | legal mean | policy KL | top-2 ties |
|---|---|---|---|---|---|---|---|---|---|
| Qeu8 seed | 0.83 | 0.596 | 0.47 | 0.1% | −0.0002 | 0 | +0.19 | 4e-6 | 1.3% |
| GLu5 41k | 1.65 | 0.484 | −10.7 | 9.0% | +0.0005 | −0.024 | +13.1 | 1.8e-4 | 4.2% |
| Lnji 67.5k | 5.39 | 0.293 | −11.5 | 33.6% | +0.0032 | −0.082 | +9.4 | 1.6e-4 | 3.2% |
| PVZp 67k | 7.74 | 0.144 | −42.1 | 35.1% | +0.0013 | −0.145 | +4.4 | 4e-5 | 2.1% |
| Ejp0 30k | 13.78 | 0.074 | −252 | 59.6% | +0.040 | −0.165 | +3.7 | 3e-5 | 2.1% |
| Ejp0 75k | 23.91 | 0.747 | +534 | 95.1% | +0.218 | −0.194 | +1.9 | 2e-5 | 1.6% |
| Ejp0 100k | 29.49 | 1.52 | +509 | 78.8% | +0.104 | −0.210 | +0.9 | 1e-5 | 1.3% |
| Ejp0 300k | 31.96 | 6.00 | +534 | 98.4% | +0.216 | −0.350 | −7.4 | 4e-5 | 1.7% |
| Ejp0 500k | 30.12 | 10.02 | +511 | 62.2% | +0.086 | −0.481 | −22.5 | 6.0e-4 | 8.3% |
| **Ejp0 681k** | **28.95** | **13.73** | **+510** | **52.5%** | **+0.145** | **−0.589** | **−42.2** | **2.4e-3** | **11.9%** |
| Ejp0 1000k | 31.12 | 20.58 | +515 | 96.6% | +0.209 | −0.763 | −85.8 | 8.6e-3 | 23.4% |
| Ejp0 1397k | 29.44 | 28.79 | +513 | 57.7% | +0.460 | −0.981 | −138.2 | 3.4e-2 | 40.4% |

- **The 1,397k row reproduces the lineage survey exactly** (ΔCE 0.4596, legal mean −138.2, fp64 CE 1.20).
- **Value:**
  - Already degraded before Ejp0 (a third of positions tied on Lnji and PVZp).
  - BAD from about 30–75k into Ejp0, when the shared logit reached ±250–530.
  - From ~100k it stays pinned just under or around 512 while the fc2 bias mean drifts linearly. ΔCE fluctuates 0.03–0.46 between checkpoints.
- **Policy:**
  - From +13 at GLu5 41k the legal level falls roughly linearly: → 0 near 130k → −42 at 681k → −138 at 1,397k. The legal spread stays ≈ 0.6.
  - Damage steps up each time the legal level crosses a bf16 binade: KL ~1e-5 below 16, 2.4e-3 in [32,64), 8.6e-3 in [64,128), 3.4e-2 in [128,256).

## Scripts (`../../scripts/`)

- `fwd4.py` — the instrumented forward, with 58 named rounding points.
- `posset_ejp0.py` — builds the position set.
- `calib.py`, `calib_scan.py`, `calib_combo.py` — fit the emulation against the records.
- `heads681.py` — head metrics and fixes.
- `net681.py stats|scan` — the body audit.
- `trend_struct.py`, `trend_fwd.py` — the trend.
- `gradcheck.py` — training gradients.
- `sfcompare.py` — needs the `ejp0_sf_full.json` copy here.
- `fwd.py` — the board encoder imported by `posset.py` and `posset_ejp0.py`; checked in so they run.

Each script reads `~/Library/Application Support/DrewsChessMachine/…` and needs numpy and python-chess.
