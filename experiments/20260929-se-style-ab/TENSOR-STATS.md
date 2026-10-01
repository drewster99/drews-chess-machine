# SE style experiment: tensor statistics

Per-tensor statistics (mean, min, max, std, abs max, RMS, L2 norm, exact-zero fraction, non-finite count) for **every checkpoint of all eight runs**: 145 checkpoints, 13,360 rows. Each run is covered from its fresh net through every enumerated 1k checkpoint to its stop save. Experiment overview: [REPORT-final.md](REPORT-final.md).

- Full data: [data/tensor_stats.csv](data/tensor_stats.csv). Columns are described in [data/README.md](data/README.md).
- Produced by [tensor_stats.py](tensor_stats.py), which reads the checkpoints. The tables below are produced by [tensor_report.py](tensor_report.py).
- Checkpoints are identified by `__metadata__`: each run's trained checkpoints were checked to share one ModelID. The `-replay-latest` files are skipped, since each is a copy of that run's stop save.

## Findings

1. **No NaN or Inf anywhere.** Every value in all 145 checkpoints is finite, including the optimizer state.

2. **Seed-1 checkpoints store bf16-rounded weights; later builds store the fp32 masters.**
   - In every build-2255 checkpoint (seed 1), 100% of parameter values lie exactly on the bf16 grid (low 16 bits zero). Those files hold the bf16 working copy, not the fp32 master weights.
   - From build 2259 on (seed 2 and zero-β), 0.00% are on the grid, so the files hold the fp32 masters (commit `d15f706`).
   - Every fresh net is on the bf16 grid, because nets are minted as bf16 models.
   - Consequence: seed-1 checkpoints have lost the masters' low bits, which is one more reason they can't be resumed exactly.

3. **Seven input planes never receive any gradient.** For stem-conv weights reading planes 19, 20, 21, 22, 24, 26 and 28:
   - their cosine with the fresh weights is exactly 1.0000;
   - their norm shrinks by exactly the weight-decay factor shared by all of them (×0.781 over seed 1's 33k steps, ×0.908 over seed 2's 7k);
   - their momentum velocity is exactly 0 in all five checkpoints that carry optimizer state.

   Planes 20–29 mark "this position occurred *i+1* plies ago":
   - **Planes 20, 22, 24, 26 and 28** (1, 3, 5, 7, 9 plies ago) **can never be set by the rules of chess**: an odd ply distance means the other side is to move, so the positions can't be equal.
   - **Plane 21** (2 plies ago) can't be set either: both sides would have to undo their own move in one ply each.
   - **Plane 19** ("seen twice before") is legal but never appears in the sampled training positions.
   - **The planes that can be set** (23, 25, 27, 29: 4, 6, 8, 10 plies ago) move only slightly: cosine with init 0.975–0.999.
   - **Plane 16** (en passant): 1/7 of its kernel weights have zero velocity, so part of the 7×7 kernel never sees a set square.

   These dead inputs cost a little compute and nothing else. They are an encoding-design observation, not a training fault.

4. **Many SE bottleneck units are dead; nothing else in the network is.**

   Two measures, per SE block (32 FC1 units each):
   - **Dead now:** the unit's FC1 bias and weight velocity are exactly 0. Momentum decays geometrically when the gradient is zero and reaches exactly 0 in fp32 only after roughly 600–2,000 consecutive zero-gradient steps (momentum 0.85–0.95), so these units have been inactive (ReLU off) for all recent inputs. Only checkpoints with optimizer state (build 2259+) have this. (FC velocity is stored in the graph's [in, out] layout. The table counts units whose bias velocity is exactly 0; counting by weight velocity gives the same units in every block except `se_att2` block 1, where unit 1's bias velocity has reached 0 while its weight velocity is down to 3.2e-38 and still decaying, i.e. the same unit one step from underflow.)
   - **Dead almost the whole run:** the unit's FC1 weight row has kept its initial direction (cosine with init > 0.999995) and shrunk by exactly the weight-decay factor, so it got essentially no gradient since the first steps. This works for every run, seed 1 included.

   | run | final step | dead now (b0 / b1 / b2) | dead almost the whole run (b0 / b1 / b2) |
   |---|---:|---|---|
   | `se_sb` (scale+bias, seed 1) | 33,014 | not saved | 2 / 4 / 0 |
   | `se_att` (attenuate-only, seed 1) | 33,012 | not saved | 8 / 1 / 2 |
   | `se_zb1` (zero-β, seed 1) | 5,030 | 13 / 6 / 4 | 8 / 9 / 1 |
   | `se_sb2` (scale+bias, seed 2) | 7,282 | 4 / 13 / 4 | 1 / 5 / 2 |
   | `se_att2` (attenuate-only, seed 2) | 7,289 | 8 / 7 / 5 | 7 / 7 / 2 |
   | `se_zb2` (zero-β, seed 2) | 5,004 | 9 / 4 / 2 | 6 / 3 / 0 |

   - **Never trained at all:** one unit in each of three runs still has its FC1 bias at exactly 0, its init value: `se_sb2` block 1 unit 19, `se_zb1` block 1 unit 20, `se_zb2` block 0 unit 26. For the two zero-β units, the β weights reading from them are also still exactly 0 (the 1/32 exact-zero fraction in the β half of fc2).
   - **Dead now ≥ dead all along:** most dead units died very early; a few more die later (e.g. `se_zb1` block 0: 8 dead almost the whole run, 13 dead now).
   - **What this means:** up to 41% of a block's SE bottleneck is inactive, which fits the finding that SE adds little here.
   - **Nothing else is dead.** Checked the same way: no conv output channel in the stem or any block, no BN γ/β, no policy channel, and none of the value head's 16 conv channels or 128 hidden units has zero velocity or a decay-only weight row. The only other dead weights in the network are the stem weights reading the seven always-zero input planes (finding 3).

5. **Invariants hold.**
   - **β bias mean** (scale+bias and zero-β) stays at 0, with |mean| ≤ 4.8e-5 at every final, as the block LayerNorm requires.
   - **The value head's WDL bias mean** stays at its init: ln 6 / 3 = 0.596354 after bf16 rounding. The finals lie in 0.595052–0.597043, because a shift common to all three logits gets no gradient through the softmax.
   - **`stem.bn.bias` matches `blocks.0.bn1.running_mean`** almost exactly in every run (e.g. −0.0184 vs −0.0184 for `se_sb`), as expected: block 0's pre-activation BN sees the stem output, whose per-channel mean is the stem BN's β.

6. **ReZero α trains up toward its cap.** The branch is scaled by the soft-bounded α_eff = C·tanh(α/C), with C = α₀ = 0.4472 (`ChessNetwork.swift`, `rezeroTanhCeilingMultiple` = 1.0).
   - The raw parameter starts at 0.4473, so α_eff starts at 0.4472·tanh(1.0) = 0.341.
   - At the finals the raw value ranges from 0.560 to 1.211, so α_eff is 0.380–0.443, i.e. 85–99% of the 0.447 cap. Seed-1 SE arms sit closest to it (raw 1.10–1.21, α_eff 0.441–0.443).
   - `se_none2` has the smallest (block 1: raw 0.560, α_eff 0.380).
   - (Corrected 2026-09-30: an earlier version of this finding used 0.4472·tanh(α) and reported 0.188 / 0.227–0.374.)

7. **Weight scale.**
   - **Tower convs** barely change RMS (0.0179 at init → 0.0159–0.0188).
   - **Stem conv** grows (0.0369 → 0.0399–0.0416), and so does **policy conv** (≈0.125 → 0.146–0.148).
   - **The value head shrinks:** fc1 0.044 → 0.035 (seed 1, 33k) and ≈0.040 (seed 2 / zero-β); wdl_fc2 ≈0.12–0.13 → 0.10–0.11.
   - **The largest |value| of any trainable parameter** in a final is a BN parameter, 1.45–2.25 (top: `se_none` `blocks.1.bn2.weight` 2.25). **BN running statistics go higher:** the largest is `se_att` `blocks.2.bn1.running_var` at 13.94, with `blocks.2.bn1` and `tower_final_bn` running variances of 3.7–13.9 in most runs (see the top-20 table below).

8. **One outlier BN channel.** `se_sb2` stem BN channel 14 has a running variance of 2.57 against a median of 0.067 (38×), with a running mean of −1.00. It peaked at 3.60 at step 3,000. Its filter weights the input on plane 1 heavily (plane RMS 0.126 vs about 0.04 for the others). BN normalizes it, so it's not an error. It is the largest *relative* outlier among the stem statistics, but not the largest running statistic overall (see below).

9. **Hot channels in the residual stream: every arm has one; attenuate-only's fluctuates.** For each BN, the *energy share* of a channel is its E[x²] = running variance + running mean², divided by the sum over all channels. With 128 channels, uniform would be 0.78%. For each run below, the channel with the largest energy share at block 2's input (`blocks.2.bn1`, i.e. block 1's LayerNorm output):

   | run | channel | energy share | variance | mean | share of its energy from variance | max ÷ mean variance | top variance share |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | `se_none` | 40 | 7.6% | 0.88 | +3.36 | 7% | 15× | 11.5% |
   | `se_att` | 37 | 18.8% | 13.94 | +3.89 | 48% | 31× | 24.3% |
   | `se_sb` | 7 | 13.8% | 7.53 | +3.41 | 39% | 27× | 21.3% |
   | `se_zb1` | 7 | 13.6% | 10.60 | +2.96 | 55% | 24× | 18.7% |
   | `se_none2` | 13 | 10.0% | 1.48 | +3.66 | 10% | 4× | 3.1% |
   | `se_att2` | 62 | 12.1% | 10.77 | +2.72 | 59% | 23× | 17.7% |
   | `se_sb2` | 83 | 8.1% | 1.34 | +3.11 | 12% | 6× | 4.9% |
   | `se_zb2` | 83 | 6.2% | 2.26 | +2.50 | 27% | 7× | 5.8% |

   The last two columns describe the highest-*variance* channel, which is not always the highest-energy one: in the no-SE runs the highest-energy channel is a near-constant offset with low variance.

   - **Every run has a hot channel** (6–19% of the energy), each carrying a large constant mean (+2.5 to +3.9).
   - **Attenuate-only's hot channel fluctuates** (48–59% of its energy is variance, 23–31× the mean variance) in both seeds.
   - **No SE's is almost pure offset** (7–10% variance) in both seeds.
   - **Scale+bias and zero-β** are strongly variable in seed 1 and weak in seed 2.
   - On energy share alone, the arms are not ordered consistently: seed 2's no SE (10.0%) is above its scale+bias (8.1%) and zero-β (6.2%).

   Block 2 input energy share over training:

   | step | none s1 | att s1 | s+b s1 | zero-β s1 | none s2 | att s2 | s+b s2 | zero-β s2 |
   |---:|---:|---:|---:|---:|---:|---:|---:|---:|
   | 1,000 | 8.2 | 4.8 | 4.3 | 5.5 | 6.3 | 8.1 | 3.8 | 3.6 |
   | 2,000 | 7.1 | 9.2 | 7.4 | 10.0 | 7.2 | 7.0 | 5.7 | 4.2 |
   | 3,000 | 6.2 | 11.3 | 9.4 | 12.2 | 8.9 | 9.4 | 6.9 | 5.3 |
   | 4,000 | 7.0 | 12.5 | 10.2 | 12.9 | 9.2 | 10.4 | 7.3 | 5.9 |
   | 5,000 | 7.1 | 12.7 | 10.3 | 13.6 | 9.5 | 11.4 | 7.9 | 6.2 |
   | 6,000 | 7.2 | 12.8 | 11.0 |  | 10.0 | 11.7 | 8.0 |  |
   | 7,000 | 7.1 | 12.9 | 12.5 |  | 10.0 | 11.9 | 8.1 |  |
   | 11,000 | 6.9 | 13.2 | 14.3 |  |  |  |  |  |
   | 20,000 | 7.2 | 13.4 | 17.4 |  |  |  |  |  |
   | 30,000 | 7.5 | 18.8 | 13.5 |  |  |  |  |  |

   Every BN input at the final checkpoint (energy share %; the value head's BN has 16 channels, so uniform there is 6.25%):

   | BN input | none | att | s+b | zero-β | none s2 | att s2 | s+b s2 | zero-β s2 |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|
   | stem.bn | 2.6 | 2.7 | 3.2 | 2.5 | 2.4 | 2.7 | 20.7 | 4.1 |
   | blocks.0.bn1 | 1.5 | 1.8 | 1.9 | 1.9 | 1.4 | 1.5 | 1.4 | 1.5 |
   | blocks.0.bn2 | 3.5 | 3.4 | 3.9 | 5.7 | 2.8 | 4.5 | 3.9 | 2.6 |
   | blocks.1.bn1 | 5.5 | 7.9 | 19.9 | 10.1 | 8.2 | 7.4 | 8.3 | 5.4 |
   | blocks.1.bn2 | 2.9 | 2.8 | 1.8 | 2.0 | 2.5 | 2.0 | 2.0 | 1.9 |
   | blocks.2.bn1 | 7.6 | 18.8 | 13.8 | 13.6 | 10.0 | 12.1 | 8.1 | 6.2 |
   | blocks.2.bn2 | 1.8 | 2.5 | 2.1 | 1.8 | 2.4 | 2.1 | 2.9 | 3.0 |
   | tower_final_bn | 6.4 | 11.2 | 8.1 | 10.6 | 4.9 | 11.1 | 4.7 | 5.4 |
   | policy.pre_bn | 6.3 | 2.6 | 2.6 | 2.3 | 2.6 | 2.3 | 2.0 | 1.9 |
   | value.bn | 16.8 | 10.5 | 16.3 | 14.0 | 13.6 | 10.3 | 10.7 | 10.0 |

   - **The concentration lives in the residual stream** (each block's `bn1` input and `tower_final_bn`), which is the clean-add sum after each block's per-position LayerNorm. Inside the blocks (`bn2`, after conv1) it stays at 2–6%.
   - **The init decides which channel wins:** zero-β and its Glorot-β parent share hottest channels 7 (seed 1) and 83 (seed 2).
   - **The value head's 10–17% is mostly there at init.** The four fresh nets checked (`se_none`, `se_sb`, `se_none2`, `se_att2`) already show 11.2–14.6% on `value.bn`, because its 16 channels are random 1×1 projections of a stream whose channels have nonzero means. At the finals, the hottest channel's `value.conv` filter norm is only 1–9% above the median filter's. This is not caused by the residual stream's hot channels.

<!-- BEGIN GENERATED TABLES -->

## Checkpoints covered

| run | arm | seed | checkpoints | steps | trained ModelID |
|---|---|---:|---:|---|---|
| `se_sb` | scale+bias | 1 | 35 | 0, 1,000 … 33,014 | `20260929-22-bWdy` |
| `se_att` | attenuate-only | 1 | 35 | 0, 1,000 … 33,012 | `20260929-23-L6Qm` |
| `se_none` | none | 1 | 34 | 0, 1,000 … 32,036 | `20260929-24-834D` |
| `se_zb1` | zero-beta scale+bias | 1 | 7 | 0, 1,000 … 5,030 | `20260930-9-RrGx` |
| `se_sb2` | scale+bias | 2 | 9 | 0, 1,000 … 7,282 | `20260930-4-k98x` |
| `se_att2` | attenuate-only | 2 | 9 | 0, 1,000 … 7,289 | `20260930-5-5TXu` |
| `se_none2` | none | 2 | 9 | 0, 1,000 … 7,019 | `20260930-6-LkS6` |
| `se_zb2` | zero-beta scale+bias | 2 | 7 | 0, 1,000 … 5,004 | `20260930-10-H51a` |

## Top 20 abs max at the final checkpoint: parameters and BN running statistics

| # | run | step | tensor | kind | abs max | min | max | mean |
|---:|---|---:|---|---|---:|---:|---:|---:|
| 1 | `se_att` | 33,012 | `blocks.2.bn1.running_var` | bn_running_stat | 13.9375 | +0.03174 | +13.94 | +0.4479 |
| 2 | `se_att2` | 7,289 | `blocks.2.bn1.running_var` | bn_running_stat | 10.7714 | +0.02062 | +10.77 | +0.4752 |
| 3 | `se_zb1` | 5,030 | `blocks.2.bn1.running_var` | bn_running_stat | 10.6035 | +0.04528 | +10.6 | +0.4431 |
| 4 | `se_att2` | 7,289 | `tower_final_bn.running_var` | bn_running_stat | 8.1271 | +0.06708 | +8.127 | +0.4957 |
| 5 | `se_att` | 33,012 | `tower_final_bn.running_var` | bn_running_stat | 8.0000 | +0.1182 | +8 | +0.4657 |
| 6 | `se_sb` | 33,014 | `blocks.2.bn1.running_var` | bn_running_stat | 7.5312 | +0.03394 | +7.531 | +0.2759 |
| 7 | `se_zb1` | 5,030 | `tower_final_bn.running_var` | bn_running_stat | 6.0784 | +0.07718 | +6.078 | +0.416 |
| 8 | `se_att2` | 7,289 | `blocks.1.bn1.running_var` | bn_running_stat | 5.8657 | +0.04462 | +5.866 | +0.4715 |
| 9 | `se_none` | 32,036 | `blocks.2.bn1.running_var` | bn_running_stat | 5.6250 | +0.05225 | +5.625 | +0.3829 |
| 10 | `se_sb` | 33,014 | `blocks.1.bn1.running_mean` | bn_running_stat | 5.5312 | -5.531 | +3.453 | -0.04565 |
| 11 | `se_zb1` | 5,030 | `blocks.1.bn1.running_var` | bn_running_stat | 5.1293 | +0.01756 | +5.129 | +0.4197 |
| 12 | `se_sb` | 33,014 | `tower_final_bn.running_var` | bn_running_stat | 4.5938 | +0.05933 | +4.594 | +0.3075 |
| 13 | `se_att` | 33,012 | `blocks.1.bn1.running_var` | bn_running_stat | 4.4688 | +0.02527 | +4.469 | +0.3143 |
| 14 | `se_sb` | 33,014 | `blocks.2.bn1.running_mean` | bn_running_stat | 4.2188 | -4.219 | +3.406 | +0.01355 |
| 15 | `se_att` | 33,012 | `blocks.2.bn1.running_mean` | bn_running_stat | 3.8906 | -2.531 | +3.891 | +0.02249 |
| 16 | `se_none` | 32,036 | `tower_final_bn.running_var` | bn_running_stat | 3.6875 | +0.1196 | +3.688 | +0.4476 |
| 17 | `se_none2` | 7,019 | `blocks.2.bn1.running_mean` | bn_running_stat | 3.6643 | -1.779 | +3.664 | +0.01832 |
| 18 | `se_zb1` | 5,030 | `policy.pre_bn.running_var` | bn_running_stat | 3.5101 | +0.4303 | +3.51 | +0.983 |
| 19 | `se_none` | 32,036 | `blocks.1.bn2.running_var` | bn_running_stat | 3.4531 | +0.459 | +3.453 | +0.7128 |
| 20 | `se_att2` | 7,289 | `blocks.2.bn1.running_mean` | bn_running_stat | 3.4362 | -2.267 | +3.436 | +0.0149 |

## Top 20 abs max at the final checkpoint: trainable parameters only

| # | run | step | tensor | kind | abs max | min | max | mean |
|---:|---|---:|---|---|---:|---:|---:|---:|
| 1 | `se_none` | 32,036 | `blocks.1.bn2.weight` | parameter | 2.2500 | +0.8828 | +2.25 | +0.9977 |
| 2 | `se_none` | 32,036 | `blocks.1.bn1.weight` | parameter | 2.2344 | +0.8086 | +2.234 | +0.9691 |
| 3 | `se_sb` | 33,014 | `blocks.2.bn1.bias` | parameter | 2.1406 | -2.141 | +0.1157 | -0.1012 |
| 4 | `se_none` | 32,036 | `blocks.2.bn1.weight` | parameter | 2.0938 | +0.8242 | +2.094 | +0.9644 |
| 5 | `se_att` | 33,012 | `blocks.1.bn1.weight` | parameter | 2.0781 | +0.7305 | +2.078 | +0.9628 |
| 6 | `se_sb` | 33,014 | `policy.pre_bn.weight` | parameter | 2.0312 | +0.832 | +2.031 | +1.06 |
| 7 | `se_sb` | 33,014 | `blocks.2.bn1.weight` | parameter | 2.0156 | +0.8047 | +2.016 | +0.9599 |
| 8 | `se_att` | 33,012 | `blocks.0.bn1.weight` | parameter | 1.9609 | +0.6992 | +1.961 | +0.9764 |
| 9 | `se_sb` | 33,014 | `blocks.1.bn1.weight` | parameter | 1.9141 | +0.7422 | +1.914 | +0.9683 |
| 10 | `se_none` | 32,036 | `tower_final_bn.weight` | parameter | 1.9062 | +0.6875 | +1.906 | +0.9812 |
| 11 | `se_sb` | 33,014 | `stem.conv.weight` | parameter | 1.8906 | -0.7695 | +1.891 | +0.0001769 |
| 12 | `se_sb` | 33,014 | `blocks.0.bn1.weight` | parameter | 1.8594 | +0.7305 | +1.859 | +0.9837 |
| 13 | `se_att` | 33,012 | `blocks.2.bn1.weight` | parameter | 1.8359 | +0.8164 | +1.836 | +0.9743 |
| 14 | `se_att` | 33,012 | `blocks.1.bn2.weight` | parameter | 1.7969 | +0.9375 | +1.797 | +1.035 |
| 15 | `se_zb1` | 5,030 | `stem.conv.weight` | parameter | 1.7852 | -0.8164 | +1.785 | +9.904e-05 |
| 16 | `se_zb1` | 5,030 | `blocks.2.bn1.weight` | parameter | 1.7804 | +0.8645 | +1.78 | +0.9878 |
| 17 | `se_none2` | 7,019 | `blocks.0.bn1.weight` | parameter | 1.7739 | +0.8616 | +1.774 | +0.9929 |
| 18 | `se_sb` | 33,014 | `tower_final_bn.weight` | parameter | 1.7578 | +0.7852 | +1.758 | +0.9839 |
| 19 | `se_att` | 33,012 | `blocks.2.bn2.weight` | parameter | 1.7578 | +0.918 | +1.758 | +1.016 |
| 20 | `se_none` | 32,036 | `policy.pre_bn.weight` | parameter | 1.7578 | +0.7539 | +1.758 | +1.084 |

## Every tensor, fresh vs final, per run

Network parameters and BN running statistics. `[gamma]` / `[beta]` rows split a scale+bias SE fc2 tensor into its γ half (rows 0–127) and β half (rows 128–255). All checkpoints, including the intermediate ones, are in [data/tensor_stats.csv](data/tensor_stats.csv).

### `se_sb` — scale+bias, seed 1 (step 0 → 33,014)

| tensor | shape | fresh mean | fresh min | fresh max | final mean | final min | final max | final std | final abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `blocks.0.bn1.bias` | 128 | 0 | 0 | 0 | -0.07488 | -0.4316 | +0.2197 | +0.07522 | +0.4316 |
| `blocks.0.bn1.running_mean` | 128 | -6.939e-05 | -0.002441 | +0.001755 | -0.01841 | -0.2656 | +0.09229 | +0.05436 | +0.2656 |
| `blocks.0.bn1.running_var` | 128 | +1 | +0.9922 | +1.008 | +0.9395 | +0.1777 | +2.281 | +0.3151 | +2.281 |
| `blocks.0.bn1.weight` | 128 | +1 | +1 | +1 | +0.9837 | +0.7305 | +1.859 | +0.148 | +1.859 |
| `blocks.0.bn2.bias` | 128 | 0 | 0 | 0 | -0.06074 | -0.4082 | +0.01965 | +0.06525 | +0.4082 |
| `blocks.0.bn2.running_mean` | 128 | -0.02343 | -0.9023 | +0.9219 | -0.543 | -2.406 | +1.086 | +0.5557 | +2.406 |
| `blocks.0.bn2.running_var` | 128 | +0.5338 | +0.2656 | +1.312 | +0.7281 | +0.4531 | +1.492 | +0.1604 | +1.492 |
| `blocks.0.bn2.weight` | 128 | +1 | +1 | +1 | +1.016 | +0.9414 | +1.305 | +0.06088 | +1.305 |
| `blocks.0.conv1.weight` | 128x128x7x7 | -6.688e-06 | -0.09814 | +0.08447 | -0.0003423 | -0.2246 | +0.457 | +0.01619 | +0.457 |
| `blocks.0.conv2.weight` | 128x128x7x7 | -3.342e-06 | -0.08496 | +0.09229 | -0.0001362 | -0.4824 | +0.2852 | +0.01593 | +0.4824 |
| `blocks.0.res_ln.bias` | 128 | 0 | 0 | 0 | -0.04485 | -0.6133 | +0.02283 | +0.06378 | +0.6133 |
| `blocks.0.res_ln.weight` | 128 | +1 | +1 | +1 | +0.9818 | +0.6367 | +1.414 | +0.1127 | +1.414 |
| `blocks.0.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +1.109 | +1.109 | +1.109 | 0 | +1.109 |
| `blocks.0.se_scalebias.fc1.bias` | 32 | 0 | 0 | 0 | +0.008288 | -0.01807 | +0.06079 | +0.01764 | +0.06079 |
| `blocks.0.se_scalebias.fc1.weight` | 32x128 | -0.0004187 | -0.4746 | +0.4902 | -0.0005747 | -0.4219 | +0.375 | +0.09983 | +0.4219 |
| `blocks.0.se_scalebias.fc2.bias` | 256 | 0 | 0 | 0 | +0.002209 | -0.09277 | +0.06299 | +0.01581 | +0.09277 |
| `blocks.0.se_scalebias.fc2.bias [gamma]` | 256 | 0 | 0 | 0 | +0.004424 | -0.02026 | +0.02673 | +0.005586 | +0.02673 |
| `blocks.0.se_scalebias.fc2.bias [beta]` | 256 | 0 | 0 | 0 | -5.357e-06 | -0.09277 | +0.06299 | +0.02143 | +0.09277 |
| `blocks.0.se_scalebias.fc2.weight` | 256x32 | -0.001157 | -0.3438 | +0.2812 | +0.002079 | -0.4102 | +0.2451 | +0.06797 | +0.4102 |
| `blocks.0.se_scalebias.fc2.weight [gamma]` | 256x32 | -0.001552 | -0.3438 | +0.2676 | +0.004751 | -0.2656 | +0.2363 | +0.06586 | +0.2656 |
| `blocks.0.se_scalebias.fc2.weight [beta]` | 256x32 | -0.0007631 | -0.291 | +0.2812 | -0.000593 | -0.4102 | +0.2451 | +0.0699 | +0.4102 |
| `blocks.1.bn1.bias` | 128 | 0 | 0 | 0 | -0.1313 | -0.9766 | +0.02576 | +0.1358 | +0.9766 |
| `blocks.1.bn1.running_mean` | 128 | +1.216e-05 | -0.3848 | +0.3945 | -0.04565 | -5.531 | +3.453 | +1.016 | +5.531 |
| `blocks.1.bn1.running_var` | 128 | +0.976 | +0.7383 | +1.219 | +0.1653 | +0.01697 | +1.242 | +0.1608 | +1.242 |
| `blocks.1.bn1.weight` | 128 | +1 | +1 | +1 | +0.9683 | +0.7422 | +1.914 | +0.1666 | +1.914 |
| `blocks.1.bn2.bias` | 128 | 0 | 0 | 0 | -0.08792 | -0.4199 | +0.01831 | +0.08196 | +0.4199 |
| `blocks.1.bn2.running_mean` | 128 | +0.004915 | -1.156 | +0.875 | -0.4847 | -1.383 | +0.9453 | +0.4563 | +1.383 |
| `blocks.1.bn2.running_var` | 128 | +0.479 | +0.3125 | +0.8242 | +0.7134 | +0.5312 | +1.508 | +0.1552 | +1.508 |
| `blocks.1.bn2.weight` | 128 | +1 | +1 | +1 | +1.02 | +0.9336 | +1.359 | +0.07369 | +1.359 |
| `blocks.1.conv1.weight` | 128x128x7x7 | -2.195e-06 | -0.08545 | +0.08545 | -0.0003997 | -0.2793 | +0.4922 | +0.01686 | +0.4922 |
| `blocks.1.conv2.weight` | 128x128x7x7 | +8.478e-06 | -0.09082 | +0.09033 | -0.0001538 | -0.4805 | +0.3613 | +0.01685 | +0.4805 |
| `blocks.1.res_ln.bias` | 128 | 0 | 0 | 0 | -0.02516 | -0.3125 | +0.07324 | +0.05245 | +0.3125 |
| `blocks.1.res_ln.weight` | 128 | +1 | +1 | +1 | +0.9783 | +0.6562 | +1.594 | +0.129 | +1.594 |
| `blocks.1.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +1.102 | +1.102 | +1.102 | 0 | +1.102 |
| `blocks.1.se_scalebias.fc1.bias` | 32 | 0 | 0 | 0 | +0.000154 | -0.01025 | +0.04736 | +0.01018 | +0.04736 |
| `blocks.1.se_scalebias.fc1.weight` | 32x128 | +0.001356 | -0.4062 | +0.4258 | +0.001075 | -0.3164 | +0.3633 | +0.09915 | +0.3633 |
| `blocks.1.se_scalebias.fc2.bias` | 256 | 0 | 0 | 0 | +0.00188 | -0.0791 | +0.06982 | +0.01463 | +0.0791 |
| `blocks.1.se_scalebias.fc2.bias [gamma]` | 256 | 0 | 0 | 0 | +0.00375 | -0.04102 | +0.04517 | +0.01535 | +0.04517 |
| `blocks.1.se_scalebias.fc2.bias [beta]` | 256 | 0 | 0 | 0 | +9.626e-06 | -0.0791 | +0.06982 | +0.01362 | +0.0791 |
| `blocks.1.se_scalebias.fc2.weight` | 256x32 | -0.0009212 | -0.2793 | +0.3398 | -3.467e-05 | -0.3945 | +0.2676 | +0.06696 | +0.3945 |
| `blocks.1.se_scalebias.fc2.weight [gamma]` | 256x32 | -0.0008021 | -0.2656 | +0.3398 | +0.0007424 | -0.3945 | +0.2676 | +0.06823 | +0.3945 |
| `blocks.1.se_scalebias.fc2.weight [beta]` | 256x32 | -0.00104 | -0.2793 | +0.3145 | -0.0008118 | -0.2305 | +0.2451 | +0.06565 | +0.2451 |
| `blocks.2.bn1.bias` | 128 | 0 | 0 | 0 | -0.1012 | -2.141 | +0.1157 | +0.2017 | +2.141 |
| `blocks.2.bn1.running_mean` | 128 | +3.231e-05 | -0.5469 | +0.5 | +0.01355 | -4.219 | +3.406 | +0.8987 | +4.219 |
| `blocks.2.bn1.running_var` | 128 | +0.9696 | +0.7305 | +1.219 | +0.2759 | +0.03394 | +7.531 | +0.6763 | +7.531 |
| `blocks.2.bn1.weight` | 128 | +1 | +1 | +1 | +0.9599 | +0.8047 | +2.016 | +0.1709 | +2.016 |
| `blocks.2.bn2.bias` | 128 | 0 | 0 | 0 | +0.01193 | -0.3418 | +0.1436 | +0.07764 | +0.3418 |
| `blocks.2.bn2.running_mean` | 128 | -0.005059 | -1.148 | +0.9414 | -0.391 | -1.883 | +1.258 | +0.6085 | +1.883 |
| `blocks.2.bn2.running_var` | 128 | +0.4917 | +0.3105 | +0.8906 | +1.122 | +0.7656 | +2.25 | +0.2523 | +2.25 |
| `blocks.2.bn2.weight` | 128 | +1 | +1 | +1 | +1.02 | +0.918 | +1.539 | +0.07488 | +1.539 |
| `blocks.2.conv1.weight` | 128x128x7x7 | +7.125e-07 | -0.0835 | +0.08789 | -0.0002664 | -0.5742 | +0.3613 | +0.01728 | +0.5742 |
| `blocks.2.conv2.weight` | 128x128x7x7 | -1.386e-05 | -0.08594 | +0.08301 | -0.0001748 | -0.3242 | +0.1914 | +0.01826 | +0.3242 |
| `blocks.2.res_ln.bias` | 128 | 0 | 0 | 0 | +0.004372 | -0.01062 | +0.03149 | +0.007692 | +0.03149 |
| `blocks.2.res_ln.weight` | 128 | +1 | +1 | +1 | +1.009 | +0.9766 | +1.07 | +0.01455 | +1.07 |
| `blocks.2.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +1.156 | +1.156 | +1.156 | 0 | +1.156 |
| `blocks.2.se_scalebias.fc1.bias` | 32 | 0 | 0 | 0 | -0.0008847 | -0.01166 | +0.03174 | +0.01016 | +0.03174 |
| `blocks.2.se_scalebias.fc1.weight` | 32x128 | -0.00436 | -0.4004 | +0.4336 | -0.003119 | -0.3105 | +0.3379 | +0.09809 | +0.3379 |
| `blocks.2.se_scalebias.fc2.bias` | 256 | 0 | 0 | 0 | +0.005656 | -0.1021 | +0.06641 | +0.02208 | +0.1021 |
| `blocks.2.se_scalebias.fc2.bias [gamma]` | 256 | 0 | 0 | 0 | +0.01136 | -0.06641 | +0.06641 | +0.02151 | +0.06641 |
| `blocks.2.se_scalebias.fc2.bias [beta]` | 256 | 0 | 0 | 0 | -4.814e-05 | -0.1021 | +0.0603 | +0.02116 | +0.1021 |
| `blocks.2.se_scalebias.fc2.weight` | 256x32 | +0.0004774 | -0.3789 | +0.2852 | +0.003143 | -0.3184 | +0.2559 | +0.06641 | +0.3184 |
| `blocks.2.se_scalebias.fc2.weight [gamma]` | 256x32 | +0.0004087 | -0.3789 | +0.2852 | +0.005885 | -0.2949 | +0.2559 | +0.06663 | +0.2949 |
| `blocks.2.se_scalebias.fc2.weight [beta]` | 256x32 | +0.0005461 | -0.2715 | +0.2637 | +0.0004013 | -0.3184 | +0.2236 | +0.06607 | +0.3184 |
| `policy.conv.bias` | 76 | 0 | 0 | 0 | -9.687e-06 | -0.08057 | +0.8945 | +0.1209 | +0.8945 |
| `policy.conv.weight` | 76x128x1x1 | -0.001031 | -0.4727 | +0.4766 | +0.01089 | -1.656 | +0.9102 | +0.1472 | +1.656 |
| `policy.pre_bn.bias` | 128 | 0 | 0 | 0 | +0.02991 | -0.1699 | +0.4277 | +0.09322 | +0.4277 |
| `policy.pre_bn.running_mean` | 128 | +0.003215 | -1.367 | +2 | -0.2789 | -1.891 | +1.055 | +0.6082 | +1.891 |
| `policy.pre_bn.running_var` | 128 | +0.7019 | +0.3906 | +1.703 | +0.8636 | +0.3887 | +2.938 | +0.3876 | +2.938 |
| `policy.pre_bn.weight` | 128 | +1 | +1 | +1 | +1.06 | +0.832 | +2.031 | +0.1358 | +2.031 |
| `policy.pre_conv.weight` | 128x128x1x1 | +1.857e-05 | -0.4629 | +0.5195 | -0.002531 | -0.4883 | +0.498 | +0.1133 | +0.498 |
| `stem.bn.bias` | 128 | 0 | 0 | 0 | -0.0184 | -0.2656 | +0.09229 | +0.05437 | +0.2656 |
| `stem.bn.running_mean` | 128 | +0.006164 | -0.4453 | +0.2871 | -0.01445 | -0.3145 | +0.375 | +0.1551 | +0.375 |
| `stem.bn.running_var` | 128 | +0.06212 | +0.0238 | +0.2383 | +0.07648 | +0.02795 | +0.2695 | +0.03501 | +0.2695 |
| `stem.bn.weight` | 128 | +1 | +1 | +1 | +0.9569 | +0.4219 | +1.516 | +0.1548 | +1.516 |
| `stem.conv.weight` | 128x30x7x7 | +4.382e-05 | -0.165 | +0.1953 | +0.0001769 | -0.7695 | +1.891 | +0.04117 | +1.891 |
| `tower_final_bn.bias` | 128 | 0 | 0 | 0 | +0.145 | -0.1338 | +0.5938 | +0.1215 | +0.5938 |
| `tower_final_bn.running_mean` | 128 | +2.36e-05 | -0.5273 | +0.625 | +0.006518 | -2.641 | +2.453 | +0.8428 | +2.641 |
| `tower_final_bn.running_var` | 128 | +0.9619 | +0.7539 | +1.219 | +0.3075 | +0.05933 | +4.594 | +0.4518 | +4.594 |
| `tower_final_bn.weight` | 128 | +1 | +1 | +1 | +0.9839 | +0.7852 | +1.758 | +0.1422 | +1.758 |
| `value.bn.bias` | 16 | 0 | 0 | 0 | -0.2321 | -0.3438 | -0.1484 | +0.05182 | +0.3438 |
| `value.bn.running_mean` | 16 | +0.04053 | -1.031 | +0.7695 | -0.1951 | -1.492 | +0.5742 | +0.4661 | +1.492 |
| `value.bn.running_var` | 16 | +0.6416 | +0.459 | +1.141 | +0.8252 | +0.4316 | +1.711 | +0.3131 | +1.711 |
| `value.bn.weight` | 16 | +1 | +1 | +1 | +0.6689 | +0.5586 | +0.9336 | +0.08499 | +0.9336 |
| `value.conv.weight` | 16x128x1x1 | +0.0007972 | -0.3984 | +0.4082 | -0.004001 | -0.3652 | +0.3516 | +0.1023 | +0.3652 |
| `value.fc1.bias` | 128 | 0 | 0 | 0 | -0.002501 | -0.04443 | +0.0459 | +0.0144 | +0.0459 |
| `value.fc1.weight` | 128x1024 | +3.011e-06 | -0.1904 | +0.1934 | -0.001629 | -0.1543 | +0.1504 | +0.03483 | +0.1543 |
| `value.wdl_fc2.bias` | 3 | +0.5964 | 0 | +1.789 | +0.597 | +0.4531 | +0.8398 | +0.1727 | +0.8398 |
| `value.wdl_fc2.weight` | 3x128 | +0.006675 | -0.3867 | +0.334 | +0.005239 | -0.4102 | +0.3145 | +0.1068 | +0.4102 |

### `se_att` — attenuate-only, seed 1 (step 0 → 33,012)

| tensor | shape | fresh mean | fresh min | fresh max | final mean | final min | final max | final std | final abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `blocks.0.bn1.bias` | 128 | 0 | 0 | 0 | -0.1087 | -0.6016 | +0.08252 | +0.08491 | +0.6016 |
| `blocks.0.bn1.running_mean` | 128 | +5.383e-05 | -0.002335 | +0.003067 | -0.02353 | -0.4727 | +0.1699 | +0.1054 | +0.4727 |
| `blocks.0.bn1.running_var` | 128 | +0.9994 | +0.9922 | +1.008 | +0.9051 | +0.1001 | +2.094 | +0.2722 | +2.094 |
| `blocks.0.bn1.weight` | 128 | +1 | +1 | +1 | +0.9764 | +0.6992 | +1.961 | +0.1697 | +1.961 |
| `blocks.0.bn2.bias` | 128 | 0 | 0 | 0 | -0.03231 | -0.4824 | +0.0625 | +0.06772 | +0.4824 |
| `blocks.0.bn2.running_mean` | 128 | -0.009793 | -0.918 | +0.8516 | -0.5367 | -2.219 | +1.133 | +0.5448 | +2.219 |
| `blocks.0.bn2.running_var` | 128 | +0.4937 | +0.2734 | +0.8125 | +0.748 | +0.5039 | +1.516 | +0.1753 | +1.516 |
| `blocks.0.bn2.weight` | 128 | +1 | +1 | +1 | +1.027 | +0.9492 | +1.375 | +0.05909 | +1.375 |
| `blocks.0.conv1.weight` | 128x128x7x7 | -2.746e-07 | -0.08447 | +0.08594 | -0.0003508 | -0.3574 | +0.3887 | +0.01648 | +0.3887 |
| `blocks.0.conv2.weight` | 128x128x7x7 | +3.403e-06 | -0.09619 | +0.08789 | -4.312e-05 | -0.4883 | +0.2871 | +0.01635 | +0.4883 |
| `blocks.0.res_ln.bias` | 128 | 0 | 0 | 0 | -0.04138 | -0.9258 | +0.1045 | +0.09803 | +0.9258 |
| `blocks.0.res_ln.weight` | 128 | +1 | +1 | +1 | +0.9503 | +0.6406 | +1.586 | +0.13 | +1.586 |
| `blocks.0.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +1.156 | +1.156 | +1.156 | 0 | +1.156 |
| `blocks.0.se_attenuate.fc1.bias` | 32 | 0 | 0 | 0 | +0.0003323 | -0.007782 | +0.01306 | +0.004489 | +0.01306 |
| `blocks.0.se_attenuate.fc1.weight` | 32x128 | -0.001985 | -0.5 | +0.4824 | -0.001559 | -0.3906 | +0.377 | +0.09828 | +0.3906 |
| `blocks.0.se_attenuate.fc2.bias` | 128 | 0 | 0 | 0 | +0.007337 | -0.005249 | +0.02869 | +0.006348 | +0.02869 |
| `blocks.0.se_attenuate.fc2.weight` | 128x32 | +0.0003264 | -0.3574 | +0.4707 | +0.00605 | -0.2793 | +0.3691 | +0.08572 | +0.3691 |
| `blocks.1.bn1.bias` | 128 | 0 | 0 | 0 | -0.1419 | -1.586 | +0.1973 | +0.1652 | +1.586 |
| `blocks.1.bn1.running_mean` | 128 | +7.153e-07 | -0.2773 | +0.2275 | -0.0152 | -2.891 | +2.672 | +0.909 | +2.891 |
| `blocks.1.bn1.running_var` | 128 | +0.9882 | +0.8398 | +1.188 | +0.3143 | +0.02527 | +4.469 | +0.482 | +4.469 |
| `blocks.1.bn1.weight` | 128 | +1 | +1 | +1 | +0.9628 | +0.7305 | +2.078 | +0.165 | +2.078 |
| `blocks.1.bn2.bias` | 128 | 0 | 0 | 0 | -0.06986 | -0.3789 | +0.007141 | +0.07113 | +0.3789 |
| `blocks.1.bn2.running_mean` | 128 | +0.001826 | -0.7422 | +0.6992 | -0.3827 | -1.594 | +0.5391 | +0.3654 | +1.594 |
| `blocks.1.bn2.running_var` | 128 | +0.4815 | +0.2773 | +0.9492 | +0.5913 | +0.4277 | +1.133 | +0.09802 | +1.133 |
| `blocks.1.bn2.weight` | 128 | +1 | +1 | +1 | +1.035 | +0.9375 | +1.797 | +0.09647 | +1.797 |
| `blocks.1.conv1.weight` | 128x128x7x7 | +2.152e-06 | -0.08252 | +0.0874 | -0.0003371 | -0.6328 | +0.2324 | +0.01671 | +0.6328 |
| `blocks.1.conv2.weight` | 128x128x7x7 | +1.765e-05 | -0.08789 | +0.07959 | +3.266e-05 | -0.8203 | +0.3809 | +0.01679 | +0.8203 |
| `blocks.1.res_ln.bias` | 128 | 0 | 0 | 0 | -0.02065 | -0.3613 | +0.1045 | +0.06578 | +0.3613 |
| `blocks.1.res_ln.weight` | 128 | +1 | +1 | +1 | +0.9689 | +0.3066 | +1.711 | +0.1626 | +1.711 |
| `blocks.1.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +1.211 | +1.211 | +1.211 | 0 | +1.211 |
| `blocks.1.se_attenuate.fc1.bias` | 32 | 0 | 0 | 0 | -0.001059 | -0.02405 | +0.03369 | +0.01127 | +0.03369 |
| `blocks.1.se_attenuate.fc1.weight` | 32x128 | -0.0002128 | -0.4141 | +0.4355 | -0.0001242 | -0.3242 | +0.334 | +0.09928 | +0.334 |
| `blocks.1.se_attenuate.fc2.bias` | 128 | 0 | 0 | 0 | +0.005942 | -0.0957 | +0.03394 | +0.01368 | +0.0957 |
| `blocks.1.se_attenuate.fc2.weight` | 128x32 | +0.0001346 | -0.4316 | +0.4023 | +0.005786 | -0.334 | +0.4863 | +0.09077 | +0.4863 |
| `blocks.2.bn1.bias` | 128 | 0 | 0 | 0 | -0.09615 | -1.469 | +0.124 | +0.1548 | +1.469 |
| `blocks.2.bn1.running_mean` | 128 | +3.219e-05 | -0.332 | +0.2471 | +0.02249 | -2.531 | +3.891 | +0.8722 | +3.891 |
| `blocks.2.bn1.running_var` | 128 | +0.9855 | +0.793 | +1.227 | +0.4479 | +0.03174 | +13.94 | +1.3 | +13.94 |
| `blocks.2.bn1.weight` | 128 | +1 | +1 | +1 | +0.9743 | +0.8164 | +1.836 | +0.1373 | +1.836 |
| `blocks.2.bn2.bias` | 128 | 0 | 0 | 0 | +0.006179 | -0.4121 | +0.08203 | +0.05527 | +0.4121 |
| `blocks.2.bn2.running_mean` | 128 | -0.03287 | -1.109 | +0.8359 | -0.3777 | -1.875 | +0.9688 | +0.5258 | +1.875 |
| `blocks.2.bn2.running_var` | 128 | +0.4721 | +0.2949 | +0.8242 | +1.045 | +0.668 | +2.609 | +0.2363 | +2.609 |
| `blocks.2.bn2.weight` | 128 | +1 | +1 | +1 | +1.016 | +0.918 | +1.758 | +0.1044 | +1.758 |
| `blocks.2.conv1.weight` | 128x128x7x7 | -1.395e-05 | -0.08936 | +0.09033 | -0.0002906 | -0.1982 | +0.3477 | +0.01706 | +0.3477 |
| `blocks.2.conv2.weight` | 128x128x7x7 | +9.495e-06 | -0.09131 | +0.08398 | -0.00021 | -0.3574 | +0.1768 | +0.01806 | +0.3574 |
| `blocks.2.res_ln.bias` | 128 | 0 | 0 | 0 | +0.004813 | -0.007935 | +0.04712 | +0.00734 | +0.04712 |
| `blocks.2.res_ln.weight` | 128 | +1 | +1 | +1 | +1.009 | +0.9883 | +1.062 | +0.01236 | +1.062 |
| `blocks.2.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +1.117 | +1.117 | +1.117 | 0 | +1.117 |
| `blocks.2.se_attenuate.fc1.bias` | 32 | 0 | 0 | 0 | +0.001709 | -0.01038 | +0.03931 | +0.01119 | +0.03931 |
| `blocks.2.se_attenuate.fc1.weight` | 32x128 | -0.0002986 | -0.4531 | +0.4668 | +3.059e-05 | -0.3516 | +0.3633 | +0.09559 | +0.3633 |
| `blocks.2.se_attenuate.fc2.bias` | 128 | 0 | 0 | 0 | +0.01189 | -0.07324 | +0.05981 | +0.02067 | +0.07324 |
| `blocks.2.se_attenuate.fc2.weight` | 128x32 | -0.0008155 | -0.4629 | +0.4023 | +0.006418 | -0.3574 | +0.3145 | +0.08812 | +0.3574 |
| `policy.conv.bias` | 76 | 0 | 0 | 0 | +4.427e-05 | -0.09082 | +0.6953 | +0.09858 | +0.6953 |
| `policy.conv.weight` | 76x128x1x1 | +0.0004662 | -0.5 | +0.4395 | +0.0145 | -1.398 | +0.8516 | +0.1462 | +1.398 |
| `policy.pre_bn.bias` | 128 | 0 | 0 | 0 | +0.02818 | -0.1309 | +0.3516 | +0.07849 | +0.3516 |
| `policy.pre_bn.running_mean` | 128 | +0.002271 | -1.328 | +1.305 | -0.4214 | -1.945 | +1.367 | +0.6289 | +1.945 |
| `policy.pre_bn.running_var` | 128 | +0.6386 | +0.3574 | +1.109 | +0.8098 | +0.4062 | +2.703 | +0.3558 | +2.703 |
| `policy.pre_bn.weight` | 128 | +1 | +1 | +1 | +1.074 | +0.8242 | +1.484 | +0.1139 | +1.484 |
| `policy.pre_conv.weight` | 128x128x1x1 | +3.854e-05 | -0.4707 | +0.5312 | -0.00413 | -0.5742 | +0.5273 | +0.1126 | +0.5742 |
| `stem.bn.bias` | 128 | 0 | 0 | 0 | -0.02351 | -0.4707 | +0.1699 | +0.1054 | +0.4707 |
| `stem.bn.running_mean` | 128 | -0.003496 | -0.3945 | +0.4355 | -0.01772 | -0.3418 | +0.3555 | +0.1703 | +0.3555 |
| `stem.bn.running_var` | 128 | +0.06817 | +0.02722 | +0.1768 | +0.08358 | +0.03564 | +0.2793 | +0.03744 | +0.2793 |
| `stem.bn.weight` | 128 | +1 | +1 | +1 | +0.9412 | +0.3164 | +1.445 | +0.1403 | +1.445 |
| `stem.conv.weight` | 128x30x7x7 | +1.62e-05 | -0.1592 | +0.1826 | +0.0002749 | -0.7188 | +1.547 | +0.04095 | +1.547 |
| `tower_final_bn.bias` | 128 | 0 | 0 | 0 | +0.1501 | -0.2305 | +0.7969 | +0.1455 | +0.7969 |
| `tower_final_bn.running_mean` | 128 | +9.924e-06 | -0.3672 | +0.3555 | +0.005105 | -1.445 | +3.312 | +0.742 | +3.312 |
| `tower_final_bn.running_var` | 128 | +0.9809 | +0.7773 | +1.195 | +0.4657 | +0.1182 | +8 | +0.7751 | +8 |
| `tower_final_bn.weight` | 128 | +1 | +1 | +1 | +0.9791 | +0.7812 | +1.727 | +0.1617 | +1.727 |
| `value.bn.bias` | 16 | 0 | 0 | 0 | -0.2423 | -0.3066 | -0.1289 | +0.0424 | +0.3066 |
| `value.bn.running_mean` | 16 | +0.1433 | -0.668 | +0.8242 | -0.0284 | -1.023 | +0.7188 | +0.4528 | +1.023 |
| `value.bn.running_var` | 16 | +0.7742 | +0.4805 | +1.711 | +0.781 | +0.5625 | +1.445 | +0.2129 | +1.445 |
| `value.bn.weight` | 16 | +1 | +1 | +1 | +0.6777 | +0.5781 | +0.9648 | +0.1047 | +0.9648 |
| `value.conv.weight` | 16x128x1x1 | +0.002821 | -0.4355 | +0.3984 | -0.0006982 | -0.3613 | +0.6133 | +0.102 | +0.6133 |
| `value.fc1.bias` | 128 | 0 | 0 | 0 | -0.003076 | -0.0481 | +0.05078 | +0.01315 | +0.05078 |
| `value.fc1.weight` | 128x1024 | +3.773e-05 | -0.1826 | +0.1973 | -0.001301 | -0.1523 | +0.1582 | +0.03505 | +0.1582 |
| `value.wdl_fc2.bias` | 3 | +0.5964 | 0 | +1.789 | +0.5951 | +0.4805 | +0.8008 | +0.1458 | +0.8008 |
| `value.wdl_fc2.weight` | 3x128 | -0.0006274 | -0.334 | +0.416 | -0.000483 | -0.2988 | +0.4492 | +0.1058 | +0.4492 |

### `se_none` — none, seed 1 (step 0 → 32,036)

| tensor | shape | fresh mean | fresh min | fresh max | final mean | final min | final max | final std | final abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `blocks.0.bn1.bias` | 128 | 0 | 0 | 0 | -0.07711 | -0.3477 | +0.08643 | +0.07138 | +0.3477 |
| `blocks.0.bn1.running_mean` | 128 | +0.0001165 | -0.00164 | +0.002853 | -0.01772 | -0.2354 | +0.1719 | +0.06871 | +0.2354 |
| `blocks.0.bn1.running_var` | 128 | +0.9993 | +0.9961 | +1 | +0.9479 | +0.3438 | +1.883 | +0.2676 | +1.883 |
| `blocks.0.bn1.weight` | 128 | +1 | +1 | +1 | +0.9829 | +0.7617 | +1.734 | +0.1544 | +1.734 |
| `blocks.0.bn2.bias` | 128 | 0 | 0 | 0 | -0.04342 | -0.8164 | +0.0481 | +0.08207 | +0.8164 |
| `blocks.0.bn2.running_mean` | 128 | +0.03009 | -0.8867 | +0.9844 | -0.6032 | -2.344 | +0.7539 | +0.5442 | +2.344 |
| `blocks.0.bn2.running_var` | 128 | +0.4769 | +0.2852 | +1.016 | +0.8363 | +0.5352 | +2.438 | +0.2626 | +2.438 |
| `blocks.0.bn2.weight` | 128 | +1 | +1 | +1 | +1.011 | +0.9219 | +1.539 | +0.06441 | +1.539 |
| `blocks.0.conv1.weight` | 128x128x7x7 | +1.697e-05 | -0.08887 | +0.08398 | -0.0003735 | -0.2969 | +0.332 | +0.01645 | +0.332 |
| `blocks.0.conv2.weight` | 128x128x7x7 | +7.671e-06 | -0.08594 | +0.08398 | +4.5e-06 | -0.9492 | +0.334 | +0.01637 | +0.9492 |
| `blocks.0.res_ln.bias` | 128 | 0 | 0 | 0 | -0.0389 | -0.5508 | +0.09033 | +0.06787 | +0.5508 |
| `blocks.0.res_ln.weight` | 128 | +1 | +1 | +1 | +0.9712 | +0.5039 | +1.312 | +0.1263 | +1.312 |
| `blocks.0.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +1.039 | +1.039 | +1.039 | 0 | +1.039 |
| `blocks.1.bn1.bias` | 128 | 0 | 0 | 0 | -0.1212 | -1 | +0.03052 | +0.1457 | +1 |
| `blocks.1.bn1.running_mean` | 128 | -5.925e-05 | -0.4941 | +0.4395 | -0.01078 | -2.062 | +2.719 | +0.8881 | +2.719 |
| `blocks.1.bn1.running_var` | 128 | +0.9736 | +0.625 | +1.234 | +0.3449 | +0.02356 | +2.5 | +0.363 | +2.5 |
| `blocks.1.bn1.weight` | 128 | +1 | +1 | +1 | +0.9691 | +0.8086 | +2.234 | +0.1615 | +2.234 |
| `blocks.1.bn2.bias` | 128 | 0 | 0 | 0 | -0.08413 | -0.5117 | +0.01263 | +0.08552 | +0.5117 |
| `blocks.1.bn2.running_mean` | 128 | +0.04068 | -0.957 | +0.8906 | -0.3285 | -1.484 | +0.9062 | +0.3768 | +1.484 |
| `blocks.1.bn2.running_var` | 128 | +0.5005 | +0.3086 | +0.8984 | +0.7128 | +0.459 | +3.453 | +0.2962 | +3.453 |
| `blocks.1.bn2.weight` | 128 | +1 | +1 | +1 | +0.9977 | +0.8828 | +2.25 | +0.1257 | +2.25 |
| `blocks.1.conv1.weight` | 128x128x7x7 | +2.934e-05 | -0.09424 | +0.08594 | -0.0002837 | -0.7148 | +0.2949 | +0.01683 | +0.7148 |
| `blocks.1.conv2.weight` | 128x128x7x7 | +1.065e-05 | -0.08643 | +0.08301 | +2.85e-05 | -1.008 | +0.2676 | +0.01684 | +1.008 |
| `blocks.1.res_ln.bias` | 128 | 0 | 0 | 0 | -0.03021 | -0.2295 | +0.1138 | +0.05579 | +0.2295 |
| `blocks.1.res_ln.weight` | 128 | +1 | +1 | +1 | +0.9847 | +0.668 | +1.547 | +0.142 | +1.547 |
| `blocks.1.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +0.9531 | +0.9531 | +0.9531 | 0 | +0.9531 |
| `blocks.2.bn1.bias` | 128 | 0 | 0 | 0 | -0.1133 | -1.195 | +0.1924 | +0.1631 | +1.195 |
| `blocks.2.bn1.running_mean` | 128 | +6.08e-05 | -0.4336 | +0.4023 | +0.02206 | -2.641 | +3.359 | +0.9295 | +3.359 |
| `blocks.2.bn1.running_var` | 128 | +0.9606 | +0.6641 | +1.273 | +0.3829 | +0.05225 | +5.625 | +0.6442 | +5.625 |
| `blocks.2.bn1.weight` | 128 | +1 | +1 | +1 | +0.9644 | +0.8242 | +2.094 | +0.1805 | +2.094 |
| `blocks.2.bn2.bias` | 128 | 0 | 0 | 0 | +0.005777 | -0.373 | +0.2188 | +0.08781 | +0.373 |
| `blocks.2.bn2.running_mean` | 128 | -0.004467 | -1.094 | +0.8164 | -0.3058 | -1.688 | +1.219 | +0.6229 | +1.688 |
| `blocks.2.bn2.running_var` | 128 | +0.5013 | +0.3105 | +0.8711 | +1.153 | +0.7656 | +2.359 | +0.2832 | +2.359 |
| `blocks.2.bn2.weight` | 128 | +1 | +1 | +1 | +1.003 | +0.8945 | +1.555 | +0.08686 | +1.555 |
| `blocks.2.conv1.weight` | 128x128x7x7 | +4.104e-06 | -0.08887 | +0.08789 | -0.0002401 | -0.3008 | +0.4141 | +0.01733 | +0.4141 |
| `blocks.2.conv2.weight` | 128x128x7x7 | -5.149e-06 | -0.08887 | +0.09131 | +2.684e-05 | -0.3789 | +0.1953 | +0.01859 | +0.3789 |
| `blocks.2.res_ln.bias` | 128 | 0 | 0 | 0 | +0.003931 | -0.01135 | +0.02222 | +0.007438 | +0.02222 |
| `blocks.2.res_ln.weight` | 128 | +1 | +1 | +1 | +1.01 | +0.9922 | +1.086 | +0.01633 | +1.086 |
| `blocks.2.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +0.957 | +0.957 | +0.957 | 0 | +0.957 |
| `policy.conv.bias` | 76 | 0 | 0 | 0 | -6.29e-05 | -0.1035 | +0.4512 | +0.0821 | +0.4512 |
| `policy.conv.weight` | 76x128x1x1 | +0.001251 | -0.5039 | +0.5195 | +0.01399 | -1.375 | +0.9062 | +0.1475 | +1.375 |
| `policy.pre_bn.bias` | 128 | 0 | 0 | 0 | +0.01745 | -0.1807 | +0.2637 | +0.07501 | +0.2637 |
| `policy.pre_bn.running_mean` | 128 | -0.05045 | -1.312 | +1.938 | -0.5908 | -3.359 | +0.9414 | +0.6891 | +3.359 |
| `policy.pre_bn.running_var` | 128 | +0.6839 | +0.3145 | +1.422 | +0.737 | +0.2617 | +2.188 | +0.2877 | +2.188 |
| `policy.pre_bn.weight` | 128 | +1 | +1 | +1 | +1.084 | +0.7539 | +1.758 | +0.1291 | +1.758 |
| `policy.pre_conv.weight` | 128x128x1x1 | -0.001009 | -0.5273 | +0.4766 | -0.00683 | -0.707 | +0.5195 | +0.1126 | +0.707 |
| `stem.bn.bias` | 128 | 0 | 0 | 0 | -0.01771 | -0.2354 | +0.1719 | +0.06871 | +0.2354 |
| `stem.bn.running_mean` | 128 | -0.01956 | -0.3926 | +0.3105 | -0.03318 | -0.3691 | +0.3438 | +0.1507 | +0.3691 |
| `stem.bn.running_var` | 128 | +0.06684 | +0.02563 | +0.2021 | +0.06914 | +0.02344 | +0.2197 | +0.03168 | +0.2197 |
| `stem.bn.weight` | 128 | +1 | +1 | +1 | +0.9646 | +0.5859 | +1.375 | +0.1335 | +1.375 |
| `stem.conv.weight` | 128x30x7x7 | -0.0001705 | -0.1602 | +0.1592 | +1.497e-05 | -0.6016 | +1.359 | +0.03989 | +1.359 |
| `tower_final_bn.bias` | 128 | 0 | 0 | 0 | +0.1209 | -0.1079 | +0.7969 | +0.1237 | +0.7969 |
| `tower_final_bn.running_mean` | 128 | -3.201e-05 | -0.5391 | +0.4531 | +0.007793 | -1.281 | +2.828 | +0.7632 | +2.828 |
| `tower_final_bn.running_var` | 128 | +0.9526 | +0.6367 | +1.32 | +0.4476 | +0.1196 | +3.688 | +0.429 | +3.688 |
| `tower_final_bn.weight` | 128 | +1 | +1 | +1 | +0.9812 | +0.6875 | +1.906 | +0.1699 | +1.906 |
| `value.bn.bias` | 16 | 0 | 0 | 0 | -0.2478 | -0.3965 | -0.1855 | +0.05137 | +0.3965 |
| `value.bn.running_mean` | 16 | +0.1224 | -0.459 | +0.7109 | -0.0945 | -0.8203 | +0.498 | +0.3557 | +0.8203 |
| `value.bn.running_var` | 16 | +0.6855 | +0.3965 | +1.047 | +0.7141 | +0.3457 | +1.617 | +0.3515 | +1.617 |
| `value.bn.weight` | 16 | +1 | +1 | +1 | +0.6831 | +0.5898 | +0.957 | +0.08118 | +0.957 |
| `value.conv.weight` | 16x128x1x1 | +0.002384 | -0.377 | +0.418 | -0.001968 | -0.3047 | +0.334 | +0.1008 | +0.334 |
| `value.fc1.bias` | 128 | 0 | 0 | 0 | -0.002758 | -0.04248 | +0.04761 | +0.01294 | +0.04761 |
| `value.fc1.weight` | 128x1024 | +5.386e-06 | -0.1904 | +0.1973 | -0.001533 | -0.1709 | +0.1553 | +0.03472 | +0.1709 |
| `value.wdl_fc2.bias` | 3 | +0.5964 | 0 | +1.789 | +0.5964 | +0.4551 | +0.8359 | +0.1703 | +0.8359 |
| `value.wdl_fc2.weight` | 3x128 | -0.002068 | -0.3711 | +0.3652 | -0.001579 | -0.3789 | +0.2793 | +0.1095 | +0.3789 |

### `se_zb1` — zero-beta scale+bias, seed 1 (step 0 → 5,030)

| tensor | shape | fresh mean | fresh min | fresh max | final mean | final min | final max | final std | final abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `blocks.0.bn1.bias` | 128 | 0 | 0 | 0 | -0.03389 | -0.2236 | +0.1143 | +0.04311 | +0.2236 |
| `blocks.0.bn1.running_mean` | 128 | -6.939e-05 | -0.002441 | +0.001755 | -0.009279 | -0.1959 | +0.1157 | +0.04875 | +0.1959 |
| `blocks.0.bn1.running_var` | 128 | +1 | +0.9922 | +1.008 | +0.978 | +0.2449 | +2.342 | +0.2355 | +2.342 |
| `blocks.0.bn1.weight` | 128 | +1 | +1 | +1 | +0.9941 | +0.8735 | +1.629 | +0.09596 | +1.629 |
| `blocks.0.bn2.bias` | 128 | 0 | 0 | 0 | -0.01714 | -0.1233 | +0.02209 | +0.02086 | +0.1233 |
| `blocks.0.bn2.running_mean` | 128 | -0.02343 | -0.9023 | +0.9219 | -0.5803 | -2.865 | +0.5791 | +0.5045 | +2.865 |
| `blocks.0.bn2.running_var` | 128 | +0.5338 | +0.2656 | +1.312 | +0.6784 | +0.3839 | +1.218 | +0.166 | +1.218 |
| `blocks.0.bn2.weight` | 128 | +1 | +1 | +1 | +1.002 | +0.9465 | +1.13 | +0.02888 | +1.13 |
| `blocks.0.conv1.weight` | 128x128x7x7 | -6.688e-06 | -0.09814 | +0.08447 | -0.0003609 | -0.155 | +0.2784 | +0.01732 | +0.2784 |
| `blocks.0.conv2.weight` | 128x128x7x7 | -3.342e-06 | -0.08496 | +0.09229 | +3.32e-05 | -0.2066 | +0.2023 | +0.01719 | +0.2066 |
| `blocks.0.res_ln.bias` | 128 | 0 | 0 | 0 | -0.01328 | -0.4455 | +0.04838 | +0.04712 | +0.4455 |
| `blocks.0.res_ln.weight` | 128 | +1 | +1 | +1 | +0.984 | +0.5242 | +1.391 | +0.09475 | +1.391 |
| `blocks.0.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +0.8499 | +0.8499 | +0.8499 | 0 | +0.8499 |
| `blocks.0.se_scalebias.fc1.bias` | 32 | 0 | 0 | 0 | +0.004479 | -0.00439 | +0.03286 | +0.008242 | +0.03286 |
| `blocks.0.se_scalebias.fc1.weight` | 32x128 | -0.0004187 | -0.4746 | +0.4902 | -0.0003189 | -0.4347 | +0.4435 | +0.1158 | +0.4435 |
| `blocks.0.se_scalebias.fc2.bias` | 256 | 0 | 0 | 0 | +0.002082 | -0.07864 | +0.04524 | +0.01448 | +0.07864 |
| `blocks.0.se_scalebias.fc2.bias [gamma]` | 256 | 0 | 0 | 0 | +0.004165 | -0.0135 | +0.02727 | +0.005569 | +0.02727 |
| `blocks.0.se_scalebias.fc2.bias [beta]` | 256 | 0 | 0 | 0 | -1.405e-06 | -0.07864 | +0.04524 | +0.01948 | +0.07864 |
| `blocks.0.se_scalebias.fc2.weight` | 256x32 | -0.0007758 | -0.3438 | +0.2676 | +0.001155 | -0.314 | +0.2823 | +0.05541 | +0.314 |
| `blocks.0.se_scalebias.fc2.weight [gamma]` | 256x32 | -0.001552 | -0.3438 | +0.2676 | +0.002313 | -0.314 | +0.2823 | +0.07538 | +0.314 |
| `blocks.0.se_scalebias.fc2.weight [beta]` | 256x32 | 0 | 0 | 0 | -2.938e-06 | -0.1691 | +0.1472 | +0.02133 | +0.1691 |
| `blocks.1.bn1.bias` | 128 | 0 | 0 | 0 | -0.04417 | -0.486 | +0.07042 | +0.06016 | +0.486 |
| `blocks.1.bn1.running_mean` | 128 | +1.216e-05 | -0.3848 | +0.3945 | +0.00226 | -2.485 | +2.997 | +0.8196 | +2.997 |
| `blocks.1.bn1.running_var` | 128 | +0.976 | +0.7383 | +1.219 | +0.4197 | +0.01756 | +5.129 | +0.5147 | +5.129 |
| `blocks.1.bn1.weight` | 128 | +1 | +1 | +1 | +0.9915 | +0.8622 | +1.589 | +0.1086 | +1.589 |
| `blocks.1.bn2.bias` | 128 | 0 | 0 | 0 | -0.01623 | -0.1674 | +0.02437 | +0.0304 | +0.1674 |
| `blocks.1.bn2.running_mean` | 128 | +0.004915 | -1.156 | +0.875 | -0.3789 | -1.276 | +0.6228 | +0.3966 | +1.276 |
| `blocks.1.bn2.running_var` | 128 | +0.479 | +0.3125 | +0.8242 | +0.6653 | +0.4851 | +0.9759 | +0.09135 | +0.9759 |
| `blocks.1.bn2.weight` | 128 | +1 | +1 | +1 | +1.012 | +0.9531 | +1.196 | +0.0375 | +1.196 |
| `blocks.1.conv1.weight` | 128x128x7x7 | -2.195e-06 | -0.08545 | +0.08545 | -0.0002827 | -0.1217 | +0.265 | +0.01765 | +0.265 |
| `blocks.1.conv2.weight` | 128x128x7x7 | +8.478e-06 | -0.09082 | +0.09033 | -6.343e-05 | -0.1212 | +0.2408 | +0.01764 | +0.2408 |
| `blocks.1.res_ln.bias` | 128 | 0 | 0 | 0 | -0.008271 | -0.2417 | +0.07144 | +0.03727 | +0.2417 |
| `blocks.1.res_ln.weight` | 128 | +1 | +1 | +1 | +0.989 | +0.5834 | +1.417 | +0.09315 | +1.417 |
| `blocks.1.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +0.9646 | +0.9646 | +0.9646 | 0 | +0.9646 |
| `blocks.1.se_scalebias.fc1.bias` | 32 | 0 | 0 | 0 | +0.0004109 | -0.006402 | +0.01917 | +0.004407 | +0.01917 |
| `blocks.1.se_scalebias.fc1.weight` | 32x128 | +0.001356 | -0.4062 | +0.4258 | +0.001301 | -0.371 | +0.3875 | +0.1154 | +0.3875 |
| `blocks.1.se_scalebias.fc2.bias` | 256 | 0 | 0 | 0 | +0.001694 | -0.0736 | +0.03443 | +0.01134 | +0.0736 |
| `blocks.1.se_scalebias.fc2.bias [gamma]` | 256 | 0 | 0 | 0 | +0.003386 | -0.04216 | +0.03443 | +0.01093 | +0.04216 |
| `blocks.1.se_scalebias.fc2.bias [beta]` | 256 | 0 | 0 | 0 | +2.28e-06 | -0.0736 | +0.02985 | +0.01149 | +0.0736 |
| `blocks.1.se_scalebias.fc2.weight` | 256x32 | -0.000401 | -0.2656 | +0.3398 | +0.0005468 | -0.2447 | +0.5088 | +0.05537 | +0.5088 |
| `blocks.1.se_scalebias.fc2.weight [gamma]` | 256x32 | -0.0008021 | -0.2656 | +0.3398 | +0.001095 | -0.2447 | +0.5088 | +0.07766 | +0.5088 |
| `blocks.1.se_scalebias.fc2.weight [beta]` | 256x32 | 0 | 0 | 0 | -1.792e-06 | -0.1393 | +0.08455 | +0.00997 | +0.1393 |
| `blocks.2.bn1.bias` | 128 | 0 | 0 | 0 | -0.03787 | -0.8283 | +0.1675 | +0.09342 | +0.8283 |
| `blocks.2.bn1.running_mean` | 128 | +3.231e-05 | -0.5469 | +0.5 | +0.01641 | -2.565 | +2.956 | +0.8172 | +2.956 |
| `blocks.2.bn1.running_var` | 128 | +0.9696 | +0.7305 | +1.219 | +0.4431 | +0.04528 | +10.6 | +0.97 | +10.6 |
| `blocks.2.bn1.weight` | 128 | +1 | +1 | +1 | +0.9878 | +0.8645 | +1.78 | +0.1219 | +1.78 |
| `blocks.2.bn2.bias` | 128 | 0 | 0 | 0 | +0.0106 | -0.08593 | +0.07969 | +0.03007 | +0.08593 |
| `blocks.2.bn2.running_mean` | 128 | -0.005059 | -1.148 | +0.9414 | -0.2897 | -1.564 | +0.963 | +0.5265 | +1.564 |
| `blocks.2.bn2.running_var` | 128 | +0.4917 | +0.3105 | +0.8906 | +1.071 | +0.8016 | +1.951 | +0.1717 | +1.951 |
| `blocks.2.bn2.weight` | 128 | +1 | +1 | +1 | +1.008 | +0.9435 | +1.139 | +0.03194 | +1.139 |
| `blocks.2.conv1.weight` | 128x128x7x7 | +7.125e-07 | -0.0835 | +0.08789 | -0.0002114 | -0.1658 | +0.212 | +0.01809 | +0.212 |
| `blocks.2.conv2.weight` | 128x128x7x7 | -1.386e-05 | -0.08594 | +0.08301 | -0.0001096 | -0.1428 | +0.1369 | +0.01846 | +0.1428 |
| `blocks.2.res_ln.bias` | 128 | 0 | 0 | 0 | +0.002408 | -0.009658 | +0.01659 | +0.004113 | +0.01659 |
| `blocks.2.res_ln.weight` | 128 | +1 | +1 | +1 | +1.003 | +0.9902 | +1.026 | +0.004704 | +1.026 |
| `blocks.2.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +0.9539 | +0.9539 | +0.9539 | 0 | +0.9539 |
| `blocks.2.se_scalebias.fc1.bias` | 32 | 0 | 0 | 0 | -0.0005057 | -0.00872 | +0.02373 | +0.006585 | +0.02373 |
| `blocks.2.se_scalebias.fc1.weight` | 32x128 | -0.00436 | -0.4004 | +0.4336 | -0.003752 | -0.3608 | +0.3965 | +0.1155 | +0.3965 |
| `blocks.2.se_scalebias.fc2.bias` | 256 | 0 | 0 | 0 | +0.002019 | -0.06971 | +0.03731 | +0.01294 | +0.06971 |
| `blocks.2.se_scalebias.fc2.bias [gamma]` | 256 | 0 | 0 | 0 | +0.004024 | -0.03198 | +0.0301 | +0.01193 | +0.03198 |
| `blocks.2.se_scalebias.fc2.bias [beta]` | 256 | 0 | 0 | 0 | +1.371e-05 | -0.06971 | +0.03731 | +0.01359 | +0.06971 |
| `blocks.2.se_scalebias.fc2.weight` | 256x32 | +0.0002044 | -0.3789 | +0.2852 | +0.001754 | -0.3461 | +0.2653 | +0.0549 | +0.3461 |
| `blocks.2.se_scalebias.fc2.weight [gamma]` | 256x32 | +0.0004087 | -0.3789 | +0.2852 | +0.003499 | -0.3461 | +0.2653 | +0.07601 | +0.3461 |
| `blocks.2.se_scalebias.fc2.weight [beta]` | 256x32 | 0 | 0 | 0 | +9.858e-06 | -0.1679 | +0.158 | +0.01563 | +0.1679 |
| `policy.conv.bias` | 76 | 0 | 0 | 0 | -3.174e-05 | -0.07297 | +0.5958 | +0.08394 | +0.5958 |
| `policy.conv.weight` | 76x128x1x1 | -0.001031 | -0.4727 | +0.4766 | +0.009894 | -1.049 | +0.8884 | +0.1462 | +1.049 |
| `policy.pre_bn.bias` | 128 | 0 | 0 | 0 | +0.03076 | -0.1107 | +0.3009 | +0.06765 | +0.3009 |
| `policy.pre_bn.running_mean` | 128 | +0.003215 | -1.367 | +2 | -0.03965 | -1.424 | +1.543 | +0.5671 | +1.543 |
| `policy.pre_bn.running_var` | 128 | +0.7019 | +0.3906 | +1.703 | +0.983 | +0.4303 | +3.51 | +0.4036 | +3.51 |
| `policy.pre_bn.weight` | 128 | +1 | +1 | +1 | +1.064 | +0.8663 | +1.421 | +0.09998 | +1.421 |
| `policy.pre_conv.weight` | 128x128x1x1 | +1.857e-05 | -0.4629 | +0.5195 | -0.0001955 | -0.4549 | +0.5165 | +0.1227 | +0.5165 |
| `stem.bn.bias` | 128 | 0 | 0 | 0 | -0.009308 | -0.1962 | +0.1165 | +0.04879 | +0.1962 |
| `stem.bn.running_mean` | 128 | +0.006164 | -0.4453 | +0.2871 | -0.001327 | -0.3179 | +0.3711 | +0.1794 | +0.3711 |
| `stem.bn.running_var` | 128 | +0.06212 | +0.0238 | +0.2383 | +0.0973 | +0.0316 | +0.2934 | +0.05257 | +0.2934 |
| `stem.bn.weight` | 128 | +1 | +1 | +1 | +0.9825 | +0.4937 | +1.53 | +0.1121 | +1.53 |
| `stem.conv.weight` | 128x30x7x7 | +4.382e-05 | -0.165 | +0.1953 | +9.904e-05 | -0.8164 | +1.785 | +0.04164 | +1.785 |
| `tower_final_bn.bias` | 128 | 0 | 0 | 0 | +0.1046 | -0.1105 | +0.3103 | +0.0814 | +0.3103 |
| `tower_final_bn.running_mean` | 128 | +2.36e-05 | -0.5273 | +0.625 | +0.002552 | -1.86 | +2.742 | +0.7673 | +2.742 |
| `tower_final_bn.running_var` | 128 | +0.9619 | +0.7539 | +1.219 | +0.416 | +0.07718 | +6.078 | +0.5805 | +6.078 |
| `tower_final_bn.weight` | 128 | +1 | +1 | +1 | +0.994 | +0.8443 | +1.367 | +0.08541 | +1.367 |
| `value.bn.bias` | 16 | 0 | 0 | 0 | -0.1546 | -0.2157 | -0.09061 | +0.02938 | +0.2157 |
| `value.bn.running_mean` | 16 | +0.04053 | -1.031 | +0.7695 | -0.1091 | -1.486 | +0.6008 | +0.5241 | +1.486 |
| `value.bn.running_var` | 16 | +0.6416 | +0.459 | +1.141 | +1.06 | +0.6252 | +1.804 | +0.2751 | +1.804 |
| `value.bn.weight` | 16 | +1 | +1 | +1 | +0.7868 | +0.7235 | +1.04 | +0.07351 | +1.04 |
| `value.conv.weight` | 16x128x1x1 | +0.0007972 | -0.3984 | +0.4082 | -0.002173 | -0.3868 | +0.3815 | +0.117 | +0.3868 |
| `value.fc1.bias` | 128 | 0 | 0 | 0 | -0.003501 | -0.02682 | +0.02497 | +0.007657 | +0.02682 |
| `value.fc1.weight` | 128x1024 | +3.011e-06 | -0.1904 | +0.1934 | -0.00158 | -0.1771 | +0.1737 | +0.04053 | +0.1771 |
| `value.wdl_fc2.bias` | 3 | +0.5964 | 0 | +1.789 | +0.597 | +0.2461 | +1.26 | +0.4693 | +1.26 |
| `value.wdl_fc2.weight` | 3x128 | +0.006675 | -0.3867 | +0.334 | +0.006118 | -0.4057 | +0.3364 | +0.1137 | +0.4057 |

### `se_sb2` — scale+bias, seed 2 (step 0 → 7,282)

| tensor | shape | fresh mean | fresh min | fresh max | final mean | final min | final max | final std | final abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `blocks.0.bn1.bias` | 128 | 0 | 0 | 0 | -0.02362 | -0.09075 | +0.1894 | +0.04875 | +0.1894 |
| `blocks.0.bn1.running_mean` | 128 | +1.776e-05 | -0.002579 | +0.002045 | -0.004566 | -0.1299 | +0.08507 | +0.03774 | +0.1299 |
| `blocks.0.bn1.running_var` | 128 | +0.9995 | +0.9922 | +1.008 | +0.9819 | +0.4842 | +1.744 | +0.2134 | +1.744 |
| `blocks.0.bn1.weight` | 128 | +1 | +1 | +1 | +0.9961 | +0.8584 | +1.248 | +0.07346 | +1.248 |
| `blocks.0.bn2.bias` | 128 | 0 | 0 | 0 | -0.01563 | -0.1901 | +0.02053 | +0.02776 | +0.1901 |
| `blocks.0.bn2.running_mean` | 128 | +0.019 | -0.7422 | +0.8828 | -0.5232 | -2.314 | +0.9622 | +0.5589 | +2.314 |
| `blocks.0.bn2.running_var` | 128 | +0.5308 | +0.2656 | +0.9688 | +0.7415 | +0.4227 | +1.8 | +0.2085 | +1.8 |
| `blocks.0.bn2.weight` | 128 | +1 | +1 | +1 | +1.006 | +0.969 | +1.178 | +0.03082 | +1.178 |
| `blocks.0.conv1.weight` | 128x128x7x7 | +1.881e-05 | -0.0835 | +0.09375 | -0.0003192 | -0.1164 | +0.142 | +0.01734 | +0.142 |
| `blocks.0.conv2.weight` | 128x128x7x7 | -1.854e-05 | -0.08301 | +0.09229 | -7.387e-05 | -0.2447 | +0.2328 | +0.01719 | +0.2447 |
| `blocks.0.res_ln.bias` | 128 | 0 | 0 | 0 | -0.01148 | -0.2001 | +0.05072 | +0.03668 | +0.2001 |
| `blocks.0.res_ln.weight` | 128 | +1 | +1 | +1 | +0.9948 | +0.8266 | +1.23 | +0.0752 | +1.23 |
| `blocks.0.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +0.9111 | +0.9111 | +0.9111 | 0 | +0.9111 |
| `blocks.0.se_scalebias.fc1.bias` | 32 | 0 | 0 | 0 | +0.004701 | -0.005059 | +0.05136 | +0.01288 | +0.05136 |
| `blocks.0.se_scalebias.fc1.weight` | 32x128 | +0.005474 | -0.4375 | +0.4746 | +0.004992 | -0.398 | +0.4291 | +0.1111 | +0.4291 |
| `blocks.0.se_scalebias.fc2.bias` | 256 | 0 | 0 | 0 | +0.002128 | -0.0576 | +0.03554 | +0.0116 | +0.0576 |
| `blocks.0.se_scalebias.fc2.bias [gamma]` | 256 | 0 | 0 | 0 | +0.004258 | -0.006772 | +0.02228 | +0.004313 | +0.02228 |
| `blocks.0.se_scalebias.fc2.bias [beta]` | 256 | 0 | 0 | 0 | -7.517e-07 | -0.0576 | +0.03554 | +0.01554 | +0.0576 |
| `blocks.0.se_scalebias.fc2.weight` | 256x32 | -0.0004752 | -0.3711 | +0.3516 | +0.00167 | -0.311 | +0.3573 | +0.07678 | +0.3573 |
| `blocks.0.se_scalebias.fc2.weight [gamma]` | 256x32 | +0.0001014 | -0.2598 | +0.3066 | +0.004294 | -0.2227 | +0.3035 | +0.07476 | +0.3035 |
| `blocks.0.se_scalebias.fc2.weight [beta]` | 256x32 | -0.001052 | -0.3711 | +0.3516 | -0.0009548 | -0.311 | +0.3573 | +0.07866 | +0.3573 |
| `blocks.1.bn1.bias` | 128 | 0 | 0 | 0 | -0.03859 | -0.4083 | +0.1012 | +0.06726 | +0.4083 |
| `blocks.1.bn1.running_mean` | 128 | -1.168e-05 | -0.3516 | +0.4219 | +0.00973 | -2.751 | +3.226 | +0.9134 | +3.226 |
| `blocks.1.bn1.running_var` | 128 | +0.969 | +0.707 | +1.25 | +0.2583 | +0.0215 | +1.201 | +0.1717 | +1.201 |
| `blocks.1.bn1.weight` | 128 | +1 | +1 | +1 | +0.9929 | +0.8494 | +1.541 | +0.09284 | +1.541 |
| `blocks.1.bn2.bias` | 128 | 0 | 0 | 0 | -0.03597 | -0.3015 | +0.01762 | +0.04118 | +0.3015 |
| `blocks.1.bn2.running_mean` | 128 | -0.03469 | -0.9141 | +0.9648 | -0.4606 | -1.457 | +0.4701 | +0.4096 | +1.457 |
| `blocks.1.bn2.running_var` | 128 | +0.4828 | +0.2598 | +0.7656 | +0.763 | +0.4438 | +1.363 | +0.1241 | +1.363 |
| `blocks.1.bn2.weight` | 128 | +1 | +1 | +1 | +1.006 | +0.9315 | +1.336 | +0.05022 | +1.336 |
| `blocks.1.conv1.weight` | 128x128x7x7 | -1.644e-05 | -0.08984 | +0.08691 | -0.0003279 | -0.1662 | +0.2127 | +0.01761 | +0.2127 |
| `blocks.1.conv2.weight` | 128x128x7x7 | -2.565e-05 | -0.07861 | +0.09033 | -2.759e-05 | -0.1621 | +0.282 | +0.01758 | +0.282 |
| `blocks.1.res_ln.bias` | 128 | 0 | 0 | 0 | -0.008798 | -0.1292 | +0.06417 | +0.02786 | +0.1292 |
| `blocks.1.res_ln.weight` | 128 | +1 | +1 | +1 | +0.9941 | +0.7871 | +1.24 | +0.08513 | +1.24 |
| `blocks.1.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +0.8551 | +0.8551 | +0.8551 | 0 | +0.8551 |
| `blocks.1.se_scalebias.fc1.bias` | 32 | 0 | 0 | 0 | -4.105e-05 | -0.009616 | +0.01502 | +0.004894 | +0.01502 |
| `blocks.1.se_scalebias.fc1.weight` | 32x128 | +0.0007602 | -0.4141 | +0.4629 | +0.0007859 | -0.3759 | +0.4201 | +0.1118 | +0.4201 |
| `blocks.1.se_scalebias.fc2.bias` | 256 | 0 | 0 | 0 | +0.001539 | -0.04183 | +0.03074 | +0.009039 | +0.04183 |
| `blocks.1.se_scalebias.fc2.bias [gamma]` | 256 | 0 | 0 | 0 | +0.003079 | -0.02204 | +0.03074 | +0.008798 | +0.03074 |
| `blocks.1.se_scalebias.fc2.bias [beta]` | 256 | 0 | 0 | 0 | -1.367e-06 | -0.04183 | +0.0218 | +0.009015 | +0.04183 |
| `blocks.1.se_scalebias.fc2.weight` | 256x32 | +0.001447 | -0.3242 | +0.3145 | +0.002738 | -0.2939 | +0.2845 | +0.0759 | +0.2939 |
| `blocks.1.se_scalebias.fc2.weight [gamma]` | 256x32 | +0.001967 | -0.2832 | +0.2891 | +0.004636 | -0.2751 | +0.2544 | +0.07605 | +0.2751 |
| `blocks.1.se_scalebias.fc2.weight [beta]` | 256x32 | +0.0009256 | -0.3242 | +0.3145 | +0.0008399 | -0.2939 | +0.2845 | +0.07571 | +0.2939 |
| `blocks.2.bn1.bias` | 128 | 0 | 0 | 0 | -0.03797 | -0.4153 | +0.1919 | +0.07995 | +0.4153 |
| `blocks.2.bn1.running_mean` | 128 | +4.53e-06 | -0.5117 | +0.625 | +0.01989 | -2.16 | +3.111 | +0.8999 | +3.111 |
| `blocks.2.bn1.running_var` | 128 | +0.9517 | +0.7188 | +1.234 | +0.2545 | +0.03357 | +1.609 | +0.2313 | +1.609 |
| `blocks.2.bn1.weight` | 128 | +1 | +1 | +1 | +0.9913 | +0.864 | +1.546 | +0.102 | +1.546 |
| `blocks.2.bn2.bias` | 128 | 0 | 0 | 0 | +0.0109 | -0.09446 | +0.0764 | +0.02868 | +0.09446 |
| `blocks.2.bn2.running_mean` | 128 | -0.00681 | -0.8867 | +0.8867 | -0.3129 | -2.012 | +0.9256 | +0.5276 | +2.012 |
| `blocks.2.bn2.running_var` | 128 | +0.4815 | +0.2988 | +0.9375 | +1.111 | +0.8253 | +1.876 | +0.171 | +1.876 |
| `blocks.2.bn2.weight` | 128 | +1 | +1 | +1 | +1.006 | +0.941 | +1.337 | +0.03937 | +1.337 |
| `blocks.2.conv1.weight` | 128x128x7x7 | -3.63e-06 | -0.0957 | +0.08447 | -0.000244 | -0.112 | +0.194 | +0.01807 | +0.194 |
| `blocks.2.conv2.weight` | 128x128x7x7 | -3.03e-06 | -0.08691 | +0.0918 | -0.0001578 | -0.2517 | +0.2158 | +0.01857 | +0.2517 |
| `blocks.2.res_ln.bias` | 128 | 0 | 0 | 0 | +0.002266 | -0.01346 | +0.03342 | +0.005296 | +0.03342 |
| `blocks.2.res_ln.weight` | 128 | +1 | +1 | +1 | +1.004 | +0.9932 | +1.033 | +0.006384 | +1.033 |
| `blocks.2.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +0.9205 | +0.9205 | +0.9205 | 0 | +0.9205 |
| `blocks.2.se_scalebias.fc1.bias` | 32 | 0 | 0 | 0 | -0.0005679 | -0.01251 | +0.02463 | +0.009193 | +0.02463 |
| `blocks.2.se_scalebias.fc1.weight` | 32x128 | -0.001246 | -0.4395 | +0.4922 | -0.0008395 | -0.3985 | +0.4477 | +0.1126 | +0.4477 |
| `blocks.2.se_scalebias.fc2.bias` | 256 | 0 | 0 | 0 | +0.002731 | -0.03343 | +0.03339 | +0.01116 | +0.03343 |
| `blocks.2.se_scalebias.fc2.bias [gamma]` | 256 | 0 | 0 | 0 | +0.005486 | -0.0258 | +0.03339 | +0.01109 | +0.03339 |
| `blocks.2.se_scalebias.fc2.bias [beta]` | 256 | 0 | 0 | 0 | -2.36e-05 | -0.03343 | +0.02784 | +0.01054 | +0.03343 |
| `blocks.2.se_scalebias.fc2.weight` | 256x32 | -2.902e-05 | -0.3496 | +0.332 | +0.002453 | -0.3173 | +0.3256 | +0.07576 | +0.3256 |
| `blocks.2.se_scalebias.fc2.weight [gamma]` | 256x32 | +0.001102 | -0.3496 | +0.332 | +0.005976 | -0.3173 | +0.3256 | +0.07685 | +0.3256 |
| `blocks.2.se_scalebias.fc2.weight [beta]` | 256x32 | -0.00116 | -0.2793 | +0.3203 | -0.001069 | -0.2521 | +0.2907 | +0.07449 | +0.2907 |
| `policy.conv.bias` | 76 | 0 | 0 | 0 | +6.034e-05 | -0.07224 | +0.5578 | +0.07909 | +0.5578 |
| `policy.conv.weight` | 76x128x1x1 | -0.001422 | -0.4648 | +0.4785 | +0.008265 | -0.7984 | +0.8049 | +0.1474 | +0.8049 |
| `policy.pre_bn.bias` | 128 | 0 | 0 | 0 | +0.0414 | -0.09472 | +0.2286 | +0.06376 | +0.2286 |
| `policy.pre_bn.running_mean` | 128 | -0.02798 | -1.469 | +1.547 | -0.1322 | -1.454 | +1.11 | +0.5633 | +1.454 |
| `policy.pre_bn.running_var` | 128 | +0.679 | +0.3496 | +1.531 | +0.9071 | +0.4291 | +2.466 | +0.323 | +2.466 |
| `policy.pre_bn.weight` | 128 | +1 | +1 | +1 | +1.078 | +0.8066 | +1.392 | +0.09868 | +1.392 |
| `policy.pre_conv.weight` | 128x128x1x1 | -0.0005189 | -0.5039 | +0.459 | -0.001816 | -0.5662 | +0.5092 | +0.1213 | +0.5662 |
| `stem.bn.bias` | 128 | 0 | 0 | 0 | -0.004569 | -0.1301 | +0.08476 | +0.03775 | +0.1301 |
| `stem.bn.running_mean` | 128 | +0.001108 | -0.2715 | +0.2695 | -0.04204 | -1.003 | +0.4043 | +0.1822 | +1.003 |
| `stem.bn.running_var` | 128 | +0.05419 | +0.02075 | +0.1416 | +0.0999 | +0.02436 | +2.57 | +0.2239 | +2.57 |
| `stem.bn.weight` | 128 | +1 | +1 | +1 | +0.9855 | +0.6962 | +1.322 | +0.1019 | +1.322 |
| `stem.conv.weight` | 128x30x7x7 | -2.108e-05 | -0.1602 | +0.1768 | -5.637e-05 | -0.644 | +1.266 | +0.04157 | +1.266 |
| `tower_final_bn.bias` | 128 | 0 | 0 | 0 | +0.1003 | -0.1875 | +0.3349 | +0.0839 | +0.3349 |
| `tower_final_bn.running_mean` | 128 | +4.53e-06 | -0.4336 | +0.5625 | +0.003265 | -2.129 | +2.148 | +0.846 | +2.148 |
| `tower_final_bn.running_var` | 128 | +0.9525 | +0.6836 | +1.227 | +0.2925 | +0.05377 | +2.162 | +0.2679 | +2.162 |
| `tower_final_bn.weight` | 128 | +1 | +1 | +1 | +0.9938 | +0.8208 | +1.426 | +0.09415 | +1.426 |
| `value.bn.bias` | 16 | 0 | 0 | 0 | -0.1292 | -0.1924 | +0.03695 | +0.05316 | +0.1924 |
| `value.bn.running_mean` | 16 | +0.1705 | -1.078 | +1.211 | +0.007298 | -1.168 | +1.24 | +0.6765 | +1.24 |
| `value.bn.running_var` | 16 | +0.7998 | +0.3867 | +1.195 | +1.079 | +0.5778 | +1.833 | +0.2856 | +1.833 |
| `value.bn.weight` | 16 | +1 | +1 | +1 | +0.7991 | +0.7405 | +1.056 | +0.07308 | +1.056 |
| `value.conv.weight` | 16x128x1x1 | +0.00335 | -0.4434 | +0.4824 | +0.0004049 | -0.4075 | +0.4895 | +0.1169 | +0.4895 |
| `value.fc1.bias` | 128 | 0 | 0 | 0 | -0.001394 | -0.01882 | +0.03015 | +0.009257 | +0.03015 |
| `value.fc1.weight` | 128x1024 | +9.948e-05 | -0.1846 | +0.2109 | -0.001261 | -0.1841 | +0.1923 | +0.04016 | +0.1923 |
| `value.wdl_fc2.bias` | 3 | +0.5964 | 0 | +1.789 | +0.596 | +0.296 | +1.189 | +0.4196 | +1.189 |
| `value.wdl_fc2.weight` | 3x128 | +0.00532 | -0.2793 | +0.3027 | +0.004869 | -0.2551 | +0.3101 | +0.1017 | +0.3101 |

### `se_att2` — attenuate-only, seed 2 (step 0 → 7,289)

| tensor | shape | fresh mean | fresh min | fresh max | final mean | final min | final max | final std | final abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `blocks.0.bn1.bias` | 128 | 0 | 0 | 0 | -0.04277 | -0.3767 | +0.1214 | +0.05682 | +0.3767 |
| `blocks.0.bn1.running_mean` | 128 | +6.71e-05 | -0.003098 | +0.001816 | -0.009909 | -0.1983 | +0.1415 | +0.06587 | +0.1983 |
| `blocks.0.bn1.running_var` | 128 | +1 | +0.9922 | +1.008 | +0.9564 | +0.1208 | +1.889 | +0.205 | +1.889 |
| `blocks.0.bn1.weight` | 128 | +1 | +1 | +1 | +0.9936 | +0.8719 | +1.515 | +0.09021 | +1.515 |
| `blocks.0.bn2.bias` | 128 | 0 | 0 | 0 | -0.008259 | -0.4293 | +0.04208 | +0.04567 | +0.4293 |
| `blocks.0.bn2.running_mean` | 128 | +0.0654 | -0.7422 | +0.9922 | -0.5882 | -2.593 | +0.7773 | +0.5381 | +2.593 |
| `blocks.0.bn2.running_var` | 128 | +0.5113 | +0.3242 | +0.9297 | +0.682 | +0.4325 | +1.341 | +0.1667 | +1.341 |
| `blocks.0.bn2.weight` | 128 | +1 | +1 | +1 | +1.012 | +0.9568 | +1.406 | +0.04717 | +1.406 |
| `blocks.0.conv1.weight` | 128x128x7x7 | +3.417e-05 | -0.08447 | +0.08594 | -0.000379 | -0.1755 | +0.3091 | +0.01741 | +0.3091 |
| `blocks.0.conv2.weight` | 128x128x7x7 | -2.21e-05 | -0.08447 | +0.07812 | -8.586e-05 | -0.3184 | +0.2642 | +0.01737 | +0.3184 |
| `blocks.0.res_ln.bias` | 128 | 0 | 0 | 0 | -0.01062 | -0.1257 | +0.09208 | +0.04346 | +0.1257 |
| `blocks.0.res_ln.weight` | 128 | +1 | +1 | +1 | +0.9822 | +0.7947 | +1.383 | +0.08767 | +1.383 |
| `blocks.0.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +1.009 | +1.009 | +1.009 | 0 | +1.009 |
| `blocks.0.se_attenuate.fc1.bias` | 32 | 0 | 0 | 0 | +0.0005168 | -0.006236 | +0.0178 | +0.004832 | +0.0178 |
| `blocks.0.se_attenuate.fc1.weight` | 32x128 | +0.0001424 | -0.5039 | +0.3945 | +0.0001414 | -0.4575 | +0.3759 | +0.1128 | +0.4575 |
| `blocks.0.se_attenuate.fc2.bias` | 128 | 0 | 0 | 0 | +0.005388 | -0.01125 | +0.02135 | +0.005144 | +0.02135 |
| `blocks.0.se_attenuate.fc2.weight` | 128x32 | +0.0009253 | -0.3789 | +0.5156 | +0.005949 | -0.3289 | +0.5136 | +0.1022 | +0.5136 |
| `blocks.1.bn1.bias` | 128 | 0 | 0 | 0 | -0.04274 | -0.2521 | +0.07864 | +0.04642 | +0.2521 |
| `blocks.1.bn1.running_mean` | 128 | +2.027e-06 | -0.2656 | +0.2676 | +0.009508 | -2.496 | +2.999 | +0.8168 | +2.999 |
| `blocks.1.bn1.running_var` | 128 | +0.9812 | +0.7344 | +1.273 | +0.4715 | +0.04462 | +5.866 | +0.5569 | +5.866 |
| `blocks.1.bn1.weight` | 128 | +1 | +1 | +1 | +0.9936 | +0.8635 | +1.48 | +0.09643 | +1.48 |
| `blocks.1.bn2.bias` | 128 | 0 | 0 | 0 | -0.01527 | -0.08837 | +0.03924 | +0.02119 | +0.08837 |
| `blocks.1.bn2.running_mean` | 128 | -0.01879 | -0.9766 | +0.6367 | -0.336 | -1.291 | +1.017 | +0.4225 | +1.291 |
| `blocks.1.bn2.running_var` | 128 | +0.4659 | +0.3301 | +0.7852 | +0.6685 | +0.4745 | +0.8966 | +0.08165 | +0.8966 |
| `blocks.1.bn2.weight` | 128 | +1 | +1 | +1 | +1.01 | +0.963 | +1.161 | +0.03138 | +1.161 |
| `blocks.1.conv1.weight` | 128x128x7x7 | -5.885e-06 | -0.08105 | +0.08301 | -0.000234 | -0.1806 | +0.1625 | +0.01755 | +0.1806 |
| `blocks.1.conv2.weight` | 128x128x7x7 | -2.394e-05 | -0.09619 | +0.09375 | -0.0001801 | -0.1582 | +0.1433 | +0.01753 | +0.1582 |
| `blocks.1.res_ln.bias` | 128 | 0 | 0 | 0 | -0.00685 | -0.1537 | +0.08965 | +0.04114 | +0.1537 |
| `blocks.1.res_ln.weight` | 128 | +1 | +1 | +1 | +0.9849 | +0.7301 | +1.538 | +0.1053 | +1.538 |
| `blocks.1.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +0.953 | +0.953 | +0.953 | 0 | +0.953 |
| `blocks.1.se_attenuate.fc1.bias` | 32 | 0 | 0 | 0 | -0.0007824 | -0.006532 | +0.01431 | +0.004275 | +0.01431 |
| `blocks.1.se_attenuate.fc1.weight` | 32x128 | +0.00393 | -0.5039 | +0.5469 | +0.003678 | -0.4573 | +0.4924 | +0.1155 | +0.4924 |
| `blocks.1.se_attenuate.fc2.bias` | 128 | 0 | 0 | 0 | +0.004497 | -0.02411 | +0.03019 | +0.00968 | +0.03019 |
| `blocks.1.se_attenuate.fc2.weight` | 128x32 | +0.002487 | -0.3691 | +0.4043 | +0.005172 | -0.3447 | +0.3667 | +0.1016 | +0.3667 |
| `blocks.2.bn1.bias` | 128 | 0 | 0 | 0 | -0.03606 | -0.2176 | +0.1158 | +0.05168 | +0.2176 |
| `blocks.2.bn1.running_mean` | 128 | +4.196e-05 | -0.3145 | +0.3242 | +0.0149 | -2.267 | +3.436 | +0.8365 | +3.436 |
| `blocks.2.bn1.running_var` | 128 | +0.9776 | +0.7266 | +1.25 | +0.4752 | +0.02062 | +10.77 | +0.9783 | +10.77 |
| `blocks.2.bn1.weight` | 128 | +1 | +1 | +1 | +0.9933 | +0.8534 | +1.549 | +0.1003 | +1.549 |
| `blocks.2.bn2.bias` | 128 | 0 | 0 | 0 | +0.009626 | -0.1566 | +0.07661 | +0.02948 | +0.1566 |
| `blocks.2.bn2.running_mean` | 128 | +0.05141 | -1.094 | +0.957 | -0.2708 | -1.485 | +0.8177 | +0.5205 | +1.485 |
| `blocks.2.bn2.running_var` | 128 | +0.4833 | +0.3281 | +0.9766 | +1.043 | +0.6334 | +1.547 | +0.1573 | +1.547 |
| `blocks.2.bn2.weight` | 128 | +1 | +1 | +1 | +1.008 | +0.9369 | +1.186 | +0.0381 | +1.186 |
| `blocks.2.conv1.weight` | 128x128x7x7 | +3.16e-05 | -0.08301 | +0.08643 | -0.000194 | -0.1925 | +0.1645 | +0.01792 | +0.1925 |
| `blocks.2.conv2.weight` | 128x128x7x7 | -3.153e-05 | -0.08545 | +0.08594 | -4.662e-05 | -0.1186 | +0.1378 | +0.01835 | +0.1378 |
| `blocks.2.res_ln.bias` | 128 | 0 | 0 | 0 | +0.002055 | -0.01058 | +0.01442 | +0.003781 | +0.01442 |
| `blocks.2.res_ln.weight` | 128 | +1 | +1 | +1 | +1.003 | +0.9936 | +1.021 | +0.004324 | +1.021 |
| `blocks.2.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +0.9644 | +0.9644 | +0.9644 | 0 | +0.9644 |
| `blocks.2.se_attenuate.fc1.bias` | 32 | 0 | 0 | 0 | -0.0006534 | -0.01078 | +0.01319 | +0.005524 | +0.01319 |
| `blocks.2.se_attenuate.fc1.weight` | 32x128 | -0.002469 | -0.4531 | +0.4453 | -0.002164 | -0.4068 | +0.3957 | +0.1126 | +0.4068 |
| `blocks.2.se_attenuate.fc2.bias` | 128 | 0 | 0 | 0 | +0.004173 | -0.03909 | +0.02591 | +0.009903 | +0.03909 |
| `blocks.2.se_attenuate.fc2.weight` | 128x32 | +0.0009354 | -0.3711 | +0.4199 | +0.005611 | -0.334 | +0.3877 | +0.1 | +0.3877 |
| `policy.conv.bias` | 76 | 0 | 0 | 0 | +3.964e-05 | -0.07132 | +0.4741 | +0.07368 | +0.4741 |
| `policy.conv.weight` | 76x128x1x1 | +0.001255 | -0.4727 | +0.4941 | +0.01093 | -0.8255 | +0.7928 | +0.1468 | +0.8255 |
| `policy.pre_bn.bias` | 128 | 0 | 0 | 0 | +0.0349 | -0.1333 | +0.2018 | +0.06292 | +0.2018 |
| `policy.pre_bn.running_mean` | 128 | -0.02069 | -1.484 | +1.516 | -0.1799 | -1.693 | +1.188 | +0.6268 | +1.693 |
| `policy.pre_bn.running_var` | 128 | +0.6771 | +0.3516 | +1.422 | +0.9091 | +0.4293 | +2.535 | +0.3158 | +2.535 |
| `policy.pre_bn.weight` | 128 | +1 | +1 | +1 | +1.079 | +0.8648 | +1.388 | +0.09588 | +1.388 |
| `policy.pre_conv.weight` | 128x128x1x1 | -0.0004165 | -0.4863 | +0.543 | -0.001878 | -0.7017 | +0.5607 | +0.1215 | +0.7017 |
| `stem.bn.bias` | 128 | 0 | 0 | 0 | -0.009898 | -0.1978 | +0.1416 | +0.0659 | +0.1978 |
| `stem.bn.running_mean` | 128 | -0.004377 | -0.252 | +0.2734 | -0.0218 | -0.3463 | +0.3729 | +0.1762 | +0.3729 |
| `stem.bn.running_var` | 128 | +0.0526 | +0.02686 | +0.1582 | +0.09026 | +0.03329 | +0.3068 | +0.05006 | +0.3068 |
| `stem.bn.weight` | 128 | +1 | +1 | +1 | +0.9722 | +0.3461 | +1.378 | +0.1088 | +1.378 |
| `stem.conv.weight` | 128x30x7x7 | -6.699e-05 | -0.1514 | +0.1641 | +3.698e-05 | -0.6191 | +1.184 | +0.04132 | +1.184 |
| `tower_final_bn.bias` | 128 | 0 | 0 | 0 | +0.09626 | -0.1051 | +0.4394 | +0.08817 | +0.4394 |
| `tower_final_bn.running_mean` | 128 | +1.121e-05 | -0.3887 | +0.3301 | +0.002631 | -1.865 | +2.568 | +0.7145 | +2.568 |
| `tower_final_bn.running_var` | 128 | +0.9749 | +0.7539 | +1.266 | +0.4957 | +0.06708 | +8.127 | +0.738 | +8.127 |
| `tower_final_bn.weight` | 128 | +1 | +1 | +1 | +0.9942 | +0.8108 | +1.668 | +0.109 | +1.668 |
| `value.bn.bias` | 16 | 0 | 0 | 0 | -0.1554 | -0.2573 | -0.09145 | +0.04271 | +0.2573 |
| `value.bn.running_mean` | 16 | +0.01633 | -0.9727 | +1.102 | -0.178 | -1.137 | +0.7132 | +0.5117 | +1.137 |
| `value.bn.running_var` | 16 | +0.7312 | +0.3438 | +1.148 | +1.038 | +0.6604 | +1.954 | +0.3611 | +1.954 |
| `value.bn.weight` | 16 | +1 | +1 | +1 | +0.79 | +0.6953 | +0.9431 | +0.06469 | +0.9431 |
| `value.conv.weight` | 16x128x1x1 | +0.0003021 | -0.3926 | +0.4395 | -0.003292 | -0.4113 | +0.4485 | +0.1178 | +0.4485 |
| `value.fc1.bias` | 128 | 0 | 0 | 0 | -0.002085 | -0.02098 | +0.01843 | +0.0079 | +0.02098 |
| `value.fc1.weight` | 128x1024 | +7.058e-05 | -0.1885 | +0.1768 | -0.001306 | -0.1748 | +0.1663 | +0.0401 | +0.1748 |
| `value.wdl_fc2.bias` | 3 | +0.5964 | 0 | +1.789 | +0.596 | +0.2436 | +1.236 | +0.4531 | +1.236 |
| `value.wdl_fc2.weight` | 3x128 | +4.868e-05 | -0.3008 | +0.3594 | +3.297e-05 | -0.3163 | +0.2855 | +0.1099 | +0.3163 |

### `se_none2` — none, seed 2 (step 0 → 7,019)

| tensor | shape | fresh mean | fresh min | fresh max | final mean | final min | final max | final std | final abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `blocks.0.bn1.bias` | 128 | 0 | 0 | 0 | -0.03332 | -0.2831 | +0.1089 | +0.04959 | +0.2831 |
| `blocks.0.bn1.running_mean` | 128 | -0.0001066 | -0.003235 | +0.001953 | -0.007135 | -0.1407 | +0.1136 | +0.0473 | +0.1407 |
| `blocks.0.bn1.running_var` | 128 | +0.9992 | +0.9961 | +1 | +0.9863 | +0.2839 | +1.717 | +0.1819 | +1.717 |
| `blocks.0.bn1.weight` | 128 | +1 | +1 | +1 | +0.9929 | +0.8616 | +1.774 | +0.1048 | +1.774 |
| `blocks.0.bn2.bias` | 128 | 0 | 0 | 0 | -0.002804 | -0.1749 | +0.04011 | +0.024 | +0.1749 |
| `blocks.0.bn2.running_mean` | 128 | -0.001006 | -0.9297 | +1.008 | -0.6676 | -2.17 | +1.065 | +0.5881 | +2.17 |
| `blocks.0.bn2.running_var` | 128 | +0.5075 | +0.3203 | +1.188 | +0.8011 | +0.4855 | +1.547 | +0.1864 | +1.547 |
| `blocks.0.bn2.weight` | 128 | +1 | +1 | +1 | +1.003 | +0.9634 | +1.162 | +0.02596 | +1.162 |
| `blocks.0.conv1.weight` | 128x128x7x7 | +1.082e-05 | -0.08252 | +0.08984 | -0.0003989 | -0.1427 | +0.2621 | +0.01748 | +0.2621 |
| `blocks.0.conv2.weight` | 128x128x7x7 | +1.173e-05 | -0.08154 | +0.08301 | +1.505e-05 | -0.1885 | +0.1576 | +0.01737 | +0.1885 |
| `blocks.0.res_ln.bias` | 128 | 0 | 0 | 0 | -0.01261 | -0.1081 | +0.1125 | +0.03565 | +0.1125 |
| `blocks.0.res_ln.weight` | 128 | +1 | +1 | +1 | +0.9948 | +0.8037 | +1.22 | +0.0772 | +1.22 |
| `blocks.0.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +0.7791 | +0.7791 | +0.7791 | 0 | +0.7791 |
| `blocks.1.bn1.bias` | 128 | 0 | 0 | 0 | -0.04269 | -0.3581 | +0.04162 | +0.05524 | +0.3581 |
| `blocks.1.bn1.running_mean` | 128 | -2.824e-06 | -0.3105 | +0.3398 | +0.004279 | -1.841 | +3.221 | +0.8389 | +3.221 |
| `blocks.1.bn1.running_var` | 128 | +0.98 | +0.7383 | +1.297 | +0.4192 | +0.02741 | +1.464 | +0.2705 | +1.464 |
| `blocks.1.bn1.weight` | 128 | +1 | +1 | +1 | +0.9928 | +0.8838 | +1.703 | +0.09985 | +1.703 |
| `blocks.1.bn2.bias` | 128 | 0 | 0 | 0 | -0.01925 | -0.1638 | +0.02789 | +0.02633 | +0.1638 |
| `blocks.1.bn2.running_mean` | 128 | -0.01181 | -0.9688 | +0.8945 | -0.2818 | -1.56 | +1.127 | +0.3798 | +1.56 |
| `blocks.1.bn2.running_var` | 128 | +0.4815 | +0.3027 | +0.7266 | +0.7513 | +0.5144 | +1.091 | +0.1073 | +1.091 |
| `blocks.1.bn2.weight` | 128 | +1 | +1 | +1 | +0.9992 | +0.9458 | +1.195 | +0.03201 | +1.195 |
| `blocks.1.conv1.weight` | 128x128x7x7 | -1.002e-05 | -0.09033 | +0.08057 | -0.0001981 | -0.1798 | +0.178 | +0.01761 | +0.1798 |
| `blocks.1.conv2.weight` | 128x128x7x7 | -3.462e-06 | -0.08984 | +0.08203 | +1.485e-06 | -0.2436 | +0.1962 | +0.01762 | +0.2436 |
| `blocks.1.res_ln.bias` | 128 | 0 | 0 | 0 | -0.009653 | -0.09542 | +0.08602 | +0.03368 | +0.09542 |
| `blocks.1.res_ln.weight` | 128 | +1 | +1 | +1 | +0.9965 | +0.7927 | +1.287 | +0.09358 | +1.287 |
| `blocks.1.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +0.56 | +0.56 | +0.56 | 0 | +0.56 |
| `blocks.2.bn1.bias` | 128 | 0 | 0 | 0 | -0.04229 | -0.4833 | +0.1768 | +0.07191 | +0.4833 |
| `blocks.2.bn1.running_mean` | 128 | -3.35e-05 | -0.3887 | +0.4414 | +0.01832 | -1.779 | +3.664 | +0.8808 | +3.664 |
| `blocks.2.bn1.running_var` | 128 | +0.9629 | +0.6602 | +1.234 | +0.3832 | +0.06524 | +1.507 | +0.296 | +1.507 |
| `blocks.2.bn1.weight` | 128 | +1 | +1 | +1 | +0.9907 | +0.8859 | +1.658 | +0.1119 | +1.658 |
| `blocks.2.bn2.bias` | 128 | 0 | 0 | 0 | +0.01148 | -0.2655 | +0.07925 | +0.03966 | +0.2655 |
| `blocks.2.bn2.running_mean` | 128 | +0.015 | -0.7891 | +1.203 | -0.3287 | -1.898 | +1.061 | +0.6186 | +1.898 |
| `blocks.2.bn2.running_var` | 128 | +0.4655 | +0.3281 | +0.7539 | +1.204 | +0.7502 | +1.864 | +0.2097 | +1.864 |
| `blocks.2.bn2.weight` | 128 | +1 | +1 | +1 | +1.001 | +0.9342 | +1.436 | +0.05784 | +1.436 |
| `blocks.2.conv1.weight` | 128x128x7x7 | +9.033e-06 | -0.08984 | +0.08154 | -0.0002664 | -0.2403 | +0.3765 | +0.0181 | +0.3765 |
| `blocks.2.conv2.weight` | 128x128x7x7 | +1.078e-05 | -0.09326 | +0.08105 | +3.509e-05 | -0.283 | +0.1542 | +0.01875 | +0.283 |
| `blocks.2.res_ln.bias` | 128 | 0 | 0 | 0 | +0.002396 | -0.01305 | +0.01469 | +0.004574 | +0.01469 |
| `blocks.2.res_ln.weight` | 128 | +1 | +1 | +1 | +1.005 | +0.9931 | +1.05 | +0.008351 | +1.05 |
| `blocks.2.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +0.7189 | +0.7189 | +0.7189 | 0 | +0.7189 |
| `policy.conv.bias` | 76 | 0 | 0 | 0 | +5.722e-05 | -0.06526 | +0.4385 | +0.0662 | +0.4385 |
| `policy.conv.weight` | 76x128x1x1 | +0.0001894 | -0.4727 | +0.4629 | +0.01391 | -0.8519 | +0.7063 | +0.1452 | +0.8519 |
| `policy.pre_bn.bias` | 128 | 0 | 0 | 0 | +0.02118 | -0.08689 | +0.2069 | +0.05202 | +0.2069 |
| `policy.pre_bn.running_mean` | 128 | -0.00795 | -1.438 | +1.414 | -0.2577 | -1.868 | +1.366 | +0.6307 | +1.868 |
| `policy.pre_bn.running_var` | 128 | +0.669 | +0.3574 | +1.148 | +0.8198 | +0.3245 | +1.647 | +0.2694 | +1.647 |
| `policy.pre_bn.weight` | 128 | +1 | +1 | +1 | +1.067 | +0.8758 | +1.317 | +0.08217 | +1.317 |
| `policy.pre_conv.weight` | 128x128x1x1 | -0.0001515 | -0.5977 | +0.5117 | -0.003103 | -0.6048 | +0.5358 | +0.1202 | +0.6048 |
| `stem.bn.bias` | 128 | 0 | 0 | 0 | -0.007121 | -0.1403 | +0.1139 | +0.04732 | +0.1403 |
| `stem.bn.running_mean` | 128 | +0.003804 | -0.4609 | +0.4922 | -0.009345 | -0.2676 | +0.3345 | +0.1588 | +0.3345 |
| `stem.bn.running_var` | 128 | +0.06548 | +0.03076 | +0.3164 | +0.07433 | +0.02348 | +0.236 | +0.03533 | +0.236 |
| `stem.bn.weight` | 128 | +1 | +1 | +1 | +0.9888 | +0.5325 | +1.31 | +0.09254 | +1.31 |
| `stem.conv.weight` | 128x30x7x7 | +5.885e-05 | -0.167 | +0.1758 | +0.0002154 | -0.4875 | +1.398 | +0.04036 | +1.398 |
| `tower_final_bn.bias` | 128 | 0 | 0 | 0 | +0.09362 | -0.1098 | +0.4739 | +0.07637 | +0.4739 |
| `tower_final_bn.running_mean` | 128 | +2.751e-05 | -0.4688 | +0.5078 | +0.003918 | -1.39 | +2.414 | +0.7299 | +2.414 |
| `tower_final_bn.running_var` | 128 | +0.9552 | +0.5781 | +1.305 | +0.4811 | +0.1114 | +1.947 | +0.3039 | +1.947 |
| `tower_final_bn.weight` | 128 | +1 | +1 | +1 | +0.9936 | +0.8402 | +1.471 | +0.1006 | +1.471 |
| `value.bn.bias` | 16 | 0 | 0 | 0 | -0.1537 | -0.2241 | -0.0684 | +0.04199 | +0.2241 |
| `value.bn.running_mean` | 16 | -0.242 | -1.211 | +0.543 | -0.3896 | -1.409 | +0.6243 | +0.5815 | +1.409 |
| `value.bn.running_var` | 16 | +0.7246 | +0.4219 | +1.109 | +1.137 | +0.6423 | +2.156 | +0.4535 | +2.156 |
| `value.bn.weight` | 16 | +1 | +1 | +1 | +0.792 | +0.7127 | +1.106 | +0.08605 | +1.106 |
| `value.conv.weight` | 16x128x1x1 | -0.004641 | -0.4746 | +0.375 | -0.007186 | -0.4494 | +0.3791 | +0.1153 | +0.4494 |
| `value.fc1.bias` | 128 | 0 | 0 | 0 | -0.002205 | -0.01645 | +0.02886 | +0.007245 | +0.02886 |
| `value.fc1.weight` | 128x1024 | -4.258e-05 | -0.1885 | +0.21 | -0.001266 | -0.1716 | +0.1974 | +0.04021 | +0.1974 |
| `value.wdl_fc2.bias` | 3 | +0.5964 | 0 | +1.789 | +0.5969 | +0.2764 | +1.227 | +0.4458 | +1.227 |
| `value.wdl_fc2.weight` | 3x128 | -0.009427 | -0.4648 | +0.3281 | -0.008551 | -0.3613 | +0.2832 | +0.1073 | +0.3613 |

### `se_zb2` — zero-beta scale+bias, seed 2 (step 0 → 5,004)

| tensor | shape | fresh mean | fresh min | fresh max | final mean | final min | final max | final std | final abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `blocks.0.bn1.bias` | 128 | 0 | 0 | 0 | -0.0282 | -0.0933 | +0.2138 | +0.05052 | +0.2138 |
| `blocks.0.bn1.running_mean` | 128 | +1.776e-05 | -0.002579 | +0.002045 | -0.006845 | -0.1677 | +0.09938 | +0.04885 | +0.1677 |
| `blocks.0.bn1.running_var` | 128 | +0.9995 | +0.9922 | +1.008 | +0.9693 | +0.4028 | +1.91 | +0.203 | +1.91 |
| `blocks.0.bn1.weight` | 128 | +1 | +1 | +1 | +0.9952 | +0.8636 | +1.333 | +0.08172 | +1.333 |
| `blocks.0.bn2.bias` | 128 | 0 | 0 | 0 | -0.01091 | -0.2171 | +0.0226 | +0.02472 | +0.2171 |
| `blocks.0.bn2.running_mean` | 128 | +0.019 | -0.7422 | +0.8828 | -0.5807 | -1.934 | +0.8911 | +0.5383 | +1.934 |
| `blocks.0.bn2.running_var` | 128 | +0.5308 | +0.2656 | +0.9688 | +0.7601 | +0.4929 | +2.596 | +0.2422 | +2.596 |
| `blocks.0.bn2.weight` | 128 | +1 | +1 | +1 | +1.01 | +0.9709 | +1.177 | +0.02832 | +1.177 |
| `blocks.0.conv1.weight` | 128x128x7x7 | +1.881e-05 | -0.0835 | +0.09375 | -0.0003411 | -0.1103 | +0.2079 | +0.0175 | +0.2079 |
| `blocks.0.conv2.weight` | 128x128x7x7 | -1.854e-05 | -0.08301 | +0.09229 | -3.187e-05 | -0.1769 | +0.2082 | +0.01737 | +0.2082 |
| `blocks.0.res_ln.bias` | 128 | 0 | 0 | 0 | -0.01181 | -0.1921 | +0.05661 | +0.03708 | +0.1921 |
| `blocks.0.res_ln.weight` | 128 | +1 | +1 | +1 | +0.9869 | +0.7376 | +1.247 | +0.08033 | +1.247 |
| `blocks.0.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +0.9757 | +0.9757 | +0.9757 | 0 | +0.9757 |
| `blocks.0.se_scalebias.fc1.bias` | 32 | 0 | 0 | 0 | +0.003513 | -0.006106 | +0.02844 | +0.008548 | +0.02844 |
| `blocks.0.se_scalebias.fc1.weight` | 32x128 | +0.005474 | -0.4375 | +0.4746 | +0.005006 | -0.3997 | +0.4375 | +0.1123 | +0.4375 |
| `blocks.0.se_scalebias.fc2.bias` | 256 | 0 | 0 | 0 | +0.002871 | -0.07903 | +0.04377 | +0.01528 | +0.07903 |
| `blocks.0.se_scalebias.fc2.bias [gamma]` | 256 | 0 | 0 | 0 | +0.005747 | -0.01501 | +0.0279 | +0.005866 | +0.0279 |
| `blocks.0.se_scalebias.fc2.bias [beta]` | 256 | 0 | 0 | 0 | -5.581e-06 | -0.07903 | +0.04377 | +0.02039 | +0.07903 |
| `blocks.0.se_scalebias.fc2.weight` | 256x32 | +5.072e-05 | -0.2598 | +0.3066 | +0.002249 | -0.2222 | +0.3077 | +0.0549 | +0.3077 |
| `blocks.0.se_scalebias.fc2.weight [gamma]` | 256x32 | +0.0001014 | -0.2598 | +0.3066 | +0.004507 | -0.2222 | +0.3077 | +0.07501 | +0.3077 |
| `blocks.0.se_scalebias.fc2.weight [beta]` | 256x32 | 0 | 0 | 0 | -8.708e-06 | -0.1361 | +0.09841 | +0.01978 | +0.1361 |
| `blocks.1.bn1.bias` | 128 | 0 | 0 | 0 | -0.04073 | -0.3207 | +0.1594 | +0.05799 | +0.3207 |
| `blocks.1.bn1.running_mean` | 128 | -1.168e-05 | -0.3516 | +0.4219 | +0.00904 | -1.811 | +2.345 | +0.8364 | +2.345 |
| `blocks.1.bn1.running_var` | 128 | +0.969 | +0.707 | +1.25 | +0.3763 | +0.02127 | +2.127 | +0.308 | +2.127 |
| `blocks.1.bn1.weight` | 128 | +1 | +1 | +1 | +0.9932 | +0.8645 | +1.438 | +0.09544 | +1.438 |
| `blocks.1.bn2.bias` | 128 | 0 | 0 | 0 | -0.02207 | -0.2706 | +0.03524 | +0.03165 | +0.2706 |
| `blocks.1.bn2.running_mean` | 128 | -0.03469 | -0.9141 | +0.9648 | -0.4377 | -1.394 | +0.7007 | +0.4016 | +1.394 |
| `blocks.1.bn2.running_var` | 128 | +0.4828 | +0.2598 | +0.7656 | +0.7479 | +0.4382 | +1.362 | +0.1211 | +1.362 |
| `blocks.1.bn2.weight` | 128 | +1 | +1 | +1 | +1.01 | +0.9529 | +1.277 | +0.03919 | +1.277 |
| `blocks.1.conv1.weight` | 128x128x7x7 | -1.644e-05 | -0.08984 | +0.08691 | -0.0003075 | -0.1609 | +0.2624 | +0.01771 | +0.2624 |
| `blocks.1.conv2.weight` | 128x128x7x7 | -2.565e-05 | -0.07861 | +0.09033 | -2.796e-05 | -0.1542 | +0.2336 | +0.01772 | +0.2336 |
| `blocks.1.res_ln.bias` | 128 | 0 | 0 | 0 | -0.007106 | -0.1122 | +0.05552 | +0.02829 | +0.1122 |
| `blocks.1.res_ln.weight` | 128 | +1 | +1 | +1 | +0.9893 | +0.7029 | +1.254 | +0.09365 | +1.254 |
| `blocks.1.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +0.9384 | +0.9384 | +0.9384 | 0 | +0.9384 |
| `blocks.1.se_scalebias.fc1.bias` | 32 | 0 | 0 | 0 | +0.000805 | -0.004677 | +0.01355 | +0.003987 | +0.01355 |
| `blocks.1.se_scalebias.fc1.weight` | 32x128 | +0.0007602 | -0.4141 | +0.4629 | +0.0007353 | -0.3724 | +0.4229 | +0.1133 | +0.4229 |
| `blocks.1.se_scalebias.fc2.bias` | 256 | 0 | 0 | 0 | +0.001793 | -0.04588 | +0.02532 | +0.009383 | +0.04588 |
| `blocks.1.se_scalebias.fc2.bias [gamma]` | 256 | 0 | 0 | 0 | +0.00359 | -0.03643 | +0.02118 | +0.007354 | +0.03643 |
| `blocks.1.se_scalebias.fc2.bias [beta]` | 256 | 0 | 0 | 0 | -3.484e-06 | -0.04588 | +0.02532 | +0.01075 | +0.04588 |
| `blocks.1.se_scalebias.fc2.weight` | 256x32 | +0.0009837 | -0.2832 | +0.2891 | +0.002709 | -0.2437 | +0.2733 | +0.05477 | +0.2733 |
| `blocks.1.se_scalebias.fc2.weight [gamma]` | 256x32 | +0.001967 | -0.2832 | +0.2891 | +0.005424 | -0.2437 | +0.2733 | +0.07614 | +0.2733 |
| `blocks.1.se_scalebias.fc2.weight [beta]` | 256x32 | 0 | 0 | 0 | -6.683e-06 | -0.0994 | +0.1105 | +0.01366 | +0.1105 |
| `blocks.2.bn1.bias` | 128 | 0 | 0 | 0 | -0.02921 | -0.2815 | +0.1821 | +0.06339 | +0.2815 |
| `blocks.2.bn1.running_mean` | 128 | +4.53e-06 | -0.5117 | +0.625 | +0.01503 | -2.25 | +2.499 | +0.8787 | +2.499 |
| `blocks.2.bn1.running_var` | 128 | +0.9517 | +0.7188 | +1.234 | +0.303 | +0.04323 | +2.259 | +0.3254 | +2.259 |
| `blocks.2.bn1.weight` | 128 | +1 | +1 | +1 | +0.9934 | +0.8611 | +1.431 | +0.09385 | +1.431 |
| `blocks.2.bn2.bias` | 128 | 0 | 0 | 0 | +0.002879 | -0.1951 | +0.05828 | +0.02897 | +0.1951 |
| `blocks.2.bn2.running_mean` | 128 | -0.00681 | -0.8867 | +0.8867 | -0.3245 | -2.048 | +1.139 | +0.5137 | +2.048 |
| `blocks.2.bn2.running_var` | 128 | +0.4815 | +0.2988 | +0.9375 | +1.046 | +0.7266 | +1.568 | +0.1263 | +1.568 |
| `blocks.2.bn2.weight` | 128 | +1 | +1 | +1 | +1.007 | +0.95 | +1.203 | +0.03312 | +1.203 |
| `blocks.2.conv1.weight` | 128x128x7x7 | -3.63e-06 | -0.0957 | +0.08447 | -0.0002426 | -0.1314 | +0.1661 | +0.01796 | +0.1661 |
| `blocks.2.conv2.weight` | 128x128x7x7 | -3.03e-06 | -0.08691 | +0.0918 | -0.0002008 | -0.1979 | +0.1691 | +0.01838 | +0.1979 |
| `blocks.2.res_ln.bias` | 128 | 0 | 0 | 0 | +0.00198 | -0.01035 | +0.0298 | +0.004865 | +0.0298 |
| `blocks.2.res_ln.weight` | 128 | +1 | +1 | +1 | +1.004 | +0.9866 | +1.028 | +0.005474 | +1.028 |
| `blocks.2.rezero_alpha` | 1 | +0.4473 | +0.4473 | +0.4473 | +0.9371 | +0.9371 | +0.9371 | 0 | +0.9371 |
| `blocks.2.se_scalebias.fc1.bias` | 32 | 0 | 0 | 0 | +0.001678 | -0.005949 | +0.01873 | +0.006482 | +0.01873 |
| `blocks.2.se_scalebias.fc1.weight` | 32x128 | -0.001246 | -0.4395 | +0.4922 | -0.0009659 | -0.3996 | +0.4561 | +0.1139 | +0.4561 |
| `blocks.2.se_scalebias.fc2.bias` | 256 | 0 | 0 | 0 | +0.002891 | -0.04902 | +0.03986 | +0.01254 | +0.04902 |
| `blocks.2.se_scalebias.fc2.bias [gamma]` | 256 | 0 | 0 | 0 | +0.005776 | -0.04902 | +0.03986 | +0.01242 | +0.04902 |
| `blocks.2.se_scalebias.fc2.bias [beta]` | 256 | 0 | 0 | 0 | +5.845e-06 | -0.03757 | +0.02595 | +0.01199 | +0.03757 |
| `blocks.2.se_scalebias.fc2.weight` | 256x32 | +0.0005512 | -0.3496 | +0.332 | +0.002653 | -0.3193 | +0.2964 | +0.05507 | +0.3193 |
| `blocks.2.se_scalebias.fc2.weight [gamma]` | 256x32 | +0.001102 | -0.3496 | +0.332 | +0.005301 | -0.3193 | +0.2964 | +0.07648 | +0.3193 |
| `blocks.2.se_scalebias.fc2.weight [beta]` | 256x32 | 0 | 0 | 0 | +4.571e-06 | -0.07813 | +0.06884 | +0.01421 | +0.07813 |
| `policy.conv.bias` | 76 | 0 | 0 | 0 | -3.011e-05 | -0.06797 | +0.5365 | +0.07582 | +0.5365 |
| `policy.conv.weight` | 76x128x1x1 | -0.001422 | -0.4648 | +0.4785 | +0.008727 | -0.7603 | +0.8053 | +0.1462 | +0.8053 |
| `policy.pre_bn.bias` | 128 | 0 | 0 | 0 | +0.03669 | -0.07407 | +0.2186 | +0.06055 | +0.2186 |
| `policy.pre_bn.running_mean` | 128 | -0.02798 | -1.469 | +1.547 | -0.1002 | -1.32 | +1.069 | +0.5381 | +1.32 |
| `policy.pre_bn.running_var` | 128 | +0.679 | +0.3496 | +1.531 | +0.9143 | +0.4285 | +2.488 | +0.3337 | +2.488 |
| `policy.pre_bn.weight` | 128 | +1 | +1 | +1 | +1.065 | +0.8841 | +1.315 | +0.09084 | +1.315 |
| `policy.pre_conv.weight` | 128x128x1x1 | -0.0005189 | -0.5039 | +0.459 | -0.001043 | -0.5621 | +0.516 | +0.1218 | +0.5621 |
| `stem.bn.bias` | 128 | 0 | 0 | 0 | -0.006864 | -0.1684 | +0.09937 | +0.04899 | +0.1684 |
| `stem.bn.running_mean` | 128 | +0.001108 | -0.2715 | +0.2695 | -0.02853 | -0.4613 | +0.3264 | +0.1735 | +0.4613 |
| `stem.bn.running_var` | 128 | +0.05419 | +0.02075 | +0.1416 | +0.0879 | +0.02503 | +0.407 | +0.05452 | +0.407 |
| `stem.bn.weight` | 128 | +1 | +1 | +1 | +0.9793 | +0.6313 | +1.383 | +0.09855 | +1.383 |
| `stem.conv.weight` | 128x30x7x7 | -2.108e-05 | -0.1602 | +0.1768 | +4.506e-05 | -0.6564 | +1.24 | +0.04154 | +1.24 |
| `tower_final_bn.bias` | 128 | 0 | 0 | 0 | +0.0977 | -0.1275 | +0.3863 | +0.0858 | +0.3863 |
| `tower_final_bn.running_mean` | 128 | +4.53e-06 | -0.4336 | +0.5625 | +0.002227 | -1.851 | +1.929 | +0.7822 | +1.929 |
| `tower_final_bn.running_var` | 128 | +0.9525 | +0.6836 | +1.227 | +0.3956 | +0.1065 | +3.276 | +0.4144 | +3.276 |
| `tower_final_bn.weight` | 128 | +1 | +1 | +1 | +0.9943 | +0.8605 | +1.454 | +0.09199 | +1.454 |
| `value.bn.bias` | 16 | 0 | 0 | 0 | -0.1346 | -0.1978 | -0.037 | +0.03682 | +0.1978 |
| `value.bn.running_mean` | 16 | +0.1705 | -1.078 | +1.211 | +0.01203 | -1.219 | +1.101 | +0.6896 | +1.219 |
| `value.bn.running_var` | 16 | +0.7998 | +0.3867 | +1.195 | +1.107 | +0.6089 | +1.831 | +0.278 | +1.831 |
| `value.bn.weight` | 16 | +1 | +1 | +1 | +0.8079 | +0.7481 | +1.036 | +0.06446 | +1.036 |
| `value.conv.weight` | 16x128x1x1 | +0.00335 | -0.4434 | +0.4824 | +0.0006374 | -0.4066 | +0.4499 | +0.1176 | +0.4499 |
| `value.fc1.bias` | 128 | 0 | 0 | 0 | -0.001275 | -0.02974 | +0.03228 | +0.009744 | +0.03228 |
| `value.fc1.weight` | 128x1024 | +9.948e-05 | -0.1846 | +0.2109 | -0.001349 | -0.1992 | +0.1898 | +0.04044 | +0.1992 |
| `value.wdl_fc2.bias` | 3 | +0.5964 | 0 | +1.789 | +0.5958 | +0.267 | +1.203 | +0.4299 | +1.203 |
| `value.wdl_fc2.weight` | 3x128 | +0.00532 | -0.2793 | +0.3027 | +0.004884 | -0.2611 | +0.3499 | +0.1027 | +0.3499 |

## Optimizer velocity at the final checkpoint

Only checkpoints from build 2259 onward carry optimizer state (seed 2 and zero-β). `zero frac` is the fraction of elements whose momentum velocity is exactly 0.

### `se_zb1` (step 5,030)

| tensor | count | mean | min | max | rms | zero frac |
|---|---:|---:|---:|---:|---:|---:|
| `opt.blocks.0.bn1.bias.velocity` | 128 | +0.0001238 | -0.004678 | +0.006563 | +0.001267 | 0.0000 |
| `opt.blocks.0.bn1.weight.velocity` | 128 | +1.543e-05 | -0.005332 | +0.004664 | +0.001274 | 0.0000 |
| `opt.blocks.0.bn2.bias.velocity` | 128 | -0.0003838 | -0.002823 | +0.002123 | +0.000939 | 0.0000 |
| `opt.blocks.0.bn2.weight.velocity` | 128 | -0.0003935 | -0.003256 | +0.003652 | +0.001003 | 0.0000 |
| `opt.blocks.0.conv1.weight.velocity` | 802,816 | -4.331e-05 | -0.005728 | +0.004853 | +0.0004687 | 0.0000 |
| `opt.blocks.0.conv2.weight.velocity` | 802,816 | -3.423e-07 | -0.009067 | +0.007178 | +0.0004001 | 0.0000 |
| `opt.blocks.0.res_ln.bias.velocity` | 128 | +8.018e-05 | -0.002623 | +0.002581 | +0.001051 | 0.0000 |
| `opt.blocks.0.res_ln.weight.velocity` | 128 | +0.000129 | -0.007833 | +0.006691 | +0.002178 | 0.0000 |
| `opt.blocks.0.rezero_alpha.velocity` | 1 | -0.01051 | -0.01051 | -0.01051 | +0.01051 | 0.0000 |
| `opt.blocks.0.se_scalebias.fc1.bias.velocity` | 32 | -0.0001705 | -0.0009766 | +0.0002048 | +0.0003421 | 0.4062 |
| `opt.blocks.0.se_scalebias.fc1.weight.velocity` | 4,096 | -1.44e-05 | -0.005671 | +0.005485 | +0.0006909 | 0.4062 |
| `opt.blocks.0.se_scalebias.fc2.bias.velocity` | 256 | -3.693e-05 | -0.003656 | +0.001286 | +0.0004212 | 0.0000 |
| `opt.blocks.0.se_scalebias.fc2.bias.velocity [gamma]` | 128 | -7.431e-05 | -0.003656 | +0.001018 | +0.0004304 | 0.0000 |
| `opt.blocks.0.se_scalebias.fc2.bias.velocity [beta]` | 128 | +4.486e-07 | -0.00102 | +0.001286 | +0.0004119 | 0.0000 |
| `opt.blocks.0.se_scalebias.fc2.weight.velocity` | 8,192 | -5.276e-05 | -0.03055 | +0.008894 | +0.001296 | 0.4062 |
| `opt.blocks.0.se_scalebias.fc2.weight.velocity [gamma]` | 4,096 | -0.0001061 | -0.03055 | +0.008499 | +0.001309 | 0.4062 |
| `opt.blocks.0.se_scalebias.fc2.weight.velocity [beta]` | 4,096 | +5.683e-07 | -0.009059 | +0.008894 | +0.001283 | 0.4062 |
| `opt.blocks.1.bn1.bias.velocity` | 128 | +0.0001817 | -0.001918 | +0.003291 | +0.0009874 | 0.0000 |
| `opt.blocks.1.bn1.weight.velocity` | 128 | +3.267e-05 | -0.004693 | +0.003948 | +0.001088 | 0.0000 |
| `opt.blocks.1.bn2.bias.velocity` | 128 | +0.0001475 | -0.001678 | +0.002675 | +0.0007495 | 0.0000 |
| `opt.blocks.1.bn2.weight.velocity` | 128 | -6.389e-05 | -0.00198 | +0.003193 | +0.0008352 | 0.0000 |
| `opt.blocks.1.conv1.weight.velocity` | 802,816 | -1.42e-05 | -0.007767 | +0.005199 | +0.0004666 | 0.0000 |
| `opt.blocks.1.conv2.weight.velocity` | 802,816 | -3.553e-06 | -0.005007 | +0.00587 | +0.0004374 | 0.0000 |
| `opt.blocks.1.res_ln.bias.velocity` | 128 | +5.047e-05 | -0.002677 | +0.00221 | +0.0009267 | 0.0000 |
| `opt.blocks.1.res_ln.weight.velocity` | 128 | +9.001e-05 | -0.005567 | +0.01226 | +0.002288 | 0.0000 |
| `opt.blocks.1.rezero_alpha.velocity` | 1 | -0.001055 | -0.001055 | -0.001055 | +0.001055 | 0.0000 |
| `opt.blocks.1.se_scalebias.fc1.bias.velocity` | 32 | -4.302e-05 | -0.0007034 | +0.0004885 | +0.0002127 | 0.1875 |
| `opt.blocks.1.se_scalebias.fc1.weight.velocity` | 4,096 | +5.279e-06 | -0.002175 | +0.002214 | +0.0002612 | 0.1875 |
| `opt.blocks.1.se_scalebias.fc2.bias.velocity` | 256 | -6.219e-06 | -0.001494 | +0.001213 | +0.0004004 | 0.0000 |
| `opt.blocks.1.se_scalebias.fc2.bias.velocity [gamma]` | 128 | -1.206e-05 | -0.001494 | +0.001097 | +0.0003823 | 0.0000 |
| `opt.blocks.1.se_scalebias.fc2.bias.velocity [beta]` | 128 | -3.823e-07 | -0.001137 | +0.001213 | +0.0004177 | 0.0000 |
| `opt.blocks.1.se_scalebias.fc2.weight.velocity` | 8,192 | -5.564e-06 | -0.009392 | +0.008373 | +0.0007054 | 0.1875 |
| `opt.blocks.1.se_scalebias.fc2.weight.velocity [gamma]` | 4,096 | -1.093e-05 | -0.009392 | +0.005502 | +0.0006014 | 0.1875 |
| `opt.blocks.1.se_scalebias.fc2.weight.velocity [beta]` | 4,096 | -2.03e-07 | -0.00578 | +0.008373 | +0.0007959 | 0.1875 |
| `opt.blocks.2.bn1.bias.velocity` | 128 | +1.387e-05 | -0.002408 | +0.002259 | +0.000921 | 0.0000 |
| `opt.blocks.2.bn1.weight.velocity` | 128 | +7.407e-05 | -0.009079 | +0.005908 | +0.001466 | 0.0000 |
| `opt.blocks.2.bn2.bias.velocity` | 128 | +1.677e-05 | -0.002369 | +0.002265 | +0.000892 | 0.0000 |
| `opt.blocks.2.bn2.weight.velocity` | 128 | +6.853e-05 | -0.004058 | +0.003822 | +0.001149 | 0.0000 |
| `opt.blocks.2.conv1.weight.velocity` | 802,816 | +4.423e-06 | -0.005923 | +0.00557 | +0.0005136 | 0.0000 |
| `opt.blocks.2.conv2.weight.velocity` | 802,816 | -6.675e-06 | -0.0059 | +0.005767 | +0.0005939 | 0.0000 |
| `opt.blocks.2.res_ln.bias.velocity` | 128 | +4.601e-05 | -0.001099 | +0.001363 | +0.0002624 | 0.0000 |
| `opt.blocks.2.res_ln.weight.velocity` | 128 | +1.154e-05 | -0.002173 | +0.001123 | +0.0002721 | 0.0000 |
| `opt.blocks.2.rezero_alpha.velocity` | 1 | +7.583e-05 | +7.583e-05 | +7.583e-05 | +7.583e-05 | 0.0000 |
| `opt.blocks.2.se_scalebias.fc1.bias.velocity` | 32 | +2.755e-05 | -0.0004419 | +0.0005275 | +0.0001704 | 0.1250 |
| `opt.blocks.2.se_scalebias.fc1.weight.velocity` | 4,096 | -7.115e-06 | -0.002173 | +0.002186 | +0.0002912 | 0.1250 |
| `opt.blocks.2.se_scalebias.fc2.bias.velocity` | 256 | -1.606e-05 | -0.002491 | +0.001423 | +0.0004595 | 0.0000 |
| `opt.blocks.2.se_scalebias.fc2.bias.velocity [gamma]` | 128 | -3.201e-05 | -0.002491 | +0.001423 | +0.0005401 | 0.0000 |
| `opt.blocks.2.se_scalebias.fc2.bias.velocity [beta]` | 128 | -1.065e-07 | -0.001122 | +0.0007384 | +0.0003613 | 0.0000 |
| `opt.blocks.2.se_scalebias.fc2.weight.velocity` | 8,192 | -1.443e-05 | -0.01737 | +0.006949 | +0.0008831 | 0.1250 |
| `opt.blocks.2.se_scalebias.fc2.weight.velocity [gamma]` | 4,096 | -2.886e-05 | -0.01737 | +0.006949 | +0.001017 | 0.1250 |
| `opt.blocks.2.se_scalebias.fc2.weight.velocity [beta]` | 4,096 | -6.725e-10 | -0.007013 | +0.006198 | +0.0007254 | 0.1250 |
| `opt.policy.conv.bias.velocity` | 76 | -3.385e-07 | -0.007619 | +0.01267 | +0.002542 | 0.0000 |
| `opt.policy.conv.weight.velocity` | 9,728 | -0.0001805 | -0.0423 | +0.04678 | +0.004345 | 0.0000 |
| `opt.policy.pre_bn.bias.velocity` | 128 | -2.457e-05 | -0.004932 | +0.009072 | +0.002781 | 0.0000 |
| `opt.policy.pre_bn.weight.velocity` | 128 | -0.0007042 | -0.02136 | +0.02575 | +0.007183 | 0.0000 |
| `opt.policy.pre_conv.weight.velocity` | 16,384 | -4.135e-05 | -0.01601 | +0.01547 | +0.002676 | 0.0000 |
| `opt.stem.bn.bias.velocity` | 128 | +3.771e-05 | -0.001821 | +0.003363 | +0.0009898 | 0.0000 |
| `opt.stem.bn.weight.velocity` | 128 | +0.0004099 | -0.006864 | +0.008104 | +0.002361 | 0.0000 |
| `opt.stem.conv.weight.velocity` | 188,160 | +2.191e-05 | -0.02797 | +0.02538 | +0.00152 | 0.2381 |
| `opt.tower_final_bn.bias.velocity` | 128 | +0.0003543 | -0.006175 | +0.008593 | +0.002328 | 0.0000 |
| `opt.tower_final_bn.weight.velocity` | 128 | +2.971e-05 | -0.03236 | +0.02338 | +0.00721 | 0.0000 |
| `opt.value.bn.bias.velocity` | 16 | -0.001216 | -0.01055 | +0.008183 | +0.004955 | 0.0000 |
| `opt.value.bn.weight.velocity` | 16 | -0.001495 | -0.011 | +0.01053 | +0.006319 | 0.0000 |
| `opt.value.conv.weight.velocity` | 2,048 | -4.422e-05 | -0.009733 | +0.005934 | +0.001508 | 0.0000 |
| `opt.value.fc1.bias.velocity` | 128 | -0.0001382 | -0.002729 | +0.002176 | +0.0006518 | 0.0000 |
| `opt.value.fc1.weight.velocity` | 131,072 | -4.372e-05 | -0.006322 | +0.005428 | +0.0004111 | 0.0020 |
| `opt.value.wdl_fc2.bias.velocity` | 3 | -1.919e-05 | -0.01556 | +0.01211 | +0.01155 | 0.0000 |
| `opt.value.wdl_fc2.weight.velocity` | 384 | +8.507e-08 | -0.03689 | +0.03573 | +0.006521 | 0.0000 |

### `se_sb2` (step 7,282)

| tensor | count | mean | min | max | rms | zero frac |
|---|---:|---:|---:|---:|---:|---:|
| `opt.blocks.0.bn1.bias.velocity` | 128 | -3.915e-05 | -0.004332 | +0.004418 | +0.001359 | 0.0000 |
| `opt.blocks.0.bn1.weight.velocity` | 128 | +1.511e-05 | -0.004745 | +0.003431 | +0.001287 | 0.0000 |
| `opt.blocks.0.bn2.bias.velocity` | 128 | -0.0001945 | -0.003608 | +0.002612 | +0.001158 | 0.0000 |
| `opt.blocks.0.bn2.weight.velocity` | 128 | -0.0002592 | -0.004557 | +0.002669 | +0.001212 | 0.0000 |
| `opt.blocks.0.conv1.weight.velocity` | 802,816 | -2.171e-05 | -0.00563 | +0.00512 | +0.0005967 | 0.0000 |
| `opt.blocks.0.conv2.weight.velocity` | 802,816 | +5.855e-06 | -0.006416 | +0.006432 | +0.000504 | 0.0000 |
| `opt.blocks.0.res_ln.bias.velocity` | 128 | +3.109e-05 | -0.003476 | +0.003525 | +0.001404 | 0.0000 |
| `opt.blocks.0.res_ln.weight.velocity` | 128 | +6.307e-05 | -0.009853 | +0.008174 | +0.002357 | 0.0000 |
| `opt.blocks.0.rezero_alpha.velocity` | 1 | -0.00555 | -0.00555 | -0.00555 | +0.00555 | 0.0000 |
| `opt.blocks.0.se_scalebias.fc1.bias.velocity` | 32 | -7.264e-05 | -0.001017 | +0.001272 | +0.0004521 | 0.1250 |
| `opt.blocks.0.se_scalebias.fc1.weight.velocity` | 4,096 | +5.237e-06 | -0.006989 | +0.007651 | +0.001092 | 0.1250 |
| `opt.blocks.0.se_scalebias.fc2.bias.velocity` | 256 | +1.287e-05 | -0.002227 | +0.001812 | +0.0003874 | 0.0000 |
| `opt.blocks.0.se_scalebias.fc2.bias.velocity [gamma]` | 128 | +2.617e-05 | -0.002227 | +0.0008878 | +0.0003443 | 0.0000 |
| `opt.blocks.0.se_scalebias.fc2.bias.velocity [beta]` | 128 | -4.332e-07 | -0.001209 | +0.001812 | +0.0004262 | 0.0000 |
| `opt.blocks.0.se_scalebias.fc2.weight.velocity` | 8,192 | +2.009e-05 | -0.03117 | +0.01609 | +0.001512 | 0.1252 |
| `opt.blocks.0.se_scalebias.fc2.weight.velocity [gamma]` | 4,096 | +4.085e-05 | -0.03117 | +0.01102 | +0.001279 | 0.1255 |
| `opt.blocks.0.se_scalebias.fc2.weight.velocity [beta]` | 4,096 | -6.793e-07 | -0.01519 | +0.01609 | +0.001713 | 0.1250 |
| `opt.blocks.1.bn1.bias.velocity` | 128 | +0.0002239 | -0.002804 | +0.003905 | +0.001356 | 0.0000 |
| `opt.blocks.1.bn1.weight.velocity` | 128 | +6.323e-05 | -0.009463 | +0.005948 | +0.001641 | 0.0000 |
| `opt.blocks.1.bn2.bias.velocity` | 128 | -0.0001294 | -0.002515 | +0.003421 | +0.001003 | 0.0000 |
| `opt.blocks.1.bn2.weight.velocity` | 128 | -8.362e-05 | -0.002688 | +0.003681 | +0.001167 | 0.0000 |
| `opt.blocks.1.conv1.weight.velocity` | 802,816 | +1.352e-05 | -0.01248 | +0.01186 | +0.0006448 | 0.0000 |
| `opt.blocks.1.conv2.weight.velocity` | 802,816 | -5.339e-06 | -0.007056 | +0.005438 | +0.0005832 | 0.0000 |
| `opt.blocks.1.res_ln.bias.velocity` | 128 | +9.423e-05 | -0.003948 | +0.003623 | +0.001328 | 0.0000 |
| `opt.blocks.1.res_ln.weight.velocity` | 128 | +0.0001039 | -0.01124 | +0.01023 | +0.002597 | 0.0000 |
| `opt.blocks.1.rezero_alpha.velocity` | 1 | -0.0007118 | -0.0007118 | -0.0007118 | +0.0007118 | 0.0000 |
| `opt.blocks.1.se_scalebias.fc1.bias.velocity` | 32 | -6.106e-05 | -0.001775 | +0.001565 | +0.0006089 | 0.4062 |
| `opt.blocks.1.se_scalebias.fc1.weight.velocity` | 4,096 | +1.769e-06 | -0.00763 | +0.005444 | +0.0006831 | 0.4062 |
| `opt.blocks.1.se_scalebias.fc2.bias.velocity` | 256 | +3.384e-05 | -0.001305 | +0.001441 | +0.0004734 | 0.0000 |
| `opt.blocks.1.se_scalebias.fc2.bias.velocity [gamma]` | 128 | +6.708e-05 | -0.00107 | +0.001441 | +0.0004522 | 0.0000 |
| `opt.blocks.1.se_scalebias.fc2.bias.velocity [beta]` | 128 | +6.024e-07 | -0.001305 | +0.001261 | +0.0004937 | 0.0000 |
| `opt.blocks.1.se_scalebias.fc2.weight.velocity` | 8,192 | +2.393e-05 | -0.01088 | +0.009803 | +0.0009217 | 0.4062 |
| `opt.blocks.1.se_scalebias.fc2.weight.velocity [gamma]` | 4,096 | +4.751e-05 | -0.006663 | +0.007124 | +0.0006899 | 0.4062 |
| `opt.blocks.1.se_scalebias.fc2.weight.velocity [beta]` | 4,096 | +3.533e-07 | -0.01088 | +0.009803 | +0.001106 | 0.4062 |
| `opt.blocks.2.bn1.bias.velocity` | 128 | +0.0003236 | -0.002605 | +0.007636 | +0.00147 | 0.0000 |
| `opt.blocks.2.bn1.weight.velocity` | 128 | +2.718e-05 | -0.004783 | +0.008197 | +0.00173 | 0.0000 |
| `opt.blocks.2.bn2.bias.velocity` | 128 | -0.0001714 | -0.003782 | +0.003102 | +0.001212 | 0.0000 |
| `opt.blocks.2.bn2.weight.velocity` | 128 | -5.419e-05 | -0.002988 | +0.004351 | +0.001309 | 0.0000 |
| `opt.blocks.2.conv1.weight.velocity` | 802,816 | +3.997e-05 | -0.006225 | +0.006635 | +0.0006774 | 0.0000 |
| `opt.blocks.2.conv2.weight.velocity` | 802,816 | +1.367e-05 | -0.007525 | +0.007353 | +0.0007854 | 0.0000 |
| `opt.blocks.2.res_ln.bias.velocity` | 128 | +2.977e-06 | -0.001658 | +0.001219 | +0.0003512 | 0.0000 |
| `opt.blocks.2.res_ln.weight.velocity` | 128 | -1.743e-05 | -0.001646 | +0.001156 | +0.0003533 | 0.0000 |
| `opt.blocks.2.rezero_alpha.velocity` | 1 | -0.0008509 | -0.0008509 | -0.0008509 | +0.0008509 | 0.0000 |
| `opt.blocks.2.se_scalebias.fc1.bias.velocity` | 32 | -0.0001663 | -0.002106 | +0.001117 | +0.0006283 | 0.1250 |
| `opt.blocks.2.se_scalebias.fc1.weight.velocity` | 4,096 | +3.672e-05 | -0.004921 | +0.006869 | +0.0009203 | 0.1250 |
| `opt.blocks.2.se_scalebias.fc2.bias.velocity` | 256 | +6.139e-05 | -0.002014 | +0.003331 | +0.0007104 | 0.0000 |
| `opt.blocks.2.se_scalebias.fc2.bias.velocity [gamma]` | 128 | +0.0001251 | -0.002014 | +0.003331 | +0.0008771 | 0.0000 |
| `opt.blocks.2.se_scalebias.fc2.bias.velocity [beta]` | 128 | -2.362e-06 | -0.001809 | +0.001491 | +0.0004898 | 0.0000 |
| `opt.blocks.2.se_scalebias.fc2.weight.velocity` | 8,192 | +5.202e-05 | -0.0119 | +0.02064 | +0.001421 | 0.1250 |
| `opt.blocks.2.se_scalebias.fc2.weight.velocity [gamma]` | 4,096 | +0.0001064 | -0.0119 | +0.02064 | +0.001678 | 0.1250 |
| `opt.blocks.2.se_scalebias.fc2.weight.velocity [beta]` | 4,096 | -2.321e-06 | -0.01078 | +0.00892 | +0.001107 | 0.1250 |
| `opt.policy.conv.bias.velocity` | 76 | +1.164e-06 | -0.01897 | +0.01647 | +0.004039 | 0.0000 |
| `opt.policy.conv.weight.velocity` | 9,728 | -0.0002785 | -0.08642 | +0.05456 | +0.006926 | 0.0000 |
| `opt.policy.pre_bn.bias.velocity` | 128 | +0.0003164 | -0.02003 | +0.01725 | +0.004887 | 0.0000 |
| `opt.policy.pre_bn.weight.velocity` | 128 | -0.001064 | -0.0696 | +0.03917 | +0.01299 | 0.0000 |
| `opt.policy.pre_conv.weight.velocity` | 16,384 | -0.0001669 | -0.02355 | +0.01847 | +0.00348 | 0.0000 |
| `opt.stem.bn.bias.velocity` | 128 | -3.802e-07 | -0.002969 | +0.003963 | +0.001055 | 0.0000 |
| `opt.stem.bn.weight.velocity` | 128 | +0.0002992 | -0.006761 | +0.01208 | +0.002686 | 0.0000 |
| `opt.stem.conv.weight.velocity` | 188,160 | -5.401e-06 | -0.02918 | +0.02863 | +0.001993 | 0.2381 |
| `opt.tower_final_bn.bias.velocity` | 128 | +0.0002203 | -0.01408 | +0.01045 | +0.003228 | 0.0000 |
| `opt.tower_final_bn.weight.velocity` | 128 | -0.0001684 | -0.03594 | +0.03921 | +0.008998 | 0.0000 |
| `opt.value.bn.bias.velocity` | 16 | +0.007514 | -0.004355 | +0.02393 | +0.01075 | 0.0000 |
| `opt.value.bn.weight.velocity` | 16 | +0.008155 | -0.00542 | +0.02354 | +0.01106 | 0.0000 |
| `opt.value.conv.weight.velocity` | 2,048 | +0.0001794 | -0.01455 | +0.01368 | +0.00195 | 0.0000 |
| `opt.value.fc1.bias.velocity` | 128 | +0.0007091 | -0.001189 | +0.007518 | +0.001732 | 0.0000 |
| `opt.value.fc1.weight.velocity` | 131,072 | +0.0001846 | -0.004624 | +0.01008 | +0.0006868 | 0.0004 |
| `opt.value.wdl_fc2.bias.velocity` | 3 | +1.013e-05 | -0.02579 | +0.01387 | +0.01826 | 0.0000 |
| `opt.value.wdl_fc2.weight.velocity` | 384 | +1.336e-06 | -0.05008 | +0.04217 | +0.008718 | 0.0000 |

### `se_att2` (step 7,289)

| tensor | count | mean | min | max | rms | zero frac |
|---|---:|---:|---:|---:|---:|---:|
| `opt.blocks.0.bn1.bias.velocity` | 128 | -5.498e-05 | -0.003577 | +0.002805 | +0.00124 | 0.0000 |
| `opt.blocks.0.bn1.weight.velocity` | 128 | +1.347e-05 | -0.003454 | +0.003949 | +0.00136 | 0.0000 |
| `opt.blocks.0.bn2.bias.velocity` | 128 | -3.422e-05 | -0.002442 | +0.005096 | +0.001127 | 0.0000 |
| `opt.blocks.0.bn2.weight.velocity` | 128 | +0.0001032 | -0.003117 | +0.007662 | +0.001304 | 0.0000 |
| `opt.blocks.0.conv1.weight.velocity` | 802,816 | +2.952e-05 | -0.007546 | +0.006555 | +0.0006211 | 0.0000 |
| `opt.blocks.0.conv2.weight.velocity` | 802,816 | +2.919e-07 | -0.006009 | +0.006775 | +0.0005408 | 0.0000 |
| `opt.blocks.0.res_ln.bias.velocity` | 128 | +5.07e-05 | -0.002764 | +0.00798 | +0.00135 | 0.0000 |
| `opt.blocks.0.res_ln.weight.velocity` | 128 | +4.115e-05 | -0.00922 | +0.006962 | +0.002523 | 0.0000 |
| `opt.blocks.0.rezero_alpha.velocity` | 1 | +0.001185 | +0.001185 | +0.001185 | +0.001185 | 0.0000 |
| `opt.blocks.0.se_attenuate.fc1.bias.velocity` | 32 | +3.254e-06 | -0.0004942 | +0.000491 | +0.0002144 | 0.2500 |
| `opt.blocks.0.se_attenuate.fc1.weight.velocity` | 4,096 | -2.081e-07 | -0.004567 | +0.004656 | +0.0006923 | 0.2500 |
| `opt.blocks.0.se_attenuate.fc2.bias.velocity` | 128 | -8.866e-06 | -0.002143 | +0.003131 | +0.0004458 | 0.0000 |
| `opt.blocks.0.se_attenuate.fc2.weight.velocity` | 4,096 | -1.834e-06 | -0.01467 | +0.02797 | +0.001414 | 0.2505 |
| `opt.blocks.1.bn1.bias.velocity` | 128 | +5.541e-05 | -0.003701 | +0.004813 | +0.001431 | 0.0000 |
| `opt.blocks.1.bn1.weight.velocity` | 128 | +1.055e-05 | -0.006794 | +0.005811 | +0.00152 | 0.0000 |
| `opt.blocks.1.bn2.bias.velocity` | 128 | -0.0002381 | -0.003419 | +0.006127 | +0.001221 | 0.0000 |
| `opt.blocks.1.bn2.weight.velocity` | 128 | -0.0002151 | -0.003077 | +0.01033 | +0.001466 | 0.0000 |
| `opt.blocks.1.conv1.weight.velocity` | 802,816 | -7.612e-06 | -0.005188 | +0.005762 | +0.0006382 | 0.0000 |
| `opt.blocks.1.conv2.weight.velocity` | 802,816 | +4.302e-06 | -0.00525 | +0.005005 | +0.0005712 | 0.0000 |
| `opt.blocks.1.res_ln.bias.velocity` | 128 | -1.078e-05 | -0.003277 | +0.005979 | +0.00129 | 0.0000 |
| `opt.blocks.1.res_ln.weight.velocity` | 128 | +0.0002007 | -0.01641 | +0.01129 | +0.003161 | 0.0000 |
| `opt.blocks.1.rezero_alpha.velocity` | 1 | -0.002085 | -0.002085 | -0.002085 | +0.002085 | 0.0000 |
| `opt.blocks.1.se_attenuate.fc1.bias.velocity` | 32 | -9.749e-05 | -0.001014 | +0.000642 | +0.0003816 | 0.2188 |
| `opt.blocks.1.se_attenuate.fc1.weight.velocity` | 4,096 | +3.374e-05 | -0.002923 | +0.003791 | +0.0005292 | 0.2092 |
| `opt.blocks.1.se_attenuate.fc2.bias.velocity` | 128 | +3.366e-05 | -0.001112 | +0.002666 | +0.00047 | 0.0000 |
| `opt.blocks.1.se_attenuate.fc2.weight.velocity` | 4,096 | +1.018e-06 | -0.00981 | +0.01769 | +0.000758 | 0.2188 |
| `opt.blocks.2.bn1.bias.velocity` | 128 | -9.734e-05 | -0.003228 | +0.004141 | +0.001231 | 0.0000 |
| `opt.blocks.2.bn1.weight.velocity` | 128 | +3.07e-06 | -0.004704 | +0.004946 | +0.001603 | 0.0000 |
| `opt.blocks.2.bn2.bias.velocity` | 128 | -0.0003884 | -0.003545 | +0.002873 | +0.001186 | 0.0000 |
| `opt.blocks.2.bn2.weight.velocity` | 128 | -0.0002062 | -0.004351 | +0.003575 | +0.001359 | 0.0000 |
| `opt.blocks.2.conv1.weight.velocity` | 802,816 | +1.133e-06 | -0.006149 | +0.006132 | +0.0006439 | 0.0000 |
| `opt.blocks.2.conv2.weight.velocity` | 802,816 | +7.446e-06 | -0.009891 | +0.007157 | +0.0007505 | 0.0000 |
| `opt.blocks.2.res_ln.bias.velocity` | 128 | -3.511e-05 | -0.0008364 | +0.0008279 | +0.0002773 | 0.0000 |
| `opt.blocks.2.res_ln.weight.velocity` | 128 | -6.041e-05 | -0.001271 | +0.0004826 | +0.000229 | 0.0000 |
| `opt.blocks.2.rezero_alpha.velocity` | 1 | -0.004765 | -0.004765 | -0.004765 | +0.004765 | 0.0000 |
| `opt.blocks.2.se_attenuate.fc1.bias.velocity` | 32 | +2.931e-05 | -0.0009793 | +0.001499 | +0.0004824 | 0.1562 |
| `opt.blocks.2.se_attenuate.fc1.weight.velocity` | 4,096 | -6.67e-06 | -0.008146 | +0.008713 | +0.0007573 | 0.1562 |
| `opt.blocks.2.se_attenuate.fc2.bias.velocity` | 128 | -7.45e-05 | -0.004421 | +0.001969 | +0.000823 | 0.0000 |
| `opt.blocks.2.se_attenuate.fc2.weight.velocity` | 4,096 | -8.481e-05 | -0.02831 | +0.01014 | +0.00162 | 0.1562 |
| `opt.policy.conv.bias.velocity` | 76 | -2.745e-08 | -0.01686 | +0.007961 | +0.003604 | 0.0000 |
| `opt.policy.conv.weight.velocity` | 9,728 | +0.0001627 | -0.0643 | +0.0501 | +0.006136 | 0.0000 |
| `opt.policy.pre_bn.bias.velocity` | 128 | -0.0009557 | -0.02284 | +0.006749 | +0.004772 | 0.0000 |
| `opt.policy.pre_bn.weight.velocity` | 128 | -0.0002008 | -0.03776 | +0.02488 | +0.01128 | 0.0000 |
| `opt.policy.pre_conv.weight.velocity` | 16,384 | -4.399e-05 | -0.02293 | +0.02457 | +0.003404 | 0.0000 |
| `opt.stem.bn.bias.velocity` | 128 | +2.128e-05 | -0.005304 | +0.008588 | +0.001652 | 0.0000 |
| `opt.stem.bn.weight.velocity` | 128 | -9.517e-05 | -0.01033 | +0.009359 | +0.002553 | 0.0000 |
| `opt.stem.conv.weight.velocity` | 188,160 | +2.943e-05 | -0.02739 | +0.03518 | +0.002186 | 0.2381 |
| `opt.tower_final_bn.bias.velocity` | 128 | -0.0007837 | -0.0127 | +0.01013 | +0.003378 | 0.0000 |
| `opt.tower_final_bn.weight.velocity` | 128 | +0.0002897 | -0.02905 | +0.03709 | +0.008727 | 0.0000 |
| `opt.value.bn.bias.velocity` | 16 | +0.001622 | -0.0323 | +0.03771 | +0.01535 | 0.0000 |
| `opt.value.bn.weight.velocity` | 16 | +0.002128 | -0.039 | +0.04695 | +0.019 | 0.0000 |
| `opt.value.conv.weight.velocity` | 2,048 | +1.121e-05 | -0.01473 | +0.01787 | +0.002219 | 0.0000 |
| `opt.value.fc1.bias.velocity` | 128 | -4.88e-05 | -0.005094 | +0.004444 | +0.001349 | 0.0000 |
| `opt.value.fc1.weight.velocity` | 131,072 | +5.167e-06 | -0.0141 | +0.009133 | +0.0006918 | 0.0048 |
| `opt.value.wdl_fc2.bias.velocity` | 3 | +2.57e-05 | -0.04269 | +0.04406 | +0.03543 | 0.0000 |
| `opt.value.wdl_fc2.weight.velocity` | 384 | +1.535e-06 | -0.06369 | +0.09195 | +0.01656 | 0.0000 |

### `se_none2` (step 7,019)

| tensor | count | mean | min | max | rms | zero frac |
|---|---:|---:|---:|---:|---:|---:|
| `opt.blocks.0.bn1.bias.velocity` | 128 | +7.468e-05 | -0.002966 | +0.006109 | +0.001386 | 0.0000 |
| `opt.blocks.0.bn1.weight.velocity` | 128 | +1.156e-05 | -0.006523 | +0.004096 | +0.001537 | 0.0000 |
| `opt.blocks.0.bn2.bias.velocity` | 128 | +0.0003327 | -0.002463 | +0.00557 | +0.001235 | 0.0000 |
| `opt.blocks.0.bn2.weight.velocity` | 128 | +0.0003759 | -0.002523 | +0.006459 | +0.001284 | 0.0000 |
| `opt.blocks.0.conv1.weight.velocity` | 802,816 | +2.976e-05 | -0.007304 | +0.007822 | +0.0006004 | 0.0000 |
| `opt.blocks.0.conv2.weight.velocity` | 802,816 | +1.205e-08 | -0.007125 | +0.01389 | +0.000561 | 0.0000 |
| `opt.blocks.0.res_ln.bias.velocity` | 128 | +0.0001046 | -0.003251 | +0.005731 | +0.001111 | 0.0000 |
| `opt.blocks.0.res_ln.weight.velocity` | 128 | +2.452e-05 | -0.00581 | +0.00572 | +0.001738 | 0.0000 |
| `opt.blocks.0.rezero_alpha.velocity` | 1 | +0.01502 | +0.01502 | +0.01502 | +0.01502 | 0.0000 |
| `opt.blocks.1.bn1.bias.velocity` | 128 | +0.000182 | -0.00334 | +0.006856 | +0.0014 | 0.0000 |
| `opt.blocks.1.bn1.weight.velocity` | 128 | +3.59e-05 | -0.004373 | +0.007284 | +0.001409 | 0.0000 |
| `opt.blocks.1.bn2.bias.velocity` | 128 | -0.0001413 | -0.002291 | +0.002434 | +0.001029 | 0.0000 |
| `opt.blocks.1.bn2.weight.velocity` | 128 | -6.562e-05 | -0.002791 | +0.005398 | +0.001119 | 0.0000 |
| `opt.blocks.1.conv1.weight.velocity` | 802,816 | -1.476e-05 | -0.00583 | +0.006839 | +0.0005975 | 0.0000 |
| `opt.blocks.1.conv2.weight.velocity` | 802,816 | -1.257e-07 | -0.007423 | +0.006949 | +0.0005633 | 0.0000 |
| `opt.blocks.1.res_ln.bias.velocity` | 128 | +3.308e-05 | -0.003457 | +0.003255 | +0.001125 | 0.0000 |
| `opt.blocks.1.res_ln.weight.velocity` | 128 | +4.259e-05 | -0.00529 | +0.009095 | +0.001953 | 0.0000 |
| `opt.blocks.1.rezero_alpha.velocity` | 1 | -0.006204 | -0.006204 | -0.006204 | +0.006204 | 0.0000 |
| `opt.blocks.2.bn1.bias.velocity` | 128 | -1.534e-05 | -0.004347 | +0.002863 | +0.001227 | 0.0000 |
| `opt.blocks.2.bn1.weight.velocity` | 128 | +2.433e-05 | -0.004422 | +0.004795 | +0.001366 | 0.0000 |
| `opt.blocks.2.bn2.bias.velocity` | 128 | -0.0002348 | -0.002423 | +0.004566 | +0.001254 | 0.0000 |
| `opt.blocks.2.bn2.weight.velocity` | 128 | -4.51e-05 | -0.004212 | +0.003692 | +0.001398 | 0.0000 |
| `opt.blocks.2.conv1.weight.velocity` | 802,816 | -9.412e-06 | -0.007349 | +0.009647 | +0.0006344 | 0.0000 |
| `opt.blocks.2.conv2.weight.velocity` | 802,816 | -7.551e-08 | -0.006868 | +0.006372 | +0.0007489 | 0.0000 |
| `opt.blocks.2.res_ln.bias.velocity` | 128 | +8.977e-06 | -0.001703 | +0.001132 | +0.0003544 | 0.0000 |
| `opt.blocks.2.res_ln.weight.velocity` | 128 | +3.198e-05 | -0.0006933 | +0.002629 | +0.0003771 | 0.0000 |
| `opt.blocks.2.rezero_alpha.velocity` | 1 | -0.004649 | -0.004649 | -0.004649 | +0.004649 | 0.0000 |
| `opt.policy.conv.bias.velocity` | 76 | -2.873e-07 | -0.01139 | +0.009247 | +0.002647 | 0.0000 |
| `opt.policy.conv.weight.velocity` | 9,728 | +4.87e-06 | -0.0436 | +0.05693 | +0.004866 | 0.0000 |
| `opt.policy.pre_bn.bias.velocity` | 128 | -0.0004237 | -0.009455 | +0.007657 | +0.003047 | 0.0000 |
| `opt.policy.pre_bn.weight.velocity` | 128 | -0.0004643 | -0.02158 | +0.02386 | +0.008217 | 0.0000 |
| `opt.policy.pre_conv.weight.velocity` | 16,384 | -8.129e-05 | -0.01938 | +0.02409 | +0.00288 | 0.0000 |
| `opt.stem.bn.bias.velocity` | 128 | +3.113e-05 | -0.0025 | +0.006612 | +0.001139 | 0.0000 |
| `opt.stem.bn.weight.velocity` | 128 | -0.000359 | -0.01013 | +0.01637 | +0.002812 | 0.0000 |
| `opt.stem.conv.weight.velocity` | 188,160 | +3.074e-05 | -0.02193 | +0.02834 | +0.001649 | 0.2381 |
| `opt.tower_final_bn.bias.velocity` | 128 | -0.0001649 | -0.00977 | +0.006657 | +0.002691 | 0.0000 |
| `opt.tower_final_bn.weight.velocity` | 128 | +0.0001436 | -0.0237 | +0.02327 | +0.005617 | 0.0000 |
| `opt.value.bn.bias.velocity` | 16 | +0.0002279 | -0.006322 | +0.007462 | +0.004045 | 0.0000 |
| `opt.value.bn.weight.velocity` | 16 | -2.203e-05 | -0.009745 | +0.006 | +0.00479 | 0.0000 |
| `opt.value.conv.weight.velocity` | 2,048 | +1.409e-05 | -0.006527 | +0.008949 | +0.001449 | 0.0000 |
| `opt.value.fc1.bias.velocity` | 128 | -1.102e-05 | -0.002581 | +0.003106 | +0.000782 | 0.0000 |
| `opt.value.fc1.weight.velocity` | 131,072 | -2.066e-06 | -0.005082 | +0.007072 | +0.0004265 | 0.0038 |
| `opt.value.wdl_fc2.bias.velocity` | 3 | +5.008e-06 | -0.01438 | +0.02044 | +0.01485 | 0.0000 |
| `opt.value.wdl_fc2.weight.velocity` | 384 | -4.889e-07 | -0.03646 | +0.04421 | +0.007688 | 0.0000 |

### `se_zb2` (step 5,004)

| tensor | count | mean | min | max | rms | zero frac |
|---|---:|---:|---:|---:|---:|---:|
| `opt.blocks.0.bn1.bias.velocity` | 128 | +9.663e-05 | -0.00294 | +0.002655 | +0.0008688 | 0.0000 |
| `opt.blocks.0.bn1.weight.velocity` | 128 | +1.311e-05 | -0.002563 | +0.004326 | +0.0008931 | 0.0000 |
| `opt.blocks.0.bn2.bias.velocity` | 128 | -4.806e-05 | -0.002286 | +0.003003 | +0.0008094 | 0.0000 |
| `opt.blocks.0.bn2.weight.velocity` | 128 | -0.000109 | -0.004059 | +0.005029 | +0.0009862 | 0.0000 |
| `opt.blocks.0.conv1.weight.velocity` | 802,816 | +1.085e-07 | -0.005849 | +0.00509 | +0.0004469 | 0.0000 |
| `opt.blocks.0.conv2.weight.velocity` | 802,816 | +1.924e-06 | -0.005081 | +0.005185 | +0.0003953 | 0.0000 |
| `opt.blocks.0.res_ln.bias.velocity` | 128 | +8.546e-05 | -0.002795 | +0.002602 | +0.0008613 | 0.0000 |
| `opt.blocks.0.res_ln.weight.velocity` | 128 | +0.0001072 | -0.006388 | +0.006027 | +0.001732 | 0.0000 |
| `opt.blocks.0.rezero_alpha.velocity` | 1 | -0.001673 | -0.001673 | -0.001673 | +0.001673 | 0.0000 |
| `opt.blocks.0.se_scalebias.fc1.bias.velocity` | 32 | +7.382e-05 | -0.0003658 | +0.0005175 | +0.0002115 | 0.2812 |
| `opt.blocks.0.se_scalebias.fc1.weight.velocity` | 4,096 | -4.576e-06 | -0.002918 | +0.002351 | +0.0005069 | 0.2812 |
| `opt.blocks.0.se_scalebias.fc2.bias.velocity` | 256 | -2.503e-05 | -0.001264 | +0.001076 | +0.0003103 | 0.0000 |
| `opt.blocks.0.se_scalebias.fc2.bias.velocity [gamma]` | 128 | -5.059e-05 | -0.001209 | +0.001019 | +0.0002785 | 0.0000 |
| `opt.blocks.0.se_scalebias.fc2.bias.velocity [beta]` | 128 | +5.278e-07 | -0.001264 | +0.001076 | +0.0003392 | 0.0000 |
| `opt.blocks.0.se_scalebias.fc2.weight.velocity` | 8,192 | -3.45e-05 | -0.01283 | +0.0122 | +0.001194 | 0.2812 |
| `opt.blocks.0.se_scalebias.fc2.weight.velocity [gamma]` | 4,096 | -6.989e-05 | -0.009164 | +0.009146 | +0.0009116 | 0.2812 |
| `opt.blocks.0.se_scalebias.fc2.weight.velocity [beta]` | 4,096 | +9.013e-07 | -0.01283 | +0.0122 | +0.001421 | 0.2812 |
| `opt.blocks.1.bn1.bias.velocity` | 128 | +0.0003482 | -0.001904 | +0.002816 | +0.000974 | 0.0000 |
| `opt.blocks.1.bn1.weight.velocity` | 128 | +6.643e-05 | -0.003723 | +0.00252 | +0.001182 | 0.0000 |
| `opt.blocks.1.bn2.bias.velocity` | 128 | +5.15e-05 | -0.001884 | +0.005398 | +0.0008709 | 0.0000 |
| `opt.blocks.1.bn2.weight.velocity` | 128 | -2.239e-05 | -0.003178 | +0.007901 | +0.001156 | 0.0000 |
| `opt.blocks.1.conv1.weight.velocity` | 802,816 | +1.048e-05 | -0.005107 | +0.005781 | +0.0004774 | 0.0000 |
| `opt.blocks.1.conv2.weight.velocity` | 802,816 | +1.413e-06 | -0.004536 | +0.003871 | +0.000444 | 0.0000 |
| `opt.blocks.1.res_ln.bias.velocity` | 128 | +2.026e-05 | -0.003648 | +0.001935 | +0.000927 | 0.0000 |
| `opt.blocks.1.res_ln.weight.velocity` | 128 | +0.0003478 | -0.01132 | +0.005694 | +0.002038 | 0.0000 |
| `opt.blocks.1.rezero_alpha.velocity` | 1 | -0.000814 | -0.000814 | -0.000814 | +0.000814 | 0.0000 |
| `opt.blocks.1.se_scalebias.fc1.bias.velocity` | 32 | -1.413e-05 | -0.001127 | +0.0004564 | +0.0002971 | 0.1250 |
| `opt.blocks.1.se_scalebias.fc1.weight.velocity` | 4,096 | +4.517e-06 | -0.004125 | +0.003047 | +0.0003882 | 0.1250 |
| `opt.blocks.1.se_scalebias.fc2.bias.velocity` | 256 | -2.693e-06 | -0.001182 | +0.001241 | +0.0003041 | 0.0000 |
| `opt.blocks.1.se_scalebias.fc2.bias.velocity [gamma]` | 128 | -5.553e-06 | -0.0009253 | +0.001241 | +0.0002899 | 0.0000 |
| `opt.blocks.1.se_scalebias.fc2.bias.velocity [beta]` | 128 | +1.667e-07 | -0.001182 | +0.0007077 | +0.0003177 | 0.0000 |
| `opt.blocks.1.se_scalebias.fc2.weight.velocity` | 8,192 | -4.178e-06 | -0.00718 | +0.009439 | +0.0007305 | 0.1250 |
| `opt.blocks.1.se_scalebias.fc2.weight.velocity [gamma]` | 4,096 | -8.548e-06 | -0.005704 | +0.006679 | +0.000596 | 0.1250 |
| `opt.blocks.1.se_scalebias.fc2.weight.velocity [beta]` | 4,096 | +1.907e-07 | -0.00718 | +0.009439 | +0.0008438 | 0.1250 |
| `opt.blocks.2.bn1.bias.velocity` | 128 | +0.0001935 | -0.002409 | +0.002554 | +0.0009295 | 0.0000 |
| `opt.blocks.2.bn1.weight.velocity` | 128 | +3.418e-05 | -0.003208 | +0.003549 | +0.001169 | 0.0000 |
| `opt.blocks.2.bn2.bias.velocity` | 128 | -0.0003561 | -0.002523 | +0.001447 | +0.0008445 | 0.0000 |
| `opt.blocks.2.bn2.weight.velocity` | 128 | -0.0003561 | -0.003708 | +0.002952 | +0.001202 | 0.0000 |
| `opt.blocks.2.conv1.weight.velocity` | 802,816 | -5.753e-06 | -0.004662 | +0.004627 | +0.0005121 | 0.0000 |
| `opt.blocks.2.conv2.weight.velocity` | 802,816 | +1.512e-07 | -0.00605 | +0.006442 | +0.000575 | 0.0000 |
| `opt.blocks.2.res_ln.bias.velocity` | 128 | +1.415e-05 | -0.000716 | +0.0007504 | +0.0002047 | 0.0000 |
| `opt.blocks.2.res_ln.weight.velocity` | 128 | -1.509e-05 | -0.0006506 | +0.000703 | +0.0001813 | 0.0000 |
| `opt.blocks.2.rezero_alpha.velocity` | 1 | -0.007238 | -0.007238 | -0.007238 | +0.007238 | 0.0000 |
| `opt.blocks.2.se_scalebias.fc1.bias.velocity` | 32 | +6.216e-05 | -0.0005209 | +0.0009734 | +0.0003136 | 0.0625 |
| `opt.blocks.2.se_scalebias.fc1.weight.velocity` | 4,096 | -1.574e-05 | -0.002654 | +0.00225 | +0.0003897 | 0.0625 |
| `opt.blocks.2.se_scalebias.fc2.bias.velocity` | 256 | -8.626e-05 | -0.001965 | +0.00183 | +0.0005203 | 0.0000 |
| `opt.blocks.2.se_scalebias.fc2.bias.velocity [gamma]` | 128 | -0.000173 | -0.001965 | +0.00183 | +0.0006321 | 0.0000 |
| `opt.blocks.2.se_scalebias.fc2.bias.velocity [beta]` | 128 | +4.835e-07 | -0.001622 | +0.0008488 | +0.0003766 | 0.0000 |
| `opt.blocks.2.se_scalebias.fc2.weight.velocity` | 8,192 | -7.335e-05 | -0.01003 | +0.009769 | +0.001048 | 0.0625 |
| `opt.blocks.2.se_scalebias.fc2.weight.velocity [gamma]` | 4,096 | -0.0001472 | -0.01003 | +0.009769 | +0.001109 | 0.0625 |
| `opt.blocks.2.se_scalebias.fc2.weight.velocity [beta]` | 4,096 | +4.611e-07 | -0.007431 | +0.007387 | +0.0009827 | 0.0625 |
| `opt.policy.conv.bias.velocity` | 76 | -8.763e-07 | -0.01644 | +0.007047 | +0.002915 | 0.0000 |
| `opt.policy.conv.weight.velocity` | 9,728 | +0.0001147 | -0.06441 | +0.03313 | +0.004746 | 0.0000 |
| `opt.policy.pre_bn.bias.velocity` | 128 | -0.0005309 | -0.01625 | +0.005174 | +0.003078 | 0.0000 |
| `opt.policy.pre_bn.weight.velocity` | 128 | -0.0004749 | -0.04212 | +0.01977 | +0.007878 | 0.0000 |
| `opt.policy.pre_conv.weight.velocity` | 16,384 | -5.131e-05 | -0.01583 | +0.01716 | +0.002496 | 0.0000 |
| `opt.stem.bn.bias.velocity` | 128 | +2.997e-05 | -0.003698 | +0.002329 | +0.0008599 | 0.0000 |
| `opt.stem.bn.weight.velocity` | 128 | +0.0001586 | -0.007753 | +0.008267 | +0.002186 | 0.0000 |
| `opt.stem.conv.weight.velocity` | 188,160 | -2.784e-05 | -0.02046 | +0.02009 | +0.001482 | 0.2381 |
| `opt.tower_final_bn.bias.velocity` | 128 | -0.0001905 | -0.006697 | +0.009974 | +0.002256 | 0.0000 |
| `opt.tower_final_bn.weight.velocity` | 128 | +6.519e-05 | -0.03109 | +0.02431 | +0.006336 | 0.0000 |
| `opt.value.bn.bias.velocity` | 16 | +0.000981 | -0.02148 | +0.01193 | +0.007785 | 0.0000 |
| `opt.value.bn.weight.velocity` | 16 | +0.001162 | -0.01705 | +0.01324 | +0.006868 | 0.0000 |
| `opt.value.conv.weight.velocity` | 2,048 | +6.249e-05 | -0.01135 | +0.009738 | +0.0014 | 0.0000 |
| `opt.value.fc1.bias.velocity` | 128 | +9.692e-05 | -0.003545 | +0.003277 | +0.0008611 | 0.0000 |
| `opt.value.fc1.weight.velocity` | 131,072 | +3.401e-05 | -0.006566 | +0.005752 | +0.0003971 | 0.0006 |
| `opt.value.wdl_fc2.bias.velocity` | 3 | +1.287e-05 | -0.01606 | +0.01635 | +0.01323 | 0.0000 |
| `opt.value.wdl_fc2.weight.velocity` | 384 | +9.837e-07 | -0.02166 | +0.02987 | +0.005135 | 0.0000 |

<!-- END GENERATED TABLES -->
