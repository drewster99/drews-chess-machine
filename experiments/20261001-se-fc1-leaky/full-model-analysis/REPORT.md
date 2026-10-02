# Leaky-FC1 run: full-model dead / stuck / always-on analysis (step 23,000)

Every layer and every unit of the leaky-FC1 net, at every 1k checkpoint from fresh to step 23,000. Compared with:
- **ReLU s1:** ReLU scale+bias seed 1, bit-identical starting weights, same steps. Weights only; this build saved no velocity.
- **ReLU s2:** ReLU scale+bias seed 2, a different init, 1k–7,282 steps, velocity saved.

Weights and optimizer state only; nothing was run.

## Headline

- **Nothing in the trunk or the heads is dead, stuck or always-on.** Across all 23 checkpoints, no stem, tower, policy or value conv channel and no BN or LayerNorm channel has zero velocity or a decay-only weight slice.
  - Every BN that feeds a ReLU has |β/|γ|| ≤ 1.03 in all three runs, so every such channel is on for roughly 15–67% of inputs. The bands "dead" (< −3), "mostly off" (< −2) and "always on" (> +3) are empty everywhere.
- **The only exactly dead weights are input-encoding ones.** These are the same in all three runs:
  - the stem weights for the 7 always-zero planes (19, 20, 21, 22, 24, 26, 28);
  - one kernel row of the en-passant plane, which never sees a set square.
- **The SE bottleneck is about half off, in both arms.** Leaky ReLU removed *exact* deadness: 0 zero-velocity FC1 units at all 23 checkpoints, against 4 / 13 / 4 for ReLU s2 at 7,282. It did not add many working units:
  - At 23k, 18 / 17 / 16 of each block's 32 FC1 units sit permanently on the negative side.
  - Their velocity is about 0.5–1.5% of an active unit's, which is what the 0.01 slope alone gives.
  - Units actually used (FC2 input column moved ≥ 5° from init): leaky 14 / 9 / 11, ReLU s1 14 / 10 / 7.
  - 29 of leaky's 34 used units are the same units ReLU uses (block 0: all 14 shared).
  - The only net gain is block 2: 4 extra units, 2 of them used since before 5k. Block 1 trades units both ways: 1 gained, 2 lost.
- **The README's "< 5% of block median: 0 / 0 / 0" is an artifact.** With more than half of each block's units weak, the median unit is itself a weak unit. Against the block's 90th percentile, 15–23 units per block are below 5% at every checkpoint (`se_weak_trend.md`).
- **ReLU-dead SE units do revive.** In ReLU s2, units leave the zero-velocity set between checkpoints, for example block 0 units 22, 30, 5, 25, 28, 12 and block 2 units 0, 10, 12, 15, 17, 23 (`se_relu_s2_dead_turnover.md`).
  - In ReLU s1, block-1 FC1 rows that never moved fell from 13 (1k) to 4 (23k).
  - FC1's input, the pooled conv2 output, keeps drifting, so "a dead SE unit cannot recover" does not hold for this architecture.
- **Heads have data-starved units, not dead ones:**
  - **Policy:** the 6 underpromotion-capture channels sit at 0.3% of the site p90 velocity at every checkpoint, and Q-NW7 at 0.5%. Q-NE7, Q-SW7 and Q-SE7 are at 1–1.5%.
  - **Value FC1:** 28 of 128 hidden units fire so rarely that their WDL output weights get < 1% of p90 velocity (median over 23 checkpoints); all 28 have negative biases.
  - None of these has zero velocity at any checkpoint. The same units are weak in ReLU s1 (Spearman 0.88 on value FC1 movement; policy row movement equal within about 1°).
- **The anomalies are high-energy channels, not dead ones** (identical channel indices in both seed-1 arms):
  - **Stream channel 103** is a near-constant offset: mean −5.1, var 0.04 at block 1's input, 16–20% of the stream's energy.
  - **Channel 7** is the high-variance channel: running variance 56× the median at block 2's input, 23× at the tower-end BN.
  - ReZero α_eff is at 97.9–98.8% of its cap, where sech² attenuates α's gradient to 2.5–4.1%.
- **Hygiene:** no NaN or Inf in any of the 55 checkpoints, optimizer state included. Leaky checkpoints store fp32 masters (≤ 0.8% of values on the bf16 grid); ReLU s1 stores bf16 working weights (100%).
- **The two arms are still close.** Per tensor at 23k, the cosine between (leaky − fresh) and (ReLU − fresh) is 0.56–1.00 (median 0.96). Tower convs are 0.73–0.79; the lowest are block-1 SE FC1 bias (0.56), block-2 LayerNorm β (0.65) and block-2 bn2 β (0.70).

## Per-layer counts at the latest step

- Columns are leaky (23k) / ReLU s1 (23k) / ReLU s2 (7,282).
- `-` means not measurable (no velocity saved); `·` means zero in all three.
- Full table, every row: `results/counts_latest.md`. Definitions are under Method.
- "weak" for scalar or bias sites in a single snapshot is mostly sign-crossing noise; use `persistent_low_velocity.md` for those.

| layer (site) | units | unmoved / unchanged | vel0 | weak (< 5% p90) | dead / mostly-off / always-on BN | rv outlier | notes |
|---|---:|---|---|---|---|---|---|
| stem conv, output rows | 128 | · | · | · | n/a | n/a | row norms 1.22–3.13 |
| stem conv, input planes | 30 | 7 / 7 / 7 | 7 / - / 7 | 9 / - / 8 | n/a | n/a | planes 19–22, 24, 26, 28; weak: also 25, 29 |
| stem BN (no activation; pre-act tower) | 128 | · | · | · | n/a (no ReLU) | 0 / 0 / 1 | min γ 0.41 (ch 103) |
| block 0–2 bn1 (→ ReLU) | 3×128 | · | · | 1 / - / 4 total | · | b2: 1 / 1 / 0 (ch 7) | β/\|γ\| −1.03…+0.18 |
| block 0–2 conv1 out / in | 3×128 each | · | · | · | n/a | n/a | velocity min 0.09× median |
| block 0–2 bn2 (→ ReLU) | 3×128 | · | · | 5 / - / 2 total | · | · | β/\|γ\| −0.37…+0.13 |
| block 0–2 conv2 out / in | 3×128 each | · | · | · | n/a | n/a | |
| SE FC1 rows, b0 / b1 / b2 | 32 each | 1/2/0 · 2/4/0 · 1/5/2 | 0/0/0 · - · 4/13/4 | 18/17/16 · - · 15/17/17 | n/a | n/a | slices are leaky · ReLU s1 · ReLU s2 |
| SE FC2 γ-columns (read FC1 unit), b0/b1/b2 | 32 each | 11/6/0 · 10/10/5 · 10/13/9 | 0 · - · 4/13/4 | 19/22/22 · - · 19/21/22 | n/a | n/a | |
| SE FC2 β-columns, b0/b1/b2 | 32 each | 0/2/0 · 4/7/3 · 5/8/4 | 0 · - · 4/13/4 | 19/21/20 · - · 18/20/21 | n/a | n/a | |
| SE FC2 output rows (γ / β halves) | 3×256 | · | · | 32 / - / 30 total | n/a | n/a | weak-gated γ outputs persist in b0 (11 rows) |
| ReZero α | 3 | · | · | · | n/a | n/a | eff 0.438 / 0.440 / 0.442 (cap 0.4472) |
| block LayerNorm (→ stream) | 3×128 | · | · | 11 / - / 10 total | n/a | n/a | γ 0.66–1.55, \|β\| ≤ 0.57; no γ ≈ 0 |
| tower-end BN (→ ReLU → both heads) | 128 | · | · | 4 / - / 2 | · | 1 / 1 / 0 (ch 7, 23×) | β/\|γ\| −0.15…+0.45 |
| policy pre_conv out / in | 128 / 128 | · | · | · | n/a | n/a | |
| policy pre_bn (→ ReLU) | 128 | · | · | 12 / - / 8 | · | · | ch 2, 116 persistently low |
| policy conv rows (76 move channels) | 76 | · | · | 22 / - / 21 | n/a | n/a | rare move types (see finding 4) |
| policy conv input columns | 128 | · | · | 3 / - / 2 | n/a | n/a | |
| value conv out / in | 16 / 128 | · | · | · | n/a | n/a | |
| value BN (→ ReLU) | 16 | · | · | 0 / - / 1 | · | · | β/\|γ\| −0.46…−0.25: all channels on 32–40% |
| value FC1 hidden units | 128 | · | · | 13 / - / 23 | n/a | n/a | 0 dead (channel, square) inputs, all checkpoints |
| value FC1 → WDL columns | 128 | · | · | 44 / - / 52 | n/a | n/a | measures hidden-unit firing; see finding 5 |
| value WDL FC2 out / bias | 3 / 3 | · | · | · | n/a | n/a | |

## Trends (fresh → 23k)

Tables: `counts_trend.md`, `se_units.md`, `se_weak_trend.md`, `rezero.md`; per-step CSV `counts_by_step.csv`.

- **SE usage settles by 5k** and barely changes after that. Used / trickle / unused, leaky vs ReLU s1:

  | block | 1k | 5k | 23k |
  |---|---|---|---|
  | b0 | 13/3/16 vs 11/4/17 | 14/2/16 vs 14/1/17 | 14/2/16 vs 14/3/15 |
  | b1 | 6/4/22 vs 6/3/23 | 8/3/21 vs 8/2/22 | 9/5/18 vs 10/2/20 |
  | b2 | 9/3/20 vs 7/5/20 | 9/3/20 vs 7/7/18 | 11/3/18 vs 7/7/18 |

  - Block 1 gains units from 15k in both arms.
  - Block 2's leaky-only units 10 and 29 crept past 5° only after 20k.
- **Leaky weak-unit count is flat:** 15–18 / 13–23 / 12–19 per block at every checkpoint, never zero. ReLU s2 zero-velocity counts move between 3–5 / 5–13 / 3–7.
- **"Unmoved since init" FC1 rows fall in both arms** (input drift revives units). Block 1: leaky 10 → 2, ReLU 13 → 4. Block 2: 5 → 0 vs 2 → 0, by 5k.
- **Dead input planes:** the 7 dead planes stay exactly zero-velocity and decay-only at every checkpoint in both arms (decay factor ×0.803 at 23k). Plane 29 ("10 plies ago") falls to 0.6% of the plane median velocity in 77% of checkpoints.
- **ReZero α rises in both arms**, accelerating after 15k (the LR cycle). The gradient scale on raw α falls from 0.42 at init to 0.025–0.041 at 23k.
- **Hot channels:** block-2-input channel 7's variance ratio and stream channel 103's energy share are present from about 5k and grow slowly. The same indices hold in both seed-1 arms, so the init decides them.

## Ranked findings

1. **SE bottleneck: about half off in both arms; leaky changes "dead" to "trickle", not "off" to "used".**
   - Tensors: `blocks.{0,1,2}.se_scalebias.fc1.weight` / `.bias` and the FC2 columns reading them.
   - Leaky units with persistently < 5% of p90 velocity (in ≥ 50% of 23 checkpoints): 17 / 16 / 18.
     - Block 0: units 2, 3, 5, 7, 8, 10, 11, 13, 15, 16, 17, 21, 26, 27, 28, 30, 31.
     - Their median velocity is 0.5–1.4% of p90, the leaky-slope-only gradient.
     - Their FC2 γ/β columns get 0.2–1% of p90, and 11 / 6 / 0 γ-columns are still decay-only.
   - Why it matters: those units' SE output is essentially constant, `sigmoid(bias + small)`, so the block's channel attention runs on about 9–14 units.
   - ReLU shows the same:
     - ReLU s2 has 15 / 17 / 17 weak, of which 4 / 13 / 4 are exactly zero.
     - ReLU s1 uses the same units; block 0 sets are identical.
   - Leaky-only working units: block 2 units 5, 30 (≥ 19° at 23k vs 2.6–2.8° in ReLU, diverged before 5k), 10, 29, and block 1 unit 27. ReLU-only: block 1 units 10, 23.
   - Files: `se_units.md` (every unit), `se_weak_trend.md`, `se_relu_s2_dead_turnover.md`.

2. **Exactly dead input weights (encoding, both arms, expected).**
   - `stem.conv.weight[:, p]` for p in {19, 20, 21, 22, 24, 26, 28}: cosine with fresh 1.000000, norm ratio equal to the decay factor (±1e-6), velocity exactly 0 at every checkpoint.
   - `stem.conv.weight[:, 16, dy=+3, :]` (en passant, 7 offsets): zero velocity for all 128 output channels at every checkpoint. EP squares sit on one rank in the encoder frame, so offset dy = +3 never reads them.
   - Nearly dead inputs: planes 25, 27, 29 (6, 8, 10 plies ago) at 1–6% of p90, cosine 0.995–0.9991 after 23k steps. These are rare repetition patterns.
   - No other plane is decay-only or unmoved. Piece planes have moved to cosine 0.38–0.56, castling 0.86, halfmove 0.92.

3. **Stream channel 103 is a constant offset and channel 7 a variance hog** (same in both arms; init-determined).
   - Channel 103 at the input to `blocks.1.bn1`: running_mean −5.10, running_var 0.042 (|mean|/std ≈ 25), 17.9% of the input energy (uniform 0.78%). ReLU s1 is the same: −5.47, 0.050, 19.8%.
   - It is pinned by the block LayerNorm: `blocks.0.res_ln.bias[103]` = −0.574 is the largest |β| of any LN channel.
   - It also has the smallest stem-BN γ (0.414) and the most-off BN-ReLU channel in the tower: `blocks.2.bn1` β = −2.08, γ = 2.02, β/|γ| = −1.03, P(on) ≈ 15%.
   - Why it matters: about a sixth of each position's LN normalisation budget is spent on a near-constant feature. It isn't dead: the next BN re-standardises it.
   - Channel 7: running variance 56× the median at the input to `blocks.2.bn1`, 23× at `tower_final_bn`, 9.5% of the tower-end energy. In TENSOR-STATS it was the 33k hot channel.
   - Other near-constant stream channels (|mean|/std > 3): 23 at block 1's input, 6 at block 2's, 4 at the tower end (ReLU s1: 15 / 8 / 5).

4. **Policy head: rare move types are data-starved, not dead.**
   - `policy.conv.weight` rows and biases for UP-{knight, rook, bishop}-cap{L,R} (channels 65, 66, 68, 69, 71, 72): median velocity 0.31–0.35% of p90 (weak in 96–100% of checkpoints).
     - Their rows moved 6–10° early and now shrink at the pure decay rate (norm/decay 0.98–1.00).
     - Their biases are negative (−0.018 to −0.053).
   - Q-NW7 is at 0.5%; Q-NE7, Q-SW7 and Q-SE7 at 1–1.5%; underpromotion-forward and the 6-square diagonals at 1.5–3%.
   - ReLU s1 is identical within about 1° of row movement.
   - `policy.pre_bn` channels 2 and 116 are persistently low-velocity (0.08–0.12× median at 23k), with normal γ, β and column norms. They are lightly used, not dead.
   - Why it matters: those logits are effectively frozen priors and will only learn when the corpus serves those moves.
   - File: `policy_channels.md`.

5. **Value head: heavy-tailed hidden-unit usage, no dead units.**
   - All 128 `value.fc1` units have nonzero velocity at every checkpoint. No (value-channel, square) input of FC1 has an all-zero velocity column at any checkpoint in either velocity run.
   - Unit firing, measured by the velocity of the 3 WDL weights reading the unit: 28 units < 1% of p90 and 47 < 5% (median over 23 checkpoints); 34 weak in ≥ 90% of checkpoints.
   - All 28 near-silent units have negative FC1 biases. Their FC1 rows moved only 1.8–9° (unit 101: 1.8°, WDL-column velocity 0.09% of p90).
   - ReLU s1 has the same pattern (Spearman 0.88 on FC1 row movement). The arms differ on a few units, for example unit 21: leaky 4.0°, ReLU 23.0°.
   - `value.bn` β/|γ| ranges −0.46 to −0.25: all 16 channels are on 32–40% of the time.
   - Why it matters: roughly a quarter of the 128-unit hidden layer contributes almost nothing to W/D/L.
   - File: `value_hidden_units.md`.

6. **ReZero α is close to its tanh cap.**
   - `blocks.{0,1,2}.rezero_alpha` raw 1.019 / 1.064 / 1.136 gives α_eff 0.4379 / 0.4396 / 0.4417 against the cap 0.4472136. d(eff)/d(raw) is 0.041 / 0.034 / 0.025.
   - ReLU s1 is the same: 1.070 / 1.070 / 1.133.
   - Not stuck in the zero-gradient sense, but α's effective learning rate is 25–40× smaller than at init.

7. **SE FC2 output gates that barely move.**
   - Block 0 γ-half outputs (gamma channels 7, 17, 25, 28, 42, 52, 73, 87, 91, 114, 115, 118, 126) are weak in ≥ 50% of checkpoints; channels 25, 87 and 114 are at 1.4–2.2% of p90.
   - These channels' attention gate is close to constant. This follows from finding 1.

8. **Tensor-level hygiene.**
   - 0 NaN or Inf in 55 checkpoints, including optimizer state.
   - Largest |parameter| at 23k: `blocks.2.bn1.bias` 2.08 (both arms), `stem.conv.weight` 2.02, `blocks.1.bn1.weight` 2.02.
   - Largest velocity spread is in SE FC2 columns: max/median 256–360× in leaky versus 375–2,314× in ReLU s2. The gap is bimodality from the off units, not instability. Every other site is ≤ 10–15×.

## Method and definitions

- **Identity:** every checkpoint is identified by safetensors `__metadata__`. Each run's trained files share one ModelID; files are keyed by `training_step`. `-replay-latest` files are excluded. See `results/manifest.csv`.
- **Architecture, verified in `ChessNetwork.swift`:**
  - stem conv 7×7 30→128 → BN, with no stem activation (`hasStemActivation` is false for a pre-act first block);
  - each block: BN1 → ReLU → conv1 → BN2 → ReLU → (dropout slot) → conv2 → SE (pool → FC1 → `se_activation` → FC2 → γ/β split → sigmoid(γ)·z + β) → ×C·tanh(α/C) → + identity skip → LayerNorm (per position, over channels);
  - tower-end BN → ReLU;
  - policy: pre_conv 1×1 → pre_bn → ReLU → conv 1×1 + bias;
  - value: conv 1×1 128→16 → BN → ReLU → flatten (c·64 + square) → FC1 128 → ReLU → FC2 3.
- **Optimizer facts used:**
  - `v ← μv + clip·g`; weight decay is decoupled and never enters the velocity.
  - Only conv and FC weights are decayed; BN/LN γ/β, biases and α are not.
- **Layouts:**
  - Saved weights are conv OIHW and linear `[out, in]`.
  - Velocity is conv OIHW and linear `[in, out]` (graph layout). Verified two ways; see `results/layout_check.md`.
- **Unit slicings ("sites"):**
  - conv output rows (`.out`) and input columns (`.in`, "is this input still read");
  - BN/LN channels;
  - SE FC1 rows, and FC2 columns per FC1 unit split into γ and β halves;
  - FC2 output rows;
  - value FC1 rows, plus FC1 columns grouped per value-conv channel and per (channel, square);
  - policy rows per move channel.
- **Flags:**
  - **unmoved** (decayed slices): cosine with fresh > 0.999995, and norm ratio within 0.2% of the decay-only factor. The factor is measured on the always-zero-plane stem weights: ×0.803 at 23k for both seed-1 arms.
  - **unchanged** (undecayed parameters): bit-identical to fresh.
  - **vel0:** every velocity element of the unit is exactly 0, meaning zero gradient long enough for momentum to underflow.
  - **weak:** unit velocity norm < 5% of the site's 90th percentile. The p90 reference replaces the median one because the median is itself a weak unit in a bimodal site.
  - **vel_low / vel_high:** < 5% / > 10× of the site median, kept for continuity with the README.
  - **persistent:** weak in ≥ 50% of the 23 leaky checkpoints.
  - **BN followed by ReLU:** dead β/|γ| < −3, mostly off < −2, always on > +3; P(on) = Φ(β/|γ|) under a Gaussian-input approximation. Velocity remains the ground truth.
  - **rv_high / rv_low:** running variance > 20× / < 0.05× the BN's median.
  - **ln_gamma_small:** LayerNorm |γ| < 0.1.
  - **near-constant channel:** |running_mean|/√running_var > 3.
  - **energy share:** (var + mean²) / Σ over channels.
- **SE usage class** (all runs, weights only): the angle between a unit's 256-weight FC2 input column and its fresh value. That column's gradient is proportional to the unit's output, so the angle integrates use: used ≥ 5°, trickle 1–5°, unused < 1°.
- **Limits:**
  - ReLU s1 has no velocity, so only weight-based flags apply to it.
  - Velocity is a short momentum average: single-snapshot "weak" for scalar parameters is noisy, so rely on the persistence table.
  - β/|γ| ignores the non-Gaussian, per-square structure of the BN inputs.
  - Weights alone cannot give exact ReLU-on fractions.

## Reproduce

```
cd experiments/20261001-se-fc1-leaky/full-model-analysis/scripts
sh run_all.sh          # layout_check.py, units.py, input_features.py, summarize.py
```

- Python 3 with numpy and pandas; CPU only.
- Reads `~/Library/Application Support/DrewsChessMachine/Models/`:
  - `20261001-test_SE_scale+bias-fc1leaky-{fresh,replay-step*}`
  - `20260929-test_SE_scale+bias-{fresh,replay-step*}`
  - `20260929-test_SE_scale+bias-seed2-{fresh,replay-step*}`
- Uses whatever latest leaky step exists at run time; this report is step 23,000.
- Policy channel labels come from `documentation/research/policy-head-2026-10-01/scripts/policy_head_lib.py`.
- All outputs are written to `../results/`; `run_all.log` holds the last run's console output.
