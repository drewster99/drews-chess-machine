# 2026-06-08 — 5-block 256-wide dual-kernel (7×7 + 3×3) tower on `full10Ply10Reps210`

**Status:** done (stopped by manual save at step 13,861 on 2026-06-08 21:58 CDT. A 10-minute resume, 23:19–23:29, ran 13,861 → 13,996 on build 1783 and was not continued. The run is kept as the reference case of the lr-1e-2 out-of-distribution blow-up.)

Migrated from `documentation/ARCH_EXPERIMENTS.md` Experiment 4 and audited 2026-09-29. The original text is kept. Corrections are marked `~~old~~ → new` or `[Audit: …]`, and all are listed in Audit notes.

## Question

This is the **first tower change in the series.** Experiments 1–3 held a fixed 5-block / 7×7 / 128-ch tower and varied only the input encoding. Experiment 4 keeps Exp 3's `full10Ply10Reps210` input but **scales the tower**, as a direct (partial) implementation of Exp 1/3 §7's "more channels at 3×3 kernels" and "shrink the over-wide stem" advice. It is therefore **not** a controlled step in the encoding ladder. It is an uncontrolled jump:
- tower width ×2
- the block's second conv 7×7→3×3
- a smaller stem kernel
- a wider value FC
- 2.1× the params

It is also a single seed, so read its deltas vs Exp 3 as suggestive, not attributable.

The literal preceding fresh build that day was a `stem 512 / single-5×5 / 67.9M`-param probe (`gViN`, build 1781), abandoned after ~822 steps. Exp 4 is the configuration that was kept. [Audit ✓: `dcm_log_20260608-115907.txt`, `Build Network (v4 pre . in full10Ply10Reps210(210) -> stem 512 (3x3) . 5x[5x5 conv, SE+/4, clean_add, ReZero] . act relu . policy intermediate_conv(4864) . value WDL(16->FC256) . bfloat16 . 67,940,116 params)`, last `[STATS]` steps=822. "single-5×5" means both block convs are 5×5; the summary collapses equal kernels.]

## Setup

- **Architecture** (`[ARCH] built champion 20260608-4-jaq1`; confirmed against the embedded JSON in the surviving `cwkO-manual` `champion.safetensors`: `channels 256`, `stem_conv_kernel_size 3`, `block_conv1_kernel_size 7`, `block_conv2_kernel_size 3`, `block_se_reduction_ratio 2`, `value_head_hidden_units 256`, `rezero_alpha_init 0.447214`):
  `v4 pre . in full10Ply10Reps210(210) -> stem 256 (3x3) . 5x[7x7,3x3 conv, SE+/2, clean_add, ReZero] . act relu . policy intermediate_conv(4864) . value WDL(16->FC256) . bfloat16 . 20,349,716 params`
- **Builds:** 1781 (fresh, git `08a74a5*`) → 1782 (resumed, main run) → 1783 (final short resume).
- **Logs:**
  - `dcm_log_20260608-133428.txt` (build 1781, steps 0→509 saved, `[STATS]` to 520)
  - `dcm_log_20260608-140857.txt` (build 1782, 509 → 13,861 saved, `[STATS]` to 13,868; the main run)
  - `dcm_log_20260608-231918.txt` (build 1783, 13,861 → 13,996, 23:19–23:29 CDT, no arenas)
- **Dates:** 2026-06-08, ~13:35 → 21:58 CDT (~8.4 h), plus the 10-minute tail.
- **Machine:** local (logs on this Mac). The hardware is not logged.
- **Training config:** constant LR **1e-2**, weight_decay **1e-4**, grad_clip 30, μ 0.90, entropy_bonus 0, draw_penalty 0; batch 4096, 800 self-play workers, `spDelay=3000ms`, promote ≥ 0.53 ~~(later 0.55)~~, 400-game arenas every 900 s. [Audit: every `[STATS]`/`[ARENA]` line in all three logs reads `promote>=0.53`. No 0.55 appears.]

### 1. Architecture (original §1)
- **Input:** **210 planes** × 8×8 (NCHW), `full10Ply10Reps210`, identical to Experiment 3 (10 stacked `basic20` frames plus the 10 `basic30` temporal-repetition planes 200–209). **Policy** 4864 logits. **Value** 3-class W/D/L head.
- **Stem:** **3×3** conv, 210 → **256** (Exp 3 was 7×7 → 128).
- **Tower:** **5** pre-activation residual blocks, **256 ch**. Each block has **two convs** (a two-conv residual block, as in Exp 1–3: `blockConv1KernelSize` + `blockConv2KernelSize`). The **second conv is now 3×3 instead of 7×7**, so block kernels are **7×7 → 3×3**. Exp 1–3 were **7×7 → 7×7**, which the summary collapses to "7x7" because the two are equal. After the convs come a scale-and-bias **SE (reduction /2)** and a clean identity add scaled by a per-block ReZero α (`1/√5`). Activation ReLU. The tower still has 10 convs (5 blocks × 2), same as Exp 3; what changed is the second kernel (7×7→3×3) and the channel width.
- **Policy head:** `intermediate_conv` → 4864 (unchanged). **Value head:** 1×1 conv → 16 → BN/ReLU → flatten(1024) → FC 1024→**256** → ReLU → FC 256→3 (W/D/L), categorical CE (head width 128→256).
- **Precision:** bfloat16. **Params:** **20,349,716 (~20.35M)**, ~2.1× Exp 3's 9.57M. **Arch version:** v4. [Audit ✓: the header tensor-shape sum is 20,349,716, and 20,349,716 / 9,574,708 = 2.13.]
- *Context:* depth was **not** increased (still 5 blocks), and the stem was **not** taken all the way to 1×1. The advice was taken on two axes (more channels, and a move toward 3×3 by shrinking the block's **second** conv from 7×7 to 3×3) and partially on a third (7×7→3×3 stem). The block's **first** conv stays 7×7, consistent with Exp 1's forensic finding that the tower uses its full 7×7 reach, so each block now pairs one wide (7×7) and one cheap (3×3) conv instead of two 7×7s.
- **Optimizer / regularization (unchanged from Exp 1–3):** constant LR **1e-2**, weight_decay **1e-4**, grad_clip 30, μ 0.90, entropy_bonus 0, draw_penalty 0; batch 4096, 800 self-play workers, promote ≥ 0.53 ~~(later 0.55)~~, 400-game arenas.

## Runs

- **Live lineage:** champion `20260608-4-jaq1` → `jaq1-4`. The final trainer is `jaq1-5`. Promotions fork the ID. **Saved-session lineage:** `20260608-5-cwkO`.
- **Arenas:** **29** (#1 at step 678 → #29 at 13,832, all in `-140857`), with **4 promotions**. After the #18 promotion the trainer lost all 11 remaining arenas (0.1613–0.4900).

### 2. Relevant saved sessions (original §2, audited)

`.dcmsession` autosaves (safetensors-native; saved lineage `cwkO`, live champion `jaq1`→`jaq1-4`). Session filenames are **UTC**-stamped. The steps below are each session's `trainingSteps`, matching the four arena promotions plus the final manual save (CDT times in parentheses).

| Saved session (`.dcmsession`) | Step @ snapshot | Trigger | On disk 2026-09-29 |
|---|--:|---|---|
| `20260608-190547-20260608-5-cwkO-manual` | 509 | manual (build 1781 → restart on 1782) [Audit: added; from `-133428` `[CHECKPOINT]`] | pruned |
| `20260608-201157-20260608-5-cwkO-promote` | 1,268 | promote → `jaq1-1` (arena #3, 15:11 CDT) | pruned |
| `20260608-205859-20260608-5-cwkO-promote` | 2,838 | promote → `jaq1-2` (arena #6, 15:58 CDT) | pruned |
| `20260608-220114-20260608-5-cwkO-promote` | 4,988 | promote → `jaq1-3` (arena #10, 17:01 CDT) | pruned |
| `20260609-000607-20260608-5-cwkO-promote` | 9,091 | promote → `jaq1-4` (arena #18, +68 Elo, 19:06 CDT) | pruned |
| `20260609-025811-20260608-5-cwkO-manual`  | ~~~13,868~~ → **13,861** | manual (final, 21:58 CDT; the best champion is the prior `jaq1-4`) | **present** (5.0 GB) |

Each `champion.safetensors` embeds the architecture in `__metadata__` (`input_encoding: full10Ply10Reps210`, `training_step`, `content_sha256`). For the surviving manual save, the champion is `20260608-4-jaq1-4` (step 13,861, `content_sha256` 3aa8b85f…, 20,349,716 params) and the trainer is `jaq1-5` (parent `jaq1-4`; 40,692,744 stored values = weights plus velocity). `session.json` holds 29 arenas and 4 promotions (1,268 / 2,838 / 4,988 / 9,091). Location: `~/Library/Application Support/DrewsChessMachine/Sessions/`.

The keeper checkpoint named in §6, `20260609-000607-…-cwkO-promote` (step 9,091), **no longer exists**, and neither does the proposed resume point `jaq1-3` (4,988). The surviving manual save's *champion* is still `jaq1-4`, with the same weights as promoted at 9,091, because champions do not train. So the best net of the run survives, inside the 13,861 manual save.

## Results

### 3. Factuals (original table)
Wide-set (4,435 puzzles) is the cross-experiment default, with the 200-set alongside (high variance). ~~`vAbs`/`pD`/`draw%` are champion self-play~~ → [Audit: **`draw%` and game length are champion self-play; `vAbs`/`pD` are trainer-side batch statistics** (the trainee's value head on replay-buffer positions).] **The probe NLL is reliable only through ~step 5,000.** It blows up after that (see Shortcomings), which also leaves `pElo` (rank-based, more robust) as the only usable tactical signal from then on. pElo is the **June 2026 in-app scale**. The selfplay_registry endpoint 540.3 / NLL 14.7415 is a later-binary re-probe and is not comparable.

| Step | pElo (wide) | NLL (wide) | pElo (200) | NLL (200) | vAbs | pD | draw% | Detail |
|--:|--:|--:|--:|--:|--:|--:|--:|---|
| 521 | 477 | 4.17 | 588 | 4.58 | ~0.085 | 0.72 | 84% | Run start (fresh 20.35M net, champion `jaq1` / trainer `jaq1-1`). Constant LR 1e-2, wd 1e-4. [Audit ✓: probe 477 / 4.170 / 588 / 4.582. `[STATS]` 522: pD 0.713, vAbs 0.093, D 82.5%.] |
| 1,268 | 563 | 3.71 | — | — | 0.089 | 0.72 | 87% | Promotion #3 → `jaq1-1`. Tactical climbing cleanly. [Audit: probe at 1,275 is 571 / 3.717. At 1,299: pD 0.715, vAbs 0.092, D ~~87%~~ → 82.3%.] |
| 2,838 | 613 | 3.78 | — | — | 0.097 | 0.75 | 87% | Promotion #6 → `jaq1-2`. [Audit: probe at 2,826 is 624 / 3.753. At 2,897: pD 0.736, vAbs 0.102, D 85.5%.] |
| ~5,000 | **635** | **3.78** | — | — | 0.088 | 0.76 | 90% | Promotion #10 → `jaq1-3`. **Peak clean tactical state**; 3 promotions and wide pElo +158 in the first ~4.5k steps. [Audit: probe at 5,000 is 626 / 3.817 and at 5,025 is 634.7 / 3.780. The clean-phase pointwise maximum is **640.7 at step 2,650**. At 5,046: pD 0.760, vAbs 0.093, D ~~90%~~ → 85.8%.] |
| ~5,500 | ~590 | **5.6 → 9.9** | — | — | — | — | — | **Probe-NLL blow-up onset.** Wide NLL leaves ~3.8 and never returns; it climbs to 15–17 over the next ~2k steps. pElo begins oscillating ~~370–665~~ (see audit). [Audit ✓: wide NLL 4.10 (5,475) → 4.73 (5,550) → 5.62 (5,626) → 9.49 (5,701) → 9.85 (5,776). Wide pElo after 5,400 (through 13,868) spans 328–780, with 5th–95th percentiles 367–628.] |
| 9,091 | ~550 | ~14 (junk) | — | — | 0.086→**0.149** | 0.73→**0.51** | 86%→**59%** | Promotion #18 → `jaq1-4`, arena **+68 Elo** (the run's only large win). **The champion's value head turns decisive here**: by ~step 9,410 vAbs 0.086→0.15, pD 0.73→0.51, draws 86%→59%, and mean self-play game length **~280→~122 plies (halved)**. The champion (`jaq1-4`) is the best net of the run. [Audit: the probe at 9,100 reads 525 / 6.45. `[STATS]` 8,957 → 9,410: pD 0.732 → 0.508, vAbs 0.086 → 0.142, D 85.7% → 58.9%, avgLen 283 → 123 ✓. The draw/length change is champion self-play. The pD/vAbs change is the trainer reading the new, more decisive buffer.] |
| 13,868 | 623 (junk NLL) | 13.2 | 727 | 13.7 | 0.161 | 0.50 | 57% | **Final step** (manual save, run stopped). 29 arenas / **4 promotions** total. The trainer (`jaq1-5`) **lost every arena after #18** (scores 0.16–0.49, Elo to −286), with zero further promotions across the last ~4.8k steps. [Audit ✓: probe 13,850 is 623 / 13.207 / 727 / 13.705. `[STATS]` 13,854: pD 0.498, vAbs 0.161, D 57.8%. Arenas #19–#29 scored 0.1613–0.4900, Elo −7 to −286. The save step is 13,861. The build-1783 resume then reached 13,996 (wide 659 / 11.75 at 14,000).] |

*Training-distribution telemetry stayed healthy the entire run: `pEnt` 2.65→2.49, `gNorm` ~1.3–3.1, `pLogitAbsMax` ~13.7 (flat), `pwNorm` 12.7→14.0 (mild), no NaN, legal-mass probe steady 0.85–0.89. The blow-up is **invisible** on `[STATS]`; only the out-of-distribution Lichess probe NLL and the arena reveal it.* **[Audit: this paragraph is wrong on the load-bearing point.** The **legal-mass probe was not steady**. `[ALARM] legal-mass probe ok` read 0.83–0.96 through ~17:30 CDT (≈ step 6k), then fell to **0.015–0.71** for the rest of the run (e.g. 0.234 at 17:46, 0.015 at 19:33, 0.148 at 21:52), and the alarm still reported "ok". On `[STATS]`, `legalMass` fell from ~0.95 (steps 3–5.5k) to 0.02–0.61 after ~6.2k, and **`pIllM` rose from 0.035 to 0.12–0.23**. The blow-up is therefore **visible on `[STATS]`** (legalMass / pIllM) from ~step 6k, about the same time as the probe NLL. The rest of the list is roughly right: pEnt 2.48–2.77 (p5–max), gNorm p5–p95 1.28–3.77 (max 5.66), pLogitAbsMax 12.4–15.7, pwNorm 12.63→14.03. `[STATS]` shows no NaN.]

## Conclusion

### 4. Wins (original)
- **Value head turns decisive, strongly.** At the `jaq1-4` promotion (~step 9k) the champion's self-play went from ~86% draws / ~280-ply shuffling to **57% draws / ~122-ply decisive games**, with vAbs ~0.086→0.16 and pD 0.73→0.50. This is the clearest decisive-value transition of any experiment so far, and the engine genuinely started converting advantages.
- **Fast, clean early tactical bootstrap** through ~step 5k: wide pElo **477→635** (+158) with 3 promotions and wide NLL falling 4.17→3.78. That is a better early tactical slope than Exp 3's 128-ch tower on the same input (≈616 wide at 5k). [Audit: Exp 3's 4–6k wide bucket mean is 616.4 ✓. Exp 4's 4.5–5.4k mean is 611.8 / 3.894, so the advantage is in the pointwise peaks (626–641), not the bucket mean.]
- **No in-distribution instability.** On its own self-play distribution the 2.1×-capacity net was stable in bf16 (entropy, gradient norm and logit max all well behaved), so the extra capacity did not cause training-loop divergence. [Audit: qualified. There was no NaN and no loss divergence, but the policy's **legal mass collapsed** (see §3 audit), so "in-distribution" health held only for the loss-side metrics.]

### 5. Shortcomings (original)
- **Catastrophic out-of-distribution calibration blow-up from ~step 5,400.** Wide-set probe NLL exploded **3.78 → 8–17** and stayed pinned there for the final ~8k steps (the 200-set likewise). Meanwhile rank-based **pElo only oscillated** (~~370–665~~ → 328–780, 5–95% 367–628). The policy still *ranks* tactical moves roughly as before but assigns **pathologically peaked, confidently wrong** distributions on positions it doesn't generate in self-play. The arena confirms the over-sharpening from the other side (arena #28: candidate played-move prob ≈ **0.97** every position, value ≈ 0). [Audit ✓: arena #28 (step 13,636) "Value + score by ply" buckets show pol 0.932–1.000 (0.97 typical), v +0.000…+0.003.]
- **Trainer-lineage strength regression.** After the `jaq1-4` promotion the trainer (`jaq1-5`) was weaker than the frozen champion in **every** subsequent arena (scores 0.16–0.49, Elo −7 to −286) and earned **no further promotions**. The last ~4.8k steps were net-negative for the trainer even though the champion was fine.
- **The failure is invisible to the standard health metrics.** `pEnt`/`gNorm`/`pLogitAbsMax`/`pwNorm` all looked healthy throughout, so a run watched only through `[STATS]` and the entropy / draw-rate alarms would read as fine. Only the manual tactical probe and the arena caught it. [Audit: **partly wrong.** `legalMass` and `pIllM` on `[STATS]`, and the legal-mass probe values, showed it clearly from ~6k. What failed was the **alarm**: `legal-mass probe ok` kept printing at legalMass 0.015.]
- **No tactical-ceiling verdict.** Clean wide pElo never cleanly exceeded ~635 before the NLL contamination, so this run cannot be compared on ceiling to Exp 1 (~879 wide). It is also a single seed on a brand-new tower, so deltas vs Exp 3 are not attributable. [Audit: bzw3's wide pElo is the same June in-app scale. `selfplay_probe/bzw3.csv` peaks at 924.4 pointwise (step 426,377); the ~879 figure is a smoothed value whose method was not recorded.]

### 6. Analysis (original)
- **The signature finding is the split:** healthy training-distribution metrics and a decisive champion value head, but an exploded out-of-distribution probe NLL and a self-degrading trainer. The net learned to play its own (increasingly narrow, decisive) self-play lines extremely confidently while becoming catastrophically mis-calibrated everywhere else. [Audit: per §3, the "healthy training-distribution metrics" half of the split does not hold for legal mass or illegal mass.]

One of these is probably the driver, possibly both:
- **Hypothesis #1: self-play distribution narrowing.** As the champion sharpened (draws 86%→57%, games halving in length), the replay buffer concentrated on a narrow band of lines; the trainer over-specialized to them and went OOD on tactical puzzles. (`diverge≈1.8` with 100% unique games is *in* the nominally healthy band, so this is not an obvious diversity collapse. It needs the diversity-histogram and per-frame stem-norm checks to confirm.) [Audit ✓: `diversity=unique=200/200(100%) diverge=1.4–1.9`. Also note that **the NLL blow-up (~5.5k) and the legal-mass collapse (~6k) both precede the 9,091 promotion** that shortened the games, so narrowing caused by the decisive champion cannot be what started it.]
- **Hypothesis #2: under-regularization at 2.1× capacity.** wd 1e-4 with a constant 1e-2, the settings that held the 9.5M nets, may simply be too weak for a 20.3M policy. That is the "weights/logits grow unbounded at wd 1e-4" concern Exp 1 §6 flagged. **A caveat complicates H2:** `pLogitAbsMax`/`pwNorm` did **not** inflate on the training distribution here (unlike Exp 1's saturated-net over-sharpening), so any over-sharpening is *distribution-specific*, not a global logit run-away. That fits H1 better than a plain weight-norm blow-up.
- **The champion is genuinely the best net of the run** (`jaq1-4`, step 9,091: decisive value head, won its arena +68). The regression is the **trainer lineage diverging**, not the champion degrading. The keeper checkpoint is `20260609-000607-…-cwkO-promote` (step 9,091). [Audit: that folder is pruned. The same `jaq1-4` champion weights survive as the champion of the 13,861 manual save.]
- **Cleanest disambiguator:** resume from the pre-blow-up `jaq1-3` checkpoint (step 4,988) with **wd ≈ 3e-4** and/or a one-shot LR anneal, and watch the **probe NLL**. If it stays bounded it was regularization (H2); if it still explodes while training metrics stay clean it was distribution narrowing (H1). [Audit: the `jaq1-3` checkpoint (4,988) is pruned, so this exact experiment is no longer possible.]

> **Conclusion (folded from overnight investigation 2026-06-11):** the wider Track-2 sweep (8 lineages, 2026-05-14→06-10, plus a retro-probe of all 80 saved checkpoints) eliminated capacity, width, depth, params and code regression as the axis, and sharpened the cause to **learning rate above an optimization-stability threshold for shallow/narrow 3×3 towers.** The decisive datapoint: **the same 8×3×3 tower (stem 3×3, 128ch, lineage KbHZ) ran ~495,000 steps stable at lr 1e-3 in fp32** (May era), then **blew up at ~73k steps at lr 1e-2 under bf16/v4** (lineage WjRY). With μ=0.90 the effective per-gradient step ≈ lr/(1−μ) ≈ 0.10 at lr 1e-2 vs ≈ 0.003 at KbHZ's lr 1e-3, a ~35× aggression gap. The threshold separates by kernel/stem geometry: every 3×3-stem run on the current (bf16/v4) era blew up (3/3), and every 7×7-stem run stayed stable (3/3). 7×7 tolerated lr 1e-2 to 471k steps (Exp 1). So Exp 4's blow-up is **not** an Exp-4 quirk. It is the common, reproducible failure mode of lr 1e-2 on 3×3-heavy towers, with the 256ch/wd-1e-4 over-fit story (H2) demoted to at most a secondary margin-shaver. Hypothesis H1 (distribution narrowing) was never cleanly confirmed; the LR threshold is the load-bearing finding. Note that `lr=1e-2·√(b/4096)` is a no-op at the production batch of 4096, so the displayed and effective LRs are equal.
>
> **[Audit of the folded conclusion; see Audit notes for evidence]:**
> - ~~"the same 8×3×3 tower"~~ → **same depth/width/kernel geometry (8 blocks × 3×3, 128 ch, 3×3 stem, basic30) but a different block recipe and heads.** KbHZ is **v3 post-activation, attenuate-only SE /4, activation-gated skip, no ReZero, `simple_conv` policy, WDL(1→FC64), 2,483,667 params, float32**. WjRY is **v4 pre-activation, scale-and-bias SE /4, clean_add + ReZero, `intermediate_conv` policy, WDL(16→FC128), 2,664,087 params, bfloat16**. The optimizer settings differ too (KbHZ μ 0.65, entropy 2.5e-3, draw_penalty 0.05; WjRY μ 0.90, entropy 0, draw_penalty 0). The comparison therefore confounds LR with precision, block recipe, heads, μ and regularization.
> - ~~"~495,000 steps stable at lr 1e-3"~~ → **KbHZ spent most of its life at lr 5e-4.** Its schedule was 1.5e-4 (0 → ~41k), 2.5e-4 (→ ~84.7k), 5e-4 (~84k → ~433k, with a brief 1e-3 at ~171.5k–181k and a rewind to 175.5k), then **1e-3 only from ~432.8k to ~495.6k** (~63k steps), and 1e-3 again for the June continuation 494.9k → 532.4k (still float32: the surviving `Ko63-manual` champion `KbHZ-22` at step 532,369 has `compute_data_type: float32`). It was stable at lr 1e-3 in fp32 for ~100k steps in total, not ~495k. The ≈0.003 effective step is right because KbHZ's μ was 0.65 (1e-3 / 0.35 = 0.0029), not 0.90. The "~35×" figure (0.10 / 0.0029 ≈ 35) holds.
> - **"WjRY blew up at ~73k at lr 1e-2" ✓:** `dcm_log_20260609-104356.txt`, lr 1e-2, μ 0.90, wd 1e-4. Wide probe NLL was 3.54–3.67 through 70k, 4.23 at 73,954, and **14.66 at 75,951**. `[STATS]` legalMass fell from 0.99 (73,305) to **0.03 (78,717)** and pIllM rose from 0.009 to 0.24, the same legal-mass-collapse signature as this run.
> - **"3/3 3×3-stem runs blew up, 3/3 7×7 stable", "8 lineages", "80 checkpoints":** unverified. The run sets are not named anywhere found. Note that eBNC (3×3 stem, bf16, v4, lr 1e-2, launched 2026-06-12, after this conclusion was written) trained 49.9k steps with no in-run blow-up (legalMass ≥ 0.906, gNorm ≤ 4.8), although every later resume of it diverged at once (see [`../20260612-block-groups-ebnc/`](../20260612-block-groups-ebnc/README.md)).
> - **pElo scale:** the KbHZ probe values (`selfplay_probe/KbHZ.csv`, starting at step 494,956) are later re-probes of saved checkpoints, not in-app June values. Do not compare them with the June in-app pElo in this file without saying so.

### 7. Suggested future variants / changes (original)
- **Make the probe NLL (and the self-play diversity histogram) first-class alarms.** This blow-up was silent to every existing alarm; a "wide-NLL rising off its floor" trip would have caught it ~4k steps before the run was stopped. [Audit: add **legal mass** to this. The legal-mass alarm existed but reported "ok" at 0.015.]
- **Re-run from the `jaq1-3` (step ~4,988) checkpoint with wd 3e-4** (± a single cosine anneal) to settle H1 vs H2 per §6. [Audit: that checkpoint is pruned. The run would have to be rebuilt.]
- **If it is distribution narrowing (H1):** raise the self-play exploration temperature or lengthen its tail, or mix a small fraction of OOD (probe-like) positions into the trainer's eval, to keep the policy honest off the self-play manifold.
- **Take the parts of Exp 1/3 §7 not yet applied:** go to a **1×1 stem** on the 210 encoding (frees ~~~1.3M~~ → **~0.43M** params that the 3×3 stem still partly wastes) and **add depth** (8–12 blocks) rather than only width. Depth was the one axis this experiment left untouched. [Audit: 256·210·9 − 256·210 = 430,080. The ~1.29M figure applies to Exp 3's 7×7/128 stem.]

## Caveats
- Single seed. It is an uncontrolled multi-axis jump from Exp 3 (width, kernel, stem, value FC, params).
- pD/vAbs are trainer-side, and draw% and game length are champion self-play. The 9k "value head turns decisive" is a champion behaviour change that the trainer's statistics then follow.
- The key checkpoints (`jaq1-3` at 4,988, `jaq1-4` promote at 9,091) are pruned. Only the 13,861 manual save survives, and its champion is `jaq1-4`.
- All pElo is June 2026 in-app scale.
- The stem-norm read of the surviving `jaq1-4` champion (3×3 stem [256,210,3,3]; ratios vs a He-normal expectation of 1.561) is F0 1.44, F1 1.16, F2–F9 0.94–0.99, rep tail 0.98. The same "reads frame 0 and 1 only" pattern as Exp 3 holds at 13.9k steps. He-normal init is assumed, not re-verified.

## Follow-ups
See §7 above. The LR-threshold reading led to later runs at lower LR / fp32 (outside this write-up's scope).

## Audit notes

Verified against `dcm_log_20260608-115907.txt` (gViN), `-133428`, `-140857`, `-231918` (`[BUTTON]`, `[ARCH]`, `[SEGMENT]`, `[ARENA] #N kv` plus the arena #28 body, `[CHECKPOINT]`, `[STATS]`, `[ALARM] legal-mass probe`, `[TACTICAL-LICHESS] tick`), the surviving `Sessions/20260609-025811-20260608-5-cwkO-manual.dcmsession` (safetensors `__metadata__`, param sums, `session.json`, stem-weight read), `documentation/dashboards/data/jaq1.csv`, `selfplay_probe/jaq1.csv`, `selfplay_registry.json`. For the folded conclusion: the KbHZ logs (all 49 in `selfplay_registry.json`: `[STATS]` lr / μ / `reg=` per log), `Sessions/20260611-212501-20260514-2-Ko63-manual.dcmsession` (KbHZ-22 header, float32, 2,483,667 params), `dcm_log_20260609-104356.txt` (WjRY), `selfplay_probe/WjRY.csv`, `selfplay_probe/KbHZ.csv`.

- **Verified:** architecture and 20,349,716 params; gViN (67,940,116 params, 822 steps); 29 arenas / 4 promotions at 1,268 (#3) / 2,838 (#6) / 4,988 (#10) / 9,091 (#18, +68, score 0.5962); trainer losing #19–#29 (0.1613–0.4900, −7…−286); start and final probe values; NLL blow-up onset ~5.5–5.7k; 9k value/draw/length transition; arena #28 pol ≈ 0.97; diversity 100% unique, diverge 1.4–1.9; wd 1e-4 / lr 1e-2 / μ 0.90; WjRY blow-up ~74–76k; the 35× effective-step arithmetic.
- **Correction:** "promote ≥ 0.53 (later 0.55)" → 0.53 throughout (evidence: every `promote>=` token in all three logs).
- **Correction:** final manual save step ~13,868 → **13,861** (evidence: `session.json` `trainingSteps` and safetensors `training_step`; 13,868 is the last `[STATS]` line in `-140857`). Also added the 509-step manual save and the build-1783 resume to 13,996.
- **Correction:** "vAbs/pD/draw% are champion self-play" → only draw% (and length) are self-play; vAbs/pD are trainer-batch `[STATS]` fields.
- **Correction:** "legal-mass probe steady 0.85–0.89 … blow-up invisible on `[STATS]`" → **legal-mass probe 0.83–0.96 until ~step 6k, then 0.015–0.71; `[STATS]` legalMass ~0.95 → 0.02–0.61 and pIllM 0.035 → 0.12–0.23** (evidence: `[ALARM] legal-mass probe ok` lines at 17:46 / 19:33 / 21:52, and `[STATS]` sampled every 25 lines).
- **Correction:** gNorm "~1.3–3.1" → p5–p95 1.28–3.77, max 5.66. pLogitAbsMax "~13.7 flat" → 12.4–15.7.
- **Correction:** post-blow-up pElo range 370–665 → 328–780 (5–95% 367–628) over steps 5,400–13,868 in `selfplay_probe/jaq1.csv`.
- **Correction:** draw% at 1,268 (87% → 82.3%) and ~5,000 (90% → 85.8%) (evidence: `comp D=` at steps 1,299 and 5,046).
- **Correction:** 1×1-stem saving on this tower ~1.3M → ~0.43M params (arithmetic above).
- **Correction (folded conclusion):** "same 8×3×3 tower" and "~495k steps stable at lr 1e-3 fp32" (see the bulleted audit under the conclusion; evidence: KbHZ-22 header, per-log `[STATS]` lr and μ).
- **Unverified:** the "8 lineages / 80 checkpoints / 3-of-3 vs 3-of-3" sweep membership; H1 vs H2 (no follow-up run exists and its resume checkpoint is pruned); hardware.
- **Scale note:** registry `endpoint_pElo` 540.3 / NLL 14.7415 (trainer `jaq1-5`) is a later-binary re-probe. KbHZ's probe CSV is re-probed checkpoint data. Everything else here is June in-app scale.
