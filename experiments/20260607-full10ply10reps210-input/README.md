# 2026-06-07 — 5-block 7×7 tower on `full10Ply10Reps210` input (10-ply history plus repetition planes)

**Status:** done (stopped at step 53,837 in `[STATS]`, 53,850 in `[BATCH-STATS]`, on 2026-06-08 11:12 CDT. A resume attempt at 11:53 loaded the step-48,825 save but never trained a step. The machine then went to the 256-wide follow-up.)

Migrated from `documentation/ARCH_EXPERIMENTS.md` Experiment 3 and audited 2026-09-29. The original text is kept. Corrections are marked `~~old~~ → new` or `[Audit: …]`, and all are listed in Audit notes. The original header said "in progress (~53k steps @ 2026-06-08 10:36)"; the data resolves the endpoint as above.

## Question

Direct successor to Experiment 2 ([`../20260606-full10ply200-input/`](../20260606-full10ply200-input/README.md)), testing its single suggested variant: put the 10 repetition planes back. `full10Ply10Reps210` = `full10ply200` (10 stacked `basic20` frames) **plus** the 10 `basic30` temporal-repetition planes (20–29), restored as a tail. Across the three experiments only the **input encoding** changes, on an identical 5-block 7×7 tower:
- Exp 1: `basic30` (reps, no history)
- Exp 2: `full10ply200` (history, no reps)
- Exp 3: `full10Ply10Reps210` (history **and** reps)

Together they form a controlled encoding ladder. Does restoring the duplication planes recover Exp 2's slow value-head bootstrap, and does the history help tactically?

## Setup

- **Architecture** (`[ARCH] built champion 20260607-6-eaRt` line; confirmed against the embedded `architecture` JSON in the surviving `KnCx-promote` `champion.safetensors`):
  `v4 pre . in full10Ply10Reps210(210) -> stem 128 (7x7) . 5x[7x7 conv, SE+/4, clean_add, ReZero] . act relu . policy intermediate_conv(4864) . value WDL(16->FC128) . bfloat16 . 9,574,708 params`
- **Build** 1770, git `ff88f64*`, branch `safetensors-storage` (safetensors-native: embedded-config identity per PLAN §6, no `arch_hash`; isolate by the `[ARCH]` line plus the log file).
- **Logs:** `dcm_log_20260607-174928.txt` (the whole run, 0 → 53,837) and `dcm_log_20260608-115345.txt` (build 1780: auto-resume of the 48,825 promote save, 11:53:45–11:54:21, **no training step**; the only `[STATS]` line reads `steps=48825`).
- **Dates:** 2026-06-07 17:50 → 2026-06-08 11:12 CDT (~17.4 h wall).
- **Training config** (`[STATS]`): constant LR 1e-2 (500-step warmup), wd 1e-4, clip 30, μ 0.90, entropy 0, draw_penalty 0. Batch 4096, 800 workers, `spDelay=3000ms`, promote ≥ 0.53, 400-game arenas every 900 s. Identical to Exp 2.
- **Machine:** local (logs on this Mac). The hardware is not logged.

### 1. Architecture (original §1)
- **Input:** **210 planes** × 8×8 (NCHW), `full10Ply10Reps210`: 10 stacked `basic20` frames (current ply N plus 9 prior plies, ply-N mover's perspective, absent frames zero), followed by the 10 `basic30` temporal-repetition planes (plane `200+i` = position `i+1` plies ago is a strict duplicate). **Policy** 4864 logits. **Value** 3-class W/D/L head.
- **Stem:** 7×7 conv, **210 → 128**.
- **Tower / policy / value:** **identical to Experiments 1 & 2**: 5 pre-activation residual blocks, 128 ch, 7×7 convs, scale-and-bias SE (/4), clean-add + ReZero (α `1/√5`), ReLU; `intermediate_conv` policy head; WDL value head (C_v=16, H=128).
- **Precision:** bfloat16. **Params:** 9,574,708 (~9.57M). [Audit: the header tensor-shape sum is exactly 9,574,708.]
- *Context:* **only the input encoding changed vs Exp 2** (full10ply200/200 → full10Ply10Reps210/210). +62,720 params, all in the stem ((210−200)×128×49). [Audit: 9,574,708 − 9,511,988 = 62,720 ✓.] Tests whether restoring the dropped duplication planes recovers Exp 2's slow value-head bootstrap.

## Runs

- **Live lineage:** champion `20260607-6-eaRt` → `eaRt-14`. The final trainer is `eaRt-15`. Promotions fork the ID. **Saved-session lineage:** `20260607-7-KnCx`.
- **Arenas:** 64 in total (#1 at 1,422 → #64 at 53,611), with **14 promotions**. The last was arena #56 at step 48,825. Arenas #57–#64 (49,282 → 53,611) did not promote; they scored 0.4250–0.5112.
- Promotion steps and scores: 1,422 (0.5925), 2,249 (0.5687), 5,013 (0.5400), 10,389 (0.5337), 12,699 (0.5463), 16,865 (0.5863), 18,642 (0.5325), 23,209 (0.5337), 25,929 (0.5312), 29,559 (0.5325), 32,272 (0.5363), 36,791 (0.5563), 38,548 (0.5463), 48,825 (0.5300).

### 2. Relevant saved sessions (original §2, expanded from the log)

The run wrote numerous post-promotion `.dcmsession` autosaves plus one periodic save (safetensors-native; live champion lineage `eaRt`→`KnCx`). The original listed one verified snapshot. [Audit: the full list from `[CHECKPOINT] Saved session` lines follows. **Only the 48,825 promote save survives on disk today.**]

| Saved session (`.dcmsession`) | Step @ snapshot | Trigger | On disk 2026-09-29 |
|---|--:|---|---|
| `20260607-231521-20260607-7-KnCx-promote` | 1,422 | promote | pruned |
| `20260607-233042-20260607-7-KnCx-promote` | 2,249 | promote | pruned |
| `20260608-001638-20260607-7-KnCx-promote` | 5,013 | promote | pruned |
| `20260608-014831-20260607-7-KnCx-promote` | 10,389 | promote | pruned |
| `20260608-023423-20260607-7-KnCx-promote` | 12,699 | promote | pruned |
| `20260608-035059-20260607-7-KnCx-promote` | 16,865 | promote | pruned |
| `20260608-042136-20260607-7-KnCx-promote` | 18,642 | promote | pruned |
| `20260608-053803-20260607-7-KnCx-promote` | 23,209 | promote | pruned |
| `20260608-062354-20260607-7-KnCx-promote` | 25,929 | promote | pruned |
| `20260608-072502-20260607-7-KnCx-promote` | 29,559 | promote | pruned |
| `20260608-081052-20260607-7-KnCx-promote` | 32,272 | promote | pruned |
| `20260608-092712-20260607-7-KnCx-promote` | 36,791 | promote | pruned |
| `20260608-095744-20260607-7-KnCx-promote` | 38,548 | promote | pruned |
| `20260608-135802-20260607-7-KnCx-periodic` | 48,774 | periodic | pruned |
| `20260608-140104-20260607-7-KnCx-promote` | 48,825 | promote | **present** (4.9 GB) |

Each `champion.safetensors` embeds the full architecture in `__metadata__` (`input_encoding: full10Ply10Reps210`, `training_step`, `content_sha256`). For the surviving save, the champion is `20260607-6-eaRt-14` at step 48,825 (9,574,708 params; `content_sha256` 7b16190c…). The trainer is `eaRt-15` (parent `eaRt-14`), step 48,825; its 19,146,056 stored values are the weights plus optimizer velocity. `session.json` `arenaHistory` holds 56 arenas and 14 promotions. Location: `~/Library/Application Support/DrewsChessMachine/Sessions/`.

No checkpoint exists for steps 48,826–53,837. The trainer weights at the stop were never saved.

## Results

### 3. Factuals (original table)
The 200-set is the cross-comparison column, because basic30 (Exp 1) lacks an early wide set; the wide set is listed where available. All pElo is the **June 2026 recording-build in-app scale**. Do not compare it with the selfplay_registry endpoint (617.9 / NLL 4.7526, a later-binary re-probe of the trainer) or with replay-era pElo.

| Step | pElo (200) | NLL (200) | pElo (wide) | NLL (wide) | vAbs | pD | draw% | Detail |
|--:|--:|--:|--:|--:|--:|--:|--:|---|
| 5k | 698 | 4.022 | 616 | 3.827 | 0.104 | 0.719 | 79% | Value head already decisive (started low, pD 0.667 / vAbs ~0.10 at 1k, unlike Exp 2's draw-prior start). |
| 10k | 744 | 3.905 | 647 | 3.726 | 0.095 | 0.757 | 85% | Leads basic30 (719) and Exp 2 (694) on 200-set pElo. ~4 promotions by 10.5k (tied). |
| 23k | 733 | 3.888 | 654 | 3.716 | 0.118 | 0.716 | 83% | Value head tracking basic30 (vAbs 0.118 vs 0.133); still ≈ level on pElo. |
| 30k | 704 | 3.929 | 652 | 3.759 | 0.130 | 0.714 | 83% | basic30 begins its surge here; 210 flat. |
| 40k | 761 | 3.909 | 670 | 3.769 | 0.162 | 0.660 | 77% | basic30 overtakes (803 vs 761). 210 value head decisive & climbing. |
| 48k | 738 | 3.928 | 663 | 3.758 | 0.163 | 0.658 | 78% | Gap to basic30 ~70 pElo / 0.43 NLL. Tied with Exp 2 on pElo (766). |
| 52.8k | 754 | 3.927 | — | — | 0.167 | 0.647 | 79% | **Plateaued** ~750; basic30 ~810. Gap stable ~55 pElo. |
| 53,837 | — | — | — | — | 0.169 | 0.645 | 77% | [Audit: added row. **Final step**, champion `eaRt-14`, trainer `eaRt-15`. Pointwise wide 2k-bucket (52k–end) 679.0 / 3.769, 200-set 765.2 / 3.921.] |

[Audit, value/draw columns: every row matches `[STATS]` (pD, vAbs, `comp D`) at the nearest step: 1,011 → 0.667 / 0.097 / 78.6%; 10,013 → 0.757 / 0.095 / 85.2%; 23,007 → 0.716 / 0.118 / 82.9%; 29,974 → 0.714 / 0.130 / 83.1%; 40,016 → 0.660 / 0.162 / 77.2%; 47,979 → 0.658 / 0.163 / 77.1%; 52,782 → 0.647 / 0.167 / 79.1%. pD/vAbs are **trainer-side** batch statistics; draw% is champion self-play. For the pElo/NLL columns, the doc's averaging method is unrecorded. 2k-bucket means starting at each step (200 / wide) are: 4k 697.8 / 616.4, 10k 757.8 / 650.4, 22k 736.9 / 651.6, 30k 721.9 / 656.9, 40k 753.3 / 672.1, 48k 748.3 / 674.0, 52k 765.2 / 679.0. The table agrees with these to within ~±15 pElo and ~±0.03 NLL.]

*Trend slopes (23k–48k): basic30 pElo **+57/10k (R²=0.53)**, NLL **−0.12/10k (R²=0.71)**, genuinely climbing. 210 pElo **+15/10k (R²=0.09, flat/noise)**, NLL ~0, not improving. The gap widened 23k→40k and then stabilized.* [Audit ✓: least-squares fits on pointwise 200-set ticks from 23k to 48k give 210 +15.7/10k, R² 0.09, NLL +0.008/10k, R² 0.01; bzw3 (`dcm_log_20260601-205349.txt`) +60.3/10k, R² 0.54, NLL −0.128/10k, R² 0.71.]

### 4. Wins
- **Value head ignites early, like basic30 and unlike Exp 2.** It is decisive from ~step 1k (vAbs ~0.10, pD climbing) and tracks basic30 (vAbs 0.118/0.162/0.167 at 23k/40k/52.8k vs basic30's 0.133/0.20/0.20). It runs **~30k steps ahead** of Exp 2's stalled head (frozen ~0.083 until ~48k). Restoring the repetition planes (the only change vs Exp 2) coincides with recovering early value-head learning. **Behaviorally this supports Exp 2's Hypothesis #2** (removing the rep planes slowed the value head), not #1 (history per se), but see the weight-forensics caveat in §6. [Audit ✓: basic30 vAbs 0.1332 / 0.2024 / 0.2025 at 22,966 / 39,987 / 52,784.]
- **Better calibration and a more decisive head than Exp 2** at matched steps (NLL 3.93 vs 3.98–4.02; vAbs 0.167 vs 0.135 at 52.8k).

### 5. Shortcomings
Compared primarily against Experiment 1 (basic30):
- **Tactically weaker than basic30, and the gap is a stable plateau, not closing.** From a roughly level start (210 *led* at 23k, 733 vs 714), basic30 surged after 30k while 210 stayed flat. By 52.8k basic30 leads by **~55 pElo (810 vs 754) and ~0.43 NLL (3.50 vs 3.93)**. 210's pElo slope is statistically flat (R²=0.09) vs basic30's real climb: the lines diverged, then locked, and are not converging. [Audit: bzw3 200-set 2k-buckets are 709.3 (22–24k), 811.7 / 3.511 (52–54k).]
- **No tactical benefit over Exp 2.** 210 and full10ply200 are **tied on pElo** (~755 at 52.8k). Restoring the rep planes helped the *value head* but did **nothing** for tactical/policy strength.
- **Wide-set NLL flat** (~3.72–3.77 across 10k–48k): the same calibration-not-improving signature as Exp 2, while basic30's NLL drops over the same window.
- [Audit, added:] **The arena ladder stalled late:** no promotion in the final ~5.0k steps (8 arenas, 48,825 → 53,837).

## Conclusion

### 6. Analysis (original)
- **Controlled encoding ladder, value-head result:** on the identical tower and optimizer, Exp 3 (history plus reps) ignites the value head as early as Exp 1 (reps, no history) and far earlier than Exp 2 (history, no reps). Read naively, the **repetition planes** drive early value-head learning (H2).
- **Weight forensics complicate the mechanism (2026-06-08, on the 48.8k `champion.safetensors`).** The per-input-plane stem L2 norm shows the net reads **only frame 0 (2.5× init) and frame 1 (1.6× init)**. **Frames 2–9 sit at initialization (0.96–0.98× init) and slightly *decay* over 47k steps** (weight decay pruning unreinforced inputs), and **the rep-plane tail is also at init (0.95×)**. Across all saved checkpoints (1.4k→48.8k), F0/F1 grow monotonically while F2–9 and the reps never move. So **the 210 net uses only "current + 1 prior ply"; the other 8 history frames and the rep planes are structurally unused.** basic30 *also* leaves its rep planes at ~0.66× init, so the rep planes are not heavily weighted in **either** net. Exp 3's early value-head advantage over Exp 2 is therefore **not** cleanly attributable to the rep planes being *used* (they're at init). It may be seed, or a small purposeful sparse projection that the norm can't see (rep planes are sparse-binary). **Open: confirm via an occlusion test** (zero the rep planes on repetition-rich positions and measure the value/policy delta). [Audit: re-derived from the surviving `eaRt-14` champion (step 48,825). Dividing by the He-normal expectation √(128·49·2/(210·49)) = 1.104 gives F0 2.51, F1 1.57, F2 1.07, F3 1.01, F4 0.97, F5 0.95, F6 0.94, F7 0.93, F8 0.93, F9 0.92, rep tail 0.95. So F2 is slightly above the quoted "0.96–0.98" band and F5–F9 slightly below it; the conclusion holds. The basic30 rep planes read 0.66× on the surviving bzw3-31 model (step 467,065; rep-count 18–19 and rep-temporal 20–29 both 0.66). The "across all saved checkpoints 1.4k→48.8k" monotonic trend is **unverifiable today**: only the 48.8k save survives.]
- **Why no tactical gain:** the deep history is unused, so Exp 3 is effectively a `basic30`-class net carrying ~150 dead input planes plus ~1 extra ply of context. On the capacity-saturated 5-block tower (Exp 1's forensics: 0% dead, ~93% rank, full kernel utilization) there is no spare capacity to exploit the richer input. It is the same capacity story as Exp 1, seen from the input side.
- **The encoding question is not answerable on this tower.** Whether 10-ply history *could* help is confounded by the tower being the bottleneck (the net declines to use the history at all). The clean test is the same deeper-tower experiment Exp 1 calls for.

## Caveats
- Single seed per arm. Exp 1/2/3 is n=1 each, so value-head timing differences of ~25–30k steps could partly be seed.
- The stem-norm "× init" ratios assume He-normal stem initialization, which was not re-verified in code. Weight norm cannot see sparse, purposeful use of binary planes.
- pD/vAbs are trainer-batch statistics; draw% is champion self-play.
- All pElo is June 2026 in-app scale (see Results).
- No checkpoint after 48,825 exists, so the final ~5k steps can only be read from the log.

## Follow-ups

### 7. Suggested future variants / changes (original)
- **Settle the encoding question on a tower that isn't the bottleneck:** rerun `full10ply200` / `full10Ply10Reps210` vs `basic30` on an **8–12 block 3×3** tower (1×1 stem). Readout: does the stem's frame-2–9 norm move off init? If yes, capacity was the bind and history helps. If it stays pinned (as here), deep history is dead weight for this engine, which closes the question.
- **Occlusion test for the rep planes** (sparse-binary, so weight norm understates them): on positions at or near 3-fold, zero the rep tail and measure the value-head shift. This settles "suppressed vs functionally unused." Ideal probe: a position with ≥2 legal moves where one forces 3-fold/50-move and the other looks otherwise equal but keeps a win.
- **Drop the dead input cost:** a 1×1 stem on the 210 encoding frees ~1.29M params (~13.5% of the model) that the current 7×7 stem spends on a spatially collapsed, mostly ignored input. [Audit ✓: 128·210·49 − 128·210 = 1,290,240 = 13.5% of 9,574,708.]
- The next run on the machine was the 256-wide dual-kernel tower on this same encoding ([`../20260608-256wide-dual-kernel/`](../20260608-256wide-dual-kernel/README.md)).

## Audit notes

Verified against `dcm_log_20260607-174928.txt` and `dcm_log_20260608-115345.txt` (`[ARCH]`, `[SEGMENT]`, `[ARENA] #N kv`, `[CHECKPOINT]`, `[STATS]`, `[BATCH-STATS]`, `[TACTICAL-LICHESS] tick`), the surviving `Sessions/20260608-140104-20260607-7-KnCx-promote.dcmsession` (safetensors `__metadata__`, tensor-shape param sums, `session.json` `arenaHistory`, stem-weight read), `documentation/dashboards/data/eaRt.csv` (ends at 53,837), `selfplay_probe/eaRt.csv` (wide set), and `selfplay_registry.json`. Basic30 comparisons were checked against the bzw3 logs `dcm_log_20260601-162715.txt` and `-205349.txt`.

- **Resolved status:** "in progress (~53k @ 2026-06-08 10:36)" → **done, stopped at 53,837** (last `[STATS]` 11:11:47 CDT; `[BATCH-STATS]` 53,850). The 11:53 relaunch (build 1780) loaded the 48,825 save and trained 0 steps before the process ended at 11:54:21. The 256-wide build started at 13:35 the same day.
- **Verified:** architecture, 9,574,708 params, +62,720; 14 promotions / 64 arenas; "~4 promotions by 10.5k (tied)" (eaRt 1,422 / 2,249 / 5,013 / 10,389; 2Gd1 1,440 / 2,414 / 7,526 / 8,557); every vAbs/pD/draw% cell; basic30 comparisons (719 at 10k ≈ bzw3 2k-bucket 710.7; 803 at 40k ≈ 808.3; ~810 / 3.50 at 52.8k ≈ 811.7 / 3.511; vAbs 0.133 / 0.20 / 0.20); Exp 2 comparisons (694 at 10k ≈ 688.3; 766 at 48k ≈ 770.1); the trend-slope statistics; the 1×1-stem savings arithmetic; stem-norm frame pattern (F0 2.51, F1 1.57, reps 0.95).
- **Correction (minor):** frames 2–9 "0.96–0.98× init" → **0.92–1.07×** (F2 1.07 … F9 0.92) (evidence: `stem.conv.weight` of `eaRt-14`).
- **Added:** the full saved-session list (14 more saves from `[CHECKPOINT]` lines; all pruned but the 48,825 promote), the final-step row, and the late-arena stall (#57–#64, no promotion).
- **Unverified:** "across all saved checkpoints (1.4k→48.8k) F0/F1 grow monotonically" (only one checkpoint survives); the exact averaging behind the table's pElo/NLL cells (2k-bucket means agree within ~±15 pElo); hardware.
- **Scale note:** the registry endpoint (617.9 / 4.7526) is a later-binary re-probe of trainer `eaRt-15` and sits on a different scale from this table. `selfplay_probe/eaRt.csv` peaks at 717.3 wide (step 39,827).
