# 2026-07-10 — nt8y stem kernel (3×3 / 5×5 / 15×15) and seed variance

**Status:** abandoned (every arm stopped early by choice; no arm reached the 31.3 h parity target. 3×3 stopped at 20k steps, the second seed at 5.8k, 15×15 at 49.75k)

## Question

- On the nt8y tower (3 blocks, 15×15 convs, 32 channels), does the **stem** conv's kernel size (3×3, 5×5, 15×15) change how fast a model learns from the std corpus?
- How big is **seed-to-seed** noise for one fixed architecture? That noise is the yardstick for any stem difference.

## Setup

- **Tower:** identical in every arm. Only `stem_conv_kernel_size` differs. Embedded architecture JSON, field by field:
  - v5 · basic30(30) → stem **k×k** → 32 · 3×[15×15+15×15 @32, SE scale+bias /4, ReLU/pre, clean_add, ReZero α init 0.333 (cap 0.333), out: layer_norm, dropout×1] · policy intermediate_conv (pre-conv 512) → 4864 · value WDL (16 → FC64) · bfloat16.
- **Parameter counts** (sum of tensor elements in the safetensors, same as the `[REPLAY-ARCH]` line):

| arm | stem | params | Δ vs 3×3 | architecture JSON sha256 (sorted keys, first 12 hex) |
|---|---|---|---|---|
| nt8y | 5×5 | 1,533,930 | +15,360 | 07785f582f1c |
| nt8y3x3 (seed 1) | 3×3 | 1,518,570 | 0 | 086f3417e4e4 |
| nt8y3x3s2 (seed 2) | 3×3 | 1,518,570 | 0 | 086f3417e4e4 |
| nt8y15x15 | 15×15 | 1,725,930 | +207,360 | 8e35370640d6 |

  - The stem is 30 → 32 channels with no bias, so its size is 960·k²: 8,640 / 24,000 / 216,000. The deltas above match that exactly.
- **Corpus:** std `20260624-192615-w3aA5b` (lichess 2026-05, 46 sealed shards) for every arm. At any given step, every arm had seen exactly the same games.
  - The log's `games=` counter and the checkpoints' `replay_next_game_index` agree across arms at every matched step: 136,804 @1k, 655,048 @5k, 2,584,454 @20k, 5,924,823 @46k.
  - So on this experiment the **games_fed axis is the same as the step axis**.
- **Hyperparameters** (from `[REPLAY-HPARAMS]`):
  - The three new arms used the same values: lr 0.01, batch 4096, **wd 0.0005, momentum 0.9**, gradClip 30, pLabelSmooth 0.1, vLabelSmooth 0.013, lrWarmup 500, bufCap 1,000,000, replayRatio 0.48, minPrefill 500,000, complementCE on, sqrtBatchLR on.
  - `--epochs 12 --enumerate-checkpoints` for all three.
  - **nt8y (5×5) used different settings.** Its segment-1 log (`dcm_log_20260701-152447.txt`, and resume2/3 likewise) shows **wd 0.00025, momentum 0.93**. Segment 0 is the only segment in the 0–49k window, and its hyperparameters are unverified (log deleted; see Caveats).
- **Builds:**

| arm | build | git |
|---|---|---|
| nt8y seg0 | 2009 | e96e0b4 |
| nt8y3x3 | 2075 | 324bf6f |
| nt8y3x3s2 | 2079 | 55f7b5a |
| nt8y15x15 | 2080 | 29ba72e |

  - Between 324bf6f and 29ba72e, only architecture-preset and new-model CLI code changed (`NewModelCLI.swift`, `NetworkArchitecture.swift`, `ArchitecturePresetStore.swift`, `DrewsChessMachineApp.swift`, one test). Training math was not touched.
- **Machine:**
  - The three new arms ran on this Mac. Their corpus path was `~/Library/Application Support/DrewsChessMachine/Corpora/…`.
  - nt8y segment 0 ran before the 2026-07-07 migration. Its checkpoints record the corpus at `/Volumes/20260624-192615-w3aA5b` (the retired DMG mount), so it ran on a different host.
- **Presets:** `nt8y_3x3stem` (commit 6aa027b) and `nt8y_15x15stem` (commit 7e7c14e) are `NetworkArchitecture.Preset` cases in code. Only `nt8y.json` exists in `~/Library/Application Support/DrewsChessMachine/Presets/`.

## Runs

Checkpoints were identified from safetensors `__metadata__` (`model_id`, `training_step`, `parent_model_id`).

| registry key | fresh model (id) | trainer lineage id | log | enumerated checkpoints | last logged step | last checkpoint |
|---|---|---|---|---|---|---|
| nt8y (5×5) | `20260701-161012-20260701-3-nT8Y-manual` (20260701-3-nT8Y) | 20260701-4-CIvL (parent nT8Y) | seg0 `dcm_log_20260701-091259.txt` **missing**; later segs `dcm_log_20260701-152447.txt`, `…20260706-125010`, `…20260706-193601`, 4th in registry | `20260701-nT8Y-fatconv-step*-frozen` (seg0) + per-resume stems | parity run (310,969 cum) | — |
| nt8y3x3 | `20260711-nt8y3x3stem-fresh` (20260711-3-pm4J) | 20260711-4-1mjX | `dcm_log_20260711-002649.txt` (00:26:49–03:08:59) | `20260711-nt8y3x3stem-std-replay-step{1000..20000}` | 20,440 | step 20000 |
| nt8y3x3s2 | `20260711-nt8y3x3stem-seed2-fresh` (20260711-9-dOjG) | 20260711-10-FxUc | `dcm_log_20260711-031104.txt` (03:11:04–03:57:05) | `20260711-nt8y3x3stem-seed2-std-replay-step{1000..5000}` | 5,800 | step 5000 |
| nt8y15x15 | `20260711-nt8y15x15stem-fresh` (20260711-11-xE1v) | 20260711-13-9mEU | `dcm_log_20260711-040005.txt` (04:00:05–10:13:15) | `20260711-nt8y15x15stem-std-replay-step{1000..49000}` | 49,750 | step 49000 (= `-latest`) |

- Only nt8y's steps 1,000–49,000 are used here. That is all segment 0 (cumstep_base 0), which ends at 65,883.
- **No two arms ran at the same time.** Each log ends before the next begins:
  - wxil ended 00:19:28, then 3×3 ran 00:26–03:08.
  - 3×3s2 ran 03:11–03:57.
  - 15×15 ran 04:00–10:13.
  - The logs in between are zero-byte (unidentifiable, likely short probe/CLI invocations).
  - ms/step stays flat within each run (3×3 414–421 ms, 15×15 390–407 ms), so no competing training job shared the GPU.

## Results

Puzzle pElo and nll come from `documentation/dashboards/data/<key>.csv`, on the replay-era (July+) probe scale. They are compared at matched step, which here also means matched games_fed. Cells are blank where an arm has no data.

| step | nt8y 5×5 pElo | nll | 3×3 s1 pElo | nll | 3×3 s2 pElo | nll | 15×15 pElo | nll |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 769.6 | 3.3984 | 740.0 | 3.4899 | 783.7 | 3.5369 | 769.6 | 3.4398 |
| 2000 | 853.5 | 3.1845 | 837.9 | 3.2575 | 831.4 | 3.2807 | 879.2 | 3.1394 |
| 3000 | 955.1 | 2.9690 | 876.3 | 3.2056 | 884.2 | 3.1267 | 932.7 | 3.0992 |
| 4000 | 986.0 | 2.9135 | 904.4 | 3.0795 | 913.9 | 3.0537 | 937.6 | 3.0234 |
| 5000 | 992.5 | 2.8746 | 962.2 | 2.9576 | 936.5 | 2.9860 | 961.1 | 2.9768 |
| 6000 | 1034.7 | 2.8190 | 994.1 | 2.9173 | | | 983.3 | 2.9268 |
| 7000 | 1014.5 | 2.8601 | 995.7 | 2.8978 | | | 1021.9 | 2.8521 |
| 8000 | 1027.8 | 2.8466 | 1034.2 | 2.8106 | | | 1033.7 | 2.8254 |
| 9000 | 1118.0 | 2.7148 | 1051.7 | 2.7910 | | | 1045.3 | 2.8390 |
| 10000 | 1091.2 | 2.7209 | 1050.6 | 2.7741 | | | 1077.6 | 2.7526 |
| 11000 | 1113.8 | 2.7124 | 1073.4 | 2.7687 | | | 1099.7 | 2.7231 |
| 12000 | 1100.7 | 2.7149 | 1089.7 | 2.7270 | | | 1064.9 | 2.7913 |
| 13000 | 1110.2 | 2.7274 | 1100.7 | 2.7164 | | | 1082.8 | 2.7454 |
| 14000 | 1146.7 | 2.6565 | 1105.4 | 2.7015 | | | 1084.4 | 2.7542 |
| 15000 | 1135.2 | 2.6673 | 1102.8 | 2.7112 | | | 1093.4 | 2.7606 |
| 16000 | 1180.0 | 2.6120 | 1139.4 | 2.6420 | | | 1143.1 | 2.6737 |
| 17000 | 1189.4 | 2.6015 | 1161.3 | 2.6261 | | | 1130.5 | 2.6786 |
| 18000 | 1191.0 | 2.6058 | 1157.7 | 2.6577 | | | 1157.2 | 2.6524 |
| 19000 | 1250.0 | 2.5147 | 1168.1 | 2.6298 | | | 1137.3 | 2.6805 |
| 20000 | 1242.8 | 2.5544 | 1189.4 | 2.5991 | | | 1157.2 | 2.6564 |
| 21000 | 1210.7 | 2.6098 | | | | | 1152.5 | 2.6419 |
| 22000 | 1256.2 | 2.5252 | | | | | 1153.5 | 2.6462 |
| 23000 | 1267.1 | 2.4906 | | | | | 1173.3 | 2.6156 |
| 24000 | 1246.4 | 2.5501 | | | | | 1127.9 | 2.6717 |
| 25000 | 1254.2 | 2.5271 | | | | | 1196.7 | 2.6073 |
| 26000 | 1249.5 | 2.5389 | | | | | 1186.3 | 2.6195 |
| 27000 | 1290.3 | 2.4798 | | | | | 1180.0 | 2.5927 |
| 28000 | 1277.9 | 2.5060 | | | | | 1152.0 | 2.6438 |
| 29000 | 1308.8 | 2.4539 | | | | | 1185.8 | 2.5777 |
| 30000 | 1335.1 | 2.4541 | | | | | 1204.5 | 2.5973 |
| 31000 | 1336.2 | 2.4318 | | | | | 1204.5 | 2.5717 |
| 32000 | 1298.0 | 2.4552 | | | | | 1247.4 | 2.5288 |
| 33000 | 1277.9 | 2.5249 | | | | | 1240.7 | 2.5233 |
| 34000 | 1269.6 | 2.5472 | | | | | 1245.4 | 2.5175 |
| 35000 | 1331.5 | 2.4546 | | | | | 1219.0 | 2.5722 |
| 36000 | 1336.2 | 2.4224 | | | | | 1226.2 | 2.5454 |
| 37000 | 1341.3 | 2.4131 | | | | | 1283.6 | 2.4650 |
| 38000 | 1352.6 | 2.4178 | | | | | 1238.6 | 2.5272 |
| 39000 | 1374.7 | 2.4001 | | | | | 1217.9 | 2.5548 |
| 40000 | 1351.1 | 2.4260 | | | | | 1218.4 | 2.5492 |
| 41000 | 1281.5 | 2.5000 | | | | | 1235.0 | 2.5510 |
| 42000 | 1338.7 | 2.4343 | | | | | 1245.4 | 2.5130 |
| 43000 | 1353.7 | 2.3994 | | | | | 1238.1 | 2.5235 |
| 44000 | 1388.6 | 2.3958 | | | | | 1241.2 | 2.5168 |
| 45000 | 1342.8 | 2.4263 | | | | | 1272.7 | 2.4565 |
| 46000 | 1409.2 | 2.3259 | | | | | 1280.5 | 2.4673 |
| 47000 | 1370.6 | 2.3710 | | | | | 1318.6 | 2.4418 |
| 48000 | 1389.1 | 2.3615 | | | | | 1298.0 | 2.4350 |
| 49000 | 1366.5 | 2.3924 | | | | | 1294.9 | 2.4692 |

### Pairwise differences at matched steps (A − B; positive pElo / negative nll = A better)

| pair | steps | n | pElo mean | pElo mean \|Δ\| | pElo max \|Δ\| (step) | A ahead | nll mean | nll mean \|Δ\| | nll max \|Δ\| |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **seed 1 − seed 2** (3×3) | 1k–5k | 5 | −5.8 | 18.6 | 43.7 (1k) | 2/5 | +0.0012 | 0.0407 | 0.0789 |
| 3×3 s1 − 15×15 | 1k–5k | 5 | −31.9 | 32.3 | 56.4 (3k) | 1/5 | +0.0623 | 0.0700 | 0.1181 |
| 3×3 s2 − 15×15 | 1k–5k | 5 | −26.1 | 31.7 | 48.5 (3k) | 1/5 | +0.0611 | 0.0611 | 0.1413 |
| 3×3 s1 − 15×15 | 1k–20k | 20 | −2.9 | 21.5 | 56.4 (3k) | 12/20 | −0.0015 | 0.0464 | 0.1181 |
| 3×3 s1 − 15×15 | 6k–20k | 15 | +6.8 | 17.9 | 32.2 (20k) | 11/15 | −0.0228 | 0.0385 | 0.0643 |
| 5×5 − 3×3 s1 | 1k–5k | 5 | +47.2 | 47.2 | 81.6 (4k) | 5/5 | −0.1300 | 0.1300 | 0.2366 |
| 5×5 − 3×3 s2 | 1k–5k | 5 | +41.4 | 47.0 | 72.1 (4k) | 4/5 | −0.1288 | 0.1288 | 0.1577 |
| 5×5 − 3×3 s1 | 1k–20k | 20 | +38.4 | 39.0 | 81.9 (19k) | 19/20 | −0.0646 | 0.0693 | 0.2366 |
| 5×5 − 15×15 | 1k–20k | 20 | +35.5 | 39.4 | 112.7 (19k) | 16/20 | −0.0661 | 0.0736 | 0.1658 |
| 5×5 − 15×15 | 21k–49k | 29 | +94.0 | 94.0 | 156.8 (39k) | 29/29 | −0.0934 | 0.0955 | 0.1547 |

- **Seed per-seed values:**
  - pElo diffs, 1k→5k: −43.7, +6.4, −7.9, −9.5, +25.7.
  - nll diffs, 1k→5k: −0.0470, −0.0232, +0.0789, +0.0258, −0.0284.
- **Within-run probe jitter** (pElo SD of each point about a centered 5-point mean, steps ≥ 6k):

| arm | SD | n |
|---|---:|---:|
| 3×3 s1 | 10.3 | 13 |
| 15×15 | 16.4 | 42 |
| nt8y | 21.6 | 42 |

  - The largest single-step pElo drops were −45.3 (15×15) and −69.6 (nt8y).
  - Probe pElo is visibly quantized: the same value repeats, e.g. 769.58 at step 1k in three arms, 1157.16 and 1204.45 each twice in 15×15.
- **By elapsed training time** (step pace, sec per 1,000 steps from `elapsed_train_sec`):

| arm | sec / 1,000 steps | log median ms/step |
|---|---:|---:|
| 3×3 | 475.7 | ~417 |
| 3×3s2 | 473.6 | ~427 |
| 15×15 | 449.9 | ~398 |
| nt8y seg0, other host | ~335–345 | — |

  - At matched elapsed time, **3×3 vs 15×15** is essentially the step comparison shifted about 5%. Example at ~9,500 s: 3×3 @20k = 1189.4, 15×15 @21k (9,388 s) = 1152.5.
  - nt8y has more steps per second, so by time it looks even further ahead (e.g. ~28k steps at 9,421 s, 1277.9). That comes from the host, not the architecture (see Caveats).

## Conclusion

- **3×3 vs 15×15 stem: no measurable difference.** They are same host, same hyperparameters, and same data order.
  - Over 1k–20k, the mean pElo gap is −2.9, with 3×3 ahead 12/20 times.
  - After 6k, the mean \|Δ\| of 17.9 pElo is about the size of one run's own step-to-step jitter (SD 10–16) and of the early seed gap (18.6).
  - nll agrees: mean gap −0.0015 over 1k–20k.
  - Early on (1k–5k), 15×15 is ahead of both 3×3 seeds by ~26–32 pElo, about 1.7× the seed spread. That gap disappears by 6k.
  - The 15×15 stem adds 207k params (+13.7%) and bought nothing measurable. A tower of three 15×15 blocks already covers the whole 8×8 board, so a bigger stem adds no reach.
- **5×5 (the original nt8y) is ahead of both new stems at every horizon, but this is NOT a clean stem result.**
  - Leads: +38 pElo vs 3×3 over 1k–20k (19/20), +94 vs 15×15 over 21k–49k (29/29), nll about 0.07–0.09 lower.
  - nt8y differs from the new arms in more than the stem: different host, build 2009 vs 2075–2080, and very likely different wd/momentum (its logged segments ran wd 0.00025 / momentum 0.93 vs 0.0005 / 0.9).
  - Any of those can explain a ~40–90 pElo gap.
  - The fair statement: the stem-kernel effect on nt8y is **not established**. 3×3 ≈ 15×15. 5×5 vs the rest is confounded, and the confounded gap favors the 5×5 run.
- **Seed variance:** only 5 matched points (1k–5k), all in the steep early part of training. Mean \|Δ\| 18.6 pElo, max 43.7 (at step 1k), mean \|Δ\| nll 0.041. This is too thin to set a noise band for later training, so any gap under about 20–40 pElo between single runs should be treated as noise.

## Caveats

- **Seed study is tiny.** Seed 2 was stopped at step 5,800 (0.77 h) to start the 15×15 run (commit 7e7c14e). The registry label says it "runs to full 31.3h", which is not what happened. Seed 1 was stopped at step 20,440. The two seeds overlap only at steps 1k–5k.
- **One seed per stem** for 5×5 and 15×15. The 3×3 seed spread is the only noise estimate, and it covers only the early steps.
- **nt8y is confounded** (host, build, hyperparameters). Its segment-0 log `dcm_log_20260701-091259.txt` is missing (deleted mid-run ~2026-07-06). Its segment-0 hyperparameters are unverified. The 0.00025 / 0.93 values come from segments 1–3.
  - The repo's `parameters.json` at e96e0b4 (wd 0.0001, lr 0.0005) matches no logged run, so the runs used saved app settings, not that file.
- **nt8y time axis:** seg0 steps 51,000–65,000 have elapsed reconstructed from frozen-file mtimes (registry note), and cum 49,000 has a blank elapsed. Neither affects the 1k–49k step comparison.
- **The 111-row nt8y notch** (cum 73,883–106,883 and 163,883–240,883) is outside this window. There is no data there, and the table does not reach it.
- **Probe noise:** single puzzle probe per checkpoint, quantized pElo, SD ~10–22 about a local mean. Replay-era probe scale only.
- **games_fed:** the registry/CSV `games_fed` column is empty for these runs. The matched-games claim is measured from each run's own log `games=` and from checkpoint `replay_next_game_index`, not modeled.
- **GPU:** the arms never overlapped each other. Zero-byte logs appear during every run (likely brief probe/CLI launches, unverified). Flat ms/step says their load was negligible.
- **Throughput differs by build as well as by arm.** 15×15 (build 2080) logged ~398 ms/step vs 3×3 (2075) ~417. The same 3×3 arch on build 2079 logged ~427, so run-to-run ms/step varies by ±2–3%. Don't read the 15×15 stem as "faster".

## Follow-ups

- A clean 5×5 arm: fresh nt8y (5×5) on the current build, this host, wd 0.0005 / momentum 0.9, to ≥20k steps. That is the only way to settle whether the 5×5 lead is the stem or the recipe.
- A real seed band: two or more seeds of one arch to ≥20k steps (ideally ≥49k), to get the noise band where the arms actually differ.
- Optionally, re-run nt8y at 0.0005 / 0.9 vs 0.00025 / 0.93 to measure the recipe gap directly.

## Audit notes

- **Verified:**
  - Each fresh model's architecture JSON (`__metadata__.architecture`) was diffed field by field against nt8y's manual export `20260701-161012-20260701-3-nT8Y-manual` (20260701-3-nT8Y):
    - 3×3 (both seeds) differs only in `stem_conv_kernel_size` (3 vs 5).
    - 15×15 differs only in `stem_conv_kernel_size` (15 vs 5).
  - Seed 1 and seed 2 architecture JSONs are identical (same hash 086f3417e4e4). Their `content_sha256` weights differ (12626df1… vs ab87ed4b…), confirming different init.
  - Enumerated checkpoints (3×3 step20000, s2 step5000, 15×15 step49000/latest) carry the same architecture hash as their fresh parents, and their `parent_model_id` chains check out.
  - Param counts 1,533,930 / 1,518,570 / 1,725,930 match the tensor-element sums, the `[REPLAY-ARCH]` lines, and the 960·k² stem arithmetic.
  - The "exactly nt8y but X stem" / "byte-identical arch" claims (registry arch_summary, commits 6aa027b, 7e7c14e) are **confirmed**.
  - Hyperparameters were taken from the `[REPLAY-HPARAMS]` lines. Start/end times and last steps were taken from each log.
- **Corrections:**
  - "3x3/5x5/15x15 stems indistinguishable within probe noise" (memory dcm-parity-goal.md) → 3×3 vs 15×15 indistinguishable (mean −2.9 pElo over 20 matched steps). 5×5 nt8y is ahead of both at every horizon (+38 vs 3×3 over 1k–20k, 19/20; +94 vs 15×15 over 21k–49k, 29/29), but it is confounded by host, build and wd/momentum, so it is not a stem effect either way. Evidence: dashboards `data/nt8y*.csv`; `[REPLAY-HPARAMS]` in `dcm_log_20260701-152447.txt` vs `dcm_log_20260711-*.txt`.
  - "seed1 vs seed2 tracked ~7-25" → \|Δ\| 6.4–43.7 pElo, mean 18.6, over only 5 matched points (steps 1k–5k). The ~7–25 range leaves out step 1k (43.7). Evidence: nt8y3x3.csv, nt8y3x3s2.csv.
  - nt8y15x15 "terminated ~cum 46000 / 5.7h" (commit 672999b, memory) → the last logged step is 49,750 at 10:13:15, and the last checkpoint/probe is step 49,000 at elapsed 22,040 s = 6.12 h. The 5.62 h / cum 45k figure was the tick before termination (commit 22d0bdc). Evidence: `dcm_log_20260711-040005.txt` tail; nt8y15x15.csv.
  - nt8y3x3s2 "runs to full 31.3h" (registry label/arch_summary, commit 0183064) → stopped at step 5,800 (~0.77 h) to launch nt8y15x15. Evidence: commit 7e7c14e; `dcm_log_20260711-031104.txt` last step 5800 at 03:56:56.
- **Unverified:**
  - nt8y segment-0 hyperparameters and exact host: the log is deleted, and checkpoint metadata carries no hyperparameters. What is known is build 2009 / git e96e0b4 and a `/Volumes` corpus path.
  - What the zero-byte logs during each run were.

## Related: wxil

- wxil (8-block 15×15 @16, 1,052,183 params) ran just before this series on the same corpus and recipe. It asks a separate question (deep-narrow vs nt8y's shallow-wide), so it has its own write-up: [`../20260710-wxil-deep-narrow/README.md`](../20260710-wxil-deep-narrow/README.md).
- At matched steps it trails every stem arm by ~136–180 pElo.
