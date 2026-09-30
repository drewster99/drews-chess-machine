# 2026-06-29 — Small-architecture corpus-replay parity (31.3 h)

**Status:** done (mini2b, qeu8, nt8y, coxw reached 31.3 h; yKkK paused at 27.87 h and never resumed; t97x abandoned after 4,433 steps)

## Question

Given the same std corpus (`20260624-192615-w3aA5b`, lichess 2026-05, 20,935,171 games) and the same replay trainer, which small architecture learns the most per hour, per step and per game? Every run was taken to the same training-time budget, 31.3 h of `elapsed_train_sec`, which was meant to match v5.

## Setup

- **Corpus:** `20260624-192615-w3aA5b` for every run. Checked against `replay_corpus_id` in each run's checkpoint headers. It is a disk-full-truncated prefix of the month (46 shards). Size of one epoch: **20,935,171 games**. Derivation: yKkK resume2's `done: … gamesFed=32315867 epochs=1` (`dcm_log_20260710-015115.txt`) minus its final header `replay_next_game_index=11380696` at `replay_epoch=1`.
- **Trainer:** offline corpus replay (`--replay-corpus`), batch 4096. Every segment reports `pre-filled: … gamesFed=7563`, so **every resume restarts the corpus at game index 0**. Resumed segments therefore replay the same opening games of the corpus again.
- **Metrics:** `pElo` and `nll` come from the puzzle probe on the replay-era probe scale, from `documentation/dashboards/data/<key>.csv`. The time axis is `elapsed_train_sec` (sleep-clamped).
- **Machine:** segments before 2026-07-07 read the corpus from `/Volumes/20260624-192615-w3aA5b` (the earlier host). Segments from 2026-07-07 on read it from the local Corpora store on the current Apple M5 Max. Per-segment `device` is unverified: the registry has no `device` field for these runs.

Architectures, taken from the embedded `architecture` JSON in each run's checkpoint headers. Every parameter count is the sum of tensor elements in the file and agrees with the log/TSV `[REPLAY-ARCH]` line:

| run | params (exact) | architecture (verified) |
|---|---:|---|
| v5 (ref) | 8,447,028 | basic30 → stem 7×7→128 · 5×[7×7+7×7 @128, SE scale+bias/4, ReLU/pre, ReZero 0.447·tanh, LN-out] · policy intermediate_conv (pre-conv 128) · value WDL(16→FC128) · bf16 |
| qeu8 | 3,929,393 | stem 7×7→64 · 2×[15×15+15×15 @64, SE scale+bias/4, ReLU/pre, ReZero 0.5, LN-out] · policy intermediate_conv (pre-conv 512) · value WDL(16→FC64) · bf16 |
| mini2b | 2,252,689 | stem 7×7→128 · blk0 7×7+7×7 @128 → blk1 3×3+3×3 @128, no SE, ReLU/pre, ReZero 0.5, LN-out · policy intermediate_conv (pre-conv 128) · value WDL(16→FC128) · bf16 |
| coxw | 1,864,336 | stem 5×5→128 · 1×[7×7+7×7 @128, no SE, ReLU/pre, ReZero 1.0, LN-out] · policy intermediate_conv (pre-conv 128) · value WDL(16→FC128) · bf16 |
| t97x | 1,696,959 | stem 7×7→64 · 2×[9×9+9×9 @64, SE scale+bias/4, **GELU**/pre, **no ReZero**, LN-out] · policy intermediate_conv (**pre-conv 5**) · value WDL(**64**→FC64) · bf16 |
| nt8y | 1,533,930 | stem 5×5→32 · 3×[15×15+15×15 @32, SE scale+bias/4, ReLU/pre, ReZero 0.333, LN-out] · policy intermediate_conv (pre-conv 512) · value WDL(16→FC64) · bf16 |
| ykkk | 242,609 | stem 3×3→64 · 2×[3×3+3×3 @64, SE **attenuate_only**/4, ReLU/pre, ReZero 0.707, LN-out] · policy **simple_conv** · value WDL(16→FC64) · bf16 |

## Runs

In the tables below, "cum" is the registry `cumstep_base + step`. "Final header" means the `-replay-latest` file for that segment, identified by its `__metadata__`.

| run | seed ModelID | segments → final ModelID (training_step, builds) | surviving logs |
|---|---|---|---|
| mini2b | 20260629-3-3MIV | y5u7 (13,464) → BEKK (120,695) → SvRu (8,000) → **znR7** (114,000); builds 2007/2009/2009/2013 | seg0–2 logs **missing** (`dcm_log_20260629-091821`, `-192108`, `20260701-080946`); seg3 `dcm_log_20260705-091255.txt` present |
| coxw | 20260629-5-Coxw | yqMI (55,550) → **avoB** (277,000); builds 2009/2033 | seg0 `dcm_log_20260629-112618` **missing**; seg1 `dcm_log_20260709-000611.txt` present |
| ykkk | 20260630-1-YkKk | 6y0s (40,677) → 0Iwe (162,330, final, exactly 1 epoch) → **amlg** (250,803, abort); builds 2009/2009/2053 | seg0 and seg1 logs **missing**; seg2 `dcm_log_20260710-015115.txt` present |
| t97x | 20260630-3-T97X | **ASdQ** (4,433, abort); build 2009 | `dcm_log_20260629-185604` **missing** |
| nt8y | 20260701-3-nT8Y | CIvL (65,883) → bOYQ (70,779) → 3CZF (15,000) → cslu (140,000) → **kEiZ** (21,086); builds 2009/2011/2013/2013/2025 | seg0 **missing**; seg1–4 present |
| qeu8 | 20260702-7-Qeu8 | GLu5 (41,407) → Lnji (67,508) → **PVZp** (67,000) = parity point; later resume3 → Ejp0 (outside this experiment) | all four present |
| v5 | see `documentation/v5-lineage.md` | parity region is inside segs 0–2 (M5 VM) | segs 0–2 logs permanently lost |

## Results

**End-of-budget table.** "At 31.3 h" means the last probed row at or before 31.3 h. A cell is blank when the run never reached that point. No values are clamped or carried forward.

| run | params | cum steps at end of budget | elapsed at end (h) | games fed (measured) | steps/h | pElo / nll at 31.3 h | peak pElo ≤ 31.3 h (cum, h) | best nll ≤ 31.3 h (cum) |
|---|---:|---:|---:|---:|---:|---|---|---|
| v5 (ref) | 8,447,028 | 84,901 | 31.248 | 10,946,978 | 2,717 | 1519.0 / 2.227 | 1589.3 (77,901, 28.67) | 2.121 (62,901) |
| qeu8 | 3,929,393 | 175,915 | 31.330 | 22,642,836 | 5,615 | 1575.4 / 2.127 at 31.33 h (1590.9 / 2.075 at 31.14 h) | **1630.9** (169,915, 30.20) | **2.0119** (171,915) |
| nt8y | 1,533,930 | 312,748 | 31.483 | 40,298,487 | 9,934 | 1509.8 / 2.198 (at 31.16 h); 1605.3 / 2.102 at 31.48 h | 1629.9 (302,662, 29.94) | 2.0564 (297,662) |
| mini2b | 2,252,689 | 256,159 | 31.294 | 33,021,918 | 8,186 | 1530.3 / 2.140 (at 31.294 h) | 1544.2 (253,159, 30.92) | 2.1401 (256,159) |
| coxw | 1,864,336 | 332,550 | 31.313 | 42,850,545 | 10,620 | 1455.9 / 2.236 (31.22 h); 1454.4 / 2.234 at 31.31 h | 1493.4 (330,550, 31.13) | 2.1937 (330,550) |
| ykkk | 242,609 | 453,007 | 27.873 | 58,496,844 (through header step 250,803 = cum 453,810) | 16,253 | | 1331.5 (452,007, 27.80) | 2.3851 (341,007) |
| t97x | 1,696,959 | 3,450 (CSV) / 4,433 (header) | 0.3135 | 581,546 (at 4,433) | | | 815.6 (3,450, 0.31) | 3.572 (3,450) |

**pElo / nll at matched training time.** Each cell is the last probed row at or before T, shown as pElo / nll (cum).

| T (h) | v5 | qeu8 | nt8y | mini2b | coxw | ykkk |
|---:|---|---|---|---|---|---|
| 3.0 | 1105 / 2.720 (8,000) | 1207 / 2.522 (16,000) | 1298 / 2.455 (32,000) | 1245 / 2.528 (24,464) | 1277 / 2.457 (29,000) | 1195 / 2.553 (67,677) |
| 5.5 | 1208 / 2.568 (15,000) | 1352 / 2.360 (30,000) | 1380 / 2.358 (58,000) | 1330 / 2.404 (46,464) | 1333 / 2.380 (54,000) | 1200 / 2.607 (124,677) |
| 8.9 | 1318 / 2.428 (23,000) | 1440 / 2.249 (49,407) | 1457 / 2.249 (72,883 @ 6.82 h †) | 1396 / 2.331 (66,464 @ 7.85 h †) | 1382 / 2.318 (90,550) | 1239 / 2.538 (201,677) |
| 17.0 | 1422 / 2.309 (45,000) | 1522 / 2.126 (97,407) | 1565 / 2.180 (163,662 @ 15.62 h †) | 1451 / 2.272 (145,159) | 1374 / 2.319 (176,550) | 1245 / 2.514 (309,007) |
| 25.9 | 1538 / 2.164 (69,901) | 1608 / 2.067 (146,915) | 1509 / 2.226 (268,662) | 1517 / 2.192 (212,159) | 1408 / 2.292 (272,550) | 1225 / 2.530 (427,007) |
| 27.8 | 1522 / 2.224 (74,901) | 1568 / 2.127 (156,915) | 1573 / 2.166 (287,662) | 1515 / 2.228 (227,159) | 1453 / 2.206 (293,550) | 1332 / 2.447 (452,007) |
| 31.3 | 1519 / 2.227 (84,901) | 1591 / 2.075 (174,915) | 1510 / 2.198 (310,662) | 1530 / 2.140 (256,159 @ 31.294 h) | 1456 / 2.236 (331,550) | |

† These runs have no probed row between the time shown and T. nt8y's 111 rows (cum 73,883–106,883 and 164,662–241,662) are blank-pElo because of a disk-full event whose checkpoints are unrecoverable. mini2b has no probes between cum 66,464 and its resume region.

**pElo / nll at matched step** (last probed row at or before S; blank when not reached):

| cum step | v5 | qeu8 | nt8y | mini2b | coxw | ykkk |
|---:|---|---|---|---|---|---|
| 20,000 | 1345 / 2.367 (7.31 h) | 1330 / 2.391 (3.55 h) | 1243 / 2.554 (1.87 h) | 1228 / 2.513 (2.30 h) | 1218 / 2.520 (2.03 h) | 1033 / 2.865 (0.87 h) |
| 40,000 | 1404 / 2.288 (14.72 h) | 1431 / 2.293 (7.21 h) | 1351 / 2.426 (3.73 h) | 1284 / 2.451 (4.67 h) | 1289 / 2.439 (4.06 h) | 1115 / 2.721 (1.69 h) |
| 73,000 | 1550 / 2.170 (26.83 h) | 1451 / 2.255 (12.63 h) | 1457 / 2.249 (6.82 h) | 1396 / 2.331 (7.85 h) | 1345 / 2.366 (7.16 h) | 1190 / 2.612 (3.21 h) |
| 175,915 | | 1575 / 2.127 (31.33 h) | 1565 / 2.180 (15.62 h) | 1476 / 2.228 (20.61 h) | 1392 / 2.314 (16.83 h) | 1237 / 2.580 (7.74 h) |
| 250,000 | | | 1557 / 2.135 (23.99 h) | 1505 / 2.164 (30.43 h) | 1384 / 2.335 (23.70 h) | 1294 / 2.476 (12.43 h) |
| 330,000 | | | | | 1456 / 2.242 (31.04 h) | 1263 / 2.506 (18.48 h) |

**Cross-model strength.** These ratings come from the 38-engine round-robin in `experiments/20260708-arena-38engines`: Ordo, Stockfish anchored at 1320, 40/5 s, `Temperature = 0` = the old decaying `.arena` schedule. Engines are mapped to model IDs via `engines_models.tsv`.

| engine (model_id) | position in its run | rating ± err |
|---|---|---:|
| DCM v5 / v5…wd2.5e4-m93 (both **20260629-1-Uf4p**, the same file) | v5 cum 100,320 (≈36.8–37.8 h on the current axis, **past** parity) | 603.8 / 605.9 ± ~12 |
| Qeu8-resume2 (PVZp) | qeu8 cum 175,915, **31.33 h (parity point)** | **588.4** ± 10.8 |
| nT8Y-resume3 (cslu) | nt8y resume3 step 140,000 (cum 291,662, ~28.36 h, the resume4 base) | 576.9 ± 11.1 |
| Qeu8 (Lnji) | qeu8 cum 108,915 (~18.9 h) | 567.1 ± 11.9 |
| mini2b-3MIV-resume3 (znR7) | mini2b cum 256,159, **31.29 h (parity)** | 561.8 ± 10.6 |
| nT8Y-resume4 (kEiZ) | nt8y cum 312,748, **31.48 h (parity run's end)** | 532.0 ± 11.2 |
| mini1b-Coxw (yqMI) | coxw step 55,550 (5.58 h) — not the parity checkpoint | 462.8 ± 11.8 |
| mini-YkKk-resume (0Iwe) | ykkk cum 203,007 (8.95 h) | 376.0 ± 10.5 |
| mini-YkKk (6y0s) | ykkk cum 40,677 | 305.4 ± 11.6 |
| T97X (ASdQ) | step 4,433 (~0.35 h) | 181.6 ± 11.0 |

Head-to-head results from `h2h.txt`, 100 games each:
- Qeu8-resume2 vs v5: 41-20-39 (51.0%).
- Qeu8-resume2 vs nT8Y-resume3: 51.0%.
- Qeu8-resume2 vs mini2b-resume3: 52.5%.
- Qeu8-resume2 vs nT8Y-resume4: 60.0%.
- Qeu8-resume2 vs coxw (5.6 h): 63.0%.
- Qeu8-resume2 vs yKkK-resume: 77.5%.
- nT8Y-resume4 vs nT8Y-resume3: 38.0%. The later nt8y checkpoint lost to the earlier one.

## Conclusion

- **Per hour:** qeu8 is the best small architecture. It has the best nll at every matched time from 17 h on (2.075 vs 2.140–2.236 near parity; at 5.5–8.9 h it is tied with nt8y), the best peak pElo within the budget (1630.9, statistically tied with nt8y's 1629.9), and the best arena result at parity (588.4, 51% against v5 cum 100,320, which had *more* training time). nt8y is a close second: its best nll ≤ 31.3 h is 2.056 vs qeu8's 2.012. mini2b is third, coxw fourth, yKkK far behind.
- **Per step:** larger is better. At 40k steps the ranking is qeu8 1431 ≈ v5 1404 > nt8y 1351 > coxw/mini2b ~1285 > yKkK 1115. The small nets win on time only because they take 2–6× more steps per hour.
- **Per game:** at parity qeu8 has seen the fewest games of the small nets (22.6M, about 1.08 epochs of distinct data counting resume repeats) and still leads. nt8y (40.3M), mini2b (33.0M) and coxw (42.9M) consumed 1.5–1.9× more games for lower nll. yKkK (58.5M games, 2.8 epoch-equivalents) is capacity-bound: nll stays ≥ 2.385.
- **Capacity floor:** yKkK (0.24M params) plateaus around pElo 1200–1330 / nll ~2.45–2.55 whatever the extra time or data. coxw (1 block) is flat from 5.6 h to 17 h (1335 → 1374) before a late rise to 1454–1493.
- **v5 is not faster per hour:** at 31.25 h v5 is at 1519 / 2.227, behind qeu8, nt8y and mini2b at the same time. Its advantage appears only much later in its lineage (peak 1770.5 at cum 638,826).
- **t97x gives no signal.** It was aborted at 4,433 steps (0.31 h in the CSV) and its log is lost. The GELU/no-ReZero/pre-conv-5 recipe was never tested at length. Its early numbers (812 at 0.22 h, below qeu8/mini2b/coxw at similar times) are not evidence either way.

## Caveats

- **Probe noise:** single-row pElo swings ±30–60 between adjacent probes (e.g. v5 1576.5 → 1519.0 between 30.88 h and 31.25 h; nt8y 1629.9 at 29.94 h vs 1509.8 at 31.16 h). Read nll and peaks together, not single endpoints.
- **One seed per architecture.** No run was repeated. The nt8y seed-variance study in `20260710-nt8y-stem-kernel-and-seed-variance` gives the scale of seed noise.
- **Resumes re-read the corpus from game 0 and reset optimizer velocity.** mini2b, ykkk and nt8y resumes were also flagged `bn_reset`. Segment boundaries differ per run, so each run saw a different games/repeat mix. Several dips line up with resumes.
- **Mixed builds** (2007 → 2053) across segments. Machine changes also occurred mid-lineage for nt8y, coxw and ykkk: the corpus path switched from `/Volumes/` to local.
- **games_fed** is the sum of each segment's final checkpoint header (`replay_epoch × 20,935,171 + replay_next_game_index`). Steps trained after a segment's last save and before a kill are not counted: mini2b seg3 reached step 114,700 vs its save at 114,000, and ykkk's header reaches 250,803 while its CSV ends at cum 453,007. The dashboard CSVs for these runs carry no measured `games_fed` (older schema; nt8y.csv and coxw.csv gained the column, blank, when `replay.py recompute` rewrote them on 2026-09-29).
- **Arena ratings** are for whichever `-replay-latest` file existed on 2026-07-08. Several are not at parity: coxw at 5.6 h, yKkK at 8.95 h, v5 at ~37 h. The arena also used sampling noise through ~ply 45.
- **pElo scale:** all replay-era values here are on the current probe scale and comparable with each other. They are not comparable to May–June self-play pElo.

## Follow-ups

- Re-run the arena with frozen parity checkpoints for every run: coxw avoB, yKkK amlg, and a v5 checkpoint at ~31.3 h. Use argmax play and an opening book.
- Seed repeats of qeu8 vs nt8y. Their ≤ 31.3 h peaks differ by 1 pElo.
- yKkK resume3, to finish 27.87 → 31.3 h, was planned but never ran.

## Audit notes

- Verified: every parameter count and architecture comes from the safetensors `__metadata__.architecture` plus a tensor-element sum of a surviving checkpoint, and matches the `[REPLAY-ARCH]` log line / `engines_models.tsv`.
- Verified: segment chaining. For every run, registry `cumstep_base` was checked against the headers (`training_step`, `parent_model_id`) and the logs (`start-model:` / `done:` / `saved trainer`).
- **Correction:** v5 param count 8,450,000 (registry) → **8,447,028**. Evidence: log `dcm_log_20260702-201756.txt` [REPLAY-ARCH], `engines_models.tsv`.
- **Correction:** coxw 1.86M / 1,860,000 (registry) → **1,864,336**. Evidence: header avoB / yqMI tensor sum.
- **Correction:** t97x 1.70M / 1,700,000 (registry) → **1,696,959**. Evidence: header ASdQ.
- **Correction:** "v5 = 31.10 h" (parity reference in memory `dcm-parity-goal.md`; the 2026-07-05 dashboard showed v5 at cum 99,901 = 31.1 h) → on the current `v5.csv` time axis, cum 99,901 is at **36.82 h**, and v5 at 31.3 h is **cum 84,901, 31.248 h**. v5's elapsed axis was re-derived after 2026-07-05. "31.3 h ≈ v5 parity" now means v5 cum ~85k, not v5's end of M5-VM training (cum 100,320).
- Verified with detail: "qeu8 DONE — 31.330h, peak pElo 1631". Confirmed at cum 175,915 = 112,789.5 s = 31.330 h. The peak ≤ 31.3 h is **1630.93** at cum 169,915 (30.20 h). The value exactly at 31.330 h is 1575.4.
- Verified: "qeu8 resume3 peak 1742.1 @ meta 681000". Confirmed: CSV cum 856,915 = 175,915 + 681,000, pElo 1742.09, nll 1.904, 154.19 h. The run's best nll is 1.8734 at cum 1,509,915. Both are outside this experiment's budget.
- **Correction (detail):** "coxw 31.31h cum 332000 pElo 1454" is correct for the *last row* (1454.35, 31.313 h). It omits that coxw's **peak** was 1493.36 at cum 330,000 (31.13 h). The endpoint understates the run.
- **Correction (detail):** "yKkK paused 27.87h cum 453007 pElo 1236" is correct for the last row (1236.05, 27.873 h). The row immediately before (cum 452,007, 27.80 h) is its **peak, 1331.51**. The 95-point drop between adjacent probes is probe noise, not a trend. yKkK's best nll is 2.3851 at cum 341,007.
- **Finding (cum axis off by a few hundred to 1,779 steps; fixed 2026-09-29 in `registry.json`, and the CSVs re-derived with `replay.py recompute`):**
  - **nt8y:** resume2 started from bOYQ (`training_step` 70,779), but the registry base 135,883 assumes 70,000. That makes every later row 779 low. resume4 started from cslu at step 140,000 (`dcm_log_20260706-193601.txt` last save `step=140000`), but base 289,883 assumes 139,000. That makes resume4 rows **1,779 low** in total.
  - **coxw:** resume started from yqMI at step 55,550 with base 55,000, so those rows are 550 low.
  - The elapsed of those steps is also missing from the time axis. The effect is small (< 0.2 h).
  - **Fix applied:** nt8y `cumstep_base` resume2 135,883 → 136,662, resume3 150,883 → 151,662, resume4 289,883 → 291,662 (from the start models' headers: bOYQ `training_step` 70,779, 3CZF 15,000, cslu 140,000), and their pinned `elapsed_base_sec` 46,279.8 / 51,951.4 / 101,447.6 → 46,534.1 / 52,205.7 / 102,090.9 s (seg 1's clamped training time to its last logged step 70,750, then resume2's to 15,000 and resume3's to 140,000, from their logs). coxw's resume `cumstep_base` 55,000 → 55,550 (yqMI header). coxw's elapsed base is unchanged: seg 0's log is gone, so the time of its steps 55,000–55,550 is unmeasured. The nt8y and coxw cells in the tables above were recomputed from the corrected CSVs. Consequences: nt8y's run actually ended at 31.483 h (cum 312,748), not 31.304 h, so its last probed row at or before 31.3 h is now 1509.8 / 2.198 at 310,662 (31.16 h); its 25.9 h cell changed from 1573 / 2.133 to 1509 / 2.226. coxw's elapsed axis is unchanged, but its cum-step cells from 55,550 on moved, which changes its 73k / 250k / 330k matched-step cells.
  - `registry.json` params for v5 / coxw / t97x were corrected to the exact counts above at the same time.
- **Unverified:**
  - Per-segment device/machine: the registry has no `device` for these runs, and the seg0 logs are missing.
  - t97x's reason for abort: log `dcm_log_20260629-185604.txt` is missing. Only the header note "corpus replay abort @ step 4433" survives.
  - nt8y's 111 blank-pElo rows are permanently unrecoverable (memory `dcm-parity-goal.md`; not re-checked beyond the blank count of 111 in `nt8y.csv`).
