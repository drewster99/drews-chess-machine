# 2026-05-14 — Self-play lineage sweep: 3×3 towers, learning rate, and the "Track-2" blow-up

**Status:** abandoned. The analysis ended on 2026-06-11. Its proposed decisive tests were never run: WjRY's exact config at lr 1e-3, and a single-axis 3×3-vs-7×7 flip. The finding therefore stays a correlation.

## Question

Why did some self-play lineages (Exp 4 / jaq1, LMGh, WjRY) "blow up" while others trained stably? A blow-up means the out-of-distribution wide-set puzzle NLL explodes while in-distribution training metrics stay flat. Is the cause capacity, width, depth, a code regression, precision, kernel/stem geometry, or learning rate?

The folded conclusion in `documentation/ARCH_EXPERIMENTS.md` (Exp 4, §6) says the cause is **lr above an optimization-stability threshold for shallow/narrow 3×3 towers**. This write-up re-checks that claim, lineage by lineage, against the logs.

## Setup

- **Source of the investigation:** `documentation/archive/OVERNIGHT_INVESTIGATION.md`, "TRACK 2", iterations 3–8 (2026-06-09 → 06-11). ARCH_EXPERIMENTS.md Exp 4 §6 holds the folded one-paragraph conclusion. The archive audit `documentation/archive/DOCS_AUDIT_2026-06-23.md` line 126 directed the fold.
- **All runs are GUI Play-and-Train self-play.** They share these settings: batch 4096, clip 30, illM 1.0, promote ≥ 0.53, 400-game arenas, sp.tau 1.00/0.50/0.007, ar.tau 0.60/0.20/0.020, basic30 input. The `·√b` factor on lr is `sqrt(b/4096)`, a no-op at batch 4096.
- **Machine:** the single local Mac (M4 Pro). The recording builds are listed per run below.
- **Lineages written up elsewhere (linked, not duplicated):**
  - bzw3 / Exp 1: `experiments/20260601-5block-7x7-rezero-se`
  - 2Gd1 / Exp 2: `experiments/20260606-full10ply200-input`
  - eaRt / Exp 3: `experiments/20260607-full10ply10reps210-input`
  - jaq1 / Exp 4: `experiments/20260608-256wide-dual-kernel`
  - JhJQ / Exp 5: `experiments/20260610-exp1-recheck-bugfixed`
  - eBNC / Exp 7: `experiments/20260612-block-groups-ebnc`

## Runs

The lineages are keyed by champion base ModelID, as in `documentation/dashboards/selfplay_registry.json`. Each log was scoped by lineage tag + build + contiguous steps. Resumes that rewound were treated as last-writer-wins.

| lineage | base ModelID | logs (first → last) | builds | architecture (source) | surviving checkpoint (by `__metadata__` / session.json) |
|---|---|---|---|---|---|
| KbHZ | 20260514-1-KbHZ | 49 logs, `dcm_log_20260514-020048` → `20260524-182125`, then June continuation `20260610-123913` | 1093 → 1357; 1804 | v3 post · stem 3×3→128 · 8×[3×3, SE/4, activation_gated, no-ReZero] · policy simple_conv · value WDL(1→FC64) · **float32** · **2,483,667** params ([ARCH] line in `dcm_log_20260610-123913.txt`; arch_hash 0x13ba0b55 in every May [APP] line) | `Sessions/20260611-212501-20260514-2-Ko63-manual.dcmsession`: champion `20260514-1-KbHZ-22` / trainer `-23`, training_step 532369 (safetensors metadata). Also the May `…-Ko63-manual` pair at step 494,927 (champion `-17`, trainer `-18`, session.json; .dcmmodel) |
| sMe9 | 20260525-1-sMe9 | 29 logs, `20260524-205101` → `20260529-135923` | 1360 → 1461 | same 8×3×3 v3 fp32 net (arch_hash 0x13ba0b55 → 2,483,667 params per the arch-hash table in memory `project_two_running_sessions_2026-06-01.md`) | `Sessions/20260529-182349-20260525-2-IWkd-manual.dcmsession`: champion `sMe9-32` / trainer `sMe9-33`, step 373,416 (session.json; .dcmmodel) |
| ysdg | 20260529-10-ysdg | `20260529-154054`, `20260529-202707`, `20260530-160307` | 1474, 1482, 1495 | 16-block v3 3×3 fp32 (arch_hash 0x5347c53d → 4,934,867 params, arch-hash table) | `Sessions/20260531-024912-20260529-11-G5w2-promote.dcmsession`: champion `ysdg-3` / trainer `ysdg-4`, step 71,604 |
| KXvb | 20260531-3-KXvb | `20260531-020125` | 1513 (git c249df2) | 12-block v4 3×3, 128ch (arch_hash 0xbad32ced → 3,898,139 params). `ChessNetwork.dataType = .bFloat16` at c249df2, which predates the fp32-master commit 0626cec | none found |
| LWKa | 20260531-9-LWKa | 9 logs, `20260531-150111` → `20260601-090740` | 1528 → 1562 | same 12-block v4 3×3 net as KXvb. **dataType is .float32 at 0626cec (first log, steps 1–7,396)** and .bFloat16 from 293f3e1 on | none (saved-session names used `WcRm`; none survive) |
| WjRY | 20260609-14-WjRY | `20260609-104356` | 1795 | v4 pre · stem 3×3→128 · 8×[3×3, SE+/4, clean_add, ReZero] · policy intermediate_conv · value WDL(16→FC128) · bfloat16 · **2,664,087** ([ARCH] "built champion") | none (the `tGOH` phantom-ID saves no longer exist) |
| LMGh | 20260609-12-LMGh | `20260609-004301` | 1793 | v4 pre · stem 3×3→32 · 50×[3×3, SE+/4, clean_add, ReZero] · WDL(16→FC32) · bfloat16 · **1,022,481** ([ARCH]). ReZero α mis-set to 1/√12 by the Build-screen regression (per the investigation) | none |
| wTp3 | 20260614-4-wTp3 | `20260614-014435` (+ three ≤4-line resumes 06-25/07-07) | 1876 | v4 · stem 3×3→128 · 4×[3×3+3×3 @128, SE+/4, relu/pre, clean_add, ReZero(0.5), drop*1] · WDL(16→FC128) · **float32** · **1,430,035** ([ARCH]) | `Sessions/20260614-133904-20260614-5-t9sX-promote.dcmsession`: champion `wTp3-3` / trainer `wTp3-4`, training_step 35496 (safetensors metadata) |

## Results

### Per-lineage table

Every value comes from the lineage's own `[STATS]` / `[ARENA]` lines unless noted.

- **Promotions** = `promoted=1` arena verdicts on the surviving (non-rewound) line.
- **pElo** is the in-app WIDE (4,435-position) lichess probe **on the recording build's scale** (`selfplay_probe/<key>.csv`). It exists only for runs recorded after the probe was added.
- **"reg. endpoint"** is the selfplay_registry `endpoint_pElo`. It comes from re-probing the final model with a **later binary**, so it is on a different scale from the in-training column. Neither is comparable with July+ replay-era pElo (~900–1770).

| lineage | dates | precision | stem | tower | params | lr (displayed = effective) | μ | wd | steps reached | promotions | in-training wide pElo: peak / end (recording scale) | reg. endpoint pElo / nll (later-binary scale) | blew up? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KbHZ | 05-14 → 05-24; 06-10 → 06-11 | fp32 | 3×3 | 8×3×3 v3 | 2,483,667 | **1.5e-4** (steps 1 → ~84.7k) → **5e-4** (~84.3k → ~435k) → **1e-3** (~435k → 532,371). A **1e-2** branch at 180,042–182,594 (builds 1237/1239) was abandoned by a rewind to 175,543 | 0.65 (0.85 only in the 05-24 fragments) | 1e-4 (1e-3 briefly) | 494,933 (May); 532,371 (June continuation) | **22** on the surviving line (24 logged, 2 of them in abandoned 181k branches) | May: no probe. June: 602.1 @ 494,956 / 574.9 @ 532,351 | 572.3 / 6.30 | **no**. NLL 4.4 → 6.3 in June while pElo held ~571–575. This is the sharpness confound (pEnt ~1.7), not a blow-up |
| sMe9 | 05-24 → 05-29 | fp32 | 3×3 | 8×3×3 v3 | 2,483,667 | **1e-3** throughout | 0.85 → 0.90 (from 135k) | 1e-3 / 1e-4 / 1e-3 | 377,302 (max seen; later launches resumed at 371,205–373,416) | **32** | no probe | 702.7 / 3.49 | no evidence of one (no OOD probe existed) |
| ysdg | 05-29 → 05-31 | fp32 | 3×3 | 16×3×3 v3 | 4,934,867 | 1e-3 | 0.90 | 1e-3 | 78,222 | **3** | no probe | 512.5 / 3.94 | no evidence of one |
| KXvb | 05-31 | bf16 (no fp32 masters yet, per commit) | 3×3 | 12×3×3 v4 | 3,898,139 | **1e-2** | 0.90 | 1e-4 | 51,551 | **0** | no probe | 529.1 / 4.41 | no evidence of one |
| LWKa | 05-31 → 06-01 | fp32 for steps 1–7,396, then bf16 + fp32 masters | 3×3 | 12×3×3 v4 | 3,898,139 | **1e-2** | 0.90 | 1e-4 | 106,740 | **10** | probed 94,807–106,664 only: 815.4 @ 104,907 / 726.6 @ 106,664. NLL flat 3.48–3.69 | 606.9 / 6.05 | **no** |
| WjRY | 06-09 → 06-10 | bf16 + fp32 masters | 3×3 | 8×3×3 v4 | 2,664,087 | **1e-2** | 0.90 | 1e-4 | 107,548 | **7** (last @ 62,415) | 1,000-step bucket peak 673.1 @ 61,010. Last probe 361.9 / NLL 16.87 @ 98,950 | 527.2 / 14.63 | **yes, ~73.7k**. NLL low of 3.36 @ 72,676, first > 6 @ 73,729. Median pElo 619 (49–66k) → ~495 (74–90k) → 392 (98.7k+) |
| LMGh | 06-09 | bf16 + fp32 masters | 3×3 | 50×3×3 v4, 32ch (α bug) | 1,022,481 | **1e-2** | 0.90 | 1e-4 | 87,507 | **3** | bucket peak 712.0 @ 36,054. Last probe 630.9 @ 79,128 | 625.6 / 7.86 | **yes by NLL, ~8.8k** (first > 6 @ 8,776; median ~10–11 thereafter). **pElo did not fall** (median ~494 → ~560–580) |
| wTp3 (after the investigation) | 06-14 | **fp32** | 3×3 | 4×[3×3+3×3] v4, dropout 0.3 | 1,430,035 | **1e-2** | 0.90 | 5e-4 | 35,586 | **3** | bucket peak 594.9 @ 32,960. Last probe 517.4 @ 35,476 | 599.7 / 4.45 | **intermittent**. 434 of 1,418 probes had NLL > 6 (max 16.50 @ 24,650), while median pElo held ~572–581 |

### What the investigation concluded, by iteration (preserved)

- **Iteration 3:** capacity refuted. The 1.02M-parameter LMGh blew up while the 8.45M/9.57M 7×7 runs did not.
- **Iteration 4:** code regression refuted. No training-path commits landed between the Exp 3 and Exp 4 builds. The lead narrowed to "3×3 second conv / 3×3 stem".
- **Iteration 5:** MPSGraph conv numerics across graph.run, executable .level0 and executable .level1 are bit-identical. This held at batch 128, 512 and 4096 and under six ULP-adversarial regimes. The report is still at `/tmp/conv_kernel_path_numerics_report.txt`.
- **Iteration 6:** the super-table of 8 lineages (LWKa, bzw3, 2Gd1, eaRt, jaq1, LMGh, WjRY, JhJQ). Its point 2: "current-era (config-builder/executable) runs with a 3×3 STEM all blew up (3/3); all 7×7-stem runs are stable (3/3)." It separately notes that the old-era 3×3 LWKa was stable to ~106k.
- **Iteration 7:** the pre-bf16 May lineages were recovered. The hypothesis sharpened to lr 1e-2 with μ 0.90, an effective step lr/(1−μ) ≈ 0.10. The comparison figure was 0.001/0.35 ≈ 0.003 "at KbHZ's lr", a ~35× gap.
- **Iteration 8:** a retro-probe of all 80 then-saved session checkpoints (`--probe-model`). Findings:
  - The damage rides inside promoted candidates at onset (jaq1 gen 4 probed NLL 6.40; LMGh promoted a champion at NLL 13.1).
  - WjRY's saved champions never degraded; the rot stayed in the post-gen-7 trainer.
  - sMe9 (11.1% top-1 / 709) is the best 3×3 ever recorded.
  - Oddly, its STATUS line still lists "learning rate" among the *eliminated* hypotheses. That contradicts iteration 7.

## Conclusion

- **Part of the folded claim holds.** WjRY is an 8×3×3, 3×3-stem v4 bf16 tower at lr 1e-2 / μ 0.90. Its OOD NLL broke at ~73.7k steps and pElo fell with it. The same 8×3×3 shape (v3, fp32) at lr ≤ 1e-3 ran to 494,933 steps (KbHZ, 22 promotions) and 377,302 steps (sMe9, 32 promotions) with no sign of instability. The caveat for May: no OOD probe existed then.
- **Much of the claim is overstated or wrong in detail (see Audit notes):**
  - **KbHZ was not "~495k steps at lr 1e-3".** Only ~60k of its May steps (≈435k → 494,933) were at 1e-3. The rest ran at 1.5e-4 and 5e-4, i.e. an even lower effective step. **sMe9 is the cleaner lr-1e-3 comparator:** 1e-3 for its whole run, but μ 0.85–0.90, so its effective step is 0.007–0.01, a ~10× gap to WjRY rather than 35×.
  - **"Every 3×3-stem run in the bf16/v4 era blew up (3/3)" is false as worded.** KXvb and LWKa were 3×3-stem, v4, bf16 (LWKa after step 7,396), at lr 1e-2 / μ 0.90, and neither blew up. LWKa reached 106,740 steps, past WjRY's onset, with flat NLL. The 3/3 count (jaq1, LMGh, WjRY) only holds for the **post-06-02 config-builder / compiled-executable era**, which is how iteration 6 actually phrased it.
  - The two surviving counter-examples are 12-block towers. Iteration 7 explains them as "a smaller per-block ReZero gain". That explanation is untested.
- **The confounds were never separated.** Between the stable May runs and the blow-ups, all of these changed at once:
  - precision (fp32 → bf16)
  - architecture (v3 → v4: pre-activation, ReZero, scale-bias SE)
  - lr (≤ 1e-3 → 1e-2)
  - momentum (0.65 → 0.90)
  - the entropy bonus / drawPen (on → off)
  - the training-step execution path (graph.run → compiled executable, 06-02)

  The only data point that varies precision alone is wTp3 (fp32, 3×3 stem, lr 1e-2). It showed intermittent NLL spikes to 16.5 with flat pElo. That is suggestive of an lr/geometry cause rather than a bf16 one, but it is a single short run with dropout 0.3 and wd 5e-4 also changed.
- **The LMGh "blow-up" is NLL-only.** Its argmax pElo kept rising, so NLL alone is a weak blow-up criterion. KbHZ also read NLL > 6 for 72% of its June probes while stable.
- **Practical guidance** is unchanged from the investigation: prefer 7×7 (stem + second conv) or a lower lr for 3×3-heavy self-play towers. "LR above a threshold" is the best-supported *hypothesis*, not an established cause.

## Caveats

- Every lineage is n = 1. The blow-up onset varies 5k–74k steps with architecture, so step counts are not comparable across runs.
- May-era (KbHZ, sMe9, ysdg) and KXvb have no in-training OOD probe at all. "Stable" for them means sustained promotions + in-distribution metrics.
- Three pElo scales are in play, as noted above. In this README, pElo is only compared within a lineage.
- The registry says pre-[ARCH] lineages have "arch not recorded". Their architectures here come from arch_hash plus the code at the logged git commit, and precision from `ChessNetwork.dataType` at that commit. The trees were dirty (`*`), so these are commit-level, not binary-level, facts.
- WjRY's probe CSV ends at 98,950 while `[STATS]` reaches 107,548.

## Follow-ups

These were never run:

1. WjRY's exact config (8×3×3 v4 bf16) at lr 1e-3 / μ 0.90: a clean single-axis lr test.
2. The same config in fp32 at lr 1e-2: a precision test.
3. A 5×7×7 control with only the stem, or only the second conv, flipped to 3×3.
4. `v4_12block_3x3` on current code, to separate the code era.

The July replay-era stem-kernel series (nt8y 3×3/5×5/15×15) is corpus replay, not self-play, and does not answer this question.

## Audit notes

- **Verified against logs** (`~/Library/Logs/DrewsChessMachine/`), by streaming every registry log per lineage (filtering `[STATS]` to the lineage tag): lr, μ, wd, ent, drawPen, builds, step ranges, `promoted=1` counts, trainer-ID progression. Params and architecture come from `[ARCH]` lines where present, otherwise from arch_hash plus the memory arch-hash table.
- **Verified against the dashboards:**
  - Probe curves: `documentation/dashboards/selfplay_probe/{KbHZ,LWKa,WjRY,LMGh,wTp3}.csv`.
  - Hours and max steps: `data/*.csv` (KbHZ 256.6h, ysdg 32.6h, KXvb 11.9h, LWKa 28.4h, WjRY 22.4h, LMGh 10.0h, wTp3 7.0h, all matching the registry labels).
  - KbHZ's lr history is corroborated by CHANGELOG.md, "2026-05-21 20:30 CDT — Learning rate 5e-4 → 1e-3 on the KbHZ run" (step ≈ 435k).
  - Precision per commit: `git grep "static let dataType"` at each logged git hash.
- **Corrections:**
  - Old claim: "KbHZ ran ~495,000 steps stable at lr 1e-3 in fp32." New: KbHZ ran 1.5e-4 → 5e-4 → 1e-3, with 1e-3 only from ≈435k; its May run ended at 494,933. Evidence: `[STATS] lr=` in `dcm_log_20260514-020048` (1.5e-04), `20260515-170436` (5.0e-04), `20260521-172508` (1.0e-03); CHANGELOG 2026-05-21 20:30 entry.
  - Old claim: effective-step gap "~35×" (0.001/0.35 vs 0.10). New: 35× only for KbHZ's last ~60k May steps. Most of KbHZ ran at 0.0014 (5e-4/0.35), a ~70× gap, and at first 0.0004 (~230×). The clean same-shape 1e-3 run, sMe9 (μ 0.85–0.90), sits at a 10–15× gap. Evidence: the `[STATS] lr=`/`μ=` lines above.
  - Old claim: "every 3×3-stem run in the bf16/v4 era blew up (3/3)." New: 3/3 holds only for the post-06-02 config-builder/executable era (jaq1, LMGh, WjRY). KXvb (bf16, v4, lr 1e-2, 51,551 steps) and LWKa (bf16 from 7.4k, v4, lr 1e-2, 106,740 steps, flat probe NLL 3.48–3.69) are 3×3-stem v4 runs that did not blow up. Evidence: `[APP]` arch_hash 0xbad32ced, `[STATS] lr=1.0e-02`, `selfplay_probe/LWKa.csv`.
  - Old claim: "8 lineages 2026-05-14→06-10." New: the 8-run super-table (iteration 6) covers 05-31 → 06-10 and does not include KbHZ. The 05-14 start comes from iteration 7/8's 13-row final table. Evidence: `documentation/archive/OVERNIGHT_INVESTIGATION.md` iterations 6–8.
  - Old claim: KbHZ-cont. "→536k+". New: `[STATS]` max is 532,371 and the last save is step 532,369. Evidence: `dcm_log_20260610-123913.txt`; `Ko63-manual` safetensors `training_step`.
  - Old claim: sMe9 "≥372,748 / ≥115,628." New: max step seen was 377,302. Evidence: `dcm_log_20260529-113855.txt`.
  - Old claim: "wd 1e-3 for the sMe9/ysdg/KXvb era." New: ysdg 1e-3. sMe9 alternated 1e-3 / 1e-4 / 1e-3. KXvb was 1e-4. Evidence: `[STATS] reg=(… decay=…)`.
  - Old claim (registry): KbHZ "24 promo." New: 24 `promoted=1` lines are logged, but 2 (steps 181,012 and 181,016) are in branches abandoned by the rewind to 175,543. The surviving line has 22, consistent with final champion `KbHZ-22` / trainer `-23`. **Fixed 2026-09-29:** registry label and `promotions` now read 22 (re-checked: `Verdict: PROMOTED` lines name KbHZ-1 … KbHZ-22, with KbHZ-4 minted twice on the abandoned 180,042 branches and a third time at 198,650 on the kept line).
  - Old claim (registry label): sMe9 "93h." New: `data/sMe9.csv` elapsed is 108.9h. Neither is exact, because summed launch elapsed double-counts rewinds; the by-step axis is authoritative.
  - Old claim (iteration 6): LWKa is an "old fixed · graph.run" era run. New: it is still pre-06-02, but it is bf16 v4 at lr 1e-2 like WjRY. Only its first log (build 1528, git 0626cec) was fp32 at the commit level.
- **Unverifiable:**
  - The 80-checkpoint retro-probe numbers (e.g. KbHZ 562 @ 495k, sMe9 709 / 11.1%, LWKa 648, jaq1 gen-4 NLL 6.40, LMGh gen-2 NLL 13.1). The raw `/tmp/checkpoint_probe_sweep.jsonl` no longer exists, most of those sessions were pruned (21 session dirs remain), and re-probing is barred while training is live.
  - The KXvb precision regime (bf16 without fp32 masters): commit-level only, since the working tree was dirty.
  - ysdg / KXvb / LWKa / sMe9 params: from arch_hash only. No safetensors header survives; the older saves are .dcmmodel.
  - The "7×7 stable 3/3" and "7×7 tolerated lr 1e-2 to 471k (Exp 1)" claims: not re-audited here (see the linked write-ups). Memory notes bzw3 ran LR cycling from ~382.7k, so "lr 1e-2 to 471k" is likely also imprecise.
