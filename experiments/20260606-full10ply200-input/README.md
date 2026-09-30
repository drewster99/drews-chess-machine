# 2026-06-06 — 5-block 7×7 tower on `full10ply200` input (10-ply history, no repetition planes)

**Status:** done (stopped at step 58,933 in `[STATS]`, ≥58,990 in `[BATCH-STATS]`, on 2026-06-07 17:46 CDT to free the machine for the `full10Ply10Reps210` follow-up)

Migrated from `documentation/ARCH_EXPERIMENTS.md` Experiment 2 and audited 2026-09-29. The original wording is kept. Corrections are marked inline as `~~old~~ → new` or `[Audit: …]`, and each one is listed in Audit notes.

## Question

Take the Experiment 1 tower (5 blocks, 7×7, 128 ch, basic30, lineage `5K7Z`/`bzw3`) and change **only the input encoding**, from `basic30` (30 planes: pieces, castling, EP, clock and repetition planes, no history) to `full10ply200` (10 stacked `basic20` frames with no temporal-repetition planes). Does the richer position history, minus the explicit duplication planes, help or hurt?

## Setup

- **Architecture** (`[BUTTON] Build Network` line in the log; confirmed against the embedded `architecture` JSON in the surviving `oItC-manual` `champion.safetensors`):
  `v4 pre . in full10ply200(200) -> stem 128 (7x7) . 5x[7x7 conv, SE+/4, clean_add, ReZero] . act relu . policy intermediate_conv(4864) . value WDL(16->FC128) . bfloat16 . 9,511,988 params`
- **Build** 1760, git `79c610b*` (dirty), branch `safetensors-storage`. Log `~/Library/Logs/DrewsChessMachine/dcm_log_20260606-213834.txt`. Dates 2026-06-06 21:39 → 2026-06-07 17:46 CDT (~20.1 h wall).
- **Machine:** local (the session log and sessions are on this Mac). The hardware is not recorded in the log.
- **Training config** (`[STATS]` `reg=`/`lr=`): constant LR 1e-2 (500-step warmup, `·√b` is a no-op at batch 4096), weight_decay 1e-4, grad_clip 30, μ 0.90, entropy_bonus 0, draw_penalty 0. Batch 4096, 800 self-play workers, `spDelay=3000ms` throughout, promote ≥ 0.53, 400-game arenas every 900 s. Self-play τ 1.00/0.50/0.007, arena τ 0.60/0.20/0.020.
- **Identification caveat (original):** the branch makes the architecture runtime-configurable, so the `[APP]` banner's `arch_hash`/`inputPlanes` are stale and **identical to Experiment 1's** (`arch_hash=0xdf23a86c inputPlanes=30`) even though the encoding differs. Isolate this run by **log file** plus **live lineage `2Gd1`**, never by `arch_hash`. This build predates the `[ARCH]` line.

### 1. Architecture (original §1)
- **Input:** **200 planes** × 8×8 (NCHW), `full10ply200`: 10 stacked `basic20` frames (current ply N plus 9 prior plies N-1…N-9), each from the ply-N mover's perspective. Absent pre-game frames are zero. **No temporal-repetition "duplication" planes**: the 10 `basic30` planes 20–29 are gone, while the 2 per-frame repetition-count planes 18/19 survive inside each `basic20` frame. **Policy** 4864 logits. **Value** 3-class W/D/L head.
- **Stem:** 7×7 conv, **200 → 128**.
- **Tower:** **identical to Experiment 1**: 5 pre-activation residual blocks, 128 ch. Each block is a 7×7 same-padded conv → scale-and-bias **SE** (reduction /4) → clean identity add scaled by a per-block ReZero α (`1/√5`, stored 0.4472136). Activation ReLU.
- **Policy head:** 1×1 conv 128 → 76 → 4864. **Value head:** 1×1 conv 128 → 16 → BN/ReLU → flatten(1024) → FC 1024→128 → ReLU → FC 128→3 (W/D/L), categorical CE.
- **Precision:** bfloat16. **Params:** 9,511,988 (~9.51M). [Audit: the tensor-shape sum of the champion header is exactly 9,511,988.] **Arch version:** v4.
- *Context:* **only the input encoding changed vs Exp 1** (basic30/30 → full10ply200/200). The +1,066,240 params over Exp 1's 8,445,748 are entirely the wider stem ((200−30)×128×7×7). [Audit: 9,511,988 − 8,445,748 = 1,066,240 = 170·128·49, exact.] So this is a controlled **encoding** experiment on a fixed tower. The question is whether richer position history, minus explicit duplication planes, helps or hurts.

## Runs

- **Live lineage:** champion `20260607-4-2Gd1` → `2Gd1-11`. The final trainer is `2Gd1-12`. **Saved-session lineage:** `20260607-5-oItC`.
- **One log:** `dcm_log_20260606-213834.txt`, a single contiguous process with segments #1–#13. Steps run 0 → 58,933 with no resume and no rewind.
- **Arenas:** 75 in total (#1 at step 1,440 → #75 at step 58,785), with **11 promotions**.

### 2. Relevant saved sessions (original §2, audited)

Twelve resumable `.dcmsession` snapshots were written: the 11 post-promotion autosaves plus one manual save. Each carries `inputPlanes: 200` and `buildNumber: 1760`, and the 11 promote steps match the arena promotion ladder exactly. [Audit: every row below matches a `[CHECKPOINT] Saved session` line in the log. **Only the manual save survives on disk today**; the 11 `-promote` folders were pruned later.]

| Saved session (`.dcmsession`) | Step @ snapshot | Trigger | On disk 2026-09-29 |
|---|--:|---|---|
| `20260607-030329-20260607-5-oItC-promote` | 1,440 | promote | pruned |
| `20260607-031849-20260607-5-oItC-promote` | 2,414 | promote | pruned |
| `20260607-043515-20260607-5-oItC-promote` | 7,526 | promote | pruned |
| `20260607-045034-20260607-5-oItC-promote` | 8,557 | promote | pruned |
| `20260607-062232-20260607-5-oItC-promote` | 14,252 | promote | pruned |
| `20260607-085540-20260607-5-oItC-promote` | 23,903 | promote | pruned |
| `20260607-114329-20260607-5-oItC-promote` | 32,095 | promote | pruned |
| `20260607-121406-20260607-5-oItC-promote` | 33,477 | promote | pruned |
| `20260607-154748-20260607-5-oItC-promote` | 45,415 | promote | pruned |
| `20260607-164849-20260607-5-oItC-promote` | 47,972 | promote | pruned |
| `20260607-191620-20260607-5-oItC-promote` | 53,749 | promote | pruned |
| `20260607-195042-20260607-5-oItC-manual`  | 54,495 | manual | **present** (4.9 GB) |

Location: `~/Library/Application Support/DrewsChessMachine/Sessions/`. Steps are each session's `trainingSteps`.

The surviving manual session's `__metadata__` gives `champion.safetensors` `model_id=20260607-4-2Gd1-11`, `training_step=54495`, `input_encoding=full10ply200`, and 9,511,988 params. `trainer.safetensors` is `20260607-4-2Gd1-12` (parent `2Gd1-11`), step 54,495; its 19,020,616 stored values are the weights plus optimizer velocity. `session.json` `arenaHistory` holds 65 arenas and 11 promotions at step 54,495.

## Results

### 3. Factuals (original table, with audit columns)

Wide-set is the cross-experiment default, and here it is available from step ~0, unlike Exp 1, which has wide only from ~190k. **Cross-comparison to Exp 1 below ~190k must use the 200-set**, because Exp 1 lacks early wide. Wide tracks ~90–110 below the 200-set here. [Audit: over 2k-step buckets the 200-set − wide gap is ~50–125. It is ~90–125 before ~32k, then ~45–110.]

The original table's pElo/NLL figures were produced by an unrecorded averaging method, and several could not be reproduced (see Audit notes). The **Audit** columns are the mean over the 2,000-step bucket starting at that step, taken from the log's `[TACTICAL-LICHESS] tick` lines (wide = `set=wide`, 4,435 puzzles; 200 = the 200-puzzle set). These values are on the **recording build's (June 2026) in-app probe scale**. They are not comparable to replay-era (July+) pElo, or to the selfplay_registry `endpoint_pElo` 627.1, which is a later-binary re-probe.

| Step | pElo (wide) | NLL (wide) | pElo (200) | NLL (200) | Audit 2k-bucket wide pElo / NLL | Audit 2k-bucket 200 pElo / NLL | Detail |
|--:|--:|--:|--:|--:|--|--|---|
| 0 | 543 | 3.80 | 677 | 4.10 | 525.5 / 3.838 (0–2k) | 649.7 / 4.150 | New full10ply200 net (Exp-1 tower; only encoding changed). **Constant LR 1e-2** (weight_decay 1e-4, grad_clip 30, entropy_bonus 0, draw_penalty 0, μ 0.90). [Audit: the first probe (step 28) is wide 515 / 3.307, 200-set 524 / 3.500.] |
| 1,440 | 543 | 3.80 | 677 | 4.10 | — | — | First promotion (arena #1, score ~~0.5625~~ → **0.5487**; 0.5625 is the candidate-as-black score). [Audit: probe at step 1,450 is wide 545 / 3.714, 200-set 711 / 3.988.] |
| ~10,000 | 607 | 3.83 | 701 | 4.09 | 595.3 / 3.858 | 688.3 / 4.089 | **Steepest-gain bucket** (+43 wide). Early bootstrap. |
| ~16,700 | 604 | 3.82 | 699 | 4.09 | 612.0 / 3.826 (16–18k) | 704.9 / 4.092 | **Value head near the draw prior:** pD 0.76, vAbs 0.085, self-play draws **88%**. (basic30 at 16.7k: pD 0.76 / vAbs 0.109 / 84%.) [Audit ✓: step 16,708 pD 0.759, vAbs 0.0848, comp D 0.882.] |
| ~32,000 | 631 | 3.82 | 738 | 4.06 | 635.1 / 3.824 | 764.2 / 4.060 | Value head still flat: pD **0.80**, vAbs 0.079, draws **90%**. (basic30 at 28.6k had already started differentiating: pD 0.71 / vAbs 0.128 / 79%.) [Audit ✓: step 31,980 pD 0.798, vAbs 0.0782, D 0.888.] |
| ~40,000 | 666 | 3.82 | 767 | 4.02 | 662.1 / 3.802 | 765.2 / 3.999 | Still stuck: pD 0.79, vAbs 0.083, draws 89%. **8 promotions by here vs Exp 1's 13;** 200-set pElo 767 vs Exp 1's 801, so a **slower bootstrap.** [Audit ✓: 8 vs 13 promotions by 40k; bzw3 200-set 2k-bucket at 40k is 808.3.] |
| 53,749 | 672 | 3.81 | 735 | 4.01 | 675.0 / 3.796 (52–54k) | 754.4 / 3.990 | Last promotion (arena #63, score ~~0.5775~~ → **0.5700**; 0.5775 is candidate-as-black). ~~66 arenas~~ → **63 arenas** / **11 promotions** at this point (75 arenas by the end of the run). |
| ~54,400 | 672 | 3.81 | 735 | 4.01 | 667.9 / 3.810 (54–56k) | 713.6 / 4.014 | **Value head turns decisive** (~25k steps later than basic30): pD 0.79→**0.716**, vAbs 0.083→**0.145**, draws 90%→**82%**, gNorm 0.98→2.66 (head now learning). [Audit: step 54,407 pD 0.712, vAbs 0.150, D 0.818, gNorm 2.74.] |
| ~55,128 | 674 | 3.80 | 749 | 3.99 | — | — | Still climbing (+8 wide last bucket), still promoting, value head decisive (pD 0.706, vAbs 0.153). No plateau yet. [Audit: pD/vAbs ✓ at step 55,128. "Still promoting" is **not** borne out: no promotion after 53,749, and the last 12 arenas (#64–#75) scored 0.389–0.525.] |
| 58,933 | — | — | — | — | 663.1 / 3.793 (58k–end) | 723.7 / 3.997 | **Final step: run stopped to start Experiment 3.** [Audit: last `[STATS]` step 58,933, last `[BATCH-STATS]` step 58,990. Champion `2Gd1-11`, trainer `2Gd1-12`.] |

**Final status (run ended step 58,933):** the value head broke its stall ~48–54k exactly as anticipated (vAbs 0.083→0.134 by ~48k → ~0.15 by 54k; pD 0.80→0.72; draws 89%→82%). At 52.8k: 200-set pElo 755 / NLL 3.98, vAbs 0.135, pD 0.730. The run **never plateaued**. It was still on its slow productive slope when stopped, so Exp 2 yields **no capacity verdict**, only the slow-bootstrap and flat-NLL findings. Direct successor: **Experiment 3**, which adds the dropped repetition planes back. [Audit: at 48,001 vAbs 0.124, pD 0.725, D 0.832. At 52,802 vAbs 0.135 ✓, pD 0.730 ✓, pointwise 200-set 781 / 4.022, 2k-bucket 754.4 / 3.990. "Never plateaued" holds for the probe (wide 2k-buckets rise to ~676 at 50–52k, then ~663–668 through the end), but the arena ladder had stalled: no promotion in the last ~5.2k steps.]

### 4. Wins
- **Encoding is learnable.** The value head turned decisive at ~54k (pD 0.79→0.71, vAbs ×~2, draws 90%→82%), so this is not a dead end.
- *(Stable training is **not** counted as a Win. Exp 1 was also stable, so this is no regression, not an improvement. The slower path to decisiveness is under Shortcomings.)*

### 5. Shortcomings
Compared primarily against Experiment 1:
- **Slow bootstrap:** wide pElo 543 → 674, **11 promotions, still promoting at 53.7k** (no arena cliff). Steepest gain in the first 10k.
- **Prolonged flat value head / high draw rate early.** For roughly the first 44k steps the value head sat at pD ~0.79 / vAbs ~0.08 with **88–90% self-play draws**, and only turned decisive at ~54k. The same tower on `basic30` did so by ~28k. (pD never approached 1.0, so this is a slow-to-differentiate value head plus a high draw rate, *not* the pD→1 value-head collapse the rubric defines.)

  | Step | full10ply200 (this) pD / vAbs / draw% | basic30 (Exp-1 lineage) pD / vAbs / draw% |
  |--:|--|--|
  | 16.7k | 0.759 / 0.085 / 88% | 0.761 / 0.109 / 84% |
  | 28.6k | 0.798 / 0.079 / 90% | 0.709 / 0.128 / 79% |
  | 44.6k | 0.794 / 0.083 / 89% | 0.649 / 0.188 / 73% |
  | 54.4k | 0.716 / 0.145 / 82% | 0.632 / 0.200 / ~~75%~~ → 73% |

  [Audit: this run's column matches `[STATS]` at steps 16,708 / 28,611 / 44,590 / 54,407. The 54.4k row reads 0.712 / 0.150 / 81.8%. The basic30 column is bzw3 in `dcm_log_20260601-205349.txt`: 16,706 0.761/0.109/83.5%, 28,616 0.709/0.128/79.4%, 44,613 0.649/0.188/73.3%, 54,389 0.634/0.201/72.5%. "draw%" is the `comp=(… D=…)` self-play draw fraction. pD/vAbs are **trainer-side** batch statistics (the trainee's value head on replay-buffer positions), not champion self-play.]
- **Slower bootstrap vs basic30** at matched steps: 8 promotions by 40k vs 13, and 200-set pElo 767 vs 801 at 40k.
- **Wide-set NLL essentially flat** (~3.80–3.83 across all 55k) despite the pElo climb. Calibration on the puzzle set is not improving even as ranking does. [Audit: 2k-bucket wide NLL spans 3.756–3.858.]
- **No capacity verdict possible.** The run ~~is in progress and~~ was stopped while still climbing on the probe, so plateau/ceiling cannot be assessed (contrast Exp 1's completed 470k run).
- **Per-step cost up:** +1.07M stem params plus the 200-plane encode (encodeMs p50 ~450) raise step time ~20% over the Exp-1-era build. [Audit: median `timing=(step=…)` is 870.3 ms here vs 723.6 ms for bzw3 in `dcm_log_20260601-205349.txt` (+20%). The median `[ENCODE-COST]` encodeMs p50 is 414 ms. The bzw3-era build did not log `[ENCODE-COST]`.]

## Conclusion

### 6. Analysis (original)
- **Controlled encoding test.** Identical tower, optimizer (constant 1e-2 / μ 0.90, single distinct LR, confirmed) and regularization as Exp 1; **only the input encoding differs.** Differences are attributable to the encoding, modulo seed (see Caveats).

One of these is probably correct, or maybe both:
- **Hypothesis #1: Adding 9 moves of history slowed learning**
- **Hypothesis #2: Removing the 10 repetition planes slowed learning**
- **AZ/Leela keep both history *and* explicit repetition planes.** This encoding diverges by dropping the latter.
- **Cadence as strength curve** still holds: promotions are decaying but ongoing (no cliff), consistent with a run still on its productive slope. [Audit: the last promotion was at arena #63 (53,749). Arenas #64–#75 did not promote.]

**Post-audit reading:** the input change (history in, explicit repetition planes out) slowed both the arena bootstrap (8 vs 13 promotions by 40k) and value-head differentiation (~54k vs ~28k) against the same tower on basic30, while tactical probe NLL stayed flat. Experiment 3 then showed the net reads almost none of the added history (see [`../20260607-full10ply10reps210-input/`](../20260607-full10ply10reps210-input/README.md)). A stem-norm read of this run's surviving checkpoint (Audit notes) shows the same pattern here: frame 0 at ~2.5× He-init scale, frame 1 ~1.6×, frames 2–9 at ~0.93–1.10×.

## Caveats
- Single seed. Exp 1 (bzw3) vs this run is n=1 vs n=1.
- pD/vAbs are trainer-batch statistics, and "draw%" is the champion's self-play `comp` draw fraction, so the two come from different networks.
- All pElo here is the recording build's in-app probe scale (June 2026). The selfplay_registry endpoint (627.1 / NLL 4.1513) is a later re-probe on a different scale and must not be compared with the table.
- `spDelay=3000ms` was on for the whole run (as in Exp 1's era).
- The original table's step-0 row cannot be reproduced from the log, and neither can the exact 543/677 values at 1,440 (see Audit notes).

## Follow-ups

### 7. Suggested future variants / changes (original)
- **Add back the 10 history repetition planes to the input tensor**: *done in Experiment 3* ([`../20260607-full10ply10reps210-input/`](../20260607-full10ply10reps210-input/README.md)).

## Audit notes

Verified against the session log `dcm_log_20260606-213834.txt` (`[BUTTON]`, `[SEGMENT]`, `[ARENA] #N kv`, `[CHECKPOINT]`, `[STATS]`, `[BATCH-STATS]`, `[TACTICAL-LICHESS] tick`, `[ENCODE-COST]`), the surviving session `Sessions/20260607-195042-20260607-5-oItC-manual.dcmsession` (safetensors `__metadata__` + tensor-shape param sums + `session.json`), `documentation/dashboards/data/2Gd1.csv`, `selfplay_probe/2Gd1.csv` (which is the wide set: step 28 = 515.468 / 3.30663, identical to the log's `set=wide` tick) and `selfplay_registry.json`. The basic30 comparisons were checked against the bzw3 logs `dcm_log_20260601-162715.txt` and `dcm_log_20260601-205349.txt`.

- **Verified:** architecture string and 9,511,988 params (header tensor sum); +1,066,240 stem arithmetic; build 1760 / git `79c610b`; 11 promotions at steps 1,440 / 2,414 / 7,526 / 8,557 / 14,252 / 23,903 / 32,095 / 33,477 / 45,415 / 47,972 / 53,749; all 12 session names and steps; 8 promotions by 40k here vs 13 for bzw3 (3,104 … 39,987); value-head / draw figures at 16.7k / 28.6k / 32k / 40k / 44.6k; basic30 figures at 16.7k / 28.6k / 44.6k; basic30 200-set ~801–810 at 40–52.8k (2k-buckets 808.3 / 811.7) and NLL ~3.50; +20% step time; final step 58,933; wide NLL flat.
- **Correction:** arena #1 score 0.5625 → **0.5487** (evidence: `[ARENA] #1 kv … score=0.5487 … cand_black_score=0.5625`). The doc quoted the per-colour score.
- **Correction:** arena #63 score 0.5775 → **0.5700** (evidence: `#63 kv step=53749 … score=0.5700`; 0.5775 is `cand_black_score`).
- **Correction:** "66 arenas / 11 promotions total" at 53,749 → **63 arenas** at that point and **75 arenas / 11 promotions** for the whole run (evidence: `#63 kv step=53749`, last `#75 kv step=58785`; `session.json` at 54,495 holds 65).
- **Correction:** "~55,128 … still promoting" → no promotion after 53,749; arenas #64–#75 scored 0.3887–0.5250 (evidence: `[ARENA] kv` lines).
- **Correction:** basic30 draw% at 54.4k 75% → **73%** (bzw3 `[STATS]` step 54,389 `comp D=0.725`).
- **Minor:** 54.4k row pD/vAbs/draws 0.716 / 0.145 / 82% vs log 0.712 / 0.150 / 81.8% at 54,407. Close; left as written with the audited value beside it.
- **Correction:** "run is in progress" in Shortcomings → the run was stopped (the migrated text predates the stop).
- **Unreproducible:** the table's step-0 and step-1,440 pElo/NLL (543 / 3.80 / 677 / 4.10). The first probe is 515 / 3.307 (wide), and the 0–1,440 mean is 512.6 / 3.863 (wide) and 624.3 / 4.185 (200). The averaging behind the other rows is unrecorded, so 2k-bucket means are shown beside them.
- **Stem-norm check (new, read-only):** from the surviving `2Gd1-11` champion (step 54,495), the per-input-plane L2 norm of `stem.conv.weight` [128,200,7,7], divided by the He-normal expectation √(128·49·2/(200·49)) = 1.131, averaged per 20-plane frame, gives F0 2.54, F1 1.60, F2 1.10, F3 1.02, F4 0.98, F5 0.96, F6 0.95, F7 0.95, F8 0.94, F9 0.93. This assumes He-normal init, which is not re-verified in code.
- **Unverified:** the machine or hardware (not logged). The 1,440-step "steepest-gain bucket (+43 wide)" arithmetic depends on the unrecorded bucketing.
- **Scale note:** all pElo in this file is the June 2026 in-app scale. The registry's `endpoint_pElo` 627.1 is a later re-probe. `selfplay_probe/2Gd1.csv` peaks at 712.7 (step 48,653).
