# Architecture Experiments

A running log of DCM architecture experiments. Each entry is self-contained; reuse the
section structure verbatim and change only the content within each section.

**Convention:** pElo / NLL are **wide-set** (4435-puzzle) tactical-probe values unless a
per-experiment note says otherwise — the lower-variance signal, and the default for
cross-experiment comparison.

---

## Experiment 1 — 5-Block 7×7-Wide (ReZero / SE)

**arch_hash** `0xdf23a86c` · **lineage** `5K7Z` (saved) / `bzw3` (live) · **dates** 2026-06-01 → 2026-06-06

### 1. Architecture
- **Input:** 30 planes × 8×8 (NCHW), current-player perspective; **policy** 4864 logits (76×64, AlphaZero encoding); **value** 3-class W/D/L head.
- **Stem:** 7×7 conv, 30 → 128.
- **Tower:** **5** pre-activation (ResNet-v2) residual blocks, 128 ch. Each block = 7×7 same-padded conv (pad 3) → scale-and-bias **SE** (reduction /4) → clean identity add scaled by a per-block ReZero/SkipInit scalar α init `1/√5`. Activation ReLU.
- **Policy head:** 1×1 conv 128 → 76 → 4864 logits.
- **Value head:** 1×1 conv 128 → 16 → BN/ReLU → flatten(1024) → FC 1024→128 → ReLU → FC 128→3 (W/D/L), categorical-CE loss.
- **Precision:** bfloat16 compute. **Params:** 8,445,748 (~8.45M). **Arch version:** v4.
- *Context:* 7×7 kernels carry 5.4× the weights of 3×3, so despite only 5 blocks this is ~2.2× the 12-block 3×3 baseline's parameter count.

### 2. Relevant saved sessions
Resumable `.dcmsession` snapshots (weights + replay buffer + params), covering steps ~276k–467k of a 0–~470k run.

| Saved session (`.dcmsession`) | Step @ snapshot |
|---|--:|
| `20260605-002429-20260601-12-5K7Z-periodic` | 276,276 |
| `20260606-002543-20260601-12-5K7Z-periodic` | 382,625 |
| `20260606-230443-20260601-12-5K7Z-periodic` | 465,652 |
| `20260607-015807-20260601-12-5K7Z-manual`   | 467,077 |

Location: `~/Library/Application Support/DrewsChessMachine/Sessions/`

### 3. Factuals

| Step | pElo (wide) | NLL (wide) | pElo (200) | NLL (200) | Detail |
|--:|--:|--:|--:|--:|---|
| 0 | — | — | 701 | 4.27 | New 5-block 7×7 network, **constant LR 1e-2** (weight_decay 1e-4, grad_clip 30, entropy_bonus 0, draw_penalty 0). |
| 3,104 | — | — | 713 | 4.12 | First promotion (arena #3). |
| ~40,000 | — | — | 801 | 3.54 | **13 promotions** reached by here — fast early bootstrap. |
| ~190,000 | 806 | 3.25 | 924 | 3.33 | **Wide-set (4435-puzzle) probe instrumented from ~here** — wide-set coverage begins. |
| 339,874 | 871 | 3.18 | 977 | 3.21 | **Turning point** — #279, the 29th and final normal-cadence promotion; promotion cadence collapses here (capacity ceiling). |
| 382,728 | 871 | 3.21 | 961 | 3.23 | **Param change:** LR cycling introduced (peaks ~3e-1, troughs ~1e-4), toggled on/off thereafter. |
| 409,245 | 879 | 3.16 | 968 | 3.17 | Last promotion — #343 → champion `bzw3-31`; landed inside a **constant-1e-2** window. |
| ~470,000 | 876 | 3.17 | 961 | 3.21 | Run assessed: 398 arenas, **30 promotions total**, plateaued. LR cycling earned **zero promotions**; under hot peaks, candidate arena scores drifted **below 0.5** (worse than the standing champion). |

*The **wide set (4435 puzzles)** is the cross-experiment default, but was only instrumented from ~step 190k — early rows show **200-set** only. Both are listed here so this run stays comparable to priors (200-set) and future runs (wide-set). Wide tracks ~90–100 pElo below the 200-set with the same shape.*

### 4. Wins
- **No training instability — including step 470k.** No entropy collapse (held ~2.55 nats), no value-head draw-collapse (pD ~0.44), no illegal-move blowup (~0.003), no gradient explosion (gNorm ~0.57), no NaN/divergence — bf16 stable even at 3e-1 LR peaks.
- **Productive early/mid-run:** pElo climbed 701 → ~970, earning **29 of 30 promotions by ~340k**, steepest in the first 60k (13 promotions by 40k).

### 5. Shortcomings
- **Strength plateaued at ~step 270k** (pElo/NLL flat thereafter); the arena eked out marginal promotions until ~step 340k, then stopped entirely. The final ~130k steps were unproductive.
- **The post-saturation phase actively regressed.** Candidates scored **below 0.5** vs the frozen champion (to ~0.42 — *worse* nets), and `pwNorm`/`pLogitAbsMax` inflated monotonically (13→22 / 20→31), over-sharpening on forced lines. No collapse, but churn producing worse-than-champion weights.
- 8.45M params in **only 5 blocks** appears to cap chess strength — **param count did not buy ceiling.**

### 6. Analysis
- **The plateau is a capacity ceiling, not a blow-up.**
- **Promotion cadence is the cleanest strength curve.** A smooth decay ending in a cliff is the saturation signature; health metrics alone look fine well into the plateau.
- **LR cycling never earned a promotion here** — constant LR did all the work. Cycling on a saturated net is churn, not damage (gNorm stable at peaks); the sub-0.5 arena scores are cycling artifacts, not degradation.
- **Capacity ≠ parameter count.** The shallow-but-wide 5-block/7×7 net underperformed; for chess, **depth likely matters more** than per-layer receptive field at equal budget.
- *Concern:* at weight_decay 1e-4, weight/logit norms grow unbounded; a longer or higher-LR run would need stronger decay to avoid logit blow-up.

### 7. Suggested future variants / changes
- **Restore depth:** 8–12 residual blocks at 3×3 — direct comparison to baseline A (`0xbad32ced`, 12-block 3×3) to confirm depth raises the ceiling.
- **Balanced middle:** 8–10 blocks at 5×5 to trade some receptive field for depth at a similar param budget.
- **Regularization:** bump weight_decay to ~3e-4 on long runs to cap logit/weight-norm growth.
- **LR:** drop cycling; at most a single cosine anneal to a low LR for a final polish. Keep constant 1e-2 as the workhorse.
- Only widen channels **after** depth is restored, not instead of it.

---

## Experiment 2 — 5-Block 7×7-Wide, `full10ply200` Input (ReZero / SE)

**arch_hash** `0xdf23a86c` (**NOT authoritative** — collides with Exp 1; banner also misreports `inputPlanes=30`) · **lineage** `oItC` (saved) / `2Gd1` (live) · **build** 1760 · **dates** 2026-06-06 → **in progress (~55k steps)**

> Identification caveat: this branch makes architecture runtime-configurable, so `arch_hash`/`inputPlanes` in the `[APP]` banner are stale and **identical to Experiment 1's** despite a different encoding. Isolate this run by **log file** (`dcm_log_20260606-213834.txt`) + **live lineage `2Gd1`**, never by `arch_hash`. (A later build adds an `[ARCH]` line carrying the true `architectureSummary`; this run predates it.)

### 1. Architecture
- **Input:** **200 planes** × 8×8 (NCHW), `full10ply200` — 10 stacked `basic20` frames (current ply N + 9 prior plies N-1…N-9), each from the ply-N mover's perspective; absent pre-game frames zero. **No temporal-repetition "duplication" planes** (the 10 `basic30` planes 20–29 are gone; the 2 per-frame repetition-count planes 18/19 survive inside each `basic20` frame). **Policy** 4864 logits; **value** 3-class W/D/L head.
- **Stem:** 7×7 conv, **200 → 128**.
- **Tower:** **identical to Experiment 1** — 5 pre-activation residual blocks, 128 ch, each 7×7 same-padded conv → scale-and-bias **SE** (reduction /4) → clean identity add scaled by per-block ReZero α (`1/√5`). Activation ReLU.
- **Policy head:** 1×1 conv 128 → 76 → 4864. **Value head:** 1×1 conv 128 → 16 → BN/ReLU → flatten(1024) → FC 1024→128 → ReLU → FC 128→3 (W/D/L), categorical-CE.
- **Precision:** bfloat16. **Params:** 9,511,988 (~9.51M). **Arch version:** v4.
- *Context:* **only the input encoding changed vs Exp 1** (basic30/30 → full10ply200/200). The +1,066,240 params over Exp 1's 8.45M are entirely the wider stem ((200−30)×128×7×7). This is therefore a controlled **encoding** experiment on a fixed tower — the question is whether richer position history minus explicit duplication planes helps or hurts.

### 2. Relevant saved sessions
Twelve resumable `.dcmsession` snapshots (saved lineage `oItC`; live champion lineage `2Gd1`) — the 11 post-promotion autosaves plus one manual save. Each carries `inputPlanes: 200`, `buildNumber: 1760`; the 11 promote steps match the arena promotion ladder exactly.

| Saved session (`.dcmsession`) | Step @ snapshot | Trigger |
|---|--:|---|
| `20260607-030329-20260607-5-oItC-promote` | 1,440 | promote |
| `20260607-031849-20260607-5-oItC-promote` | 2,414 | promote |
| `20260607-043515-20260607-5-oItC-promote` | 7,526 | promote |
| `20260607-045034-20260607-5-oItC-promote` | 8,557 | promote |
| `20260607-062232-20260607-5-oItC-promote` | 14,252 | promote |
| `20260607-085540-20260607-5-oItC-promote` | 23,903 | promote |
| `20260607-114329-20260607-5-oItC-promote` | 32,095 | promote |
| `20260607-121406-20260607-5-oItC-promote` | 33,477 | promote |
| `20260607-154748-20260607-5-oItC-promote` | 45,415 | promote |
| `20260607-164849-20260607-5-oItC-promote` | 47,972 | promote |
| `20260607-191620-20260607-5-oItC-promote` | 53,749 | promote |
| `20260607-195042-20260607-5-oItC-manual`  | 54,495 | manual |

Location: `~/Library/Application Support/DrewsChessMachine/Sessions/`. Steps = each session's `trainingSteps`.

### 3. Factuals
Wide-set is the cross-experiment default and is available here from step ~0 (unlike Exp 1, which has wide only from ~190k). **Cross-comparison to Exp 1 below ~190k must use the 200-set** (Exp 1 lacks early wide); wide tracks ~90–110 below the 200-set here.

| Step | pElo (wide) | NLL (wide) | pElo (200) | NLL (200) | Detail |
|--:|--:|--:|--:|--:|---|
| 0 | 543 | 3.80 | 677 | 4.10 | New full10ply200 net (Exp-1 tower; only encoding changed). **Constant LR 1e-2** (weight_decay 1e-4, grad_clip 30, entropy_bonus 0, draw_penalty 0, μ 0.90). |
| 1,440 | 543 | 3.80 | 677 | 4.10 | First promotion (arena #1, score 0.5625). |
| ~10,000 | 607 | 3.83 | 701 | 4.09 | **Steepest-gain bucket** (+43 wide). Early bootstrap. |
| ~16,700 | 604 | 3.82 | 699 | 4.09 | **Value head near the draw prior:** pD 0.76, vAbs 0.085, self-play draws **88%**. (basic30 at 16.7k: pD 0.76 / vAbs 0.109 / 84%.) |
| ~32,000 | 631 | 3.82 | 738 | 4.06 | Value head still flat: pD **0.80**, vAbs 0.079, draws **90%**. (basic30 at 28.6k had already started differentiating: pD 0.71 / vAbs 0.128 / 79%.) |
| ~40,000 | 666 | 3.82 | 767 | 4.02 | Still stuck: pD 0.79, vAbs 0.083, draws 89%. **8 promotions by here vs Exp 1's 13;** 200-set pElo 767 vs Exp 1's 801 — **slower bootstrap.** |
| 53,749 | 672 | 3.81 | 735 | 4.01 | Last promotion so far (arena #63, score 0.5775). 66 arenas / **11 promotions** total. |
| ~54,400 | 672 | 3.81 | 735 | 4.01 | **Value head turns decisive** (~25k steps later than basic30): pD 0.79→**0.716**, vAbs 0.083→**0.145**, draws 90%→**82%**, gNorm 0.98→2.66 (head now learning). |
| ~55,128 | 674 | 3.80 | 749 | 3.99 | **Current (in progress).** Still climbing (+8 wide last bucket), still promoting, value head decisive (pD 0.706, vAbs 0.153). No plateau yet. |

### 4. Wins
- **No training instability through ~55k.** pEnt healthy (~2.65 nats), pIllM ~0.011, gNorm ~3.08, value head decisive at the final step, no NaN — bf16 stable.
- **Value head did eventually turn decisive (~54k)** — pD 0.79→0.71, vAbs ×~2, draws 90%→82% — so the encoding is *learnable*, not a dead end.

### 5. Shortcomings
Comparing primarily against Experiment 1
- **Slow bootstrap:** wide pElo 543 → 674, **11 promotions, still promoting at 53.7k** (no arena cliff). Steepest gain in the first 10k.
- **Prolonged flat value head / high draw rate early.** For ~the first 44k steps the value head sat at pD ~0.79 / vAbs ~0.08 with **88–90% self-play draws**, only turning decisive at ~54k. The same tower on `basic30` did so by ~28k. (pD never approached 1.0 — this is a slow-to-differentiate value head plus a high draw rate, *not* the pD→1 value-head collapse the rubric defines.)

  | Step | full10ply200 (this) pD / vAbs / draw% | basic30 (Exp-1 lineage) pD / vAbs / draw% |
  |--:|--|--|
  | 16.7k | 0.759 / 0.085 / 88% | 0.761 / 0.109 / 84% |
  | 28.6k | 0.798 / 0.079 / 90% | 0.709 / 0.128 / 79% |
  | 44.6k | 0.794 / 0.083 / 89% | 0.649 / 0.188 / 73% |
  | 54.4k | 0.716 / 0.145 / 82% | 0.632 / 0.200 / 75% |

- **Slower bootstrap vs basic30** at matched steps: 8 promotions by 40k vs 13; 200-set pElo 767 vs 801 at 40k.
- **Wide-set NLL essentially flat** (~3.80–3.83 across all 55k) despite the pElo climb — calibration on the puzzle set is not improving even as ranking does.
- **No capacity verdict possible** — run is in progress and still climbing; plateau/ceiling cannot be assessed yet (contrast Exp 1's completed 470k run).
- **Per-step cost up:** +1.07M stem params + 200-plane encode (encodeMs p50 ~450) raise step time ~20% over the Exp-1-era build.

### 6. Analysis
- **Controlled encoding test.** Identical tower, optimizer (constant 1e-2 / μ 0.90, single distinct LR — confirmed), and regularization as Exp 1; **only the input encoding differs.** Differences are attributable to the encoding (modulo seed — see caveat).

One of these is probably correct (or maybe even both):
- **Hypothesis #1: Adding 9 move history slowed learning**
- **Hypothesis #2: Removing the 10 repetition planes slowed learning**
- **AZ/Leela keep both history *and* explicit repetition planes** — this encoding diverges by dropping the latter.
- **Cadence-as-strength-curve** still holds: promotions are decaying-but-ongoing (no cliff), consistent with a run still on its productive slope.

### 7. Suggested future variants / changes
- **Add back 10 history repetition planes to the input tensor**

