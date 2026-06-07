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
