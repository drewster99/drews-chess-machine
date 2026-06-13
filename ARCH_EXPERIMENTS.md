# Architecture Experiments

A running log of DCM architecture experiments. Each entry is self-contained; reuse the
section structure verbatim and change only the content within each section.

**Convention:** pElo / NLL are **wide-set** (4435-puzzle) tactical-probe values unless a
per-experiment note says otherwise — the lower-variance signal, and the default for
cross-experiment comparison.

**Architecture identity (per `RUNTIME_ARCHITECTURE_CONFIG_PLAN.md` §6):** identity is the
**embedded architecture config** (the `[ARCH]` / `architectureSummary` line), **not**
`arch_hash`. On this branch `arch_hash` is non-authoritative — it collides across different
encodings (same plane *count*) and survives only as a legacy-`.dcmmodel` lookup key.
Identify a run by its `architectureSummary` + lineage + log file; a safetensors file's
integrity is `content_sha256`. Each experiment's **arch** line below is the canonical
summary in the plan's decomposed-axis vocabulary.

**Summary-format change (2026-06-12, block groups — one-time mapping):** with
`ARCHITECTURE_EXPANSION_PLAN.md` Feature 2, `architectureSummary` switched to a
fully-explicit per-group form (no silent defaults). The **arch** lines of Experiments 1–6
below are in the OLD format and remain valid identities; map them to the new form as:
`v4 pre` / `v3 post` → `v4` / `v3` (the activation style moved into each group);
`Nx[<k>x<k> conv, <SE>, <merge>, ReZero]` →
`Nx[<k1>x<k1>+<k2>x<k2> @<channels>, <SE>, <fn>/<style>, <merge>, ReZero(<α-init>), drop*<mult>]`
(both kernels always shown; width, activation function/style, α init, and dropout
multiplier explicit; legacy multiplier ≡ 1); the dual-kernel comma form `7x7,3x3 conv`
becomes `7x7+3x3`. Multi-group towers join group terms with `->`. Concretely, Exp 1's line
maps to:
`v4 . in basic30(30) -> stem 128 (7x7) . 5x[7x7+7x7 @128, SE+/4, relu/pre, clean_add, ReZero(0.447), drop*1] . act relu . policy intermediate_conv(4864) . value WDL(16->FC128) . bfloat16 . 8,445,748 params`.
New experiments record the explicit grouped summary verbatim.

---

## Experiment 1 — 5-Block 7×7-Wide (ReZero / SE)

**arch** `v4 pre . in basic30(30) -> stem 128 (7x7) . 5x[7x7 conv, SE+/4, clean_add, ReZero] . act relu . policy intermediate_conv(4864) . value WDL(16->FC128) . bfloat16` · **lineage** `5K7Z` (saved) / `bzw3` (live) · **dates** 2026-06-01 → 2026-06-06 · *(legacy `.dcmmodel` tag `0xdf23a86c` — non-authoritative; identity is the embedded config per PLAN §6)*

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

Resume-point characterization (log forensics 2026-06-11) — the saves differ
materially in what the trainer state carries:

- **276,276** — pre-cliff (marginal-promotion era; the cadence cliff is at ~340k).
- **382,625** — **the only post-cliff, pre-LR-cycling checkpoint**: LR cycling
  began at step 382,728, ~100 steps and four minutes after this save. Chosen
  starting point for the ceiling-vs-stall resume probe (§8).
- **465,652** — mid-cycling snapshot; `trainer.safetensors` includes the SGD
  velocity tensors (`opt.*.velocity`), captured during an lr≈1.8e-1 hot phase.
- **467,077** — taken right after the run's final constant-1e-1 segment (§3),
  so the trainer weights and velocity carry that 10×-LR kick.

### 3. Factuals

| Step | pElo (wide) | NLL (wide) | pElo (200) | NLL (200) | Detail |
|--:|--:|--:|--:|--:|---|
| 0 | — | — | 701 | 4.27 | New 5-block 7×7 network, **constant LR 1e-2** (weight_decay 1e-4, grad_clip 30, entropy_bonus 0, draw_penalty 0). |
| 3,104 | — | — | 713 | 4.12 | First promotion (arena #3). |
| ~40,000 | — | — | 801 | 3.54 | **13 promotions** reached by here — fast early bootstrap. |
| ~190,000 | 806 | 3.25 | 924 | 3.33 | **Wide-set (4435-puzzle) probe instrumented from ~here** — wide-set coverage begins. |
| 339,874 | 871 | 3.18 | 977 | 3.21 | **Turning point** — #279, the 29th and final normal-cadence promotion; promotion cadence collapses here (capacity ceiling). |
| 382,728 | 871 | 3.21 | 961 | 3.23 | **Param change:** LR cycling introduced (peaks ~3e-1, troughs ~1e-4), toggled on/off thereafter; **momentum cycles with it** (μ 0.90 ↔ ~0.855). A one-minute constant-**1e-1** poke at step ~382,722 (06-05 19:27) preceded it and was rolled back by a session resume. |
| 409,245 | 879 | 3.16 | 968 | 3.17 | Last promotion — #343 → champion `bzw3-31`; landed inside a **constant-1e-2** window. |
| 465,739 | — | — | — | — | **Param change (final):** cycling off → **constant LR 1e-1** (10× base) for the last ~1,360 steps, until the manual save/shutdown at 467,099 (06-06 20:23–20:58). The end-of-run trainer state carries this hot segment. |
| ~470,000 | 876 | 3.17 | 961 | 3.21 | Run assessed: 398 arenas, **30 promotions total**, plateaued. LR cycling earned **zero promotions**; under hot peaks, candidate arena scores drifted **below 0.5** (worse than the standing champion). |

*Run-config note (2026-06-11 forensics): `selfPlayDelay` was **3000 ms** over the entire verified span (step ~251k → end; every `[STATS]` line), alongside workers=800, batch=4096, decay=1e-4, promote≥0.53, unchanged taus. The run still averaged ~3.7k steps/hr — keep this in mind when comparing step rates across runs with different spDelay settings.*

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
- **Capacity ceiling confirmed by weight forensics (2026-06-08, on the 465k `champion.safetensors`).** Reading the saved weights directly: **0% dead channels** in every conv (weakest channel 0.6–0.9× its layer mean), **0%** near-zero BN γ, and **~91–95% effective rank** (participation ratio of singular values) in all ten tower convs. The net populated every channel and nearly every representational dimension and *still* couldn't pass pElo ~965 for 210k steps — i.e. the plateau is **not** unused capacity (which would show dead units / low rank), it's a packed net with nowhere to write new knowledge. The 7×7 tower kernels also use their full spatial extent (outer ring holds **~41%** of each kernel's energy; a 3×3 truncation would discard ~74%) — no spatial slack either. The over-sharpening the logs showed (pwNorm 13.8→22.3, pLogitAbsMax 15.6→30.7 while gNorm fell 2.0→0.56 and pElo stayed flat) is the saturated-net signature: SGD spent its remaining budget inflating confidence on known lines, not learning. *Caveat:* the forensics confirm capacity is fully **utilized** but can't fully separate a hard ceiling from a fixable weak-regularization over-sharpening stall (wd 1e-4 lets logits run; the 400k→467k blow-up is partly the LR-cycling experiment). Clean disambiguator: a **resume-from-checkpoint run with wd≈3e-4 / a one-shot LR anneal** — if pElo breaks ~977 it was regularization, if it stays pinned it was capacity.
- **Stem is over-wide (incidental forensic finding).** The 7×7 stem collapsed to ~1×1 (center holds 30% of energy, outer ring only 21% — vs the tower's 41% edge share): board planes are per-square one-hot, so the stem's real job is a pointwise per-square embedding and it zeroed the kernel periphery. A 1×1/3×3 stem costs ~nothing and frees ~150k params (far more on wider-input encodings). Spatial reasoning is the tower's job.
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
- **Refined by the weight forensics:** the tower is reach-hungry (uses the full 7×7) **and** channel-packed (0% dead, ~93% rank), so the move is **more depth + more channels at 3×3 kernels** (depth delivers the reach the tower wants, more parameter-efficiently than wide kernels — every competitive chess net is deep/wide/3×3) plus a **1×1 stem**. Do **not** narrow the tower kernels in place — that amputates needed reach. Wider *convs* are the lowest-value axis on an 8×8 board (receptive field is global after ~2 layers).

### 8. Resume probe — ceiling vs stall (protocol set 2026-06-11; results pending)

Executes the §6 disambiguator: is the plateau a hard capacity ceiling, or a
weak-regularization over-sharpening stall (wd 1e-4 letting logit/weight norms
inflate, saturating the softmax and shrinking effective gradients)?

- **Resume point: `20260606-002543-…-5K7Z-periodic` (step 382,625)** — not the
  465k/467k saves. Rationale: the 465k trainer is post-83k-steps of cycling
  churn with the 200-set pElo already down 977→961 and hot velocity tensors in
  the save; a *null* result from there can't distinguish "ceiling" from
  "cycling-damaged starting point". 382,625 is post-cliff but pre-cycling, with
  the inflation pathology already present (pwNorm ~17 of the 13.8→22.3 climb) —
  the stall hypothesis is fully testable and a null is clean.
- **Phase A:** cycling off, LR constant 1e-2, weight_decay 1e-4 → **3e-4**,
  ~5–10k steps. Health signature that decay is biting: pwNorm/pLogitAbsMax
  deflate, gNorm recovers. Then arena.
- **Phase B (if Phase A stays pinned):** keep wd 3e-4, one-shot anneal to LR
  constant **1e-3**, ~5k steps, arena.
- **Primary readout: the tactical-battery pElo ceiling, not arena promotions.**
  The lineage never broke **~977 (200-set) / ~879 (wide)** from anywhere in
  470k steps; breaking it post-intervention is clean signal. Promotions are
  secondary evidence here: this save's champion is the older `bzw3-30`, and the
  original run still squeezed straggler promotion #343 out of this region, so
  a lone promotion is ambiguous (base rate ≈ 1 per ~85k steps from this point).
- **Verdict rule:** breaks the ceiling → it was the stall (and the phase that
  broke it names the lever); pinned through both phases → capacity ceiling
  confirmed, depth is the answer.

---

## Experiment 2 — 5-Block 7×7-Wide, `full10ply200` Input (ReZero / SE)

**arch** `v4 pre . in full10ply200(200) -> stem 128 (7x7) . 5x[7x7 conv, SE+/4, clean_add, ReZero] . act relu . policy intermediate_conv(4864) . value WDL(16->FC128) . bfloat16` · **lineage** `oItC` (saved) / `2Gd1` (live) · **build** 1760 · **log** `dcm_log_20260606-213834.txt` · **dates** 2026-06-06 → 2026-06-07 (**completed at step 58,933**; stopped to start Experiment 3) · *(legacy `arch_hash` `0xdf23a86c` non-authoritative — collides with Exp 1, banner also misreports `inputPlanes=30`; identity is the embedded config per PLAN §6)*

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
| ~55,128 | 674 | 3.80 | 749 | 3.99 | Still climbing (+8 wide last bucket), still promoting, value head decisive (pD 0.706, vAbs 0.153). No plateau yet. |
| 58,933 | — | — | — | — | **Final step — run stopped to start Experiment 3.** |

**Final status (run ended step 58,933):** the value head broke its stall ~48–54k exactly as anticipated (vAbs 0.083→0.134 by ~48k → ~0.15 by 54k; pD 0.80→0.72; draws 89%→82%). At 52.8k: 200-set pElo 755 / NLL 3.98, vAbs 0.135, pD 0.730. The run **never plateaued** — still on its slow productive slope when stopped — so Exp 2 yields **no capacity verdict**, only the slow-bootstrap + flat-NLL findings. Direct successor: **Experiment 3** (adds the dropped repetition planes back).

### 4. Wins
- **Encoding is learnable.** The value head turned decisive at ~54k (pD 0.79→0.71, vAbs ×~2, draws 90%→82%) — not a dead end.
- *(Stable training is **not** counted as a Win: Exp 1 was also stable, so this is no regression, not an improvement. The slower path to decisiveness is in Shortcomings.)*

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
- **Add back 10 history repetition planes to the input tensor** — *done: Experiment 3.*

---

## Experiment 3 — 5-Block 7×7-Wide, `full10Ply10Reps210` Input (ReZero / SE)

**arch** `v4 pre . in full10Ply10Reps210(210) -> stem 128 (7x7) . 5x[7x7 conv, SE+/4, clean_add, ReZero] . act relu . policy intermediate_conv(4864) . value WDL(16->FC128) . bfloat16` · **lineage** `eaRt`→`KnCx` (live champion; promotions fork the ID) · **build** 1770 · **log** `dcm_log_20260607-174928.txt` · **dates** 2026-06-07 → in progress (~53k steps @ 2026-06-08 10:36) · *(safetensors-native, embedded-config identity per PLAN §6 — no `arch_hash`; isolate by the `[ARCH]` summary line + log file)*

> Direct successor to Experiment 2 — its single suggested variant (add the 10 repetition planes back). `full10Ply10Reps210` = `full10ply200` (10 stacked `basic20` frames) **+** the 10 `basic30` temporal-repetition planes (20–29) restored as a tail. So across the three experiments only the **input encoding** changes on an identical 5-block-7×7 tower: Exp 1 `basic30` (reps, no history), Exp 2 `full10ply200` (history, no reps), Exp 3 `full10Ply10Reps210` (history **and** reps). A controlled encoding ladder.

### 1. Architecture
- **Input:** **210 planes** × 8×8 (NCHW), `full10Ply10Reps210` — 10 stacked `basic20` frames (current ply N + 9 prior plies, ply-N mover's perspective, absent frames zero) followed by the 10 `basic30` temporal-repetition planes (plane `200+i` = position `i+1` plies ago is a strict duplicate). **Policy** 4864 logits; **value** 3-class W/D/L head.
- **Stem:** 7×7 conv, **210 → 128**.
- **Tower / policy / value:** **identical to Experiments 1 & 2** — 5 pre-activation residual blocks, 128 ch, 7×7 convs, scale-and-bias SE (/4), clean-add + ReZero (α `1/√5`), ReLU; `intermediate_conv` policy head; WDL value head (C_v=16, H=128).
- **Precision:** bfloat16. **Params:** 9,574,708 (~9.57M).
- *Context:* **only the input encoding changed vs Exp 2** (full10ply200/200 → full10Ply10Reps210/210). +62,720 params, all stem ((210−200)×128×49). Tests whether restoring the dropped duplication planes recovers Exp 2's slow value-head bootstrap.

### 2. Relevant saved sessions
Numerous post-promotion + periodic `.dcmsession` autosaves (safetensors-native; live champion lineage `eaRt`→`KnCx`). Verified snapshot:

| Saved session (`.dcmsession`) | Step @ snapshot | Trigger |
|---|--:|---|
| `20260608-140104-20260607-7-KnCx-promote` | 48,825 | promote |

Each `champion.safetensors` embeds the full architecture in `__metadata__` (`input_encoding: full10Ply10Reps210`, `training_step`, `content_sha256`). Location: `~/Library/Application Support/DrewsChessMachine/Sessions/`.

### 3. Factuals
200-set is the cross-comparison column (basic30/Exp 1 lacks early wide-set); wide-set listed where available.

| Step | pElo (200) | NLL (200) | pElo (wide) | NLL (wide) | vAbs | pD | draw% | Detail |
|--:|--:|--:|--:|--:|--:|--:|--:|---|
| 5k | 698 | 4.022 | 616 | 3.827 | 0.104 | 0.719 | 79% | Value head already decisive (started low pD 0.667 / vAbs ~0.10 at 1k — unlike Exp 2's draw-prior start). |
| 10k | 744 | 3.905 | 647 | 3.726 | 0.095 | 0.757 | 85% | Leads basic30 (719) and Exp 2 (694) on 200-set pElo. ~4 promotions by 10.5k (tied). |
| 23k | 733 | 3.888 | 654 | 3.716 | 0.118 | 0.716 | 83% | Value head tracking basic30 (vAbs 0.118 vs 0.133); still ≈ level on pElo. |
| 30k | 704 | 3.929 | 652 | 3.759 | 0.130 | 0.714 | 83% | basic30 begins its surge here; 210 flat. |
| 40k | 761 | 3.909 | 670 | 3.769 | 0.162 | 0.660 | 77% | basic30 overtakes (803 vs 761). 210 value head decisive & climbing. |
| 48k | 738 | 3.928 | 663 | 3.758 | 0.163 | 0.658 | 78% | Gap to basic30 ~70 pElo / 0.43 NLL. Tied with Exp 2 on pElo (766). |
| 52.8k | 754 | 3.927 | — | — | 0.167 | 0.647 | 79% | **Plateaued** ~750; basic30 ~810. Gap stable ~55 pElo. |

*Trend slopes (23k–48k): basic30 pElo **+57/10k (R²=0.53)**, NLL **−0.12/10k (R²=0.71)** — genuinely climbing. 210 pElo **+15/10k (R²=0.09 — flat/noise)**, NLL ~0 — not improving. The gap widened 23k→40k then stabilized.*

### 4. Wins
- **Value head ignites early — like basic30, unlike Exp 2.** Decisive from ~step 1k (vAbs ~0.10, pD climbing); tracks basic30 (vAbs 0.118/0.162/0.167 at 23k/40k/52.8k vs basic30's 0.133/0.20/0.20) and runs **~30k steps ahead** of Exp 2's stalled head (frozen ~0.083 until ~48k). Restoring the repetition planes (the only change vs Exp 2) coincides with recovering early value-head learning. **Behaviorally supports Exp 2's Hypothesis #2** (removing the rep planes slowed the value head), not #1 (history per se) — but see the weight-forensics caveat in §6.
- **Better calibration + more decisive head than Exp 2** at matched steps (NLL 3.93 vs 3.98–4.02; vAbs 0.167 vs 0.135 at 52.8k).

### 5. Shortcomings
Comparing primarily against Experiment 1 (basic30).
- **Tactically weaker than basic30, and the gap is a stable plateau, not closing.** From a ~level start (210 *led* at 23k: 733 vs 714), basic30 surged after 30k while 210 stayed flat; by 52.8k basic30 leads **~55 pElo (810 vs 754) and ~0.43 NLL (3.50 vs 3.93)**. 210's pElo slope is statistically flat (R²=0.09) vs basic30's real climb — the lines diverged then locked, they are not converging.
- **No tactical benefit over Exp 2.** 210 and full10ply200 are **tied on pElo** (~755 at 52.8k) — restoring the rep planes helped the *value head* but did **nothing** for tactical/policy strength.
- **Wide-set NLL flat** (~3.72–3.77 across 10k–48k) — the same calibration-not-improving signature as Exp 2; basic30's NLL drops over the same window.

### 6. Analysis
- **Controlled encoding ladder, value-head result:** on the identical tower/optimizer, Exp 3 (history+reps) ignites the value head as early as Exp 1 (reps, no history) and far earlier than Exp 2 (history, no reps). Read naively, the **repetition planes** drive early value-head learning (H2).
- **Weight forensics complicate the mechanism (2026-06-08, on the 48.8k `champion.safetensors`).** Per-input-plane stem L2 norm shows the net reads **only frame 0 (2.5× init) and frame 1 (1.6× init)**; **frames 2–9 sit at initialization (0.96–0.98× init) and slightly *decay* over 47k steps** (weight-decay pruning unreinforced inputs), and **the rep-plane tail is also at init (0.95×)**. Across all saved checkpoints (1.4k→48.8k), F0/F1 grow monotonically while F2–9/reps never move. So **the 210 net uses only "current + 1 prior ply"; the other 8 history frames and the rep planes are structurally unused.** And basic30 *also* leaves its rep planes at ~0.66× init — so the rep planes are not heavily weighted in **either** net. Exp 3's early-value-head advantage over Exp 2 is therefore **not** cleanly attributable to the rep planes being *used* (they're at init); it may be seed, or a small purposeful sparse projection the norm can't see (rep planes are sparse-binary). **Open: confirm via an occlusion test** (zero the rep planes on repetition-rich positions, measure the value/policy delta).
- **Why no tactical gain:** the deep history is unused, so Exp 3 is effectively a `basic30`-class net carrying ~150 dead input planes + ~1 extra ply of context. On the capacity-saturated 5-block tower (Exp 1's forensics: 0% dead, ~93% rank, full kernel utilization) there is no spare capacity to exploit the richer input. Same capacity story as Exp 1, viewed from the input side.
- **The encoding question is not answerable on this tower** — whether 10-ply history *could* help is confounded by the tower being the bottleneck (the net declines to use the history at all). The clean test is the same deeper-tower experiment Exp 1 calls for.

### 7. Suggested future variants / changes
- **Settle the encoding question on a tower that isn't the bottleneck:** rerun `full10ply200` / `full10Ply10Reps210` vs `basic30` on an **8–12 block 3×3** tower (1×1 stem). Readout: does the stem's frame-2–9 norm move off init? If yes, capacity was the bind and history helps; if it stays pinned (as here), deep history is dead weight for this engine — close the question.
- **Occlusion test for the rep planes** (sparse-binary, so weight-norm understates them): on positions at/near 3-fold, zero the rep tail and measure the value-head shift — settles "suppressed vs functionally unused." Ideal probe: a position with ≥2 legal moves where one forces 3-fold/50-move and the other is otherwise equal-looking but keeps a win.
- **Drop the dead input cost:** a 1×1 stem on the 210 encoding frees ~1.29M params (~13.5% of the model) the current 7×7 stem spends on a spatially-collapsed, mostly-ignored input.

---

## Experiment 4 — 5-Block 256-Wide Dual-Kernel (7×7+3×3), `full10Ply10Reps210` Input (ReZero / SE)

**arch** `v4 pre . in full10Ply10Reps210(210) -> stem 256 (3x3) . 5x[7x7,3x3 conv, SE+/2, clean_add, ReZero] . act relu . policy intermediate_conv(4864) . value WDL(16->FC256) . bfloat16` · **lineage** `cwkO` (saved) / `jaq1`→`jaq1-4` (live champion; promotions fork the ID) · **build** 1781 (fresh) → 1782 (resumed) · **logs** `dcm_log_20260608-133428.txt` (steps 1–520) + `dcm_log_20260608-140857.txt` (steps 509–13,868, the main run) · **dates** 2026-06-08 (~13:34 → 21:58 CDT, ~8.5h; stopped by manual save) · *(safetensors-native, embedded-config identity per PLAN §6 — no `arch_hash`; isolate by the `[ARCH]` summary line + log file)*

> **First tower change in the series.** Experiments 1–3 held a fixed 5-block / 7×7 / 128-ch tower and varied only the input encoding. Experiment 4 keeps Exp 3's `full10Ply10Reps210` input but **scales the tower** — a direct (partial) implementation of Exp 1/3 §7's "more channels at 3×3 kernels" + "shrink the over-wide stem" advice. It is therefore **not** a controlled step in the encoding ladder; it is an uncontrolled jump (tower width ×2, the block's second conv 7×7→3×3, smaller stem kernel, wider value FC, 2.1× params) and a single seed — read its deltas vs Exp 3 as suggestive, not attributable. The literal preceding fresh build that day was a `stem 512 / single-5×5 / 67.9M`-param probe (`gViN`, build 1781) abandoned after ~822 steps; Exp 4 is the configuration that was kept.

### 1. Architecture
- **Input:** **210 planes** × 8×8 (NCHW), `full10Ply10Reps210` — identical to Experiment 3 (10 stacked `basic20` frames + the 10 `basic30` temporal-repetition planes 200–209). **Policy** 4864 logits; **value** 3-class W/D/L head.
- **Stem:** **3×3** conv, 210 → **256** (Exp 3 was 7×7 → 128).
- **Tower:** **5** pre-activation residual blocks, **256 ch**. Each block has **two convs** (a two-conv residual block, same as Exp 1–3 — `blockConv1KernelSize` + `blockConv2KernelSize`), but the **second conv is now 3×3 instead of 7×7** (block kernels **7×7 → 3×3**; Exp 1–3 were **7×7 → 7×7**, which the summary collapses to "7x7" because the two are equal) → scale-and-bias **SE (reduction /2)** → clean identity add scaled by per-block ReZero α (`1/√5`). Activation ReLU. The tower still has 10 convs (5 blocks × 2), same as Exp 3; what changed is the second kernel (7×7→3×3) and the channel width.
- **Policy head:** `intermediate_conv` → 4864 (unchanged). **Value head:** 1×1 conv → 16 → BN/ReLU → flatten(1024) → FC 1024→**256** → ReLU → FC 256→3 (W/D/L), categorical-CE (head width 128→256).
- **Precision:** bfloat16. **Params:** **20,349,716 (~20.35M)** — ~2.1× Exp 3's 9.57M. **Arch version:** v4.
- *Context:* depth was **not** increased (still 5 blocks) and the stem was **not** taken all the way to 1×1; the advice was taken on two axes (more channels, and moving toward 3×3 by shrinking the block's **second** conv from 7×7 to 3×3) and partially on a third (7×7→3×3 stem). The block's **first** conv stays 7×7 — consistent with Exp 1's forensic finding that the tower uses its full 7×7 reach — so each block now pairs one wide (7×7) and one cheap (3×3) conv instead of two 7×7s.
- **Optimizer / regularization (unchanged from Exp 1–3):** constant LR **1e-2**, weight_decay **1e-4**, grad_clip 30, μ 0.90, entropy_bonus 0, draw_penalty 0; batch 4096, 800 self-play workers, promote ≥ 0.53 (later 0.55), 400-game arenas.

### 2. Relevant saved sessions
`.dcmsession` autosaves (safetensors-native; saved lineage `cwkO`, live champion `jaq1`→`jaq1-4`). Session filenames are **UTC**-stamped; the steps below are each session's `trainingSteps`, matching the four arena promotions plus the final manual save (CDT times in parentheses).

| Saved session (`.dcmsession`) | Step @ snapshot | Trigger |
|---|--:|---|
| `20260608-201157-20260608-5-cwkO-promote` | 1,268 | promote → `jaq1-1` (arena #3, 15:11 CDT) |
| `20260608-205859-20260608-5-cwkO-promote` | 2,838 | promote → `jaq1-2` (arena #6, 15:58 CDT) |
| `20260608-220114-20260608-5-cwkO-promote` | 4,988 | promote → `jaq1-3` (arena #10, 17:01 CDT) |
| `20260609-000607-20260608-5-cwkO-promote` | 9,091 | promote → `jaq1-4` (arena #18, +68 Elo, 19:06 CDT) |
| `20260609-025811-20260608-5-cwkO-manual`  | ~13,868 | manual (final, 21:58 CDT — best champion is the prior `jaq1-4`) |

Each `champion.safetensors` embeds the architecture in `__metadata__` (`input_encoding: full10Ply10Reps210`, `training_step`, `content_sha256`). Location: `~/Library/Application Support/DrewsChessMachine/Sessions/`.

### 3. Factuals
Wide-set (4435-puzzle) is the cross-experiment default; 200-set listed alongside (high variance). `vAbs`/`pD`/`draw%` are champion self-play. **The probe NLL is reliable only through ~step 5,000** — it blows up thereafter (see Shortcomings), which also makes `pElo` (rank-based, more robust) the only usable tactical signal after that point.

| Step | pElo (wide) | NLL (wide) | pElo (200) | NLL (200) | vAbs | pD | draw% | Detail |
|--:|--:|--:|--:|--:|--:|--:|--:|---|
| 521 | 477 | 4.17 | 588 | 4.58 | ~0.085 | 0.72 | 84% | Run start (fresh 20.35M net, champion `jaq1` / trainer `jaq1-1`). Constant LR 1e-2, wd 1e-4. |
| 1,268 | 563 | 3.71 | — | — | 0.089 | 0.72 | 87% | Promotion #3 → `jaq1-1`. Tactical climbing cleanly. |
| 2,838 | 613 | 3.78 | — | — | 0.097 | 0.75 | 87% | Promotion #6 → `jaq1-2`. |
| ~5,000 | **635** | **3.78** | — | — | 0.088 | 0.76 | 90% | Promotion #10 → `jaq1-3`. **Peak clean tactical state**; 3 promotions and wide pElo +158 in the first ~4.5k steps. |
| ~5,500 | ~590 | **5.6 → 9.9** | — | — | — | — | — | **Probe-NLL blow-up onset.** Wide NLL leaves ~3.8 and never returns; climbs to 15–17 over the next ~2k steps. pElo begins oscillating 370–665. |
| 9,091 | ~550 | ~14 (junk) | — | — | 0.086→**0.149** | 0.73→**0.51** | 86%→**59%** | Promotion #18 → `jaq1-4`, arena **+68 Elo** (the run's only large win). **Champion value head turns decisive here**: by ~step 9,410 vAbs 0.086→0.15, pD 0.73→0.51, draws 86%→59%, mean self-play game length **~280→~122 plies (halved)**. The champion (`jaq1-4`) is the best net of the run. |
| 13,868 | 623 (junk NLL) | 13.2 | 727 | 13.7 | 0.161 | 0.50 | 57% | **Final step** (manual save, run stopped). 29 arenas / **4 promotions** total; trainer (`jaq1-5`) **lost every arena after #18** (scores 0.16–0.49, Elo to −286), zero further promotions across the last ~4.8k steps. |

*Training-distribution telemetry stayed healthy the entire run — `pEnt` 2.65→2.49, `gNorm` ~1.3–3.1, `pLogitAbsMax` ~13.7 (flat), `pwNorm` 12.7→14.0 (mild), no NaN, legal-mass probe steady 0.85–0.89. The blow-up is **invisible** on `[STATS]`; only the out-of-distribution Lichess probe NLL and the arena reveal it.*

### 4. Wins
- **Value head turns decisive, strongly.** At the `jaq1-4` promotion (~step 9k) the champion's self-play went from ~86% draws / ~280-ply shuffling to **57% draws / ~122-ply decisive games**, with vAbs ~0.086→0.16 and pD 0.73→0.50. The clearest decisive-value transition of any experiment so far, and the engine genuinely started converting advantages.
- **Fast, clean early tactical bootstrap** through ~step 5k: wide pElo **477→635** (+158) with 3 promotions and wide NLL falling 4.17→3.78 — a better early tactical slope than Exp 3's 128-ch tower on the same input (≈616 wide at 5k).
- **No in-distribution instability.** On its own self-play distribution the 2.1×-capacity net was stable in bf16 (entropy, gradient norm, logit max all well-behaved) — the extra capacity did not cause training-loop divergence.

### 5. Shortcomings
- **Catastrophic out-of-distribution calibration blow-up from ~step 5,400.** Wide-set probe NLL exploded **3.78 → 8–17** and stayed pinned there for the final ~8k steps (200-set likewise). Meanwhile rank-based **pElo only oscillated** (370–665) — i.e. the policy still *ranks* tactical moves roughly as before but assigns **pathologically peaked, confidently-wrong** distributions on positions it doesn't generate in self-play. The arena confirms the over-sharpening from the other side (arena #28: candidate played-move prob ≈ **0.97** every position, value ≈ 0).
- **Trainer-lineage strength regression.** After the `jaq1-4` promotion the trainer (`jaq1-5`) was weaker than the frozen champion in **every** subsequent arena (scores 0.16–0.49, Elo −7 to −286) and earned **no further promotions** — the last ~4.8k steps were net-negative for the trainer even though the champion was fine.
- **The failure is invisible to the standard health metrics.** `pEnt`/`gNorm`/`pLogitAbsMax`/`pwNorm` all looked healthy throughout; a run watched only by `[STATS]`/the entropy & draw-collapse alarms would read as fine. Only the manual tactical probe and the arena caught it.
- **No tactical-ceiling verdict.** Clean wide pElo never cleanly exceeded ~635 before the NLL contamination, so this run cannot be compared on ceiling to Exp 1 (~879 wide) — and it is a single seed on a brand-new tower, so deltas vs Exp 3 are not attributable.

### 6. Analysis
- **The signature finding is the split:** healthy training-distribution metrics + decisive champion value head, but an exploded out-of-distribution probe NLL and a self-degrading trainer. The net learned to play its own (increasingly narrow, decisive) self-play lines extremely confidently while becoming catastrophically mis-calibrated everywhere else.

One of these is probably the driver (possibly both):
- **Hypothesis #1 — self-play distribution narrowing.** As the champion sharpened (draws 86%→57%, games halving in length), the replay buffer concentrated on a narrow band of lines; the trainer over-specialized to them and went OOD on tactical puzzles. (`diverge≈1.8` with 100% unique games is *in* the nominally-healthy band, so this is not an obvious diversity collapse — it needs the diversity-histogram + per-frame stem-norm check to confirm.)
- **Hypothesis #2 — under-regularization at 2.1× capacity.** wd 1e-4 / constant 1e-2 — the settings that held the 9.5M nets — may simply be too weak for a 20.3M policy, exactly the "weights/logits grow unbounded at wd 1e-4" concern Exp 1 §6 flagged. **Caveat that complicates H2:** `pLogitAbsMax`/`pwNorm` did **not** inflate on the training distribution here (unlike Exp 1's saturated-net over-sharpening), so any over-sharpening is *distribution-specific*, not a global logit run-away — which is more consistent with H1 than with a plain weight-norm blow-up.
- **The champion is genuinely the best net of the run** (`jaq1-4`, step 9,091 — decisive value head, won its arena +68). The regression is the **trainer lineage diverging**, not the champion degrading; the keeper checkpoint is `20260609-000607-…-cwkO-promote` (step 9,091).
- **Cleanest disambiguator:** resume from the pre-blow-up `jaq1-3` checkpoint (step 4,988) with **wd ≈ 3e-4** and/or a one-shot LR anneal, and watch the **probe NLL**: if it stays bounded it was regularization (H2); if it still explodes while training metrics stay clean it was distribution narrowing (H1).

### 7. Suggested future variants / changes
- **Make the probe NLL (and self-play diversity histogram) first-class alarms.** This blow-up was silent to every existing alarm; a "wide-NLL rising off its floor" trip would have caught it ~4k steps before the run was stopped.
- **Re-run from the `jaq1-3` (step ~4,988) checkpoint with wd 3e-4** (± single cosine anneal) to settle H1 vs H2 per §6.
- **If distribution-narrowing (H1):** raise the self-play exploration temperature / lengthen its tail, or mix a small fraction of OOD (probe-like) positions into the trainer's eval, to keep the policy honest off the self-play manifold.
- **Take the parts of Exp 1/3 §7 not yet applied:** go to a **1×1 stem** on the 210 encoding (frees ~1.3M params the 3×3 stem still partly wastes) and **add depth** (8–12 blocks) rather than only width — depth was the one axis this experiment left untouched.


---

## Experiment 5 — Re-check: Exp 1 Architecture Re-run on Bug-Fixed Code

**arch** identical to Exp 1: `v4 pre . in basic30(30) -> stem 128 (7x7) . 5x[7x7 conv, SE+/4, clean_add, ReZero] . act relu . policy intermediate_conv(4864) . value WDL(16->FC128) . bfloat16 . 8,445,748 params` · **lineage** `3p0G` (saved) / `JhJQ` (live champion) · **builds** 1795 → 1806 · **logs** `dcm_log_20260610-090909.txt` (fresh build) through `dcm_log_20260611-081931.txt` · **dates** 2026-06-10 09:09 CDT → 2026-06-11 (stopped to start Experiment 6)

**Why:** two proven training-loop concurrency bugs — the probe staging-buffer
clobber and the `exportWeights`/SGD race — were found 2026-06-09 and fixed
before this run. Every prior experiment trained with those bugs present, so
this is a clean re-run of Exp 1's exact architecture + input on fixed code:
a re-baseline, and a health check of the fixes under full load.

**Status at stop (~step 62k, ~31h):** healthy and unremarkable — **9
promotions / 109 arenas**, pEnt ~2.70, pIllM ~0.009, value head still
draw-heavy (pD ~0.76, vAbs ~0.14). Promotion cadence trails Exp 1 at matched
steps (9 vs 13 by ~40–60k), but config differences muddy the comparison: this
run carried `spDelay=3000ms` until 2026-06-11 ~16:45 CDT (set to 0 at ~step
61.8k) and stepped at roughly half Exp 1's rate (~1.9k vs ~3.7k steps/hr).
No bug-fix regression signature observed. Resumable from the `3p0G`
post-promotion autosaves.

---

## Experiment 6 — Exp 1 Resume Probe: Capacity Ceiling vs Over-Sharpening Stall

**arch** identical to Exp 1 (resumed weights, not a fresh build) · **resume point** `20260606-002543-20260601-12-5K7Z-periodic.dcmsession`, step 382,625 — the only post-cliff, pre-LR-cycling checkpoint of the Exp 1 run · **lineage / log / dates** TBD at launch (2026-06-11 →)

**Why:** Exp 1's verdict — "capacity ceiling, 5 blocks too shallow" — carries
one unresolved caveat (Exp 1 §6): at wd 1e-4 the logit/weight norms inflated
unopposed (pwNorm 13.8→22.3, pLogitAbsMax →31, gNorm →0.56), so the plateau
could instead be a weak-regularization over-sharpening stall — SGD spending
its budget inflating confidence on known lines while effective gradients
shrink. The depth-vs-capacity conclusion feeds every future architecture
choice, so it's worth one cheap resume (~a day) to pin down before investing
in depth runs.

**Protocol** (full rationale in Exp 1 §8): resume 382,625 → cycling off, LR
constant 1e-2, **wd 1e-4 → 3e-4**, ~5–10k steps (Phase A); if pinned, one-shot
anneal to LR 1e-3 (Phase B). Primary readout: does the tactical-battery pElo
break the lineage's all-run ceiling (**~977** 200-set / **~879** wide)?
Promotions are secondary (this save's champion is the older `bzw3-30`).
Verdict rule: ceiling breaks → stall (the breaking phase names the lever);
pinned through both phases → capacity ceiling confirmed, depth is the answer.

*(results pending)*
