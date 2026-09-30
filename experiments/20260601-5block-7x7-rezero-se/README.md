# 2026-06-01 — Experiment 1: 5-block 7×7-wide tower (ReZero / SE), self-play

**Status:** done — ran 0 → 467,099 steps (2026-06-01 → 2026-06-06); plateaued, stopped. Its ceiling question was followed up by the [resume probe](../20260611-exp1-resume-ceiling-probe/README.md) and the fixed-code [re-run](../20260610-exp1-recheck-bugfixed/README.md).

Migrated from `documentation/ARCH_EXPERIMENTS.md` Experiment 1 (all original detail kept; corrections are marked inline and listed under Audit notes). Part of the 2026-06-01 architecture series: **Session A** = the 12-block 3×3 v4 baseline (lineage `LWKa`, saved-session tag `WcRm`), **Session B** = this run. The wider May–June self-play sweep that includes `LWKa` is written up in [`../20260514-selfplay-lr-threshold-sweep/`](../20260514-selfplay-lr-threshold-sweep/README.md); only the A-vs-B comparison is repeated here.

## Question

With the 12-block 3×3 v4 tower as baseline, does a **shallow but wide-receptive-field** tower (5 blocks, 7×7 kernels, ~2.2× the parameters) train a stronger self-play network? Where does it plateau, and why?

## Setup

- **arch** (old summary format, as logged by the `[ARCH]`/`[BUILD]` lines of every later log of this lineage):
  `v4 pre . in basic30(30) -> stem 128 (7x7) . 5x[7x7 conv, SE+/4, clean_add, ReZero] . act relu . policy intermediate_conv(4864) . value WDL(16->FC128) . bfloat16 . 8,445,748 params`
  New grouped form (per the 2026-06-12 mapping in ARCH_EXPERIMENTS.md):
  `v4 . in basic30(30) -> stem 128 (7x7) . 5x[7x7+7x7 @128, SE+/4, relu/pre, clean_add, ReZero(0.447), drop*1] . act relu . policy intermediate_conv(4864) . value WDL(16->FC128) . bfloat16 . 8,445,748 params`
- **lineage** `5K7Z` (saved-session tag) / `bzw3` (live champion/trainer IDs, `20260601-11-bzw3-*`) · **dates** 2026-06-01 16:27 CDT → 2026-06-06 20:58 CDT · legacy `.dcmmodel` tag `arch_hash=0xdf23a86c` (non-authoritative; it collides with Exp 2 — identity is the embedded config per RUNTIME_ARCHITECTURE_CONFIG_PLAN §6).
- **Builds** 1566 (git `4f9456b*`, branch `bf16-trainer`) → 1752 (git `73978b7*`, branch `safetensors-storage`), 17 launches (see Runs). Machine: this Mac (all logs local).
- **Training config** (every `[STATS]` line of the kept chain): batch 4096, `lr=1.0e-02·√b` base, momentum 0.9, weight_decay 1e-4, grad_clip 30, entropy_bonus 0, draw_penalty 0, complement-CE on, promote ≥ 0.53, 400 arena games every 900 s, workers 800, replay-ratio target 0.48 (auto off), `spDelay=3000ms` on **every** `[STATS]` line from step 1 to the end, self-play tau 1.00/0.50/0.007, arena tau 0.60/0.20/0.020, replay buffer 1,000,000.

### 1. Architecture
- **Input:** 30 planes × 8×8 (NCHW), current-player perspective; **policy** 4864 logits (76×64, AlphaZero encoding); **value** 3-class W/D/L head.
- **Stem:** 7×7 conv, 30 → 128.
- **Tower:** **5** pre-activation (ResNet-v2) residual blocks, 128 ch. Each block = 7×7 same-padded conv (pad 3) → scale-and-bias **SE** (reduction /4) → clean identity add scaled by a per-block ReZero/SkipInit scalar α init `1/√5`. Activation ReLU. [Audit: each block has **two** 7×7 convs (`conv1`, `conv2`, both `[128,128,7,7]`), each with its own BN; `rezero_alpha_init` = 0.4472136 = 1/√5 confirmed in the embedded architecture JSON.]
- **Policy head:** ~~1×1 conv 128 → 76 → 4864 logits.~~ → 1×1 pre-conv 128 → 128 (`policy.pre_conv`, bias-free) → BN → ReLU → 1×1 conv 128 → 76 (+bias) → 4864 logits (see Audit notes).
- **Value head:** 1×1 conv 128 → 16 → BN/ReLU → flatten(1024) → FC 1024→128 → ReLU → FC 128→3 (W/D/L), categorical-CE loss.
- **Precision:** bfloat16 compute. **Params:** 8,445,748 (~8.45M) [Audit: verified — 100 weight tensors summing to 8,445,748 in both `champion.safetensors` and `trainer.safetensors` of the surviving session; `parameterCount` 8,445,748 in the embedded architecture]. **Arch version:** v4.
- *Context:* 7×7 kernels carry 5.4× the weights of 3×3, so despite only 5 blocks this is ~2.2× the 12-block 3×3 baseline's parameter count. [Audit: 8,445,748 / 3,898,139 = 2.17×; the baseline count is from the arch-lineage table in memory / `selfplay_registry.json`, not re-verifiable from a header (no `LWKa`/`WcRm` checkpoint survives).]

## Runs

Kept chain (a resume that rewinds to an earlier save abandons the tail of the previous log; those tails are listed separately). Steps from the first/last `[STATS]` line of each log.

| Log (`dcm_log_…`) | Build | Steps | Kept? |
|---|--:|---|---|
| 20260601-162715 | 1566 | 1 → 14,183 | to 14,014 |
| 20260601-205349 | 1569 | 14,014 → 60,493 | to 60,131 |
| 20260602-102407 | 1576 | 60,131 → 70,903 | **abandoned** (next log re-resumed 60,131) |
| 20260602-132758 | 1588 | 60,131 → 66,653 | to 66,605 |
| 20260602-145046 | 1590 | 66,605 → 85,039 | to 84,721 |
| 20260602-193914 | 1609 | 84,721 → 90,069 | to 90,043 |
| 20260602-204349 | 1610 | 90,043 → 92,569 | **abandoned** |
| 20260602-211401 | 1611 | 90,043 → 90,131 | **abandoned** |
| 20260602-211938 | 1612 | 90,043 → 152,402 | to 152,390 |
| 20260603-104854 | 1620 | 152,390 → 153,785 | **abandoned** |
| 20260603-142226 | 1632 | 152,390 → 159,090 | to 156,390 |
| 20260603-161620 | 1636 | 156,390 → 189,724 | to 189,639 |
| 20260604-020243 | 1642 | 189,639 → 253,796 | to 251,597 |
| 20260604-145335 | 1645 | 251,597 → 383,155 | to 382,635 |
| 20260605-193530 | 1693 | 382,635 → 399,919 | **abandoned** (constant LR 1e-2, wd 1e-4; 14 arenas, 0 promotions) — the next log re-resumed 382,635 |
| 20260606-002811 | 1716 | 382,635 → 470,822 | to 465,670 (the 465,670 → 470,822 tail, arenas #393–#399, was abandoned) |
| 20260606-202202 | 1752 | 465,670 → 467,099 | yes (final segment; manual save + stop) |

Kept chain: 100.9 h of active training (sum of `[STATS] elapsed` spans), ~124.5 h wall (06-01 16:27 → 06-06 20:58).

Surviving checkpoints (identified by safetensors `__metadata__`):
- `Sessions/20260607-015807-20260601-12-5K7Z-manual.dcmsession` — `champion.safetensors`: `model_id` 20260601-11-bzw3-31, `training_step` 467099, `content_sha256` ee784465…1868; `trainer.safetensors`: `model_id` 20260601-11-bzw3-32, `training_step` 467099, `content_sha256` 0399d95b…f6fa, 72 `opt.*.velocity` tensors. Replay buffer 7.2 GB.
- `Models/20260607-015745-20260601-11-bzw3-31-manual.safetensors` — `model_id` 20260601-11-bzw3-31, `training_step` 467065, same `content_sha256` as the session champion (identical weights).
- Every other 5K7Z session listed below has been pruned from disk.

### 2. Relevant saved sessions
Resumable `.dcmsession` snapshots (weights + replay buffer + params), covering steps ~276k–467k of a 0–~470k run.

| Saved session (`.dcmsession`) | Step @ snapshot | On disk (2026-09-29) |
|---|--:|---|
| `20260605-002429-20260601-12-5K7Z-periodic` | ~~276,276~~ → 276,293 | pruned |
| `20260606-002543-20260601-12-5K7Z-periodic` | ~~382,625~~ → 382,635 | pruned |
| `20260606-230443-20260601-12-5K7Z-periodic` | ~~465,652~~ → 465,670 | pruned |
| `20260607-015807-20260601-12-5K7Z-manual`   | ~~467,077~~ → 467,099 | **present** |

[Audit: steps from the `[SEGMENT] close (save) … -> N` line paired with each `[CHECKPOINT] Saved session` line, and for the survivor also from `training_step` in both safetensors headers; every resume of `…-002543` starts at `steps=382635`.]

Location: `~/Library/Application Support/DrewsChessMachine/Sessions/`

Resume-point characterization (log forensics 2026-06-11) — the saves differ
materially in what the trainer state carries:

- **276,276** [Audit: 276,293] — pre-cliff (marginal-promotion era; the cadence cliff is at ~340k).
- **382,625** [Audit: 382,635] — **the only post-cliff, pre-LR-cycling checkpoint**: LR cycling
  began at step 382,728, ~100 steps and four minutes after this save. [Audit: 93 steps and ~1 min of training *after the resume* in log 20260606-002811 (00:29:57 CDT 06-06); the save itself was at 19:25:56 CDT 06-05, ~5 h of wall time earlier. And LR cycling ran only 382,728 → ~382,820 at that point — see §3 audit rows.] Chosen
  starting point for the ceiling-vs-stall resume probe (§8).
- **465,652** [Audit: 465,670] — mid-cycling snapshot; `trainer.safetensors` includes the SGD
  velocity tensors (`opt.*.velocity`), captured during an lr≈1.8e-1 hot phase. [Audit: first `[STATS]` after resuming it reads `lr=1.8e-01·cyc`; the save is pruned so its velocity tensors can't be re-read, but the surviving 467,099 trainer carries them.]
- **467,077** [Audit: 467,099] — taken right after the run's final constant-1e-1 segment (§3),
  so the trainer weights and velocity carry that 10×-LR kick.

## Results

### 3. Factuals

Original table (values as recorded in ARCH_EXPERIMENTS.md), with an audit column of the in-app probe re-derived from the logs as the mean of all `[TACTICAL-LICHESS]` ticks within ±1,000 steps on the kept chain (per-tick SD at this stage ≈ 17–19 pElo, so single values move by ±20).

| Step | pElo (wide) | NLL (wide) | pElo (200) | NLL (200) | Detail | Audit: ±1k-step tick mean (wide / 200) |
|--:|--:|--:|--:|--:|---|---|
| 0 | — | — | 701 | 4.27 | New 5-block 7×7 network, **constant LR 1e-2** (weight_decay 1e-4, grad_clip 30, entropy_bonus 0, draw_penalty 0). | — / 637, 4.88 (first ticks 727, 614, 637: too noisy for a single value; 701 not reproducible) |
| 3,104 | — | — | 713 | 4.12 | First promotion (arena #3). | — / 725, 3.99 |
| ~40,000 | — | — | 801 | 3.54 | **13 promotions** reached by here — fast early bootstrap. | — / 800, 3.51 (13th promotion = arena #36 at 39,987 — confirmed) |
| ~190,000 | 806 | 3.25 | 924 | 3.33 | **Wide-set (4435-puzzle) probe instrumented from ~here** — wide-set coverage begins. | 803, 3.26 / 917, 3.35 (first wide tick at 189,736) |
| 339,874 | 871 | 3.18 | 977 | 3.21 | **Turning point** — #279, the ~~29th~~ **30th** and final normal-cadence promotion (→ `bzw3-30`); promotion cadence collapses here (capacity ceiling). | 870, 3.17 / 976, 3.21 |
| 382,728 | 871 | 3.21 | 961 | 3.23 | **Param change:** LR cycling introduced (peaks ~3e-1, troughs ~1e-4), toggled on/off thereafter; **momentum cycles with it** (μ 0.90 ↔ ~0.855). A one-minute constant-**1e-1** poke at step ~382,722 (06-05 19:27) preceded it and was rolled back by a session resume. [Audit: see "LR/momentum schedule, verified" below — on the kept chain LR cycling was on only ~92 steps here; 382,820 → 432,577 was constant LR 1e-2 with **momentum-only** cycling; LR cycling (1e-4 ↔ 3e-1) plus momentum cycling ran 432,652 → 465,670. The 1e-1 poke is confirmed: `[PARAM] learningRate 1e-2 -> 1e-1` 19:26:15, back 19:27:16, in log 20260604-145335, after the 382,635 save.] | 874, 3.17 / 971, 3.19 |
| 409,245 | 879 | 3.16 | 968 | 3.17 | Last promotion — #343 → champion `bzw3-31`; landed inside a **constant-1e-2** window. [Audit: LR constant 1e-2, but momentum was cycling (`μ=0.873·cyc` at 409,265).] | 883, 3.16 / 968, 3.16 |
| 465,739 | — | — | — | — | **Param change (final):** cycling off → **constant LR 1e-1** (10× base) for the last ~1,360 steps, until the manual save/shutdown at 467,099 (06-06 20:23–20:58). The end-of-run trainer state carries this hot segment. [Audit: confirmed — `[PARAM] learningRate 1e-2 -> 1e-1` 20:22:32 and `lr_momentum_cycle: lr=off mom=off`; first 1e-1 `[STATS]` at 465,739.] | — |
| ~470,000 | 876 | 3.17 | 961 | 3.21 | Run assessed: ~~398 arenas, **30 promotions total**~~ → **392 arenas, 31 promotions** on the kept chain (398/399 arena numbers include the abandoned 465,670–470,822 tail), plateaued. LR cycling earned **zero promotions**; under hot peaks, candidate arena scores drifted **below 0.5** (worse than the standing champion). | 460k–467k window: 875, 3.17 / 957, 3.21 |

*Run-config note (2026-06-11 forensics): `selfPlayDelay` was **3000 ms** over the entire verified span (step ~251k → end; every `[STATS]` line), alongside workers=800, batch=4096, decay=1e-4, promote≥0.53, unchanged taus. The run still averaged ~3.7k steps/hr — keep this in mind when comparing step rates across runs with different spDelay settings.* [Audit: `spDelay=3000ms` on every `[STATS]` line of **every** log from step 1, not just from ~251k. ~3.7k steps/hr is the **wall-clock** average (467,099 steps / ~124.5 h); per active-training hour it is **~4.6k steps/hr** (100.9 h). Per-log active rates: ~3.5k/h (0–60k), ~5.0–5.2k/h (60k–253k), ~4.5–4.6k/h (253k–465k).]

*The **wide set (4435 puzzles)** is the cross-experiment default, but was only instrumented from ~step 190k — early rows show **200-set** only. Both are listed here so this run stays comparable to priors (200-set) and future runs (wide-set). Wide tracks ~90–100 pElo below the 200-set with the same shape.* [Audit: 10k-window means give a 200-minus-wide gap of 90–117 (e.g. 113 at 190–200k, 100 at 340–350k, 82 at 460–467k); "~90–115" is more accurate.]

**pElo scale:** every pElo in this write-up is the in-app `[TACTICAL-LICHESS]` probe on the June 2026 recording builds (1566–1752). It is **not** on the scale of `selfplay_registry.json`'s `endpoint_pElo` for bzw3 (471.2, NLL 5.1859 — a later re-probe with a different binary), nor on the July+ replay-era probe scale (~900–1770). Do not compare across these without saying which scale.

#### Promotion list (kept chain, from `[ARENA] … Verdict: PROMOTED`)

| # | Arena | Step | Score | → | # | Arena | Step | Score | → |
|--:|--:|--:|--:|---|--:|--:|--:|--:|---|
| 1 | 3 | 3,104 | 57.6% | bzw3-1 | 17 | 71 | 69,179 | 53.1% | bzw3-17 |
| 2 | 5 | 4,857 | 54.1% | bzw3-2 | 18 | 85 | 88,673 | 54.1% | bzw3-18 |
| 3 | 6 | 5,707 | 54.2% | bzw3-3 | 19 | 119 | 132,483 | 53.5% | bzw3-19 |
| 4 | 12 | 10,911 | 53.2% | bzw3-4 | 20 | 137 | 156,390 | 53.2% | bzw3-20 |
| 5 | 15 | 15,292 | 54.9% | bzw3-5 | 21 | 146 | 168,818 | 53.0% | bzw3-21 |
| 6 | 19 | 20,254 | 58.5% | bzw3-6 | 22 | 162 | 189,639 | 53.0% | bzw3-22 |
| 7 | 21 | 22,421 | 54.9% | bzw3-7 | 23 | 191 | 227,167 | 53.0% | bzw3-23 |
| 8 | 28 | 30,653 | 54.9% | bzw3-8 | 24 | 196 | 233,890 | 53.4% | bzw3-24 |
| 9 | 29 | 31,891 | 57.1% | bzw3-9 | 25 | 209 | 251,597 | 53.6% | bzw3-25 |
| 10 | 31 | 34,392 | 53.2% | bzw3-10 | 26 | 211 | 254,431 | 53.1% | bzw3-26 |
| 11 | 33 | 36,896 | 53.2% | bzw3-11 | 27 | 237 | 290,200 | 53.4% | bzw3-27 |
| 12 | 35 | 39,121 | 53.1% | bzw3-12 | 28 | 261 | 318,371 | 54.2% | bzw3-28 |
| 13 | 36 | 39,987 | 54.5% | bzw3-13 | 29 | 272 | 331,675 | 53.0% | bzw3-29 |
| 14 | 47 | 49,163 | 54.9% | bzw3-14 | 30 | 279 | 339,874 | 57.9% | bzw3-30 |
| 15 | 61 | 57,957 | 53.9% | bzw3-15 | 31 | 343 | 409,245 | 53.1% | bzw3-31 |
| 16 | 62 | 58,515 | 55.6% | bzw3-16 | | | | | |

Cumulative promotions by step: 13 by 40k, 16 by 60k, 18 by 106.7k, 26 by 270k, 30 by 339,874, 31 total.

#### Arena scores by phase (kept chain)

| Phase | Steps | Arenas | Promotions | Mean score | Min | Arenas < 50% |
|---|---|--:|--:|--:|--:|--:|
| Pre-cliff, constant LR 1e-2 | 1 – 339,874 | 279 | 30 | 50.1% | 42.0% | 136 |
| Post-cliff, constant LR 1e-2 | 339,875 – 382,635 | 42 | 0 | 49.1% | 44.5% | 24 |
| Momentum-only cycling, LR 1e-2 | 382,636 – 432,600 | 41 | 1 | 49.0% | 44.9% | 23 |
| LR + momentum cycling | 432,601 – 465,670 | 28 | 0 | 46.7% | 39.8% | 27 |
| Constant LR 1e-1 | 465,671 – 467,099 | 2 | 0 | 45.0% | 44.0% | 2 |

#### LR/momentum schedule, verified (`[PARAM]` lines + `lr=`/`μ=` fields of `[STATS]`)

- 1 → 382,635: `lr=1.0e-02·√b`, μ 0.900, decay 1e-4 (plus the 1e-1 poke of ~56 steps at 382,722–382,778 in log 20260604-145335, rolled back).
- 382,635 → 382,820 (log 20260606-002811, 00:29:14–00:29:59): LR cycling enabled while its bounds were being dialled (min 1e-4, max briefly 3e-1 then set back to 3e-2), then **disabled**.
- 382,820 → 432,577: LR constant 1e-2; **momentum cycling only** (μ 0.85 ↔ 0.95, period 8,000 steps).
- 432,652 → 465,670 (from 11:07 06-06): LR cycling on (1e-4 ↔ 3e-1, period 4,000 steps) together with momentum cycling (inverted, μ 0.85 ↔ 0.95).
- 465,739 → 467,099: both cycles off; constant LR 1e-1.

#### Late-run health (median of `[STATS]` per 20k-step window, kept chain)

| Window | pwNorm | pLogitAbsMax | gNorm | pEnt | pD | vAbs | pIllM |
|---|--:|--:|--:|--:|--:|--:|--:|
| 20k | 13.81 | 15.72 | 1.80 | 2.634 | 0.704 | 0.130 | 0.0121 |
| 140k | 15.04 | 15.28 | 1.90 | 2.627 | 0.623 | 0.215 | 0.0034 |
| 280k | 16.00 | 13.35 | 1.18 | 2.587 | 0.473 | 0.286 | 0.0018 |
| 340k | 16.50 | 14.22 | 1.05 | 2.560 | 0.421 | 0.338 | 0.0016 |
| 400k | 17.25 | 18.39 | 1.02 | 2.555 | 0.434 | 0.312 | 0.0015 |
| 440k | 20.67 | 26.98 | 0.57 | 2.554 | 0.439 | 0.309 | 0.0021 |
| 460k | 21.97 | 29.92 | 0.66 | 2.554 | 0.439 | 0.309 | 0.0019 |

End of run (467,094): pwNorm 22.26, pLogitAbsMax 30.66.

#### Probe curve (10k-step window means of all ticks, kept chain)

| Steps | 200-set pElo / NLL | Wide pElo / NLL |
|---|---|---|
| 0–10k | 718 / 4.05 | — |
| 30–40k | 775 / 3.60 | — |
| 90–100k | 850 / 3.44 | — |
| 190–200k | 924 / 3.32 | 811 / 3.25 |
| 260–270k | 974 / 3.21 | 859 / 3.17 |
| 340–350k | 978 / 3.20 | 878 / 3.17 |
| 400–410k | 972 / 3.18 | 883 / 3.16 |
| 430–440k | 961 / 3.21 | 871 / 3.18 |
| 460–467k | 957 / 3.21 | 875 / 3.17 |

Best 5k-step window mean: 200-set 981.4 (from 339,426); wide 892.2 (from 397,626).

### Session A vs Session B (2026-06-01 architecture pair)

A ran concurrently with B's start on 2026-06-01; its in-series log is `dcm_log_20260601-090740.txt` (build 1562, git `c78d7c2*`, `arch_hash=0xbad32ced`, resumed from `20260601-133857-20260531-10-WcRm-periodic.dcmsession`, steps 69,644 → 106,740, trainer `LWKa-11` / champion `LWKa-10` at the end). Its full history (9 logs from 2026-05-31) belongs to the [self-play sweep write-up](../20260514-selfplay-lr-threshold-sweep/README.md).

| | A — 12-block 3×3 (`LWKa`/`WcRm`) | B — 5-block 7×7 (`bzw3`/`5K7Z`, this run) |
|---|---|---|
| Params | 3,898,139 (arch-lineage table; no checkpoint survives to re-verify) | 8,445,748 (verified from header) |
| Config | LR 1e-2, wd 1e-4, spDelay 3000 ms, batch 4096 (same as B) | same |
| Promotions by ~40k / by 106.7k | 6 / 10 (last `LWKa-10` at 75,830) | 13 / 18 |
| 200-set pElo / NLL, 70–80k | 708 / 3.62 | 838 / 3.45 |
| 200-set pElo / NLL, 90–100k | 721 / 3.59 | 850 / 3.44 |
| 200-set pElo / NLL, 100–106.7k (B: 100–110k) | 721 / 3.59 | 850 / 3.43 |
| Health at ~106k (median of last 100 `[STATS]`) | pEnt 2.62, pIllM 0.006, pD 0.69, vAbs 0.20, gNorm 2.59 | (at 100–120k) pEnt 2.60, pIllM 0.004, pD 0.71, vAbs 0.17, gNorm 2.46 |
| Active steps/hr (spDelay 3000) | ~3.4k (log 20260601-090740) | ~5.0k (logs at 90–150k) |

At matched steps B was ~130 pElo (200-set) ahead of A and promoted nearly twice as often. A was stopped at 106,740, so this says nothing about A's ceiling; the "depth matters more" conclusion below is an inference from B's saturation, not an A-vs-B measurement at equal budget.

## Conclusion

### 4. Wins
- **No training instability — including step 470k.** No entropy collapse (held ~2.55 nats), no value-head ~~draw-collapse~~ collapse (pD ~0.44), no illegal-move blowup (~~~0.003~~ → ~0.0015–0.002), no gradient explosion (gNorm ~0.57), no NaN/divergence — bf16 stable even at 3e-1 LR peaks.
- **Productive early/mid-run:** pElo climbed 701 → ~970 (200-set) [Audit: 10k-window means 718 → 978], earning ~~29 of 30~~ **30 of 31** promotions by ~340k, steepest in the first 60k (13 promotions by 40k; 16 by 60k).

### 5. Shortcomings
- **Strength plateaued at ~step 270k** (pElo/NLL flat thereafter); the arena eked out marginal promotions until ~step 340k, then stopped entirely. The final ~130k steps were unproductive. [Audit: flat on the 200-set from ~260k (window means 955–978); the wide set still crept up ~+20 (859 at 260–270k → 883 at 400–410k) before slipping back to ~871–875. There was one more promotion after 340k (#343 at 409,245).]
- **The post-saturation phase actively regressed.** Candidates scored **below 0.5** vs the frozen champion (to ~~~0.42~~ → min 0.398, arena #364 at 433,845 — *worse* nets), and `pwNorm`/`pLogitAbsMax` inflated monotonically (13→22 / 20→31), over-sharpening on forced lines. No collapse, but churn producing worse-than-champion weights. [Audit: pwNorm did rise monotonically (13.1 at 3k → 22.26 at 467k). pLogitAbsMax was **not** monotone: ~15–16 early, down to ~13.3 at 280–310k, then up to 30.7 — "20→31" holds from ~400k. Sub-0.5 scores were common even pre-cliff (136 of 279 arenas; mean 50.1%); the phase that clearly shifted the mean down was LR cycling (mean 46.7%, 27 of 28 below 0.5).]
- 8.45M params in **only 5 blocks** appears to cap chess strength — **param count did not buy ceiling.**

### 6. Analysis
- **The plateau is a capacity ceiling, not a blow-up.**
- **Capacity ceiling confirmed by weight forensics (2026-06-08, on the 465k `champion.safetensors`).** Reading the saved weights directly: **0% dead channels** in every conv (weakest channel 0.6–0.9× its layer mean), **0%** near-zero BN γ, and **~91–95% effective rank** (participation ratio of singular values) in all ten tower convs. The net populated every channel and nearly every representational dimension and *still* couldn't pass pElo ~965 for 210k steps — i.e. the plateau is **not** unused capacity (which would show dead units / low rank), it's a packed net with nowhere to write new knowledge. The 7×7 tower kernels also use their full spatial extent (outer ring holds **~41%** of each kernel's energy; a 3×3 truncation would discard ~74%) — no spatial slack either. The over-sharpening the logs showed (pwNorm 13.8→22.3, pLogitAbsMax 15.6→30.7 while gNorm fell 2.0→0.56 and pElo stayed flat) is the saturated-net signature: SGD spent its remaining budget inflating confidence on known lines, not learning. *Caveat:* the forensics confirm capacity is fully **utilized** but can't fully separate a hard ceiling from a fixable weak-regularization over-sharpening stall (wd 1e-4 lets logits run; the 400k→467k blow-up is partly the LR-cycling experiment). Clean disambiguator: a **resume-from-checkpoint run with wd≈3e-4 / a one-shot LR anneal** — if pElo breaks ~977 it was regularization, if it stays pinned it was capacity. [Audit: re-derived on the surviving 467,099 `champion.safetensors` (champion `bzw3-31`, unchanged since the 409,245 promotion, so the same weights as the 465k save's champion): weakest channel 0.73–0.87× layer mean; 0 BN γ with |γ| < 1e-3; tower outer ring 39.7–43.2% of energy; energy outside the central 3×3 72.4–75.7%. Effective rank 90.7–94.9% **when defined as (Σσ)²/Σσ²/n** (reproduces the claim); the variance-weighted form (Σσ²)²/Σσ⁴/n gives only 62–76%, so "nearly every dimension" depends on the definition. ReZero α grew to 0.78, 0.93, 1.83, 3.25, 4.03 across blocks 0–4. The resume probe was run — see [Exp 6](../20260611-exp1-resume-ceiling-probe/README.md).]
- **Stem is over-wide (incidental forensic finding).** The 7×7 stem collapsed to ~1×1 (center holds 30% of energy, outer ring only 21% — vs the tower's 41% edge share): board planes are per-square one-hot, so the stem's real job is a pointwise per-square embedding and it zeroed the kernel periphery. A 1×1/3×3 stem costs ~nothing and frees ~150k params (far more on wider-input encodings). Spatial reasoning is the tower's job. [Audit: center 30.2%, outer ring 20.7%, outside-3×3 42.5% — confirmed. The 7×7 stem has 188,160 weights; a 3×3 stem saves 153,600, a 1×1 stem 184,320. "Zeroed the periphery" overstates it: 42.5% of the stem's energy is still outside the central 3×3.]
- **Promotion cadence is the cleanest strength curve.** A smooth decay ending in a cliff is the saturation signature; health metrics alone look fine well into the plateau.
- **LR cycling never earned a promotion here** — constant LR did all the work. Cycling on a saturated net is churn, not damage (gNorm stable at peaks); the sub-0.5 arena scores are cycling artifacts, not degradation. [Audit: the one post-cliff promotion (#343) came during momentum-only cycling at constant LR 1e-2.]
- **Capacity ≠ parameter count.** The shallow-but-wide 5-block/7×7 net underperformed; for chess, **depth likely matters more** than per-layer receptive field at equal budget. [Audit: "underperformed" isn't supported by the in-series comparison — B beat the 12-block A at every matched step A reached (see A vs B). What B shows is that it *saturated*; there was no equal-budget deeper run to compare against.]
- *Concern:* at weight_decay 1e-4, weight/logit norms grow unbounded; a longer or higher-LR run would need stronger decay to avoid logit blow-up.

## Caveats

- One seed; no replicate.
- Training ran with the two probe/training concurrency bugs later fixed in `2e58830` (2026-06-10); [Exp 5](../20260610-exp1-recheck-bugfixed/README.md) re-ran this architecture on fixed code.
- Probe values are noisy: per-tick SD ≈ 17–19 pElo late in the run; the 200-set is small (200 puzzles), and the wide set only exists from 189,736.
- Rewinds: five log tails were abandoned by later resumes (see Runs). Arena numbers and chart history on disk include some of them.
- The 382k–467k region mixes three schedule changes (momentum cycling, LR cycling, LR 1e-1), so it can't cleanly test anything alone.

## Follow-ups

### 7. Suggested future variants / changes
- **Restore depth:** 8–12 residual blocks at 3×3 — direct comparison to baseline A (`0xbad32ced`, 12-block 3×3) to confirm depth raises the ceiling.
- **Balanced middle:** 8–10 blocks at 5×5 to trade some receptive field for depth at a similar param budget.
- **Regularization:** bump weight_decay to ~3e-4 on long runs to cap logit/weight-norm growth.
- **LR:** drop cycling; at most a single cosine anneal to a low LR for a final polish. Keep constant 1e-2 as the workhorse.
- Only widen channels **after** depth is restored, not instead of it.
- **Refined by the weight forensics:** the tower is reach-hungry (uses the full 7×7) **and** channel-packed (0% dead, ~93% rank), so the move is **more depth + more channels at 3×3 kernels** (depth delivers the reach the tower wants, more parameter-efficiently than wide kernels — every competitive chess net is deep/wide/3×3) plus a **1×1 stem**. Do **not** narrow the tower kernels in place — that amputates needed reach. Wider *convs* are the lowest-value axis on an 8×8 board (receptive field is global after ~2 layers).

### 8. Resume probe — ceiling vs stall (protocol set 2026-06-11; ~~results pending~~ → run 2026-06-11/12, see [Exp 6](../20260611-exp1-resume-ceiling-probe/README.md))

Executes the §6 disambiguator: is the plateau a hard capacity ceiling, or a
weak-regularization over-sharpening stall (wd 1e-4 letting logit/weight norms
inflate, saturating the softmax and shrinking effective gradients)?

- **Resume point: `20260606-002543-…-5K7Z-periodic` (step 382,625 [Audit: 382,635])** — not the
  465k/467k saves. Rationale: the 465k trainer is post-83k-steps of cycling
  churn with the 200-set pElo already down 977→961 and hot velocity tensors in
  the save; a *null* result from there can't distinguish "ceiling" from
  "cycling-damaged starting point". 382,625 is post-cliff but pre-cycling, with
  the inflation pathology already present (pwNorm ~17 of the 13.8→22.3 climb) —
  the stall hypothesis is fully testable and a null is clean. [Audit: pwNorm 16.88 at 382,728.]
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

## Audit notes

Verified against: the 17 session logs listed in Runs (all present), parsed for `[STATS]`, `[ARENA]`, `[TACTICAL-LICHESS]`, `[PARAM]`, `[SEGMENT]`, `[CHECKPOINT]` lines and scoped to the kept chain; `documentation/dashboards/selfplay_probe/bzw3.csv` (11,095 rows, 189,736 → 467,077 — identical to the kept-chain wide ticks from the logs); safetensors headers + tensors of the surviving session and model.

Corrections:
- Promotions: "30 promotions total" → **31** (`bzw3-1` … `bzw3-31`); "#279, the 29th" → **30th**; "29 of 30 by ~340k" → **30 of 31** (evidence: `Verdict: PROMOTED` lines, table above).
- Arenas: "398 arenas" → **392** on the kept chain to 467,099 (arena #392 at 466,727); #393–#399 were on the abandoned 465,670–470,822 tail of log 20260606-002811.
- Saved-session steps: 276,276 → 276,293; 382,625 → 382,635; 465,652 → 465,670; 467,077 → 467,099 (`[SEGMENT] close (save)` lines; survivor's `training_step`).
- LR schedule: "LR cycling introduced at 382,728 … toggled on/off thereafter" → LR cycling on only 382,728–~382,820, then momentum-only cycling at LR 1e-2 to 432,577, then LR+momentum cycling 432,652–465,670 (no further toggling; `[PARAM]` lines of log 20260606-002811 at 00:29–00:30 and 11:07).
- spDelay 3000 ms "step ~251k → end" → from step 1 (every log).
- ~3.7k steps/hr is wall-clock; active-training rate is ~4.6k/hr.
- Policy head "1×1 conv 128 → 76" → 1×1 128→128 pre-conv + BN + ReLU, then 1×1 128→76 (tensors `policy.pre_conv.weight [128,128,1,1]`, `policy.pre_bn.*`, `policy.conv.weight [76,128,1,1]`).
- pIllM "~0.003" late → ~0.0015–0.002 (20k-window medians 360k–467k).
- Min candidate score "~0.42" → 0.398 (#364, 433,845).
- pLogitAbsMax "inflated monotonically" → not monotone (min ~13.3 at 280–310k).
- Wide-vs-200 gap "~90–100" → ~90–115.
- "Wide plateau from ~270k" → wide still rose ~+20 to 400–410k.
- "value-head draw-collapse" wording → "value-head collapse" (pD ~0.44 is nowhere near pD → 1).
- "The shallow-but-wide net underperformed" → not measured against an equal-budget deeper run; it beat the 12-block A at every step A reached.
- `data/bzw3.csv` / `selfplay_registry.json` (not edited, reported): the registry's bzw3 entry ("33 promo, 115h") merges this run, the abandoned tails, Exp 6, and four 600-step dropout-A/B fork runs; see the Exp 6 write-up's Audit notes for the full finding.

Unverifiable:
- Step-0 "701 / 4.27" and step-3,104 "713 / 4.12": no single canonical tick; the ±1k tick means (637/4.88, 725/3.99) differ. The original values were probably read off a smoothed chart.
- Session A parameter count 3,898,139: no `LWKa`/`WcRm` checkpoint survives and build 1562 predates `[ARCH]` lines; the value comes from the arch-lineage table (memory) and `selfplay_registry.json`.
- The 2026-06-08 forensics were run on the since-pruned 465,670 session; re-derived instead on the surviving 467,099 champion (same `bzw3-31` weights, since no promotion happened between them).
- Velocity tensors in the 465,670 trainer: session pruned.
