# 2026-06-10 — Experiment 5: re-check of the Exp 1 architecture on bug-fixed code

**Status:** abandoned — stopped at step 61,816 (~30 h active) on 2026-06-11 to free the machine for [Exp 6](../20260611-exp1-resume-ceiling-probe/README.md); healthy but not run long enough to answer its question.

Migrated from `documentation/ARCH_EXPERIMENTS.md` Experiment 5 (all original detail kept; corrections marked inline and listed under Audit notes).

## Question

Two proven training-loop concurrency bugs — the probe staging-buffer
clobber and the `exportWeights`/SGD race — were found 2026-06-09 and fixed
before this run. Every prior experiment trained with those bugs present, so
this is a clean re-run of Exp 1's exact architecture + input on fixed code:
a re-baseline, and a health check of the fixes under full load. Does the fixed code train differently from [Exp 1](../20260601-5block-7x7-rezero-se/README.md) at matched steps?

## Setup

- **arch** identical to Exp 1 (`[ARCH]` line of log 20260610-090909): `v4 pre . in basic30(30) -> stem 128 (7x7) . 5x[7x7 conv, SE+/4, clean_add, ReZero] . act relu . policy intermediate_conv(4864) . value WDL(16->FC128) . bfloat16 . 8,445,748 params` (fresh build, not resumed weights).
- **lineage** `3p0G` (saved-session tag, `20260610-2-3p0G`) / `JhJQ` (live champion/trainer, `20260610-1-JhJQ-*`) · **builds** 1795 → 1806 · **logs** `dcm_log_20260610-090909.txt` (fresh build) through `dcm_log_20260611-081931.txt` · **dates** 2026-06-10 09:09 CDT → 2026-06-11 17:37 CDT (stopped to start Experiment 6; Exp 6's first log opened at 17:37:57).
- **Config** (every `[STATS]` line): batch 4096, LR 1e-2·√b, momentum 0.9, wd 1e-4, workers 800, replay-ratio target 0.48, complement-CE on, draw_penalty 0, promote ≥ 0.53, 400-game arenas — the same as Exp 1. `spDelay=3000ms` until 2026-06-11 16:49:17 CDT, then 0.
- **Bug-fix provenance** [Audit]: the fix is commit `2e58830` (2026-06-10 11:22 CDT). Builds 1795 (`2a73f5d*`), 1800 (`a0fd962*`) and 1801 (`73f539f*`) are dirty trees of commits that do **not** contain it; build 1804 (`2e58830*`, from step 10,350) is the first that does. See Caveats.
- Machine: this Mac.

## Runs

| Log (`dcm_log_…`) | Build / git | Steps | Kept |
|---|---|---|---|
| 20260610-090909 | 1795 / `2a73f5d*` | 1 → 2,623 | to 1,655 (resumed from the post-promotion save) |
| 20260610-101741 | 1800 / `a0fd962*` | 1,655 → 2,188 | to 2,131 |
| 20260610-105214 | 1801 / `73f539f*` | 2,131 → 10,324 | to 10,350 |
| 20260610-123844 | 1804 / `2e58830*` | 10,350 → 40,947 | to 38,795 |
| 20260611-081931 | 1806 / `c3eb430*` | 38,795 → 61,816 | yes |

Kept chain: 30.1 h active training, ~32.5 h wall. Final IDs: trainer `20260610-1-JhJQ-10`, champion `20260610-1-JhJQ-9`.

Checkpoints: no `3p0G` session and no `JhJQ` model survive on disk (2026-09-29); the last saves were `20260611-220045-20260610-2-3p0G-periodic.dcmsession` (17:00 CDT, 06-11) and the post-promotion saves listed in the logs. The `[PARAM] selfPlayConcurrency 4000 -> 800`, `replayRatioTarget 1.00 -> 0.48`, `signedAdvantageComplementCE false -> true`, `maxDrawPercentPerBatch 70 -> 100` lines at the 06-11 08:20 resume are the app's defaults being overwritten by the session's saved values; `[STATS]` shows workers=800, target=0.48, complCE=on before and after, so they are not a config change.

## Results

**Status at stop (~step 62k, ~31h):** healthy and unremarkable — **9
promotions / 109 arenas**, pEnt ~2.70, pIllM ~0.009, value head still
draw-heavy (pD ~0.76, vAbs ~0.14). Promotion cadence trails Exp 1 at matched
steps (~~9 vs 13 by ~40–60k~~ → 8 vs 13 by 40k, 9 vs 16 by 60k), but config differences muddy the comparison: this
run carried `spDelay=3000ms` until 2026-06-11 ~~~16:45 CDT (set to 0 at ~step
61.8k)~~ → 16:49:17 CDT (first 0 ms `[STATS]` at step 57,671) and stepped at roughly half Exp 1's rate (~1.9k vs ~3.7k steps/hr). [Audit: both rates are wall-clock; active-training rates are ~2.05k vs ~4.6k steps/hr. Exp 1 **also** ran spDelay 3000 ms throughout, so spDelay is not a difference between the two runs over the 0–57.7k range; the halved step rate is unexplained by config (same workers, batch, ratio target).]
No bug-fix regression signature observed. Resumable from the `3p0G`
post-promotion autosaves. [Audit: no longer — all `3p0G` sessions have been pruned.]

| Metric | Exp 5 (JhJQ) | Exp 1 (bzw3) at matched steps |
|---|---|---|
| Steps / active hours | 61,816 / 30.1 h | ~61.8k reached after ~17 h active |
| Arenas / promotions | 109 / 9 | 62 arenas and 16 promotions by 58,515 |
| Promotions by 40k / 60k | 8 / 9 | 13 / 16 |
| Final health (median of last 200 `[STATS]`) | pEnt 2.69, pIllM 0.010, pD 0.754, vAbs 0.137, gNorm 2.87 | (40–80k windows) pEnt 2.61–2.62, pIllM 0.007–0.009, pD 0.64–0.72, vAbs 0.16–0.20, gNorm 3.15–3.17 |
| 200-set pElo / NLL, 20–30k | 718 / 3.69 | 706 / 3.74 |
| 200-set pElo / NLL, 40–50k | 705 / 3.63 | 809 / 3.51 |
| 200-set pElo / NLL, 50–60k | 736 / 3.54 | 808 / 3.51 |
| Wide pElo / NLL, 50–60k | 643 / 3.48 | — (not instrumented before 189,736) |
| Wide pElo / NLL, 60–61.8k | 650 / 3.47 | — |

Promotions (from `Verdict: PROMOTED`): arena #1 @ 1,655 (→JhJQ-1), #14 @ 13,455, #17 @ 14,927, #24 @ 18,430, #32 @ 22,638, #39 @ 23,499, #64 @ 33,926, #74 @ 38,795, #92 @ 48,952 (→JhJQ-9). Last two arenas #108 (48.2%) and #109 (51.4%), kept.

**pElo scale:** in-app probe values on builds 1795–1806. `selfplay_registry.json`'s `endpoint_pElo` 573.2 / NLL 3.9288 for JhJQ is a later re-probe on a different binary and is not comparable.

## Conclusion

- The fixed code trained stably at this architecture — no NaN, no entropy or value-head collapse (pD 0.75, well away from 1), pIllM ~0.01 by 60k.
- Through 20–30k the two runs matched on the 200-set probe (718 vs 706). From 40k on, Exp 5 fell ~70–100 pElo behind Exp 1 and promoted about half as often. With one seed each, and the bug fix only covering Exp 5 from step 10,350, this run can't tell whether that gap comes from the fix, from seed noise, or from the halved step rate (wall-clock time per step affects how many self-play games land between steps).
- Nothing in 62k steps says anything about Exp 1's ~340k ceiling, which was the question that mattered; the machine went to Exp 6 instead.

## Caveats

- **Bugs present for the first 10,350 steps:** only build 1804 onward contains `2e58830`. Whether the dirty working trees of builds 1795/1800/1801 already carried the fix can't be checked from the logs.
- One seed per run; the Exp 1 comparison is a single other seed.
- Step rate halved vs Exp 1 at identical listed config (workers 800, spDelay 3000, ratio 0.48). Games per training step (the replay-ratio producer/consumer) were not compared.
- spDelay changed to 0 for the last 4,145 steps only.
- `documentation/dashboards/selfplay_probe/JhJQ.csv` stops at step 58,626 (2,347 rows); the logs have wide ticks to 61,816 (2,475 on the kept chain).

## Follow-ups

- If the fixed-code baseline matters, re-run it past ~150k steps (with a second seed) or resume from a fixed-code Exp 1-architecture checkpoint; none survives today.
- Work out why the step rate halved at unchanged config before comparing step-matched curves across builds.

## Audit notes

Verified against: the five logs listed in Runs (all present; `[STATS]`, `[ARENA]`, `[TACTICAL-LICHESS]`, `[PARAM]`, `[CHECKPOINT]` lines, scoped to the kept chain); `git merge-base --is-ancestor 2e58830 <build commit>` for each build; CHANGELOG entry "2026-06-10 CDT — Fix two probe/training concurrency bugs" (`2e58830`); `documentation/dashboards/data/JhJQ.csv` (ends 61,816) and `selfplay_probe/JhJQ.csv`; `selfplay_registry.json` (9 promotions, 8,445,748 params — consistent).

Confirmed: 9 promotions, 109 arenas (arena numbers 1–109, contiguous on the kept chain), pEnt ~2.70, pIllM ~0.009 (last line 0.0084), pD ~0.76, vAbs ~0.14, builds 1795 → 1806, stop at ~62k (61,816), ~31 h (30.1 h active / 32.5 h wall), ~1.9k steps/hr (wall).

Corrections:
- "fixed before this run" → fixed from build 1804 / step 10,350 (the first three builds lack commit `2e58830`).
- "9 vs 13 by ~40–60k" → 8 vs 13 by 40k; 9 vs 16 by 60k.
- spDelay → 0 "~16:45 CDT, ~step 61.8k" → 16:49:17 CDT (`[PARAM] selfPlayDelayMs: 3000 -> 0`), step 57,671.
- The spDelay difference cited as a confound doesn't exist: Exp 1 also ran 3000 ms throughout.
- "Resumable from the 3p0G post-promotion autosaves" → all pruned.

Unverifiable:
- Whether builds 1795/1800/1801 carried the bug fix uncommitted.
- Checkpoint `__metadata__`: no JhJQ/3p0G safetensors survive.
