# 2026-06-11 — Experiment 6: Exp 1 resume probe — capacity ceiling vs over-sharpening stall

**Status:** done — ran 2026-06-11 17:39 → 2026-06-12 18:48 CDT (382,635 → 503,570 steps). ARCH_EXPERIMENTS.md still says "results pending"; the logs show it ran. Verdict: the wide-set ceiling did not break under stronger weight decay at LR 1e-2. The planned Phase B (LR 1e-3 anneal) never ran, so the protocol's full verdict rule was never completed.

Migrated from `documentation/ARCH_EXPERIMENTS.md` Experiment 6 (original text kept below; corrections marked inline and listed under Audit notes). Parent run: [Exp 1](../20260601-5block-7x7-rezero-se/README.md).

## Question

Exp 1's verdict — "capacity ceiling, 5 blocks too shallow" — carries
one unresolved caveat (Exp 1 §6): at wd 1e-4 the logit/weight norms inflated
unopposed (pwNorm 13.8→22.3, pLogitAbsMax →31, gNorm →0.56), so the plateau
could instead be a weak-regularization over-sharpening stall — SGD spending
its budget inflating confidence on known lines while effective gradients
shrink. The depth-vs-capacity conclusion feeds every future architecture
choice, so it's worth one cheap resume (~a day) to pin down before investing
in depth runs.

**Protocol** (full rationale in Exp 1 §8): resume 382,625 [Audit: 382,635] → cycling off, LR
constant 1e-2, **wd 1e-4 → 3e-4**, ~5–10k steps (Phase A); if pinned, one-shot
anneal to LR 1e-3 (Phase B). Primary readout: does the tactical-battery pElo
break the lineage's all-run ceiling (**~977** 200-set / **~879** wide)?
Promotions are secondary (this save's champion is the older `bzw3-30`).
Verdict rule: ceiling breaks → stall (the breaking phase names the lever);
pinned through both phases → capacity ceiling confirmed, depth is the answer.

## Setup

- **arch** identical to Exp 1 (resumed weights, not a fresh build); every log's `[ARCH] resumed session 20260601-12-5K7Z (champion …)` line reads `v4 pre . in basic30(30) -> stem 128 (7x7) . 5x[7x7 conv, SE+/4, clean_add, ReZero] . act relu . policy intermediate_conv(4864) . value WDL(16->FC128) . bfloat16 . 8,445,748 params`.
- **Resume point** `20260606-002543-20260601-12-5K7Z-periodic.dcmsession`, step ~~382,625~~ 382,635 (`[CHECKPOINT] Loaded session … savedBuild=1645 savedGit=eac5113`), champion `bzw3-30`, trainer `bzw3-31` — the only post-cliff, pre-LR-cycling checkpoint of the Exp 1 run.
- **Lineage / log / dates** (~~TBD at launch~~): saved-session tag `5K7Z` (unchanged), live IDs `20260601-11-bzw3-*`; logs `dcm_log_20260611-173757.txt`, `…-180921`, `dcm_log_20260612-110030.txt`, `…-140430`, `…-153226`; builds 1810/1811 (`c04fe86*`), 1818 and 1824 (`6e4e233*`), 1827 (`0cb7ad7*`); 2026-06-11 17:37 → 2026-06-12 18:49 CDT. Machine: this Mac.
- **Settings actually used** (`[PARAM]`, `[RESUME-PARAM]`, and the `lr=` / `decay=` / `μ=` / `spDelay=` / `workers=` fields of every `[STATS]` line):

| Phase | Steps | LR | Momentum | wd | Other changes vs Exp 1 |
|---|---|---|---|--:|---|
| A | 382,635 → 430,928 (48.3k steps) | 1e-2 constant (no `cyc` on any line) | 0.9 constant | **3e-4** (from 17:40:00 06-11) | spDelay 3000 → **0** ms; workers 800 → **170** (18:31–18:35 06-11, steps 385,238–385,650) |
| B′ (unplanned) | 430,928 → 498,397 (67.5k) | 1e-2 constant | 0.9 | **5e-4** (`[PARAM] weightDecay 3e-4 -> 5e-4`, 01:36:24 06-12) | same |
| C′ (unplanned) | 498,397 → 503,570 (5.2k) | 1e-2 constant | 0.9 | 5e-4 | **dropout 0.7** — `[RESUME-PARAM] dropout_rate: saved=nil applied=0.7 (defaulted)` on the build-1824 and -1827 resumes (the session predated the dropout parameter) |

The planned Phase B (LR 1e-3) was never applied: no `[PARAM] learningRate` change appears in any of the five logs, and every `[STATS]` line reads `lr=1.0e-02·√b`.

## Runs

| Log (`dcm_log_…`) | Build | Steps | Notes |
|---|--:|---|---|
| 20260611-173757 | 1810 | 382,635 → 383,471 | wd 3e-4 + spDelay 0 set; manual save at 383,242 (`20260611-230110-…-5K7Z-manual`); kept to 383,242 |
| 20260611-180921 | 1811 | 383,242 → 486,688 | auto-resume of 383,242; workers → 170; wd → 5e-4 at 430,928; saves at 408,794, 434,389 (periodic), 442,911 (promote), 468,057 (periodic), 475,864 (promote), 486,383 (manual); kept to 486,383 |
| 20260612-110030 | 1818 | 486,383 → 498,378 | SIGUSR2 save at 498,397 (`20260612-175413-…-sigusr2`) |
| 20260612-140430 | 1824 | 498,397 → 498,665 | dropout 0.7 defaulted in; SIGUSR2 save at 498,650 (`20260612-191329-…-sigusr2`) |
| 20260612-153226 | 1827 | 498,650 → 503,589 | dropout 0.7; manual save at 503,570 (`20260612-234833-…-manual`); run ends 18:48 CDT |

~22.5 h of active training. All Exp 6 sessions have since been pruned (the only 5K7Z session on disk is Exp 1's 467,099 save), so no Exp 6 checkpoint `__metadata__` can be read.

**ID collision:** this branch re-minted IDs Exp 1 had already used. The resumed trainer is `bzw3-31` (Exp 1's trainer at that save), promotion #359 at 442,911 minted champion **`bzw3-31`** (Exp 1 used the same ID for its #343 promotion at 409,245, which has different weights), and #380 at 475,864 minted `bzw3-32`; the trainer finished as `bzw3-33`. A `bzw3-31` or `bzw3-32` ID alone doesn't say which branch it came from.

## Results

Arenas: 86 on this branch (#322–#407; the arena counter continued from the restored history).

| Phase | Arenas | Promotions | Mean score | Min | < 50% |
|---|--:|---|--:|--:|--:|
| A (wd 3e-4) | 30 | 0 | 49.2% | 47.1% | 19 |
| B′ (wd 5e-4) | 44 | 2 — #359 @ 442,911 (54.4%, 77W/281D/42L), #380 @ 475,864 (54.2%, 75W/284D/41L) | 49.6% | 45.0% | 24 |
| C′ (wd 5e-4 + dropout 0.7) | 12 | 0 | 50.1% | 46.8% | 5 |

In-app probe (`[TACTICAL-LICHESS] tick` lines; the probe runs on the live trainer). Means of all ticks in each range; per-tick SD ≈ 12–20:

| Range | Exp 6 200-set pElo / NLL | Exp 6 wide pElo / NLL | Exp 1, same steps (200 / wide) |
|---|---|---|---|
| 382,635 → 399,919 | 964.6 / 3.189 | 876.8 / 3.167 | 966.5 / 878.9 (kept chain); the abandoned constant-LR, wd 1e-4 control log 20260605-193530: 958.7 / 875.7 |
| Phase A (→ 430,928) | 971.3 / 3.193 | 878.3 / 3.167 | 966.3 / 879.2 (momentum-only cycling, LR 1e-2) |
| Phase B′ (430,929 → 498,397) | 989.0 / 3.195 | 885.2 / 3.155 | 962.4 / 871.7 (LR + momentum cycling, then LR 1e-1; to 467k only) |
| Phase C′ (→ 503,589) | 988.6 / 3.211 | 881.9 / 3.159 | — |

Best windows vs Exp 1's whole run:

| Readout | Exp 1 (0 → 467,099) | Exp 6 (382,635 → 503,589) | Δ |
|---|---|---|--:|
| Wide, best 10k-window mean | 882.7 (400–410k) | 889.2 (460–470k) | +6.5 |
| Wide, best 5k-window mean | 892.2 (from 397,626) | 892.7 (from 463,302) | +0.5 |
| Wide NLL, best 10k-window mean | 3.160 (390–410k) | 3.142 (460–470k) | −0.018 |
| 200-set, best 10k-window mean | 977.7 (340–350k) | 997.9 (480–490k) | +20.2 |
| 200-set, best 5k-window mean | 981.4 (from 339,426) | 1001.2 (from 473,777) | +19.8 |
| Max single tick (wide / 200) | 924 / 1039 | 933 / 1058 | — |

Health (median `[STATS]` per 20k window) — did the decay bite?

| Window | pwNorm | pLogitAbsMax | gNorm | pEnt | pD | vAbs |
|---|--:|--:|--:|--:|--:|--:|
| 380k (start) | 16.76 | 16.31 | 1.07 | 2.561 | 0.420 | 0.338 |
| 420k (wd 3e-4) | 16.14 | 18.06 | 1.21 | 2.562 | 0.419 | 0.340 |
| 460k (wd 5e-4) | 14.91 | 19.15 | 1.38 | 2.539 | 0.435 | 0.316 |
| 500k (end) | 14.25 | 20.62 | 1.74 | 2.560 | 0.391 | 0.332 |

For comparison, Exp 1 over the same steps went pwNorm 16.9 → 22.3, pLogitAbsMax 16.4 → 30.7, gNorm 1.09 → 0.57.

pElo scale: every value here is the in-app probe on builds 1810–1827, comparable to Exp 1's in-app values (same probe sets, and the build-1810 resume restored Exp 1's probe history as the continuation). It is not on the scale of the registry's re-probed `endpoint_pElo` (471.2 for bzw3) or of the July+ replay-era probe.

## Conclusion

- **The decay bit, and strength didn't follow.** wd 3e-4 then 5e-4 reversed pwNorm (16.8 → 14.3) and more than doubled gNorm relative to Exp 1's same steps (1.74 vs ~0.6). Those are the recovery signs Exp 1 §8 said to watch for. pLogitAbsMax still rose (16.3 → 20.6), though far less than Exp 1's 30.7.
- **Wide set (the declared primary, lower-variance readout): pinned.** Phase A was indistinguishable from Exp 1 and from the constant-LR control (878 vs 879/876). Phase B′ averaged 885 vs 872 for Exp 1's LR-cycling arm at the same steps. But against Exp 1's own best, the best windows moved +6.5 (10k) and +0.5 (5k) pElo — well inside tick noise. NLL improved by 0.018. The ~879 ceiling (≈ 883–892 by window mean) was not broken in any meaningful sense.
- **200-set: a small rise.** +20 pElo on best-window means (978 → 998) and about +25 on phase means. That is roughly one tick SD on a 200-puzzle set, and the wide set doesn't show it, so on its own it isn't strong evidence.
- **Promotions:** 2 in 121k steps, against Exp 1's 1 in the 83k steps after 382k — the "≈1 per 85k" base rate the protocol called ambiguous. Arena means stayed ~49–50%.
- **Verdict:** under the pre-registered rule this is "pinned through Phase A (and an extended wd-5e-4 phase)". That rules out weak weight decay at LR 1e-2 as the cause of the plateau. It can't formally confirm capacity, for two reasons. Phase B (LR 1e-3 anneal) never ran. And the extra phases changed several things at once: wd 3e-4 → 5e-4, workers 800 → 170, spDelay 3000 → 0, and a defaulted dropout of 0.7 for the last 5k steps. On the evidence that does exist, the stall hypothesis (norm inflation choking learning) is weakened: norms deflated and gradients recovered without a wide-set gain. That leaves capacity, or an LR-noise floor only an anneal would test, as the likely explanations. **Result: inconclusive, leaning capacity.**

## Caveats

- Phase A ran ~48k steps, not the planned 5–10k; the wd 5e-4 phase and the dropout segment weren't part of the protocol.
- Self-play generation changed along with regularization: spDelay 3000 → 0 and workers 800 → 170 from the start of Phase A. Active step rate was 6.2k steps/h (log 20260611-180921), against ~4.5k for Exp 1 over the same steps. That changes how fresh the replay buffer is relative to the trainer.
- Dropout 0.7 was applied silently, by the resume defaulting a parameter the saved session didn't have. It covers 498,397 → 503,589 (the step rate fell to ~1.5k/h there).
- The probe measures the live trainer, not the champion; per-tick SD 12–20 pElo; adjacent ticks can differ by 20+.
- One seed; the only same-point control is the 17k-step abandoned log 20260605-193530 (constant LR 1e-2, wd 1e-4).

## Follow-ups

- Run the missing Phase B from a Phase A/B′ state: LR 1e-3 constant, ~5k steps, fixed workers/spDelay, dropout explicitly 0. None of the Exp 6 sessions survives, so it would have to start from Exp 1's 467,099 save (post-LR-cycling, carrying the hot 1e-1 velocity — the start point the protocol rejected) or from a fresh run.
- Resumed sessions should log (or refuse) parameters that get defaulted because the save predates them. `dropout_rate … (defaulted)` changed this experiment without a `[PARAM]` action.

## Audit notes

Source: the five logs in Runs, all present. Parsed `[STATS]` (lr/decay/μ/spDelay/workers/pwNorm/pLogitAbsMax/gNorm/pEnt/pD/vAbs), `[ARENA]`, `[TACTICAL-LICHESS] tick` (200-set and `set=wide`), `[PARAM]`, `[RESUME-PARAM]`, `[SEGMENT]`, `[CHECKPOINT]`. The chain was scoped by `[CHECKPOINT] Loaded session` → the next log's start step. Exp 1 comparison values come from its kept chain (see the Exp 1 write-up).

Corrections:
- "results pending" / "lineage / log / dates TBD" → ran 2026-06-11 17:37 → 2026-06-12 18:49 CDT, logs and builds as listed.
- Resume step 382,625 → 382,635.
- Protocol vs actual: Phase A ran 48.3k steps (not 5–10k). Phase B (LR 1e-3) was never run. wd was instead raised to 5e-4 at 430,928. Workers went 800 → 170, spDelay 3000 → 0, and dropout 0.7 was defaulted in from 498,397.

Finding — `selfplay_registry.json` bzw3 entry and `documentation/dashboards/data/bzw3.csv` (fixed 2026-09-29; see the last bullet):
- The four launches `dcm_log_20260612-141454`, `-143424`, `-145341`, `-151312` are **not** a continuation of the bzw3 lineage. Each is a headless `--train` run (`autoTrain=on`, builds 1824/1825/1827) that loads `Sessions/20260612-191329-20260601-12-5K7Z-sigusr2.dcmsession/champion.safetensors` (→ `20260601-11-bzw3-32`, the Exp 6 champion at 498,650) as a fresh champion. They run with `--parameters` overrides `dropout_rate` = 0.0 (runs A and B), 0.3, 0.7, plus `weight_decay=0.0005`, `replay_buffer_min_positions_before_training=200000`, `arena_auto_interval_sec=36000`, `training_step_limit=600`. Each writes `experiments/dropout-ab-trained-20260612/result_{0.00A,0.00B,0.30,0.70}.json` and exits at step 567–601. Step counters restart at 1 (with LR warm-up); the IDs are inherited (`[STATS]` shows trainer `bzw3-33`, champion `bzw3-32`), which is why they look like part of bzw3. pLoss ≈ 2.2 at step 1 because the weights are the trained champion, not a random init. They are 600-step forks belonging to the dropout A/B study, not training of this lineage.
- `data/bzw3.csv` interleaves several branches at overlapping `cum_step`s: Exp 1's kept chain; the abandoned log 20260605-193530 (segment 14); log 20260606-002811 including its abandoned 465,670–470,822 tail (segment 15); Exp 6 (segments 17–19); and the dropout A/B forks (segments 21, 23, 24, with `cum_step` = 498,665 + fork step). The last segment (25, log 20260612-153226) then gets `cum_step` = `meta_step` + 500,371, so it runs 999,054 → **1,003,960**. The real step at the end of the lineage is 503,589 (`meta_step`). The ~1.0M figure is an artifact of stacking the forks' steps into the base; the lineage never trained a million steps.
- The registry's "33 promo, 115h" counts Exp 1's 31 promotions plus Exp 6's 2, whose IDs collide with Exp 1's (see ID collision).
- **Fix (2026-09-29):** Exp 6 is its own registry entry, `bzw3e6` (logs 20260611-173757, -180921, 20260612-110030, -140430, -153226, each cut at the step the next resumed from: 383,242 / 486,383 / 498,397 / 498,650). Its `[STATS]` rows run 382,635 → 503,589 (`data/bzw3e6.csv` first bucket 382,992, last 503,589), 22.3 h summed elapsed, 2 promotions, final trainer `bzw3-33`, no endpoint re-probe (no checkpoint survives). Its wide-probe curve `selfplay_probe/bzw3e6.csv` (465 marks) was extracted from the same logs' `[TACTICAL-LICHESS] tick set=wide` lines with `selfplay_probe_append.py`. The four dropout forks are listed under `bzw3`'s `excluded_logs` and belong to neither entry. `bzw3` keeps Exp 1's kept chain (1 → 467,099, 31 promotions, 100.9 h). The tracker no longer adds its random-init anchor (pElo 450 at step 1000) to a run whose logs begin mid-lineage.

Unverifiable:
- Checkpoint `__metadata__` for every Exp 6 save: all pruned from `Sessions/` and nothing in `Models/`.
- Whether build 1818 (log 20260612-110030) applied any dropout: that log has no dropout line, and the dropout parameter first appears in build 1824.
