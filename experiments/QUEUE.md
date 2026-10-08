# Experiment queue and decision log

Standing instruction (owner, 2026-10-02): keep the GPU busy; chain runs back to
back; make hard calls autonomously, log them here, report them at the next
conversation.

## Running

| started | experiment | ends (est.) |
|---|---|---|
| 2026-10-08 09:46 | R-replay: ZlrA's settings on corpus replay (`20261008-zlra-selfplay-lr/`) | 60,000 steps |
| 2026-10-08 09:46 | R-fixedlr: ZlrA's settings on self-play, LR 0.01 / momentum 0.90 fixed (`20261008-zlra-selfplay-lr/`) | until stopped |


## Next (in order)

1. ~~**Mixed-tail timing, 128-channel models** — fp32 tail vs mixed, A-B-B-A, 600
   steps each, on the v5-style SE net and `v4_5block_7x7`. Needs the GPU to
   itself: runs when every replay run has ended (moved behind C and D; see decisions).~~
   Done 2026-10-05 01:32, with R7 33k added (owner-requested); E-0011.
2. ~~**Full test suite** for the layer-health tracking batch (in a gap).~~ Done
   2026-10-02 03:35 alongside the Lichess shutdown fix: 1701 passed, 0 failed, 1 skipped.
3. ~~**Label smoothing C:** policy ε 0.1 → 0.03 (`plans-active/POLICY_LABEL_SMOOTHING_EXPERIMENTS.md`).
   Launches automatically when leaky-FC1 ends.~~ Launched 2026-10-02 06:14.
4. ~~**Zero-init ReZero** (`20261002-rezero-zero-init/`) — first in the queue; launches
   when a slot frees once its build is frozen.~~ Done (R9, E-0001).
5. ~~**Label smoothing D:** value ε 0.013 → 0. Launches when a slot frees (queue order).~~ Done (R12, E-0003).
6. ~~**Label smoothing C seed 2** — from the scale+bias seed-2 fresh net; launches when
   a slot frees (queue order).~~ Ran; stopped by the owner at 31,906 (R11, E-0002).
7. **Label smoothing B:** per-move policy smoothing (code in place; its complement
   target is being fixed first — `plans-completed/REVIEW_2026-10-02_FIXES_PLAN.md` B2).
8. ~~Second no-ReZero seed if the ReZero result is close.~~ Launched 2026-10-02 03:55.

## Finished

- 2026-10-07 16:47 → 2026-10-08 09:41 — ZlrA GUI self-play on the 0.001–1.0 LR cycle (`20261008-zlra-selfplay-lr/`, E-0025):
  stopped by the owner at 51,135 steps. Wide pElo flat at about 540 from 20k (every earlier fresh self-play run 630–680 by
  40–50k) while pLoss / vLoss sat far below every other run (0.72 / 0.16 at 50k); BN running-variance ratio ratcheted to
  1,655. Cause open; follow-ups R-replay and R-fixedlr launched.
- 2026-10-07 06:07 → 07:28 — Relative gradient cap validation V-1 and V-3 (`20261005-lr-schedule-ab/`, E-0024; plan
  `RELATIVE_GRADIENT_CAP_PLAN.md` Part V). V-1: exact log-only rerun of B-silu from 18k to 21k, byte-identical to B-silu; healthy
  steps ≤ 1.39× their trailing median, the breaking steps 18–48× (19,785, 19,795 and the 11-step burst 20,599–20,609) → k = 3.
  V-3: fresh start on B's recipe with the cap on (k = 3), 3,000 steps, no relative clip, byte-identical to B. Both pass; P5 (default
  to clip) waits on the owner's go (the edit was refused by the permission check). V-2 (k = 3 clip from 18k to 23k) not run.
- 2026-10-07 03:50 → 05:47 — LR arms B-silu-clip2 and B-silu-clip5 (`20261005-lr-schedule-ab/`, E-0023): exact resumes of B-silu
  from 18k with `grad_clip_max_norm` 2.0 / 5.0, to 23k (both clean finishes, rc 0). Both left B-silu's path at 19,800 like clip1
  (so B-silu had an unlogged step with a pre-clip norm above 5) and neither blew up: 21k pElo 1373.2 / 1353.6 vs B-silu 457.4,
  0 parked channels at 21k–23k, 23k 1560.6 / 1567.3 vs ReLU B 1573.4. Largest logged gNorm 0.453 / 0.475.
- 2026-10-06 17:00 → 2026-10-07 02:45 — LR arm B-silu-clip1 (`20261005-lr-schedule-ab/`, E-0020): exact resume of B-silu
  from 18k with `grad_clip_max_norm` 1.0, to 40k (clean finish). Never blew up (the cap-15 control reproduced the 20,600 blowup
  bit for bit); low-LR probes −5.3 ± 6.7 pElo vs ReLU B, best 1639.2 at 38k (B 1632.0); 0 parked / 0 mostly-off BN channels at
  40k (B: value.bn 6 parked). Largest logged gNorm 0.462.
- 2026-10-05 23:44 → 2026-10-07 00:12 — LR arm B-leakyall (`20261005-lr-schedule-ab/`, E-0022): B with leaky ReLU at every
  activation, to 40k (clean finish). Value head kept every channel (`value.bn` 0 of 16 parked vs B's 6; value FC1 0 of 128 at
  zero velocity vs 27); tower `blocks.2.bn1` still drifted (4 mostly off, worst β/|γ| −2.70 vs B −2.53); policy probes −12.5 ± 13.0
  pElo vs B at low LR; best 1629.4 at 38k (B 1632.0, B-leaky 1641.7).
- 2026-10-05 23:44 → 2026-10-06 23:10 — LR arm B-silu (`20261005-lr-schedule-ab/`, E-0021): SiLU tower + leaky heads on B's
  LR cycle at cap 15. Tracked B to 19k (mean −3.7 pElo), blew up at 20,600 (E-0020); 20 of 128 policy pre-BN channels
  stayed parked from 21k to the end; pElo 1250.0 at 36k vs B 1620.7. Stopped by the owner at 36,066 (clean abort save).
- 2026-10-06 18:15 → 21:57 — LR arm B-silu-ctl15 (`20261005-lr-schedule-ab/`, E-0020): exact resume of B-silu from 18k with its own cap 15;
  reproduced B-silu bit for bit through the step-20,600 blowup (equal probes to the last digit at 19k–22k), stopped by the owner after
  the 22k probe matched; clean abort save at 22,030. B-silu-clip1 (cap 1.0, same resume) did not blow up (0 dead channels at 22k).
- 2026-10-05 20:41 → 2026-10-06 16:24 — LR arm B-leaky (`20261005-lr-schedule-ab/`, E-0019): B with leaky ReLU in the value head
  (conv + FC1), to 40k. Value head kept every channel (`value.bn` 0 of 16 dead vs B's 6; value FC1 0 of 128 units at zero
  velocity vs 27); value loss unchanged (0.8032 vs 0.8031 over 37k–40k); policy probes −1.7 ± 9.3 pElo vs B at low LR.
  Best 1641.7 at 38k (B 1632.0 at 38k).
- 2026-10-05 23:13 → 23:43 — LR arm C-leaky (`20261005-lr-schedule-ab/`): C with leaky ReLU everywhere; trained like C to LR 2,
  then ran away from LR 3 (loss to 4.5M, finite weights, 543 dead channels); stopped by the owner at step 1,175.
- 2026-10-05 01:32 → 18:51 — LR schedule A/B/C on a fresh basic24 R7-shape net (`20261005-lr-schedule-ab/`, E-0017): A constant 0.01,
  B cycle 1.0 ↔ 0.001 (10k period), to 36k then (owner) continued by exact resume to 40k; C (10 ↔ 0.01) diverged at ~300.
  B best 1632.0 at 38k, A best 1453.8 at 36k; B ahead at 39 of 40 probes.
- 2026-10-04 23:34 → 2026-10-05 01:32 — fp32 vs mixed policy-tail timing (A-B-B-A on R7 33k, the SE scale+bias net, a fresh
  `v4_5block_7x7`) and nt8y's step time (`20261004-policy-tail-precision/`, E-0011).
- Earlier runs (R1–R16, numbered in `rchart.py`) have all ended; summaries of those since 2026-10-02 are in `summaries/`
  (open `summaries/index.html`).

- **Leaky ReLU in SE FC1** (`20261001-se-fc1-leaky/`), 2026-10-01 15:18 → 2026-10-02
  06:13, 33k steps. Final 1492.8 pElo / 2.2381 NLL vs ReLU twin 1463.1 / 2.2614;
  conclusions in its README.

- **SE / ReZero conclusions so far** (2026-10-02): `20260929-se-style-ab/FOLLOW-UPS.md` —
  neither helps at this scale; use no SE and no ReZero as the baseline for this family.

## Deferred (owner: interested, not spending the compute now)

- **Do the SE / leaky-FC1 / ReZero findings hold at ~130k steps?** Every arm so far
  is ≤ 33k steps (one LR-cycle region), where no-SE leads leaky-SE by ~15 pElo and
  leaky leads its ReLU twin by ~15 (inside seed noise). Long-run behavior is
  untested; owner raised it 2026-10-02 and deferred it.

## Decisions

- **2026-10-02 01:11 — start no-SE/no-ReZero now, in parallel with leaky-FC1.**
  Two replay runs share the GPU at roughly unchanged total throughput (the SE
  experiment ran three at once); comparisons are on step. Cost: neither run's step
  times are clean speed numbers from here (leaky-FC1 already has 14k clean steps at
  818 ms median). Timing benchmarks wait for an idle GPU.
- **2026-10-02 01:11 — fp32 tail for experiment arms that compare against the SE
  experiment.** The baselines trained with `fp32_from_pre_bn`; the new default
  (`mixed_final_projection`) would be a second variable.
- **2026-10-02 01:28 — full test suite run during both training runs** (layer-health batch).
  The GPU is shared for ~20–30 min; both runs' step times in that window are not
  speed data. Training math unaffected. Accepted to keep the batch moving rather
  than leave it untested until the runs finish.
- **2026-10-02 03:55 — second no-ReZero seed now, as a third concurrent run.** Seed 1
  was level with both ReZero seeds at 6k (inside seed noise), and the owner asked for
  another experiment rather than an idle GPU. It decides whether ReZero is worth
  keeping on LayerNorm-out nets, which the zero-init / decoupled-cap idea depends on.
  The original chain waited only for the first two runs and would have timed the
  benchmark beside seed 2, so it was replaced: C and D now launch as runs end (three
  at most), and the timing benchmark waits for every replay run to end.
- **2026-10-02 06:15 — the chain records the wrong log for launched runs.** It takes
  the newest `dcm_log_*` by name, but each probe also opens an (empty) log, so C's
  chain line names `dcm_log_20261002-061436.txt` (empty); the real log is
  `dcm_log_20261002-061425.txt`. Launch records are filled in from the log whose
  `[REPLAY] start-model` line matches, not from the chain line. Left `chain2.sh`
  running rather than edit a script that is executing.
- **2026-10-02 14:39 — chain2 replaced by chain3** to queue label smoothing C seed 2
  (owner-approved) after no-ReZero seed 2; D still launches after no-ReZero seed 1,
  the timing benchmark still waits for every run. chain3 records each run's log by its
  `[REPLAY] start-model` line instead of "newest log".
- **2026-10-02 14:40 — full test suite during three training runs** for the
  `--probe-positions-out` change (shared NLL definition + per-position records).
  Same trade as at 01:28: the runs' step times in that window are not speed data;
  training math is unaffected.
- **2026-10-02 17:58 — chain3 replaced by chain4** so zero-init ReZero (owner: "sooner
  rather than later") takes the first free slot, ahead of D and C seed 2, as soon as
  its format-v6 build is frozen; if no build is ready when a slot frees, the slot goes
  to D instead of waiting.
- **2026-10-04 23:31 — chain4 stopped; timing benchmark restarted clean.** chain4 (from
  2026-10-02) was still alive. When fatconv's continuation ended at 22:59 it started its
  queued timing benchmark (build 2275, `bench3/`), and the owner-requested benchmark started
  at 23:04 on build 2320, so the two shared the GPU for ~30 minutes and every timing in that
  window is unusable. Both were stopped (chain4's remaining work was that benchmark, which the
  new one covers); the contended results are set aside, and the restart refuses to start any
  run while another DCM job is on the GPU.
- **2026-10-04 23:56 — value-head vs Stockfish analysis during the benchmark** (owner: "whatever
  you can do in less than 10 minutes"). CPU only, 3 minutes; it overlapped the timed window of
  `r7-mixedB`, whose wall time (832.3 ms/step vs 801.4 for `r7-mixedA`) is excluded; its
  sampled step median is unaffected. Not re-run (owner: no benchmark re-runs for it).
- **2026-10-05 00:57 — LR A/B queued (owner: "go with A and B after the brief timing run";
  "if B doesn't blow up badly within the first 10k, run it to 36k").** Fresh basic24 R7-shape
  net minted with `--init-seed 20261005`; both arms use `--seed 20261005`, so they see the same
  batches; R7/R8's parameters otherwise, with the sampler set to the nearest uniform equivalent
  (build 2320 applies batch-composition constraints that 2275 ignored).
- **2026-10-05 09:04 — arm C added (owner: "identical to B, but with 10 and 0.01").** `lr_cycle_max` was declared
  with an upper bound of 1.0, so the bound was raised to 10 (owner chose 10 over 100, and launching now over
  waiting for A/B to finish). Build 2323, frozen as `DCM-2323-58e9f952-lrmax10`. A and B slow to ~2 s/step while
  three trainers share the GPU.
- **2026-10-05 09:22 — arm C stopped at step 513** under the owner's rule for B (stop if it blows up badly):
  the policy diverged as the LR passed ~3 (illegal-move mass 0.997 at step 300, then ~0.945 with gNorm
  0.02–0.06; 339 of 1,040 channels dead at the abort save). Details in `20261005-lr-schedule-ab/README.md`.
- **2026-10-05 12:18 — arm C continued (owner), 15:34 stopped (owner).** Resumed exactly from step 513 to see whether
  it recovers or goes non-finite; it did neither (frozen at pElo ~585 through trainer step 6,116, no NaN/Inf). Details
  in `20261005-lr-schedule-ab/README.md`.
- **2026-10-05 17:14 — A and B continued to 40k (owner: "allow them to go to 40k, then stop there").** Each was resumed
  exactly from its 36,000-step final save with a 4,000-step segment limit (same build, parameters and seed); both ended
  18:51 at trainer step 40,000. Details in `20261005-lr-schedule-ab/README.md`.
