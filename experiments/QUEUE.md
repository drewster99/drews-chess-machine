# Experiment queue and decision log

Standing instruction (owner, 2026-10-02): keep the GPU busy; chain runs back to
back; make hard calls autonomously, log them here, report them at the next
conversation.

## Running

| started | experiment | ends (est.) |
|---|---|---|
| 2026-10-05 01:32 | LR schedule A/B on a fresh basic24 R7-shape net (`20261005-lr-schedule-ab/`, E-0017): A constant LR 0.01, B cycle 1.0 ↔ 0.001 (10k period), both to 36k, same `--seed` | ~2026-10-05 16:35 |

Queue script: `chain4.sh` was stopped 2026-10-04 23:31 (see decisions). Both A/B arms were launched at 01:32 by a
local script (`lrab_chain.sh`, not in the repo), which keeps running only to stop arm B on a non-finite loss.

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
