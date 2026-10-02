# Experiment queue and decision log

Standing instruction (owner, 2026-10-02): keep the GPU busy; chain runs back to
back; make hard calls autonomously, log them here, report them at the next
conversation.

## Running

| started | experiment | log | ends (est.) |
|---|---|---|---|
| 2026-10-02 06:14 | label smoothing C, policy ε 0.03 (`20261002-label-smoothing-C/`), to 33k | `dcm_log_20261002-061425.txt` | after the no-ReZero runs |
| 2026-10-02 01:11 | no SE, no ReZero (`20261002-noSE-noReZero/`), to 33k | `dcm_log_20261002-011124.txt` | ~2026-10-02 16:00 |
| 2026-10-02 03:55 | no SE, no ReZero **seed 2** (same folder), to 33k | `dcm_log_20261002-035513.txt` | later than seed 1 (three runs share the GPU) |

Chain (`chain2.sh`, scratchpad): label smoothing C launches when leaky-FC1 ends, D
when no-ReZero seed 1 ends (each with `experiments/probe_loop.sh`); the mixed-tail
timing runs when every replay run has ended.

## Next (in order)

1. **Mixed-tail timing, 128-channel models** — fp32 tail vs mixed, A-B-B-A, 600
   steps each, on the v5-style SE net and `v4_5block_7x7`. Needs the GPU to
   itself: runs when every replay run has ended (moved behind C and D; see decisions).
2. ~~**Full test suite** for the layer-health tracking batch (in a gap).~~ Done
   2026-10-02 03:35 alongside the Lichess shutdown fix: 1701 passed, 0 failed, 1 skipped.
3. ~~**Label smoothing C:** policy ε 0.1 → 0.03 (`plans-active/POLICY_LABEL_SMOOTHING_EXPERIMENTS.md`).
   Launches automatically when leaky-FC1 ends.~~ Launched 2026-10-02 06:14.
4. **Label smoothing D:** value ε 0.013 → 0. Launches automatically when no-ReZero
   seed 1 ends.
5. **Label smoothing C seed 2** — from the scale+bias seed-2 fresh net; launches
   when no-ReZero seed 2 ends (`chain3.sh`).
6. **Label smoothing B:** per-move policy smoothing (needs code).
7. ~~Second no-ReZero seed if the ReZero result is close.~~ Launched 2026-10-02 03:55.

## Finished

- **Leaky ReLU in SE FC1** (`20261001-se-fc1-leaky/`), 2026-10-01 15:18 → 2026-10-02
  06:13, 33k steps. Final 1492.8 pElo / 2.2381 NLL vs ReLU twin 1463.1 / 2.2614;
  conclusions in its README.

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

