# Experiment queue and decision log

Standing instruction (owner, 2026-10-02): keep the GPU busy; chain runs back to
back; make hard calls autonomously, log them here, report them at the next
conversation.

## Running

| started | experiment | log | ends (est.) |
|---|---|---|---|
| 2026-10-01 15:18 | leaky ReLU in SE FC1 only (`20261001-se-fc1-leaky/`), to 33k | `dcm_log_20261001-151822.txt` | ~2026-10-02 06:00 (shared GPU from 01:11) |
| 2026-10-02 01:11 | no SE, no ReZero (`20261002-noSE-noReZero/`), to 33k | `dcm_log_20261002-011124.txt` | ~2026-10-02 16:00 |

## Next (in order)

1. **Mixed-tail timing, 128-channel models** — fp32 tail vs mixed, A-B-B-A, 600
   steps each, on the v5-style SE net and `v4_5block_7x7`. Needs the GPU to
   itself: runs in the first gap when both training runs are done.
2. **Full test suite** for the layer-health tracking batch (in a gap).
3. **Label smoothing C:** policy ε 0.1 → 0.03 (`plans-active/POLICY_LABEL_SMOOTHING_EXPERIMENTS.md`).
4. **Label smoothing D:** value ε 0.013 → 0.
5. **Label smoothing B:** per-move policy smoothing (needs code).
6. Second no-ReZero seed if the ReZero result is close.

## Decisions

- **2026-10-02 01:11 — start no-SE/no-ReZero now, in parallel with leaky-FC1.**
  Two replay runs share the GPU at roughly unchanged total throughput (the SE
  experiment ran three at once); comparisons are on step. Cost: neither run's step
  times are clean speed numbers from here (leaky-FC1 already has 14k clean steps at
  818 ms median). Timing benchmarks wait for an idle GPU.
- **2026-10-02 01:11 — fp32 tail for experiment arms that compare against the SE
  experiment.** The baselines trained with `fp32_from_pre_bn`; the new default
  (`mixed_final_projection`) would be a second variable.
