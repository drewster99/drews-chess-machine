# 2026-10-02 — Label smoothing arm D: value label smoothing ε 0.013 → 0 (`value_label_smoothing_epsilon`)

**Status:** queued — the local queue script launches it when a training slot frees,
after zero-init ReZero (see `experiments/QUEUE.md`); it shares the GPU with whichever
runs are still training then.

## Question

Is value-target smoothing still needed now that the shared value offset can't drift? Proposal and background:
`documentation/plans-active/POLICY_LABEL_SMOOTHING_EXPERIMENTS.md`.

## Design

- **Only variable:** value label smoothing ε 0.013 → 0 (`value_label_smoothing_epsilon`). `parameters.json` here is the SE experiment's pinned
  file with that one key changed.
- **Starting net:** the SE experiment's scale+bias seed-1 fresh net
  (`20260929-test_SE_scale+bias-fresh.safetensors`, ModelID `20260929-12-JZOe`) —
  bit-identical to the baseline's start.
- **Baseline (not re-run):** ReLU scale+bias seed 1 (`se_sb`, 33,014 steps).
- **Everything else identical:** corpus `20260624-192615-w3aA5b`, 12 epochs, step
  limit 33,000, `--policy-tail-precision fp32_from_pre_bn` (the baseline's
  numerics), build 2275 (= `de0f22b`'s app code; stamped `f6fdd88`).
- **Measurements:** pElo / NLL every 1,000 steps (`--probe-set wide`); for C also
  NLL / top-1 by legal-move count bucket and policy entropy, `pLogitAbsMax`; for D
  value loss and W/D/L calibration.

## Launch record

(filled in at launch)
