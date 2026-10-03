# 2026-10-02 — Label smoothing arm D: value label smoothing ε 0.013 → 0 (`value_label_smoothing_epsilon`)

**Status:** running since 2026-10-03 00:28 CDT, sharing the GPU with label smoothing C and
zero-init ReZero.

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

- **Launched** 2026-10-03 00:28:16 CDT (session log `dcm_log_20261003-002816.txt`), by the
  experiment queue in the slot freed by no-ReZero seed 2.
- **Build** 2275 (`de0f22b` app code), frozen as `FrozenBuilds/DCM-2275-de0f22b.app` — the
  baseline's build, so this run's fed stream and sampling match the baseline's (it predates the
  corpus-adjudication fix and the replay sampling-constraint change).
- **Run model ID** `20261003-21-yEjN` (parent `20260929-12-JZOe`, the shared starting net).
- **Command**

```
"$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2275-de0f22b.app/Contents/MacOS/DrewsChessMachine" \
  --replay-corpus 20260624-192615-w3aA5b \
  --start-model "$HOME/Library/Application Support/DrewsChessMachine/Models/20260929-test_SE_scale+bias-fresh.safetensors" \
  --out-model "$HOME/Library/Application Support/DrewsChessMachine/Models/20261002-label-smoothing-D-replay-latest.safetensors" \
  --parameters experiments/20261002-label-smoothing-D/parameters.json \
  --epochs 12 --training-step-limit 33000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn
```

- **Probes** `experiments/probe_loop.sh 20261002-label-smoothing-D experiments/20261002-label-smoothing-D/probes.jsonl`
  with the same build.
