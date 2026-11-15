# fp64 graph audit (2026-11-14)

Every forward op of the app's own MPSGraph network graphs (training and inference mode, at real batch sizes)
recomputed in float64 on the CPU from the operand values MPSGraph held, and every tensor's gradient recomputed from
its consumers' output gradients. Plan: `documentation/plans-active/FP64_GRAPH_AUDIT_PLAN.md`. Code:
`Network/GraphAudit.swift`, `Network/GraphAuditReference.swift`, `App/GraphAuditCLI.swift`.

## Method

- `--graph-audit <config> --graph-audit-out <new file>` builds the network with an audit recording
  (`InferenceGraphRecorder(purpose: .audit)`), feeds the bundled puzzle positions cycled to the batch, and for training
  graphs adds a fixed random linear loss on the policy logits, value logits and value, asking MPSGraph for the gradient
  of every recorded tensor the loss depends on.
- Ops run in chunks; each chunk rebuilds the graph (MPSGraph caches an executable with its buffers per target set: one
  graph across chunks grew the process ~10 GB per chunk at batch 5376) and checks the rebuild records the same ops.
- Judged error: `scale_rel_err` = ‖MPSGraph − fp64‖ / ‖magnitude of the summed terms‖ (a gradient that cancels to ~0,
  e.g. a γ/β followed by a BatchNorm, is judged on what was added up). Flag thresholds (`analyze_audit.py`): fp32
  1e-4, fp16 1e-2, bf16 3e-2.
- `--graph-audit-selftest`: every fp64 backward reference against central finite differences of its own forward
  (worst 4.4e-9).

## Memory (the first full-size run hung the Mac: watchdog panic 2026-11-14 17:30)

Chunks are sized from real tensor shapes (1 GiB of float64 by default); the process stops a case above
`memory_limit_gib` (default 12); `run_guarded.sh` kills the audit if system free memory drops below 20%. A fp32 case past
2³¹ elements needs MPS's own 8-bytes-per-element buffer on top (17.3 GB at 7×7 / batch 5376). The `-limit*`, `-pool`,
`-rebuild24` files are the stopped attempts that found these limits.

## Validation

| case | result |
|---|---|
| `validate.json`: 9×9 model, fp32 and bf16 training, bf16 inference, batch 64 | 0 flags; fp32 ≤ 2e-7, bf16 ≤ 5e-3 (rounding) |
| `known_defect.json`: `v4_5block_7x7` fp32 training, batch 5376 (past 7×7's 5349 limit) | every block's conv1 weight gradient rel 0.095–0.099 (repro: 0.094), block 4's conv2 0.103; other conv weights ≤ 1.5e-5 |

The audit finds the known defect, so its silence elsewhere means something.

## Reproduce

```sh
BIN=<Release build>/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine
$BIN --graph-audit-selftest
DCM_AUDIT_BIN=$BIN ./run_guarded.sh validate.json validate.jsonl
DCM_AUDIT_BIN=$BIN ./run_guarded.sh known_defect.json known_defect.jsonl
python3 analyze_audit.py known_defect.jsonl
```
