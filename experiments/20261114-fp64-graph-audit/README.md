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

## Matrix results (2026-11-14 20:50 → 11-15 01:19, frozen build `DCM-2686-85443e2f-audit`)

58 cases (`matrix/*.json`, run by `run_matrix.sh`): A = the real R-elitecyc models (bf16 and fp32fix at 400k, training
4096, inference 256 and 1, the bf16 model rebuilt fp16 and with an fp32 policy tail); B = fp32 / fp16 / fp32-tail
overrides on presets; C = every preset in its own precision, training 4096; D = every preset, inference 256.
Table: `python3 matrix_table.py matrix/*.jsonl` (worst `scale_rel_err` of fp32-only checks vs checks with a 16-bit
operand).

**No MPSGraph defect beyond the known fp32 conv weight-gradient one.**

- **fp32 (14 cases incl. the fp32fix model with its split convolutions at 4096):** worst forward 2.0e-7, worst gradient
  1.4e-7 — rounding. The split fp32fix graph's conv weight gradients are exact (1.1e-7), confirming the workaround.
- **bf16 (40 cases):** worst forward 4.5e-3, worst gradient 4.0e-3: bf16 rounding (8 significant bits, 3.9e-3 per
  element), far under the 3e-2 threshold. The fp32 policy tail changes nothing measurable.
- **Two fp16 flags, both fp16's range, not MPSGraph:**
  - `A06` (9×9 bf16 model rebuilt fp16, training 4096): the backward pass overflows to inf from `block2_bn1`'s
    variance gradient back. Its exact fp64 value reaches 5.56e4 (fp16 max 65,504), because the audit loss sums over
    the 4,096 positions where training averages them (gradients 4,096× larger). An audit artifact.
  - `B-v4_5block_7x7-fp16`: one forward multiply at 3.2e-2 on values ~1e-6, below fp16's smallest normal (6.1e-5);
    the error equals the subnormal rounding floor.
- **`032-C-test_22_skinny_48` training (22 blocks × 48 channels, bf16, 883 ops / 671 gradient tensors) is not
  audited.** Both attempts were SIGKILLed (exit 137) before the first chunk wrote a record. The first was jetsam
  (`JetsamEvent-2026-11-15-005019.ips`, reason low swap): the audit process held 17,114,014 pages × 16 KiB = 261 GiB,
  mostly compressed, while `ps` RSS read ~16 GB (`matrix/032-rerun.mem.log`, the rerun's samples: RSS peak 38.6 GB in
  the first 3 min). The rerun died at 18 min the same way without a jetsam report. This is the audit's own memory on a
  graph this deep (the in-process footprint check runs only after a chunk completes; the outside guard reads free
  memory, which compression keeps up), not a finding about the model. Its inference case (`D`) is clean. The preset
  trains nowhere.

## Reproduce

```sh
BIN=<Release build>/DrewsChessMachine.app/Contents/MacOS/DrewsChessMachine
$BIN --graph-audit-selftest
DCM_AUDIT_BIN=$BIN ./run_guarded.sh validate.json validate.jsonl
DCM_AUDIT_BIN=$BIN ./run_guarded.sh known_defect.json known_defect.jsonl
python3 analyze_audit.py known_defect.jsonl
```
