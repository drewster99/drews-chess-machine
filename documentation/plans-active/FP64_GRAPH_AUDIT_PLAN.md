# fp64 audit of the training and inference graphs (2026-11-14)

**Status: forward and backward graph audit implemented and run (owner's go 2026-11-14, on macOS 27.2 before the
update); results in `experiments/20261114-fp64-graph-audit/README.md`: no defect beyond the known fp32 conv weight
gradient. Not done: the trainer's loss subgraph (`buildTrainingOps`' raw ops) and the optimizer update; a re-run on
the updated OS.**

## Why

MPSGraph returned a wrong fp32 convolution weight gradient past 2³¹ elements and nothing in the app noticed for
28,814 steps (`FP32_CONV_WEIGHT_GRADIENT_WORKAROUND_PLAN.md`). The defect depended on dtype, tensor size and the
graph around the op (a second consumer), so small isolated tests missed it. The question (owner, 2026-11-14): are
there other ops, shapes or dtypes where MPSGraph is silently wrong in the graphs we actually train and play with?

## Approach: check every op of the real graph against fp64, at the real sizes

Two levels, because size-dependent bugs need real sizes and a CPU can't run a whole network in fp64 at batch 4096:

1. **Per-op local checks at full size (the main tool).** Build the app's own graph (`ChessNetwork`, training and
   inference mode) with taps on every op's inputs, output and output gradient. For each op, recompute in fp64 on the
   CPU, from the graph's own tensors:
   - the forward output from its inputs;
   - in training graphs, the input gradient and the weight gradient from its output gradient.

   Compare with what MPSGraph returned: relative error ‖mps − fp64‖ / ‖fp64‖ per tensor, and the worst element.
   This is how the conv defect was found (`analysis-fp32/repro/block_conv1_weight_gradient.swift`): one op at a
   time, exact reference, real batch. Each op needs a fp64 reference: convolution (Accelerate / tiled loops),
   BatchNorm (training: batch statistics; inference: running statistics), LayerNorm, activations (ReLU, leaky,
   SiLU, GELU, tanh), SE (pool, FC, gates), ReZero, residual merges, 1×1 skip projections, dense layers, softmax,
   the losses (policy CE with label smoothing and advantage weighting, illegal-mass penalty, WDL CE), dtype casts.
2. **End-to-end check at small batch.** An fp64 CPU forward of the whole network and loss (from the recorded
   `InferenceGraph`, extended to the training forward), and directional finite differences: for random directions
   d per tensor, (L(w + εd) − L(w − εd)) / 2ε against ⟨g_mps, d⟩. This catches wiring errors the local checks
   can't (a gradient routed to the wrong tensor, a missing term), at batch 32–256.

## Matrix

- **Architectures:** every built-in preset, every user preset in `Presets/`, and the shapes of every model trained
  since 2026-10 (from `--describe-model` over `Models/`: elitecyc 3 × 9×9, the SE / SiLU / leaky variants, 15×15
  stems, heterogeneous towers, feature skip).
- **Dtypes:** bf16 and fp16 under both policy tails, fp32.
- **Batch sizes:** 1, 256 (self-play / arena), 4096 (training), and each plan's largest correct batch + 1 (to map the
  edge), within GPU memory.
- **Graph contexts:** training graph, inference graph, the compiled executables the trainer and evaluators use
  (`.level1`), not only `graph.run`.

## Output

- `experiments/<date>-fp64-graph-audit/`: per (architecture, dtype, batch, op) relative errors, a table of every
  result above dtype rounding level (thresholds per dtype, set from the clean majority), standalone repros for
  anything wrong.
- A gated test class (`DCM_RUN_SLOW_TESTS`) that re-runs the per-op checks on a small set of shapes, as a bug map
  like `MPSGraphFP32ConvWeightGradientBugMapTests`.

## Cost and risks

- CPU fp64 convolutions at batch 4096 are slow (a 9×9 / 128 layer is ~3.5·10¹² multiply-adds): minutes per op;
  hours for the full matrix. Runs beside training only on the CPU side; GPU use is short graph runs.
- Taps change the graph (extra consumers), and the conv defect depended on consumers. So each op is checked both
  with taps and, for the gradients, through the per-op reconstruction used in
  `ConvolutionWeightGradientWorkaroundGraphTests` (taps only where the production graph already has a consumer).
- Tolerances: bf16 / fp16 errors are expected at rounding level; a finding is an error well above the clean
  majority for the same op and dtype, or one that grows with size.

## Validation

- The audit must find the known defect: fp32, 9×9 / 128, batch 4096, conv1 weight gradient on build 2669 (unsplit).
  If it doesn't, the audit is broken.
- Every other result either matches fp64 at rounding level or comes with a standalone repro.
