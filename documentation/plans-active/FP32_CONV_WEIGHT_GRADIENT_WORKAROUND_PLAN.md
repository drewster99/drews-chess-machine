# fp32 convolution weight-gradient workaround (owner request 2026-11-14)

## Problem

MPSGraph (measured on macOS 27.2 26B123, M5 Max) computes a wrong **fp32** convolution weight gradient when the
convolution's output has another consumer (a training-mode BatchNorm reads it three times) and
batch × 64 × C_in × k² exceeds 2³¹ − 1 for 128-channel inputs. The forward output and the input gradient stay
exact; bf16 and fp16 are correct at every size measured. R-elitecyc-fp32 trained on it for 28,814 steps (every
block's 9×9 `conv1`, 2.72·10⁹ elements at batch 4096) and lost 330–380 pElo. Evidence and standalone repro:
`experiments/20261008-zlra-selfplay-lr/analysis-fp32/`.

## Workaround (owner-approved)

Split the convolution over its **input channels**: Y = Σ_g conv(X[:, g], W[:, g]). Mathematically identical
(the weight gradient of each slice is that slice of the true gradient), batch-independent (channel counts are
static at graph build; the batch is not), exact in measurement (relative error 7·10⁻⁶ at batch 4096 inside the
real block structure, against 0.76 unsplit; exact at batches 4096–32768).

- **Where:** training-mode graphs only (inference graphs are untouched: their forward is exact, and Core ML export
  records only inference graphs), on every convolution that runs in fp32 and whose
  `TrainingBatchSize` upper bound × 64 × C_in × k² exceeds 2³¹ − 1 (C_in·k² > 512 at the 65,536 bound). The
  declared bound is the one source of the batch ceiling, so any batch the app accepts is covered.
- **Groups:** the smallest divisor of C_in that is ≥ 2 (two halves for every even width). Measured exact with two
  halves at 9×9 / 128 channels up to batch 32,768. An affected convolution whose C_in has no divisor ≥ 2 (C_in = 1)
  cannot be split: building that training graph is an error.
- **One rule:** `ConvolutionWeightGradientWorkaround` (pure) decides, from the architecture, which convolutions
  split and into how many groups. The graph builder takes its plan; the builder checks every convolution it builds
  against it (a convolution the rule covers but the plan does not name is an error, never a silent unsplit
  build).
- **Warnings:** `[ARCH] fp32 conv weight-gradient workaround: …` when a training graph is built with splits;
  the same notice from `--new-model`, `--derive-model` (a derive to fp32, or any derive whose output is fp32 with
  affected convolutions) and Build New Model (shown with the architecture when fp32 is chosen).

## Tests

1. **MPSGraph bug map** (`MPSGraphFP32ConvWeightGradientTests`, gated by `DCM_RUN_SLOW_TESTS`: it allocates
   ~20 GiB): the unsplit fp32 conv at batch 4096 / 9×9 / 128 with its output read must still be **wrong** — the
   test fails when MPSGraph changes, so we notice a fix (or a change in where it breaks); the channel-split and
   bf16 forms must be exact.
2. **Plan rule** (`ConvolutionWeightGradientWorkaroundTests`): which convolutions of a given architecture split,
   their groups, dtype gating (bf16/fp16 and the bf16 policy tail), the 512 boundary, C_in without a divisor.
3. **The app's own training graph** (`ChessNetwork` fp32 training mode, one 9×9 / 128 block, batch 4096): the
   weight gradient of the BatchNorm-fed `conv1`, taken from the network's graph, is orthogonal to its weights per
   output channel (BatchNorm scale invariance), as it is in bf16.

## Validation

- Build with no new warnings; the three test classes pass; full suite passes.
- A 1-step fp32 resume of the R-elitecyc 384,964 state on the new build: every tensor's gradient matches the bf16
  1-step gradient to cos ≥ 0.99 (conv1 was 0.42–0.57 before).
- Restart R-elitecyc-fp32 from 384,964 under a new stem; compare with R-elitecyc at 390k / 400k.

## Status (2026-11-14)

- Implemented (build 2672+). Build: no errors; the only warnings are pre-existing type-check-time warnings in files
  this change does not touch.
- 1-step validation done: gNorm 0.745 in fp32 and bf16 (build 2669 fp32: 0.877); every convolution weight's gradient
  matches bf16 at cos ≥ 0.998.
- **The three test classes have not been compiled or run** (the test suite is run only on the owner's word).
- Audit done (`experiments/20261008-zlra-selfplay-lr/analysis-fp32/README.md`): only R-elitecyc-fp32 was affected.

## Also

- Audit every earlier fp32 training run against the rule.
- Apple Feedback report and blog-post draft from the repro.
