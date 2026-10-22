# Core ML export

A model file can be exported to a Core ML model (`.mlpackage`) for inference: the same policy logits, value and W/D/L probabilities the MPSGraph network computes, runnable on the CPU, GPU or Neural Engine through Core ML. Training stays on MPSGraph.

## One definition of the network

The network is defined once, by `ChessNetwork`'s graph builders. Nothing else describes it:

- Every forward-pass op the builders emit goes through `ForwardGraphOps` (`Network/ForwardGraphOps.swift`), which builds it into the MPSGraph exactly as a direct call would. When an `InferenceGraphRecorder` is attached, it also records the op. Production networks attach none.
- The recording (`InferenceGraph`, `Network/InferenceGraph.swift`) is the inference forward pass as primitive ops in build order:
  - each op's operands, its arguments and the data type the MPSGraph computes it in;
  - the weights, by plan name, in the layout the MPSGraph variables hold.
- `scripts/dcm_coreml_export.py` translates each primitive to its Core ML MIL equivalent. It knows no blocks, heads or architecture options, so a new option reaches Core ML with no change there. An op kind it doesn't know is refused.
- A forward op built on the raw MPSGraph instead of through `ForwardGraphOps` fails the recording: the next recorded op reads a tensor no recorded op produced. A plan tensor the forward pass never reads fails it too. **Never build a forward-pass op with the MPSGraph directly.** Only training-only ops may use the raw graph: dropout's random draws, comparisons and `minimum`, BN running-stat EMA assigns, the dropout RNG state, the weight-load assigns and the analysis-tap readbacks. None of them is part of the inference pass, and a training-mode network refuses a recorder.

The architecture a file resolves to is also read only from the app: `--describe-model <file>` prints it as JSON after every format-version rule and legacy resolution the loader applies, with the expanded blocks and the weight plan. `scripts/dcm_arch.py` is a thin, file-based wrapper over it (`describe`, `architecture`, `site_activations`, `rezero_blocks`, the scripts' ReLU guards); no Python code resolves an architecture itself.

## Commands

The app (headless, then exit):

| Command | Does |
|---|---|
| `--describe-model <file> [<file> ...]` | Prints one JSON line per file: its `ModelDescription` (header-only: the architecture the loader resolves, the lineage record, the step reading, the test-set results; every header rule a load applies, through `SafetensorsModelIO.inspect`), or the reason a load would refuse it. Exit 1 when any file was refused. |
| `--export-inference-graph <file> --out <new folder>` | Writes `graph.json` (the recording, format 2), `weights.bin` (little-endian float32, plan order, graph layout, each value rounded to its variable's type — what `loadWeights` gives the network, value head recentered) and `model.json` (the description). The folder is staged and moved into place whole. |
| `--coreml-parity <file> --coreml <model.mlpackage> [--compute-units cpu\|cpu-gpu\|cpu-ne\|all] [--tolerance <x>]` | Runs the bundled test-set positions (lichess-200 + lichess-wide, 4,635 positions) through both and prints the largest and mean differences of every output, and how often both rank the same legal move first, at every batch size the Core ML model accepts. Non-finite outputs are counted on each side independently. With `--tolerance`, exits 3 when a difference exceeds it or an output on either side is non-finite. |
| `--coreml-benchmark <config.json> [--json-out <new file>]` | Times Core ML exports against the MPSGraph network on the same weights and positions: each config arm (`mpsgraph`, or `coreml` with compute units) at each batch size and in-flight count, warm-up untimed, arm order rotated each round. Reports latency percentiles, positions/hour, load and first-call times, and Core ML's planned device per op (`MLComputePlan`). Core ML arms must pass parity first. Refuses a Debug build. `scripts/dcm_coreml_benchmark.py` prepares the exports and configs for a list of models. Rules: `documentation/plans-completed/COREML_BENCHMARK_PLAN.md`; results: `experiments/20261022-coreml-benchmark/`. |

The exporter (Python 3.13 with coremltools):

```sh
uv venv --python /opt/homebrew/bin/python3.13 ~/.venvs/dcm-coreml
uv pip install --python ~/.venvs/dcm-coreml/bin/python coremltools numpy

~/.venvs/dcm-coreml/bin/python scripts/dcm_coreml_export.py MODEL.safetensors --out MODEL.mlpackage \
    [--precision mirror|fp16|fp32] [--batch-sizes 1,8,32,64,128,256] [--parity [--compute-units cpu-ne] [--tolerance X]]
# or from a folder --export-inference-graph wrote:
~/.venvs/dcm-coreml/bin/python scripts/dcm_coreml_export.py --graph FOLDER --out MODEL.mlpackage
```

The app binary is `--app`, else `$DCM_BINARY`, else the newest build in DerivedData.

- **Input:** `board`, float32 `[batch, planes, 8, 8]`. The batch sizes are enumerated shapes, which the Neural Engine needs; 1 is always included.
- **Outputs:** `policy` `[batch, 4864]`, `value` `[batch, 1]` and `value_probs` `[batch, 3]` (`[batch, 1]` for a `scalar_tanh` head), all float32.

## Precision

| `--precision` | Ops in fp16 | Notes |
|---|---|---|
| `mirror` (default) | Exactly those the MPSGraph runs in bf16 / fp16 | The policy tail (`fp32_from_pre_bn` or `mixed_final_projection`), the value tail and an fp32 model stay fp32. |
| `fp16` | Every op | What the Neural Engine runs fastest. |
| `fp32` | None | CPU / GPU only. The parity reference. |

Every op is emitted with an explicit type (the recorded casts become real `cast` ops) and the converter is told to add no precision pass, so a fusion pass can't move a precision boundary. coremltools 9's `fuse_elementwise_to_batchnorm` pass is disabled: on fp16 constants it builds an invalid `batch_norm` (fp32 mean beside fp16 gamma/beta); the convolution fusion passes fold the BN anyway.

Core ML has no bf16, so a bf16 model's reduced-precision ops run in fp16. fp16's range is far smaller (largest finite value 65504, against bf16's ~3.4e38): an activation the bf16 network carries above it becomes inf in Core ML. `--coreml-parity` reports that as `non_finite_coreml_positions`. Even an fp16 model differs between the two engines: Core ML and MPSGraph accumulate fp16 in different orders, so mirror is the closest equivalent, not a bit-exact one.

## Measured (2026-10-21, Apple M5 Max)

**`zlra-elitecyc` step 314767** (bf16, 3 × 128 ch 9 × 9, `fp32_from_pre_bn`), 4,635 positions, against its bf16 MPSGraph; batch 1 and batch 64 give the same numbers:

| Export | Compute units | Max \|Δ\| policy logit | Mean \|Δ\| | Max \|Δ\| value | Same top move |
|---|---|---|---|---|---|
| fp32 | cpu | 0.54 | 0.020 | 0.032 | 4,619 (99.7%) |
| mirror | cpu | 1.71 | 0.046 | 0.067 | 4,558 (98.3%) |
| mirror | cpu-ne | 0.83 | 0.026 | 0.052 | 4,610 (99.5%) |
| fp16 | cpu-ne | 0.83 | 0.026 | 0.052 | 4,610 (99.5%) |

Every export's differences here include the MPSGraph's own bf16 rounding. Requesting the Neural Engine is not a measurement of where each op ran (Core ML decides that).

**peGM champion, derived to fp32** (`--derive-model --set-compute-dtype float32`), fp32 export, cpu, against its fp32 MPSGraph: max \|Δ\| policy logit 6.0e-4, mean 4.0e-5, value 3.1e-4, same top move 4,635 (100%). The translation is exact up to float32 accumulation order.

**The same weights derived to fp16**, mirror export, cpu, against its fp16 MPSGraph: max \|Δ\| policy logit 0.75, mean 0.029, value 0.16, same top move 4,550 (98.2%) — fp16 accumulation order.

## Tests

- `InferenceGraphRecorderTests`:
  - every architecture option records completely (`InferenceGraphTestArchitectures`);
  - attaching a recorder doesn't change the built graph (bit-identical outputs);
  - the refusals: a training-mode network, an op that bypasses `ForwardGraphOps`, an unread weight;
  - data types follow the policy tail;
  - `graph.json` round-trips; `weights.bin` is laid out as described and holds bf16-rounded values for a bf16 network.
- `CoreMLExportParityTests` gives every BatchNorm per-channel γ, β, mean and variance, so a swapped or misbroadcast BN parameter shows. It then exports every architecture variant at fp32 through the Python script and requires every output within 5e-3 on all lichess-200 positions at every batch size the export accepts. It skips without a Python that has coremltools (`$DCM_COREML_PYTHON`, else `~/.venvs/dcm-coreml/bin/python`).
- `documentation/dashboards/tests/test_coreml_export.py` covers the script's own rules: graph-format, weights-layout and duplicate-id refusals, unknown ops, constant folding, and that mirror runs each op in its recorded type.
