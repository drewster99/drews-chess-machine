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

The architecture a file resolves to is also read only from the app: `--describe-model <file>` prints it as JSON after every format-version rule and legacy resolution the loader applies, with the expanded blocks and the weight plan.

## Commands

The app (headless, then exit):

| Command | Does |
|---|---|
| `--describe-model <file>` | Prints the `ModelDescription` JSON of a model file. |
| `--export-inference-graph <file> --out <new folder>` | Writes `graph.json` (the recording), `weights.bin` (little-endian float32, plan order, graph layout; the value head recentered as every load does) and `model.json` (the description). |
| `--coreml-parity <file> --coreml <model.mlpackage> [--compute-units cpu\|cpu-gpu\|cpu-ne\|all] [--tolerance <x>]` | Runs the bundled test-set positions (lichess-200 + lichess-wide, 4,635 positions) through both and prints the largest and mean differences of every output, and how often both rank the same legal move first. Non-finite outputs are counted on both sides. With `--tolerance`, exits 3 when a difference exceeds it or an output on either side is non-finite. |

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

Core ML has no bf16, so a bf16 model's reduced-precision ops run in fp16. fp16's range is far smaller (largest finite value 65504, against bf16's ~3.4e38): an activation the bf16 network carries above it becomes inf in Core ML. `--coreml-parity` reports that as `non_finite_coreml_positions`.

In mirror mode, ops the converter adds itself follow the network: fp16 when it runs any op in reduced precision, fp32 for an all-fp32 model.

## Measured (2026-10-20, Apple M5 Max)

**`zlra-elitecyc` step 314767** (bf16, 3 × 128 ch 9 × 9, `fp32_from_pre_bn`), 4,635 positions, against its bf16 MPSGraph:

| Export | Compute units | Max \|Δ\| policy logit | Mean \|Δ\| | Max \|Δ\| value | Same top move |
|---|---|---|---|---|---|
| fp32 | cpu | 1.29 | 0.045 | 0.071 | 4,578 (98.8%) |
| mirror | cpu | 1.86 | 0.070 | 0.116 | 4,547 (98.1%) |
| mirror | cpu-ne | 1.54 | 0.047 | 0.069 | 4,578 (98.8%) |
| fp16 | cpu | 1.87 | 0.070 | 0.116 | 4,545 (98.1%) |
| fp16 | cpu-ne | 7.42 | 0.202 | 0.589 | 4,373 (94.3%) |

Every export's differences here include the MPSGraph's own bf16 rounding.

**peGM champion, derived to fp32** (`--derive-model --set-compute-dtype float32`), fp32 export, cpu, against its fp32 MPSGraph:

| Max \|Δ\| policy logit | Mean \|Δ\| | Max \|Δ\| value | Same top move |
|---|---|---|---|
| 1.04e-3 | 6.8e-5 | 4.6e-4 | 4,635 (100%) |

The translation is exact up to float32 accumulation order.

## Tests

- `InferenceGraphRecorderTests`:
  - every architecture option records completely (`InferenceGraphTestArchitectures`);
  - a recorded network computes bit-identical outputs to an unrecorded one;
  - the refusals: a training-mode network, an op that bypasses `ForwardGraphOps`, an unread weight;
  - data types follow the policy tail;
  - `graph.json` round-trips and `weights.bin` is laid out as described.
- `CoreMLExportParityTests` gives every BatchNorm per-channel γ, β, mean and variance, so a swapped or misbroadcast BN parameter shows. It then exports every architecture variant at fp32 through the Python script and requires each output within 5e-3 and the same top move on all lichess-200 positions. It skips without a Python that has coremltools (`$DCM_COREML_PYTHON`, else `~/.venvs/dcm-coreml/bin/python`).
- `documentation/dashboards/tests/test_coreml_export.py` covers the script's own rules: graph-format and weights-size refusals, unknown ops, constant folding, and mirror precision selection.
