# Core ML vs MPSGraph inference benchmark

Owner request 2026-10-22: "a fair test to run our Core ML inference and benchmark it against our MPSGraph".

## Question

For the same weights and the same positions, how fast is an exported Core ML model (`documentation/coreml-export.md`) against the app's MPSGraph forward pass on each compute unit, at the batch sizes the app uses: 1 for the Lichess bot and `--uci`, 64–256 for self-play and arena ticks, and up to 1024 to find each engine's ceiling?

## What makes it fair

- **One process, one harness, one clock.** `--coreml-benchmark` times both engines the same way, with `ContinuousClock`, and loads them through the code `--coreml-parity` already uses.
- **The same work boundary.** The timer runs from encoded boards in CPU memory to every output (policy logits, value, W/D/L) copied into caller-owned CPU buffers. Board encoding is shared and stays outside the timer.
  - **MPSGraph:** `ChessNetwork.evaluateBatched(batchBoardsPointer:…)`, the self-play hot path through `GPUSubmission`. Its `consume` copies the three outputs into the stream's buffers.
  - **Core ML:** one preallocated float32 `MLMultiArray` input per stream and batch size, filled by `memcpy`, then `MLModel.prediction(from:)` (async), then the three outputs copied into the stream's buffers (a contiguous float32 output by `memcpy`, any other layout by conversion; the conversions are counted in the results).
- **Correct outputs first.** Every Core ML arm runs the parity check (`CoreMLParity.compare`) against its own model file's MPSGraph network at every batch size before it is timed. An arm with a non-finite output, or with top-move agreement below the config's floor, is refused.
- **Precision pairs named, not hidden.** Core ML has no bf16. The arms:

  | Arm | Engine | Precision | Compute units | Answers |
  |---|---|---|---|---|
  | `mps-<dtype>` (e.g. `mps-bfloat16`) | MPSGraph | as trained (bf16) | GPU | Production baseline |
  | `mps-float32` | MPSGraph | fp32 (`--derive-model --set-compute-dtype float32`) | GPU | Same-math baseline |
  | `coreml-fp32-gpu` | Core ML | fp32 export | cpu-gpu | Engine overhead at equal math (vs `mps-float32`) |
  | `coreml-mirror-gpu` | Core ML | mirror export | cpu-gpu | Could Core ML on the GPU replace MPSGraph (vs `mps-bfloat16`)? |
  | `coreml-fp16-ne` | Core ML | fp16 export | cpu-ne | What the Neural Engine buys |
  | `coreml-fp16-all` | Core ML | fp16 export | all | Core ML's own placement |
  | `coreml-fp32-cpu` | Core ML | fp32 export | cpu | CPU reference |

- **Compile and first runs excluded, but reported.** Load time is recorded per arm: graph build plus `loadWeights`, or Core ML compile plus load. Each arm's first call at each batch size is recorded as `first_call_ms`. The first steady-state round for a cell is preceded by a warm-up that is not timed.
- **Interleaved, rotated rounds.** For each batch size and in-flight count, the arms run in `rounds` rounds of `round_seconds` each. Round *r* starts at arm *r* mod *A*, so drift (heat, background load) is spread over every arm. No random draws.
- **Two concurrency modes, reported apart.**
  - In flight 1: one request at a time, as the self-play and arena ticks do today.
  - In flight N: N concurrent streams. For MPSGraph that means N `ChessNetwork` instances, each with its own queue, because a network serializes its own calls. For Core ML it means N concurrent predictions on one `MLModel`. Each is that engine's idiomatic way to overlap work.
- **Varied inputs.** Each stream walks the bundled test-set positions (4,635, encoded once) from its own offset, so no two consecutive calls see the same batch.
- **Machine state recorded.**
  - Per run: build, configuration (the CLI refuses a Debug build unless `allow_debug_build` is set, because Swift debug code would slow only the MPSGraph side), chip, OS, and Low Power Mode.
  - Per round: thermal state; the report gives the worst seen per cell.
  - The wrapper script refuses to run while another DrewsChessMachine process other than the Lichess bot is up, and records the bot's CPU seconds before and after (a game played mid-run shows there).

## Config

`--coreml-benchmark <config.json> [--json-out <new file>]`. Every key is required except `allow_debug_build`. An unknown key is refused. Relative paths are read from the config's folder.

```json
{
  "batch_sizes": [1, 8, 32, 64, 128, 256, 512, 1024],
  "in_flight": [1, 4],
  "rounds": 5,
  "round_seconds": 1.5,
  "min_top_move_agreement": 0.95,
  "allow_debug_build": false,
  "arms": [
    {"label": "mps-bfloat16", "engine": "mpsgraph", "model": "model.safetensors"},
    {"label": "coreml-fp16-ne", "engine": "coreml", "model": "model.safetensors",
     "coreml": "model-fp16.mlpackage", "compute_units": "cpu-ne"}
  ]
}
```

The first arm is the baseline for the table's relative column. `scripts/dcm_coreml_benchmark.py` writes this config for each model.

## Statistics

Per arm × batch size × in-flight count:
- latency: median, p10, p90 and min over every timed call of every round (ms);
- throughput: per-round positions/s; the report gives the median of the rounds, with min and max, and positions/hour (× 3600);
- calls, rounds, worst thermal state.

## Models

Timing doesn't depend on weight values, so one untrained model covers a size no trained model has.

| Model | Parameters | Shape | Why |
|---|---|---|---|
| `mini-YkKk` step 250803 | 0.24M | 2 × 64, 3×3, SE | Small: per-call overhead dominates |
| `zlra-elitecyc` step 314767 | 8.27M | 3 × 128, 9×9 | A current line; large kernels |
| `v5cont-resume4` step 2000 | 8.45M | 5 × 128, 7×7, SE (default preset shape) | The default architecture |
| `bench-10x256` (untrained, `--new-model`) | ~12M | 10 × 256, 3×3, SE | Larger than any trained model |

## Deliverables

1. `Network/CoreMLBenchmark.swift`: config, runners, statistics.
2. `--coreml-benchmark <config.json> [--json-out <new file>]` in `InferenceGraphExportCLI`, with help text.
3. `scripts/dcm_coreml_benchmark.py`: derive the fp32 copy, export the three Core ML packages, write the config, check the machine, run, save the JSON. Work files (derived models, packages) go to a work folder outside the repo.
4. Tests (`CoreMLBenchmarkTests`): config refusals, statistics and rotation. Plus an MPSGraph-only end-to-end run on a tiny model, which proves every cell gets numbers.
5. `experiments/20261022-coreml-benchmark/`: README with results and a Reproduce section, and the JSON.
6. `documentation/coreml-export.md`: the command, and a pointer to the results.

## Validation

- Build Release with no new warnings; `CoreMLBenchmarkTests` pass; the full suite passes before commit.
- Every Core ML arm passes parity in the run (no non-finite output, top-move agreement ≥ 0.95).
- Spread check: each cell's round-throughput max/min ratio is in the JSON. A cell above 1.25 is flagged in the report as noisy.

## As implemented (2026-10-22)

- **Done as planned:** the command, the runners, the statistics, the script, the tests and the write-up (`experiments/20261022-coreml-benchmark/`, E-0028).
- **Added: planned placement.** Every Core ML arm's `MLComputePlan` placement (preferred device per operation, and share of estimated cost) is in the report. The smoke run showed Core ML putting `cpu-gpu` arms on the CPU, which the compute-units setting alone can't show.
- **Changed: models.** Three of the four are untrained nets of the planned shapes, from `--new-model` with seed 20261022 or an existing fresh file.
  - `--derive-model` refuses trainer-state files, and the trained files of those shapes are trainer-state files.
  - The `v5cont` file was refused by the parity check (86.97% top-move agreement on the mirror arm), because of its pre-fix bf16 policy-head ties. It was replaced by the `v4_5block_7x7` preset.
- **Changed: parity floor.** It was run at 0.90, not 0.95, because fp16 flips some of an untrained net's near-tied top moves. Every arm's measured agreement is reported.
- **Added after run 1: fp16 vs fp16 and real batch sizes (run 2).** The owner asked whether fp16 had been compared with fp16, and why only batch 256 was reported. The script now also derives a float16 copy and runs `mps-float16`, `coreml-fp16-gpu` and `coreml-float16-mirror-gpu` (the float16 copy's mirror export, the exact fp16-vs-fp16 pair), and drops `coreml-fp16-all` and `coreml-fp32-cpu`. Run 2 used batches 100, 180 and 250 (the arena's call, default and owner self-play). It writes absolute paths into the config.
- **Changed: models can be added.** `scripts/dcm_coreml_benchmark.py` adds a model to an existing results folder and refuses one already there.
- **Added after the 2026-10-23 review.**
  - The arm-order rotation continues from cell to cell (`armOrder(armCount:cell:round:roundsPerCell:)`). Restarting it in every cell left the arms' average places unequal in both runs; the README's caveats say so. Over a run the bias is now spread across the arms (exactly balanced when cells × rounds is a multiple of the arm count); one cell is balanced only when its rounds are.
  - `first_call_ms` is documented as cold only for MPSGraph arms; a Core ML arm's parity check has already run every batch size.
  - The non-contiguous-input error names the arm and the stream.
  - The script reuses a derived copy or a package only when its `<artifact>.source.json` (source SHA-256, operation and settings, app version) matches what it would make, and refuses otherwise.
