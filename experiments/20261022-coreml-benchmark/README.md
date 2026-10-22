# Core ML vs MPSGraph inference benchmark (E-0028)

2026-10-22, Apple M5 Max (64 GB), macOS 27.2 (26B5112a), build 2607 Release.

## Question

The owner asked for a fair test of Core ML inference against our MPSGraph forward pass. For the same weights and positions, how fast is an exported Core ML model on each compute unit?
- Batch 1 matters for the Lichess bot and `--uci`.
- Batches 64–256 match self-play and arena ticks.
- Batches up to 1024 show each engine's ceiling.

Rules and method are in [the plan](../../documentation/plans-completed/COREML_BENCHMARK_PLAN.md). In short:
- Both engines run in one process with one timer.
- The timed span runs from encoded boards in CPU memory to every output copied into the caller's buffers.
- MPSGraph is timed through `evaluateBatched`, the call self-play makes.
- Every Core ML arm must pass a parity check against its own model's MPSGraph network at every timed batch size before it is timed.
- Warm-up is untimed; arm order rotates each round.
- Core ML's planned device for each op (`MLComputePlan`) is recorded.

## Setup

| Model | `model_id` | Parameters | Shape | Encoding | Trained |
|---|---|---|---|---|---|
| `20260630-mini-YkKk-resume2-replay-step250803` | `20260710-1-amlg` | 242,609 | 2 × 64, 3×3, SE attenuate | basic30 | yes |
| `20261008-zlra-ab-fresh` | `20261008-4-Ew5t` | 8,271,279 | 3 × 128, 9×9, no SE, `fp32_from_pre_bn` tail | basic24 | no |
| `bench-v4_5block_7x7-untrained` | `20261022-9-Hhuy` | 8,445,748 | `v4_5block_7x7` preset (5 × 128, 7×7, SE scale+bias) | basic30 | no |
| `bench-10x256-untrained` | `20261022-2-PswQ` | 12,919,577 | 10 × 256, 3×3, SE scale+bias ([`bench-10x256.json`](bench-10x256.json)) | basic30 | no |

- **Precision:** all four models are bf16. Each also has a float32 copy made with `--derive-model --set-compute-dtype float32`.
- **Why untrained models are fine:** timing doesn't depend on weight values. The trained files of these shapes in `Models/` are trainer-state files, and `--derive-model` refuses those.
- **Inputs:** the 4,635 bundled test-set positions (lichess-200 + lichess-wide). Each stream walks them from its own offset.
- **Arms:**

  | Arm | Engine | Precision | Compute units |
  |---|---|---|---|
  | `mps-bfloat16` | MPSGraph | the model's own (bf16) | GPU |
  | `mps-float32` | MPSGraph | the float32 copy | GPU |
  | `coreml-fp32-gpu` | Core ML | fp32 export | `cpu-gpu` |
  | `coreml-mirror-gpu` | Core ML | mirror export (fp16 where the MPSGraph runs bf16) | `cpu-gpu` |
  | `coreml-fp16-ne` | Core ML | fp16 export | `cpu-ne` |
  | `coreml-fp16-all` | Core ML | fp16 export | `all` |
  | `coreml-fp32-cpu` | Core ML | fp32 export | `cpu` |

- **Grid:** batch sizes 1, 8, 32, 64, 128, 256, 512, 1024 × in-flight 1 and 4. Each cell gets a 0.5 s untimed warm-up per arm, then 5 rounds of 1.5 s, with arm order rotated each round.
- **Throughput** is the median of the 5 rounds. **Spread** is the fastest round over the slowest.
- **Machine state:** no other DrewsChessMachine process was running, except the Lichess bot (frozen build 2590). The bot used 5.6–8.8 CPU seconds during each model's run of 16–22 minutes (`results/machine.json`). Thermal state was "fair" in 91–105 of each model's 112 cells and "nominal" in the rest. Low Power Mode was off.

## Runs

1. **First attempt** used the trained `zlra-elitecyc` step 314767 file. `--derive-model` refused it: it is a trainer-state file with optimizer velocity. Replaced by `20261008-zlra-ab-fresh`, the same architecture, untrained.
2. **Second attempt** (`run.log`) finished mini and zlra.
   - On `20260804-v5cont-resume4-replay-step2000` the parity check refused the mirror arm: 86.97% top-move agreement at batch 1, below the 0.90 floor.
   - That model comes from the August v5 line, whose bf16 policy head has the near-tied logits fixed on 2026-09-30 (`project_bf16_head_shared_offset`). bf16 and fp16 break those ties differently, so the low agreement comes from the reference model, not from the export. Its fp32 arm agreed 100%.
   - Replaced by a fresh `v4_5block_7x7` net of the same shape. `results/20260804-v5cont-resume4-replay-step2000.config.json` is kept as the record of that attempt.
3. **Third invocation** (`run2.log`) ran the v4 preset model and the 10 × 256 model.
- The parity floor was 0.90, not the planned 0.95, because fp16 can flip an untrained net's near-tied top moves. Non-finite outputs were always refused. Every arm's measured agreement is below.

## Results

The full tables, with p10/p90 latency, minimum latency, first-call times and every cell, are in `results/<model>.txt` and `results/<model>.json`. [`summary.md`](summary.md) has every throughput cell, built by `summarize.py`. Throughput is in **millions of positions per hour**; * marks a round spread above 1.25.

### Where Core ML put the work, and parity

| Model | `coreml-*-gpu` arms planned on | `fp16-ne` / `fp16-all` planned on | ANE compile + load | Top-1 agreement: fp32 / mirror / fp16 |
|---|---|---|---|---|
| mini 0.24M | CPU 100% | CPU 100% | 0.2 s | 100% / 96.38% / 96.42% |
| zlra 8.27M (9×9) | GPU 100% | ANE 95%, CPU 5% | 5.1 s | 100% / 99.01% / 97.73% |
| v4 preset 8.45M | GPU 100% | ANE 96%, CPU 4% | 2.1 s | 100% / 99.53% / 99.59% |
| 10 × 256 12.9M | **CPU 100%** | ANE 98%, CPU 2% | 20.7 s | 100% / 98.19% / 98.53% |

Shares are by Core ML's estimated cost. The compute-units setting only lists the devices Core ML may use: under `cpu-gpu`, Core ML chose the CPU for the 10 × 256 net and for the mini net.

The worst fp32 difference was below 0.0005 on every model: the export reproduces the network exactly up to float32 accumulation order.

### Batch-1 latency, median ms (Lichess bot, `--uci`)

| Model | mps-bfloat16 | mps-float32 | coreml-fp32-gpu | coreml-mirror-gpu | coreml-fp16-ne | coreml-fp32-cpu |
|---|---|---|---|---|---|---|
| mini 0.24M | 0.670 | 0.764 | 0.123 | **0.106** | 0.122 | 0.123 |
| zlra 8.27M | 1.381 | 2.061 | 1.960 | 1.207 | **0.544** | 1.313 |
| v4 preset 8.45M | 1.389 | 2.069 | 2.332 | 1.369 | **0.424** | 1.525 |
| 10 × 256 12.9M | 2.400 | 1.767 | 2.181 | 0.745 (CPU) | **0.467** | 2.164 |

### Self-play batch sizes, in flight 1 (today's tick: one batch at a time)

| Model | Batch | mps-bfloat16 | mps-float32 | coreml-fp32-gpu | coreml-mirror-gpu | coreml-fp16-ne |
|---|---|---|---|---|---|---|
| mini 0.24M | 64 | 245.3 | **258.4** | 115.7 | 235.0 | 223.7 |
| | 256 | **509.4** | 499.1 | 119.6 | 260.8 | 247.2 |
| zlra 8.27M | 64 | 93.7 | 33.7* | 33.5* | **96.3** | 14.3 |
| | 256 | **145.2** | 39.7 | 37.3 | 115.4 | 14.5 |
| v4 preset 8.45M | 64 | **81.2** | 24.3* | 22.8 | 73.2 | 50.4 |
| | 256 | **130.3*** | 33.8* | 29.2* | 92.2 | 51.5 |
| 10 × 256 12.9M | 64 | **46.7** | 30.1* | 2.4 | 7.3 | 25.2 |
| | 256 | **73.9** | 34.9* | 2.6 | 7.8 | 26.0 |

### In flight 4 (four overlapping streams)

| Model | Batch | mps-bfloat16 | coreml-mirror-gpu | coreml-fp16-ne | mps-bfloat16 gain over in-flight 1 |
|---|---|---|---|---|---|
| zlra 8.27M | 64 | **142.8*** | 110.6 | 14.9 | +52% |
| | 256 | **174.0** | 122.4 | 14.9 | +20% |
| v4 preset 8.45M | 64 | **110.7*** | 90.6 | 59.7 | +36% |
| | 256 | **139.0*** | 101.1 | 56.3 | +7% |
| 10 × 256 12.9M | 64 | **66.9** | 9.5 | 27.7 | +43% |
| | 256 | **85.9** | 9.3 | 27.1 | +16% |

## Conclusion

- **At self-play batch sizes, our MPSGraph bf16 path is the fastest engine on every model of real size from batch 128 up.**
  - At batch 64 the zlra-shape mirror arm ties it (1.03×).
  - At batch 256 the best Core ML arm reaches 0.79× on zlra and 0.71× on the v4 preset (mirror fp16 on the GPU), and 0.35× on 10 × 256, where Core ML moved the GPU arms to the CPU and the ANE is the best it offers.
  - The ANE tops out early: about 51M/h on the v4 preset model, 26M/h on 10 × 256, and 14.5M/h on 9×9 kernels.
  - Core ML offers nothing for training-loop self-play.
- **At equal math (fp32 on the GPU) the two engines are close.**
  - `coreml-fp32-gpu` vs `mps-float32`: 0.86–0.99× on the v4 and zlra models from batch 64 up.
  - MPSGraph's lead comes from bf16, which Core ML lacks; its fp16 is the nearest substitute.
- **At batch 1 the Neural Engine is 2.5–5× faster:** 0.42–0.54 ms vs 1.38–2.40 ms. For the Lichess bot and `--uci` (one forward pass per move) that saves about 1–2 ms per move, which no time control notices.
- **Tiny networks (0.24M) are a different regime.**
  - Core ML runs them on the CPU in 0.1 ms per call, 6× faster than MPSGraph's 0.67 ms.
  - Most of MPSGraph's 0.67 ms is fixed per-call cost, not math: a net with 35× fewer parameters than the v4 preset still takes half its time (0.67 vs 1.39 ms).
- **Overlap helps MPSGraph more than any engine swap would.** Four overlapping streams give +36–52% at batch 64 and +7–20% at batch 256 over one stream. Self-play now runs one batch per tick on one network. This is an observation, not a plan.
- **Core ML's placement can't be assumed.** Asked for `cpu-gpu`, it put the 12.9M-parameter 3×3 net on the CPU (2.6M/h at batch 256, 28× slower than MPSGraph).
- **ANE start-up cost:** compiling for the ANE takes 2–21 s per model at load. That is an extra cost at bot or UCI start-up, small next to a game.

## Caveats

- **One machine, one run per model.** No repeat on another day or Mac.
  - 3, 19, 19 and 33 of the four models' 112 cells have round spread above 1.25, mostly MPSGraph arms. Ratios under about 1.2× between arms should not be over-read.
  - The rankings claimed above rest on gaps of 1.26× or more (MPSGraph bf16 vs mirror at batch 256). The batch-64 tie and the fp32 pair are given as ratios.
- **Thermal state was "fair" for most cells.** The machine had run builds and exports just before. Every arm saw the same conditions, through the rotated order.
- **`MLComputePlan` is Core ML's plan for the default input shape,** not a trace of each timed prediction.
- **Power was not measured.** The ANE may win on energy; that would need `sudo powermetrics`.
- **Three of the four models are untrained.** Timing doesn't depend on weight values, but parity agreement does: untrained nets have near-tied logits, which is why the floor was lowered to 0.90.
- **Configuration is the same as production but not identical.**
  - MPSGraph overlap was measured with four separate networks, not a pipelined single network.
  - Core ML overlap is four concurrent predictions on one model.

## Follow-ups (not planned; owner's call)

- Overlap self-play's MPSGraph batches: two networks, or a pipelined tick. The in-flight-4 gain above is the upper bound.
- If bot latency or energy ever matters: an fp16 Core ML model on the ANE for `--lichess-bot` / `--uci`.

## Reproduce

**Status: full.** Every model file is in [`models/`](models/) (Git LFS). The two minted ones can also be recreated bit-identically from their seeds.

- **Commit and build:** run at build 2607 (Release), git `a74e2c8f` plus the uncommitted benchmark code, committed right after as the commit that adds this folder.
- **Corpus:** none (bundled test-set positions).
- **Fresh nets:**
  ```sh
  BIN=<Release DrewsChessMachine binary>
  W=~/Library/Caches/DrewsChessMachine/coreml-benchmark
  $BIN --new-model --architecture v4_5block_7x7 --name bench-v4-5block --init-seed 20261022 \
      --out-model $W/bench-v4_5block_7x7-untrained.safetensors
  $BIN --new-model --architecture experiments/20261022-coreml-benchmark/bench-10x256.json --name bench-10x256 \
      --init-seed 20261022 --out-model $W/bench-10x256-untrained.safetensors
  ```
- **Exporter:** Python 3.13 venv with coremltools (`documentation/coreml-export.md`).
- **Commands, as run:**
  ```sh
  python3 scripts/dcm_coreml_benchmark.py --app "$BIN" --work $W --out experiments/20261022-coreml-benchmark/results \
      --min-top-move-agreement 0.90 \
      "$M/20260630-mini-YkKk-resume2-replay-step250803.safetensors" "$M/20261008-zlra-ab-fresh.safetensors" \
      "$M/20260804-v5cont-resume4-replay-step2000.safetensors" "$W/bench-10x256-untrained.safetensors"   # stopped at v5cont
  python3 scripts/dcm_coreml_benchmark.py --app "$BIN" --work $W --out experiments/20261022-coreml-benchmark/results \
      --min-top-move-agreement 0.90 "$W/bench-v4_5block_7x7-untrained.safetensors" "$W/bench-10x256-untrained.safetensors"
  python3 experiments/20261022-coreml-benchmark/summarize.py > experiments/20261022-coreml-benchmark/summary.md
  ```
  The script writes each model's config (`results/<model>.config.json`), then runs `--coreml-benchmark` on it.
- **Expected exactness:** the parity numbers repeat exactly. Timings are machine- and load-dependent. Expect rankings and rough ratios to repeat on an M5 Max, not the numbers.
