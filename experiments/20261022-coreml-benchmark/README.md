# Core ML vs MPSGraph inference benchmark (E-0028)

2026-10-22, Apple M5 Max (64 GB), macOS 27.2 (26B5112a), build 2607 Release. Two runs: run 1 (bf16 vs Core ML) and run 2 (adds fp16 vs fp16 and the batch sizes self-play and the arena actually use). **Run 2's results replace run 1's conclusion**; run 1's tables are kept below.

## Question

The owner asked for a fair test of Core ML inference against our MPSGraph forward pass. For the same weights and positions, how fast is an exported Core ML model on each compute unit?
- Batch 1 matters for the Lichess bot and `--uci`.
- Self-play evaluates every game in one `evaluateBatched` call per tick: batch = `self_play_concurrency`, 250 in the owner's recent GUI runs (`workers=250` on the `[STATS]` lines of the 2026-10-10 to 10-15 logs), 180 by default.
- The arena splits `arena_concurrency` (200 in those runs) between two networks evaluated concurrently: about 100 per call, two in flight.
- Batches up to 1024 show each engine's ceiling.

After run 1, the owner asked: was fp16 compared with fp16, and why was only batch 256 reported? Neither had been done: run 1 compared MPSGraph bf16 with Core ML fp16, and its summary used batch 256 without checking self-play's batch. Run 2 answers both.

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

- **Precision:** all four models are bf16. Each also has a float32 copy (runs 1 and 2) and a float16 copy (run 2), made with `--derive-model --set-compute-dtype float32|float16`.
- **Why untrained models are fine:** timing doesn't depend on weight values. The trained files of these shapes in `Models/` are trainer-state files, and `--derive-model` refuses those.
- **Inputs:** the 4,635 bundled test-set positions (lichess-200 + lichess-wide). Each stream walks them from its own offset.
- **Arms:**

  | Arm | Engine | Precision | Compute units | Runs |
  |---|---|---|---|---|
  | `mps-bfloat16` | MPSGraph | the model's own (bf16) | GPU | 1, 2 |
  | `mps-float32` | MPSGraph | the float32 copy | GPU | 1, 2 |
  | `mps-float16` | MPSGraph | the float16 copy | GPU | 2 |
  | `coreml-fp32-gpu` | Core ML | fp32 export | `cpu-gpu` | 1, 2 |
  | `coreml-mirror-gpu` | Core ML | mirror export of the bf16 model (fp16 where the MPSGraph runs bf16) | `cpu-gpu` | 1, 2 |
  | `coreml-fp16-gpu` | Core ML | fp16 export (every op fp16) | `cpu-gpu` | 2 |
  | `coreml-float16-mirror-gpu` | Core ML | mirror export of the float16 copy: exactly the ops `mps-float16` runs in fp16 | `cpu-gpu` | 2 |
  | `coreml-fp16-ne` | Core ML | fp16 export | `cpu-ne` | 1, 2 |
  | `coreml-fp16-all` | Core ML | fp16 export | `all` | 1 |
  | `coreml-fp32-cpu` | Core ML | fp32 export | `cpu` | 1 |

  **The fp16-vs-fp16 pair is `mps-float16` vs `coreml-float16-mirror-gpu`:** same weights, same ops in fp16, same fp32 tails. Run 2 dropped `coreml-fp16-all` (the same package as `fp16-ne`, planned on the same devices on every model, batch-1 latency within 0.006 ms of it) and `coreml-fp32-cpu` (fastest in no run-1 cell), to fit the larger grid in the time.
- **Grid:** run 1: batch 1, 8, 32, 64, 128, 256, 512, 1024; run 2: 1, 8, 32, 64, 100, 128, 180, 250, 512, 1024. Both × in-flight 1 and 4. Each cell gets a 0.5 s untimed warm-up per arm, then 5 rounds of 1.5 s, with arm order rotated each round.
- **Throughput** is the median of the 5 rounds. **Spread** is the fastest round over the slowest.
- **Machine state:** no other DrewsChessMachine process was running, except the Lichess bot (frozen build 2590). Low Power Mode was off.
  - Run 1: the bot used 5.6–8.8 CPU seconds per model's run of 16–22 minutes (`results/machine.json`). Thermal state "fair" in 91–105 of each model's 112 cells, "nominal" in the rest.
  - Run 2 (13:07–14:45): the bot played four games: 13:47–13:56 (during zlra), 14:10–14:17 (v4 preset), 14:25–14:34 and 14:32–14:42 (10 × 256). It used 8.7, 212, 146 and 442 CPU seconds during the four models' runs of 23–28 minutes (`results-run2/machine.json`). Its GPU work is one forward pass per own move, about 30–40 per game. Thermal state "fair" in 136–160 of each model's 160 cells, "nominal" in the rest.

## Runs

1. **First attempt** used the trained `zlra-elitecyc` step 314767 file. `--derive-model` refused it: it is a trainer-state file with optimizer velocity. Replaced by `20261008-zlra-ab-fresh`, the same architecture, untrained.
2. **Second attempt** (`run.log`) finished mini and zlra.
   - On `20260804-v5cont-resume4-replay-step2000` the parity check refused the mirror arm: 86.97% top-move agreement at batch 1, below the 0.90 floor.
   - That model comes from the August v5 line, whose bf16 policy head has the near-tied logits fixed on 2026-09-30 (`project_bf16_head_shared_offset`). bf16 and fp16 break those ties differently, so the low agreement comes from the reference model, not from the export. Its fp32 arm agreed 100%.
   - Replaced by a fresh `v4_5block_7x7` net of the same shape. `results/20260804-v5cont-resume4-replay-step2000.config.json` is kept as the record of that attempt.
3. **Third invocation** (`run2.log`) ran the v4 preset model and the 10 × 256 model. Attempts 2 and 3 together are **run 1** (`results/`).
4. **Run 2** (`results-run2/`, log `results-run2/run.log`): all four models, eight arms, run-2 grid, same binary. Its first try failed at once (`results-run2/run-failed-relative-paths.log`): the script wrote repo-relative model paths into the config, which the app resolves against the config's own folder. The script now writes absolute paths.
- The parity floor was 0.90, not the planned 0.95, because fp16 can flip an untrained net's near-tied top moves. Non-finite outputs were always refused. Every arm's measured agreement is below.

## Results (run 2)

Every cell, with p10/p90 and minimum latency and first-call times, is in `results-run2/<model>.txt` and `.json`; [`summary-run2.md`](summary-run2.md) has every throughput cell (`summarize.py results-run2`). Throughput is in **millions of positions per hour**, in flight 1 unless stated; * marks a round spread above 1.25.

### Where Core ML put the work, and parity

| Model | `coreml-*-gpu` arms planned on | `fp16-ne` planned on | ANE compile + load | Top-1 agreement: fp32 / mirror / fp16 / float16-mirror |
|---|---|---|---|---|
| mini 0.24M | CPU 100% | CPU 100% | 0.3 s | 100% / 96.38% / 96.42% / 97.52% |
| zlra 8.27M (9×9) | GPU 100% | ANE 95%, CPU 5% | 6.0 s | 100% / 99.01% / 98.99% / 99.74% |
| v4 preset 8.45M | GPU 100% | ANE 96%, CPU 4% | 3.1 s | 100% / 99.53% / 99.48% / 99.89% |
| 10 × 256 12.9M | **CPU 100%** | ANE 98%, CPU 2% | 24.0 s | 100% / 98.19% / 98.19% / 99.14% |

Shares are by Core ML's estimated cost. Under `cpu-gpu`, Core ML chose the CPU for the 10 × 256 net and the mini net, in both runs. No Core ML output needed a conversion copy.

### fp16 vs fp16: Core ML ÷ MPSGraph (`coreml-float16-mirror-gpu` ÷ `mps-float16`)

| Model | 1 | 8 | 32 | 64 | 100 | 128 | 180 | 250 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|---|---|
| zlra 8.27M (GPU) | 1.14 | 1.23 | 1.21 | 1.15 | 1.04 | 1.00 | 1.06 | 0.96 | 1.03 | 0.99 |
| v4 preset 8.45M (GPU) | 1.02 | 1.12 | 1.15 | 1.11 | 1.11 | 1.08 | 1.11 | 1.09 | 0.97 | 1.01 |
| 10 × 256 12.9M (Core ML on CPU) | 3.21 | 0.67 | 0.25 | 0.16 | 0.14 | 0.12 | 0.12 | 0.12 | 0.10 | 0.12 |
| mini 0.24M (Core ML on CPU) | 7.19 | 6.04 | 1.62 | 0.94 | 0.76 | 0.67 | 0.57 | 0.51 | 0.44 | 0.45 |

Where both run on the GPU, Core ML is 0.96–1.23× MPSGraph at equal fp16 math: level from batch 100 up, ahead below it.

### bf16 vs fp16 inside MPSGraph (`mps-bfloat16` ÷ `mps-float16`)

| Model | 1 | 8 | 32 | 64 | 100 | 128 | 180 | 250 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|---|---|
| zlra 8.27M | 1.00 | 1.00 | 1.06 | 1.13 | 1.12 | 1.10 | 1.14 | 1.05 | 1.15 | 1.06 |
| v4 preset 8.45M | 0.97 | 0.98 | 1.00 | 1.05 | 1.10 | 1.05 | 1.12 | 1.27 | 1.10 | 1.18 |
| 10 × 256 12.9M | 1.00 | 0.97 | 0.99 | 0.99 | 1.03 | 1.00 | 1.08 | 1.03 | 1.01 | 1.14 |
| mini 0.24M | 1.10 | 1.04 | 1.01 | 0.99 | 0.96 | 1.02 | 0.99 | 1.00 | 0.98 | 1.00 |

### Every batch size: MPSGraph bf16 against the best Core ML arm

| Model | | 1 | 8 | 32 | 64 | 100 | 128 | 180 | 250 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zlra 8.27M | mps-bfloat16 | 2.6 | 20.4 | 61.9 | 96.9 | 108.8 | 125.2 | 131.4 | 131.5 | 133.5 | 133.4 |
| | best Core ML | 6.5 ne | 25.1 f16m | 71.8 fp16 | 100.2 fp16 | 103.7 mir | 117.7 mir | 121.8 f16m | 124.7 mir | 119.3 f16m | 131.9* fp16 |
| | ratio | 2.49 | 1.23 | 1.16 | 1.03 | 0.95 | 0.94 | 0.93 | 0.95 | 0.89 | 0.99 |
| v4 preset 8.45M | mps-bfloat16 | 2.4 | 19.0 | 58.1 | 89.0* | 84.9* | 92.6 | 100.8 | 120.9* | 132.9 | 120.9 |
| | best Core ML | 8.3 ne | 33.8 ne | 67.8 fp16 | 96.1 mir | 85.7 f16m | 94.8 f16m | 99.7 f16m | 104.0 f16m | 122.2 fp16 | 111.1 mir |
| | ratio | 3.39 | 1.78 | 1.17 | 1.08 | 1.01 | 1.02 | 0.99 | 0.86 | 0.92 | 0.92 |
| 10 × 256 12.9M | mps-bfloat16 | 1.4 | 9.7 | 29.3 | 47.2 | 55.6 | 62.8 | 66.3 | 73.1 | 82.3* | 79.8* |
| | best Core ML | 7.3 ne | 21.2 ne | 24.4 ne | 25.8 ne | 25.9* ne | 25.3 ne | 25.6 ne | 25.9 ne | 25.1 ne | 25.2 ne |
| | ratio | 5.17 | 2.20 | 0.84 | 0.55 | 0.47 | 0.40 | 0.39 | 0.35 | 0.30 | 0.32 |
| mini 0.24M | mps-bfloat16 | 4.7 | 21.2* | 123.1* | 244.6 | 310.3* | 391.9 | 451.3 | 531.2 | 644.1 | 686.1 |
| | best Core ML (CPU) | 31.8 fp16 | 126.1* fp16 | 209.1 fp16 | 253.5 fp16 | 256.7 fp16 | 288.0 fp16 | 288.4 fp16 | 296.0 fp16 | 318.1 fp16 | 330.6 fp16 |
| | ratio | 6.75 | 5.94 | 1.70 | 1.04 | 0.83 | 0.73 | 0.64 | 0.56 | 0.49 | 0.48 |

Arm abbreviations: ne = `coreml-fp16-ne`, fp16 = `coreml-fp16-gpu`, mir = `coreml-mirror-gpu`, f16m = `coreml-float16-mirror-gpu`. The ratio is the best Core ML arm over `mps-bfloat16`. Columns 100, 180 and 250 are the arena call, default self-play and the owner's self-play.

### Batch-1 latency, median ms (Lichess bot, `--uci`)

| Model | mps-bfloat16 | mps-float16 | coreml-mirror-gpu | coreml-float16-mirror-gpu | coreml-fp16-ne |
|---|---|---|---|---|---|
| mini 0.24M | 0.704 | 0.785 | 0.105 (CPU) | 0.103 (CPU) | 0.119 (CPU) |
| zlra 8.27M | 1.361 | 1.358 | 1.199 | 1.191 | **0.535** |
| v4 preset 8.45M | 1.374 | 1.368 | 1.338 | 1.345 | **0.408** |
| 10 × 256 12.9M | 2.421 | 2.459 | 0.754 (CPU) | 0.749 (CPU) | **0.467** |

### In flight 4, self-play-size batches

| Model | Batch | mps-bfloat16 | gain over in flight 1 | mps-float16 | coreml-float16-mirror-gpu | coreml-fp16-ne |
|---|---|---|---|---|---|---|
| zlra 8.27M | 100 | **147.2** | +35% | 118.9 | 125.3 | 15.1 |
| | 250 | **157.3** | +20% | 125.2 | 128.2 | 14.7 |
| v4 preset 8.45M | 100 | **117.7** | +39% | 110.6 | 113.5 | 54.7 |
| | 250 | **151.4*** | +25% | 119.6 | 131.4 | 57.2 |
| 10 × 256 12.9M | 100 | 84.2 | +52% | **84.6*** | 9.2 | 27.7 |
| | 250 | **92.0** | +26% | 76.1* | 10.9 | 27.3 |

### Run 2 against run 1

On the cells both runs measured (same arm, batch and in-flight), run 2 ÷ run 1 has a median of 0.94–1.06 for every arm on the mini, zlra and 10 × 256 models. On the v4 preset model Core ML ran faster in run 2: median 1.30 for `coreml-mirror-gpu` and 1.20 for `coreml-fp32-gpu`, against 1.08 for `mps-bfloat16`. The bot's games would slow run 2, not speed it up, so this is run-to-run variation: a gap below about 1.3× between two engines in one run may not repeat.

## Conclusion

- **fp16 vs fp16 on the GPU, Core ML is level with MPSGraph or ahead.** At equal math on the 8M-parameter models, Core ML is 0.96–1.23× MPSGraph: level from batch 100 up (0.96–1.11×), ahead below that (1.11–1.23× at batches 8–64). Run 1's "Core ML's fp16 is slower" compared Core ML fp16 with MPSGraph *bf16*, and was wrong as a statement about fp16.
- **MPSGraph's edge is bf16, which Core ML lacks.** On the 8M models, MPSGraph bf16 is 1.05–1.27× its own fp16 from batch 64 up, and 0.97–1.06× below.
- **At the batch sizes self-play and the arena use, no engine wins meaningfully on the 8M models.** At batches 100–250, the best Core ML arm is 0.86–1.02× MPSGraph bf16 in run 2; run 1 measured 0.71–0.79× at 256 on the same models. Both are within the run-to-run variation above. Moving self-play to Core ML would gain nothing and would likely lose 5–15%.
- **Core ML's device placement decides the rest.** Under `cpu-gpu` it ran the 12.9M-parameter 10 × 256 net on the CPU, in both runs. There, the best Core ML option (the ANE) is 0.35–0.47× MPSGraph at batches 100–250.
- **Below batch 64, Core ML is faster.** On the GPU, 1.03–1.23× at batches 8–64 on the 8M models. The ANE is 1.8–2.2× at batch 8 on the v4 and 10 × 256 nets, and 2.5–5.2× at batch 1 (0.41–0.54 ms against 1.36–2.42 ms). For the Lichess bot and `--uci` that saves 1–2 ms per move, which no time control notices.
- **The ANE saturates early**, at 14–53M positions/hour depending on the shape. 9×9 kernels suit it worst.
- **Tiny networks (0.24M) run on the CPU under Core ML,** 5.9–6.8× faster than MPSGraph at batches 1–8: most of MPSGraph's 0.7 ms per call is fixed cost. MPSGraph leads from batch 100.
- **In isolation, overlap helps MPSGraph more than any engine swap.** Four overlapping streams gain +35% to +52% at batch 100 and +20% to +26% at 250 on the 8M and 12.9M models (run 2; run 1: +36% to +52% at 64, +7% to +20% at 256). Most of that gain needs four times the positions in flight: at equal positions in flight it is 0.85–1.06× at about 250 and 1.05–1.25× at about 1,000. Real self-play runs at 9% of the GPU-only rate, so the forward pass is not its bottleneck. Full analysis: [`OVERLAP.md`](OVERLAP.md); implementation issue [#23](https://github.com/drewster99/drews-chess-machine/issues/23).

## Caveats

- **One machine, two runs.** Run-to-run variation reached 1.3× on one model (see "Run 2 against run 1").
  - Round spread above 1.25: run 1, 3, 19, 19 and 33 of the four models' 112 cells; run 2, 28, 12, 22 and 35 of 160.
  - Only gaps of about 1.3× or more should be read as real. The fp16-vs-fp16 ratios and the self-play-batch ratios on the 8M models are all smaller: they say "about equal", not which is faster.
- **The Lichess bot played four games during run 2,** during the zlra, v4 and 10 × 256 models (Setup). Arm order rotated within each cell, so every arm saw the same conditions on average. Its GPU use was about one forward pass per 15 s.
- **Production runs self-play beside the trainer on the same GPU.** Neither run measured that contention.
- **Thermal state was "fair" for most cells.** Every arm saw the same conditions, through the rotated order.
- **`MLComputePlan` is Core ML's plan for the default input shape,** not a trace of each timed prediction.
- **Power was not measured.** The ANE may win on energy; that would need `sudo powermetrics`.
- **Three of the four models are untrained.** Timing doesn't depend on weight values, but parity agreement does: untrained nets have near-tied logits, which is why the floor was lowered to 0.90.
- **Configuration is the same as production but not identical.**
  - MPSGraph overlap was measured with four separate networks, not a pipelined single network.
  - Core ML overlap is four concurrent predictions on one model.

## Run 1 results (conclusion superseded)

Every run-1 cell is in [`summary.md`](summary.md) and `results/`. Run 1's conclusion said MPSGraph bf16 was the fastest engine from batch 128 up (Core ML at most 0.79× at 256) and that Core ML's fp16 was slower. Run 2 corrects both (Conclusion above). The tables are kept as measured.

### Where Core ML put the work, and parity (run 1)

| Model | `coreml-*-gpu` arms planned on | `fp16-ne` / `fp16-all` planned on | ANE compile + load | Top-1 agreement: fp32 / mirror / fp16 |
|---|---|---|---|---|
| mini 0.24M | CPU 100% | CPU 100% | 0.2 s | 100% / 96.38% / 96.42% |
| zlra 8.27M (9×9) | GPU 100% | ANE 95%, CPU 5% | 5.1 s | 100% / 99.01% / 97.73% |
| v4 preset 8.45M | GPU 100% | ANE 96%, CPU 4% | 2.1 s | 100% / 99.53% / 99.59% |
| 10 × 256 12.9M | **CPU 100%** | ANE 98%, CPU 2% | 20.7 s | 100% / 98.19% / 98.53% |

The worst fp32 difference was below 0.0005 on every model: the export reproduces the network exactly up to float32 accumulation order.

### Batch-1 latency, median ms (run 1)

| Model | mps-bfloat16 | mps-float32 | coreml-fp32-gpu | coreml-mirror-gpu | coreml-fp16-ne | coreml-fp32-cpu |
|---|---|---|---|---|---|---|
| mini 0.24M | 0.670 | 0.764 | 0.123 | **0.106** | 0.122 | 0.123 |
| zlra 8.27M | 1.381 | 2.061 | 1.960 | 1.207 | **0.544** | 1.313 |
| v4 preset 8.45M | 1.389 | 2.069 | 2.332 | 1.369 | **0.424** | 1.525 |
| 10 × 256 12.9M | 2.400 | 1.767 | 2.181 | 0.745 (CPU) | **0.467** | 2.164 |

### Batches 64 and 256, in flight 1 (run 1)

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

### In flight 4 (run 1)

| Model | Batch | mps-bfloat16 | coreml-mirror-gpu | coreml-fp16-ne | mps-bfloat16 gain over in-flight 1 |
|---|---|---|---|---|---|
| zlra 8.27M | 64 | **142.8*** | 110.6 | 14.9 | +52% |
| | 256 | **174.0** | 122.4 | 14.9 | +20% |
| v4 preset 8.45M | 64 | **110.7*** | 90.6 | 59.7 | +36% |
| | 256 | **139.0*** | 101.1 | 56.3 | +7% |
| 10 × 256 12.9M | 64 | **66.9** | 9.5 | 27.7 | +43% |
| | 256 | **85.9** | 9.3 | 27.1 | +16% |

At equal fp32 math on the GPU (`coreml-fp32-gpu` vs `mps-float32`), run 1 measured 0.86–0.99× on the v4 and zlra models from batch 64 up.

## Follow-ups (not planned; owner's call)

- Overlap self-play's MPSGraph batches: [#23](https://github.com/drewster99/drews-chess-machine/issues/23), analysis in [`OVERLAP.md`](OVERLAP.md). Measure a tick's phases first.
- If bot latency or energy ever matters: an fp16 Core ML model on the ANE for `--lichess-bot` / `--uci`.
- A benchmark with the trainer running, to measure self-play under production GPU contention.

## Reproduce

**Status: full.** Every model file is in [`models/`](models/) (Git LFS). The two minted ones can also be recreated bit-identically from their seeds.

- **Commit and build:** both runs used build 2607 (Release), git `a74e2c8f` plus the benchmark code, committed as `1e4969f1`. After run 1 only the plan path in comments and help text changed. After run 2 the text table's label column was widened to fit long arm labels (no timing change).
- **Script:** run 1 used `scripts/dcm_coreml_benchmark.py` as of `1e4969f1` (seven arms). Run 2 used the version committed with this write-up (eight arms, float16 copies, absolute paths in the config).
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
- **Commands, as run (run 1):**
  ```sh
  python3 scripts/dcm_coreml_benchmark.py --app "$BIN" --work $W --out experiments/20261022-coreml-benchmark/results \
      --min-top-move-agreement 0.90 \
      "$M/20260630-mini-YkKk-resume2-replay-step250803.safetensors" "$M/20261008-zlra-ab-fresh.safetensors" \
      "$M/20260804-v5cont-resume4-replay-step2000.safetensors" "$W/bench-10x256-untrained.safetensors"   # stopped at v5cont
  python3 scripts/dcm_coreml_benchmark.py --app "$BIN" --work $W --out experiments/20261022-coreml-benchmark/results \
      --min-top-move-agreement 0.90 "$W/bench-v4_5block_7x7-untrained.safetensors" "$W/bench-10x256-untrained.safetensors"
  python3 experiments/20261022-coreml-benchmark/summarize.py > experiments/20261022-coreml-benchmark/summary.md
  ```
- **Commands, as run (run 2),** from the repository root, with a fresh `--work` folder (packages are reused by name, and their batch sizes are fixed at export):
  ```sh
  E=experiments/20261022-coreml-benchmark
  python3 scripts/dcm_coreml_benchmark.py --app "$BIN" --work ~/Library/Caches/DrewsChessMachine/coreml-benchmark-run2 \
      --out $E/results-run2 --min-top-move-agreement 0.90 --batch-sizes 1,8,32,64,100,128,180,250,512,1024 \
      $E/models/20260630-mini-YkKk-resume2-replay-step250803.safetensors $E/models/20261008-zlra-ab-fresh.safetensors \
      $E/models/bench-v4_5block_7x7-untrained.safetensors $E/models/bench-10x256-untrained.safetensors
  python3 $E/summarize.py $E/results-run2 > $E/summary-run2.md
  ```
  The script writes each model's config (copied to `results*/<model>.config.json`), then runs `--coreml-benchmark` on it.
- **Expected exactness:** the parity numbers repeat exactly. Timings are machine- and load-dependent. Expect rankings and rough ratios to repeat on an M5 Max, not the numbers; gaps under about 1.3× may flip.
