# Overlapping MPSGraph forward passes (from E-0028)

An analysis of the in-flight-4 cells of the Core ML benchmark ([README](README.md), E-0028). No new run: the numbers are from the two runs already recorded, `results/` (run 1) and `results-run2/` (run 2), and from one production self-play log. Tables come from [`overlap.py`](overlap.py).

Implementation issue: [#23](https://github.com/drewster99/drews-chess-machine/issues/23).

## Summary

- **In isolation, four overlapping MPSGraph streams beat one.** On the 8M and 12.9M models, in-flight 4 beats in-flight 1 by +35% to +52% at batch 100 and +20% to +26% at batch 250 (run 2, `mps-bfloat16`). Below batch 64 the gain is +73% to +154%; on the 0.24M model it is +147% to +385% at every batch size.
- **Most of that gain needs four times as many positions in flight.** In-flight 4 at batch 250 is 1,000 positions in flight. Splitting today's 250 games into four streams of about 64 gives 0.98–1.06× on the 8M+ models in run 2, and 0.85–0.98× in run 1: nothing. Four streams of 250 against one batch of 1,024 gives 1.15–1.25× (run 2) and 1.05–1.17× (run 1).
- **Real self-play is nowhere near the GPU-only rate.** The owner's 250-game zlra run ran at a median of 11.9M positions/hour, a 75.7 ms tick. The same model's forward pass at batch 250 took 6.67 ms in isolation (131.5M positions/hour). The other ~69 ms of each tick is CPU work, waiting behind the trainer on the shared GPU, or both; nothing logged today says which.
- **So the measured gain is not a forecast for self-play.** What overlap can do depends on where those ~69 ms go. The first step of the issue is to measure that.

## What the benchmark measured

- `--coreml-benchmark` builds one `ChessNetwork` per stream for an MPSGraph arm: four separate networks, each with its own command queue and serial `executionQueue` (`InferenceGraphExportCLI+Benchmark.swift`). A single `ChessNetwork` cannot overlap with itself: its calls serialize on its queue.
- In-flight N runs N streams together for each 1.5 s round (`CoreMLBenchmark.measureRound`). Each stream calls `evaluateBatched` back to back from its own starting position.
- Throughput is all streams' positions ÷ the round's wall time. Latency is each call's own time, so at in-flight 4 it includes waiting behind the other streams.
- Boards are encoded before timing. No CPU work runs between calls: no encoding, legal-move generation, sampling, game-end handling or replay-buffer writes. Self-play does all of these every tick.
- Nothing else used the GPU, apart from the Lichess bot's four games during run 2 (one forward pass about every 15 s).

## Results, run 2: every batch size

Throughput in millions of positions/hour (median round); * marks a round spread above 1.25. "Gain" is in-flight 4 ÷ in-flight 1 − 1 for the same arm and batch. The last column puts the best Core ML arm at in-flight 4 against `mps-bfloat16` at in-flight 4.

#### mini YkKk (0.24M)

| Batch | mps-bfloat16 in flight 1 | in flight 4 | gain | ms/call in flight 1 | ms/call in flight 4 | mps-float32 gain | mps-float16 gain | best Core ML in flight 4 ÷ bf16 in flight 4 |
|---|---|---|---|---|---|---|---|---|
| 1 | 4.7 | 12.9* | +174% | 0.70 | 1.00 | +199% | +177% | 5.60 (coreml-mirror-gpu) |
| 8 | 21.2* | 103.0 | +385% | 1.32 | 1.04 | +477% | +372% | 3.62 (coreml-float16-mirror-gpu) |
| 32 | 123.1* | 429.3 | +249% | 0.88 | 1.01 | +224% | +241% | 1.31 (coreml-fp16-gpu) |
| 64 | 244.6 | 838.0 | +243% | 0.84 | 1.02 | +220% | +230% | 0.67 (coreml-fp16-gpu) |
| 100 | 310.3* | 1146.0 | +269% | 1.05 | 1.19 | +217% | +243% | 0.51 (coreml-fp16-gpu) |
| 128 | 391.9 | 1276.5 | +226% | 1.13 | 1.38 | +206% | +240% | 0.46 (coreml-fp16-gpu) |
| 180 | 451.3 | 1510.8 | +235% | 1.37 | 1.65 | +210% | +227% | 0.38 (coreml-fp16-gpu) |
| 250 | 531.2 | 1612.2 | +204% | 1.63 | 2.19 | +186% | +198% | 0.35 (coreml-fp16-gpu) |
| 512 | 644.1 | 1592.2 | +147% | 2.74 | 4.57 | +103% | +145% | 0.37 (coreml-fp16-gpu) |
| 1024 | 686.1 | 1738.0 | +153% | 5.19 | 8.49 | +98% | +168% | 0.33 (coreml-mirror-gpu) |

#### zlra shape (8.27M)

| Batch | mps-bfloat16 in flight 1 | in flight 4 | gain | ms/call in flight 1 | ms/call in flight 4 | mps-float32 gain | mps-float16 gain | best Core ML in flight 4 ÷ bf16 in flight 4 |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.6 | 5.9 | +129% | 1.36 | 2.38 | +129% | +133% | 2.39 (coreml-fp16-ne) |
| 8 | 20.4 | 42.8 | +110% | 1.39 | 2.69 | +45% | +74% | 1.41 (coreml-float16-mirror-gpu) |
| 32 | 61.9 | 109.1 | +76% | 1.82 | 4.30 | +11% | +60% | 1.00 (coreml-mirror-gpu) |
| 64 | 96.9 | 137.0 | +41% | 2.36 | 6.54 | +24% | +38% | 0.89 (coreml-float16-mirror-gpu) |
| 100 | 108.8 | 147.2 | +35% | 3.26 | 9.61 | +12% | +22% | 0.85 (coreml-float16-mirror-gpu) |
| 128 | 125.2 | 158.2* | +26% | 3.68 | 11.34 | +12% | +15% | 0.82 (coreml-float16-mirror-gpu) |
| 180 | 131.4 | 167.3 | +27% | 4.84 | 15.37 | +9% | +21% | 0.85 (coreml-mirror-gpu) |
| 250 | 131.5 | 157.3 | +20% | 6.67 | 22.55 | +5% | -0% | 0.82 (coreml-fp16-gpu) |
| 512 | 133.5 | 161.0* | +21% | 14.50 | 45.38 | +15% | +18% | 0.79 (coreml-float16-mirror-gpu) |
| 1024 | 133.4 | 174.4 | +31% | 25.62 | 84.69 | +4% | +2% | 0.75 (coreml-fp16-gpu) |

#### v4_5block_7x7 preset (8.45M)

| Batch | mps-bfloat16 in flight 1 | in flight 4 | gain | ms/call in flight 1 | ms/call in flight 4 | mps-float32 gain | mps-float16 gain | best Core ML in flight 4 ÷ bf16 in flight 4 |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.4 | 5.7 | +134% | 1.37 | 2.47 | +101% | +130% | 3.44 (coreml-fp16-ne) |
| 8 | 19.0 | 41.6 | +118% | 1.46 | 2.71 | +70% | +108% | 1.44 (coreml-mirror-gpu) |
| 32 | 58.1 | 100.5 | +73% | 1.91 | 4.37 | +26% | +56% | 1.18 (coreml-float16-mirror-gpu) |
| 64 | 89.0* | 128.1 | +44% | 2.50 | 7.15 | +20% | +29% | 1.01 (coreml-float16-mirror-gpu) |
| 100 | 84.9* | 117.7 | +39% | 4.13 | 11.43 | +15% | +43% | 0.96 (coreml-float16-mirror-gpu) |
| 128 | 92.6 | 123.1 | +33% | 4.75 | 15.88 | +3% | +19% | 0.91 (coreml-fp16-gpu) |
| 180 | 100.8 | 116.9 | +16% | 6.38 | 22.21 | +3% | +17% | 0.95 (coreml-mirror-gpu) |
| 250 | 120.9* | 151.4* | +25% | 7.64 | 24.13 | +17% | +26% | 0.89 (coreml-fp16-gpu) |
| 512 | 132.9 | 139.6 | +5% | 13.80 | 53.17 | +2% | +2% | 0.89 (coreml-fp16-gpu) |
| 1024 | 120.9 | 131.6 | +9% | 29.46 | 110.05 | -0% | +5% | 0.82 (coreml-mirror-gpu) |

#### 10 x 256 (12.9M)

| Batch | mps-bfloat16 in flight 1 | in flight 4 | gain | ms/call in flight 1 | ms/call in flight 4 | mps-float32 gain | mps-float16 gain | best Core ML in flight 4 ÷ bf16 in flight 4 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.4 | 3.5 | +147% | 2.42 | 4.07 | +167% | +138% | 5.37 (coreml-fp16-ne) |
| 8 | 9.7 | 24.5 | +154% | 2.75 | 4.61 | +111% | +142% | 1.23 (coreml-fp16-ne) |
| 32 | 29.3 | 59.0 | +102% | 3.86 | 7.72 | +35% | +105% | 0.48 (coreml-fp16-ne) |
| 64 | 47.2 | 71.3 | +51% | 4.79 | 12.77 | +16% | +44% | 0.39 (coreml-fp16-ne) |
| 100 | 55.6 | 84.2 | +52% | 6.45 | 17.03 | +17% | +57% | 0.33 (coreml-fp16-ne) |
| 128 | 62.8 | 82.2 | +31% | 7.24 | 21.63 | +4% | +12% | 0.33 (coreml-fp16-ne) |
| 180 | 66.3 | 85.0* | +28% | 9.60 | 29.68 | +2% | +11% | 0.32 (coreml-fp16-ne) |
| 250 | 73.1 | 92.0 | +26% | 12.18 | 38.88 | +0% | +8% | 0.30 (coreml-fp16-ne) |
| 512 | 82.3* | 92.9 | +13% | 22.19 | 78.63 | +1% | -5% | 0.28 (coreml-fp16-ne) |
| 1024 | 79.8* | 91.0* | +14% | 45.74 | 162.93 | +1% | +5% | 0.29 (coreml-fp16-ne) |

The `mps-float32` arm gains least on the 8M+ models (+0% to +17% from batch 100 up): at in-flight 1 it is already 2–4× slower than bf16 there, so the GPU is busy. `mps-float16` falls between. The Core ML arms do not catch up under overlap: from batch 100 up, the best of them is 0.75–0.96× `mps-bfloat16` at in-flight 4 on the 8M models, and 0.28–0.33× on 10 × 256.

## Equal positions in flight

The comparison that matters for self-play: the same number of positions in flight, as one batch or as four streams.

Run 2:

| Model | 4 × 64 ÷ 1 × 250 | 4 × 128 ÷ 1 × 512 | 4 × 250 ÷ 1 × 1024 |
|---|---|---|---|
| mini YkKk (0.24M) | 838.0 ÷ 531.2 = 1.58 | 1276.5 ÷ 644.1 = 1.98 | 1612.2 ÷ 686.1 = 2.35 |
| zlra shape (8.27M) | 137.0 ÷ 131.5 = 1.04 | 158.2* ÷ 133.5 = 1.19 | 157.3 ÷ 133.4 = 1.18 |
| v4_5block_7x7 preset (8.45M) | 128.1 ÷ 120.9* = 1.06 | 123.1 ÷ 132.9 = 0.93 | 151.4* ÷ 120.9 = 1.25 |
| 10 x 256 (12.9M) | 71.3 ÷ 73.1 = 0.98 | 82.2 ÷ 82.3* = 1.00 | 92.0 ÷ 79.8* = 1.15 |

Run 1 (batches 256 in place of 250):

| Model | 4 × 64 ÷ 1 × 256 | 4 × 128 ÷ 1 × 512 | 4 × 256 ÷ 1 × 1024 |
|---|---|---|---|
| mini YkKk (0.24M) | 847.1 ÷ 509.4 = 1.66 | 1196.3 ÷ 608.3 = 1.97 | 1612.0 ÷ 633.9 = 2.54 |
| zlra shape (8.27M) | 142.8* ÷ 145.2 = 0.98 | 155.3* ÷ 151.7 = 1.02 | 174.0 ÷ 155.0 = 1.12 |
| v4_5block_7x7 preset (8.45M) | 110.7* ÷ 130.3* = 0.85 | 139.0* ÷ 124.3 = 1.12 | 139.0* ÷ 132.6* = 1.05 |
| 10 x 256 (12.9M) | 66.9 ÷ 73.9 = 0.91 | 79.0 ÷ 81.1 = 0.97 | 85.9 ÷ 73.2 = 1.17 |

- **At today's game count (about 250 positions in flight), splitting gains nothing** on the 8M and 12.9M models: 0.98–1.06 in run 2, 0.85–0.98 in run 1.
- **At 1,000 positions in flight, four streams beat one batch by 5–25%** on the same models. Getting there means four times the games, with each game's moves four times further apart in wall time.
- **At 512, the two runs disagree** (0.93–1.19 in run 2, 0.97–1.12 in run 1). That is inside the run-to-run variation the README records (up to 1.3×).
- **The 0.24M model gains 1.58–2.54× at every size.** Its forward pass is mostly fixed cost per call (0.7 ms at batch 1), and the streams hide it.

## Real self-play against the benchmark

Source: `~/Library/Logs/DrewsChessMachine/dcm_log_20261010-224411.txt`.
- **Run:** build 2501, M5 Max with 64 GB, GUI Play-and-Train resumed from session `20261008-7-peGM`.
- **Champion:** `20261008-4-Ew5t`, the same zlra-shape model as the benchmark's.
- **Settings:** 250 self-play games (`workers=250`), trainer batch 4096, SPRT arenas.

| | Positions/hour (millions) | Time per 250 positions |
|---|---|---|
| Self-play, median of 300 `[STATS]` intervals (p10–p90: 10.4–13.1) | 11.9 | 75.7 ms per tick |
| Benchmark, `mps-bfloat16`, batch 250, in flight 1 | 131.5 | 6.67 ms per call |
| Benchmark, `mps-bfloat16`, batch 250, in flight 4 | 157.3 | 22.55 ms per call (4 streams) |

- **The forward pass, at its isolated speed, is under 9% of a self-play tick.** If the GPU call were the bottleneck, self-play would run near 131M positions/hour; it runs at 9% of that.
- **Where the other ~69 ms go is not measured.** No log line breaks a tick into its phases (encode, `evaluateBatched`, sample and apply, game-end pass, throttle). Candidates:
  - **The trainer's GPU work.** The trainer ran about 1,400 steps/hour at batch 4096 beside self-play. Over the run, the medians of its `[ENCODE-COST]` / `[LEGAL-COST]` lines are 1.28 s of GPU wait per step against 2.55 s between steps (two sample lines' p50s): the trainer kept the GPU busy about half the time. A self-play call submitted behind a training step waits for it.
  - **CPU phases.** These are encoding 250 boards, legal-move masks and sampling over 4,864 logits for 250 games, and the game-end pass. They run on 18 parallel tasks (`P=18`) while the GPU sits idle for self-play.
  - **Arenas** ran in some intervals and add their own GPU calls and CPU work.
  - **Other processes** on the machine that day are not recorded in the log.
- **What follows for overlap:**
  - **If the trainer dominates,** more self-play streams mostly buy self-play a bigger share of the GPU, at the trainer's expense. That changes the replay ratio, and `ReplayRatioController` would then throttle one side or the other.
  - **If CPU phases dominate,** two pipelined groups (one group's CPU work during the other's GPU call) could approach the CPU-bound rate. The benchmark doesn't measure that case at all.
  - **If the GPU call itself dominates** (unlikely at 9%), the equal-positions table above applies: nothing at 250 games, 5–25% at 1,000.

## Caveats

- One machine (M5 Max, 64 GB, macOS 27.2), two runs. Gaps under about 1.3× may not repeat (README, "Run 2 against run 1").
- **The benchmark overlapped four networks.** Self-play overlap would use either several copies of the champion (each copy is the weights again: about 16 MB for an 8M-parameter bf16 model) or one network that accepts concurrent calls, which `ChessNetwork` does not today.
- **The self-play figure comes from one run on an older build (2501), with arenas included.** It shows the size of the gap, not its cause.

## Reproduce

From `experiments/20261022-coreml-benchmark/`:
```sh
python3 overlap.py                # run 2 tables (results-run2/)
python3 overlap.py results        # run 1 tables
```
Self-play rate: the median over consecutive `[STATS]` lines of Δ`spMoves` ÷ Δ`elapsed`, keeping intervals of 60–400 s with `spMoves` increasing. The log file is on the owner's Mac and not in the repository.
