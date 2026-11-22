# 2026-11-22 — Where a corpus-replay trainer's memory goes

Owner request: explain why each label-smoothing arm (6×3×3@64, 694,767 parameters, batch 4096, replay
buffer 1,000,000 positions, frozen build 2687) has a 21.7 GB footprint. Measured on arm A (pid 57064) while it
trained, with the owner's OK to suspend it: `footprint -w/-v`, `heap`, `leaks --outputGraph`, and LLDB attached
for ~30 s to read every Metal heap and every buffer of 1 MiB or more (`mtl_probe.py`). The arm kept training.

## Totals (arm A at trainer step ~8,000)

| part | size | notes |
|---|---|---|
| GPU (IOAccelerator) | 13 GB | 4.5 GB of it swapped/compressed (idle), 4.45 GB wired |
| Malloc Large | 6.2 GB | the replay buffer's board store |
| Malloc Small | 2.1 GB | 1.4 GB of it the corpus feeder's decoded shard |

## GPU: three ~4 GiB MPSGraph heaps

`mtl_objects.txt`: 10 `MTLHeap`s, none labeled (MPSGraph does not label its heaps or the buffers it places
in them; the graph's tensor names never reach Metal):

| heap | size | live buffers ≥1 MiB in it |
|---|---|---|
| 0x74f59b9980 | 4.07 GiB | 59 bufs, 2.24 GiB |
| 0x74b4637300 | 4.00 GiB | 84 bufs, 3.10 GiB |
| 0x74b4634c00 | 3.92 GiB | 56 bufs, 2.10 GiB |
| 0x74b40f36c0 | 0.47 GiB | 4 bufs, 0.27 GiB |
| 0x74b4606040 | 0.20 GiB | 2 bufs |
| 4 empty, 1 of 2 MiB | | |

Plus buffers outside any heap: 2 × 76 MiB and 2 × 25 MiB (the batch feeds: 4096 × 4,864 fp32 legal masks and
4096 × 24 × 8 × 8 fp32 boards).

Buffer sizes identify the tensors:
- **32 MiB = [4096 × 64 × 8 × 8] bf16** — trunk activations; 47–61 per heap (≈1.5–1.9 GiB). Six blocks of two
  convs with BN, activation and LayerNorm, kept for the backward pass, at batch 4096.
- **76 MiB = [4096 × 4,864] fp32** — policy-head / loss tensors; 6–12 per heap (≈0.45–0.9 GiB).
- **64 MiB = [4096 × 64 × 8 × 8] fp32** — 3–4 per heap.

**Inference (not yet confirmed by labeled logging):** the three ~4 GiB heaps are the three compiled training-graph
variants this run uses — `TrainingExecutableKey` (batch, includeDiagnostics, KL-probe stash): plain steps,
diagnostic steps (every 10) and diagnostic + KL-stash steps (every 100, always also a diagnostic step). The heap
sizes rise in that order (3.92 / 4.00 / 4.07 GiB), and one heap's pages are mostly swapped out, matching a variant
used rarely. Each variant keeps its own heap for its intermediates, so the training graph's working memory is
paid three times. The smaller heaps are presumably the baseline forward pass and the test-set evaluator.

Correction to an earlier remark: GPU memory does depend on the network — on activation size (blocks × channels ×
batch), not on parameter count. The 8.27M-parameter v6 net (3 blocks @128, 9×9) has fewer, larger activations
and used 15.5 GB.

## CPU

`heap_armA_top.txt`:
- `ReplayBuffer.boardStorage` 6,144,000,000 bytes = 1,000,000 × 24 planes × 64 × 4 (fp32 boards; a
  deliberate layout so the trainer copies batches with no conversion).
- 368,840 `[ChessMove]` arrays, 1.36 GB: one corpus shard's games decoded at once. `leaks --traceTree` shows them
  held by a `[GameRecord]` array (11.9 MB) captured by the `CorpusReplayFeeder.feed(_:)` task. `ChessMove` is
  four `Int`s and an optional promotion (stride ≈ 40 bytes), ~92 moves per game on average.
- `ChessTrainer.replayBatchLegalMasks` 80 MB, `replayBatchBoards` 26 MB, the per-position stats dictionary 51 MB.

## Ways to cut it (each changes code; owner decision)

1. One training executable whose diagnostics and KL stash are switched by feeds instead of separate compiled
   variants: ≈ 8 GiB less GPU memory per run.
2. Stream the corpus shard, or store its moves compactly (e.g. 2-byte move codes): ≈ 1.3 GB less per run.
   Owner 2026-11-22: stream it, or memory-map it with a sequential-access hint, so it needs no fixed allocation.
3. Smaller `replay_buffer_capacity` for small-model tests (500k saves ≈ 3 GB; owner 2026-11-22 chose 200k for
   the label-smoothing arms from step 10,000); bit-packed boards would save
   ~5.9 GB but adds an unpack to every batch (the fp32 layout was chosen for speed).
