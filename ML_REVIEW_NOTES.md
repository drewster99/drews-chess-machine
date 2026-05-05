# ML Expert Review Notes

This document records the completed ML expert review findings for `drews-chess-machine`, organized by priority.

## Critical Issues

- No clear critical correctness issue found that would guarantee training failure, data corruption, or invalid labels. The core board encoding / policy-index wiring appears deliberately designed and tested, legal-move sampling is masked on the inference side, and the replay buffer writes outcomes only at game end.

## High Priority

### 1. Optimizer / `ChessTrainer.swift`, `TrainingParameters.swift`

- The exposed training parameters and documentation are Adam-oriented, but the actual optimizer is plain SGD with global gradient clipping and L2 weight decay. `TrainingParameters.swift:207-224` describes "Adam optimizer learning rate" and "standard practice for Adam" sqrt-batch scaling, while `ChessTrainer.swift:941-959` and `ChessTrainer.swift:2312-2361` implement `v_new = v - lr * (clipped_grad + weightDecayC * v)` with no Adam moments or momentum state.
- Hyperparameters tuned under an Adam mental model may be badly calibrated for plain SGD. LR scale, warmup length, weight decay strength, and batch-size scaling all have different behavior under SGD, which can cause slow convergence, excessive reliance on clipping, or undertraining.

### 2. Policy loss design / `ChessTrainer.swift`

- Policy CE and entropy are computed on `maskedLogits`, where illegal cells receive `-1e9` before softmax (`ChessTrainer.swift:1560-1565`, `1592-1598`, `1792-1796`). This makes the policy objective optimize only among legal moves for each sampled position.
- This is safe for selecting legal actions, but it gives almost no direct gradient pressure to lower raw illegal logits. If rising "legal mass" / top-1-legal behavior is an intended learned property, this objective may not teach it except indirectly through shared features and weight decay. The model may remain dependent on external legal masking at inference.

### 3. Training efficiency / `ChessTrainer.swift` — Policy readback

- The fresh-baseline pass in `trainStep(replayBuffer:)` calls `network.evaluate(...)` and retrieves both policy and values, but only `freshValues` are used; `freshPolicy` is only referenced by a commented-out diagnostic block (`ChessTrainer.swift:2664-2672`, `2691`, `2764-2810`).
- At batch size 4096, the unused policy output is `4096 * 4864` floats, roughly 80 MB of readback/allocation per training step. This is a large avoidable cost if the baseline pass only needs the value head.

### 4. Training efficiency / `ChessTrainer.swift` — Legal mask allocation

- Each real-data training step constructs and feeds a dense legal mask of shape `[batch, 4864]`: zeroing/filling the mask after decoding each board and generating legal moves (`ChessTrainer.swift:2727-2749`), allocating/caching an MPSNDArray of the same shape (`ChessTrainer.swift:3372-3378`), and writing it into the feed each step (`ChessTrainer.swift:3270-3273`).
- With batch size 4096 this is another roughly 80 MB per step, plus CPU move generation for every sampled position. It may be necessary for masked CE, but it is a major bottleneck candidate. A sparse gathered-loss design or persisted compact legal-index representation would likely be much cheaper.

## Medium Priority

### 5. Data pipeline / `ReplayBuffer.swift`

- Replay minibatches are sampled uniformly with replacement from individual positions (`ReplayBuffer.swift:483-525`). There is no prioritization, no game-level balancing, no phase/outcome balancing, and no recency weighting beyond FIFO buffer eviction.
- Long games and common repeated positions can dominate. The same position can appear multiple times in one batch, and rare decisive/late-game positions can be underrepresented. This increases gradient variance and may slow learning in a draw-heavy self-play setting.

### 6. Reward/value shaping / `ChessTrainer.swift`, `parameters.json`

- Draws are rewritten from `z = 0` to `-drawPenalty` when the knob is positive (`ChessTrainer.swift:1041-1070`, `2712-2725`); current `parameters.json:9` sets `draw_penalty` to `0.1`.
- This may be useful as bootstrap shaping, but it changes the value target for all draw types and can bias a draw-heavy domain. The code comment itself notes `drawPenalty=0` is a reasonable steady-state default, so this should be treated as a tuning choice rather than neutral labeling.

### 7. Loss function / `ChessTrainer.swift`

- Policy advantage is standardized per batch: `(A - mean(A)) / sqrt(var(A)+eps)` before multiplying CE (`ChessTrainer.swift:1621-1687`).
- This stabilizes scale, but removes absolute advantage magnitude and can distort or nearly erase policy signal in homogeneous batches. Because replay sampling is uniform with replacement, batch composition can strongly affect the sign/scale of policy updates.

### 8. Batch processing robustness / `ChessTrainer.swift`

- `trainStep(replayBuffer:)` has three phases: phase 1 and phase 3 are serialized via `enqueue`, but the fresh-baseline `await network.evaluate(...)` occurs between them while trainer-owned staging buffers remain shared object state (`ChessTrainer.swift:2585-2691`).
- The current app likely has a single training loop, so this may be fine in practice. But if two callers ever invoke replay training concurrently, phase interleaving could overwrite staging buffers between sample and train. A precondition/single-flight guard would make the invariant explicit.

## Low Priority

### 9. Sampling strategy / `MPSChessPlayer.swift`, `parameters.json`

- Current self-play and arena schedules are highly stochastic: self-play starts at tau 2.0 and floors at 0.8 in `parameters.json:10-12`; arena starts at tau 2.0 and floors at 0.5 in `parameters.json:13-15`. The default self-play schedule also supports opening Dirichlet noise (`MPSChessPlayer.swift:26-52`, `56-152`).
- This improves coverage and diversity, but it also injects noise into training labels and arena comparisons. For promotion decisions, high arena temperature can make strength estimates noisier than a lower-temperature or paired-seed evaluation.

### 10. Observability overhead / `ChessTrainer.swift`

- Many diagnostic target tensors are fetched every training step, including a batch-sized `advantageRaw` tensor that is read back into a freshly allocated `[Float]` (`ChessTrainer.swift:3436-3449`, `3592-3605`).
- The diagnostics are useful, but if throughput is limiting, some could be sampled every N steps rather than every step to reduce readback/allocation overhead.
