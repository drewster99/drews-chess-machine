# GPU call audit — 2026-10-09

Phase 1 (A1 inventory) of `GPU_FAULT_FORENSICS_PLAN.md`. Read-only: no app
code changed, nothing built, no test run, no running process touched.

- Scope: production code under `DrewsChessMachine/DrewsChessMachine/` (paths
  below are relative to it). Tests excluded.
- Code state: `main` at `7292c0eb`.
- SDK headers: Xcode 27.2 beta 2 and Xcode 27.1 beta (`MPSGraph.h` and
  `MPSGraphExecutable.h` are byte-identical in both).
- Runtime facts marked **[probe]** come from small standalone Swift programs
  run in the session scratchpad (64×64 matmul chains, empty command buffers;
  no app code). Their outputs are quoted in "Probe results" at the end.

## Key facts that change the picture

1. **MPSGraph splits large work across several command buffers.**
   `executable.encode(to:)` and `graph.encode(to:)` call `commitAndContinue`
   on the `MPSCommandBuffer`. **[probe]** A 100-op graph made 1 continuation,
   400 ops 4, 8,000 ops 10. With 400 ops the caller's original buffer was
   already `.completed` before `encode` returned. A training step
   (forward + backward + optimizer, 8.3M params) is far above those sizes. Its
   segment count is unmeasured, but it is almost certainly several.
2. **After a continuation, `MPSCommandBuffer.commandBuffer` and
   `.rootCommandBuffer` both return the newest buffer.** **[probe]** This
   contradicts the header comment on `commandBuffer` ("The Metal Command
   Buffer that was used to initialize this object"). The original buffer is
   reachable only through the caller's own reference. Middle segments are not
   reachable through any public property.
3. **Continued buffers do not inherit `MTLCommandBufferDescriptor.errorOptions`.**
   **[probe]** The original had `errorOptions=1`
   (`.encoderExecutionStatus`); every MPS-created continuation had `0`. The
   plan's A4 proposal (set `.encoderExecutionStatus`) would cover only the
   first segment.
4. **No app code sets `MTLCommandBufferDescriptor`, `errorOptions`, an
   `MPSGraphExecutionDescriptor` or an `MPSGraphExecutableExecutionDescriptor`.**
   Every `run` / `encode` passes `executionDescriptor: nil`.
5. **No app code calls** `makeBuffer`, `makeHeap`, `makeTexture`, `newBuffer`,
   `MTLBuffer.contents()`, `MTLCopyAllDevices`, `runAsync`, MPS kernels
   (`MPSMatrix*`, `MPSImage*`, `MPSCNN*`), `makeComputePipelineState`, or
   compute / blit encoders. All GPU work is MPSGraph.

## Inventory

Columns as requested. "Segment" = one command buffer of an MPS
`commitAndContinue` chain. "Original" = the buffer the app created, which is
the first segment. "Result-key guard" means `results[x] != nil`: a CPU-side
dictionary check, not a GPU status. Whether MPSGraph ever omits a key on a
GPU fault is unverified (the comments at `ChessTrainer.swift:5893-5899`
assume it does).

### A. Device, queue, command buffers, own submissions

| # | file:line | function | call | GPU work | caller path(s) | sync/async | failure signal | checked? | when checked | logged? | consumer if silent failure | risk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Network/ChessNetwork.swift:587 | `ChessNetwork.init` | `MTLCreateSystemDefaultDevice()` | none | init of every network (champion, trainer, arena candidate / champion, probe networks, UCI, Lichess, CLI, audits) | sync | nil | yes → `metalNotSupported` | at creation | not at the site; depends on the caller | n/a | low |
| 2 | Network/ChessNetwork.swift:590 | `ChessNetwork.init` | `makeCommandQueue()` (one queue per network) | none | same as #1 | sync | nil | yes → `commandQueueCreationFailed` | at creation | not at the site; depends on the caller | n/a | low |
| 3 | Training/BehaviorFingerprint.swift:239 | behavior fingerprint | `MTLCreateSystemDefaultDevice()` | none | resume fingerprint (GUI, corpus replay, train-vs-UCI start) | sync | nil | yes → `FingerprintError.noMetalDevice` | at creation | depends on the caller | n/a | low |
| 4 | Training/BehaviorFingerprint.swift:240 | behavior fingerprint | `makeCommandQueue()` | none | same as #3 | sync | nil | yes → `noCommandQueue` | at creation | depends on the caller | n/a | low |
| 5 | Network/ChessNetwork.swift:1700 | `internalComputeValueBaseline` | `makeCommandBuffer()` | — | value baseline | sync | nil | yes → `outputMissing("value-baseline command buffer")` (misnamed error) | at creation | via the caller's catch (§4) | n/a | low |
| 6 | Network/ChessNetwork.swift:1703 | `internalComputeValueBaseline` | `MPSCommandBuffer(commandBuffer:)` | — | value baseline | sync | none (nonnull) | n/a | — | — | — | low |
| 7 | Network/ChessNetwork.swift:1704 (+1710 `commit`, 1716 `waitUntilCompleted` only when `blockingValueBaseline`, 1719 retain, 1662 status read) | `internalComputeValueBaseline` | `executable.encode(to:inputs:results:[resultTD]:executionDescriptor:nil)` | value-only forward, batch 4096, into the reused `resultTD` (`[count,1]` fp32) | **every training step** with real data: GUI Play-and-Train / `--train`, corpus replay, train-vs-UCI, behavior fingerprint (`ChessTrainer.swift:4979`) | **async**: committed, never waited (`blockingValueBaseline` defaults false at :570; no app code sets it) | status / error on every segment | **partial**: original segment only, and only `== .error` (`:1662`) | at the **next** baseline call, after the training step that read it has run and updated the weights; the run's last baseline is never checked | thrown `ChessNetworkError.gpuCommandFailed("value baseline")`; logged by the path's catch (§4) | training step's `vBaseline` (`ChessTrainer.swift:7127-7128`) → advantages → policy gradient → weights | **high** |
| 8 | Network/ChessNetwork.swift:1973 | `internalLoadWeights` | `makeCommandBuffer()` | — | weight load (see #10) | sync | nil | yes → `commandBufferCreationFailed` | at creation | via the caller | n/a | low |
| 9 | Network/ChessNetwork.swift:1976 | `internalLoadWeights` | `MPSCommandBuffer(commandBuffer:)` | — | weight load | sync | none | n/a | — | — | — | low |
| 10 | Network/ChessNetwork.swift:1978 (+1985 `commit`, 1986 wait on current, 1990 wait on original, 1991-1995 `requireCompleted` ×2) | `internalLoadWeights` | `graph.encode(to:feeds:targetTensors:targetOperations:)` | variable assigns for all weights + BN stats | arena candidate / champion sync (`SessionController+Arena.swift:215, 255`), promotion (`:482`), Promote Trainee Now (`SessionController+ManualPromote.swift:159`), resume (`SessionController+Checkpoint.swift:864, 1069`; `TrainerResumeState.swift:418`), trainer load/fork (`ChessTrainer.swift:5747, 5765`), probe-network refresh (candidate, tactical, Lichess, legal mass), UCI model load, train-vs-UCI eval sync, CLI start models, test-set evaluator, save verification (`CheckpointManager.swift:2161, 2173`), network-build BN warmup | sync | status / error per segment | **partial**: original and last segment (`.commandBuffer` = last, **[probe]**); middle segments unchecked; `!= .completed` | after the wait, before the load gate clears | thrown `ChessNetworkError.gpuCommandFailed("weight load")`; logged depends on the caller (§4) | network weights for every later use | med |
| 11 | Training/ChessTrainer.swift:7145 | `runPreparedStep` | `makeCommandBuffer()` | — | training step | sync | nil | yes → `ChessTrainerError.lossOutputMissing` (misnamed error) | at creation | via the path's catch | n/a | low |
| 12 | Training/ChessTrainer.swift:7148 | `runPreparedStep` | `MPSCommandBuffer(commandBuffer:)` | — | training step | sync | none | n/a | — | — | — | low |
| 13 | Training/ChessTrainer.swift:7155 (+7177 `commit`, 7178 wait on current, 7179 status read, 7186 check) | `runPreparedStep` | `executable.encode(to:inputs:results:nil:executionDescriptor:nil)` | forward + backward + optimizer assigns (weights, velocity, fp32 masters, BN stats, dropout-RNG advance), loss / norm / diagnostic reductions; batch 4096 | every training step: GUI, corpus replay, train-vs-UCI, sweeps (`SessionController.swift:1049`, `ArchSweepCLI.swift:127`), fingerprint | sync (commit + wait) | status / error per segment | **partial**: original (first) segment only; `== .error` only | after the wait, before readback | thrown `ChessTrainerError.gpuCommandFailed("training step")` (§4) | loss / grad-norm / diagnostics readback (`:7272-7368`) → health monitor, `[GRAD-CLIP]`, gradient-norm history; the in-place weight / velocity / master updates | **high** |
| 14 | Training/ChessTrainer.swift:7215 | `runPreparedStep` | `makeCommandBuffer()` | — | working-weight sync | sync | nil | yes → `lossOutputMissing` (misnamed) | at creation | via the catch | n/a | low |
| 15 | Training/ChessTrainer.swift:7218 | `runPreparedStep` | `MPSCommandBuffer(commandBuffer:)` | — | working-weight sync | sync | none | n/a | — | — | — | low |
| 16 | Training/ChessTrainer.swift:7220 (+7227 `commit`, 7228 wait, 7229 status read, 7231 check) | `runPreparedStep` | `graph.encode(to:feeds:[:]:targetTensors:targetOperations:workingSyncOps)` | `working = cast(master)` per trainable (bf16 / fp16 only, `splitWorkingWeightSync`) | every training step when split sync is on | sync | status / error per segment | **partial**: original segment only; `== .error` | after the wait | thrown `gpuCommandFailed("working-weight sync")` | bf16 working weights for the next step, exports and probes | high |

### B. Synchronous `executable.run` / `graph.run` (no status available)

| # | file:line | function | call | GPU work | caller path(s) | sync/async | failure signal | checked? | when checked | logged? | consumer if silent failure | risk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17 | Network/ChessNetwork.swift:1490 | `internalEvaluate(batchBoards:)` | `executable.run(with:inputs:results:nil:executionDescriptor:nil)` | batched inference: policy logits, value, W/D/L | **self-play** (`BatchedSelfPlayDriver.swift:460`, K=250 in R-fixedlr), **arena** (`TickTournamentDriver.swift:557, 581`), train-vs-UCI trainer moves (`TrainVsUciDriver.swift:337`), tactical / Lichess / model-test-set batteries (`SessionController+TacticalProbe.swift:433`, one call of 4,635 positions), legal-mass probe (`ChessTrainer.swift:5423, 5430`) | sync | none (an execution descriptor `completionHandler` with an `NSError` could be passed; not used) | **no** (key guards only) | — | none for GPU faults; a *thrown* error logs `[SP-TICK] network error` / `[VS-UCI] trainer network error`, or aborts the arena | moves sampled from garbage logits; draw-watch reads garbage pDraw (can terminate games); arena results feed SPRT → promotion; probe results / the pElo written into model files; legal-mass alarms | **high** (self-play, arena) |
| 18 | Network/ChessNetwork.swift:1210 | `internalEvaluateCore` | `graph.run(with:feeds:targetTensors:targetOperations:)` | single-position inference | UCI (`MoveEvaluationSource.swift:146`), Lichess bot (`LichessBotMoveChooser.swift:91` via `evaluateWithValueDistribution`), Play Game / Human-vs-Network (`ChessRunner.swift:51`), candidate probe (`SessionController+CandidateProbe.swift:199`), tactical single probes (`SessionController+TacticalProbe.swift:346, 349`), Run Forward Pass (`SessionController+Diagnostics.swift:147, 230, 240`), replay-buffer analyzer (`ReplayBufferAnalyzer.swift:860`), save verification (`CheckpointManager.swift:2162, 2174`) | sync | none | no (key guards) | — | none | move played (Lichess games are real); probe readouts; save verification compares two forwards | med |
| 19 | Network/ChessNetwork.swift:1322 | `internalEvaluateValueDistribution` | `graph.run` | single-position value head | candidate probe (`:206`), tactical (`:371`), `ChessRunner.swift:72` | sync | none | no | — | none | diagnostic W/D/L readout | low |
| 20 | Network/ChessNetwork.swift:1889 | `internalExportWeights` | `graph.run` (variables as targets) | read every weight + BN stat | arena champion snapshot (`SessionController+Arena.swift:254`), arena candidate (`TrainerResumeState.swift:392`), promotion (`SessionController+Arena.swift:481`), session saves (`SessionController+Checkpoint.swift:74, 417`), trainer fork from champion (`SessionController+Training.swift:974, 990`), trainer exports for saves (`ChessTrainer.swift:4537, 4586, 5716` via `exportWeightsBlocking`), probe snapshots (`SessionController+CandidateProbe.swift:55`, `TacticalProbeWatcher.swift:234`, `LichessProbeWatcher.swift:235`, `ChessTrainer.swift:5421`), train-vs-UCI eval sync (`TrainVsUciRunner.swift:497, 611`), UCI live trainer (`MoveEvaluationSource.swift:215`), Lichess model provider, `--new-model`, graft, analysis snapshot | sync | none | no (key guards) | — | none | **weights written to disk**, promoted, or copied to the arena networks | **high** |
| 21 | Network/ChessNetwork.swift:2066 | `internalComputeBatchStats` | `graph.run` | forward pass, BN batch mean / var | network-build BN warmup (`ChessMPSNetwork.swift:247`) | sync | none | no | — | none | BN running stats of a new inference network (#23) | med |
| 22 | Network/ChessNetwork.swift:2131 | `internalEvaluateAnalysisTaps` | `graph.run` | forward pass + analysis taps | numerics audit (`NumericsAudit+Dynamic.swift:171`; GUI and CLI) | sync | none | no | — | none | audit verdicts | low |
| 23 | Network/ChessNetwork.swift:2230 | `internalLoadBNRunningStats` | `graph.run` (assign ops) | write BN running stats | network-build BN warmup (`ChessMPSNetwork.swift:251`) | sync | none | **no** (unlike `loadWeights`, which checks) | — | none | inference BN stats for every later forward | med |
| 24 | Training/ChessTrainer.swift:5838 | `internalReadVelocityValues` | `graph.run` | read every velocity tensor | `exportTrainerWeights` (every trainer save: GUI sessions, corpus replay / train-vs-UCI checkpoints and session folders, Promote Trainee Now, analysis snapshot, fingerprint), arena-start capture (`TrainerResumeState.swift:393`) | sync | none | no (key guard) | — | none | **velocity written to disk**; restored by a promotion rewind | **high** |
| 25 | Training/ChessTrainer.swift:5900 | `writeVelocityValues` | `graph.run` (assign ops) | write velocity | resume (`loadTrainerWeights`), branch / fork (`loadBaseWeightsResetVelocity` → `resetVelocitiesToZero`), promotion rewind (`TrainerResumeState.swift:422`) | sync | none | **no** (key guard only, though the comment calls it a GPU signal) | — | none | optimizer state of every later step; an exact resume still reports EXACT | high |
| 26 | Training/ChessTrainer.swift:5940 | `internalReadMasterValues` | `graph.run` | read fp32 masters | `exportTrainerWeights` (all trainer saves, bf16 / fp16) | sync | none | no (key guard) | — | none | **masters written to disk** | **high** |
| 27 | Training/ChessTrainer.swift:5991 | `writeMasterValues` | `graph.run` (assign ops) | write fp32 masters | resume, branch / fork | sync | none | no (key guard) | — | none | masters → every later step | high |
| 28 | Training/ChessTrainer.swift:6075 | `readTrainableVelocity` | `graph.run` | read one velocity tensor | value-FC1 health read (`TrainingHealthReads.swift:61`) | sync | none | no (key guard) | — | none | `value_fc1_zero_velocity` alarm (an action can halt a run) | med |
| 29 | Training/ChessTrainer.swift:6120 | `internalReadLayerHealthLiveState` | `graph.run` | read BN γ/β/stats, ReZero α | `[LAYER-HEALTH] live` and the health monitor every 50 steps (`LayerHealthLog.swift:46`) | sync | none | no (key guard) | — | none | health alarms (`non_finite`, `dead_channels`, …) | med |
| 30 | Training/ChessTrainer.swift:6153 | `syncMastersFromWorking` | `graph.run` (assign ops) | master = cast(working) | promotion rewind (`TrainerResumeState.swift:419`) | sync | none | no (key guard) | — | none | masters → all later training | high |
| 31 | Training/ChessTrainer.swift:6180 | `runSyncMastersOnQueue` | `graph.run` (assign ops) | master = cast(working) | trainer init (`:2423`), `resetNetwork` (`:2651`) | sync | none | **no** (result discarded) | — | none | masters → all later training | high |
| 32 | Training/ChessTrainer.swift:6253 | `writeDropoutStateOnQueue` | `graph.run` (seed assign) | write Philox state | seed at init / reset / `beginDropoutStream`, restore on resume and promotion rewind | sync | none | **no** (result discarded) | — | none | dropout masks; RNG exactness claims in lineage | med |
| 33 | Training/ChessTrainer.swift:6274 | `readDropoutStateOnQueue` | `graph.run` | read Philox state | `captureDropoutState`: saves (`exportResumeSnapshot`), arena-start capture | sync | none | no (key guard) | — | none | Philox state persisted in the lineage record | med |
| 34 | Training/ChessTrainer.swift:7479 | `runPreparedStep` (KL probe) | `graph.run` | post-update forward, KL reductions | every `klProbeInterval` training step (100 in the 2026-10-09 runs) | sync | none | no (finite check on 2 scalars) | after readback | `[KL-PROBE] skipped` only for thrown / non-finite | KL telemetry | low |
| 35 | Training/ChessTrainer.swift:7549 | `runPreparedStep` | `graph.run` (dropout-RNG advance) | Philox state advance | KL-probe steps | sync | none | **no** (result discarded) | — | none | next step's dropout mask | med |
| 36 | Training/DropoutPhiloxState.swift:53 | `DropoutPhiloxState.derived` | `graph.run` (`randomPhiloxStateTensor`) | seed → Philox state | `runDropoutSeedOnQueue` (`ChessTrainer.swift:6234`), fingerprint (`BehaviorFingerprint.swift:241`) | sync | none | no (key guard → `missingResult`) | — | none | dropout seed state; fingerprint hash | med |

### C. Compilation

| # | file:line | function | call | GPU work | caller path(s) | sync/async | failure signal | checked? | when checked | logged? | consumer if silent failure | risk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 37 | Network/ChessNetwork.swift:1604 (descriptor 1588) | `inferenceExecutable` | `graph.compile(with:feeds:targetTensors:targetOperations:compilationDescriptor:)` | compile; cached per batch size, so arena / self-play compile once per distinct K | #17 | sync | none: nonnull return; `compilationCompletionHandler` (`MPSGraph.h:134`) not set | large-stack thread failure only → `preconditionFailure` (`:1613`) | — | crash | — | low |
| 38 | Network/ChessNetwork.swift:1756 (descriptor 1743) | `valueBaselineExecutable` | `graph.compile` | compile | #7 | sync | none | stack-thread failure → `preconditionFailure` (`:1765`) | — | crash | — | low |
| 39 | Training/ChessTrainer.swift:6986 (descriptor 6946) | `trainingExecutable` | `graph.compile` | compile; logs `[EXEC] compiled …` | #13 | sync | none | stack-thread failure throws | — | `[EXEC]` on success only | — | low |

### D. Allocations, CPU↔GPU copies, capture (no failure signal at any of these)

| # | file:line(s) | function | call | GPU work | caller path(s) | sync/async | failure signal | checked? | when | logged? | consumer if silent failure | risk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | Network/ChessNetwork.swift:754, 758, 1040, 1059, 1069, 1544, 1778 (7) | `init`, `batchInputEntry`, `valueBaselineResultTD` | `MPSNDArray(device:descriptor:)` | allocation (storage lazily backed) | network build; first use of each batch size | sync | Swift: none. The header (`MPSNDArray.h:287-289`) says "A valid MPSNDArray object or nil, if failure" but declares the result `nonnull` | no | — | none | feeds and results of #7, #10, #17-#23 | low (out of memory would surface elsewhere first; unverified) |
| 41 | Training/ChessTrainer.swift:2354, 2358, 2362, 2366, 2370, 2374, 2378, 2382, 2386, 2390, 2394, 2398, 2402, 2406 (init, 14); 2580, 2583, 2586, 2589, 2592, 2595, 2598, 2601, 2604, 2607, 2610, 2613, 2616, 2619 (`internalResetNetwork`, 14); 4277, 4324 (`buildTrainingOps`, 2); 6636, 6649, 6662, 6671, 6679 (`feedsForBatch`, 5) | as listed | `MPSNDArray(device:descriptor:)` (35) | allocation | trainer build / reset / first step per batch size | sync | none (same as #40) | no | — | none | training feeds, velocity / master loads | low |
| 42 | Network/ChessNetwork.swift:757, 762, 1042, 1062, 1075, 1546, 1780 (7); Training/ChessTrainer.swift:2357, 2361, 2365, 2369, 2373, 2377, 2381, 2385, 2389, 2393, 2397, 2401, 2405, 2409, 2582, 2585, 2588, 2591, 2594, 2597, 2600, 2603, 2606, 2609, 2612, 2615, 2618, 2621, 4279, 4326, 6643, 6651, 6664, 6673, 6681 (35) | as #40 / #41 | `MPSGraphTensorData(MPSNDArray)` (42) | wrapper | as #40 / #41 | sync | none (`instancetype`, nonnull) | no | — | none | — | low |
| 43 | Training/DropoutPhiloxState.swift:83 | `tensorData(device:)` | `MPSGraphTensorData(device:data:shape:dataType:)` | allocation + copy from `Data` | #32 | sync | none | no | — | none | Philox state feed | low |
| 44 | Network/ChessNetwork.swift:756, 760, 3534, 3563, 3622 (5); Training/ChessTrainer.swift:2664, 6220, 6472, 6556, 6564, 6582, 6599, 6602, 6605 (9) | `init`, `writeInferenceInput`, `writeFloats`, `writeFloatsFP32`, `internalResetNetwork`, `pushDropoutRateToGraph`, `buildFeeds`, `writeRealValuedFeed`, `writeScalarFeed` | `MPSNDArray.writeBytes` (14) | CPU → GPU copy | every feed: inference inputs, weight / velocity / master loads, training feeds, scalar hyperparameters, dropout rate | sync | none (`void`) | no | — | none | GPU inputs | low |
| 45 | Network/ChessNetwork.swift:3578, 3607, 3768, 3780, 3801, 3848, 3858, 3886 (8); Training/DropoutPhiloxState.swift:74 (1) | `readFloatsFP32` ×2, `readFloats` ×2, `DropoutPhiloxState.init(reading:)` | `MPSGraphTensorData.mpsndarray().readBytes` (9) | GPU → CPU copy | every readback in #13 and #17-#36 | sync | none (`void`); copies whatever the buffer holds | no (dtype / count preconditions only, `:3598-3606`) | — | none | every consumer listed in sections A-B | n/a: inherits the risk of the run that produced the data |
| 46 | CLI/CorpusReplayRunner.swift:931-943 | `beginGPUCapture` | `MTLCaptureManager.startCapture(with:)` | GPU trace | `--capture-gpu-step` | sync | throws | yes → `gpuCaptureFailed` | before the step | `[ALARM] [REPLAY] GPU trace capture … could not start` | — | low |
| 47 | CLI/CorpusReplayRunner.swift:2128, 2133 | replay loop | `MTLCaptureManager.stopCapture()` | — | same | sync | none | no | — | — | trace file | low |

Not counted as failure sites: `device.currentAllocatedSize` /
`recommendedMaxWorkingSetSize` / `maxBufferLength` reads
(`ChessTrainer.swift:7676, 8032-8037`, `SessionController+Sweep.swift:157`),
and `graph.variable(with:)` allocations at build time (no failure signal).

## 1. Summary counts

| Category | Sites | Checked fully | Partial | Unchecked | No signal available at the call |
|---|---|---|---|---|---|
| `MTLCreateSystemDefaultDevice` / `makeCommandQueue` | 4 | 4 | – | – | – |
| `makeCommandBuffer` | 4 | 4 | – | – | – |
| `MPSCommandBuffer(commandBuffer:)` | 4 | – | – | – | 4 |
| Own-buffer submissions (`encode` + `commit` + status) | 4 | 0 | 4 (#7, #10, #13, #16) | – | – |
| ↳ `commit` / `waitUntilCompleted` / status reads within them | 4 / 5 / 5 | – | – | – | (these are the checks) |
| `executable.run` | 1 | 0 | – | 1 | – |
| `graph.run` | 19 | 0 | – | 19 (14 key guard only, 4 result discarded, 1 key guard + finite check) | – |
| `graph.compile` (+3 descriptors) | 3 | – | – | – | 3 |
| `MPSNDArray` init | 42 | – | – | – | 42 |
| `MPSGraphTensorData` init | 43 | – | – | – | 43 |
| `writeBytes` / `readBytes` | 14 / 9 | – | – | – | 23 |
| `MTLCaptureManager` start / stop | 1 / 2 | 1 | – | – | 2 |
| **Total** | **164** | **9** | **4** | **20** | **117** (+14 commit / wait / status inside the 4 partial submissions) |

**GPU work submissions: 24.** None is fully checked; 4 are partly checked;
20 have no check at all, because synchronous `run` exposes no status (§3).

Logged with the Metal error at the failure site: **0**. Every check throws;
logging happens only in the path's catch, as `localizedDescription` (§4).

## 2. Gaps, worst first

1. **Training step checks only its first segment** —
   `Training/ChessTrainer.swift:7179-7186`.
   - The step reads `mtlCommandBuffer.status`: the buffer passed to
     `MPSCommandBuffer(commandBuffer:)` at `:7148`, which is the first segment.
   - `commit` / `waitUntilCompleted` at `:7177-7178` act on the *current*
     segment (**[probe]**: `mps.status` and `.rootCommandBuffer` follow the
     newest buffer).
   - MPS commits the original early, during `encode` at `:7155`
     (**[probe]**: original already `.completed` when encode returned, for a
     400-op graph). So the check covers only the first chunk of the forward
     pass.
   - A discarded later segment (backward, optimizer assigns, loss
     reductions) passes the check. `:7272-7296` then reads loss /
     `gradGlobalNorm` from garbage, and the in-place weight / velocity /
     master writes are partial.
   - B-siluall's 21:37:09.834 halt was the non-finite **readback** alarm
     (`:7410-7423`), not `gpuCommandFailed`. So that step's first segment
     reported `.completed`. This is consistent with a later segment (or the
     baseline, gap 2) being the discarded one. It is not proof.
   - Only `== .error` is tested, not `!= .completed`.
   - **Confirmed:** the training step checks only the root (first) buffer.
   - **The weight load (`ChessNetwork.swift:1990-1995`) checks two buffers:**
     the local `rootCommandBuffer` (the original) and
     `commandBuffer.commandBuffer`. At runtime that property is the *last*
     segment (**[probe]**), not the original as the header says. So the load
     checks the first and last segments and misses the middle ones
     (**[probe]**: 4-10 continuations for 400-8,000-op graphs).
   - **Can MPS `commitAndContinue` into a different buffer for
     `executable.encode(to: MPSCommandBuffer)`?** Yes:
     - Header, `MPSGraphExecutable.h:224`: "commitAndContinue might be
       called, please don't rely on underlying MTLCommandBuffer to remain
       uncommitted".
     - **[probe]**: `executable.encode` continued 1× at about 100 ops and 4×
       at about 400 ops.

2. **Value baseline: committed without a wait, checked one call late, root
   only** — `Network/ChessNetwork.swift:1704-1720`, check at `:1662`.
   - The baseline for step N is encoded and committed (`:1710`) with no wait.
     The training step for N binds `resultTD` as its `vBaseline`
     (`ChessTrainer.swift:7127-7128`) and runs to completion, updating the
     weights.
   - The baseline buffer's status is read only when step N+1 calls
     `computeValueBaselineGPU` (`:1662`). Even then it reads only the
     original segment, only `== .error`, and only if that segment was the
     victim. The last baseline of a run is never checked.
   - What a discarded baseline leaves in `resultTD`: either the previous
     step's v(s) (a different batch) or a partial overwrite. Both are values
     in [−1, 1], so the advantages are wrong but bounded.
   - Whether that alone can produce a 10.6M gradient norm or NaN is
     unverified. The policy loss normalizes advantages by RMS (CLAUDE.md
     `pLoss`), which limits the scale; gap 1 is the stronger suspect for the
     blow-ups.
   - `blockingValueBaseline` (`:570`) is never set by app code.

3. **Weight and optimizer-state reads for saves, promotion and the arena have
   no check** — `ChessNetwork.swift:1889`, `ChessTrainer.swift:5838, 5940,
   6274`.
   - A discarded `graph.run` returns result buffers holding whatever was
     there. The data is written to disk or promoted.
   - Save verification (`CheckpointManager.swift:2120-2180`) loads the
     exported arrays and the file read-back into a scratch network and
     compares two forwards. Both come from the same bad export, so it
     passes.
   - Test-set results are computed from those weights, so a garbage pElo
     would be recorded in the file. That is the only (weak) trace.

4. **Optimizer-state writes on resume, branch and promotion rewind have no
   check** — `ChessTrainer.swift:5900, 5991, 6153, 6180`.
   - Their `results[x] != nil` guard is not a GPU status (the comment at
     `:5893-5899` assumes it is; unverified).
   - `rewindToArenaStart` (`TrainerResumeState.swift:414-424`) mixes the
     checked `loadWeights` with the unchecked `syncMastersFromWorking` and
     `loadVelocitySnapshot`.
   - An exact resume would log `[RESUME] EXACT` on top of bad velocity or
     masters.

5. **Self-play and arena inference have no check** —
   `ChessNetwork.swift:1490` (`executable.run`, sync, `executionDescriptor:
   nil`).
   - Callers: `BatchedSelfPlayDriver.swift:460` and
     `TickTournamentDriver.swift:557, 581`.
   - The only error path, `[SP-TICK] network error … skipping tick`
     (`BatchedSelfPlayDriver.swift:525-528`), catches *thrown* errors, which
     a GPU fault never produces.
   - Garbage logits are masked to legal moves and sampled. The moves are
     legal, but the games no longer come from the network's policy.
   - Draw-watch reads garbage pDraw and can terminate games
     (`drawWatchTerminate`).
   - Arena games decided this way are SPRT evidence for promotion.
   - Train-vs-UCI trainer moves (`TrainVsUciDriver.swift:337`) are the same.

6. **Dropout-RNG state ops have no check** — `ChessTrainer.swift:6253,
   6274, 7549`, `DropoutPhiloxState.swift:53`.
   - `:6253` and `:7549` discard the result entirely.
   - A bad state means wrong masks, and the RNG exactness recorded in
     lineage becomes false.

7. **Health-monitor inputs have no check** — `ChessTrainer.swift:6075,
   6120`. A garbage read can raise `non_finite` / `dead_channels` /
   `value_fc1_zero_velocity`. With a non-default action, that can halt a
   run on a GPU fault while the log names the wrong cause.

8. **BN warmup has no check** — `ChessNetwork.swift:2066, 2230`.
   `internalLoadBNRunningStats` uses `graph.run`, while `internalLoadWeights`
   (same kind of assign) uses the checked encode path.

9. **Error reporting is incomplete.** No site logs at the point of failure.
   - The catches log `error.localizedDescription` only. That string does
     carry the IOGPU text, for example
     `GPU command buffer failed during working-weight sync:
     status=MTLCommandBufferStatus(rawValue: 5), error=Internal Error
     (00000001:Internal Error)` (session log 2026-07-14 18:13:36).
   - `NSError.domain`, `code` and `userInfo` are not logged.
   - `MTLCommandBufferEncoderInfoErrorKey` is never available:
     `errorOptions` is never set, and continued segments would not inherit it
     anyway (**[probe]**).
   - Arena failure catches write no session-log line:
     `SessionController+Arena.swift:224-228` (candidate sync — this one *can*
     carry a weight-load `gpuCommandFailed`), `:257-261` (champion sync) and
     `:382-386` (tournament) call only `trainingBox?.recordError`.

10. **Single-position inference has no check** — `ChessNetwork.swift:1210`.
    Lichess games are real games. UCI and probes are lower stakes.

11. **Misnamed nil-command-buffer errors.**
    - `ChessTrainer.swift:7146, 7216` throw `lossOutputMissing`.
    - `ChessNetwork.swift:1701` throws `outputMissing("value-baseline command
      buffer")`.
    - Only `:1973` uses `commandBufferCreationFailed`.

12. **Compile has no error path.** `ChessNetwork.swift:1604, 1756`;
    `ChessTrainer.swift:6986`. No `compilationCompletionHandler` is set; an
    MPSGraph compile failure asserts inside the framework (unverified how).

13. **Allocation failure is invisible.** `MPSNDArray(device:descriptor:)` is
    imported nonnull although the header documents a nil return (#40-#41).

## 3. Can synchronous `run` report a GPU error?

Header facts (Xcode 27.2 beta 2; identical in 27.1 beta):

- `MPSGraph.h:249-277`: `run(feeds:…)`, `run(with:feeds:targetTensors:targetOperations:)`,
  `run(with:feeds:targetOperations:resultsDictionary:)`. "This call blocks
  until execution has completed." These take **no execution descriptor and
  return no error**: only "A valid MPSGraphTensor : MPSGraphTensorData
  dictionary". All 19 app `graph.run` sites use this form.
- `MPSGraphExecutable.h:163-167`:
  `run(with:inputs:results:executionDescriptor:)`, "synchronous and will
  return on completion of execution". It takes an optional
  `MPSGraphExecutableExecutionDescriptor`. The app passes `nil`
  (`ChessNetwork.swift:1494`).
- `MPSGraphExecutable.h:22-23`:
  `typedef void (^MPSGraphExecutableCompletionHandler)(NSArray<MPSGraphTensorData *> * results, NSError * _Nullable error);`
  ("If an error occurs, more information might be found here"). Also
  `MPSGraphExecutableExecutionDescriptor.waitUntilCompleted` ("Flag for the
  graph executable to wait till the execution has completed", default
  false), and `scheduledHandler` with the same `NSError` parameter.
- `MPSGraph.h:90-91, 163-173`: `MPSGraphCompletionHandler(resultsDictionary,
  NSError * _Nullable error)` on `MPSGraphExecutionDescriptor.completionHandler`,
  plus `waitUntilCompleted`. These are usable with `runAsync(with:feeds:targetTensors:targetOperations:executionDescriptor:)`
  (`:316-320`) and `encode(to:…executionDescriptor:)`.
- `MPSGraph.h:387`, `MPSGraphExecutable.h:224`: encode — "commitAndContinue
  might be called, please don't rely on underlying MTLCommandBuffer to remain
  uncommitted".
- `MPSCommandBuffer.h:214-223`: "be sure to use the appropriate command buffer
  when querying the [MTLCommandBuffer status] property".
- `MPSGraphOptions` (`MPSGraph.h:26-32`) has only none / SynchronizeResults /
  Verbose. **No public way to stop MPSGraph from splitting work.**

Answer: synchronous `graph.run` **cannot** report a GPU error.
`executable.run` can only through an execution descriptor's
`completionHandler`.

- **[probe]**: with a descriptor on `encode`, the handler fired once with
  `error=nil` on success.
- Whether it carries the error of a *middle* segment, or any IOGPU
  hang / victim error at all, is **unverified**. A real fault can't be forced
  safely while training shares the GPU.

What the app could use instead (to verify on macOS 27 before relying on it):

- **Own command buffers everywhere:** convert all 19 `graph.run` sites and
  the `executable.run` at :1490 to `encode` into an `MPSCommandBuffer`, then
  `commit` + `waitUntilCompleted`, as #10 / #13 / #16 already do. Then check
  every segment you can reach: the original reference and
  `mps.rootCommandBuffer` after commit (the last). Middle segments stay
  unreachable.
- **Add an execution descriptor `completionHandler`** (`MPSGraphExecutableExecutionDescriptor`
  / `MPSGraphExecutionDescriptor`) on every `encode` / `run`, and treat a
  non-nil `NSError` as failure. This is the only API that could plausibly
  cover every segment MPSGraph created. Needs a fault test, for example in a
  dedicated session with no training running.
- **Segment tracking via a forwarding command buffer:** `MPSCommandBuffer.h:225-227`
  says `commitAndContinue` "will be forwarded" to an underlying object that
  implements it. A wrapper that records each segment would let every
  segment's status be checked. Unverified, and heavy: it needs an Objective-C
  `NSProxy` around `MTLCommandBuffer`.
- **`errorOptions = .encoderExecutionStatus`** only helps the first segment
  (**[probe]**), so it is not a complete answer for A4.

## 4. Existing error and log plumbing

Throw sites:

| Error | Thrown at | Covers |
|---|---|---|
| `ChessNetworkError.gpuCommandFailed` | `ChessNetwork.swift:1664` | previous value baseline, original segment, `== .error`, one call late |
| `ChessNetworkError.gpuCommandFailed` (via `requireCompleted`, `:2006-2010`) | `:1991`, `:1993` | weight load, first + last segment; `requireCompleted` has no other caller (trainer included) |
| `ChessTrainerError.gpuCommandFailed` | `ChessTrainer.swift:7187`, `:7232` | training step, working-weight sync; original segment only |
| `ChessNetworkError.commandBufferCreationFailed` | `ChessNetwork.swift:1974` | weight load only |

All carry `stage`, `status` and `error?.localizedDescription` as a `String`.
The `NSError` is dropped, so its domain, code and `userInfo` are lost.

What each path does after a throw:

| Path | Catch | Logged | Afterwards |
|---|---|---|---|
| Corpus replay | `CLI/CorpusReplayRunner.swift:2124-2130` (stops any GPU capture, rethrows) → `:1083-1087` | `[REPLAY] failed: <localizedDescription>` to session log + stderr. A non-finite readback first logs `[ALARM] loss non-finite …` (`ChessTrainer.swift:7415`) | exit 33. The throw leaves `runReplay` before the post-loop final save, so there is no save (matches B-siluall: "no save after it") |
| Train-vs-UCI | `CLI/TrainVsUciRunner.swift:1064-1070` (tears down the engine producer, rethrows) → `:145-149` / `:167-171` | `[VS-UCI] failed: <localizedDescription>` | exit 33; no save on this path (by reading; the catch rethrows without saving) |
| GUI Play-and-Train / `--train` | `App/SessionController+Training.swift:1330-1346` | `box.recordError` (UI) + `suspendTrainingOnDivergence` → `[DIVERGE] training suspended …: <reason>` (`:2586`) | trainer worker exits. Self-play, heartbeat and stats continue; arenas and periodic autosave are gated |
| Self-play | `Training/BatchedSelfPlayDriver.swift:525-528` | `[SP-TICK] network error: …; skipping tick` | tick skipped. **Unreachable for GPU faults** (#17 never throws one) |
| Arena tournament | `TickTournamentDriver` rethrows → `App/SessionController+Arena.swift:382-386` | **none** in the session log (`recordError` only) | arena cleaned up, no promotion. **Unreachable for GPU faults** |
| Arena candidate / champion sync | `SessionController+Arena.swift:224-228`, `:257-261` | **none** in the session log | arena skipped. Reachable: `loadWeights` can throw `gpuCommandFailed` |
| Train-vs-UCI play | `Training/TrainVsUciDriver.swift:345-347` | `[VS-UCI] trainer network error … skipping tick batch` | unreachable for GPU faults |
| Probes | e.g. `SessionController+TacticalProbe.swift:440-444` `[TACTICAL] evaluateBatched failed`, `LichessProbeWatcher.swift` `[TACTICAL-LICHESS] monitor trainer-snapshot failed` | thrown errors only | probe result discarded |

## 5. Arena GPU work

- **Per tick:** two `evaluateBatched` calls, run concurrently with `async let`
  (`Arena/TickTournamentDriver.swift:557, 581, 606-609`). Each is one
  synchronous `executable.run` (`ChessNetwork.swift:1490`) on its network's
  own `MTLCommandQueue`.
  - `candidateInferenceNetwork` and `arenaChampionNetwork` are separate
    `ChessNetwork` instances; each `init` makes its own queue
    (`ChessNetwork.swift:590`).
  - MPSGraph may split each run into several command buffers internally.
    The count is unmeasured; **[probe]** suggests more than one for a full
    network.
- **Batch size:** candidate-to-move games + champion-to-move games = K.
  - In SPRT mode `initialK = max(1, concurrency)` (`:144-146`), which was 400
    in R-fixedlr. Each call therefore holds at most 400 positions, about 200
    each on average.
  - `P = activeProcessorCount` (`:147`, 18) is CPU workers for encode and
    sample, not GPU parallelism.
  - The executable is compiled and cached per distinct batch size
    (`inferenceExecutables[count]`, `ChessNetwork.swift:1577`). Varying K can
    therefore compile on the first use of each new count, synchronously on
    that network's queue.
- **Other GPU work in the same process during an arena** (R-fixedlr, a GUI
  `--train` run, session log `dcm_log_20261008-094623.txt`):
  - **Self-play** resumes after the champion snapshot
    (`SessionController+Arena.swift:263`). It ran K=250 per
    `executable.run` (`[SP-TICK] paused: dropped 250 in-flight games`).
  - **Training continues** (`trainingGate.resume()`, `:230`). Training step
    batch 4096:
    - `[ENCODE-COST] gpuWaitMs` (commit → wait) p50 about 1.1-2.1 s.
    - The window containing the 21:37:09.567 hang (steps 46500-46509, line at
      21:37:18.915) shows **p99 4,775 ms** against p50 1,839 ms.
    - The window around the 19:36:49 hang (steps 44640-44649, 19:36:51.695)
      shows encodeMs p99 8,034 ms, and `[LEGAL-COST] p3ms` p99 9,968 ms.
    - KL-probe steps log `gpu=18-19 s` on 2026-10-09. That figure includes
      the encode, which is 16-18 s at p99 on those steps.
  - **Value baseline:** batch 4096, not waited (#7).
  - **Tactical / Lichess probe batteries:** one `executable.run` of **4,635
    positions**, `gpuMs` 300-1,100 ms. One ran 21:36:34-21:36:39.9, ending
    about 30 s before the 21:37:09 hang; none overlapped either hang.
- **Largest single submissions:** the training step (multi-second
  commit → wait, split by MPS into an unmeasured number of segments) and the
  4,635-position probe battery. Arena calls are at most 400 positions.
- **Arena per-call GPU duration is not logged anywhere; unmeasured.** It can
  only be measured on our own command buffers (`gpuStartTime` /
  `gpuEndTime`), which requires converting #17 to own-buffer encode.
- **Which buffer hung is unknown.** R-fixedlr logged nothing at 21:37:09:
  1 Hang + 3 InnocentVictim, no `gpuCommandFailed`, no `[DIVERGE]`, no
  `[GRAD-CLIP]` event. So none of its four failed buffers was the *first*
  segment of a training step, a working-weight sync or a weight load. Every
  other R-fixedlr submission is unchecked, so the hang could be in:
  - the arena,
  - self-play,
  - the baseline,
  - a later training segment,
  - a probe.

  The macOS watchdog threshold is not documented in the SDK headers;
  unverified.

## Probe results

Standalone programs in the session scratchpad (`mpscb/probe*.swift`), built
with `swiftc -O` against the default SDK, run on this Mac (M5 Max, macOS 27.2
beta) alongside the live runs. Workloads were tiny: empty buffers, and 64×64
matmul chains.

```
probe (empty buffer, explicit commitAndContinue):
before: original=…d0b0 .commandBuffer=…d0b0 .rootCommandBuffer=…d0b0
after commitAndContinue: .commandBuffer=…19f0 label=original+ .rootCommandBuffer=…19f0 label=original+
original.status=2 root.status=0 mps.status=0
after commit+wait: original.status=4 .commandBuffer.status=4 .rootCommandBuffer.status=4 mps.status=4

probe2 (does MPSGraph encode continue?):
depth=10   executable.encode: continued=false originalStatusAfterEncode=0
depth=10   graph.encode:      continued=false originalStatusAfterEncode=0
depth=200  executable.encode: continued=true  originalStatusAfterEncode=4
depth=200  graph.encode:      continued=true  originalStatusAfterEncode=4
depth=2000 executable.encode: continued=true  originalStatusAfterEncode=4
depth=2000 graph.encode:      continued=true  originalStatusAfterEncode=4

probe3 (segment count; MPS appends "+" to the label per continuation):
depth=50 ops~100 continuations=1
depth=100 ops~200 continuations=2
depth=200 ops~400 continuations=4
depth=1000 ops~2000 continuations=5
depth=4000 ops~8000 continuations=10

probe4 (errorOptions inheritance, explicit commitAndContinue):
original errorOptions=1 retained=false; continued errorOptions=0 retained=false

probe5 (MPSGraph executable.encode, descriptor-created buffer, completionHandler):
completionHandler error=nil
segments label=L++++ originalErrOpts=1 lastErrOpts=0 handlerCalls=1
```

The "+ per continuation" count is inferred from MPS's labeling; the
identity changes in probe 1 / probe 2 are direct.
