# GPU Utilization Plan — training step is CPU-encode-bound

Status: **PLAN — not started.** Phases 1 and 2 only; Phase 3 (kernel fusion /
baseline-forward merge) is explicitly deferred per direction on 2026-06-02.

## Diagnosis

The training step (`p3` in `[LEGAL-COST]`) is a steady **~570 ms** (median
572 ms over 558 windows; min 525, max 950). For a ~3.9M-parameter network at
batch 4096 on an M5 Max, the compute-ideal is ~150–170 ms — so the step runs
at roughly **~25–30% of GPU peak**, and the GPU is idle the rest of the time.

Instruments (2026-06-02 capture) shows *why*, and it is **not** GPU throughput:

- **`Encode MPSGraph` CPU interval = 575 ms** for a single encode (one training
  step). That is essentially all of `p3`. It is CPU time with the GPU idle.
- Individual GPU compute commands are **~260–530 µs** each, but carry
  **~60–106 ms CPU-to-GPU latency** — tiny kernels, delayed because the CPU
  encoder is busy.
- A representative MPSGraph command buffer executes **~9.6 ms** on the GPU.
- When the GPU *does* run, the **Last-Level-Cache Limiter averages ~86%** and is
  the Top Performance Limiter (~80%): the kernels are **memory/cache-bandwidth
  bound**, not compute bound — expected for 8×8×128 tensors with a BN/SE op
  between every conv.
- The Metal driver shows large rhythmic **"Wait for GPU"** blocks (synchronous
  `graph.run` blocking after each encode).

**Root cause.** The network is small, so the forward + backward + optimizer
graph is hundreds of tiny kernels. `network.graph.run(...)`
(`ChessTrainer.swift:5382`) is invoked **every step**, and the per-step
`Encode MPSGraph` cost (~575 ms) dwarfs the actual GPU execution. At ~358
kernels/step that is **~1.6 ms of "encode" per kernel** — 100–1000× slower than
raw Metal command encoding (~1–10 µs/kernel), which strongly implies the cost is
**graph planning/lowering redone per run**, not raw command encoding. That is
the lever Phase 2 targets. The GPU is starved waiting for the encoder and is
LLC-bound when it runs.

This also reconciles the earlier confusion: the GPU timeline gaps and the
60–106 ms "CPU-to-GPU latency" are the encoder being the bottleneck, not a
queue-hop bubble (`[LEGAL-COST]` `gapMs` is ~3 ms, so cross-phase dispatch
overhead is negligible).

## What does NOT consume telemetry per step (verified)

The per-training-step loop (`SessionController+Training.swift:1349–1403`) does:
`box.recordStep(timing)`, `pStatsBox.recordTrainingStep()`,
`fireCandidateProbeIfNeeded()`, an arena auto-trigger keyed on **step count +
wall-clock** (not loss), and `ratioController.recordTrainingBatchAndGetDelay(...)`
keyed on **production/consumption counts + timing** (not loss).

- `[STATS]` (line 1866) and the entropy `[ALARM]` (line 2021) run in a
  **separate periodic loop** and read box's rolling-mean **snapshot**, not
  per-step values.
- The legal-mass collapse alarm runs in its **own probe loop** (2249+), not the
  training step.
- There is **no per-step NaN/divergence guard** reading the loss. The only loss
  guard (`lossOutputMissing`, 5434) just checks the requested target tensors
  came back.

**Conclusion:** nothing in the per-step path reads any telemetry tensor for a
control decision. They only feed rolling means emitted on the periodic
`[STATS]` cadence. This is what makes Phase 1 safe.

---

## Phase 1 — Telemetry gating — IMPLEMENTED 2026-06-02

Compute the diagnostic-only graph outputs **only on stats steps**, reusing the
existing `isStatsStep` gate (`ChessTrainer.swift`,
`nextStep % batchStatsInterval == 0`) that already gates the replay metadata and
`[LEGAL-COST]` line.

**As implemented:** `runPreparedStep` gained an `includeDiagnostics` flag. It is
gated on a **diagnostics cadence** that equals `batchStatsInterval` when that is
set (so the reductions coincide with stats steps) but falls back to a fixed
interval (`diagnosticsFallbackInterval = 10`) when `batchStatsInterval == 0` —
because the diagnostics feed the `[STATS]` line and the entropy/draw-collapse
alarms, not just `[BATCH-STATS]`, so disabling batch stats must not silently kill
them. The random-data sweep passes `true` to keep its measured step unchanged.
On non-diagnostic steps only the lean targets (totalLoss,
policyLoss, valueLoss, illegalMassPenalty, gradGlobalNorm) are requested, so
MPSGraph never encodes the diagnostic reductions; the diagnostic `TrainStepTiming`
fields are `.nan` / `nil` and `hasDiagnostics` is `false`. `recordStep` appends
the loss/grad-norm + timing windows every step but gates the diagnostic windows,
skip-windows, advantage-percentile ring, and `_lastTiming` on `hasDiagnostics`,
so the rolling means accumulate only real samples (now at stats-step cadence) and
the "Last Step" UI panel never shows NaN. Pending: measure the `p3` / encode-time
drop on a non-stats step once training is idle.

### Honest bound on the win

`graph.gradients(of: totalLoss, with: trainableVariables)` →`assignOps` forces
the **entire forward + backward + optimizer** to be encoded regardless of which
telemetry we request. So gating **cannot** remove the core encode cost — only
the *extra diagnostic ops* layered on top. The win is real but **bounded**;
Phase 2 is what attacks the core. Several diagnostic ops are nonetheless large
(`[batch, 4864]` ≈ 20M-element reductions), so the savings are worth measuring,
not dismissing.

### Proposed split (verify subexpression sharing during implementation)

**Keep every step** (on the loss/clip path — computed anyway, ~free readback;
also satisfies MPSGraph's "≥1 target tensor" requirement):
- `totalLoss` (drives the gradient)
- `policyLossTensor`, `valueLossTensor`, `illegalMassPenaltyTensor` — terms summed
  into `totalLoss` (`3259`), computed en route
- `gradGlobalNormTensor` — computed for gradient clipping (`3322–3392`)

**Gate to stats steps** (extra diagnostic ops; the starred ones are the large
`[batch, 4864]` passes and the real savings):
- `policyEntropyTensor` ★ (entropy coeff is currently 0, so this is diagnostic)
- `policyLogitAbsMaxTensor` ★
- `playedMoveProbTensor`, `playedMoveProbPosAdvTensor`, `playedMoveProbNegAdvTensor` ★
- `policyLossWinTensor`, `policyLossLossTensor` ★
- `policyNonNegCountTensor`, `policyNonNegIllegalCountTensor` ★
- `valueMeanTensor`, `valueAbsMeanTensor`, `valueProbWin/Draw/LossTensor`
- `policyHeadWeightNormTensor`, `velocityGlobalNormTensor`
- `advantageMean/Std/Min/Max/FracPos/FracSmall/RawTensor`

### Design

- `runPreparedStep` takes the target-tensor list as a parameter (or two
  precomputed arrays: `everyStepTargets`, `fullTargets`). Phase 3 of `trainStep`
  passes the full list when `isStatsStep`, the lean list otherwise.
- The readback `guard` block splits accordingly: on a lean step only the
  every-step results are read; `TrainStepTiming` carries `nil`/`.nan` for the
  gated fields, and `box.recordStep` must tolerate that (record only present
  fields into the rolling means).
- **Semantics change to accept:** rolling means become "last N **stats** steps"
  instead of "last N steps." At the `[STATS]` cadence this is fine (arguably
  cleaner — regular sampling). Document it.

### Validation
- Build; with training idle, run the suite.
- Confirm `[STATS]` still renders every field (sampled on stats steps).
- Measure `p3` and the `Encode MPSGraph` interval on a **non-stats** step vs a
  stats step — quantify the gated savings. Log both.
- Confirm the entropy `[ALARM]` and draw-collapse signals still fire (they read
  the snapshot, which now updates on stats steps).

---

## Phase 2 — Compile the training step to an `MPSGraphExecutable`

Replace per-step `network.graph.run(...)` with a once-compiled
`MPSGraphExecutable` (via `graph.compile(...)`), executed each step with
`encode`/`run`. The hypothesis (supported by the ~1.6 ms/kernel "encode" figure)
is that the 575 ms is **planning/lowering redone per run**, which a compiled
executable caches.

### Regeneration triggers (the lifetime question)

An executable is fixed for a given **(graph topology, input shapes, target set,
target operations)**. Regenerate **only** when one changes:

1. **Graph topology** = trainer network rebuilt = **`resetNetwork()`** (training
   start, session resume, sweeps; `Training.swift:1022/1039`, `Sweep.swift:35`,
   `ChessTrainer.swift:5852`). ≈ once per session.
2. **Input shapes** = **batch size** changes. Already cached per batch size
   (`feedCache`); compile one executable per batch size, lazily, same pattern.
3. **Target set** = if Phase 1 lands, compile **two** executables per batch size
   (lean + full). Each compiled once, reused.

**Does NOT regenerate on:**
- **Weight updates** — weights are MPSGraph variables; `assignOps` mutate them
  in place; the executable reads current values each run.
- **Arena promotion** — `champion.loadWeights` / `trainer.network.loadWeights`
  (`Arena.swift:285–286`) **copy weight values into existing variables**; the
  graph/topology is unchanged. **Confirmed:** promotion does not rebuild.
- **Hyperparameter changes** (LR, weight decay, momentum, entropy coeff, label
  smoothing, etc.) — all fed as scalar placeholders each step.

**Net:** in steady-state training the executable is compiled **once at session
start** and reused for the entire run, **including across promotions**. Rebuild
count ≈ number of `resetNetwork` calls + distinct batch sizes.

### Weight-state architecture (investigated 2026-06-02)

Weights are stateful `graph.variable(...)` ops. **Every** weight write goes
through `graph.run` on the one `MPSGraph`:
- SGD update: `graph.assign` ops run as `targetOperations` each step.
- `loadWeights` (`ChessNetwork.swift:686–707`, run at `1431`): feeds
  `weightLoadPlaceholders` and runs `weightLoadAssignOps` (`g.assign(v, ph)`).
- Promotion (`Arena.swift:285–286`) and checkpoint load both call `loadWeights`.

So today there is a **single owner of weight state** — the graph — and it stays
consistent because training and loading share the same `graph.run` path.

### Variable-state sharing (probed 2026-06-02; one direction confirmed)

The risk was that a separately-compiled `MPSGraphExecutable` captures its **own**
variable storage, forking from the graph: the training executable would train a
snapshot while `loadWeights` (promotion/checkpoint, via `graph.run`) writes the
graph's copy — a silent correctness break.

`DrewsChessMachineTests/MPSGraphExecutableVariableSemanticsTests.swift` probes
it. Findings:

- **Forward (`graph.run` write → executable read): SHARED — confirmed on-device
  (M5 Max, 2026-06-02).** `testExecutableObservesExternalGraphAssignToVariable`:
  after a `graph.run`+assign sets `v=9`, a re-run of a previously-compiled
  executable reads `9`. So `loadWeights`/promotion/checkpoint assigns (all via
  `graph.run`) **are** visible to a compiled training executable. Locked with an
  `XCTAssertEqual`.
- **Reverse (executable write → `graph.run` read): SHARED — confirmed on-device
  (M5 Max, 2026-06-02).** `testGraphRunObservesExecutableAssignToVariable`: after
  an executable assigns `v=7` (then `v=3`), a `graph.run` read returns `7` (then
  `3`). So the training executable's own SGD assigns are visible to the
  `graph.run` readers (arena candidate snapshot, `evaluate`, checkpoint
  `makeWeightData`) and persist across executable runs. Locked with
  `XCTAssertEqual`.

**Both directions share one variable buffer ⇒ the variable-state fork risk is
retired.** Phase 2 is the drop-in path, not the functional-weights refactor.

Also learned (and codified in the test): MPSGraph does **not** order a
`targetOperations` assign before a `targetTensors` read of the same variable
*within one run* — assigns and reads of the same variable must be in separate
runs (the app already does this; the probe's first draft did not, and failed its
own sanity check).

**Cost note (answers "are we running `graph.run` anyway?"):** yes, but only for
the **rare** writers — `loadWeights`/promotion/checkpoint, ~once per arena win.
Their encode cost is negligible. Phase 2 only replaces the **per-step** training
`graph.run` with `executable.run`; mixing a rare `graph.run` with a per-step
executable is the intended design, and the shared-storage result is what makes
it sound.

**Resolved:** Phase 2 is the drop-in path — compile the training step (with the
SGD assigns as `targetOperations`) into an executable, run it per step, and
leave `loadWeights`/snapshot/inference on `graph.run`; all share one variable
storage. The remaining gate is the **perf** question below (does compiling
actually cut the ~575 ms encode?), not correctness of weight sharing.

### Other validation (only if the fork risk clears)

- **Planning-vs-encoding.** Measure whether the `Encode MPSGraph` interval
  actually drops. If 575 ms is raw per-kernel encoding rather than planning, the
  executable won't help much and the remaining lever is Phase 3 (deferred) —
  surface that rather than proceeding.
- **Numerical equivalence** vs `graph.run` for one step (within fp tolerance).
- Confirm the optimizer assigns work as the executable's target operations, and
  that the weight-read/snapshot paths (arena snapshot, `makeWeightData`) still
  read the live variables.

### Validation
- Build; with training idle, run the suite + a one-step `graph.run` vs
  executable numerical compare.
- Measure `p3` / `Encode MPSGraph` before/after. Gate the rest of the work on a
  real drop.

---

## Phase 3 — Encode-ahead pipeline (4–8 steps in flight)

Phase 2's measurement settled the open question: the executable cut per-submission
**CPU-to-GPU latency ~20%** (≈61→49 ms) but barely moved step time, so the cost is
**raw command encoding** of the hundreds of tiny kernels, not planning. The win is
to overlap CPU encode with GPU execution — keep the one command queue full so the
GPU never stalls on the encoder.

### The structural reality that shapes everything

Training steps are a **sequential dependency chain**: step N+1's forward reads the
weights step N's SGD `assignOps` just wrote. They therefore CANNOT run in parallel
on the GPU — Metal serializes them via the weight-variable resource hazard.
Submitting 4–8 in flight does not parallelize the training math; it keeps the
**command queue full** so the GPU never waits on the CPU encoder ("always full for
certain"), and gives slack against CPU encode jitter.

The ordering question, resolved against Apple's docs:

- Command buffers on one queue are **guaranteed to execute in enqueue order**
  (archived Metal Programming Guide: "All command buffers sent to a single queue are
  guaranteed to execute in the order in which the command buffers were enqueued").
  That is *start* order, not non-overlap — Apple GPUs may overlap consecutive command
  buffers, so order alone doesn't guarantee N's weight writes are visible to N+1's
  reads.
- BUT for **tracked** resources, "Metal automatically uses synchronization primitives
  to avoid hazards on the GPU timeline" across passes/command buffers within a single
  queue (`MTLHazardTrackingMode` docs). Device-allocated `MTLBuffer`s — which MPSGraph
  `variable` weight buffers are — are **tracked by default**. Explicit Fences/Events
  are only required for **untracked** resources (heap-allocated `.untracked`).

So the weight read-after-write hazard across command buffers is **handled
automatically by Metal**, provided the weight buffers are tracked (the default). We
therefore likely need NO explicit `MTLEvent`/fence — just `enqueue()` for order +
`commit`, relying on automatic hazard tracking, while the CPU encodes N+1…N+k ahead
to keep the queue full. One queue (`network.commandQueue`), N command buffers.

**Verify, don't assume:** confirm MPSGraph's variable buffers are tracked (they
should be — device-allocated), with the loss-doesn't-diverge test (Increment 2) as
the gate. If they turn out untracked, the fallback is a single `MTLFence` on the
weight buffer. Never rely on bare in-order *start* without the tracking guarantee.

### Pieces

1. **`executable.encode(to: commandBuffer)` instead of `executable.run`.** `run`
   commits-and-waits (the current stall). `encode(to:)` records into a command
   buffer we own; `enqueue()` → encode → `commit` without waiting, then move on.
2. **N-deep resource ring.** While the GPU reads step N's feeds / writes its
   results, the CPU fills step N+1's — so feeds AND result buffers must be
   N-buffered (currently single). `feedCache` → an N-slot ring per batch size;
   pre-allocated result `MPSGraphTensorData` per slot (the `results:` argument,
   now N-buffered — also removes the per-step result allocation).
3. **GPU→GPU baseline handoff.** Co-encode the fresh-baseline forward and the
   training step. The baseline's `v(s)` flows into the training step's `vBaseline`
   as a **GPU buffer** (the baseline's output ndarray fed as the placeholder's
   data) — no CPU readback (which would break the pipeline). Encoded in order, the
   GPU runs `train(N-1)` [writes W] → `baseline(N)` [reads updated W] → `train(N)`,
   so the baseline stays **fresh** AND the CPU round-trip is gone. The placeholder
   boundary still detaches the gradient (value head only trains via its own CE).
4. **Completion → readback → recordStep, off the handler.** The command-buffer
   completion handler only *signals*; a consumer drains the result buffers and
   calls `recordStep`. No work in the handler.
5. **Replay-ratio controller** counts consumed batches at completion — same
   cadence, now async.

### Risks / validation
- **Read-before-write on the weights** — handled by Metal's automatic hazard
  tracking *if* the variable buffers are tracked (the default for device buffers);
  the loss-doesn't-diverge test confirms it. Fallback if untracked: one `MTLFence`
  on the weight buffer. A divergence is the signature of an unguarded hazard.
- **vBaseline GPU-handoff detach** — confirm no gradient reaches the value head
  through the baseline (the placeholder boundary must hold).
- **N-buffer lifetime** — the GPU must finish reading slot k before the CPU
  overwrites it N steps later; the in-flight cap (N) enforces this.
- Success signature: CPU-to-GPU latency collapses toward µs, the GPU track goes
  solid, step throughput rises.

### Increments (each built + validated before the next)
1. **Plumbing** — `executable.run` → `encode(to:)` + `commit` + completion handler,
   still 1-deep (await completion). Same perf; proves the encode/commit/completion
   path with zero pipelining risk.
2. **Pipeline depth** — N-deep feed/result ring + encode-ahead (don't await; keep N
   in flight, cap configurable 4–8). The actual win.
3. **GPU→GPU baseline** — remove the CPU baseline readback; co-encode baseline+train.

## Phase 4 — Deferred (kernel-count reduction)

Fuse BN into convs, simplify the SE block — fewer kernels to encode. Revisit after
the pipeline; with encode overlapped, fewer kernels compounds the win.
