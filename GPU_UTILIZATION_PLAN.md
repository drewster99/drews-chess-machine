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

## Phase 3 — Parallel-encoder pipeline (measurement-driven)

**The decomposition (`[ENCODE-COST]` probe, build 1620, 2026-06-03):**
- pure CPU `executable.encode()`: **p50 ≈ 340 ms/step**
- `commit` + GPU + `waitUntilCompleted`: **p50 ≈ 175 ms/step**

The training step is **CPU-encode-bound ~2:1, not GPU-bound.** (~340 ms / a few-
hundred tiny kernels ≈ ~1 ms/kernel — per-kernel encode overhead × the kernel count
of a tiny 8×8 net.) This **killed the original single-encoder design**: one encoder
produces a step every ~340 ms; the GPU wants one every ~175 ms, so a single encoder
is 2× too slow — the GPU starves no matter how deep the slot pool. Increment 1
(encode/commit/completion plumbing, done + validated) stands; the *single-encoder*
Increment 2 does not.

### The corrected architecture: parallel encoders feeding one ordered queue

Encoding is the bottleneck, so **encode in parallel across ~2–3 threads** (a single
`MTLCommandBuffer` can't be split across threads, so each thread encodes a *different
step's* buffer). The steps are still a **dependency chain** (step N+1 reads step N's
updated weights), so the GPU must run them in order — `enqueue()` locks queue order
*before* encoding, regardless of which thread finishes first (the Metal "multiple
command buffers across threads + `enqueue()` for strict order" pattern). This is the
"N workers" intuition, correctly understood: **N parallel *encoders* of one ordered
chain, not N independent training loops.**

CPU units (serial `DispatchQueue`s + a concurrent encode pool + semaphore + completion
handlers — no actors):
- **Control (serial queue):** per step, in order — acquire a slot (semaphore, cap N);
  sample + `buildFeeds` into the slot (~25 ms, fast, stays ahead of the 175 ms GPU);
  create an `MTLCommandBuffer` and **`enqueue()` it (locks GPU order)**; dispatch the
  encode job (buffer + slot) to the worker pool.
- **Encode pool (M ≈ 2–3 threads):** encode the step's baseline + train into the
  reserved buffer (slot's feeds / result / baseline-output ndarrays), attach a
  completion handler, `commit`. M chosen so M/340 ≥ 1/175 ⇒ M ≈ 2–3.
- **Completion handler (Metal thread, minimal):** hand `(slot, results)` to the consumer.
- **Consumer (serial queue):** readback + `recordStep` + free the slot. Steps complete
  in GPU (enqueue) order, so this stays ordered.
- **Slot ring (N ≥ M+1):** each slot = feed buffers + per-slot baseline-output ndarray
  + result buffers. The consumer releases the slot.

If this works, the GPU runs back-to-back at ~175 ms/step ⇒ train step **~550 → ~175 ms
(~3×)**, pure infrastructure (no network/math change).

### Ordering correctness
- `enqueue()` in step order locks GPU execution order regardless of encode-finish order.
- The **weight RAW hazard** across the dependent steps is handled by Metal's automatic
  tracking (device `MTLBuffer`s are tracked by default), so no explicit `MTLEvent`
  needed — verify via loss-doesn't-diverge; `MTLFence` fallback if untracked.

### GPU→GPU baseline handoff (folded in — can't be separated)
Each step's command buffer encodes **baseline forward → train**, with the baseline's
per-slot `v(s)` output ndarray bound directly as the train step's `vBaseline` input —
no CPU readback (a readback would re-serialize the pipeline). Intra-buffer hazard
tracking orders the baseline-write before the train-read; the GPU's step ordering
keeps the baseline reading the latest weights (fresh). The placeholder boundary still
detaches the gradient.

### KEY open risk — verify before building (the make-or-break, like variable-sharing was for Phase 2)
**Is `MPSGraphExecutable.encode(to:)` thread-safe for concurrent calls from multiple
worker threads (each with its own command buffer, sharing one executable)?**
- If **yes** → share the one cached executable across encoders. Done.
- If **no** → need **per-thread executable instances** compiled from the same graph,
  and we must verify *those* share the weight variables (extend
  `MPSGraphExecutableVariableSemanticsTests`). This is the gating prerequisite —
  resolve it with a focused probe/test first, exactly as we resolved variable-sharing
  before committing to Phase 2.

### Other caveats
- **CPU cost:** ~2–3 cores continuously encoding — competes with the 800 self-play
  workers for CPU. Watch self-play throughput doesn't regress.
- **Not unit-testable:** correctness is concurrency-under-real-training; the gate is a
  **live run** watching loss-doesn't-diverge + self-play health.
- **Complementary lever (Phase 4):** kernel fusion cuts the ~340 ms encode at the root
  (fewer kernels → less encode *and* less GPU), reducing the M needed — or removing the
  need for parallel encoders entirely if it drops encode below the 175 ms GPU.

## Phase 4 — Deferred (kernel-count reduction)

Fuse BN into convs, simplify the SE block — fewer kernels to encode. Revisit after
the pipeline; with encode overlapped, fewer kernels compounds the win.
