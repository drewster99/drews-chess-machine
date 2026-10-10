# GPU fault forensics — plan

**Status: started 2026-10-09 on the owner's go ("execute that plan").**
Revised after an independent review (`GPU_FAULT_FORENSICS_REVIEW_2026-10-09.md`
in this folder; every finding is answered in "Review findings" below).
Owner request (2026-10-09, after the 21:37 incident below):

1. When training has to stop (a NaN or similar), write a crash dump that saves
   everything needed to investigate.
2. Hash every batch (at least the logged ones), so a resumed run can prove it
   saw the same batches instead of assuming it.
3. Audit GPU and memory error handling: we may not be watching, or logging,
   every GPU error; check and log every failure.

## As built (deviations from the design below)

- **Batch hashes (Part B):** the cheap per-step identity was dropped — a
  resumed run rebuilds its replay buffer, so ring slots and stream counters
  differ from the original's and could never match. Instead every step's full
  content (boards, moves, outcomes after the draw penalty) is hashed on a
  utility queue (≈12 ms of one core per step, never waited on by the step —
  the owner allowed asynchronous CPU work) and chained per 1,000-step window.
  Only `[BATCH-HASH]` lines carry them (every 100 trainer steps); step lines
  are unchanged, so no log parser changes.
- **Self-play:** any ledger fault, including the driver's own failed
  evaluation, abandons the games in progress (more conservative than "skip
  the tick" for layer 1).
- **Monitor availability** is in `results.json` `gpu_faults.monitor` and the
  `[GPU-SYSLOG] monitoring …` / `unavailable` line, not on `[RUN]` (its
  format is shared and parsed).
- **Crash dump:** records the run's `[RUN]` line (seed, parameters hash,
  build, device, lineage run and segment) instead of the full lineage record,
  and no per-sample source slots (the positions are in `batch.safetensors`
  and can be decoded from the boards).
- **Voided arena:** `[ARENA] voided: GPU fault …` precedes the usual verdict
  line, which still shows the SPRT outcome; no promotion happens.
- `ChessNetwork.requireCompleted` stays as the single-buffer form of
  `GPUSubmissionReport`'s rule (its test pins it).

## The incident that prompted this (2026-10-09 21:37:09)

Three training processes shared the GPU (build 2440, macOS 27.2 beta
26B5091g, Apple M5 Max): R-replay (pid 20981), B-siluall (pid 19816) and
R-fixedlr (pid 20985, GUI self-play, an arena in progress since 21:36:18).

System log (`/usr/bin/log show`), none of it in any session log:

| Time | Process | Message |
|---|---|---|
| 21:37:00.7–21:37:09 | TGOnDeviceInferenceProviderService, textunderstandingd, GenerativeExperiencesSafetyInferenceProvider | macOS on-device language-model requests (MPSGraph models loading, one-shot inference) |
| 21:37:09.567 | DrewsChessMachine 20985 (R-fixedlr) | `Execution of the command buffer was aborted … GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)` |
| 21:37:09.569–.570 | DrewsChessMachine 20985 (×3) | `Discarded (victim of GPU error/recovery) (00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)` |
| 21:37:09.570 | DrewsChessMachine 20981 (R-replay) | same, InnocentVictim |
| 21:37:09.571 | DrewsChessMachine 19816 (B-siluall) | same, InnocentVictim |
| 21:37:09.633 | osanalyticshelper | wrote `gpuEvent-DrewsChessMachin-2026-10-09-213709.ips` (later found moved to `/Library/Logs/DiagnosticReports/Retired/`, as were the 19:36 and 22:59 reports: `restart_reason_desc` "firmware-detected lockup", `guilty_dm` 3, `signature` 579 in all three; no pid, `process_name` "DrewsChessMachin") |

What our own logs showed:

| Run | Session log | Effect |
|---|---|---|
| B-siluall | 21:37:09.834 `[ALARM] loss non-finite … grad=nan`, halted | NaN gradient at trainer step ~45,975; no save after it |
| R-replay | 21:37:09.960 `[GRAD-CLIP] trainerStep=43220 preNorm=10624371` | relative cap clipped the step to 0.85; training continued |
| R-fixedlr | nothing | unknown which work was lost (self-play, arena or training) |

**How far back the evidence goes.** The system log keeps error entries only
back to 2026-10-09 10:56 (`/private/var/db/diagnostics/Persist`, 51 files,
499 MB cap); earlier GPU faults are gone from it. Our own session logs (9,870
files back to 2026-04-14) record a GPU fault only when our code caught one:

| Log | Build | Event |
|---|---|---|
| 2026-06-14 10:49 | 1883 | NaN halt, losses blown up (total 1.0e13) |
| 2026-06-14 19:53 | 1845 | NaN gradient, finite losses (total 5.69), 28 s after launch |
| 2026-06-15 07:55 | 1891 | inf gradient, losses blown up (total 1152) |
| 2026-06-16 19:27 | 1921 | NaN halt, losses blown up (total 6.1e30) |
| 2026-06-24 15:01 | — | corpus replay, inf gradient, losses blown up (4.5e7) |
| 2026-06-24 16:17 | — | corpus replay, NaN, losses blown up (1.2e20) |
| 2026-06-25 18:06 | 1977 | NaN halt, losses blown up (4.6e17) |
| 2026-07-14 18:13 | — | train-vs-UCI halted: `GPU command buffer failed during working-weight sync: status=5, error=Internal Error` |
| 2026-10-09 21:37 | 2440 | this incident |

The June halts are the reduced-precision divergence period (losses grew
first). The July 14 one is the only GPU fault our code has ever caught and
logged. No other session log has a gradient spike ≥ 100 × its reference
(`[GRAD-CLIP]` lines exist only in recent builds).

**Earlier hang, same day.** A search of the system log
(`eventMessage CONTAINS "kIOGPUCommandBufferCallbackError"`, run 22:30) finds
exactly one other event, 2026-10-09 19:36:49: R-fixedlr 1 Hang + 3
InnocentVictim, R-replay 1 InnocentVictim; B-siluall none. Nothing in any
session log; R-replay's step lines around it (41,330) look normal, so the
discarded buffer there was likely not one that fed training — unknown.

**Third hang, 2026-10-09 22:59:04.127:** R-fixedlr Hang + 3 InnocentVictim,
R-replay 1 and B-siluall (resumed, pid 63702) 1 InnocentVictim. Effects, all
clipped by the relative cap, none halted: R-fixedlr pre-clip norm 13,487,039 at
trainer step 47,924 (and `[ALARM] Critical Training Divergence … gNorm=26343`),
B-siluall 7,425,386 at 46,697, R-replay 2.35 (9.4× its reference) at 44,616.
8.5 s after R-fixedlr's `[ARENA] start` at 22:58:55.602.

All three hangs came early in an R-fixedlr arena: 15.2 s after `[ARENA] start` at
19:36:33.910 (hang 19:36:49.102; that arena ran to 19:38:31) and 51.3 s after
the one at 21:36:18.260 (hang 21:37:09.567; SPRT decided 0.6 s later, arena
ended 21:37:36) (SPRT tick driver, initialK=400
games in flight). R-fixedlr ran 260 arenas from 2026-10-08 10:25 to
2026-10-09 22:34, so 3 hangs in about 265 arenas, all three within 3.5 h on 2026-10-09. macOS language-model activity at
19:36:30–19:36:50 was only 4 `textunderstandingd` lines, so it is not a
common factor. Pattern, not proof: the arena's GPU work, with two other
trainers on the GPU, is the first suspect.

B-siluall's exact resume from step 45,000 replayed the step-~45,975 batch at
22:25 with no fault — consistent with the GPU reset, not the batch. Without
batch hashes that "same batch" is an assumption (Part B).

Memory at the time of writing: swap 25.3 GB used of 26.0 GB
(`sysctl vm.swapusage`, 22:3x). Whether memory pressure contributed to the hang
is unknown; nothing records it (A6).

## Part A — GPU and memory errors: detect, log, act

### A1. Inventory — done

`GPU_CALL_AUDIT_2026-10-09.md` (this folder). All GPU work is MPSGraph (no app
`makeBuffer`, compute pipelines or MPS kernels). 24 GPU submissions: none fully
checked, 4 partly (training step, working-weight sync, value baseline, weight
load), 20 not at all (19 synchronous `graph.run`, 1 `executable.run` —
self-play / arena / probe inference). 9 nil- or throw-returning creations
(device, queues, `makeCommandBuffer`, capture start), all checked. 117 calls
have no failure signal at all (`MPSNDArray` / `MPSGraphTensorData` init —
imported non-nil — `readBytes` / `writeBytes` — no return — and `compile` — no
error path). No site logs the Metal error. (The audit's "164" counts commit /
wait / status reads and `MPSCommandBuffer` wrappers as sites too.)

**Limit of "check every call" (owner decision, 2026-10-09):** the 117 calls
with no failure signal cannot be checked where they are made. Their effects
are covered where they surface: a bad allocation or copy shows in the
submission that uses it (A2) or in the values read back (non-finite checks,
A4). Reported to the owner.

Facts from the audit's probes that shape the design:

1. MPSGraph `encode` splits work across several command buffers
   (`commitAndContinue`): 1 extra buffer at ~100 ops, 4 at ~400, 10 at ~8,000.
   A training step spans several.
2. After a split, `MPSCommandBuffer.commandBuffer` and `.rootCommandBuffer`
   both name the newest (last) buffer; middle buffers are unreachable.
3. Continued buffers don't inherit `errorOptions`.
4. Synchronous `graph.run` returns no error. `encode` / `run` with an execution
   descriptor gets a `completionHandler(…, NSError?)`; whether that error
   covers a middle buffer, or a hang / victim, is unverified. The completion
   handler can run after `waitUntilCompleted` returns, so it must be awaited.

So detection has three layers; none alone is complete.

### A2. Layer 1 — `GPUSubmission`, the one checked submission path

`Network/GPUSubmission.swift`. Every GPU submission in the app goes through
it: all 19 `graph.run` sites, the `executable.run` site and the four `encode`
sites (training step, working-weight sync, value baseline, weight load; plus
the KL probe and its dropout-RNG advance, converted from `graph.run`).

- Creates the first command buffer with `errorOptions = .encoderExecutionStatus`
  (first buffer only; see fact 3) and labels it with the stage.
- Hands out one execution descriptor (graph or executable) whose completion
  handler records the `NSError`.
- `commit()`; `waitUntilCompleted()` (never throws — usable inside the
  `weightAccessLock` section, which must reach its unlock); `verify()` waits
  for the buffers and for the completion handler (10 s grace after the last
  buffer completes; silence is a failure), then fails unless the first buffer,
  the last buffer and the handler all report success.
- A failure logs one `[GPU-ERR] stage=… first=… last=… handler=… gpuMs=…`
  line with each Metal error's domain, code and text and each encoder's state,
  records it in `GPUFaultLedger`, and throws
  `ChessNetworkError.gpuCommandFailed(stage:status:error:)` — the one GPU
  failure error (the trainer's duplicate case is removed).
- `[GPU-SLOW] stage=… firstMs=… lastMs=… spanMs=…` when either reachable
  buffer's own GPU time reaches 5 s (the watchdog acts per buffer; the span
  includes CPU encode gaps, so it is reported but not tested against).
- `GPUStage` names every kind of submission (enum, not strings).
- The value baseline keeps its no-wait overlap: the training step verifies the
  baseline's submission (`verifyPendingValueBaseline`, which waits for it)
  right after verifying its own step and before reading any result. Because
  the step's update has already run by then, a failed baseline cannot be
  prevented, only caught — it leads to the training-fault action (A5).

### A3. Layer 2 — `GPUFaultMonitor`: macOS's own fault messages

macOS logs every hang / victim / page fault in the affected process
(`kIOGPUCommandBufferCallbackError…`), including for buffers the app can't
reach. `OSLogStore(scope: .currentProcessIdentifier)` reads them without an
entitlement (verified with a standalone probe; 1.8–2.0 s per query beside
three trainers — to be re-measured in a long-lived GUI process, M2).

- One monitor per process, started by every training path (GUI Play-and-Train
  and `--train`, corpus replay, train-vs-UCI); polls every 10 s on a utility
  queue.
- **Allowlist filter:** an entry counts only when its text contains
  `kIOGPUCommandBufferCallbackError` (the three incidents' messages: `…Hang`
  `00000003`, `…InnocentVictim` `00000005`); the incidents' captured lines are
  test fixtures. Other `IOGPUMetalError` lines are logged, not acted on.
- Each new entry: `[GPU-SYSLOG] <log time> <message>` once, and one ledger
  record per entry (time = macOS's log time).
- **Barrier:** `checkNow()` runs a synchronous poll (≈2 s) and returns the
  faults it found. Every action that would make post-fault state durable or
  authoritative calls it first and refuses on a new fault (A5): promotion,
  every save, `LastSessionPointer` updates, the rolling `--out-model`
  replacement. Saves and promotions happen at most every few minutes, so the
  ≈2 s is acceptable.
- If `OSLogStore` can't be opened or queried: `[GPU-SYSLOG] unavailable: …`
  once, `gpu_fault_monitor: unavailable` in `results.json` and on the `[RUN]`
  line, and the run continues on layers 1 and 3 — so "no faults" is never
  read from a run whose monitor didn't work.

### A4. Layer 3 — numeric guards (exist; what they really cover)

- The non-finite check halts on a NaN / inf loss or gradient (B-siluall at
  21:37).
- The relative gradient cap clips only in `mode=clip` (the default mode logs
  and doesn't clip, CLAUDE.md); all three runs on 2026-10-09 used `clip`.
  Its `applied=true` is inferred on the CPU from a readback that may itself be
  faulted.
- What the three hangs showed (from the logs): 19:36:49 — no visible effect;
  21:37:09 — R-replay 37,469,639× clipped, B-siluall NaN (halt); 22:59:04 —
  R-fixedlr 10,343,145×, B-siluall 32,736,274× (both clipped), R-replay 9.38×
  (clipped). So a hang-corrupted step can look ordinary (9.38×) or show
  nothing; layers 1 and 2 are the real detection, layer 3 limits damage.

### A5. What happens on a GPU fault

"Training fault" = a training-step / baseline / working-weight-sync /
KL-probe / dropout-advance / promotion-copy failure (layer 1), or any ledger
fault while the trainer is active (layer 2). It means the weights, optimizer
state or RNG state may be wrong in ways nothing can locate, so the run
returns to its last fault-free save.

| Where | Layer 1 (immediate) | Layer 2 (late) |
|---|---|---|
| Training fault, corpus replay / train-vs-UCI | crash dump (C), no further saves, `results.json` `gpu_fault` + `crash_dump`, exit 36 | same, at the step it is seen |
| Training fault, GUI Play-and-Train | crash dump; `trainingSuspension = .divergence(reason: "GPU fault: …")` — the existing case: arenas, Promote Trainee Now and periodic autosave all skipped, so suspect weights are never saved | same |
| Training fault, GUI `--train` | crash dump; ends through `AutoTrainTermination` (results with `gpu_fault`) | same |
| Promotion copy (sync masters, velocity, dropout restore) | crash dump + the training-fault action (the trainer may be half-rewound) | — |
| Arena | the arena's own evaluation fails → arena ends without a verdict, logged `[ARENA] voided: GPU fault` | barrier before `shouldPromote` acts: a fault since arena start voids it — no promotion, no "kept" |
| Self-play inference (GUI) | the tick throws before any move is sampled; existing skip-tick behavior, now logged | abandon every game in progress at detection, logged with the count; games that finished within the lag stay (logged as possibly affected) |
| Save (any path, any trigger) | an export that fails fails the save (existing path; nothing written) | barrier before writing: a fault since the previous save → no save, training-fault action |
| Weight / optimizer load, BN warmup, dropout seed at start | the load / start fails with its error | — |
| Health reads, test-set evaluation, probes | logged and discarded (existing catches); the ledger fault still triggers the training-fault action | same |
| Lichess bot / UCI / interactive | the move fails with the error (existing behavior), logged | logged |

Exit codes: 0 done, 2 refused, 33 failed, 35 health stop, **36 GPU fault**
(new; same results-writing path as the health stop). The recovery point is
"the last save written before any fault", which the barrier guarantees is the
last save.

### A6. Memory and thermal visibility

- `DispatchSource.makeMemoryPressureSource` (normal / warning / critical) →
  `[MEM]` line on every change.
- `[MEM]` every 10 minutes and on every change: `phys_footprint`,
  `MTLDevice.currentAllocatedSize`, `recommendedMaxWorkingSetSize`, swap used
  and total (`vm.swapusage`), pressure level, `thermalState`.

### A7. Validation

- Unit (pure): the failure rule for every status × handler combination; the
  `[GPU-ERR]` text; the single-buffer rule equals `requireCompleted`; ledger
  order and retention; monitor allowlist on the incidents' captured lines;
  suspension gates (`.divergence` skips autosave).
- Unit (GPU): `runGraph` / `runExecutable` return bit-identical results to
  `graph.run` / `executable.run`, for a small graph and a 400-op graph that
  MPSGraph splits; the completion handler reports for every encode variant
  used (graph, executable; with and without `results:`).
- **Forced-fault experiment (pending; needs a GPU with no training on it):** a
  standalone probe that faults a middle buffer of a split submission (page
  fault from an out-of-bounds read, then a watchdog trip) and records: the
  handler's error, the last buffer's status, the exact system-log text. Can't
  run beside live training — a forced reset would make every trainer a victim.
- **Performance (pending the same idle GPU):** steps/s and plies per hour
  before and after on an idle GPU; acceptance: within 1% for the training step
  and self-play (the added work is one descriptor, one semaphore wait and two
  status reads per submission).
- Live: the next real fault must produce `[GPU-SYSLOG]` and, for training, a
  dump and the abort.

## Part B — Batch identity and content hashes

### B1. What is hashed

- **Identity (every step, all paths):** each sample's absolute position in the
  buffer's write stream (positions written before it, not its ring slot), in
  batch order — 4,096 × 8 bytes, SHA-256 ≈ 0.01 ms.
- **Content (step-line steps, checkpoint saves, crash dumps):** the exact bytes
  the training step consumes — boards, moves, outcomes after the draw-penalty
  rewrite — copied in phase 1 (moves and outcomes are small; boards are
  already copied). SHA-256 ≈ 12 ms, on a background CPU thread; the step line
  waits for it (it finishes long before the GPU step does).

### B2. Identity chain

`chain[n] = SHA-256(chain[n−1] ‖ identity[n])`, restarting at every multiple
of 1,000 trainer steps from `SHA-256("dcm-batch-chain-v1" ‖ window start)`.
Checkpoints land on multiples of 1,000, so an exact resume recomputes the
original run's values without storing anything (no lineage change). A segment
that starts mid-window, and a GUI promotion rewind (which repeats trainer
steps), logs `chain=partial` until the next window start.

**What a match proves, per path:**
- corpus replay — same samples and, with the deterministic feed, same contents;
  a matching content hash confirms the bytes;
- GUI self-play and train-vs-UCI — same positions in the write stream only; the
  buffer's contents differ after a resume (new games), so only the content
  hash says anything about the bytes.

### B3. Where it appears

- Step lines: `batch=<16 hex content> chain=<16 hex>`.
- Checkpoint saves: both in the save's log line.
- `[BATCH-HASH] trainerStep=N chain=…` every 100 trainer steps.
- `results.json` rows; the last 1,000 identities in memory for dumps.
- `scripts/compare_batch_hashes.py <log A> <log B>`: first common step where
  chain or content differs, or "identical through step N".
- Checked: `scripts/` and dashboard parsers read step lines by `key=value`;
  the new keys must not break them (B4).

### B4. Validation

Same arrays → same hash; one changed value → different; window restart and
`partial` labelling; identity uses stream positions (same positions in a
wrapped and an unwrapped ring → same identity); two corpus-replay runs from one
checkpoint give identical chains (extends the existing exact-resume test);
parsers accept the new keys.

## Part C — Crash dump

### C1. Triggers

- A training fault (A5, either layer).
- A non-finite loss / gradient halt.
- **Near miss, no halt:** pre-clip gradient norm ≥ 1,000 × the relative cap's
  reference median (none while the cap has no reference yet). Batch + state,
  no weights; at most one per 1,000 trainer steps.

### C2. Where

`~/Library/Application Support/DrewsChessMachine/CrashDumps/<YYYYMMDD-HHMMSS>-<modelID>-step<N>-<reason>[-2…]/`,
staged under `.tmp`, created exclusively, never overwritten (`FileSafety`); the
GUI-launch orphan sweep covers `CrashDumps/` staging. No automatic deletion.

### C3. Contents, in write order (CPU-side first)

1. `manifest.json`: reason and error text; trainer and segment step; **which
   step the dumped batch belongs to** (for a late trigger the staging batch is
   a later step's — the faulted step's identity is in the identity ring); the
   lineage record; parameter snapshot; LR and momentum; batch identity, content
   hash and chain; the last 1,000 identities; the last 200 step records; ledger
   faults; memory and thermal state (A6); other DrewsChessMachine processes
   (pid and redacted argv).
2. `batch.safetensors` (plain `SafetensorsFile`, not a model file): boards,
   moves, outcomes, each sample's stream position, ring slot, packed
   worker/game id and ply.
3. `log-tail.txt` (session logger flushed first): the last 5,000 lines;
   `system-log.txt`: this process's log entries for the last 120 s; copies of
   `/Library/Logs/DiagnosticReports/gpuEvent-*` reports for this pid, if
   readable.
4. `weights-after.safetensors` (plain `SafetensorsFile`, not a model file — no
   test-set results, no lineage): trainer weights + velocity as they are, and a
   per-tensor non-finite census in `manifest.json`. GPU reads, so last and
   under a 60 s budget on a separate thread; on timeout or error the manifest
   says so and the halt proceeds.
- No per-step weight copy (owner).

### C4. After the dump

The halt proceeds as A5 (same exit codes); `[CRASH-DUMP] wrote <path>`;
`results.json` gets `crash_dump`. A dump failure logs `[CRASH-DUMP-ERR]` and
never blocks the halt.

### C5. Reader

`scripts/dcm_crash_dump.py <dir>`: reason, step, batch sources, non-finite
tensors, memory, GPU faults.

### C6. Validation

Dump on an injected non-finite step and an injected GPU fault; exclusive
staging and `-2` suffix; a write failure and a hanging weights read (stubbed
reader that never returns) don't block the halt; manifest decodes; the dumped
batch's content hash equals the logged one.

## Phases

1. **A1 inventory** — done.
2. **A2** `GPUSubmission` + `GPUStage` + `GPUFaultLedger` + every submission
   converted + `[GPU-ERR]` / `[GPU-SLOW]`; tests. Build, recheck, commit, push.
3. **A3–A6** `GPUFaultMonitor` + barrier + per-path fault actions + `[MEM]`;
   tests. Build, recheck, commit, push.
4. **Part B**; tests. Build, recheck, commit, push.
5. **Part C**; tests. Build, recheck, commit, push.
6. **Docs**: CLAUDE.md (log tags, exit 36, the three layers), training-health
   doc, CHANGELOG. Commit, push.

Running training is never touched; new builds apply to new runs only.

## Decisions

Owner (2026-10-09):
- Training-step GPU fault: full abort — crash dump, halt, exact resume from the
  last checkpoint.
- Audit scope: every GPU-touching call; every one checks its failure and logs
  it (limit for calls with no failure signal: A1).
- No per-step copy of the weights.
- Batch content hashed only at step lines, checkpoints and crash dumps.
- Apple Intelligence: owner asked for it off; macOS 27.2 beta (26B5091g) has no
  switch for it (Settings offers only "Turn Off Siri", which doesn't name the
  model). Left on; owner said not to worry about it.

Mine, under the owner's "make the best decision and note it" rule:
- Self-play: skip the tick on a layer-1 fault (no garbage move is ever
  sampled — the throw comes before sampling); abandon in-progress games on a
  layer-2 fault. Arena: void on any fault in its span.
- GUI training fault reuses `.divergence` (skips autosave) instead of a new
  suspension case — one fewer state, and the gates are already right.
- Exit 36 for a CLI GPU fault, through the health stop's results path.
- Near-miss threshold 1,000 × reference, at most one dump per 1,000 steps.
- Batch chain in 1,000-step windows, no lineage field.
- The barrier (≈2 s synchronous monitor poll) before promotion and every save.
- `weights-after` is a plain safetensors file, not a DCM model file, so the
  test-set rule doesn't apply.

## Review findings and how the plan answers them

| Finding | Answer |
|---|---|
| C1 late fault can't void promotion / saves | barrier before promotion, saves, pointer, rolling out-model (A3, A5) |
| C2 `.gpuFault` would autosave suspect weights | reuse `.divergence`, which skips autosave (A5) |
| H1 promotion-copy failure leaves trainer half-rewound | training-fault action (A5 row) |
| H2 dropout advance is training state | its failure throws out of the step → training fault (done in phase 2) |
| H3 dump can block / wrong batch | CPU parts first, GPU reads last with a 60 s budget; manifest names the batch's step (C3) |
| H4 chain proves less outside corpus replay; rewinds repeat steps | per-path guarantee stated; rewind → `partial`; identity = stream positions (B1, B2) |
| H5 layer 1 unverified, no fault injection | forced-fault experiment listed as pending; needs an idle GPU (A7) |
| H6 layer 3 overstated | A4 rewritten from the logs |
| M1 `[GPU-SLOW]` span includes CPU gaps | per-buffer durations; span reported only (A2) |
| M2 monitor cost / filter / availability | allowlist + fixtures; GUI-process measurement; unavailability recorded (A3) |
| M3 GUI `--train` | through `AutoTrainTermination` (A5) |
| M4 second stop channel | `.divergence` reused; exit 36 through the health stop's results path (A5) |
| M5 content-hash inputs and timing | boards + moves + outcomes after draw penalty; step line waits (B1) |
| M6 no corpus game per slot | stream position recorded per sample; maps to the corpus by feed order (C3) |
| M7 117 calls can't be checked | limit stated, reported to the owner (A1) |
| M8 rows for newly-throwing sites | A5 table rows |
| M9 abandoning games on layer 1 unnecessary | layer 1 → skip tick only (A5) |
| M10 dump weights vs test-set rule | plain safetensors, not a model file (C3) |
| M11 no performance criteria | added, pending an idle GPU (A7) |
| L1 cross-references | fixed |
| L2 names drifted; handler can fire late | plan uses the code's names; handler awaited (A2) |
| L3 baseline finish assumed | `verifyPendingValueBaseline` waits (A2) |
| L4 "164" overstated | A1 says what the count includes |
| L5 dump collisions / staging sweep / no cap | `-2` suffix, sweep, near-miss rate limit (C1, C2) |
| L6 secrets in process list | redacted argv (C3) |
| L7 keep macOS's report | copy `gpuEvent-*` (C3) |
| L8 flush logger | flushed first (C3) |
| L9 parser compatibility | checked in phase 4 (B3) |
| L10 possible baseline input-buffer hazard | follow-up below |
| Process killed outright (assertion, jetsam) | out of scope; nothing in-process can write then |

## Follow-ups

- **L10 (unverified hazard, existing):** the no-wait baseline reads the cached
  input array for its batch size; another job on the trainer network with the
  same batch size before the baseline finishes would overwrite that input
  (`ChessNetwork.swift` batch input entries; candidate: the legal-mass probe's
  fallback in `ChessTrainer.swift`). To be checked separately.
- **Arena GPU work:** all three hangs began in R-fixedlr's process early in an
  arena (two concurrent `executable.run` per tick beside self-play, training
  and the baseline, audit §5). `[GPU-ERR]` / `[GPU-SLOW]` now name the stage
  of a reachable failed or slow buffer.
- Forced-fault experiment and performance measurement (A7), when the GPU is
  idle.
