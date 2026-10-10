# GPU fault forensics — plan

**Status: proposed 2026-10-09, not started.** Owner request (2026-10-09, after
the 21:37 incident below):

1. When training has to stop (a NaN or similar), write a crash dump that saves
   everything needed to investigate.
2. Hash every batch (at least the logged ones), so a resumed run can prove it
   saw the same batches instead of assuming it.
3. Audit GPU and memory error handling: we may not be watching, or logging,
   every GPU error.

Nothing here is implemented until the owner says start.

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
| 21:37:09.633 | osanalyticshelper | wrote `gpuEvent-DrewsChessMachin-2026-10-09-213709.ips` (not present afterwards) |

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
is unknown; nothing records it (A5).

## Part A — GPU and memory error audit, then handling

### A1. Inventory — done

`GPU_CALL_AUDIT_2026-10-09.md` (this folder): 164 call sites. All GPU work is
MPSGraph (no app `makeBuffer`, compute pipelines or MPS kernels). 24 GPU
submissions: none fully checked, 4 partly (training step, working-weight sync,
value baseline, weight load), 20 not at all (19 synchronous `graph.run`, 1
`executable.run` — self-play / arena / probe inference). 9 nil- or
throw-returning creations (device, queues, `makeCommandBuffer`, capture
start), all checked. 117 calls have no failure signal (`MPSNDArray`,
`MPSGraphTensorData`, `readBytes` / `writeBytes`, `compile`). No site logs the
Metal error.

Facts from the audit's probes that shape the design:

1. MPSGraph `encode` splits work across several command buffers
   (`commitAndContinue`): 1 extra buffer at ~100 ops, 4 at ~400, 10 at ~8,000.
   A training step spans several.
2. After a split, `MPSCommandBuffer.commandBuffer` and `.rootCommandBuffer`
   both name the newest buffer; middle buffers are unreachable. So "check
   every buffer" is impossible through public API.
3. Continued buffers don't inherit `errorOptions`.
4. Synchronous `graph.run` returns no error at all. `executable` / graph
   `encode` / `run` with an execution descriptor gets a
   `completionHandler(…, NSError?)`; whether that error covers a middle
   buffer, or a hang / victim, is unverified (a real fault can't be forced
   safely beside live training).

So detection has three layers, because no single public signal is complete.

### A2. Layer 1 — one checked submission helper (`GPUWork`)

Every GPU submission goes through one helper:

- creates the command buffer itself, with `errorOptions = .encoderExecutionStatus`;
- encodes (graph or executable) with an execution descriptor whose
  `completionHandler` records the `NSError`;
- commits, waits, then checks the first buffer (our reference), the last
  (`mpsCommandBuffer.commandBuffer`) and the handler's error;
- on any failure throws `GPUWorkError(stage:, firstStatus:, lastStatus:,
  error domain / code / userInfo, encoder statuses)` and logs one
  `[GPU-ERR] stage=… first=… last=… handler=… code=… (kIOGPUCommandBufferCallbackError…)`.

It also measures each submission's GPU time from the first and last
buffers' `gpuStartTime` / `gpuEndTime`, and logs `[GPU-SLOW] stage=… gpuMs=…`
for any submission over 5 s (the hang watchdog fires on a long-running
buffer, so this names the stage that comes closest).

All 19 `graph.run` sites, the `executable.run` site and the four `encode`
sites move to it (`graph.run(with:queue…)` is encode + commit + wait inside,
so results are unchanged; tests pin that). The value baseline keeps its
no-wait overlap: the training step checks the baseline's buffers and handler
before it reads its own results (the step's buffers follow the baseline's on
the same queue, so the baseline has finished by then).

### A3. Layer 2 — the process's own GPU fault messages (`GPUFaultMonitor`)

macOS logs every hang / victim / page fault in the affected process
(`IOGPUMetalError`, `kIOGPUCommandBufferCallbackError…`), including for
buffers the app can't reach. `OSLogStore(scope: .currentProcessIdentifier)`
reads them without an entitlement (verified 2026-10-09 with a standalone
probe: found its own error entry; 1.8–2.0 s per query with three trainers
running). One monitor per process polls every 10 s on a utility queue:

- logs each new entry once as `[GPU-SYSLOG] <time> <message>`;
- keeps fault times; `faults(since:)` answers in memory, no query.
- A failed `OSLogStore` open or query logs `[GPU-SYSLOG] unavailable: …`
  once and the run continues (layers 1 and 3 still apply).

Detection lag is up to ~12 s, so its consumers act on faults that already
happened (A4).

### A4. Layer 3 — numeric guard (exists)

The relative gradient cap already clips a faulted step's gradient (3 of 3
hangs today showed 2.35× … 32,736,274× spikes, all clipped); the non-finite
check halts on NaN. Both stay; Part C dumps on either.

### A5. What each path does on a fault

| Where | Immediate (layer 1) | Late (layer 2) |
|---|---|---|
| Training step / baseline / working-weight sync (all paths) | crash dump (Part C), then abort: CLI exits 36 (new: GPU fault), GUI suspends training (`trainingSuspension = .gpuFault`, like a health stop: arenas and promotion refused, self-play and autosave continue) | same — a fault at any time while the trainer is active means the weights since may be poisoned; recovery is an exact resume from the last checkpoint |
| Self-play inference (GUI) | abandon the games in that tick (their next move would come from garbage); log count | abandon every game in progress when the fault is reported; games that ended inside the lag window are already in the buffer — logged as possibly affected, not removed |
| Arena (GUI) | void the arena: no promotion, no "kept", logged `[ARENA] voided: GPU fault` | void the arena if a fault time falls between its start and end |
| Weight / optimizer-state export for a save | the save fails through its existing failure path (nothing written) | the save that overlapped the fault is logged as possibly affected (not deleted) |
| Weight / optimizer-state load, BN warmup, dropout-state writes | the load / resume fails with its error | logged |
| Probes, health reads, test-set evaluation | result discarded, logged; training continues | logged |
| Lichess bot / UCI / interactive | the move fails with the error (existing behavior) | logged |

Exit codes: 33 stays "failed", 35 health stop; new 36 = GPU fault (CLI).

### A6. Memory and thermal visibility

- `DispatchSource.makeMemoryPressureSource` (warning / critical / normal) →
  `[MEM]` line on every change.
- `[MEM]` every 10 minutes and on every change: `phys_footprint`,
  `MTLDevice.currentAllocatedSize`, `recommendedMaxWorkingSetSize`, swap used
  and total (`vm.swapusage`), memory-pressure level, `thermalState`.

### A7. Validation

- Unit tests: `GPUWork` returns identical results to `graph.run` /
  `executable.run` on small graphs; throws and logs on an injected failed
  status / handler error (the failure check is a pure function over
  statuses + error, tested directly); the A5 policy for each path driven by
  an injected fault; the monitor's entry parsing and de-duplication from
  fixed log text.
- Live: the first real fault must produce `[GPU-SYSLOG]` (always) and, when a
  reachable buffer or the handler reports it, `[GPU-ERR]`; a training fault
  must produce a dump and the abort.

## Part B — Batch hashes

### B1. What is hashed

The exact sampler output the training step consumes, in batch order: boards
(`Float`), moves (`Int32`), outcomes (`Float`). Not the value baseline (it
depends on the weights). SHA-256 (CryptoKit); logs show the first 16 hex
digits, dumps keep all 64.

### B2. Cost (owner 2026-10-09: no per-step content hashing)

Measured on this Mac (Python `hashlib`, one core, three trainers running):
SHA-256 of a whole batch (4096 × 30 × 64 floats ≈ 30 MB) takes 11.9 ms; of the
batch's 4,096 sample indices (16 KB), 0.006 ms. So:

- **Every step:** hash only the batch's identity: the sampler's buffer-slot
  indices in batch order, plus the buffer's write position (how much has been
  fed). 0.006 ms; no measurable cost.
- **Content hash only where it's recorded:** step-line steps, checkpoint saves
  and crash dumps hash the full batch bytes (boards, moves, outcomes). It runs
  on a background CPU thread from the existing `boardsCopy`, so the trainer
  never waits; the step line is written after the step, by which time the hash
  is done.

### B3. Identity chain

`chain[n] = SHA-256(chain[n−1] ‖ identity[n])`, restarting at every multiple
of 1,000 trainer steps from a fixed seed (`SHA-256("dcm-batch-chain-v1" ‖
window start step)`). Checkpoints land on multiples of 1,000, so an exact
resume and the original run compute the same chain values without storing
anything in the checkpoint (no lineage schema change). A matching chain at a
common step proves every batch in that window drew the same samples; a
matching content hash at a logged step proves the bytes matched. A segment
that starts mid-window (a resume from a final or autosave not on a multiple
of 1,000) logs `chain=partial` until its first window boundary.

### B4. Where it appears

- Every step line: `batch=<16 hex content> chain=<16 hex>`.
- Checkpoint saves: both values in the save's log line.
- `[BATCH-HASH] trainerStep=N chain=…` every 100 trainer steps (chain only;
  step lines are partly time-scheduled, so a resumed run's step lines land on
  other steps; the 100-step cadence always overlaps).
- `results.json` rows; the last 1,000 per-step identities held in memory for
  dumps.
- `scripts/compare_batch_hashes.py <log A> <log B>`: first common step where
  the chains or content hashes differ, or "identical through step N".

### B5. Validation

- Same arrays → same hash; one changed float → different hash.
- Window restarts: same identities → same chain from any window start; a
  partial window is labelled.
- Two corpus-replay runs from one checkpoint, N steps each: identical chains
  (extend the existing exact-resume test).
- Live: B-siluall-style resume shows matching `chain=` at the overlapping
  100-step marks.

## Part C — Crash dump

### C1. Triggers

- A halt on a non-finite loss / gradient.
- A GPU fault while the trainer is active (A5, either layer).
- **Near miss, no halt:** a pre-clip gradient norm ≥ a threshold × reference
  (R-replay's 21:37 step was 37 million ×). Writes batch + state, not weights,
  and training continues. Threshold: 1,000 × the relative cap's median
  reference (decided 2026-10-09; today's three hang-corrupted steps were
  32.7M ×, 10.3M × and 37.5M ×; ordinary clips are under 10 ×).

Every training path: corpus replay, train-vs-UCI, GUI Play-and-Train /
`--train`.

### C2. Where

`~/Library/Application Support/DrewsChessMachine/CrashDumps/<YYYYMMDD-HHMMSS>-<modelID>-step<N>-<reason>/`,
staged under `.tmp`, created exclusively, never overwritten (`FileSafety`).
No automatic deletion.

### C3. Contents

- `manifest.json`: reason and error text; trainer and segment step; the full
  lineage record (run, segment, parameters, build, device, RNG states); LR and
  momentum at the step; the batch hash and chain, plus the last 1,000 per-step
  hashes; the last 200 step records (losses, pre-clip gradient norm, clips);
  memory and thermal state (A5); other DrewsChessMachine processes running;
  GPU errors seen (A4, A6).
- `batch.safetensors`: boards, moves, outcomes, the value baseline as computed
  (if readable), per-position advantages (when read back), and each sample's
  source: buffer slot plus game identity (corpus shard + game + ply for corpus
  replay; game serial + ply for self-play).
- `weights-after.safetensors`: trainer weights + optimizer velocity as they
  are after the failing step, and a per-tensor non-finite census in the
  manifest (locates where the damage is).
- **Weights before the step: never copied per step** (owner 2026-10-09: too
  slow). They are reproducible by an exact resume from the last checkpoint,
  which the batch chain proves reaches the same batch.
- `log-tail.txt`: the session log's last 5,000 lines; `system-log.txt`: our
  process's system-log entries for the last 60 s (A6).

### C4. After the dump

Halt exactly as today (same exit codes); log `[CRASH-DUMP] wrote <path>`;
`results.json` gets `crash_dump`. A dump that fails to write logs
`[CRASH-DUMP-ERR]` and never blocks the halt.

### C5. Reader

`scripts/dcm_crash_dump.py <dir>`: one-page summary (reason, step, batch
sources, non-finite tensors, memory, GPU errors). Re-running the failing batch
on the pre-step weights with per-tensor gradient norms is a follow-up, not in
this plan.

### C6. Validation

- Unit tests: dump written on an injected non-finite step; staging never
  overwrites; a write failure doesn't block the halt; the manifest decodes.
- A dump's batch, re-fed to the sampler-free trainer path, reproduces its
  recorded batch hash.

## Phases

1. **A1 inventory** — done (`GPU_CALL_AUDIT_2026-10-09.md`).
2. **A2** `GPUWork` + every submission converted + `[GPU-ERR]`; tests. Build,
   recheck, commit, push.
3. **A3–A6** `GPUFaultMonitor` + per-path fault policy (exit 36, GUI
   `.gpuFault` suspension, self-play abandon, arena void) + `[MEM]`; tests.
   Build, recheck, commit, push.
4. **Part B** batch identity + content hashes + chain + compare script; tests.
   Build, recheck, commit, push.
5. **Part C** crash dumps + reader; tests. Build, recheck, commit, push.
6. **Docs**: CLAUDE.md log tags and exit codes,
   `documentation/training-health-alarms.md`, CHANGELOG. Commit, push.

Running training is never touched; new builds apply to new runs only.

## Owner decisions

Decided by me under the owner's "make the best decision and note it"
standing rule (2026-10-09), owner may revisit:

- **Self-play / arena fault:** abandon the affected games, void the arena
  (A5). Re-running a tick was the first proposal, but most faults are seen
  only through the monitor, seconds late, when the moves are already played.
- **Near-miss dump threshold:** 1,000 × reference (C1).
- **Batch chain without a lineage field** (B3): windows of 1,000 trainer
  steps instead of a chain stored in checkpoints.

Decided (owner, 2026-10-09):

- **Training-step GPU fault: full abort** — crash dump, halt, recover by exact
  resume from the last checkpoint.
- **Audit scope: every GPU-touching call** — `MTLCreateSystemDefaultDevice`,
  command-queue and command-buffer creation, every `MTLBuffer` / heap /
  texture allocation, `MPSGraph` `run` / `runAsync` / `encode` / `compile`,
  `MPSGraphExecutable` `run` / `encode`, `MPSGraphTensorData` / `MPSNDArray`
  creation and reads/writes. Every one checks its failure (nil, status, error)
  and logs it.
- No per-step copy of the weights.
- Batch content hashed only at step lines, checkpoints and crash dumps (B2).
- GPU errors are logged (Part A).
- Apple Intelligence turned off on the training Mac, to rule it out.
  **Not possible from Settings on macOS 27.2 beta (26B5091g):** the Siri pane
  has no Apple Intelligence switch; the only control is "Turn Off Siri", whose
  dialog lists Siri, the Siri app, Visual Intelligence and HomePod, not the
  on-device model. Left unchanged (2026-10-09 22:5x); owner to decide.

## Follow-up suspect: arena GPU work

All three hangs began in R-fixedlr's process early in an arena. The audit
(section 5) found two concurrent `executable.run` calls per arena tick (one
per network, own queues, ≤ ~200 positions each) beside self-play, training
steps and the baseline in the same process; nothing logs their GPU time, and
which buffer hung is unknown (none of the four checked buffers reported it).
After phase 2, `[GPU-ERR]` names the stage of any reachable failed buffer, and
`GPUWork` records each submission's GPU time, so the next hang identifies its
source.
