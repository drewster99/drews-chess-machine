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

B-siluall's exact resume from step 45,000 replayed the step-~45,975 batch at
22:25 with no fault — consistent with the GPU reset, not the batch. Without
batch hashes that "same batch" is an assumption (Part B).

Memory at the time of writing: swap 25.3 GB used of 26.0 GB
(`sysctl vm.swapusage`, 22:3x). Whether memory pressure contributed to the hang
is unknown; nothing records it (A5).

## Part A — GPU and memory error audit, then handling

### A1. Inventory (read-only, first)

List every GPU submission site and, for each: synchronous `graph.run` /
`executable.run` or our own command buffer; which command buffers it can
produce (`MPSCommandBuffer` may `commitAndContinue`, so the root buffer and
`mpsCommandBuffer.commandBuffer` can differ); whether status is checked;
when; and what consumes the output. Sites found so far (2026-10-09 grep):
`ChessTrainer.swift` (training step encode at ~7148, working-weight sync
~7218, ~11 `graph.run` sites for probes / exports / KL probe), `ChessNetwork.swift`
(inference `executable.run` ~1490, value baseline encode ~1703, weight load
~1976 — the one site that checks both buffers, ~6 `graph.run`),
`DropoutPhiloxState.swift`, `ChessMPSNetwork.swift`. The inventory table goes
into this file.

Known gaps already visible:

- **Training step** checks only the root `mtlCommandBuffer.status`, not the
  continued `mpsCommandBuffer.commandBuffer` (the weight load checks both).
- **Value baseline** is committed without a wait and checked at the *next*
  call (`lastBaselineCommandBuffer`) — after the training step that used it
  has already consumed garbage. A discarded baseline gives garbage advantages,
  which fits a 10M gradient or a NaN.
- **Synchronous `graph.run`** (self-play / arena inference, probes) returns
  results with no status at all; a discarded buffer's outputs are used as if
  valid.
- **No logging**: a `gpuCommandFailed` is thrown, never logged with the Metal
  error code, and nothing mirrors the system's IOGPU / Metal messages.

### A2. One status rule

Every command buffer we commit is checked (root and continued) before any of
its outputs are read or consumed by later GPU work. One helper
(`ChessNetwork.requireCompleted`, extended to take an `MPSCommandBuffer`)
used everywhere. Synchronous `graph.run` sites move to encode + commit + wait +
check, or to an execution descriptor whose completion handler reports the
error (API to verify on macOS 27).

### A3. The baseline before the step

The training step checks the baseline's command buffers before it reads its
own results. A failed baseline fails the step.

### A4. What happens on a GPU fault

- Log `[GPU-ERR] stage=… status=… code=… (kIOGPUCommandBufferCallbackError…) domain=… pid=…`
  with `MTLCommandBufferDescriptor.errorOptions = .encoderExecutionStatus` so
  the error names the encoder that faulted.
- **Training step or its baseline faulted:** the weights may be partly written
  → halt with a crash dump (Part C). Recovery stays an exact resume.
  (Owner decision 1.)
- **Self-play / arena inference faulted:** discard that tick's evaluations
  and run the tick again, logging it; no move is played from garbage.
  (Owner decision 2.)
- **Probe faulted:** discard the probe result, log, continue.

### A5. Memory and thermal visibility

- `DispatchSource.makeMemoryPressureSource` (warning / critical) → `[MEM]` line
  on every change.
- On step lines and every `[MEM]` line: `phys_footprint`,
  `MTLDevice.currentAllocatedSize`, `recommendedMaxWorkingSetSize`, swap used
  and total (`vm.swapusage`), memory-pressure level, `thermalState`.
- `[MEM]` on every thermal-state change.

### A6. Mirror the system's GPU messages

Poll `OSLogStore(scope: .currentProcessIdentifier)` (our own process's entries,
no entitlement needed — the 21:37 errors were attributed to our pids) for
Metal / IOGPU error entries and copy them into the session log as
`[GPU-SYSLOG]`. Verify the API on macOS 27 before building on it.

### A7. Validation

- Inventory table complete: every site has a checked status or a written
  reason.
- Unit tests: the helper throws on an `.error` root, an `.error` continued
  buffer, and passes `.completed`; the fault policy (halt / re-run tick /
  discard probe) driven by an injected `gpuCommandFailed`.
- A real fault can't be forced on demand; the first real one must produce
  `[GPU-ERR]` + `[GPU-SYSLOG]` lines and (for training) a dump.

## Part B — Batch hashes

### B1. What is hashed

The exact sampler output the training step consumes, in batch order: boards
(`Float`), moves (`Int32`), outcomes (`Float`). Not the value baseline (it
depends on the weights). SHA-256 (CryptoKit); logs show the first 16 hex
digits, dumps keep all 64.

### B2. Cost

Boards dominate: batch 4096 × 30 planes × 64 squares × 4 bytes ≈ 30 MB. SHA-256
at ~2 GB/s ≈ 15 ms per step, computed on the existing `boardsCopy` off the
trainer queue, in parallel with the baseline forward, so it doesn't lengthen a
~3–4 s step. Every step, every path; no parameter (it never changes training).

### B3. Hash chain

`chain[n] = SHA-256(chain[n−1] ‖ batch[n])`, carried in the trainer
checkpoint's lineage record (with the RNG states) and restored by an exact
resume. A matching chain value at any common step proves every batch since the
checkpoint matched — not just the logged ones.

### B4. Where it appears

- Every step line: `batch=<16 hex> chain=<16 hex>`.
- `[BATCH-HASH] trainerStep=N batch=… chain=…` every 100 trainer steps (step
  lines are partly time-scheduled, so a resumed run's step lines land on other
  steps; the 100-step cadence always overlaps).
- `results.json` rows; the last 1,000 per-step hashes held in memory for dumps.
- `scripts/compare_batch_hashes.py <log A> <log B>`: first common step where
  the chains differ, or "identical through step N".

### B5. Validation

- Same arrays → same hash; one changed float → different hash.
- Chain survives save → exact resume (unit test on the lineage round-trip).
- Two corpus-replay runs from one checkpoint, N steps each: identical chains
  (extend the existing exact-resume test).
- Live: B-siluall-style resume shows matching `chain=` at the overlapping
  100-step marks.

## Part C — Crash dump

### C1. Triggers

- A halt on a non-finite loss / gradient.
- A training-step GPU fault (A4).
- **Near miss, no halt:** a pre-clip gradient norm ≥ a threshold × reference
  (R-replay's 21:37 step was 37 million ×). Writes batch + state, not weights,
  and training continues. (Owner decision 3: threshold; proposal 1,000×.)

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
- **Weights before the step: not kept by default.** They are reproducible by
  an exact resume from the last checkpoint, which the batch chain now proves
  reaches the same batch. An optional per-step copy costs a GPU → CPU copy of
  weights + velocity every step (~65 MB for 8.3M parameters). (Owner
  decision 4; proposal: off.)
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

1. **A1 inventory** (read-only) → table in this file. Commit.
2. **A2–A6** status rule, fault policy, `[GPU-ERR]`, `[MEM]`, `[GPU-SYSLOG]`;
   tests. Build, commit.
3. **Part B** batch hashes + chain + compare script; tests. Build, commit.
4. **Part C** crash dumps + reader; tests. Build, commit.
5. **Docs**: CLAUDE.md log tags, `documentation/training-health-alarms.md`,
   CHANGELOG. Commit.

Running training is never touched; new builds apply to new runs only.

## Owner decisions

1. **Training-step GPU fault:** halt + dump, recover by exact resume
   (proposed) — or roll back in memory to the last checkpoint and continue.
2. **Self-play / arena inference fault:** re-run the tick (proposed) — or halt.
3. **Near-miss dump threshold:** pre-clip norm ≥ 1,000 × reference (proposed).
4. **Per-step pre-step weight copy for dumps:** off (proposed).
5. **This Mac:** macOS on-device language-model requests started 9 s before the
   hang. Turning Apple Intelligence off on the training Mac is the owner's
   call; this plan doesn't depend on it.
