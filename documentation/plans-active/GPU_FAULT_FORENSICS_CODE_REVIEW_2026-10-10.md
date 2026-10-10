# Independent code review: GPU fault forensics (2026-10-10)

I read it only: no edits, no build, no tests. I found no unchecked GPU submission left. A grep finds no `MPSGraph.run`, `MPSGraphExecutable.run` or raw `commit` outside `GPUSubmission`; the only match is a doc comment. The batch hash draws no randomness and changes no training state.

## High

**H1. A continue after Stop forgets the fault, so training resumes on suspect weights and saves succeed again.**
- **Where:** `App/SessionController+Training.swift:665`, together with `stopRealTraining` (`:2648`).
- **Scenario:**
  1. A GPU fault suspends training as `.divergence`.
  2. The user presses Stop, then Play-and-Train. That is a continue on the in-memory trainer, and the banner itself says "…or Stop".
  3. `trainingSuspension = nil` (`:565`), and `GPUFaultWatch.startForRun()` sets a new baseline equal to `ledger.latestSequence`, so the fault no longer counts.
  4. Training continues from the post-fault weights and optimizer/RNG state. Periodic and promotion saves pass the barrier and move `LastSessionPointer`.
- **Why it matters:** this breaks the owner decision "full abort; exact resume from the last checkpoint" and the guarantee that the last save comes before any fault.
- **Fix:** add a trainer-tainted flag that survives Stop, set it in `handleGPUFault`, and clear it only when weights load from disk or a new network is built. Refuse a continue while it is set. Alternatively, carry the previous watch's baseline into a continue.

## Medium

**M1. The post-promotion save has no barrier after the promotion's own GPU work.**
- **Where:** `SessionController+Arena.swift` — the arena barrier is at about `:408`; the copy at `:498–510`; the detached save at `:860`.
- **Scenario:**
  - The barrier runs before `candidateInference.exportWeights()`, `champion.loadWeights` and `rewindToArenaStart`.
  - A fault that only the system log reports (an unreachable middle buffer discarded during that export or rewind) is missed.
  - `promotedChampionWeights` is then written to the `-promote` session and `LastSessionPointer` moves to it.
- **Fix:** run `gpuFaultBarrier()` after the copy, before `promotionSaveWillRun` (or before building the cut). On a fault, call `failBeforeWriting` and `handleGPUFault`.

**M2. A non-finite halt caused by a GPU reset is classified as non-finite, not as a GPU fault.**
- **Where:**
  - `CLI/CorpusReplayRunner.swift:2172`
  - `CLI/TrainVsUciRunner.swift:1137`
  - `App/SessionController+Training.swift` around `:1374`
- **Scenario:** this is exactly B-siluall at 21:37. The non-finite check throws before the system-log entry arrives (it can be up to one poll late).
  - The CLI exits 33 with no `results.json`, no `gpu_faults` and no `gpu_fault` reason.
  - The dump's reason is `non-finite`.
  - In the GUI, `handleGPUFault` is never called.
- **Fix:** on `nonFiniteLoss`, await `faultWatch.barrier()` first. If it finds a fault, take the GPU-fault path (exit 36 / `gpu_fault`) and note the non-finite values in the detail.

**M3. `handleGPUFault` writes the dump before it suspends or claims termination.**
- **Where:** `App/SessionController+GPUFault.swift:28–51`.
- **Scenario:**
  - The dump can take several seconds of synchronous work plus up to 60 s of weights read.
  - When the fault was seen by a save barrier or an arena, `trainingSuspension` is still nil throughout. An arena can start, pause the gates and export the suspect trainer into the candidate, competing with the dump's weight read on the trainer queue.
  - For `--train`, another ending (time limit) can claim first, and the GPU fault is not the recorded reason.
- **Fix:** call `suspendTrainingOnDivergence` (interactive) or `termination.claim()` (`--train`) before awaiting the dump. Then call `writeResultsAndExit` after it.

**M4. Promote Trainee Now runs its barrier too early, and a failed copy is not treated as a fault.**
- **Where:** `App/SessionController+ManualPromote.swift:87` and `:175–183`.
- **Scenario:**
  - The barrier runs before the gates pause and before `exportWeightsWithCompletedSteps`, leaving a ~2 s+ window. A system-log-only fault in the export loads garbage into the champion, and self-play plays it.
  - If the copy fails with `gpuCommandFailed`, the code only resumes the gates. The champion may be left mid-load: the load gate is still set, so every self-play tick throws.
- **Fix:** move the barrier to after both gates are acquired, immediately before the export. Add a second barrier after `champion.loadWeights`. Route a `gpuCommandFailed` copy error to `handleGPUFault`.

**M5. The crash dump does seconds of blocking work on the cooperative pool, and reads the whole session log into memory.**
- **Where:** `Training/CrashDump.swift:119–157`, `:214–226`, `:229–244`, `:280–301`.
- **Scenario:**
  - `writeMemoryParts` runs synchronously inside an async function: `SessionLogger.flush()` (`queue.sync`), `String(contentsOf:)` of the entire log (multi-day logs can be very large), an OSLogStore query over 120 s, `ps` plus `waitUntilExit`, and the file writes.
  - The near-miss dump runs this inline in every training loop. That breaks the CLAUDE.md rule "No long synchronous work inside a Task".
- **Fix:**
  - Run the memory parts on a utility `DispatchQueue` behind a continuation.
  - Read the log tail by seeking backwards from the end of the file (for example, the last 4 MB).

**M6. Tests don't cover the completion handler for the variants production uses.**
- **Where:** `DrewsChessMachineTests/GPUSubmissionTests.swift:152–202`.
- **Scenario:**
  - Only `results: nil` executables and `targetOperations: nil` graphs are tested.
  - Untested: the value baseline (`executable.encode(... results: [resultTD])`), and graph encodes with `targetOperations` (weight load, working-weight sync, dropout advance, master sync).
  - If MPS doesn't call the handler for one of these, every such submission waits 10 s and then fails as `handler=did-not-report`. Every training step would then become a "GPU fault".
- **Fix:** add GPU tests for both variants. Plan section A7 already requires "with and without `results:`".

**M7. The system-log monitor is fragile and costly.**
- **Where:** `Network/GPUFaultMonitor.swift:178–183` and `:104–106`.
- **Problems:**
  - One failed `getEntries` permanently disables that layer for the life of the process; there is no retry.
  - In the GUI it keeps polling after Stop, forever: about 2 s per 10 s, roughly 20% of one core, with no run to protect.
  - `start()` does a `queue.sync` before its "already started" guard. Each Play-and-Train start therefore blocks the main actor behind any poll in flight (about 2 s). `MemoryStatusMonitor.start()` has the same pattern.
- **Fix:**
  - Retry with backoff, and log `[GPU-SYSLOG] unavailable` only after N consecutive failures.
  - Suspend the timer when no run is active.
  - Check whether the monitor has started (`availabilityBox`) before `queue.sync`, and open `OSLogStore` off the main actor.

## Low

1. **Stale watch after a load.** `SessionController.swift:352` (`gpuFaultWatch` is never reset).
   - After a fault, then Stop, then loading a good session from disk, File ▸ Save Session and Promote Trainee Now are refused "since this run started" even though no run is active.
   - Fix: reset the watch on load or new network (alongside the H1 flag).
2. **Save Champion as Model has no barrier.** `handleSaveChampionAsModel` (`SessionController+Checkpoint.swift:23`); the plan says "every save". Fix: add the barrier.
3. **Train-vs-UCI labels some failed steps' batches with the wrong step.** `TrainVsUciRunner.swift:1134`.
   - A `dropoutAdvance` or `workingWeightSync` failure happens inside step N+1, but the batch is recorded as step N.
   - Fix: decide by "the error came out of `trainStep`" (as corpus replay does), not by stage.
4. **Command-buffer creation failure isn't recorded.** `GPUSubmission.swift:96–100`.
   - `commandBufferCreationFailed` logs `[GPU-ERR]` but writes no ledger entry. The CLI then exits 33 and the GUI suspends it as a generic divergence.
   - Fix: record it in the ledger and treat it as `gpuCommandFailed`.
5. **GUI `--train` results show a stale monitor state.** `SessionController+Training.swift:673`.
   - `gpu_faults` is set at start. A later monitor failure followed by a normal ending (time or step limit) still reports `available`.
   - Fix: refresh the report inside `AutoTrainTermination.writeResults`.
6. **The batch-hash chain survives a session load.** `ChessTrainer.swift:1664`.
   - The GUI trainer outlives runs. If a session is loaded at the trainer's current step, the next step extends the old chain instead of reporting `partial`.
   - Fix: add `BatchHashChain.reset()` on weight load or resume.
7. **gpuEvent report copying has two gaps.** `CrashDump.swift:250–256`.
   - It filters by name prefix, not pid, so other DrewsChessMachine processes' reports are copied.
   - It only looks in `/Library/Logs/DiagnosticReports`, not `~/Library/Logs/DiagnosticReports`.
8. **`ps` parsing breaks on spaces.** `CrashDump.swift:297–300`.
   - Splitting on spaces drops binaries whose paths contain spaces.
   - It can also misalign `--flag value` pairs for redaction.
   - Fix: use `sysctl KERN_PROCARGS2` or `proc_pidpath` with argv.
9. **Self-play abandons games more than once per reset.** `BatchedSelfPlayDriver.swift:278`.
   - One reset produces several ledger records across one or two polls, so fresh games get abandoned two or three times.
   - Fix: abandon only for faults newer than the last abandon time.
10. **The GPU-fault exit can hide a failed autosave.** `CorpusReplayRunner.swift` around `:2398`.
    - It skips `requireLastSaveSucceeded`, then claims the last save is the fault-free recovery point even if the last autosave had failed (one failure is tolerated).
    - Fix: say so in the exit message and in `results.json`.
11. **Hash backlog can hold a lot of memory.** `BatchHashChain` on a utility queue.
    - Each queued step holds about 30 MB of board copies. Waits happen only at `[BATCH-HASH]` steps, so a starved queue could hold up to about 3 GB.
    - Fix: cap the backlog, for example by waiting when more than N hashes are queued.
12. **Test-run dumps go to the real folder.** `CrashDumpWriter.dump` has no directory parameter, so a runner test that hits a near miss writes into the real `~/Library/…/CrashDumps`.

## Checked, no issue found
- **Batch-hash ordering:** a serial queue with FIFO submit; entry and recent reads wait for queued hashes.
- **Window maths:** resume at a multiple of 1,000, mid-window starts and the rewind-to-`partial` case are correct.
- **Locks and the handler wait:** `waitUntilCompleted` inside `weightAccessLock`, `verify` after it. The 10 s handler wait runs inside the lock only in `defer`-guarded sections, and only on failure.
- **Value baseline:** verified by the consuming step before any readback.
- **Gate holds:** released on every refusal path.
- **Dump inputs:** `GradientNormHistory` is always finite, so encoding the manifest can't throw on NaN.
- **Fault-path exits:** corpus-replay and train-vs-UCI fault paths tear down the producer and engines, write results and exit 36.

## Summary

The fault detection layers are sound and complete. The gaps are in the policy: a few paths still let suspect state continue or get persisted. The most important is H1: a continue after Stop silently drops the fault and resumes training on the post-fault weights. Next:
- the post-promotion save and Promote Trainee Now have no barrier after their own GPU exports (M1, M4);
- a GPU reset that shows up first as a NaN is classified as non-finite (M2);
- `handleGPUFault` dumps before it suspends (M3).

Operationally, the crash dump's synchronous I/O and full-log read (M5), the system-log monitor's permanent disable and endless polling (M7), and the untested handler variants (M6) should be fixed before relying on this in long runs. Everything else is low-severity accuracy or tidiness.


## How each finding was answered

| Finding | Fix |
|---|---|
| H1 continue after Stop drops the fault | `GPUFaultTaint` (trainer / champion parts) outlives Stop; a start that keeps the trainer is refused while it is tainted, one that forks the champion while the champion is; cleared when the weights are replaced (start from a loaded session or the champion; network built; model or session loaded) |
| M1 post-promotion save has no barrier after the copy | a second barrier after the promotion's copy; on a fault no post-promotion save, and the fault is handled with the champion marked tainted |
| M2 NaN from a GPU reset classified as non-finite | every path runs the barrier on a non-finite halt; a fault makes it a GPU fault (exit 36 / `gpu_fault`) |
| M3 dump before suspend / claim | `handleGPUFault` records the taint and suspends (or claims the `--train` ending) before the dump; `writeResultsAndExit` after it |
| M4 Promote Trainee Now barrier too early; copy failure | barrier again under the pause right before the export, and after the copy; a copy fault is handled with the champion tainted |
| M5 dump does blocking I/O on the cooperative pool; reads the whole log | the in-memory parts run on a utility `DispatchQueue` behind a continuation; the log tail reads at most the last 8 MB |
| M6 handler untested for production variants | tests for executable `results:` (the baseline's form) and graph `targetOperations` (loads, syncs) |
| M7 monitor permanently disabled / polls after Stop / blocks start | retries every poll (unavailable after 3 consecutive failures, available again on success); paused at Stop, resumed at start; `start()` and `MemoryStatusMonitor.start()` return at once (work on their queues) |
| L1 stale watch after load | the watch is cleared at Stop; suspect weights are tracked by the taint |
| L2 Save Champion as Model unguarded | refused while the champion is tainted |
| L3 vs-UCI batch step by stage | every stage inside a training step counts as the failing step |
| L4 command-buffer creation not recorded | recorded in the ledger and thrown as `gpuCommandFailed` (the separate error case is removed) |
| L5 `--train` results show a stale monitor state | the recorder builds `gpu_faults` at write time from a provider |
| L6 chain survives a load | `BatchHashChain.reset()` whenever the trainer clock is set or the network reset |
| L7 gpuEvent reports by name, system folder only | filtered by this pid in the report; both report folders |
| L8 ps parsing on spaces | the command line is matched as a whole; redaction per word |
| L9 games abandoned several times per reset | once per reset (faults within 30 s of the last abandon are the same reset) |
| L10 fault exit hides a failed autosave | the exit warns when the last autosave failed |
| L11 hash backlog unbounded | at most 8 batches wait; `submit` blocks the trainer queue beyond that |
| L12 test dumps into the real folder | the default dump folder is a scratch folder under XCTest |
