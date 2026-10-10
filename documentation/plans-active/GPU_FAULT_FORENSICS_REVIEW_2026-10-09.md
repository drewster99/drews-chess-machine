# Independent review: GPU fault forensics plan and GPU call audit (2026-10-09)

I checked the plan and audit against committed `HEAD` (`b43d5cf8`). Its code is identical to the audit's baseline `7292c0eb`. The incident numbers were checked against the session logs, read-only.

**The working tree changed while I was reviewing.** Phase 2 is already being implemented, although the plan says "not started":
- new, untracked: `Network/GPUSubmission.swift` and `Network/GPUFaultLedger.swift`
- modified: `ChessNetwork.swift` and `ChessTrainer.swift`

## Verified correct

- **Inventory counts all match the code.** 19 `graph.run`, 1 `executable.run`, 4 `encode`, 4 `makeCommandBuffer`, 42 `MPSNDArray` inits, 43 `MPSGraphTensorData` inits, 14 `writeBytes`, 9 `readBytes`.
- **I found no GPU submission the audit missed** within its stated scope.
- **Error checks are as described:**
  - The baseline check reads only `== .error`, one call late (`ChessNetwork.swift:1662`).
  - The training step checks only its first buffer, `== .error` (`ChessTrainer.swift:7179-7186`).
  - The weight load checks the first and last buffers (`ChessNetwork.swift:1990-1995`, `2006-2010`).
- **Other code claims hold:**
  - The arena catches write no session-log line (`SessionController+Arena.swift:224-228, 257-261, 382-386`).
  - The CLI failure paths exit 33 (`CorpusReplayRunner.swift:1083-1087`, `TrainVsUciRunner.swift:145-149, 167-171`).
  - The velocity-load comment is at `ChessTrainer.swift:5893-5899`.
  - The promotion rewind order matches `TrainerResumeState.swift:414-424`.
  - Save verification compares two forwards built from the same export (`CheckpointManager.swift:2119-2184`).
- **Incident gradient ratios match the logs:** 37,469,639×, 32,736,274×, 10,343,145× and 9.38×.

## Critical

**C1. A late (layer 2) fault can't void a promotion or a save that has already happened.** The monitor lags by up to about 12 s. At 21:37, SPRT decided 0.6 s after the hang.
- **Promotion goes through first.** "Void the arena if a fault time falls between its start and end" (plan A5) is checked only after `shouldPromote` (`SessionController+Arena.swift:430-500`) has already run. By then the champion has been replaced, the trainer rewound, the `-promote` save written and `LastSessionPointer` moved.
- **Saves inside the lag window carry post-fault weights.**
  - In the CLI, any checkpoint written in that window holds post-fault weights.
  - "Recover by exact resume from the last checkpoint" would pick that bad checkpoint.
  - The rolling `--out-model` is replaced in place, so the pre-fault copy is gone.
- **Fix:** add a fault barrier before every promotion, save and pointer update. Either force a synchronous monitor poll (about 2 s), or wait until the monitor has read past the decision time. Record `fault_checked_through` in each save, and define recovery as "the last save checked through with no fault after it."

**C2. The new `.gpuFault` suspension would re-enable periodic autosave of suspect weights.**
- Today a GPU throw in the GUI becomes `.divergence`, which skips periodic autosave (`TrainingSuspension.swift:49-55`, `SessionController+Training.swift:1330-1348`).
- The plan models `.gpuFault` on the health stop instead ("self-play and autosave continue").
- That would save post-fault trainer weights and move `LastSessionPointer` to them, so "Resume Training from Autosave" would resume poisoned weights.
- **Fix:** `.gpuFault` must skip autosave, as `.divergence` does.

## High

**H1. A failed promotion copy leaves the trainer half-rewound and training continues.**
- The catch at `SessionController+Arena.swift:558-560` only calls `recordError`.
- Once `syncMastersFromWorking`, `loadVelocitySnapshot` and `restoreDropoutState` throw (they will under the new submission helper), the result can be: rewound weights, stale fp32 masters and velocity, and a champion that may already hold the candidate's weights. Training then resumes on that state.
- The A5 table has no row for this sequence. It must suspend training.

**H2. The dropout-RNG advance at `ChessTrainer.swift:7549` is training state, not a probe.**
- A5 puts it under "probes: result discarded, training continues", and the code comment says it is "logged and swallowed".
- If it fails, the next step reuses this step's dropout mask, and the RNG exactness recorded in lineage becomes false.
- It belongs in the training-step row, so a failure there aborts.

**H3. The crash dump can block the halt, and can dump the wrong batch.**
- **GPU reads right after a fault.** `weights-after`, velocity, masters and the Philox state all need GPU reads (`ChessNetwork.swift:1889`; `ChessTrainer.swift:5838, 5940, 6274`). Right after a hang, the GPU can hang again, and `waitUntilCompleted` has no timeout. C4's "a dump never blocks the halt" isn't achievable as written.
  - **Fix:** write the CPU-side parts first and flush them, give the GPU reads a time budget, and mark them unverified if they fail.
- **Wrong batch on a late trigger.** For a layer 2 trigger, the staging batch has already been overwritten by later steps (`replayBatch*`, `ChessTrainer.swift:4886-4897`). The dump must say which step its batch belongs to; only the identities of earlier steps survive.

**H4. The batch identity chain proves less than B3 claims, outside corpus replay.**
- The identity is slot indices plus write position. That proves "same samples" only when the buffer's contents are a deterministic function of the feed, which is true for corpus replay only.
- **GUI:** a resume refills the buffer from new games (`NOT EXACT: buffer` by default) while the sampler stream may continue. Chains can match while the contents differ.
- **Train-vs-UCI:** games come from external engines, so contents aren't reproducible either.
- **GUI promotion:** the rewind resets the trainer clock (`TrainerResumeState.swift:421`), so step numbers repeat within one log. Chain state and `[BATCH-HASH]` lines must reset or be marked after a rewind.
- **Fix:** state the guarantee per path, and label the chain as covering indices only.

**H5. Layer 1 rests on unverified behavior, and A7 has no fault-injection test.**
- The audit (§3) says outright that it needs a fault test in a dedicated session with no training running.
- A7 only waits for "the first real fault" to show up live.
- **Fix:** add a controlled fault in a standalone probe with no training running (for example, a watchdog trip or an out-of-bounds page fault). Verify, for a fault in a middle buffer: whether the completion handler reports it, what the last buffer's status shows, and the exact system-log text.

**H6. Layer 3's protection is overstated.**
- **A4's "3 of 3 hangs showed 2.35× … 32,736,274× spikes, all clipped" is wrong:**
  - The 19:36:49 hang showed no spike (the plan's own lines 62-67 say so).
  - 2.35 is a pre-clip norm, not a ratio; the ratio was 9.38× (`dcm_log_20261008-094622.txt:13073`).
  - The largest ratio was 37,469,639× (same file, line 12668), not 32,736,274×.
  - B-siluall's 21:37 step was a NaN halt, not a clip (`dcm_log_20261009-163612.txt:1760`).
- **Clipping only happens in `mode=clip`.** All three runs had `mode=clip applied=true`. CLAUDE.md says the default mode only logs, so a default run has no layer 3 beyond the hard maximum.
- **`applied=true` doesn't prove the GPU applied the clip.** It is a CPU-side inference (`clipped(preClipNorm:)`, `RelativeGradientCap.swift:253`) from the same readback that may be faulted.
- **The 1,000× near-miss threshold has gaps.**
  - It misses a real hang-corrupted step: R-replay's 22:59 step was only 9.38×.
  - It is undefined when there is no reference median (mode off, or warm-up; `RelativeGradientCap.swift:238-240`).

## Medium

- **M1. `[GPU-SLOW]` measures the wrong span.**
  - First buffer start to last buffer end includes CPU encode gaps: MPS commits the first buffer during encode, and step encode is 1–18 s (`[ENCODE-COST]` p99 8 s).
  - It will fire on ordinary steps. The watchdog works per buffer, so log each buffer's own duration as well as the span.
- **M2. The `OSLogStore` monitor's cost and filter are unspecified.**
  - The 1.8–2.0 s query time came from a small standalone probe. The GUI process emits about 13 `os_log` lines/s from the heartbeat (`SessionController+Heartbeat.swift:5-8`) and runs for days, so queries will likely be slower there.
  - The machine was already at 25 GB of swap. Measure in a long-lived GUI process.
  - Define the filter as an allowlist built from the three incidents' captured message text, and use those messages as test fixtures.
  - Record "layer 2 unavailable" in `results.json` and the `[RUN]` line, so "no faults" isn't misread.
- **M3. GUI `--train` isn't covered.** R-fixedlr was a GUI `--train` run, and CLAUDE.md requires every `--train` termination to go through `AutoTrainTermination`. A5 only says "GUI suspends".
- **M4. It adds a second stop channel.** Exit 36 and `.gpuFault` sit beside the health stop (exit 35, `.healthAlarm`), with separate `results.json` fields and banners. Reuse the existing termination plumbing (one source of truth). Also confirm the exit-33 path can write `results.json` at all, since C4 assumes it can add `crash_dump`.
- **M5. The content hash's inputs and timing need defining.**
  - B2 says the hash runs from `boardsCopy` (`ChessTrainer.swift:4952`), but moves and outcomes exist only in staging. That staging is overwritten by the next `sample()`, and the draw penalty rewrites outcomes in place (`5028-5033`).
  - Copy moves and outcomes too, and say whether outcomes are hashed before or after the draw penalty.
  - "The hash is done by the time the step line is written" is an assumption; the step line must wait for it or mark it pending.
- **M6. The replay buffer doesn't store corpus shard and game per slot.** It keeps only a packed `workerGameId` and the ply (`ReplayBuffer.swift:137, 770`). C3's "corpus shard + game + ply" needs either new per-slot storage, which affects memory and `replay_buffer.bin` compatibility, or a map from feed position to corpus game.
- **M7. The owner decision "every GPU-touching call checks its failure" can't be met for the 117 calls with no failure signal:**
  - `MPSNDArray` init is imported as non-nil.
  - `readBytes` / `writeBytes` return nothing.
  - Compile has no error path.

  The plan silently drops these calls; state the limit and get the owner's sign-off.
- **M8. A5 needs a row for every call site that newly throws.** Examples: trainer init and reset (`6180`), the dropout seed write at run start (`6253`), BN warmup (`2230`), and health reads (`6075`, `6120`). The health reads currently "never stop training", which contradicts A5's "any fault while the trainer is active aborts".
- **M9. Abandoning self-play games on a layer 1 fault is unnecessary.** With the new helper, the throw happens before `consume` (`ChessNetwork.swift:1490-1516`), so no garbage logits are sampled. The existing skip-tick behavior is correct; only a layer 2 fault justifies abandoning games.
- **M10. The dump's weights file collides with the "every model file carries test-set results" rule.** `SafetensorsModelIO.encode` requires `testSetResults:`. Specify that `weights-after` is not a DCM model file, or get the owner's OK for an explicit "not evaluated" status.
- **M11. Validation has no performance acceptance criteria.** Add before/after steps per second and plies per hour on an idle GPU, with bounds, for the new helper (self-play, training step, `errorOptions`, the completion-handler wait).

## Low

- **L1. Wrong cross-references in the plan.**
  - Line 92's "(A5)" should be A6.
  - In C3, "memory and thermal state (A5)" should be A6.
  - In C3, "GPU errors seen (A4, A6)" should be A2/A3.
  - In C3, `system-log.txt` "(A6)" should be A3.
- **L2. The plan text has drifted from the in-progress code.**
  - Names differ: the plan says `GPUWork` / `GPUWorkError`; the code has `GPUSubmission` and throws `ChessNetworkError.gpuCommandFailed`, even for stages that used to throw `ChessTrainerError.gpuCommandFailed`.
  - The plan should say the completion handler can fire after `waitUntilCompleted` returns, so it must be awaited. The in-progress code already does this, with a semaphore and a 10 s grace period.
  - Test that the handler fires for every encode variant; otherwise every submission stalls 10 s and then fails.
- **L3. Wait for the baseline instead of assuming it finished.** Metal doesn't guarantee "the step's buffers follow the baseline's, so it has finished" for MPS heap resources. Also, the check runs after the step's update, so it can only trigger the abort, not prevent the update.
- **L4. "164 call sites" overstates the count.** It includes 14 commit/wait/status reads and 4 non-failing `MPSCommandBuffer` inits. Audit #7's "fingerprint (`ChessTrainer.swift:4979`)" is actually the general baseline call.
- **L5. Crash-dump folders need housekeeping.**
  - Two dumps with the same name collide; add a `-2` suffix as the session logs do.
  - `.tmp` staging in `CrashDumps/` is never swept.
  - Near-miss dumps have no rate limit or disk cap.
- **L6. Secrets in the process list.** "Other DrewsChessMachine processes running" must use the lineage record's argv redaction.
- **L7. Keep macOS's own report.** Copy `gpuEvent-*.ips` reports for our PID into the dump; the incident's report was gone afterwards.
- **L8. Flush the session logger** before writing `log-tail.txt`.
- **L9. Log-parser compatibility.** Step lines are parsed by the dashboards, `replay.py` and `--replay-health-log`; confirm they accept the new `batch=` / `chain=` fields.
- **L10. Possible existing input-buffer hazard (unverified).**
  - The baseline is committed without a wait, and it reads the cached input array for its batch size.
  - Any other job on the trainer network with the same batch size, run before the baseline completes, overwrites that input from the CPU (`ChessNetwork.swift:1534-1563, 1670-1674`).
  - One candidate: the legal-mass probe's fallback at `ChessTrainer.swift:5430`, if its count equals the batch size.

## Error paths the plan doesn't cover

Nothing is written when the process dies outright:
- an MPSGraph framework assertion (like the earlier `mps.placeholder` crash)
- a `preconditionFailure` in compile or feed binding
- jetsam (memory-pressure kill)

## Missing tests

- a forced-fault experiment (H5)
- the arena-void and late-detection race, including promotion (C1)
- `.gpuFault` gating autosave (C2)
- promotion-copy failure (H1)
- a failure in the dropout advance (H2)
- a hanging GPU read during the dump (H3)
- chain behavior across a promotion rewind and on non-corpus paths (H4)
- content hash after the draw penalty (M5)
- exit code 36 and the `crash_dump` field in `results.json`
- performance bounds (M11)

## Summary

The audit's inventory is accurate and complete for `HEAD`. Its line numbers and probe-backed findings hold up, and the incident numbers check out against the logs, except A4's summary of them, which has the wrong figures. The plan's weak points are when it acts and how much it promises. Detection that lags by up to 12 s can't stop a promotion or a save that happens within that window; the 21:37 incident is exactly that case. The proposed `.gpuFault` suspension would turn periodic autosave back on, which today's divergence handling deliberately blocks. Promotion-copy and dropout-advance failures go to the wrong outcome. The crash dump depends on GPU reads that can hang. The batch identity chain is only proof of matching batches for corpus replay. The layer 3 numeric guard is described as stronger than it is. Fix C1–C2 and H1–H6 before implementing phase 3. Phase 2, the checked-submission helper, is sound but needs the forced-fault experiment and performance bounds. It is already being implemented, under different names than the plan uses, while the plan still says "not started".

