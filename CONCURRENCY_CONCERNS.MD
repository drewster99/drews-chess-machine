# Concurrency Concerns Evaluation

This document summarizes the prioritized concurrency issues identified during the evaluation of `drews-chess-machine`, along with reviewed areas that did not produce findings.

## Prioritized Issues

### 1. High — Unsynchronized cross-thread access to `ChessTrainer` mutable configuration and diagnostics

#### Files / lines

- `DrewsChessMachine/DrewsChessMachine/Training/ChessTrainer.swift:983-1039` — mutable training parameters (`learningRate`, `sqrtBatchScalingForLR`, `lrWarmupSteps`, `entropyRegularizationCoeff`, `weightDecayC`, `gradClipMaxNorm`, `policyLossWeight`, `valueLossWeight`)
- `DrewsChessMachine/DrewsChessMachine/Training/ChessTrainer.swift:1070` — mutable `drawPenalty`
- `DrewsChessMachine/DrewsChessMachine/Training/ChessTrainer.swift:1252-1264` — mutable `batchStatsInterval`, `lastBatchStatsUniquePct`, `lastBatchStatsSummary`
- `DrewsChessMachine/DrewsChessMachine/Training/ChessTrainer.swift:2457-2474` and `3289-3312` — background/execution-queue reads of those mutable parameters
- `DrewsChessMachine/DrewsChessMachine/Training/ChessTrainer.swift:2618-2653` — execution-queue reads/writes of `batchStatsInterval`, `lastBatchStatsUniquePct`, `lastBatchStatsSummary`
- `DrewsChessMachine/DrewsChessMachine/App/UpperContentView/UpperContentView.swift:9688-9890` — SwiftUI/MainActor writes to trainer parameters while training can be active
- `DrewsChessMachine/DrewsChessMachine/App/UpperContentView/UpperContentView.swift:7236-7355` — reads `trainer.lastBatchStatsUniquePct` / `trainer.lastBatchStatsSummary` from the UI/training orchestration side while the trainer queue may update them

#### Description

`ChessTrainer` is marked `@unchecked Sendable` and serializes graph/training work on `executionQueue`, but several ordinary `var` properties are also read/written directly from SwiftUI/MainActor and other async contexts. Some fields are read inside `executionQueue` (`buildFeeds`, replay sampling/stats), while UI controls write them directly.

#### Why it is a problem

Plain Swift stored properties are not atomic and are not protected by an actor, lock, or the trainer queue. Concurrent reads/writes across the main actor and `executionQueue` are real data races. Even for scalar `Float`/`Bool`/`Int` values, Swift does not guarantee race-free behavior; for optional struct diagnostics like `lastBatchStatsSummary`, torn or stale observations are possible. The comment at `ChessTrainer.swift:1260-1263` explicitly acknowledges unsynchronized access for `lastBatchStatsSummary`, but it is still a concurrency defect.

---

### 2. Medium — `SessionLogger` uses shared `DateFormatter` instances from arbitrary threads

#### Files / lines

- `DrewsChessMachine/DrewsChessMachine/Logging/SessionLogger.swift:33-48` — static `DateFormatter` instances
- `DrewsChessMachine/DrewsChessMachine/Logging/SessionLogger.swift:104-107` — `lineTimestampFormatter.string(from:)` is called before dispatching onto the serial logger queue

#### Description

`SessionLogger.log(_:)` is documented as safe from any thread/actor, but it formats timestamps synchronously on the caller's thread using a shared static `DateFormatter`.

#### Why it is a problem

`DateFormatter` is not reliably thread-safe for concurrent use. Since many background tasks call `SessionLogger.shared.log(...)`, multiple threads can enter `lineTimestampFormatter.string(from:)` concurrently, causing undefined behavior, corrupted output, or rare crashes. File writes themselves are serialized correctly; the timestamp formatting is not.

---

### 3. Medium — Log-analysis cancellation has a race that can leave the subprocess running and can publish stale results after cancellation

#### Files / lines

- `DrewsChessMachine/DrewsChessMachine/Views/LogAnalysis/LogAnalysisViewModel.swift:54-59` — starts detached analysis task
- `DrewsChessMachine/DrewsChessMachine/Views/LogAnalysis/LogAnalysisViewModel.swift:65-69` — `cancel()` cancels task and terminates only the currently stored `activeProcess`
- `DrewsChessMachine/DrewsChessMachine/Views/LogAnalysis/LogAnalysisViewModel.swift:151-169` — background queue publishes/clears `activeProcess` asynchronously on main
- `DrewsChessMachine/DrewsChessMachine/Views/LogAnalysis/LogAnalysisViewModel.swift:193-203` — result/error is applied without checking cancellation

#### Description

The subprocess is launched on a GCD queue, then `activeProcess` is assigned via `DispatchQueue.main.async`. If `cancel()` runs after `claudeProc.run()` but before that async assignment executes, `activeProcess` is still nil and the running `claude` process is not terminated. Separately, cancelling `analysisTask` does not cancel the blocking `waitUntilExit()` work; when it eventually completes, `runAnalysis` still sets `isAnalyzing`, `claudeResponse`, or `errorMessage` without checking `Task.isCancelled`.

#### Why it is a problem

Closing/cancelling the analysis window can leave an external process running unnecessarily, and stale output can be published after the user cancelled. This is a task-cancellation correctness issue, not just style.

---

### 4. Low / Medium — Queued `ChessNetwork` and `ChessTrainer` work ignores Swift task cancellation once enqueued

#### Files / lines

- `DrewsChessMachine/DrewsChessMachine/Network/ChessNetwork.swift:1002-1011` — `enqueue` always runs queued work and resumes the continuation even if the awaiting task was cancelled
- `DrewsChessMachine/DrewsChessMachine/Training/ChessTrainer.swift:3208-3217` — same pattern for trainer work
- Representative callers: `ChessNetwork.evaluate` at `518-524` and `608-615`; `ChessTrainer.trainStep` at `2398-2403` and replay training path `2585-2692`

#### Description

Both queue wrappers bridge async work onto serial `DispatchQueue`s with `withCheckedThrowingContinuation`, but they do not check cancellation before executing queued work, and they do not use `withTaskCancellationHandler` to short-circuit queued items that have not started.

#### Why it is a problem

If the parent task is cancelled while work is waiting behind other graph/training operations, the cancelled operation still runs to completion and may perform expensive GPU work or mutate trainer/network state before the caller observes cancellation. This does not appear to corrupt queue serialization, but it can cause stop/abort latency and unexpected work after cancellation. Severity is lower because MPSGraph work may not be practically cancellable once running, but queued-not-yet-started work could still be skipped.

## Reviewed Areas With No Issues Found

- `BatchedMoveEvaluationSource` is actor-isolated and has explicit cancellation handling for parked continuations; no continuation leak or double-resume issue found.
- `BatchedSelfPlayDriver` and `TournamentDriver` use task groups/unstructured slot tasks with explicit drain/retirement handling; no shared mutable parent-state races found.
- `ChessNetwork` serializes graph access and scratch-buffer reuse through its private `executionQueue`; no reentrant scratch-buffer races found in current public entry points.
