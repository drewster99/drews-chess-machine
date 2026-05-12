import SwiftUI

/// `SessionController`'s arena machinery — split out of `SessionController.swift`
/// to keep that file navigable. Holds the parallel-mode tournament
/// (`runArenaParallel`, ~690 lines), its result-logging + teardown helpers, and
/// the "Recover Arena History from logs" scanner. All state is `var` (internal)
/// on `SessionController`, so this extension has full access.
extension SessionController {

    // MARK: - Arena-history recovery (Stage 4r)

    func runArenaHistoryRecovery() {
        guard !arenaRecoveryInProgress else { return }
        arenaRecoveryInProgress = true
        SessionLogger.shared.log("[BUTTON] Recover Arena History from logs (start)")

        Task.detached(priority: .userInitiated) { [tournamentHistory] in
            let logsDir: URL
            do {
                logsDir = try ArenaLogRecovery.defaultLogsDirectory()
            } catch {
                await MainActor.run {
                    SessionLogger.shared.log(
                        "[RECOVER] failed: cannot resolve logs directory: \(error.localizedDescription)"
                    )
                    self.arenaRecoveryInProgress = false
                }
                return
            }
            let recovered = ArenaLogRecovery.scan(logsDirectory: logsDir)

            // Merge in-place on a copy; only adopt the new array
            // if at least one record actually changed.
            var updated = tournamentHistory
            var changedCount = 0
            for (i, record) in tournamentHistory.enumerated() {
                guard let hit = recovered[record.finishedAtStep] else { continue }
                guard hit.candidateWins == record.candidateWins,
                      hit.draws == record.draws,
                      hit.championWins == record.championWins else {
                    continue
                }
                let newFinishedAt = record.finishedAt ?? hit.finishedAt
                let newCandidateID = record.candidateID
                    ?? hit.candidateID.map { ModelID(value: $0) }
                let newChampionID = record.championID
                    ?? hit.championID.map { ModelID(value: $0) }
                if newFinishedAt != record.finishedAt
                    || newCandidateID != record.candidateID
                    || newChampionID != record.championID {
                    updated[i] = TournamentRecord(
                        finishedAtStep: record.finishedAtStep,
                        finishedAt: newFinishedAt,
                        candidateID: newCandidateID,
                        championID: newChampionID,
                        gamesPlayed: record.gamesPlayed,
                        candidateWins: record.candidateWins,
                        championWins: record.championWins,
                        draws: record.draws,
                        score: record.score,
                        promoted: record.promoted,
                        promotionKind: record.promotionKind,
                        promotedID: record.promotedID,
                        durationSec: record.durationSec,
                        candidateWinsAsWhite: record.candidateWinsAsWhite,
                        candidateWinsAsBlack: record.candidateWinsAsBlack,
                        candidateLossesAsWhite: record.candidateLossesAsWhite,
                        candidateLossesAsBlack: record.candidateLossesAsBlack,
                        candidateDrawsAsWhite: record.candidateDrawsAsWhite,
                        candidateDrawsAsBlack: record.candidateDrawsAsBlack
                    )
                    changedCount += 1
                }
            }

            await MainActor.run {
                if changedCount > 0 {
                    self.tournamentHistory = updated
                }
                SessionLogger.shared.log(
                    "[RECOVER] Arena history scan: \(recovered.count) kv lines mapped, \(changedCount) records updated (of \(tournamentHistory.count) total). Save the session to persist."
                )
                self.arenaRecoveryInProgress = false
            }
        }
    }

    // MARK: - Arena tournament (Stage 4n)

    /// Max barrier-wait for the arena's batched move evaluator (ms). Arena
    /// games run a small concurrency so the batch rarely fills; the wait caps
    /// per-ply latency.
    nonisolated static let arenaBatchWaitMs: Double = 100.0

    /// Whether to write a `-promote.dcmsession` autosave after each arena
    /// promotion (re-using the weight snapshots taken under the arena's
    /// pauses). On by default.
    nonisolated static let autosaveSessionsOnPromote: Bool = true


    /// Run one arena tournament in parallel mode — 200 games between
    /// the candidate (synced from trainer at start) and the arena
    /// champion (synced from the real champion at start), while
    /// self-play and training continue running in the background.
    /// Promotes the candidate into the real champion iff the score
    /// meets the 0.55 threshold.
    ///
    /// Synchronization: this is called from the arena coordinator
    /// task, which is a peer to the self-play and training workers.
    /// Training and self-play are briefly paused at arena start so
    /// the method can take trainer and champion snapshots into
    /// dedicated arena-only networks; after that the arena runs
    /// exclusively on those two snapshots and doesn't touch the
    /// "live" trainer or champion again until promotion (which
    /// briefly re-pauses self-play to write into the champion).
    /// Candidate test probes skip while `arenaFlag.isActive` so
    /// they don't race with arena reads of the candidate inference
    /// network.
    @MainActor
    func runArenaParallel(
        trainer: ChessTrainer,
        champion: ChessMPSNetwork,
        candidateInference: ChessMPSNetwork,
        arenaChampion: ChessMPSNetwork,
        tBox: TournamentLiveBox,
        selfPlayGate: WorkerPauseGate,
        trainingGate: WorkerPauseGate,
        arenaFlag: ArenaActiveFlag,
        overrideBox: ArenaOverrideBox
    ) async {
        // Clear any stale decision from a previous tournament so this
        // run starts with a clean override slate. Normal completion
        // `consume()`s the box at the end, but early-return paths
        // (cancellation, sync errors) don't — clearing here keeps
        // all exit paths honest.
        _ = overrideBox.consume()
        let (steps, totalGames, startTime, arenaStartTrainingSnapshot) = await beginArenaRun(
            trainer: trainer, champion: champion, tBox: tBox, arenaFlag: arenaFlag
        )

        // --- Trainer → candidate inference snapshot ---
        //
        // Pause the training worker briefly (a few ms, at most one
        // SGD step) so we can export trainer weights without racing
        // against a concurrent `trainer.trainStep`. Release training
        // as soon as the snapshot lands — the rest of the arena
        // runs on `candidateInference`, not on `trainer`, so training
        // can continue through the 200 games.
        await trainingGate.pauseAndWait()
        if Task.isCancelled {
            trainingGate.resume()
            cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
            return
        }
        // Capture trainer weights here and hold them through the
        // rest of the arena. At arena end we use them to autosave
        // the session without needing another training pause
        // (and therefore without any gate interaction from the
        // autosave task, which is critical to avoid deadlocking
        // a save that runs past a session cancel — unstructured
        // save tasks don't inherit realTrainingTask cancellation).
        //
        // We also snapshot the optimizer velocity. If the candidate
        // wins the arena, we'll restore THIS snapshot when we copy
        // the candidate weights back into the trainer — the velocity
        // that built the validated candidate is the right velocity
        // for the candidate's weight surface. (Earlier behavior
        // zeroed velocity on promotion, throwing away accumulated
        // gradient signal.)
        var trainerSnapshotWeights: [[Float]] = []
        var trainerSnapshotVelocity: [[Float]] = []
        do {
            let snapshot: ([[Float]], [[Float]]) = try await Task.detached(priority: .userInitiated) {
                let weights = try await trainer.network.exportWeights()
                let velocity = try await trainer.exportVelocitySnapshot()
                try await candidateInference.loadWeights(weights)
                return (weights, velocity)
            }.value
            trainerSnapshotWeights = snapshot.0
            trainerSnapshotVelocity = snapshot.1
        } catch {
            trainingBox?.recordError("Arena candidate sync failed: \(error.localizedDescription)")
            trainingGate.resume()
            cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
            return
        }
        trainingGate.resume()

        // Arena candidate inherits the trainer's current generation
        // ID. If it gets promoted, the promoted champion should keep
        // that exact identifier and the live trainer will roll forward
        // to the next generation after being rewound to the promoted
        // weights.
        candidateInference.identifier = trainer.identifier

        // --- Champion → arena champion snapshot ---
        //
        // Same pattern for self-play: brief pause, copy weights from
        // the real champion into the arena-only champion network,
        // release. Arena games from here on only read
        // `arenaChampion`, leaving the real champion free for
        // continuous self-play through the tournament.
        await selfPlayGate.pauseAndWait()
        if Task.isCancelled {
            selfPlayGate.resume()
            cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
            return
        }
        do {
            try await Task.detached(priority: .userInitiated) {
                let weights = try await champion.exportWeights()
                try await arenaChampion.loadWeights(weights)
            }.value
        } catch {
            trainingBox?.recordError("Arena champion sync failed: \(error.localizedDescription)")
            selfPlayGate.resume()
            cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
            return
        }
        selfPlayGate.resume()

        // arenaChampion is a pure copy of champion's current weights,
        // so it inherits the champion's ID verbatim — no mint.
        arenaChampion.identifier = champion.identifier

        // --- Tournament games ---
        //
        // Cancellation: the detached tournament task is wrapped in
        // `withTaskCancellationHandler` so clicking Stop flips a
        // `CancelBox` that `TournamentDriver.run` checks between
        // games. The worst-case delay from Stop to actually breaking
        // out is one in-flight arena game (~400 ms).
        let cancelBox = CancelBox()
        let arenaDiversity = GameDiversityTracker(windowSize: totalGames)
        // Snapshot the current arena schedule once at tournament start
        // so every game in this arena uses the same tau settings, even
        // if the user edits the fields mid-tournament.
        let arenaScheduleSnapshot = samplingScheduleBox?.arena ?? buildArenaSchedule()

        // --- Concurrent arena: build per-network batchers ---
        //
        // K parallel arena games share two `BatchedMoveEvaluationSource`
        // batchers, one per network. Each game alternates candidate ↔
        // champion moves, so at any instant ~K/2 are pending on each
        // side; a strict count-only barrier (the self-play model) would
        // never fire either batcher in steady state. The coalescing-
        // window timer (`maxBatchWaitMs`) makes the barrier fire on
        // whichever happens first — count met OR window expired —
        // which captures the early-game synchronized phase as a full
        // K-batch and the desynchronized steady state as ~K/2 partial
        // batches.
        //
        // Live count tracking: `expectedSlotCount` starts at `liveK`
        // (clamped to game count) and stays there while replacements
        // keep the pool full. The `onSlotRetired` callback below
        // fires only when a slot leaves the pool (no replacement
        // spawned) — i.e. exactly when each of the final K games
        // finishes — so each batcher's `expectedSlotCount` decrements
        // 1:1 with live game count during the tail. Decrementing on
        // every completion (the prior approach) drove the count to
        // 0 partway through the tournament and forced the rest of
        // the run through drain mode = single-position batches.
        let liveK = max(1, min(TrainingParameters.shared.arenaConcurrency, totalGames))
        // Joint GPU-busy-wall accumulator shared across both batchers.
        // Each fire reports start/end around its `network.evaluate`
        // await; the timer accumulates wall time during which AT
        // LEAST ONE side was on the GPU. Read once at the end of
        // arena to emit a single `[ARENA] timing joint` line that's
        // directly comparable to arena wall.
        let arenaGpuTimer = ArenaGpuTimer()
        let candidateBatcher = BatchedMoveEvaluationSource(
            network: candidateInference,
            maxBatchWaitMs: Self.arenaBatchWaitMs,
            name: "candidate",
            logBatchTimings: true,
            gpuTimer: arenaGpuTimer
        )
        let championBatcher = BatchedMoveEvaluationSource(
            network: arenaChampion,
            maxBatchWaitMs: Self.arenaBatchWaitMs,
            name: "champion",
            logBatchTimings: true,
            gpuTimer: arenaGpuTimer
        )
        await candidateBatcher.setExpectedSlotCount(liveK)
        await championBatcher.setExpectedSlotCount(liveK)

        // Update the live progress box with the chosen concurrency so
        // the arena's busy-label suffix can render "(×K concurrent)"
        // for the duration of this run.
        tBox.update(TournamentProgress(
            currentGame: 0,
            totalGames: totalGames,
            candidateWins: 0,
            championWins: 0,
            draws: 0,
            startTime: startTime,
            concurrency: liveK
        ))
        tournamentProgress = tBox.snapshot()

        // Records collector. The driver fires `onGameRecorded` from
        // its parent harvest loop — already serial in that task — so
        // this Box's append happens on a single task and needs no
        // synchronization. We lift it to a `final class` only because
        // the closure must be `@Sendable`; the lock is a defensive
        // belt for that contract, not for actual contention.
        let recordsBox = TournamentRecordsBox()
        let stats: TournamentStats
        do {
            stats = try await withTaskCancellationHandler {
                try await Task.detached(priority: .userInitiated) {
                    [candidateBatcher, championBatcher, tBox, cancelBox, overrideBox, arenaDiversity, arenaScheduleSnapshot, liveK, recordsBox] in
                    let driver = TournamentDriver()
                    return try await driver.run(
                        playerA: {
                            MPSChessPlayer(
                                name: "Candidate",
                                source: candidateBatcher,
                                schedule: arenaScheduleSnapshot
                            )
                        },
                        playerB: {
                            MPSChessPlayer(
                                name: "Champion",
                                source: championBatcher,
                                schedule: arenaScheduleSnapshot
                            )
                        },
                        games: totalGames,
                        concurrency: liveK,
                        diversityTracker: arenaDiversity,
                        // The driver checks this between games. Either a
                        // task-cancel (session Stop) or a user Abort /
                        // Promote click breaks the game loop early; the
                        // caller below disambiguates the two via the
                        // override box's `consume()`.
                        isCancelled: { cancelBox.isCancelled || overrideBox.isActive },
                        onGameCompleted: { gameIndex, aWins, bWins, draws in
                            tBox.update(TournamentProgress(
                                currentGame: gameIndex,
                                totalGames: totalGames,
                                candidateWins: aWins,
                                championWins: bWins,
                                draws: draws,
                                startTime: startTime,
                                concurrency: liveK
                            ))
                        },
                        onSlotRetired: {
                            // Fires only when a slot leaves the
                            // live pool (no replacement spawned).
                            // Decrementing on every completion would
                            // peg the count at 0 mid-tournament and
                            // collapse all remaining batches to size 1.
                            await candidateBatcher.decrementExpectedSlotCount()
                            await championBatcher.decrementExpectedSlotCount()
                        },
                        onGameRecorded: { record in
                            recordsBox.append(record)
                        }
                    )
                }.value
            } onCancel: {
                cancelBox.cancel()
            }
        } catch {
            // Error path. Drain both batchers so any parked
            // continuations resume, then emit whatever telemetry we
            // can over the partial records captured before the throw
            // (so the user can still see how much concurrency we got
            // up to the failure and confirm whether the captured
            // games were internally consistent before bailing).
            // After that, surface the error and tear down arena
            // state.
            await candidateBatcher.setExpectedSlotCount(0)
            await championBatcher.setExpectedSlotCount(0)
            await ArenaTelemetryFormatter.emitPostRunTelemetry(
                candidateBatcher: candidateBatcher,
                championBatcher: championBatcher,
                gpuTimer: arenaGpuTimer,
                recordsBox: recordsBox,
                wasCancelled: cancelBox.isCancelled || overrideBox.isActive,
                context: "after error",
                arenaStartTime: startTime,
                trainingBox: trainingBox
            )
            trainingBox?.recordError("Arena tournament failed: \(error.localizedDescription)")
            cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
            return
        }

        // Drain the batchers explicitly. Drain mode fires any
        // straggler pendings immediately so no continuation is left
        // parked, even on cancellation paths where the driver may
        // have stopped harvesting before all in-flight slots returned.
        await candidateBatcher.setExpectedSlotCount(0)
        await championBatcher.setExpectedSlotCount(0)

        await ArenaTelemetryFormatter.emitPostRunTelemetry(
            candidateBatcher: candidateBatcher,
            championBatcher: championBatcher,
            gpuTimer: arenaGpuTimer,
            recordsBox: recordsBox,
            wasCancelled: cancelBox.isCancelled || overrideBox.isActive,
            context: nil,
            arenaStartTime: startTime,
            trainingBox: trainingBox
        )

        // --- Score and promotion ---
        //
        // A user Abort ends the tournament with no promotion regardless
        // of score. Otherwise the usual score-threshold check decides:
        // a full tournament must have been played AND the candidate's
        // score must meet `arenaPromoteThreshold`. (There is no
        // force-promote override anymore — `Engine ▸ Promote Trainee
        // Now` covers "promote the current trainer right now" without
        // an arena.) The consume also clears the box for the next
        // tournament.
        let aborted = overrideBox.consume()
        let playedGames = stats.gamesPlayed
        let score: Double
        if playedGames > 0 {
            score = (Double(stats.playerAWins) + 0.5 * Double(stats.draws)) / Double(playedGames)
        } else {
            score = 0
        }
        var promoted = false
        var promotedID: ModelID?
        let shouldPromote = !aborted
            && playedGames >= totalGames
            && score >= TrainingParameters.shared.arenaPromoteThreshold
        // `promotionKind` is `.automatic` for any arena-driven
        // promotion (the only kind this path can produce); `.manual`
        // is reserved for `promoteTrainerNow()`. Only read if
        // `promoted` ends up true.
        let promotionKind: PromotionKind? = shouldPromote ? .automatic : nil
        // Holds the new champion weights if promotion succeeds,
        // so we can hand them to a detached autosave task at the
        // end without needing to re-read them from the live
        // network (which would race against self-play again).
        var promotedChampionWeights: [[Float]] = []
        if shouldPromote {
            // Pause both self-play and training, then copy the
            // promoted candidate into both the champion and the live
            // trainer. This keeps self-play and SGD aligned on the
            // exact promoted weights rather than letting training
            // continue from a later, unvalidated post-arena state.
            await selfPlayGate.pauseAndWait()
            await trainingGate.pauseAndWait()
            if !Task.isCancelled {
                do {
                    promotedChampionWeights = try await Task.detached(priority: .userInitiated) {
                        [candidateInference, champion, trainer, trainerSnapshotVelocity, steps] in
                        let weights = try await candidateInference.exportWeights()
                        try await champion.loadWeights(weights)
                        try await trainer.network.loadWeights(weights)
                        // The trainer's CURRENT velocity was built up
                        // against the post-arena weight surface (which
                        // we just discarded by overwriting with the
                        // candidate weights). Restore the velocity we
                        // snapshotted at arena-start instead — that
                        // velocity is the EMA of gradients that built
                        // the validated candidate, so it's the right
                        // accumulator for the candidate's weight
                        // surface. Both gates are paused at this point,
                        // so the trainer's velocity I/O is safe to
                        // drive directly on network.graph.
                        try await trainer.loadVelocitySnapshot(trainerSnapshotVelocity)
                        // CRITICAL: Rewind the trainer's completed step
                        // count to match the snapshotted weights. Without
                        // this, the trainer keeps its post-arena step
                        // count but uses arena-start weights, causing
                        // the LR warmup multiplier to jump ahead of
                        // the weights and drive the immature network
                        // into collapse with an oversized LR.
                        trainer.completedTrainSteps = steps
                        return weights
                    }.value
                    // Promoted: champion now holds the arena candidate's
                    // exact weights, so it inherits that snapshot ID,
                    // while the rewound live trainer rolls forward to
                    // the next mutable generation in the same lineage.
                    champion.identifier = candidateInference.identifier
                    trainer.identifier = ModelIDMinter.mintTrainerGeneration(
                        from: champion.identifier ?? candidateInference.identifier ?? ModelIDMinter.mint()
                    )
                    promoted = true
                    promotedID = candidateInference.identifier
                    if let arenaStartTrainingSnapshot {
                        trainingBox?.seed(arenaStartTrainingSnapshot.stats)
                        trainingStats = arenaStartTrainingSnapshot.stats
                    } else {
                        trainingBox?.setStepCount(steps)
                        if var stats = trainingStats {
                            stats.steps = steps
                            trainingStats = stats
                        }
                    }
                    trainingBox?.resetRollingWindows()
                    trainingAlarm?.resetStreaks()
                    trainingAlarm?.clear()
                } catch {
                    trainingBox?.recordError("Promotion copy failed: \(error.localizedDescription)")
                }
            }
            trainingGate.resume()
            selfPlayGate.resume()
        }

        // Append to history and clear arena state.
        let durationSec = Date().timeIntervalSince(startTime)
        let record = TournamentRecord(
            finishedAtStep: steps,
            finishedAt: Date(),
            candidateID: candidateInference.identifier,
            championID: arenaChampion.identifier,
            gamesPlayed: playedGames,
            candidateWins: stats.playerAWins,
            championWins: stats.playerBWins,
            draws: stats.draws,
            score: score,
            promoted: promoted,
            promotionKind: promoted ? promotionKind : nil,
            promotedID: promotedID,
            durationSec: durationSec,
            candidateWinsAsWhite: stats.playerAWinsAsWhite,
            candidateWinsAsBlack: stats.playerAWinsAsBlack,
            candidateLossesAsWhite: stats.playerALossesAsWhite,
            candidateLossesAsBlack: stats.playerALossesAsBlack,
            candidateDrawsAsWhite: stats.playerADrawsAsWhite,
            candidateDrawsAsBlack: stats.playerADrawsAsBlack
        )
        tournamentHistory.append(record)
        // Mirror into the chart-tile event stream. Compute the
        // elapsed-second start/end against the chart-coordinator's
        // anchor so the band lands on the same X axis as the
        // time-series charts. (See `recordArenaStarted` site for
        // why this MUST be the chart anchor and not
        // `parallelStats.sessionStart` — the chart anchor is
        // back-dated on resume so the band lines up with restored
        // chart data; the parallel-stats anchor is intentionally
        // fresh per segment for rate-display.)
        let endElapsed = max(0, Date().timeIntervalSince(chartCoordinator?.chartElapsedAnchor ?? Date()))
        // Prefer the live start mark captured at arena begin —
        // it avoids a ~5-second drift from backward-inferring
        // startElapsed out of (end - durationSec) after the
        // promotion work ran. Fall back to the durationSec math
        // only if the live mark is somehow nil.
        let startElapsed = chartCoordinator?.activeArenaStartElapsed
        ?? max(0, endElapsed - durationSec)
        // `recordArenaCompleted` appends the event AND clears
        // the live-band marker, so the chart drops back to just
        // the completed events on the next render.
        chartCoordinator?.recordArenaCompleted(ArenaChartEvent(
            id: (chartCoordinator?.arenaChartEvents.count ?? 0),
            startElapsedSec: startElapsed,
            endElapsedSec: endElapsed,
            score: score,
            promoted: promoted
        ))
        logArenaResult(
            record: record,
            index: tournamentHistory.count,
            trainer: trainer,
            candidate: candidateInference,
            championSide: arenaChampion,
            diversity: arenaDiversity.snapshot()
        )
        cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)

        // On promotion: reset game-play stats so the display
        // reflects only the new champion's self-play performance,
        // and emit a STATS log line so the post-promotion state
        // is visible in the session log (the fixed STATS ticker
        // may not fire for up to an hour at this point in the
        // schedule).
        if promoted {
            parallelWorkerStatsBox?.resetGameStats()
            let trainerIDStr = trainer.identifier?.description ?? "?"
            let championIDStr = champion.identifier?.description ?? "?"
            SessionLogger.shared.log(
                "[STATS] post-promote  steps=\(trainingStats?.steps ?? 0) champion=\(championIDStr) trainer=\(trainerIDStr)"
            )
        }

        // Post-promotion autosave. Fires a detached Task that
        // writes a full session snapshot using the weights we
        // already captured above under the arena-start training
        // pause and the promotion self-play pause. The detached
        // task touches no live networks and no pause gates, so
        // it is safe to run past a session cancel — unstructured
        // save tasks don't inherit `realTrainingTask`
        // cancellation, and any post-return gate interaction
        // here would potentially deadlock against workers that
        // have already exited their loops.
        //
        // Status row: we publish the "Saving… / Saved" progression
        // via the same `setCheckpointStatus` channel the manual
        // save uses, tagged "(post-promotion)" so users can tell
        // at a glance which autosave trigger produced the line.
        // `periodicSaveController` sees the success callback and
        // resets its 4-hour deadline, so a promotion that happens
        // shortly before a scheduled periodic save implicitly
        // defers the periodic one — which matches the spec: if a
        // post-promotion save already covered the window, the
        // next periodic tick runs a full 4 hours later from now.
        if promoted && Self.autosaveSessionsOnPromote && !promotedChampionWeights.isEmpty {
            let championID = champion.identifier?.description ?? "unknown"
            let trainerID = trainer.identifier?.description ?? "unknown"
            let sessionState = buildCurrentSessionState(
                championID: championID,
                trainerID: trainerID
            )
            let championMetadata = ModelCheckpointMetadata(
                creator: "promote",
                trainingStep: trainingStats?.steps ?? 0,
                parentModelID: "",
                notes: "Post-arena autosave after promotion"
            )
            let trainerMetadata = ModelCheckpointMetadata(
                creator: "promote",
                trainingStep: trainingStats?.steps ?? 0,
                parentModelID: championID,
                notes: "Trainer lineage at arena-start pause with optimizer velocity"
            )
            let createdAtUnix = Int64(Date().timeIntervalSince1970)
            // Copy captured arrays for clean Sendable semantics
            // (they're already Sendable but this makes the
            // transfer to the detached task explicit).
            let championWeightsSnapshot = promotedChampionWeights
            let trainerWeightsSnapshot = trainerSnapshotWeights + trainerSnapshotVelocity
            let bufferForAutosave = replayBuffer
            // Same main-actor snapshot rule as the manual/periodic
            // path — rings are @MainActor-isolated, so the array
            // copies have to happen here before we go detached.
            let chartSnapshotForAutosave = chartCoordinator?.buildSnapshot()

            checkpoint?.setCheckpointStatus("Saving session (post-promotion)…", kind: .progress)
            checkpoint?.checkpointSaveInFlight = true
            checkpoint?.startSlowSaveWatchdog(label: "session save (post-promotion)")
            // Fire-and-forget detached task. The closure captures
            // only Sendable value types (weight arrays, metadata
            // structs, the session state snapshot, and a few
            // strings) — explicitly NOT `self` — so we can safely
            // run past a session end without touching any View
            // @State. The completion handler hops back to the
            // MainActor to surface the outcome in the status row
            // and to reset `periodicSaveController`'s deadline;
            // that hop is safe because the handler only touches
            // the View's @State (no network or gate reads).
            Task.detached(priority: .utility) {
                [bufferForAutosave, chartSnapshotForAutosave] in
                let outcome: Result<(URL, String), Error>
                do {
                    let url = try await CheckpointManager.saveSession(
                        championWeights: championWeightsSnapshot,
                        championID: championID,
                        championMetadata: championMetadata,
                        championCreatedAtUnix: createdAtUnix,
                        trainerWeights: trainerWeightsSnapshot,
                        trainerID: trainerID,
                        trainerMetadata: trainerMetadata,
                        trainerCreatedAtUnix: createdAtUnix,
                        state: sessionState,
                        replayBuffer: bufferForAutosave,
                        chartSnapshot: chartSnapshotForAutosave,
                        trigger: "promote"
                    )
                    let bufStr: String
                    if let snap = bufferForAutosave?.stateSnapshot() {
                        bufStr = " replay=\(snap.storedCount)/\(snap.capacity)"
                    } else {
                        bufStr = ""
                    }
                    outcome = .success((url, bufStr))
                } catch {
                    outcome = .failure(error)
                }
                await MainActor.run {
                    switch outcome {
                    case .success(let (url, bufStr)):
                        self.checkpoint?.setCheckpointStatus(
                            "Saved \(url.lastPathComponent) (post-promotion)",
                            kind: .success
                        )
                        SessionLogger.shared.log(
                            "[CHECKPOINT] Saved session (post-promotion): \(url.lastPathComponent) build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(bufStr)"
                        )
                        self.checkpoint?.recordLastSessionPointer(
                            directoryURL: url,
                            sessionID: sessionState.sessionID,
                            trigger: "post-promotion"
                        )
                        self.periodicSaveController?.noteSuccessfulSave(at: Date())
                        self.checkpoint?.lastSavedAt = Date()
                    case .failure(let error):
                        self.checkpoint?.setCheckpointStatus(
                            "Autosave failed (post-promotion): \(error.localizedDescription)",
                            kind: .error
                        )
                        SessionLogger.shared.log(
                            "[CHECKPOINT] Saved session (post-promotion) failed: \(error.localizedDescription)"
                        )
                    }
                    self.checkpoint?.cancelSlowSaveWatchdog()
                    self.checkpoint?.checkpointSaveInFlight = false
                }
            }
        }
    }

    /// Emit an expanded multi-line summary of a just-finished arena
    /// tournament to stdout and the session log. Format:
    ///
    ///   [ARENA] #N Candidate vs Champion @ step S
    ///   [ARENA]     Games: G       (played/total)
    ///   [ARENA]     Result: W wins / D draws / L losses (candidate perspective)
    ///   [ARENA]     Score:  51.2% [48.1%, 54.3%]
    ///   [ARENA]     Elo diff: +28 [+13, +34]
    ///   [ARENA]     Draw rate: 17.5%
    ///   [ARENA]     By side:
    ///   [ARENA]       Candidate as white: 54.1%  (W/D/L n/n/n)
    ///   [ARENA]       Candidate as black: 48.3%  (W/D/L n/n/n)
    ///   [ARENA]     batch=… lr=… promote>=… sp.tau=… ar.tau=… workers=… build=…
    ///   [ARENA]     candidate=…  champion=…  trainer=…
    ///   [ARENA]     diversity: unique=…/… (…%) avgDiverge=…
    ///   [ARENA]     Verdict: PROMOTED(auto)=ID | kept    dur=m:ss
    ///   [ARENA] #N kv step=… games=… w=… d=… l=… score=… elo=… elo_lo=… elo_hi=… draw_rate=… cand_white_score=… cand_black_score=… promoted=… kind=… dur_sec=… build=… candidate=… champion=… trainer=…
    ///
    /// The trailing `kv` line stays a single key=value line so
    /// grep-based tooling can pull every arena outcome without
    /// reassembling the multi-line block.
    @MainActor
    private func logArenaResult(
        record: TournamentRecord,
        index: Int,
        trainer: ChessTrainer,
        candidate: ChessMPSNetwork,
        championSide: ChessMPSNetwork,
        diversity: GameDiversityTracker.Snapshot
    ) {
        let sp = samplingScheduleBox?.selfPlay ?? buildSelfPlaySchedule()
        let ar = samplingScheduleBox?.arena ?? buildArenaSchedule()
        let candidateIDStr = candidate.identifier?.description ?? "?"
        let championIDStr = championSide.identifier?.description ?? "?"
        let trainerIDStr = trainer.identifier?.description ?? "?"

        let parameters = ArenaLogFormatter.Parameters(
            batchSize: TrainingParameters.shared.trainingBatchSize,
            learningRate: trainer.learningRate,
            promoteThreshold: TrainingParameters.shared.arenaPromoteThreshold,
            tournamentGames: TrainingParameters.shared.arenaGamesPerTournament,
            spStartTau: sp.startTau,
            spFloorTau: sp.floorTau,
            spDecayPerPly: sp.decayPerPly,
            arStartTau: ar.startTau,
            arFloorTau: ar.floorTau,
            arDecayPerPly: ar.decayPerPly,
            workerCount: TrainingParameters.shared.selfPlayWorkers,
            buildNumber: BuildInfo.buildNumber
        )
        let diversityCtx = ArenaLogFormatter.Diversity(
            uniqueGames: diversity.uniqueGames,
            gamesInWindow: diversity.gamesInWindow,
            uniquePercent: diversity.uniquePercent,
            avgDivergencePly: diversity.avgDivergencePly
        )

        let lines = ArenaLogFormatter.formatHumanReadable(
            record: record,
            index: index,
            candidateID: candidateIDStr,
            championID: championIDStr,
            trainerID: trainerIDStr,
            parameters: parameters,
            diversity: diversityCtx
        )
        let kv = ArenaLogFormatter.formatKVLine(
            record: record,
            index: index,
            candidateID: candidateIDStr,
            championID: championIDStr,
            trainerID: trainerIDStr,
            buildNumber: BuildInfo.buildNumber
        )

        for line in lines {
            print(line)
            SessionLogger.shared.log(line)
        }
        print(kv)
        SessionLogger.shared.log(kv)

        // CLI-mode capture: mirror the arena block's parse-target
        // fields (KV line) plus the extras from the human-readable
        // block that don't appear in the KV but are useful for
        // downstream tooling (tau schedules, diversity snapshot,
        // per-worker count). The recorder serializes these into
        // the arena_results[] array of the `--output` JSON.
        if let recorder = cliRecorder {
            let elo = record.eloSummary
            let entry = CliTrainingRecorder.Arena(
                index: index,
                finishedAtStep: record.finishedAtStep,
                gamesPlayed: record.gamesPlayed,
                tournamentGames: TrainingParameters.shared.arenaGamesPerTournament,
                candidateWins: record.candidateWins,
                championWins: record.championWins,
                draws: record.draws,
                score: record.score,
                drawRate: ArenaLogFormatter.drawRateFraction(record: record),
                elo: elo.elo,
                eloLo: elo.eloLo,
                eloHi: elo.eloHi,
                scoreLo: elo.scoreLo,
                scoreHi: elo.scoreHi,
                candidateWinsAsWhite: record.candidateWinsAsWhite,
                candidateDrawsAsWhite: record.candidateDrawsAsWhite,
                candidateLossesAsWhite: record.candidateLossesAsWhite,
                candidateWinsAsBlack: record.candidateWinsAsBlack,
                candidateDrawsAsBlack: record.candidateDrawsAsBlack,
                candidateLossesAsBlack: record.candidateLossesAsBlack,
                candidateScoreAsWhite: record.candidateScoreAsWhite,
                candidateScoreAsBlack: record.candidateScoreAsBlack,
                promoted: record.promoted,
                promotionKind: record.promotionKind?.rawValue,
                promotedID: record.promotedID?.description,
                durationSec: record.durationSec,
                candidateID: candidateIDStr,
                championID: championIDStr,
                trainerID: trainerIDStr,
                learningRate: Double(trainer.learningRate),
                promoteThreshold: TrainingParameters.shared.arenaPromoteThreshold,
                batchSize: TrainingParameters.shared.trainingBatchSize,
                workerCount: TrainingParameters.shared.selfPlayWorkers,
                spStartTau: Double(sp.startTau),
                spFloorTau: Double(sp.floorTau),
                spDecayPerPly: Double(sp.decayPerPly),
                arStartTau: Double(ar.startTau),
                arFloorTau: Double(ar.floorTau),
                arDecayPerPly: Double(ar.decayPerPly),
                diversityUniqueGames: diversity.uniqueGames,
                diversityGamesInWindow: diversity.gamesInWindow,
                diversityUniquePercent: diversity.uniquePercent,
                diversityAvgDivergencePly: diversity.avgDivergencePly,
                buildNumber: BuildInfo.buildNumber
            )
            recorder.appendArena(entry)
        }
    }

    /// Release arena-active state on an early return from
    /// `runArenaParallel`. Clears the active flag (so the candidate
    /// probe resumes), clears the live-progress box and mirror (so
    /// the busy label reverts to normal Play and Train mode), and
    /// resets the UI's `isArenaRunning` mirror.
    @MainActor
    private func cleanupArenaState(arenaFlag: ArenaActiveFlag, tBox: TournamentLiveBox) {
        arenaFlag.clear()
        isArenaRunning = false
        tBox.clear()
        tournamentProgress = nil
        // Early-exit cleanup — make sure the live arena band on
        // the chart grid isn't left in an "arena active" state
        // after cancellation / error paths that skipped the
        // normal append-then-clear sequence. A no-op on the happy
        // path (the append site already cleared this).
        chartCoordinator?.cancelActiveArena()
        // Release any pending periodic-save fire that was held
        // back during the tournament. The controller decides on
        // the next `decide(now:)` call whether a (post-promotion)
        // successful save landed during the arena window (swallow
        // the pending fire) or not (dispatch the save a little
        // late).
        periodicSaveController?.noteArenaEnded()
    }

    /// Open an arena run: pull the trainer's start-of-arena snapshot (and
    /// mirror it into the live training-stats fields), emit the start log
    /// lines, mark the arena active for the probe / periodic-save / UI
    /// machinery, anchor the chart's live arena band, and seed the
    /// tournament-progress box. Returns the step count, configured game
    /// count, and start time the rest of `runArenaParallel` needs. Split
    /// out so `runArenaParallel`'s body stays under the long-type-check
    /// budget; it's straight-line setup with no behavioral change.
    private func beginArenaRun(
        trainer: ChessTrainer,
        champion: ChessMPSNetwork,
        tBox: TournamentLiveBox,
        arenaFlag: ArenaActiveFlag
    ) async -> (steps: Int, totalGames: Int, startTime: Date, startSnapshot: TrainingLiveStatsBox.Snapshot?) {
        let arenaStartTrainingSnapshot = await trainingBox?.snapshot()
        let steps = arenaStartTrainingSnapshot?.stats.steps ?? trainingStats?.steps ?? 0
        if let arenaStartTrainingSnapshot {
            trainingStats = arenaStartTrainingSnapshot.stats
            lastTrainStep = arenaStartTrainingSnapshot.lastTiming
            realRollingPolicyLoss = arenaStartTrainingSnapshot.rollingPolicyLoss
            realRollingValueLoss = arenaStartTrainingSnapshot.rollingValueLoss
        }

        logArenaStart(steps: steps, trainer: trainer, champion: champion, startSnapshot: arenaStartTrainingSnapshot)

        // Mark arena active and seed live progress. Arena-active
        // suppresses the candidate test probe for the duration so
        // probe and arena don't race on the candidate inference
        // network. isArenaRunning is @State mirror the UI reads to
        // disable the Run Arena button and adjust the busy label.
        arenaFlag.set()
        isArenaRunning = true
        // Let the periodic-save scheduler know an arena is in
        // progress. A 4-hour deadline that crosses while an arena
        // runs will be held as a pending fire and only dispatched
        // once the arena ends — unless a post-promotion autosave
        // lands in the meantime, in which case the pending fire is
        // swallowed (see `PeriodicSaveController`).
        periodicSaveController?.noteArenaBegan()
        // Record the arena's start elapsed-second position so the
        // chart grid's arena activity tile can render a live band
        // as the arena progresses, rather than only showing the
        // arena post-hoc when the completed ArenaChartEvent lands.
        // Anchors off `chartCoordinator?.chartElapsedAnchor` so the
        // arena's x-position lands on the same axis as the training
        // + progress-rate samples. (Routing this off
        // `parallelStats.sessionStart` instead would put restored
        // chart data and post-resume arena bands in different
        // coordinate spaces — the back-dated chart axis vs. the
        // fresh per-segment parallel-stats axis.)
        chartCoordinator?.recordArenaStarted(
            elapsedSec: Date().timeIntervalSince(chartCoordinator?.chartElapsedAnchor ?? Date())
        )

        let totalGames = TrainingParameters.shared.arenaGamesPerTournament
        let startTime = Date()
        tBox.update(TournamentProgress(
            currentGame: 0,
            totalGames: totalGames,
            candidateWins: 0,
            championWins: 0,
            draws: 0,
            startTime: startTime
        ))
        tournamentProgress = tBox.snapshot()
        return (steps, totalGames, startTime, arenaStartTrainingSnapshot)
    }

    /// Emit the `[ARENA] start` line and (if a trainer snapshot is
    /// available) the `[STATS] arena-start` line, capturing the trainer's
    /// loss/entropy/grad-norm state entering the arena — useful for
    /// diagnosing whether divergence was already underway before the arena
    /// ran. Split out of `runArenaParallel` so that function's body stays
    /// under the long-type-check budget; this is pure logging with no
    /// side effects on arena state.
    private func logArenaStart(
        steps: Int,
        trainer: ChessTrainer,
        champion: ChessMPSNetwork,
        startSnapshot: TrainingLiveStatsBox.Snapshot?
    ) {
        let trainerIDStart = trainer.identifier?.description ?? "?"
        let championIDStart = champion.identifier?.description ?? "?"
        SessionLogger.shared.log(
            "[ARENA] start  step=\(steps) trainer=\(trainerIDStart) champion=\(championIDStart)"
        )
        guard let snap = startSnapshot else { return }
        let pStr = snap.rollingPolicyLoss.map { String(format: "%+.4f", $0) } ?? "--"
        let vStr = snap.rollingValueLoss.map { String(format: "%+.4f", $0) } ?? "--"
        let eStr = snap.rollingPolicyEntropy.map { String(format: "%.4f", $0) } ?? "--"
        let gStr = snap.rollingGradGlobalNorm.map { String(format: "%.3f", $0) } ?? "--"
        let vmStr = snap.rollingValueMean.map { String(format: "%+.4f", $0) } ?? "--"
        let vaStr = snap.rollingValueAbsMean.map { String(format: "%.4f", $0) } ?? "--"
        let bufCount = replayBuffer?.count ?? 0
        let bufCap = replayBuffer?.capacity ?? TrainingParameters.shared.replayBufferCapacity
        SessionLogger.shared.log(
            "[STATS] arena-start  steps=\(steps) buffer=\(bufCount)/\(bufCap) pLoss=\(pStr) vLoss=\(vStr) pEnt=\(eStr) gNorm=\(gStr) vMean=\(vmStr) vAbs=\(vaStr) trainer=\(trainerIDStart) champion=\(championIDStart)"
        )
    }

}
