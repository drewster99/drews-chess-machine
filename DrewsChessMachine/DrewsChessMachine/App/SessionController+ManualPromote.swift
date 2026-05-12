import SwiftUI

/// `SessionController`'s "Promote Trainee Now" path — split out of
/// `SessionController.swift` so the arena coordinator (`SessionController+Arena.swift`)
/// stays focused on the tournament loop.
///
/// This is the deliberate, no-arena counterpart to arena promotion:
/// it copies the *current* trainer weights straight into the live
/// champion with no score gate. Use cases are sanity moves the user
/// makes by hand — "the trainee has obviously diverged from the
/// champion and I want it in" — not part of the normal self-play →
/// train → arena loop. Because there is no arena validation, the UI
/// confirms first (see `UpperContentView.promoteTrainerNowFromMenu()`).
///
/// Asymmetry vs. `runArenaParallel`'s promotion block (intentional):
/// the arena path is *rewinding* the champion to an arena-start
/// snapshot, so it restores the optimizer velocity it snapshotted at
/// arena start and rewinds `trainer.completedTrainSteps` to match. We
/// are NOT rewinding — the champion is taking the trainer's *current*
/// state verbatim, so the trainer's velocity and step count are
/// already correct for the weight surface both networks now share.
/// We therefore touch neither, and we leave `trainingBox` / alarms /
/// rolling windows alone too: training continues seamlessly against
/// the new champion.
extension SessionController {

    /// Promote the current trainer weights into the champion. Caller
    /// (the confirmation dialog) has already confirmed; we re-check
    /// the same preconditions here because a stray keyboard-shortcut
    /// / URL-scheme invocation could reach `promoteTrainerNow()`
    /// without the menu disable having gated it.
    @MainActor
    func promoteTrainerNow() {
        // Belt-and-suspenders guards.
        guard realTraining,
              let champion = network,
              let trainer,
              let selfPlayGate = activeSelfPlayGate,
              let trainingGate = activeTrainingGate else {
            onRefuseMenuAction("No active training session — start Play and Train first.")
            return
        }
        if isArenaRunning {
            onRefuseMenuAction("An arena is running. Wait for it to finish, then try again.")
            return
        }
        if checkpoint?.checkpointSaveInFlight == true {
            onRefuseMenuAction("A save is in progress. Wait for it to finish, then try again.")
            return
        }

        // IDs: the trainer's current ID becomes the champion's ID (the
        // champion now holds exactly the trainer's weights), and the
        // trainer rolls forward to a fresh next-generation ID forked
        // from the new champion — same mint/inherit rule as arena
        // promotion (see sampling-parameters.md). `oldChampionID` is
        // recorded on the history entry so the lineage stays traceable.
        let oldChampionID = champion.identifier
        let newChampionID = trainer.identifier ?? ModelIDMinter.mint()

        let trainerNet = trainer.network
        // Mark a checkpoint-affecting operation in flight up front
        // (same pattern as `handleSaveSessionManual`), so a concurrent
        // File ▸ Save Session can't start and fight us for the pause
        // gates during the copy. `saveSessionInternal` re-sets and
        // ultimately clears the flag on the success path; every other
        // exit path below clears it explicitly.
        checkpoint?.checkpointSaveInFlight = true
        checkpoint?.setCheckpointStatus("Promoting trainee → champion…", kind: .progress)

        Task {
            // 1) Pause self-play (evaluates against `champion`) and
            //    training (drives `trainerNet`) so the export/load is
            //    race-free. Bounded waits so a session end mid-promote
            //    can't spin forever.
            let selfPlayAcquired = await selfPlayGate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
            guard selfPlayAcquired else {
                checkpoint?.checkpointSaveInFlight = false
                checkpoint?.setCheckpointStatus("Promotion aborted: could not pause self-play (timeout)", kind: .error)
                SessionLogger.shared.log("[STATS] promote(manual) aborted — self-play pause timeout")
                return
            }
            let trainingAcquired = await trainingGate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
            guard trainingAcquired else {
                selfPlayGate.resume()
                checkpoint?.checkpointSaveInFlight = false
                checkpoint?.setCheckpointStatus("Promotion aborted: could not pause training (timeout)", kind: .error)
                SessionLogger.shared.log("[STATS] promote(manual) aborted — training pause timeout")
                return
            }

            // 2) Copy live trainer weights → champion, on a detached
            //    task so the GPU work doesn't sit on the cooperative
            //    pool. All errors surfaced — never swallowed.
            var copyError: Error?
            do {
                try await Task.detached(priority: .userInitiated) {
                    let weights = try await trainerNet.exportWeights()
                    try await champion.loadWeights(weights)
                }.value
            } catch {
                copyError = error
            }

            if let copyError {
                trainingGate.resume()
                selfPlayGate.resume()
                checkpoint?.checkpointSaveInFlight = false
                trainingBox?.recordError("Promote-trainee copy failed: \(copyError.localizedDescription)")
                checkpoint?.setCheckpointStatus("Promotion failed: \(copyError.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[STATS] promote(manual) failed — \(copyError.localizedDescription)")
                return
            }

            // 3) Re-stamp identities and resume both gates.
            champion.identifier = newChampionID
            trainer.identifier = ModelIDMinter.mintTrainerGeneration(from: newChampionID)
            trainingGate.resume()
            selfPlayGate.resume()

            // 4) Record it. Synthetic `TournamentRecord` with
            //    `gamesPlayed == 0` (no arena was played) so the
            //    promotion shows up in arena history, the promotions
            //    counter, and the persisted session. `ArenaEloStats`
            //    treats a 0-game W/D/L as "no data" (score 0, elo nil),
            //    and the history tile renders "—" for the per-side
            //    breakdown — both safe with zero games.
            let steps = trainingStats?.steps ?? 0
            let record = TournamentRecord(
                finishedAtStep: steps,
                finishedAt: Date(),
                candidateID: newChampionID,
                championID: oldChampionID,
                gamesPlayed: 0,
                candidateWins: 0,
                championWins: 0,
                draws: 0,
                score: 0,
                promoted: true,
                promotionKind: .manual,
                promotedID: newChampionID,
                durationSec: 0,
                candidateWinsAsWhite: 0,
                candidateWinsAsBlack: 0,
                candidateLossesAsWhite: 0,
                candidateLossesAsBlack: 0,
                candidateDrawsAsWhite: 0,
                candidateDrawsAsBlack: 0
            )
            tournamentHistory.append(record)

            // Chart marker — a zero-width "arena" event at the current
            // elapsed second so the activity strip shows the promotion.
            if let chartCoordinator {
                let elapsed = max(0, Date().timeIntervalSince(chartCoordinator.chartElapsedAnchor))
                chartCoordinator.recordArenaCompleted(ArenaChartEvent(
                    id: chartCoordinator.arenaChartEvents.count,
                    startElapsedSec: elapsed,
                    endElapsedSec: elapsed,
                    score: 0,
                    promoted: true
                ))
            }

            // Reset game-play stats so the display reflects only the
            // new champion's self-play, mirroring arena promotion.
            parallelWorkerStatsBox?.resetGameStats()

            let championIDStr = newChampionID.description
            let oldChampionIDStr = oldChampionID?.description ?? "?"
            let trainerIDStr = trainer.identifier?.description ?? "?"
            SessionLogger.shared.log(
                "[STATS] post-promote(manual-no-arena)  steps=\(steps) champion=\(championIDStr) (was \(oldChampionIDStr)) trainer=\(trainerIDStr)"
            )
            checkpoint?.setCheckpointStatus("Promoted trainee → champion \(championIDStr)", kind: .success)

            // 5) Autosave a `.dcmsession` snapshot the same way an arena
            //    promotion does (gated on the same flag). Goes through
            //    the shared save helper, which re-sets `checkpointSaveInFlight`,
            //    does its own gate dance, exports both networks fresh —
            //    by now the champion and trainer are both quiesced and
            //    consistent — and clears the flag when the write
            //    finishes. If the autosave is disabled, clear the flag
            //    here since nothing downstream will.
            if Self.autosaveSessionsOnPromote {
                saveSessionInternal(
                    champion: champion,
                    trainer: trainer,
                    selfPlayGate: selfPlayGate,
                    trainingGate: trainingGate,
                    trigger: .manualPromote
                )
            } else {
                checkpoint?.checkpointSaveInFlight = false
            }
        }
    }
}
