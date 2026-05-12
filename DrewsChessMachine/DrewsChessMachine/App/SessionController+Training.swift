import SwiftUI
import OSLog

/// `SessionController`'s Play-and-Train orchestration — split out of
/// `SessionController.swift` (which had grown past ~6k lines) so the self-play
/// + training + arena-coordination `TaskGroup` (`startRealTraining`, ~2,100
/// lines) and `stopRealTraining` live in their own file. All state these touch
/// is `var` (internal) on `SessionController`, so an extension here has full
/// access without any access-level changes.
extension SessionController {

    // MARK: - Real (self-play) training run (Stage 4o)

    /// Kick off real-data training in parallel mode: self-play, training,
    /// and arena coordination run as three independent tasks inside a
    /// `TaskGroup`, sharing state only through the lock-protected
    /// replay buffer, stats boxes, pause gates, and arena-trigger box.
    /// Self-play plays one game at a time on the champion network and
    /// streams labeled positions into the replay buffer. Training runs
    /// a tight-loop SGD on the trainer network, sampling the buffer
    /// for each batch. The arena coordinator sleeps until triggered
    /// (either by the 30-minute auto-fire or the Run Arena button),
    /// then runs 200 games between the candidate inference network
    /// and a fourth "arena champion" network — both snapshots taken
    /// under brief per-worker pauses so game play and training never
    /// actually stop, even during a tournament.
    ///
    /// `mode` picks how in-memory state from a prior Stop is handled:
    /// see `TrainingStartMode` for the four cases.
    func startRealTraining(mode: TrainingStartMode = .freshOrFromLoadedSession) {
        SessionLogger.shared.log("[BUTTON] Play and Train")
        // Begin a new training segment for cumulative wall-time
        // tracking. Closed via `closeActiveTrainingSegment` on Stop or
        // at save time. Don't try to open one if the previous Stop
        // didn't actually close (defensive — `closeActiveTrainingSegment`
        // is idempotent on nil but we want the log line to be clean).
        if checkpoint?.activeSegmentStart != nil {
            checkpoint?.closeActiveTrainingSegment(reason: "restart-without-stop")
        }
        checkpoint?.beginActiveTrainingSegment()
        precondition(
            UpperContentView.absoluteMaxSelfPlayWorkers >= 1,
            "absoluteMaxSelfPlayWorkers must be >= 1; got \(UpperContentView.absoluteMaxSelfPlayWorkers)"
        )
        // Snap the live N into the [1, absoluteMaxSelfPlayWorkers] range
        // before doing anything else. The Stepper enforces this
        // for user input but `TrainingParameters.shared.selfPlayWorkers` is centrally managed
        // so the value could in principle be edited elsewhere.
        let initialWorkerCount = max(1, min(UpperContentView.absoluteMaxSelfPlayWorkers, TrainingParameters.shared.selfPlayWorkers))
        if initialWorkerCount != TrainingParameters.shared.selfPlayWorkers {
            TrainingParameters.shared.selfPlayWorkers = initialWorkerCount
        }
        guard let trainer = ensureTrainer(), let network else { return }
        let gameWatcher = gameWatcherProvider()
        onResetBoardDisplay()
        trainingAlarm?.clear()
        trainingAlarm?.resetStreaks()

        // `continueMode` controls whether we preserve in-memory state
        // from a prior Stop. When true: reuse replay buffer, stats
        // boxes, session ID, tournament history, chart samples, and
        // trainer weights. The only fresh objects are the transient
        // per-task gates and live schedule boxes (those get cancelled
        // with the task on Stop, so they don't survive).
        let continueMode = (mode == .continueAfterStop)
        if !continueMode {
            onClearTrainingDisplay()
        }

        let buffer: ReplayBuffer
        if continueMode, let existing = replayBuffer {
            buffer = existing
        } else {
            buffer = ReplayBuffer(capacity: TrainingParameters.shared.replayBufferCapacity)
            replayBuffer = buffer
        }
        let box: TrainingLiveStatsBox
        if continueMode, let existing = trainingBox {
            box = existing
        } else {
            let fresh = TrainingLiveStatsBox(rollingWindow: SessionController.rollingLossWindow)
            if let rs = pendingLoadedSession?.state {
                var seeded = TrainingRunStats()
                seeded.steps = rs.trainingSteps
                fresh.seed(seeded)
            }
            trainingBox = fresh
            realRollingPolicyLoss = nil
            realRollingValueLoss = nil
            box = fresh
        }
        // Seed counters from the loaded session if resuming, or
        // start fresh. This covers the stats box (game/move/result
        // counters), training step count, tournament history, and
        // worker count. In `.continueAfterStop` the trainer's
        // hyperparameters and `trainingStats` stay untouched — the
        // user wants to pick up exactly where they left off.
        let resumeState = pendingLoadedSession?.state
        if !continueMode {
            if let rs = resumeState {
                SessionLogger.shared.log(
                    "[RESUME-PARAM] learning_rate: \(TrainingParameters.shared.learningRate) -> \(rs.learningRate) (from session)"
                )
                trainer.learningRate = rs.learningRate
                TrainingParameters.shared.learningRate = Double(rs.learningRate)
                if let entropyCoeff = rs.entropyRegularizationCoeff {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] entropy_bonus: \(TrainingParameters.shared.entropyBonus) -> \(entropyCoeff) (from session)"
                    )
                    trainer.entropyRegularizationCoeff = entropyCoeff
                    TrainingParameters.shared.entropyBonus = Double(entropyCoeff)
                } else {
                    trainer.entropyRegularizationCoeff = Float(TrainingParameters.shared.entropyBonus)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] entropy_bonus: saved=nil applied=\(TrainingParameters.shared.entropyBonus) (defaulted)"
                    )
                }
                if let dp = rs.drawPenalty {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] draw_penalty: \(TrainingParameters.shared.drawPenalty) -> \(dp) (from session)"
                    )
                    trainer.drawPenalty = dp
                    TrainingParameters.shared.drawPenalty = Double(dp)
                } else {
                    trainer.drawPenalty = Float(TrainingParameters.shared.drawPenalty)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] draw_penalty: saved=nil applied=\(TrainingParameters.shared.drawPenalty) (defaulted)"
                    )
                }
                // Regularization knobs that became editable post-v1 session
                // files: hydrate when present, otherwise leave the current
                // @AppStorage-backed value alone but log the fallthrough
                // so a session saved before this field existed never
                // resumes silently under different defaults.
                if let wd = rs.weightDecayCoeff {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] weight_decay: \(TrainingParameters.shared.weightDecay) -> \(wd) (from session)"
                    )
                    trainer.weightDecayC = wd
                    TrainingParameters.shared.weightDecay = Double(wd)
                } else {
                    trainer.weightDecayC = Float(TrainingParameters.shared.weightDecay)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] weight_decay: saved=nil applied=\(TrainingParameters.shared.weightDecay) (defaulted)"
                    )
                }
                if let clip = rs.gradClipMaxNorm {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] grad_clip_max_norm: \(TrainingParameters.shared.gradClipMaxNorm) -> \(clip) (from session)"
                    )
                    trainer.gradClipMaxNorm = clip
                    TrainingParameters.shared.gradClipMaxNorm = Double(clip)
                } else {
                    trainer.gradClipMaxNorm = Float(TrainingParameters.shared.gradClipMaxNorm)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] grad_clip_max_norm: saved=nil applied=\(TrainingParameters.shared.gradClipMaxNorm) (defaulted)"
                    )
                }
                if let plw = rs.policyLossWeight {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] policy_loss_weight: \(TrainingParameters.shared.policyLossWeight) -> \(plw) (from session)"
                    )
                    trainer.policyLossWeight = plw
                    TrainingParameters.shared.policyLossWeight = Double(plw)
                } else {
                    trainer.policyLossWeight = Float(TrainingParameters.shared.policyLossWeight)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] policy_loss_weight: saved=nil applied=\(TrainingParameters.shared.policyLossWeight) (defaulted)"
                    )
                }
                if let vlw = rs.valueLossWeight {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] value_loss_weight: \(TrainingParameters.shared.valueLossWeight) -> \(vlw) (from session)"
                    )
                    trainer.valueLossWeight = vlw
                    TrainingParameters.shared.valueLossWeight = Double(vlw)
                } else {
                    trainer.valueLossWeight = Float(TrainingParameters.shared.valueLossWeight)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] value_loss_weight: saved=nil applied=\(TrainingParameters.shared.valueLossWeight) (defaulted)"
                    )
                }
                if let mu = rs.momentumCoeff {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] momentum_coeff: \(TrainingParameters.shared.momentumCoeff) -> \(mu) (from session)"
                    )
                    trainer.momentumCoeff = mu
                    TrainingParameters.shared.momentumCoeff = Double(mu)
                } else {
                    trainer.momentumCoeff = Float(TrainingParameters.shared.momentumCoeff)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] momentum_coeff: saved=nil applied=\(TrainingParameters.shared.momentumCoeff) (defaulted)"
                    )
                }
                if let imw = rs.illegalMassPenaltyWeight {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] illegal_mass_weight: \(TrainingParameters.shared.illegalMassWeight) -> \(imw) (from session)"
                    )
                    trainer.illegalMassPenaltyWeight = imw
                    TrainingParameters.shared.illegalMassWeight = Double(imw)
                } else {
                    trainer.illegalMassPenaltyWeight = Float(TrainingParameters.shared.illegalMassWeight)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] illegal_mass_weight: saved=nil applied=\(TrainingParameters.shared.illegalMassWeight) (defaulted)"
                    )
                }
                if let lse = rs.policyLabelSmoothingEpsilon {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] policy_label_smoothing_epsilon: \(TrainingParameters.shared.policyLabelSmoothingEpsilon) -> \(lse) (from session)"
                    )
                    trainer.policyLabelSmoothingEpsilon = lse
                    TrainingParameters.shared.policyLabelSmoothingEpsilon = Double(lse)
                } else {
                    trainer.policyLabelSmoothingEpsilon = Float(TrainingParameters.shared.policyLabelSmoothingEpsilon)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] policy_label_smoothing_epsilon: saved=nil applied=\(TrainingParameters.shared.policyLabelSmoothingEpsilon) (defaulted)"
                    )
                }
                if let bsi = rs.batchStatsInterval {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] batch_stats_interval: \(TrainingParameters.shared.batchStatsInterval) -> \(bsi) (from session)"
                    )
                    trainer.batchStatsInterval = bsi
                    TrainingParameters.shared.batchStatsInterval = bsi
                } else {
                    trainer.batchStatsInterval = TrainingParameters.shared.batchStatsInterval
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] batch_stats_interval: saved=nil applied=\(TrainingParameters.shared.batchStatsInterval) (defaulted)"
                    )
                }
                // LR warmup length and sqrt-batch LR scaling are now
                // part of the session schema (Optional, for back-compat
                // with older `.dcmsession` files that pre-date the
                // expansion). When the saved value is present, we
                // restore it onto both the trainer and the @AppStorage
                // mirror so the UI shows what the session was running
                // with — not whatever the user's current global
                // preference happens to be. When absent (older session),
                // we fall through to the @AppStorage value as before.
                // The trainer's internal completed-step counter is
                // seeded from `trainingSteps` so warmup scaling resumes
                // mid-session instead of restarting from zero.
                //
                // Every applied parameter emits a `[RESUME-PARAM]`
                // log line — both the "from session" and the
                // "saved=nil applied=… (defaulted)" branches — so
                // resuming an older session under post-schema-
                // expansion defaults can never silently drift the
                // applied value away from what the session was
                // trained under.
                if let savedSqrt = rs.sqrtBatchScalingForLR {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] sqrt_batch_scaling_lr: \(TrainingParameters.shared.sqrtBatchScalingLR) -> \(savedSqrt) (from session)"
                    )
                    trainer.sqrtBatchScalingForLR = savedSqrt
                    TrainingParameters.shared.sqrtBatchScalingLR = savedSqrt
                } else {
                    trainer.sqrtBatchScalingForLR = TrainingParameters.shared.sqrtBatchScalingLR
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] sqrt_batch_scaling_lr: saved=nil applied=\(TrainingParameters.shared.sqrtBatchScalingLR) (defaulted)"
                    )
                }
                if let savedWarmup = rs.lrWarmupSteps, savedWarmup >= 0 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] lr_warmup_steps: \(TrainingParameters.shared.lrWarmupSteps) -> \(savedWarmup) (from session)"
                    )
                    trainer.lrWarmupSteps = savedWarmup
                    TrainingParameters.shared.lrWarmupSteps = savedWarmup
                } else {
                    trainer.lrWarmupSteps = TrainingParameters.shared.lrWarmupSteps
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] lr_warmup_steps: saved=nil applied=\(TrainingParameters.shared.lrWarmupSteps) (defaulted)"
                    )
                }
                trainer.completedTrainSteps = rs.trainingSteps
                // Run-management knobs that previously lived only in
                // @AppStorage. Same Optional-with-fallback pattern but
                // logged on both branches: a saved=nil line when an
                // older session is resumed under post-`74839ee`
                // defaults makes the silent-fallback regression that
                // motivated this audit impossible.
                if let v = rs.replayBufferMinPositionsBeforeTraining, v >= 0 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] replay_buffer_min_positions_before_training: \(TrainingParameters.shared.replayBufferMinPositionsBeforeTraining) -> \(v) (from session)"
                    )
                    TrainingParameters.shared.replayBufferMinPositionsBeforeTraining = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] replay_buffer_min_positions_before_training: saved=nil applied=\(TrainingParameters.shared.replayBufferMinPositionsBeforeTraining) (defaulted)"
                    )
                }
                if let v = rs.arenaAutoIntervalSec, v > 0 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] arena_auto_interval_sec: \(TrainingParameters.shared.arenaAutoIntervalSec) -> \(v) (from session)"
                    )
                    TrainingParameters.shared.arenaAutoIntervalSec = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] arena_auto_interval_sec: saved=nil applied=\(TrainingParameters.shared.arenaAutoIntervalSec) (defaulted)"
                    )
                }
                if let v = rs.arenaConcurrency, v >= 1 {
                    let clamped = min(UpperContentView.absoluteMaxArenaConcurrency, v)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] arena_concurrency: \(TrainingParameters.shared.arenaConcurrency) -> \(clamped) (from session)"
                    )
                    TrainingParameters.shared.arenaConcurrency = clamped
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] arena_concurrency: saved=nil applied=\(TrainingParameters.shared.arenaConcurrency) (defaulted)"
                    )
                }
                if let v = rs.candidateProbeIntervalSec, v > 0 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] candidate_probe_interval_sec: \(TrainingParameters.shared.candidateProbeIntervalSec) -> \(v) (from session)"
                    )
                    TrainingParameters.shared.candidateProbeIntervalSec = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] candidate_probe_interval_sec: saved=nil applied=\(TrainingParameters.shared.candidateProbeIntervalSec) (defaulted)"
                    )
                }
                if let v = rs.legalMassCollapseThreshold, v > 0, v < 1 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_threshold: \(TrainingParameters.shared.legalMassCollapseThreshold) -> \(v) (from session)"
                    )
                    TrainingParameters.shared.legalMassCollapseThreshold = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_threshold: saved=nil applied=\(TrainingParameters.shared.legalMassCollapseThreshold) (defaulted)"
                    )
                }
                if let v = rs.legalMassCollapseGraceSeconds, v >= 0 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_grace_seconds: \(TrainingParameters.shared.legalMassCollapseGraceSeconds) -> \(v) (from session)"
                    )
                    TrainingParameters.shared.legalMassCollapseGraceSeconds = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_grace_seconds: saved=nil applied=\(TrainingParameters.shared.legalMassCollapseGraceSeconds) (defaulted)"
                    )
                }
                if let v = rs.legalMassCollapseNoImprovementProbes, v >= 1 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_no_improvement_probes: \(TrainingParameters.shared.legalMassCollapseNoImprovementProbes) -> \(v) (from session)"
                    )
                    TrainingParameters.shared.legalMassCollapseNoImprovementProbes = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_no_improvement_probes: saved=nil applied=\(TrainingParameters.shared.legalMassCollapseNoImprovementProbes) (defaulted)"
                    )
                }
                // Sampling schedule — TauConfigCodable is non-Optional on
                // the session schema (added in v1), so no fallback branch.
                // Writes to @AppStorage propagate through
                // `buildSelfPlaySchedule` / `buildArenaSchedule` the next
                // time the schedule box is built below.
                TrainingParameters.shared.selfPlayStartTau = Double(rs.selfPlayTau.startTau)
                TrainingParameters.shared.selfPlayTargetTau = Double(rs.selfPlayTau.floorTau)
                TrainingParameters.shared.selfPlayTauDecayPerPly = Double(rs.selfPlayTau.decayPerPly)
                TrainingParameters.shared.arenaStartTau = Double(rs.arenaTau.startTau)
                TrainingParameters.shared.arenaTargetTau = Double(rs.arenaTau.floorTau)
                TrainingParameters.shared.arenaTauDecayPerPly = Double(rs.arenaTau.decayPerPly)
            } else {
                trainer.learningRate = Float(TrainingParameters.shared.learningRate)
                trainer.entropyRegularizationCoeff = Float(TrainingParameters.shared.entropyBonus)
                trainer.drawPenalty = Float(TrainingParameters.shared.drawPenalty)
                trainer.weightDecayC = Float(TrainingParameters.shared.weightDecay)
                trainer.gradClipMaxNorm = Float(TrainingParameters.shared.gradClipMaxNorm)
                trainer.policyLossWeight = Float(TrainingParameters.shared.policyLossWeight)
                trainer.valueLossWeight = Float(TrainingParameters.shared.valueLossWeight)
                trainer.momentumCoeff = Float(TrainingParameters.shared.momentumCoeff)
                trainer.sqrtBatchScalingForLR = TrainingParameters.shared.sqrtBatchScalingLR
                trainer.lrWarmupSteps = TrainingParameters.shared.lrWarmupSteps
            }
            var initialTrainingStats = TrainingRunStats()
            if let rs = resumeState {
                initialTrainingStats.steps = rs.trainingSteps
            }
            trainingStats = initialTrainingStats
        }
        // Game-run mode is meaningless with N > 1 self-play workers
        // (the live board hides itself behind a "N concurrent games"
        // overlay) and the Game-run / Candidate-test picker is hidden
        // in that case anyway. Default the mode to Candidate test in
        // multi-worker sessions so the user gets a usable left-side
        // panel out of the gate; single-worker sessions keep the
        // historical Game-run default.
        playAndTrainBoardMode = TrainingParameters.shared.selfPlayWorkers > 1 ? .candidateTest : .gameRun
        probeNetworkTarget = .candidate
        candidateProbeDirty = false
        lastCandidateProbeTime = .distantPast
        candidateProbeCount = 0
        // The Training and Arena popovers' edit fields are owned by
        // `trainingSettingsPopover` / `arenaSettingsPopover` and re-seed
        // themselves from `trainingParams` each time the popover
        // opens (its `onAppear`), so there's nothing to re-seed here.
        if continueMode {
            // Preserve `checkpoint?.completedTrainingSegments` and
            // `tournamentHistory` as they stood at Stop. The new
            // training segment has already been opened by
            // `beginActiveTrainingSegment` above.
        } else if let rs = resumeState {
            // Hydrate prior training segments so cumulative wall-time
            // and run-count metrics carry across save/load. Missing
            // (older session files) → empty array, which means this
            // becomes the first segment in the session's history.
            checkpoint?.completedTrainingSegments = rs.trainingSegments ?? []
            tournamentHistory = rs.arenaHistory.map { entry in
                // Legacy session files don't store `gamesPlayed` —
                // reconstruct it from the W/L/D totals (same identity
                // the driver uses when building the live record).
                // Legacy files also don't store `promotionKind` — the
                // manual Promote button didn't exist then, so treat
                // any recorded promotion as automatic.
                let gp = entry.gamesPlayed
                ?? (entry.candidateWins + entry.championWins + entry.draws)
                let kind: PromotionKind?
                if entry.promoted {
                    if let raw = entry.promotionKind,
                       let parsed = PromotionKind(rawValue: raw) {
                        kind = parsed
                    } else {
                        kind = .automatic
                    }
                } else {
                    kind = nil
                }
                return TournamentRecord(
                    finishedAtStep: entry.finishedAtStep,
                    finishedAt: entry.finishedAtUnix.map {
                        Date(timeIntervalSince1970: TimeInterval($0))
                    },
                    candidateID: entry.candidateID.map { ModelID(value: $0) },
                    championID: entry.championID.map { ModelID(value: $0) },
                    gamesPlayed: gp,
                    candidateWins: entry.candidateWins,
                    championWins: entry.championWins,
                    draws: entry.draws,
                    score: entry.score,
                    promoted: entry.promoted,
                    promotionKind: kind,
                    promotedID: entry.promotedID.map { ModelID(value: $0) },
                    durationSec: entry.durationSec,
                    candidateWinsAsWhite: entry.candidateWinsAsWhite ?? 0,
                    candidateWinsAsBlack: entry.candidateWinsAsBlack ?? 0,
                    candidateLossesAsWhite: entry.candidateLossesAsWhite ?? 0,
                    candidateLossesAsBlack: entry.candidateLossesAsBlack ?? 0,
                    candidateDrawsAsWhite: entry.candidateDrawsAsWhite ?? 0,
                    candidateDrawsAsBlack: entry.candidateDrawsAsBlack ?? 0
                )
            }
        } else {
            // Fresh session (first Start of the launch, or one of
            // the new-session modes from the Start dialog). Clear
            // both the arena history and the training-segment
            // wall-time history — "new session" means fresh
            // counters. Pre-existing first-launch behavior left
            // `checkpoint?.completedTrainingSegments` alone (benign, since it
            // was already empty at first launch), but after a
            // Stop→"New session" pick the old value would bleed
            // through, so normalize to "always fresh" here.
            tournamentHistory = []
            checkpoint?.completedTrainingSegments = []
        }
        tournamentProgress = nil
        let tBox = TournamentLiveBox()
        tournamentBox = tBox
        let pStatsBox: ParallelWorkerStatsBox
        if continueMode, let existing = parallelWorkerStatsBox {
            pStatsBox = existing
            // continueMode preserves the prior box's sessionStart, so
            // the segment baseline must stay where it was — don't
            // reset it here, otherwise the very next Stop/Start cycle
            // would re-zero a still-open segment's rate.
        } else if let rs = resumeState {
            // Resumed session: trainer was seeded with `rs.trainingSteps`
            // (see `liveStatsBox.seed(seeded)` in `runRealTraining`),
            // so the rate numerator must subtract that as the
            // segment-start baseline to express this-segment steps
            // over this-segment wall time.
            checkpoint?.trainingStepsAtSegmentStart = rs.trainingSteps
            pStatsBox = ParallelWorkerStatsBox(
                sessionStart: Date(),
                totalGames: rs.selfPlayGames,
                totalMoves: rs.selfPlayMoves,
                totalGameWallMs: rs.totalGameWallMs ?? 0,
                whiteCheckmates: rs.whiteCheckmates ?? 0,
                blackCheckmates: rs.blackCheckmates ?? 0,
                stalemates: rs.stalemates ?? 0,
                fiftyMoveDraws: rs.fiftyMoveDraws ?? 0,
                threefoldRepetitionDraws: rs.threefoldRepetitionDraws ?? 0,
                insufficientMaterialDraws: rs.insufficientMaterialDraws ?? 0,
                trainingSteps: rs.trainingSteps
            )
            if let workerCount = resumeState?.selfPlayWorkerCount {
                TrainingParameters.shared.selfPlayWorkers = max(1, min(UpperContentView.absoluteMaxSelfPlayWorkers, workerCount))
            }
            if let delay = rs.stepDelayMs {
                TrainingParameters.shared.trainingStepDelayMs = delay
            }
            if let spDelay = rs.selfPlayDelayMs {
                TrainingParameters.shared.selfPlayDelayMs = spDelay
            }
            if let autoDelay = rs.lastAutoComputedDelayMs {
                lastAutoComputedDelayMs = autoDelay
            }
            TrainingParameters.shared.replayRatioTarget = rs.replayRatioTarget ?? 1.0
            TrainingParameters.shared.replayRatioAutoAdjust = rs.replayRatioAutoAdjust ?? true
        } else {
            // Fresh session — no resumed steps to subtract.
            checkpoint?.trainingStepsAtSegmentStart = 0
            pStatsBox = ParallelWorkerStatsBox(sessionStart: Date())
        }
        parallelWorkerStatsBox = pStatsBox
        parallelStats = pStatsBox.snapshot()
        let spDiversityTracker: GameDiversityTracker
        if continueMode, let existing = selfPlayDiversityTracker {
            spDiversityTracker = existing
        } else {
            spDiversityTracker = GameDiversityTracker(windowSize: 200)
            selfPlayDiversityTracker = spDiversityTracker
            chartCoordinator?.setDiversityHistogramBars([])
        }
        if !continueMode {
            // Fresh session — wipe every chart-layer field back to a
            // zero state so the new session's chart starts at t=0
            // and doesn't show a visible "step" from the previous
            // session's trailing values.
            chartCoordinator?.reset()
            // Resume path: if a session is being restored AND it was
            // saved with chart-companion files, decode them here on
            // the main actor and seed the chart coordinator. The
            // JSON decode takes ~1-2 s on a 24h session — acceptable
            // as a one-time resume cost, on the same order as the
            // replay-buffer restore that already runs at this point.
            // A decode failure logs a warning and skips the restore;
            // the rest of the session-resume flow continues so a
            // corrupt chart file never blocks the (more important)
            // network/replay-buffer load.
            if let chartURLs = pendingLoadedSession?.chartDataURLs {
                seedChartCoordinatorFromLoadedSession(chartURLs: chartURLs)
            }
        }
        // Single self-play gate. All self-play workers now share one
        // `BatchedMoveEvaluationSource` on the champion network, driven
        // by `BatchedSelfPlayDriver`, so there is exactly one consumer
        // for the arena coordinator to pause. The previous per-secondary
        // gate array is gone along with the secondary networks.
        let selfPlayGate = WorkerPauseGate()
        // Shared current-N holder. Workers poll this to decide
        // whether to play another game or sit in their idle wait.
        // The Stepper writes through it (and to `@State
        // TrainingParameters.shared.selfPlayWorkers simultaneously). Exposed via
        // so the UI can disable the buttons when the box is gone
        // (between sessions).
        let countBox = WorkerCountBox(initial: initialWorkerCount)
        workerCountBox = countBox
        // Live schedule box, seeded from the current @AppStorage tau
        // values. Edits in the tau text fields push new schedules into
        // this box; the self-play driver's slots read at the top of
        // each new game, so the next game played uses the edited
        // values.
        let spSchedule = buildSelfPlaySchedule()
        let arSchedule = buildArenaSchedule()
        let scheduleBox = SamplingScheduleBox(
            selfPlay: spSchedule,
            arena: arSchedule
        )
        samplingScheduleBox = scheduleBox
        let ratioController = ReplayRatioController(
            batchSize: TrainingParameters.shared.trainingBatchSize,
            targetRatio: TrainingParameters.shared.replayRatioTarget,
            autoAdjust: TrainingParameters.shared.replayRatioAutoAdjust,
            initialDelayMs: TrainingParameters.shared.replayRatioAutoAdjust
            ? lastAutoComputedDelayMs
            : TrainingParameters.shared.trainingStepDelayMs,
            maxTrainingStepDelayMs: UpperContentView.stepDelayMaxMs,
            maxSelfPlayDelayMs: UpperContentView.selfPlayDelayMaxMs
        )
        // Seed the controller's manual SP-delay slot from the
        // persisted training parameter so a session that starts in
        // manual mode inherits whatever the user last left in the
        // SP-delay stepper, instead of falling back to 0.
        ratioController.manualSelfPlayDelayMs = TrainingParameters.shared.selfPlayDelayMs
        replayRatioController = ratioController
        let trainingGate = WorkerPauseGate()
        let arenaFlag = ArenaActiveFlag()
        arenaActiveFlag = arenaFlag
        let triggerBox = ArenaTriggerBox()
        arenaTriggerBox = triggerBox
        let overrideBox = ArenaOverrideBox()
        arenaOverrideBox = overrideBox
        isArenaRunning = false
        realTraining = true
        // Arm the 4-hour periodic-save scheduler. Always construct
        // a fresh controller on each start — a previous stop will
        // have nil'd it out, and `.continueAfterStop` intentionally
        // resets the next-save deadline to a full interval from
        // now rather than inheriting whatever time was left on the
        // prior arm, since the whole point of the scheduler is
        // "saved within the last 4 hours" and the session was not
        // being saved while stopped.
        let controller = PeriodicSaveController(interval: UpperContentView.periodicSaveIntervalSec)
        controller.arm(now: Date())
        periodicSaveController = controller
        periodicSaveLastPollAt = Date()
        periodicSaveInFlight = false

        // Expose the two gates the checkpoint save path needs and
        // anchor the session ID + wall clock. `checkpoint?.currentSessionID`
        // is either a fresh mint or the loaded session's ID when
        // resuming. `checkpoint?.currentSessionStart` is back-dated on
        // resume by the loaded session's `elapsedTrainingSec`, so
        // successive save-resume-save cycles accumulate elapsed
        // time monotonically. This anchor is only read by the
        // save path (`buildCurrentSessionState`) — the parallel
        // worker stats box keeps its own fresh `sessionStart`
        // anchor for rate-display purposes, so games/hr doesn't
        // get polluted by the back-dated hours.
        activeSelfPlayGate = selfPlayGate
        activeTrainingGate = trainingGate
        if continueMode {
            // Preserve `checkpoint?.currentSessionID`, `checkpoint?.currentSessionStart`,
            // `replayRatioTarget`, `replayRatioAutoAdjust`, and any
            // trainer-hyperparameter edits the user made between
            // Stop and Start. Nothing to do here.
        } else if let resumed = pendingLoadedSession {
            checkpoint?.currentSessionID = resumed.state.sessionID
            checkpoint?.currentSessionStart = Date().addingTimeInterval(-resumed.state.elapsedTrainingSec)
            TrainingParameters.shared.learningRate = Double(resumed.state.learningRate)
            if let entropyCoeff = resumed.state.entropyRegularizationCoeff {
                TrainingParameters.shared.entropyBonus = Double(entropyCoeff)
            }
        } else {
            checkpoint?.currentSessionID = ModelIDMinter.mint().value
            checkpoint?.currentSessionStart = Date()
        }

        // CLI capture mode: allocate the recorder before the
        // training Task starts so every downstream capture site
        // (stats logger, arena, candidate probe) can append to it.
        // Allocation fires when `--train` is on (headless CLI run,
        // with a potential `training_time_limit` that needs a JSON
        // artifact at expiry) OR when `--output <file>` was
        // supplied (user wants artifact regardless of train mode).
        // The recorder is `final class @unchecked Sendable` so we
        // can capture the reference into each child task without
        // wrestling MainActor isolation at each append. Cleared in
        // the teardown block along with the other per-session state.
        let recorder: CliTrainingRecorder?
        if cliOutputURL != nil || autoTrainOnLaunch {
            let r = CliTrainingRecorder()
            r.setSessionID(checkpoint?.currentSessionID)
            recorder = r
            cliRecorder = r
        } else {
            recorder = nil
        }
        let outputURL = cliOutputURL
        let cliTrainingTimeLimitSec = cliConfig?.trainingTimeLimitSec
        let isAutoTrainRun = autoTrainOnLaunch
        let runStart = Date()

        // Register the early-stop flush handler so SIGUSR1 / SIGHUP /
        // applicationShouldTerminate can write `result.json` cleanly
        // before exiting. Cleared in the teardown block. The closure
        // captures the recorder, outputURL, and runStart so the
        // coordinator doesn't need to know about ContentView's state
        // shape — it just calls the closure with the termination reason.
        if let recorder {
            EarlyStopCoordinator.shared.earlyStopHandler = { reason in
                let elapsed = Date().timeIntervalSince(runStart)
                let destDescription = outputURL?.path ?? "<stdout>"
                SessionLogger.shared.log(
                    "[APP] --train: early-stop on \(reason.rawValue) at elapsed=\(String(format: "%.1f", elapsed))s; writing snapshot to \(destDescription)"
                )
                recorder.setTerminationReason(reason)
                let counts = recorder.countsSnapshot()
                do {
                    if let url = outputURL {
                        try recorder.writeJSON(to: url, totalTrainingSeconds: elapsed)
                    } else {
                        try recorder.writeJSONToStdout(totalTrainingSeconds: elapsed)
                    }
                    SessionLogger.shared.log(
                        "[APP] --train: wrote snapshot to \(destDescription) (arenas=\(counts.arenas), stats=\(counts.stats), probes=\(counts.probes))"
                    )
                } catch {
                    SessionLogger.shared.log(
                        "[APP] --train: early-stop snapshot write FAILED for \(destDescription): \(error.localizedDescription)"
                    )
                }
                SessionLogger.shared.log("[APP] --train: exiting process after early-stop snapshot")
                Darwin._exit(0)
            }
        }

        // Snapshot the CLI-overridable effective values into plain
        // `let`s so the detached Task bodies below can read them
        // without a MainActor hop. These values don't change after
        // session start in the current model, so a one-time capture
        // is safe and keeps the Task-side code MainActor-free.
        let sessionTrainingBatchSize = TrainingParameters.shared.trainingBatchSize
        let sessionMinBufferBeforeTraining = TrainingParameters.shared.replayBufferMinPositionsBeforeTraining
        let sessionTournamentGames = TrainingParameters.shared.arenaGamesPerTournament
        let sessionPromoteThreshold = TrainingParameters.shared.arenaPromoteThreshold

        realTrainingTask = Task(priority: .high) {
            [trainer, network, buffer, box, tBox, pStatsBox, spDiversityTracker,
             selfPlayGate, trainingGate, arenaFlag, triggerBox, overrideBox, countBox,
             gameWatcher, ratioController, recorder, outputURL, cliTrainingTimeLimitSec,
             isAutoTrainRun,
             sessionTrainingBatchSize, sessionMinBufferBeforeTraining,
             sessionTournamentGames, sessionPromoteThreshold] in

            // --- Setup: build any missing networks, reset the trainer ---

            let needsCandidateBuild = await MainActor.run { candidateInferenceNetwork == nil }
            if needsCandidateBuild {
                do {
                    let built = try await Task.detached(priority: .userInitiated) {
                        try ChessMPSNetwork(.randomWeights)
                    }.value
                    built.network.commandQueue.label = "startrealTraining candidate Inference"
                    await MainActor.run {
                        candidateInferenceNetwork = built
                    }
                } catch {
                    box.recordError("Candidate network init failed: \(error.localizedDescription)")
                    await MainActor.run {
                        realTraining = false
                        realTrainingTask = nil
                    }
                    return
                }
            }

            let needsProbeBuild = await MainActor.run { probeInferenceNetwork == nil }
            if needsProbeBuild {
                do {
                    let built = try await Task.detached(priority: .userInitiated) {
                        try ChessMPSNetwork(.randomWeights)
                    }.value
                    built.network.commandQueue.label = "startrealTraining probe Inference"
                    await MainActor.run {
                        probeInferenceNetwork = built
                        probeRunner = ChessRunner(network: built)
                    }
                } catch {
                    box.recordError("Probe network init failed: \(error.localizedDescription)")
                    await MainActor.run {
                        realTraining = false
                        realTrainingTask = nil
                    }
                    return
                }
            }

            let needsArenaChampionBuild = await MainActor.run { arenaChampionNetwork == nil }
            if needsArenaChampionBuild {
                do {
                    let built = try await Task.detached(priority: .userInitiated) {
                        try ChessMPSNetwork(.randomWeights)
                    }.value
                    built.network.commandQueue.label = "startrealTraining arena champion"
                    await MainActor.run {
                        arenaChampionNetwork = built
                    }
                } catch {
                    box.recordError("Arena champion init failed: \(error.localizedDescription)")
                    await MainActor.run {
                        realTraining = false
                        realTrainingTask = nil
                    }
                    return
                }
            }

            // Reset the trainer's graph AND initialize its weights,
            // branched by `mode`:
            //
            // - `.freshOrFromLoadedSession`: rebuild graph, then
            //   load trainer weights from the loaded session's
            //   `trainer.dcmmodel` if present, or fork them from
            //   the champion. This is the default first-launch and
            //   disk-resume path.
            // - `.newSessionResetTrainerFromChampion`: rebuild
            //   graph, fork trainer weights from the live champion.
            //   User explicitly asked to throw away accumulated
            //   trainer weights and start fresh from champion.
            // - `.continueAfterStop` and `.newSessionKeepTrainer`:
            //   leave the trainer's graph and weights untouched —
            //   the trainer's MPSGraph stayed alive across Stop,
            //   and preserving its weights is the whole point of
            //   both modes.
            //
            // In the fork-from-champion branch, champion weights
            // loaded from disk at file-load time are already live
            // in `network`, so the batcher (which wraps `network`)
            // picks up the restored weights automatically.
            let resumedTrainerWeights: [[Float]]? = await MainActor.run {
                pendingLoadedSession?.trainerFile.weights
            }
            let resumedBufferURL: URL? = await MainActor.run {
                pendingLoadedSession?.replayBufferURL
            }
            do {
                switch mode {
                case .continueAfterStop, .newSessionKeepTrainer:
                    break
                case .freshOrFromLoadedSession:
                    try await trainer.resetNetwork()
                    try await Task.detached(priority: .userInitiated) {
                        [resumedTrainerWeights] in
                        if let trainerWeights = resumedTrainerWeights {
                            // Session resume requires exact trainer
                            // state, including optimizer velocity.
                            try await trainer.loadTrainerWeights(trainerWeights)
                        } else {
                            // No prior trainer file (fresh session or
                            // pre-existing session without trainer.dcmmodel):
                            // fork from champion and intentionally
                            // start optimizer velocity from zero.
                            let championWeights = try await network.exportWeights()
                            try await trainer.loadBaseWeightsResetVelocity(championWeights)
                        }
                    }.value
                case .newSessionResetTrainerFromChampion:
                    try await trainer.resetNetwork()
                    try await Task.detached(priority: .userInitiated) {
                        // User explicitly asked to discard trainer state
                        // and re-fork from champion. Velocity goes back
                        // to zero, which is the correct semantics for
                        // a fresh fork.
                        let championWeights = try await network.exportWeights()
                        try await trainer.loadBaseWeightsResetVelocity(championWeights)
                    }.value
                }
            } catch {
                box.recordError("Reset failed: \(error.localizedDescription)")
                await MainActor.run {
                    realTraining = false
                    realTrainingTask = nil
                }
                return
            }

            // Dump every live MTLCommandQueue's (label, address) pair
            // so subsequent Metal-trace captures can be matched back
            // to the network they came from. The address printed here
            // is what Xcode's Metal frame-capture UI shows under
            // "Command Queues" on the trace selection sheet — pick by
            // label, then verify by address. Emitted once per session
            // start; queues persist for the app lifetime so this is
            // accurate for every later capture in the same launch.
            do {
                let candidateQ = await MainActor.run {
                    candidateInferenceNetwork?.network.commandQueue
                }
                let probeQ = await MainActor.run {
                    probeInferenceNetwork?.network.commandQueue
                }
                let arenaChampionQ = await MainActor.run {
                    arenaChampionNetwork?.network.commandQueue
                }
                let championQ = network.network.commandQueue
                let trainerQ = trainer.network.commandQueue
                func desc(_ label: String, _ q: MTLCommandQueue?) -> String {
                    guard let q else { return "\(label)=<not built>" }
                    // Pointer that matches the address Xcode's Metal
                    // frame-capture UI shows in the queue selector.
                    let opaque = Unmanaged.passUnretained(q as AnyObject).toOpaque()
                    let addr = String(format: "0x%016lx", UInt(bitPattern: opaque))
                    return "\(label)=\(q.label ?? "<no-label>") @ \(addr)"
                }
                SessionLogger.shared.log(
                    "[QUEUES] champion: \(desc("champion", championQ)); "
                    + "trainer: \(desc("trainer", trainerQ)); "
                    + "candidate: \(desc("candidate", candidateQ)); "
                    + "probe: \(desc("probe", probeQ)); "
                    + "arenaChampion: \(desc("arenaChampion", arenaChampionQ))"
                )
            }

            // Restore replay buffer before any self-play worker or
            // the training worker starts — training samples from the
            // buffer, and any worker appends after restore resets
            // would either be clobbered or (worse) race with the
            // restore's counter reset. The restore runs on a
            // detached I/O task so ~GB-scale reads don't block the
            // cooperative hop cadence.
            if let bufferURL = resumedBufferURL {
                let resumedState: SessionCheckpointState? = await MainActor.run {
                    pendingLoadedSession?.state
                }
                do {
                    try await Task.detached(priority: .userInitiated) {
                        [buffer, bufferURL] in
                        try buffer.restore(from: bufferURL)
                    }.value
                    // Cross-check lifetime counter against session.json.
                    // Mismatch here indicates file-pairing error or
                    // residual corruption that happened to SHA-match.
                    if let resumedState {
                        try CheckpointManager.verifyReplayBufferMatchesSession(
                            buffer: buffer,
                            state: resumedState
                        )
                    }
                    let snap = buffer.stateSnapshot()
                    SessionLogger.shared.log(
                        "[CHECKPOINT] Restored replay buffer: stored=\(snap.storedCount)/\(snap.capacity) totalAdded=\(snap.totalPositionsAdded) writeIndex=\(snap.writeIndex)"
                    )
                } catch {
                    box.recordError("Replay buffer restore failed: \(error.localizedDescription)")
                    SessionLogger.shared.log(
                        "[CHECKPOINT] Replay buffer restore failed: \(error.localizedDescription) — continuing with empty buffer"
                    )
                }
            }

            // Trainer ID — branched by `mode`, matching the
            // trainer-weights logic above:
            //
            // - `.freshOrFromLoadedSession`: inherit from the
            //   loaded session's trainer file if present, else
            //   mint a new generation off the champion.
            // - `.newSessionResetTrainerFromChampion`: mint a new
            //   generation off the current champion (trainer
            //   weights were just forked from champion).
            // - `.continueAfterStop` and `.newSessionKeepTrainer`:
            //   keep the trainer's existing ID — its weights
            //   weren't touched, so the lineage is continuous.
            await MainActor.run {
                switch mode {
                case .continueAfterStop, .newSessionKeepTrainer:
                    break
                case .newSessionResetTrainerFromChampion:
                    trainer.identifier = ModelIDMinter.mintTrainerGeneration(
                        from: network.identifier ?? ModelIDMinter.mint()
                    )
                case .freshOrFromLoadedSession:
                    if let resumed = pendingLoadedSession {
                        trainer.identifier = ModelID(value: resumed.trainerFile.modelID)
                    } else {
                        trainer.identifier = ModelIDMinter.mintTrainerGeneration(
                            from: network.identifier ?? ModelIDMinter.mint()
                        )
                    }
                }
                // Consume the pending load — from here on, the
                // running session owns the restored state.
                pendingLoadedSession = nil
                // A training segment has started, so clear the
                // "champion replaced since last training" flag
                // (the Start dialog's annotation is resolved).
                championLoadedSinceLastTrainingSegment = false
            }

            // Grab the candidate inference network and arena champion
            // network references on the main actor once — both are
            // now guaranteed non-nil from the setup above. The
            // workers capture them as values for the duration of the
            // 
            let candidateInference = await MainActor.run { candidateInferenceNetwork }
            let arenaChampion = await MainActor.run { arenaChampionNetwork }
            guard let candidateInference, let arenaChampion else {
                box.recordError("Networks missing after setup")
                await MainActor.run {
                    realTraining = false
                    realTrainingTask = nil
                }
                return
            }

            // --- Spawn the worker tasks ---
            //
            // absoluteMaxSelfPlayWorkers self-play tasks, one training
            // worker, one arena coordinator, one session-log
            // ticker. The Stepper picks how many of the self-play
            // tasks are *active* at any moment — the rest sit in
            // their pause gate's wait state until the user raises
            // N enough to include them.

            // Anchor the session wall-clock to *now*, after all the
            // synchronous and MainActor-hop setup above has finished.
            // Rate denominators ("steps/sec", "games/hr", "avg move
            // ms", "Total session time", ...) are computed as `Date()
            // - sessionStart`, so leaving the original `Date()`-at-
            // button-press anchor in place would bake the setup
            // delay into every average for the life of the 
            pStatsBox.markWorkersStarted()

            // Build the shared self-play batcher and driver. All slots
            // play against one `ChessMPSNetwork` (the champion, via
            // `network`) through a `BatchedMoveEvaluationSource` actor
            // that coalesces N per-ply forward passes into one batched
            // `graph.run`. The driver owns the slot lifecycle: it
            // spawns up to `countBox.count` child tasks, responds to
            // Stepper-driven count changes, and pauses the whole
            // self-play subsystem through the shared `selfPlayGate`
            // when arena requests it (replacing the previous
            // per-worker gate array).
            let selfPlayBatcher = BatchedMoveEvaluationSource(network: network)
            // Wire the ratio controller into the batcher so every
            // barrier fire reports an aggregate (positions, elapsed)
            // measurement. `setReplayRatioController` is actor-
            // isolated, so it's dispatched via a `Task`.
            Task { [ratioController] in
                await selfPlayBatcher.setReplayRatioController(ratioController)
            }
            let selfPlayDriver = BatchedSelfPlayDriver(
                batcher: selfPlayBatcher,
                buffer: buffer,
                statsBox: pStatsBox,
                diversityTracker: spDiversityTracker,
                countBox: countBox,
                pauseGate: selfPlayGate,
                gameWatcher: gameWatcher,
                scheduleBox: scheduleBox,
                replayRatioController: ratioController
            )

            // Pin the probe inference network into a local the child
            // tasks can capture. It's a @State on this view, so
            // reading it requires a MainActor hop that child tasks
            // would otherwise repeat. Built lazily earlier in this
            // function (lines ~7694-7703); should be non-nil here but
            // we tolerate nil and skip probes in that case rather
            // than fall back to the pollution-prone trainer-network
            // probe path.
            let probeInferenceForProbes: ChessMPSNetwork? = await MainActor.run {
                probeInferenceNetwork
            }

            await withTaskGroup(of: Void.self) { group in
                // Self-play driver: manages N concurrent
                // `ChessMachine` game loops against the shared
                // batcher. Slot count tracks `countBox.count` live;
                // pause requests on `selfPlayGate` flow through the
                // driver and stop every slot at its next game
                // boundary. Each slot streams positions into the
                // shared replay buffer via its white/black
                // `MPSChessPlayer` pair, records stats through
                // `pStatsBox`, and feeds `spDiversityTracker`.
                //
                // Slot 0 drives the live-display `GameWatcher` when
                // there is exactly one active slot; all slots
                // contribute identically to `pStatsBox` /
                // `spDiversityTracker`.
                let selfPlayTaskCreatedAt = Date()
                group.addTask(priority: .high) {
                    [selfPlayDriver, selfPlayTaskCreatedAt] in
                    let creatingMs = Int(Date().timeIntervalSince(selfPlayTaskCreatedAt) * 1000)
                    SessionLogger.shared.log("[TASK] self-play driver: created→exec=\(creatingMs)ms")
                    await selfPlayDriver.run()
                }

                // Training worker: tight-loop SGD on the trainer,
                // sampling batches from the replay buffer. Fires the
                // candidate probe at its own 15 s cadence between
                // steps, and nudges the arena trigger box when the
                // 30 min auto cadence elapses. Pauses at `trainingGate`
                // so the arena coordinator can briefly snapshot
                // trainer weights.
                let trainingTaskCreatedAt = Date()
                group.addTask(priority: .high) {
                    [trainer, buffer, box, pStatsBox, trainingGate, triggerBox, ratioController,
                     sessionTrainingBatchSize, sessionMinBufferBeforeTraining, trainingTaskCreatedAt] in
                    let creatingMs = Int(Date().timeIntervalSince(trainingTaskCreatedAt) * 1000)
                    SessionLogger.shared.log("[TASK] training worker: created→exec=\(creatingMs)ms")
                    // Track the previous step's applied delay so the
                    // next `recordTrainingBatchAndGetDelay` can report
                    // it as the current per-batch training-side delay
                    // setting. The controller owns the inter-call
                    // wall-clock measurement directly; the caller
                    // just reports its current configured delay.
                    var lastTrainingDelaySettingMs: Int = 0
                    while !Task.isCancelled {
                        // Pause gate check (between steps).
                        if trainingGate.isRequestedToPause {
                            trainingGate.markWaiting()
                            while trainingGate.isRequestedToPause && !Task.isCancelled {
                                // `try?` is intentional: `Task.sleep` only
                                // throws `CancellationError`, and the loop
                                // condition re-checks `Task.isCancelled` on
                                // the very next iteration.
                                try? await Task.sleep(for: .milliseconds(5))
                            }
                            trainingGate.markRunning()
                        }
                        if Task.isCancelled { break }

                        // Wait for the replay buffer to warm up before
                        // starting to train — the first few games
                        // haven't produced enough decorrelated samples
                        // yet. Short sleep + retry keeps the worker
                        // responsive to Stop and to pause requests.
                        if buffer.count < sessionMinBufferBeforeTraining {
                            // `try?` is intentional: only `CancellationError`
                            // can be thrown; the enclosing
                            // `while !Task.isCancelled` exits on the next
                            // iteration.
                            try? await Task.sleep(for: .milliseconds(100))
                            continue
                        }

                        // The enclosing worker already runs at
                        // `.userInitiated` so there's nothing to escape
                        // to — run the SGD step inline and skip the
                        // per-step detached-task + continuation
                        // allocation pair. Sampling happens inside the
                        // trainer on its serial queue so replay-buffer
                        // rows are copied directly into trainer-owned
                        // staging buffers.
                        let timing: TrainStepTiming
                        do {
                            guard let sampledTiming = try await trainer.trainStep(
                                replayBuffer: buffer,
                                batchSize: sessionTrainingBatchSize
                            ) else {
                                // `try?` is intentional: only `CancellationError`
                                // is throwable; the enclosing loop's
                                // `!Task.isCancelled` check exits cleanly.
                                try? await Task.sleep(for: .milliseconds(100))
                                continue
                            }
                            timing = sampledTiming
                        } catch {
                            box.recordError(error.localizedDescription)
                            return
                        }

                        box.recordStep(timing)
                        pStatsBox.recordTrainingStep()

                        // Candidate-test probe firing check. Method
                        // guards internally on all preconditions
                        // including arena-active, so an unconditional
                        // call here is safe — it no-ops when nothing
                        // is due or when the arena is running.
                        await self.fireCandidateProbeIfNeeded()

                        // Auto-trigger the arena on the configured
                        // cadence. Fires the trigger inbox; the arena
                        // coordinator task picks it up and runs the
                        // tournament. Reads `arenaAutoIntervalSec` live
                        // (the parameter is `liveTunable: true`) so the
                        // status-bar Arena popover's Save edits take
                        // effect on the next poll without restarting
                        // the 
                        //
                        // CRITICAL: The arena clock only begins ticking
                        // once the model has reached stability (buffer
                        // prefill AND LR warmup complete). While either
                        // is in progress, we keep resetting the anchor
                        // to 'now' so the first auto-arena occurs exactly
                        // one interval after the model matures.
                        let liveParams = await TrainingParameters.shared.snapshot()
                        let liveInterval = liveParams.arenaAutoIntervalSec
                        let isWarmup = trainer.completedTrainSteps < liveParams.lrWarmupSteps
                        if isWarmup {
                            triggerBox.resetLastArenaTime(to: Date())
                        } else if triggerBox.shouldAutoTrigger(interval: liveInterval) {
                            triggerBox.trigger()
                        }

                        // Post-step pause. Symmetric API with the
                        // self-play barrier tick: caller reports its
                        // currently-applied per-batch delay setting,
                        // controller owns the wall-clock and does
                        // the subtraction. The "current setting"
                        // we report is whatever sleep we just
                        // applied on the previous loop pass — that
                        // IS the configured delay during the
                        // controller-owned inter-call period that
                        // ends right now.
                        let stepDelayMs = ratioController.recordTrainingBatchAndGetDelay(
                            currentDelaySettingMs: Double(lastTrainingDelaySettingMs)
                        )
                        lastTrainingDelaySettingMs = stepDelayMs
                        if stepDelayMs > 0 {
                            // `try?` is intentional: cancellation is the
                            // normal exit path; the enclosing
                            // `while !Task.isCancelled` re-checks
                            // cancellation on the next pass.
                            try? await Task.sleep(for: .milliseconds(stepDelayMs))
                        }
                    }
                }

                // Arena coordinator: polls the trigger inbox and runs
                // a tournament whenever one is pending. Blocks its
                // own loop (not the worker tasks) during arena
                // execution. Both the 30-minute auto-fire and the
                // Run Arena button enter here via `triggerBox.trigger()`.
                let arenaTaskCreatedAt = Date()
                group.addTask(priority: .high) {
                    [arenaTaskCreatedAt] in
                    let creatingMs = Int(Date().timeIntervalSince(arenaTaskCreatedAt) * 1000)
                    SessionLogger.shared.log("[TASK] arena coordinator: created→exec=\(creatingMs)ms")
                    // Event-driven wait — sleeps until a trigger
                    // fires, the box is cancelled, or the Task is
                    // cancelled. The trigger-box's tri-state return
                    // is the single source of truth for what to do
                    // next — no separate `Task.isCancelled` check
                    // needed after the await.
                    arenaLoop: while true {
                        switch await triggerBox.waitForTrigger() {
                        case .cancelled:
                            break arenaLoop
                        case .falseAlarm:
                            continue arenaLoop
                        case .fire:
                            await self.runArenaParallel(
                                trainer: trainer,
                                champion: network,
                                candidateInference: candidateInference,
                                arenaChampion: arenaChampion,
                                tBox: tBox,
                                selfPlayGate: selfPlayGate,
                                trainingGate: trainingGate,
                                arenaFlag: arenaFlag,
                                overrideBox: overrideBox
                            )
                            triggerBox.recordArenaCompleted()
                        }
                    }
                }

                // Periodic session-log ticker. Emits one [STATS] line
                // per training step for the first 500 steps (every step
                // matters during bootstrap — you want to see the curve
                // shape of the first few hundred updates) then drops to
                // one line per 60 seconds for the rest of the 
                //
                // Each wake-up snapshots the thread-safe stats boxes,
                // optionally refreshes `legalMass` via a sampled
                // forward pass (cadence-gated so the CPU work doesn't
                // pile up during the per-step bootstrap window), and
                // writes one `[STATS]` line. Identifiers are pulled
                // through a brief MainActor hop since they live on
                // classes whose var mutation is otherwise main-actor-
                // driven.
                group.addTask(priority: .utility) {
                    [trainer, network, box, pStatsBox, buffer, spDiversityTracker, ratioController, countBox, scheduleBox, recorder,
                     sessionTrainingBatchSize, sessionTournamentGames, sessionPromoteThreshold, probeInferenceForProbes] in
                    let sessionStart = Date()
                    // Bootstrap-phase step threshold for the per-step
                    // emit. `UpperContentView.bootstrapStatsStepCount` is tunable
                    // on the view; at default 500 steps this covers
                    // roughly the first 1-3 minutes of real-data
                    // training at typical throughput.
                    let bootstrapSteps = UpperContentView.bootstrapStatsStepCount
                    // Time between STATS emits after the bootstrap
                    // window closes. 60 s chosen so a session's
                    // steady-state log file grows at a manageable rate
                    // (~60 lines/hr) while still capturing drift
                    // inside the typical 30-minute arena cadence.
                    let steadyInterval: TimeInterval = 60
                    // Cadence for refreshing legalMass during the
                    // per-step bootstrap window — refreshing every
                    // step would double per-step CPU cost for little
                    // additional signal. Every 25 steps roughly
                    // matches the 60-second cadence used afterwards
                    // at typical throughput.
                    let legalMassBootstrapStride = 25
                    let legalMassSampleSize = 128

                    // First-observed pwNorm becomes the session baseline
                    // so each [STATS] line can report the absolute value
                    // alongside the drift since session start. Captured
                    // lazily on the first non-nil reading rather than at
                    // task launch — when training has just kicked off
                    // the trainer's rolling weight-norm window may not
                    // be populated yet. Mutated from inside the nested
                    // `logOne` closure; safe because every call site
                    // runs serially on this task.
                    var pwNormBaseline: Double? = nil

                    // Carried across [STATS] emits so each line can
                    // report the delta since the previous tick. nil
                    // until the first sample lands; the very first emit
                    // shows `drss=+0` etc., which is fine — the deltas
                    // are only meaningful from the second tick onward.
                    var prevRssBytes: UInt64 = 0
                    var prevVmTotal: UInt32 = 0
                    var prevVmIoAccel: UInt32 = 0

                    func logOne(elapsedTarget: TimeInterval, legalMassOverride: ChessTrainer.LegalMassSnapshot?) async {
                        let trainingSnap = await box.snapshot()
                        let parallelSnap = pStatsBox.snapshot()
                        let bufCount = buffer.count
                        let bufCap = buffer.capacity
                        let ratioSnap = ratioController.snapshot()
                        let workerN = countBox.count
                        let spSched = scheduleBox.selfPlay
                        let arSched = scheduleBox.arena
                        let (trainerID, championID, lr, entropyCoeff, illegalMassW, drawPen, weightDec, gradClip, policyW, valueW, momentum, sqrtLR, warmupSteps, completedSteps, arenaAutoSec, livePromoteThreshold, liveTournamentGames) = await MainActor.run {
                            (
                                trainer.identifier?.description ?? "?",
                                network.identifier?.description ?? "?",
                                trainer.learningRate,
                                trainer.entropyRegularizationCoeff,
                                trainer.illegalMassPenaltyWeight,
                                trainer.drawPenalty,
                                trainer.weightDecayC,
                                trainer.gradClipMaxNorm,
                                trainer.policyLossWeight,
                                trainer.valueLossWeight,
                                trainer.momentumCoeff,
                                trainer.sqrtBatchScalingForLR,
                                trainer.lrWarmupSteps,
                                trainer.completedTrainSteps,
                                TrainingParameters.shared.arenaAutoIntervalSec,
                                TrainingParameters.shared.arenaPromoteThreshold,
                                TrainingParameters.shared.arenaGamesPerTournament
                            )
                        }
                        let policyStr: String
                        if let p = trainingSnap.rollingPolicyLoss {
                            policyStr = String(format: "%+.4f", p)
                        } else {
                            policyStr = "--"
                        }
                        let pLossWinStr: String
                        if let p = trainingSnap.rollingPolicyLossWin {
                            pLossWinStr = String(format: "%+.4f", p)
                        } else {
                            pLossWinStr = "--"
                        }
                        let pLossLossStr: String
                        if let p = trainingSnap.rollingPolicyLossLoss {
                            pLossLossStr = String(format: "%+.4f", p)
                        } else {
                            pLossLossStr = "--"
                        }
                        let valueStr: String
                        if let v = trainingSnap.rollingValueLoss {
                            valueStr = String(format: "%+.4f", v)
                        } else {
                            valueStr = "--"
                        }
                        let entropyStr = String(format: "%.4f", trainingSnap.rollingPolicyEntropy ?? 0)
                        let illegalPenaltyStr = String(format: "%.4f", trainingSnap.rollingIllegalMassPenalty ?? 0)
                        let gradNormStr: String
                        if let g = trainingSnap.rollingGradGlobalNorm {
                            gradNormStr = String(format: "%.3f", g)
                        } else {
                            gradNormStr = "--"
                        }
                        let vNormStr: String
                        if let vn = trainingSnap.rollingVelocityNorm {
                            vNormStr = String(format: "%.3f", vn)
                        } else {
                            vNormStr = "--"
                        }
                        let muStr = String(format: "%.3f", momentum)
                        let vMeanStr: String
                        if let vm = trainingSnap.rollingValueMean {
                            vMeanStr = String(format: "%+.4f", vm)
                        } else {
                            vMeanStr = "--"
                        }
                        let vAbsStr: String
                        if let va = trainingSnap.rollingValueAbsMean {
                            vAbsStr = String(format: "%.4f", va)
                        } else {
                            vAbsStr = "--"
                        }
                        // vBaseDelta = mean abs delta between trainer's
                        // current v(s) and the play-time-frozen vBaseline
                        // from the random-init champion. Higher = trainer
                        // is genuinely diverging from the champion.
                        let vBaseDeltaStr: String
                        if let vbd = trainingSnap.rollingVBaselineDelta {
                            vBaseDeltaStr = String(format: "%.4f", vbd)
                        } else {
                            vBaseDeltaStr = "--"
                        }
                        let h = Int(elapsedTarget) / 3600
                        let m = (Int(elapsedTarget) % 3600) / 60
                        let s = Int(elapsedTarget) % 60
                        let elapsedStr = String(format: "%d:%02d:%02d", h, m, s)
                        let spTau = String(format: "%.2f/%.2f/%.3f", spSched.startTau, spSched.floorTau, spSched.decayPerPly)
                        let arTau = String(format: "%.2f/%.2f/%.3f", arSched.startTau, arSched.floorTau, arSched.decayPerPly)
                        let pwNormStr: String
                        if let pwn = trainingSnap.rollingPolicyHeadWeightNorm {
                            if pwNormBaseline == nil {
                                pwNormBaseline = pwn
                            }
                            let delta = pwn - (pwNormBaseline ?? pwn)
                            pwNormStr = String(format: "%.5f(Δ%+.5f)", pwn, delta)
                        } else {
                            pwNormStr = "--"
                        }
                        let divSnap = spDiversityTracker.snapshot()
                        let divStr = divSnap.gamesInWindow > 0
                        ? String(format: "unique=%d/%d(%.0f%%) diverge=%.1f", divSnap.uniqueGames, divSnap.gamesInWindow, divSnap.uniquePercent, divSnap.avgDivergencePly)
                        : "n/a"
                        // Append a `·√b` marker when sqrt-batch scaling is on
                        // and a `·warmup(i/N)` marker while warmup is still
                        // active, so the reader can tell at a glance what
                        // per-step multipliers the optimizer is actually
                        // seeing. The displayed LR is always the base value.
                        var lrStr = String(format: "%.1e", lr) + (sqrtLR ? "·√b" : "")
                        if warmupSteps > 0 && completedSteps < warmupSteps {
                            lrStr += "·warmup(\(completedSteps)/\(warmupSteps))"
                        }
                        let spMsStr: String
                        if let sp = ratioSnap.selfPlayMsPerMove {
                            spMsStr = String(format: "%.2f", sp)
                        } else {
                            spMsStr = "--"
                        }
                        let trMsStr: String
                        if let tr = ratioSnap.trainingMsPerMove {
                            trMsStr = String(format: "%.2f", tr)
                        } else {
                            trMsStr = "--"
                        }
                        // moves/hour companions to prod/cons (which are
                        // pos/sec). Same rolling 60-s window — these are
                        // a pure unit conversion, exposed here so the
                        // [STATS] line and result.json carry the rate in
                        // the unit a human asks for ("how many moves per
                        // hour are we training on?").
                        let spMovesPerHour = ratioSnap.productionRate * 3600.0
                        let trainMovesPerHour = ratioSnap.consumptionRate * 3600.0
                        let ratioStr = String(format: "target=%.2f cur=%.2f prod=%.1f cons=%.1f spRate=%.0f/hr trainRate=%.0f/hr auto=%@ delay=%dms spDelay=%dms spMs=%@ trMs=%@ workers=%d",
                                              ratioSnap.targetRatio, ratioSnap.currentRatio,
                                              ratioSnap.productionRate, ratioSnap.consumptionRate,
                                              spMovesPerHour, trainMovesPerHour,
                                              ratioSnap.autoAdjust ? "on" : "off",
                                              ratioSnap.computedDelayMs, ratioSnap.computedSelfPlayDelayMs,
                                              spMsStr, trMsStr, ratioSnap.workerCount)
                        let outcomeStr = String(format: "wMate=%d bMate=%d stale=%d 50mv=%d 3fold=%d insuf=%d",
                                                parallelSnap.whiteCheckmates, parallelSnap.blackCheckmates,
                                                parallelSnap.stalemates, parallelSnap.fiftyMoveDraws,
                                                parallelSnap.threefoldRepetitionDraws, parallelSnap.insufficientMaterialDraws)
                        let cfgStr = "batch=\(sessionTrainingBatchSize) lr=\(lrStr) promote>=\(String(format: "%.2f", livePromoteThreshold)) arenaGames=\(liveTournamentGames) arenaAutoSec=\(Int(arenaAutoSec)) workers=\(workerN)"
                        let regStr = String(
                            format: "clip=%.1f decay=%.0e ent=%.1e illM=%.1e drawPen=%.3f pLossW=%.2f vLossW=%.2f μ=%.2f",
                            gradClip,
                            weightDec,
                            entropyCoeff,
                            illegalMassW,
                            drawPen,
                            policyW,
                            valueW,
                            momentum
                        )
                        // Average game length: lifetime and 10-min
                        // rolling window. `selfPlayPositions` counts
                        // every ply played, so dividing by the number
                        // of completed games gives the mean plies-per-
                        // game. Rolling avg tracks recent behavior;
                        // lifetime avg catches longer-term drift.
                        let lifetimeAvgLen: Double = parallelSnap.selfPlayGames > 0
                        ? Double(parallelSnap.selfPlayPositions) / Double(parallelSnap.selfPlayGames)
                        : 0
                        let rollingAvgLen: Double = parallelSnap.recentGames > 0
                        ? Double(parallelSnap.recentMoves) / Double(parallelSnap.recentGames)
                        : 0
                        let p50Str: String
                        let p95Str: String
                        if let p50 = parallelSnap.gameLenP50 {
                            p50Str = String(p50)
                        } else {
                            p50Str = "--"
                        }
                        if let p95 = parallelSnap.gameLenP95 {
                            p95Str = String(p95)
                        } else {
                            p95Str = "--"
                        }
                        let gameLenStr = String(format: "avgLen=%.1f rollingAvgLen=%.1f p50=\(p50Str) p95=\(p95Str)", lifetimeAvgLen, rollingAvgLen)
                        // New encoding/gradient-health signals.
                        let playedProbStr: String
                        if let pm = trainingSnap.rollingPlayedMoveProb {
                            playedProbStr = String(format: "%.4f", pm)
                        } else {
                            playedProbStr = "--"
                        }
                        // Advantage-sign-conditional played-move
                        // probabilities. Under a working training loop
                        // `posAdv` rises above `1/policySize` and
                        // `negAdv` falls — divergence between the two
                        // is the real action-index/direction-of-
                        // learning signal (the unconditional
                        // `playedMoveProb` is ambiguous under adv-
                        // normalized loss). The `skip=K/N` suffix
                        // reports how many steps in the rolling window
                        // dropped out because their batch had no
                        // positions on that side of A — that's the
                        // effective sample count behind the mean.
                        let condWindowSize = trainingSnap.rollingPlayedMoveCondWindowSize
                        let playedProbPosStr: String
                        if let pm = trainingSnap.rollingPlayedMoveProbPosAdv {
                            playedProbPosStr = String(
                                format: "%.4f(skip=%d/%d)",
                                pm,
                                trainingSnap.rollingPlayedMoveProbPosAdvSkipped,
                                condWindowSize
                            )
                        } else {
                            playedProbPosStr = String(
                                format: "--(skip=%d/%d)",
                                trainingSnap.rollingPlayedMoveProbPosAdvSkipped,
                                condWindowSize
                            )
                        }
                        let playedProbNegStr: String
                        if let pm = trainingSnap.rollingPlayedMoveProbNegAdv {
                            playedProbNegStr = String(
                                format: "%.4f(skip=%d/%d)",
                                pm,
                                trainingSnap.rollingPlayedMoveProbNegAdvSkipped,
                                condWindowSize
                            )
                        } else {
                            playedProbNegStr = String(
                                format: "--(skip=%d/%d)",
                                trainingSnap.rollingPlayedMoveProbNegAdvSkipped,
                                condWindowSize
                            )
                        }
                        let pLogitMaxStr: String
                        if let pm = trainingSnap.rollingPolicyLogitAbsMax {
                            pLogitMaxStr = String(format: "%.3f", pm)
                        } else {
                            pLogitMaxStr = "--"
                        }
                        // Advantage distribution summary. Lots of
                        // fields but they go in one parenthesized
                        // block in the line so grep for
                        // "adv=(" when analyzing.
                        func advFmt(_ d: Double?) -> String {
                            guard let d else { return "--" }
                            return String(format: "%+.4f", d)
                        }
                        func advFracFmt(_ d: Double?) -> String {
                            guard let d else { return "--" }
                            return String(format: "%.2f", d)
                        }
                        let advStr = "mean=\(advFmt(trainingSnap.rollingAdvMean)) std=\(advFmt(trainingSnap.rollingAdvStd)) min=\(advFmt(trainingSnap.rollingAdvMin)) max=\(advFmt(trainingSnap.rollingAdvMax)) frac+=\(advFracFmt(trainingSnap.rollingAdvFracPositive)) fracSmall=\(advFracFmt(trainingSnap.rollingAdvFracSmall)) p05=\(advFmt(trainingSnap.advantageP05)) p50=\(advFmt(trainingSnap.advantageP50)) p95=\(advFmt(trainingSnap.advantageP95))"
                        let legalMassStr: String
                        let top1LegalStr: String
                        let pEntLegalStr: String
                        if let lm = legalMassOverride {
                            legalMassStr = String(format: "%.4f", lm.legalMass)
                            top1LegalStr = String(format: "%.2f", lm.top1LegalFraction)
                            pEntLegalStr = String(format: "%.4f", lm.legalEntropy)
                        } else {
                            legalMassStr = "--"
                            top1LegalStr = "--"
                            pEntLegalStr = "--"
                        }
                        // Surface the most-recent batch unique-position
                        // ratio so it scrolls past the eye in the same
                        // line as pEnt / gNorm. Reads NaN until the
                        // first stats-collection batch lands.
                        let bufUniqStr: String
                        let bufUniqPct = trainer.lastBatchStatsUniquePct
                        if !bufUniqPct.isNaN {
                            bufUniqStr = String(format: "%.4f", bufUniqPct)
                        } else {
                            bufUniqStr = "--"
                        }
                        // Per-step timing means over the rolling timing
                        // window. Splits `recentStepMs` into prep/gpu/
                        // read/queueWait/step so a slowdown can be
                        // attributed to one component (CPU prep, GPU
                        // compute, readback marshalling, executor
                        // queue backlog, or unaccounted overhead).
                        let timingStr: String
                        if let stepMs = trainingSnap.recentStepMs {
                            let prep = trainingSnap.recentDataPrepMs ?? 0
                            let gpu = trainingSnap.recentGpuRunMs ?? 0
                            let read = trainingSnap.recentReadbackMs ?? 0
                            let wait = trainingSnap.recentQueueWaitMs ?? 0
                            timingStr = String(
                                format: "step=%.1f gpu=%.1f prep=%.2f read=%.2f wait=%.2f n=%d",
                                stepMs, gpu, prep, read, wait, trainingSnap.recentTimingSamples
                            )
                        } else {
                            timingStr = "n=0"
                        }

                        // Process resident memory + delta since previous
                        // [STATS] tick. The delta lets a reader see
                        // sustained leak rate (e.g. "+8 MB / minute")
                        // at a glance rather than having to diff two
                        // log lines mentally.
                        let rssBytes = DiagSampler.currentResidentBytes()
                        let drss: Int64 = prevRssBytes == 0
                            ? 0
                            : Int64(rssBytes) - Int64(prevRssBytes)
                        prevRssBytes = rssBytes
                        let memStr = String(
                            format: "rss=%.2fGB drss=%+.1fMB",
                            Double(rssBytes) / 1024.0 / 1024.0 / 1024.0,
                            Double(drss) / 1024.0 / 1024.0
                        )

                        // VM region count + IOAccelerator-tagged
                        // subset. Cost ~1-3 ms per call; safe at this
                        // 60 s cadence. Each AGX-mapped GPU buffer
                        // shows up as one IOAccelerator region, so
                        // `vmAccel` growing in step with `trMs` is
                        // direct evidence that the per-`commit` cost
                        // is climbing in the kernel's residency walk.
                        let vm = DiagSampler.currentVMRegionCount()
                        let dvmTotal = prevVmTotal == 0
                            ? 0
                            : Int64(vm.total) - Int64(prevVmTotal)
                        let dvmAccel = prevVmIoAccel == 0
                            ? 0
                            : Int64(vm.ioAccelerator) - Int64(prevVmIoAccel)
                        prevVmTotal = vm.total
                        prevVmIoAccel = vm.ioAccelerator
                        let vmStr = String(
                            format: "total=%u dtotal=%+d ioAccel=%u dioAccel=%+d",
                            vm.total, dvmTotal, vm.ioAccelerator, dvmAccel
                        )

                        // Trainer feed-cache size. Stable at 1 means
                        // the trainer is calling MPSGraph with one
                        // feed shape (so any pipeline accumulation in
                        // MPSGraph is not because we're varying our
                        // inputs).
                        let shapesStr = "feedCache=\(trainer.feedCacheCount)"

                        let line = "[STATS] elapsed=\(elapsedStr) steps=\(trainingSnap.stats.steps) spGames=\(parallelSnap.selfPlayGames) spMoves=\(parallelSnap.selfPlayPositions) \(gameLenStr) buffer=\(bufCount)/\(bufCap) pLoss=\(policyStr) pLossWin=\(pLossWinStr) pLossLoss=\(pLossLossStr) vLoss=\(valueStr) pEnt=\(entropyStr) pIllM=\(illegalPenaltyStr) gNorm=\(gradNormStr) vNorm=\(vNormStr) μ=\(muStr) pwNorm=\(pwNormStr) pLogitAbsMax=\(pLogitMaxStr) playedMoveProb=\(playedProbStr) playedMoveProbPosAdv=\(playedProbPosStr) playedMoveProbNegAdv=\(playedProbNegStr) legalMass=\(legalMassStr) top1Legal=\(top1LegalStr) pEntLegal=\(pEntLegalStr) vMean=\(vMeanStr) vAbs=\(vAbsStr) vBaseDelta=\(vBaseDeltaStr) adv=(\(advStr)) sp.tau=\(spTau) ar.tau=\(arTau) diversity=\(divStr) ratio=(\(ratioStr)) outcomes=(\(outcomeStr)) bufUniq=\(bufUniqStr) \(cfgStr) reg=(\(regStr)) timing=(\(timingStr)) mem=(\(memStr)) vm=(\(vmStr)) shapes=(\(shapesStr)) build=\(BuildInfo.buildNumber) trainer=\(trainerID) champion=\(championID)"
                        SessionLogger.shared.log(line)

                        // CLI `--output` capture: one StatsLine per
                        // `[STATS]` log emit. All values come from the
                        // same thread-safe snapshots the log line just
                        // used, so the JSON and the log stay perfectly
                        // consistent. No-op when `recorder == nil`.
                        if let recorder {
                            let entry = CliTrainingRecorder.StatsLine(
                                elapsedSec: elapsedTarget,
                                steps: trainingSnap.stats.steps,
                                selfPlayGames: parallelSnap.selfPlayGames,
                                positionsTrained: parallelSnap.selfPlayPositions,
                                avgLen: lifetimeAvgLen,
                                rollingAvgLen: rollingAvgLen,
                                gameLenP50: parallelSnap.gameLenP50,
                                gameLenP95: parallelSnap.gameLenP95,
                                bufferCount: bufCount,
                                bufferCapacity: bufCap,
                                policyLoss: trainingSnap.rollingPolicyLoss,
                                valueLoss: trainingSnap.rollingValueLoss,
                                policyEntropy: trainingSnap.rollingPolicyEntropy,
                                policyIllegalMassPenalty: trainingSnap.rollingIllegalMassPenalty,
                                gradGlobalNorm: trainingSnap.rollingGradGlobalNorm,
                                policyHeadWeightNorm: trainingSnap.rollingPolicyHeadWeightNorm,
                                policyLogitAbsMax: trainingSnap.rollingPolicyLogitAbsMax,
                                playedMoveProb: trainingSnap.rollingPlayedMoveProb,
                                playedMoveProbPosAdv: trainingSnap.rollingPlayedMoveProbPosAdv,
                                playedMoveProbPosAdvSkipped: trainingSnap.rollingPlayedMoveProbPosAdvSkipped,
                                playedMoveProbNegAdv: trainingSnap.rollingPlayedMoveProbNegAdv,
                                playedMoveProbNegAdvSkipped: trainingSnap.rollingPlayedMoveProbNegAdvSkipped,
                                playedMoveCondWindowSize: trainingSnap.rollingPlayedMoveCondWindowSize,
                                legalMass: legalMassOverride.map { Double($0.legalMass) },
                                top1LegalFraction: legalMassOverride.map { Double($0.top1LegalFraction) },
                                legalEntropy: legalMassOverride.map { Double($0.legalEntropy) },
                                policyLossWin: trainingSnap.rollingPolicyLossWin,
                                policyLossLoss: trainingSnap.rollingPolicyLossLoss,
                                batchStats: trainer.lastBatchStatsSummary.map { s in
                                    CliTrainingRecorder.BatchStatsSnapshot(
                                        step: s.step,
                                        batchSize: s.batchSize,
                                        uniqueCount: s.uniqueCount,
                                        uniquePct: s.uniquePct,
                                        dupMax: s.dupMax,
                                        dupDistribution: Dictionary(uniqueKeysWithValues:
                                            s.dupDistribution.map { (String($0.key), $0.value) }
                                        ),
                                        phaseByPlyHistogram: s.phaseByPlyHistogram,
                                        phaseByMaterialHistogram: s.phaseByMaterialHistogram,
                                        gameLengthHistogram: s.gameLengthHistogram,
                                        samplingTauHistogram: s.samplingTauHistogram,
                                        workerIdHistogram: s.workerIdHistogram,
                                        outcomeHistogram: s.outcomeHistogram,
                                        phaseByPlyXOutcomeHistogram: s.phaseByPlyXOutcomeHistogram,
                                        bufferUniquePositions: s.bufferUniquePositions,
                                        bufferStoredCount: s.bufferStoredCount
                                    )
                                },
                                valueMean: trainingSnap.rollingValueMean,
                                valueAbsMean: trainingSnap.rollingValueAbsMean,
                                vBaselineDelta: trainingSnap.rollingVBaselineDelta,
                                advMean: trainingSnap.rollingAdvMean,
                                advStd: trainingSnap.rollingAdvStd,
                                advMin: trainingSnap.rollingAdvMin,
                                advMax: trainingSnap.rollingAdvMax,
                                advFracPositive: trainingSnap.rollingAdvFracPositive,
                                advFracSmall: trainingSnap.rollingAdvFracSmall,
                                advP05: trainingSnap.advantageP05,
                                advP50: trainingSnap.advantageP50,
                                advP95: trainingSnap.advantageP95,
                                spStartTau: Double(spSched.startTau),
                                spFloorTau: Double(spSched.floorTau),
                                spDecayPerPly: Double(spSched.decayPerPly),
                                arStartTau: Double(arSched.startTau),
                                arFloorTau: Double(arSched.floorTau),
                                arDecayPerPly: Double(arSched.decayPerPly),
                                diversityUniqueGames: divSnap.uniqueGames,
                                diversityGamesInWindow: divSnap.gamesInWindow,
                                diversityUniquePercent: divSnap.uniquePercent,
                                diversityAvgDivergencePly: divSnap.avgDivergencePly,
                                ratioTarget: ratioSnap.targetRatio,
                                ratioCurrent: ratioSnap.currentRatio,
                                ratioProductionRate: ratioSnap.productionRate,
                                ratioConsumptionRate: ratioSnap.consumptionRate,
                                selfPlayMovesPerHour: spMovesPerHour,
                                trainingMovesPerHour: trainMovesPerHour,
                                ratioAutoAdjust: ratioSnap.autoAdjust,
                                ratioComputedDelayMs: ratioSnap.computedDelayMs,
                                whiteCheckmates: parallelSnap.whiteCheckmates,
                                blackCheckmates: parallelSnap.blackCheckmates,
                                stalemates: parallelSnap.stalemates,
                                fiftyMoveDraws: parallelSnap.fiftyMoveDraws,
                                threefoldRepetitionDraws: parallelSnap.threefoldRepetitionDraws,
                                insufficientMaterialDraws: parallelSnap.insufficientMaterialDraws,
                                batchSize: sessionTrainingBatchSize,
                                learningRate: Double(lr),
                                promoteThreshold: sessionPromoteThreshold,
                                arenaGames: sessionTournamentGames,
                                workerCount: workerN,
                                gradClipMaxNorm: Double(gradClip),
                                weightDecayC: Double(weightDec),
                                entropyRegularizationCoeff: Double(entropyCoeff),
                                drawPenalty: Double(drawPen),
                                policyLossWeight: Double(policyW),
                                valueLossWeight: Double(valueW),
                                buildNumber: BuildInfo.buildNumber,
                                trainerID: trainerID,
                                championID: championID
                            )
                            recorder.appendStats(entry)
                        }

                        // Policy-entropy alarm: fires whenever the
                        // rolling entropy (computed over the training
                        // stats window, same as logged above) is below
                        // the threshold. Co-located with the [STATS]
                        // emit so the cadence matches — the log
                        // adjacent lines always tell a consistent
                        // story. Skipped if entropy isn't yet
                        // available (training hasn't started).
                        if let entropy = trainingSnap.rollingPolicyEntropy,
                           entropy < TrainingAlarmController.policyEntropyAlarmThreshold {
                            SessionLogger.shared.log(
                                "[ALARM] policy entropy \(String(format: "%.4f", entropy)) < \(String(format: "%.2f", TrainingAlarmController.policyEntropyAlarmThreshold)) — policy may be collapsing (steps=\(trainingSnap.stats.steps))"
                            )
                        }
                    }

                    // Cache the most recent legalMass probe result so
                    // we can include it in back-to-back per-step emits
                    // without paying the ~5-20 ms forward-pass cost on
                    // every single step. Refreshed every
                    // `legalMassBootstrapStride` steps during bootstrap
                    // and every time-based emit afterward.
                    var lastLegalMass: ChessTrainer.LegalMassSnapshot? = nil
                    var lastEmittedStep: Int = -1
                    var bootstrapDone = false

                    // Bootstrap phase: poll at short interval, emit one
                    // line per new training step until
                    // bootstrapSteps steps have been logged.
                    while !Task.isCancelled && !bootstrapDone {
                        let trainingSnap = await box.snapshot()
                        let steps = trainingSnap.stats.steps
                        if steps > lastEmittedStep && steps > 0 {
                            // Refresh legalMass on a stride so
                            // back-to-back per-step emits share the
                            // most recent probe.
                            if steps == 1 || (steps - max(0, lastEmittedStep)) >= legalMassBootstrapStride {
                                if buffer.count >= legalMassSampleSize, let probeNet = probeInferenceForProbes {
                                    do {
                                        lastLegalMass = try await trainer.legalMassSnapshot(
                                            replayBuffer: buffer,
                                            sampleSize: legalMassSampleSize,
                                            inferenceNetwork: probeNet
                                        )
                                        // Mirror to @State so the
                                        // chart-sample heartbeat can
                                        // render the legal-entropy
                                        // trace at its own cadence.
                                        let snap = lastLegalMass
                                        await MainActor.run {
                                            self.realLastLegalMassSnapshot = snap
                                        }
                                    } catch {
                                        lastLegalMass = nil
                                        SessionLogger.shared.log(
                                            "[STATS-ERR] legalMassSnapshot failed at step \(steps): \(error.localizedDescription)"
                                        )
                                    }
                                }
                            }
                            let elapsed = Date().timeIntervalSince(sessionStart)
                            await logOne(elapsedTarget: elapsed, legalMassOverride: lastLegalMass)
                            lastEmittedStep = steps
                            if steps >= bootstrapSteps {
                                bootstrapDone = true
                                break
                            }
                        }
                        // Short poll — a training step at typical
                        // throughput completes in 50-200 ms, and we
                        // want the per-step cadence to track closely.
                        do {
                            try await Task.sleep(for: .milliseconds(50))
                        } catch {
                            return
                        }
                    }
                    if Task.isCancelled { return }

                    // Steady-state: one emit every `steadyInterval`
                    // seconds. Each emit refreshes the legalMass probe
                    // too.
                    while !Task.isCancelled {
                        do {
                            try await Task.sleep(for: .seconds(steadyInterval))
                        } catch {
                            return
                        }
                        if Task.isCancelled { return }
                        if buffer.count >= legalMassSampleSize, let probeNet = probeInferenceForProbes {
                            let trainingSnap = await box.snapshot()
                            let steps = trainingSnap.stats.steps
                            do {
                                lastLegalMass = try await trainer.legalMassSnapshot(
                                    replayBuffer: buffer,
                                    sampleSize: legalMassSampleSize,
                                    inferenceNetwork: probeNet
                                )
                                // Mirror to @State so the chart-sample
                                // heartbeat sees a fresh legal-entropy
                                // value at the steady-state cadence.
                                let snap = lastLegalMass
                                await MainActor.run {
                                    self.realLastLegalMassSnapshot = snap
                                }
                            } catch {
                                lastLegalMass = nil
                                SessionLogger.shared.log(
                                    "[STATS-ERR] legalMassSnapshot failed at step \(steps): \(error.localizedDescription)"
                                )
                            }
                        }
                        let elapsed = Date().timeIntervalSince(sessionStart)
                        await logOne(elapsedTarget: elapsed, legalMassOverride: lastLegalMass)
                    }
                }

                // Headless CLI deadline. Armed whenever the run
                // started with `--train` AND `training_time_limit`
                // is set — the time limit is the primary driver of
                // output generation and exit, regardless of
                // whether `--output <file>` was supplied. When
                // `--output` is present the JSON snapshot is
                // written to that file (overwriting); when absent
                // the snapshot goes to stdout so the user can
                // redirect or pipe it. Outside of `--train` mode
                // (interactive use) the deadline is ignored — the
                // user is driving the UI and an unexpected
                // termination would be hostile.
                //
                // On wake-up: log the deadline event, emit the
                // recorder's snapshot to the configured sink, then
                // call `Darwin._exit(0)`. The crucial detail is
                // `_exit` vs `exit`: `exit` runs C++ atexit
                // handlers (including CoreAnalytics' exit barrier)
                // while the MPS self-play worker is still mid-
                // `graph.run` on its dedicated serial dispatch
                // queue — the handler tears down global state
                // that `MPSGraphOSLog` still reads, producing
                // EXC_BAD_ACCESS at 0x8 inside the MPSGraph
                // executable run path. `_exit` terminates the
                // process immediately without running atexit or
                // stdio cleanup, which sidesteps the race. It's
                // safe because:
                //   - snapshot file writes use `Data.write(atomic:)`
                //     which fsyncs before rename,
                //   - stdout writes go through `FileHandle`
                //     which is an unbuffered syscall (no libc
                //     stdio buffer to flush),
                //   - session log writes have already been
                //     flushed by SessionLogger before this point.
                if isAutoTrainRun, let recorder, let deadlineSec = cliTrainingTimeLimitSec, deadlineSec > 0 {
                    group.addTask(priority: .utility) {
                        do {
                            try await Task.sleep(for: .seconds(deadlineSec))
                        } catch {
                            // Cancelled before the deadline hit —
                            // session ended for another reason.
                            return
                        }
                        if Task.isCancelled { return }
                        let elapsed = Date().timeIntervalSince(runStart)
                        let destDescription = outputURL?.path ?? "<stdout>"
                        SessionLogger.shared.log(
                            "[APP] --train: training_time_limit=\(deadlineSec)s reached at elapsed=\(String(format: "%.1f", elapsed))s; writing snapshot to \(destDescription)"
                        )
                        recorder.setTerminationReason(.timerExpired)
                        let counts = recorder.countsSnapshot()
                        do {
                            if let outputURL {
                                try recorder.writeJSON(to: outputURL, totalTrainingSeconds: elapsed)
                            } else {
                                try recorder.writeJSONToStdout(totalTrainingSeconds: elapsed)
                            }
                            SessionLogger.shared.log(
                                "[APP] --train: wrote snapshot to \(destDescription) (arenas=\(counts.arenas), stats=\(counts.stats), probes=\(counts.probes))"
                            )
                        } catch {
                            SessionLogger.shared.log(
                                "[APP] --train: snapshot write FAILED for \(destDescription): \(error.localizedDescription)"
                            )
                        }
                        SessionLogger.shared.log("[APP] --train: exiting process after snapshot")
                        Darwin._exit(0)
                    }
                }

                // Legal-mass collapse detector. Runs in every session
                // (interactive or --train), probes at 60 s cadence
                // with a 120 s grace period measured from the first
                // observed SGD step (i.e. earliest possible detection
                // is training_start + 120 s), flags the collapse
                // banner once the network has shown
                // `illegalMass > 0.99` for `collapseProbesToAbort`
                // consecutive probes. In interactive mode we stop at
                // the banner — the user sees it and decides what to
                // do. In --train mode we ALSO write the snapshot with
                // `termination_reason = legal_mass_collapse` and exit
                // the process (status 0 — a clean early abort, not a
                // crash; scripting distinguishes via the JSON field
                // rather than exit code per the user's spec).
                // Local "session start" for the collapse detector.
                // The --train timer task uses its own `runStart`;
                // both are set at roughly the same moment in the
                // task-group setup (drift < 1 ms) and the two are
                // only used for "elapsed since session start"
                // display in logs/alarms, so a separate declaration
                // is intentional — it keeps the detector
                // self-contained and doesn't require plumbing the
                // timer task's runStart into this closure.
                let collapseRunStart = Date()
                let collapseRecorder: CliTrainingRecorder? = (isAutoTrainRun ? recorder : nil)
                let collapseOutputURL: URL? = outputURL
                // Configuration snapshot taken at task start. All three
                // are user-tunable via parameters JSON / @AppStorage; we
                // capture once so the running detector has stable behavior
                // even if the user edits the values mid-session (changes
                // take effect on the next training run).
                let collapseIllegalMassThreshold = TrainingParameters.shared.legalMassCollapseThreshold
                let collapseGracePeriodSec = TrainingParameters.shared.legalMassCollapseGraceSeconds
                let collapseNoImprovementProbeCount = max(1, TrainingParameters.shared.legalMassCollapseNoImprovementProbes)
                group.addTask(priority: .utility) {
                    [trainer, buffer, box, probeInferenceForProbes] in
                    let probeIntervalSec: UInt64 = 60
                    let sampleSize = 128
                    let gracePeriodSec: TimeInterval = collapseGracePeriodSec
                    let illegalMassThreshold: Double = collapseIllegalMassThreshold
                    let noImprovementProbeCount = collapseNoImprovementProbeCount
                    // Sliding window of recent legal-mass readings,
                    // newest at end. Trip condition: window is full,
                    // every reading's illegalMass is above threshold,
                    // AND the newest legal_mass is no better than the
                    // oldest (no upward improvement across the window).
                    // This catches *stuck* collapse — slow climbs out
                    // of near-uniform won't fire even if absolute
                    // legal mass is still below threshold for a while.
                    var legalMassWindow: [Double] = []
                    var aborted = false
                    // Anchor for the grace countdown. Lazily set the
                    // first time we observe at least one completed SGD
                    // step. The replay buffer has to fill enough for
                    // training to begin (minutes in practice), and
                    // before that first step lands, every probe on the
                    // fresh random-init network will naturally see
                    // illegalMass ≈ 0.994 — firing the alarm during
                    // that window is a false positive. Grace is 120 s
                    // measured from TRAINING start, not session start.
                    var trainingStartAt: Date? = nil
                    while !Task.isCancelled && !aborted {
                        do {
                            try await Task.sleep(for: .seconds(probeIntervalSec))
                        } catch {
                            return
                        }
                        if Task.isCancelled { return }
                        // Training-start gate. If no SGD steps have
                        // landed yet, reset the anchor and skip — the
                        // network is still at random init so no
                        // learning-progress signal exists. Once steps
                        // > 0, stamp the anchor on the first qualifying
                        // iteration and measure grace from there.
                        let trainingSteps = await box.snapshot().stats.steps
                        guard trainingSteps > 0 else {
                            trainingStartAt = nil
                            continue
                        }
                        if trainingStartAt == nil {
                            trainingStartAt = Date()
                        }
                        guard let startAt = trainingStartAt else { continue }
                        let trainingElapsed = Date().timeIntervalSince(startAt)
                        if trainingElapsed < gracePeriodSec { continue }
                        // Session-elapsed for snapshot-write / log
                        // consistency with the timer task, which also
                        // reports `totalTrainingSeconds` as
                        // session-elapsed rather than training-elapsed.
                        let elapsed = Date().timeIntervalSince(collapseRunStart)
                        // Skip the probe if the inference network isn't
                        // available — the pollution-prone trainer-network
                        // fallback would silently corrupt BN running stats
                        // and defeat the reason we switched detectors off
                        // the trainer network in the first place.
                        guard let probeNet = probeInferenceForProbes else { continue }

                        let snap: ChessTrainer.LegalMassSnapshot?
                        do {
                            snap = try await trainer.legalMassSnapshot(
                                replayBuffer: buffer,
                                sampleSize: sampleSize,
                                inferenceNetwork: probeNet
                            )
                        } catch {
                            SessionLogger.shared.log(
                                "[STATS-ERR] collapse-detector legalMassSnapshot failed: \(error.localizedDescription)"
                            )
                            continue
                        }
                        guard let snap else { continue }
                        let illegalMass = 1.0 - snap.legalMass
                        legalMassWindow.append(snap.legalMass)
                        if legalMassWindow.count > noImprovementProbeCount {
                            legalMassWindow.removeFirst()
                        }
                        // Window-full tripwire: we need
                        // `noImprovementProbeCount` samples before
                        // declaring "no improvement"; until then we
                        // log the probe and continue.
                        let windowFull = legalMassWindow.count >= noImprovementProbeCount
                        let allAboveThreshold: Bool = {
                            guard windowFull else { return false }
                            for legalMass in legalMassWindow where (1.0 - legalMass) <= illegalMassThreshold {
                                return false
                            }
                            return true
                        }()
                        let noImprovement: Bool = {
                            guard windowFull,
                                  let oldest = legalMassWindow.first,
                                  let newest = legalMassWindow.last else {
                                return false
                            }
                            return newest <= oldest
                        }()
                        let probeMatches = allAboveThreshold && noImprovement
                        if probeMatches {
                            SessionLogger.shared.log(
                                String(format: "[ALARM] legal-mass collapse window: %d/%d probes illegalMass>%.4f, newest legalMass=%.5f ≤ oldest=%.5f",
                                    legalMassWindow.count, noImprovementProbeCount,
                                    illegalMassThreshold,
                                    legalMassWindow.last ?? 0, legalMassWindow.first ?? 0)
                            )
                            await MainActor.run {
                                self.trainingAlarm?.raise(
                                    severity: .critical,
                                    title: "Policy Collapse (legal mass)",
                                    detail: String(format: "illegalMass>%.4f for %d probes with no improvement (legal_mass %.5f→%.5f)",
                                        illegalMassThreshold, legalMassWindow.count,
                                        legalMassWindow.first ?? 0, legalMassWindow.last ?? 0)
                                )
                            }
                        } else {
                            SessionLogger.shared.log(
                                String(format: "[ALARM] legal-mass probe ok: legalMass=%.5f illegalMass=%.4f window=%d/%d",
                                    snap.legalMass, illegalMass,
                                    legalMassWindow.count, noImprovementProbeCount)
                            )
                        }

                        if probeMatches {
                            aborted = true
                            SessionLogger.shared.log(
                                String(format: "[ALARM] legal-mass collapse confirmed: %d probes above %.4f with no improvement (legal_mass %.5f→%.5f) — aborting",
                                    legalMassWindow.count, illegalMassThreshold,
                                    legalMassWindow.first ?? 0, legalMassWindow.last ?? 0)
                            )
                            if let rec = collapseRecorder {
                                let destDescription = collapseOutputURL?.path ?? "<stdout>"
                                SessionLogger.shared.log(
                                    "[APP] --train: legal-mass collapse abort at elapsed=\(String(format: "%.1f", elapsed))s; writing snapshot to \(destDescription)"
                                )
                                rec.setTerminationReason(.legalMassCollapse)
                                let counts = rec.countsSnapshot()
                                do {
                                    if let url = collapseOutputURL {
                                        try rec.writeJSON(to: url, totalTrainingSeconds: elapsed)
                                    } else {
                                        try rec.writeJSONToStdout(totalTrainingSeconds: elapsed)
                                    }
                                    SessionLogger.shared.log(
                                        "[APP] --train: wrote snapshot to \(destDescription) (arenas=\(counts.arenas), stats=\(counts.stats), probes=\(counts.probes))"
                                    )
                                } catch {
                                    SessionLogger.shared.log(
                                        "[APP] --train: snapshot write FAILED for \(destDescription): \(error.localizedDescription)"
                                    )
                                }
                                SessionLogger.shared.log("[APP] --train: exiting process after collapse-abort snapshot")
                                Darwin._exit(0)
                            }
                            // Interactive session: alarm already
                            // raised, loop stays exited so we stop
                            // probing (no value continuing once a
                            // confirmed collapse has been flagged).
                        }
                    }
                }

                // Wait for all four tasks to complete (only happens
                // on cancellation since each loops forever).
                for await _ in group { }
            }

            await MainActor.run {
                trainingAlarm?.clear()
                realTraining = false
                realTrainingTask = nil
                isArenaRunning = false
                arenaActiveFlag = nil
                arenaTriggerBox = nil
                arenaOverrideBox = nil
                parallelWorkerStatsBox = nil
                parallelStats = nil
                chartCoordinator?.setDiversityHistogramBars([])
                chartCoordinator?.arenaChartEvents = []
                chartCoordinator?.cancelActiveArena()
                workerCountBox = nil
                samplingScheduleBox = nil
                activeSelfPlayGate = nil
                activeTrainingGate = nil
                checkpoint?.currentSessionID = nil
                checkpoint?.currentSessionStart = nil
                replayRatioController = nil
                replayRatioSnapshot = nil
                effectiveReplayRatioTarget = nil
                lastReplayRatioCompensatorAt = nil
                cliRecorder = nil
                EarlyStopCoordinator.shared.earlyStopHandler = nil
            }
        }
    }

    func stopRealTraining() {
        realTrainingTask?.cancel()
        realTrainingTask = nil
        trainingAlarm?.clear()
        // Close the in-progress training segment so cumulative wall-time
        // totals exclude post-Stop idle. If saving immediately after,
        // buildCurrentSessionState will see the segment already closed
        // and won't double-count.
        checkpoint?.closeActiveTrainingSegment(reason: "stop")
        // Disarm the periodic-save scheduler so a Stop-then-Start
        // doesn't fire an immediate save on Start. The next Start
        // constructs a fresh controller with a fresh 4-hour
        // deadline.
        periodicSaveController?.disarm()
        periodicSaveController = nil
        periodicSaveLastPollAt = nil
        periodicSaveInFlight = false
    }

}
