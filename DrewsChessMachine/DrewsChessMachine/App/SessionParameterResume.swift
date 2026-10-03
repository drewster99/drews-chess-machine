import Foundation

/// Applies `TrainingParameterResolution` to the live settings while a session
/// resumes, and logs every decision.
///
/// For each key it writes the resolved value onto `TrainingParameters` with
/// the persistence that value deserves:
/// - a value the checkpoint carried is restored through
///   `restoreFromSession` (saved to app settings, as resume always has;
///   held for the run only if it lies outside today's declared range);
/// - a declared pre-feature value is held for this run only — it describes
///   the old run, not the user's preference;
/// - an operational knob or a not-exact training parameter keeps the live
///   setting untouched.
///
/// Each call emits exactly one `[RESUME-DIFF]` line (saved, applied, current,
/// reason), so a resume can never change a parameter silently. Training
/// parameters with no recorded value are collected in `notExactParameterIDs`
/// for the resume's NOT EXACT summary. The trainer is then configured from
/// the resulting settings through the same `TrainerHyperparameters` path a
/// fresh start uses.
@MainActor
final class SessionParameterResume {
    let parameters: TrainingParameters
    private let log: (String) -> Void
    private(set) var notExactParameterIDs: [String] = []

    init(parameters: TrainingParameters, log: @escaping (String) -> Void) {
        self.parameters = parameters
        self.log = log
    }

    /// Resolve and apply one key held on the settings as its own value type.
    @discardableResult
    func restore<K: TrainingParameterKey>(
        _ key: K.Type,
        saved: K.Value?,
        into keyPath: ReferenceWritableKeyPath<TrainingParameters, K.Value>
    ) -> K.Value {
        restore(
            K.self,
            saved: saved,
            current: parameters[keyPath: keyPath],
            describe: { "\($0)" },
            write: { value, source in
                switch source {
                case .session:
                    parameters.restoreFromSession(K.self, value, into: keyPath)
                case .preFeature:
                    parameters.holdForThisRun(K.self, value, into: keyPath)
                case .currentSetting, .notExact:
                    break
                }
            }
        )
    }

    /// Resolve and apply one `Double` key whose session copy is a `Float`
    /// (the trainer's hyperparameters). The saved float is widened through
    /// its shortest decimal text, so a saved 0.1 is restored as 0.1.
    @discardableResult
    func restore<K: TrainingParameterKey>(
        _ key: K.Type,
        savedFloat: Float?,
        into keyPath: ReferenceWritableKeyPath<TrainingParameters, Double>
    ) -> Double where K.Value == Double {
        restore(K.self, saved: savedFloat.map(TrainingParameters.doubleFromSavedFloat), into: keyPath)
    }

    /// Resolve one key and hand the decision to `write`, for a key the
    /// settings expose as a richer type than its stored value (an enum over
    /// an `Int` raw value). `describe` renders a value for the log line.
    /// `write` is called for every source; it decides how to apply each
    /// (`holdForThisRun` for `.preFeature`, nothing for the live-setting
    /// sources).
    @discardableResult
    func restore<K: TrainingParameterKey>(
        _ key: K.Type,
        saved: K.Value?,
        current: K.Value,
        describe: (K.Value) -> String,
        write: (K.Value, TrainingParameterResolution.Source) -> Void
    ) -> K.Value {
        let resolved = TrainingParameterResolution.resolve(K.self, saved: saved, current: current)
        log(TrainingParameterResolution.diffLine(K.self, resolved, describe: describe))
        if resolved.source == .notExact {
            notExactParameterIDs.append(K.id)
        }
        write(resolved.applied, resolved.source)
        return resolved.applied
    }

    /// A saved value that was unusable and that the user, reviewing the
    /// session at load, chose to replace with the current setting. Nothing is
    /// written; the line records the replacement.
    func logReplacedAtLoad<K: TrainingParameterKey>(
        _ key: K.Type,
        savedDescription: String,
        currentDescription: String,
        why: String
    ) {
        log("[RESUME-DIFF] \(K.id): saved=\(savedDescription) applied=\(currentDescription) current=\(currentDescription) "
            + "(saved value \(why); replaced with the current setting, accepted by the user at load)")
    }

    /// The resume's NOT EXACT summary for parameters the session did not
    /// record, or nothing when every training parameter was resolved.
    func notExactSummary() -> String? {
        guard !notExactParameterIDs.isEmpty else { return nil }
        return "[RESUME] NOT EXACT: parameters absent from the session (current settings used): "
            + notExactParameterIDs.joined(separator: ", ")
    }
}

// MARK: - GUI session resume

extension SessionParameterResume {

    /// Apply a resumed GUI session's training parameters to the live
    /// settings. Every saved training parameter goes through one resolver
    /// (`TrainingParameterResolution`, driven by each key's declared
    /// `absentValue`): a value the session carried is restored; a key the
    /// session predates gets its declared pre-feature value, held for this
    /// run only; an operational knob keeps the live setting; a training
    /// parameter with no recorded value keeps the live setting and is
    /// reported NOT EXACT. Each decision logs one `[RESUME-DIFF]` line.
    /// `acceptedReplacements` names the saved settings the user, reviewing
    /// the session at load, chose to replace with the current ones.
    func applyGuiSession(_ rs: SessionCheckpointState, acceptedReplacements: Set<String>) {
        let p = parameters
        restore(LearningRate.self, savedFloat: rs.learningRate, into: \.learningRate)
        restore(EntropyBonus.self, savedFloat: rs.entropyRegularizationCoeff, into: \.entropyBonus)
        restore(DrawPenalty.self, savedFloat: rs.drawPenalty, into: \.drawPenalty)
        restore(WeightDecay.self, savedFloat: rs.weightDecayCoeff, into: \.weightDecay)
        restore(DropoutRate.self, savedFloat: rs.dropoutRate, into: \.dropoutRate)
        restore(GradClipMaxNorm.self, savedFloat: rs.gradClipMaxNorm, into: \.gradClipMaxNorm)
        restore(PolicyLossWeight.self, savedFloat: rs.policyLossWeight, into: \.policyLossWeight)
        restore(ValueLossWeight.self, savedFloat: rs.valueLossWeight, into: \.valueLossWeight)
        restore(MomentumCoeff.self, savedFloat: rs.momentumCoeff, into: \.momentumCoeff)
        restore(IllegalMassWeight.self, savedFloat: rs.illegalMassPenaltyWeight, into: \.illegalMassWeight)
        restore(PolicyLabelSmoothingEpsilon.self, savedFloat: rs.policyLabelSmoothingEpsilon, into: \.policyLabelSmoothingEpsilon)
        restorePolicyLabelSmoothingSet(from: rs, acceptedReplacements: acceptedReplacements)
        restore(ValueLabelSmoothingEpsilon.self, savedFloat: rs.valueLabelSmoothingEpsilon, into: \.valueLabelSmoothingEpsilon)
        restore(BatchStatsInterval.self, saved: rs.batchStatsInterval, into: \.batchStatsInterval)
        restore(KLProbeInterval.self, saved: rs.klProbeInterval, into: \.klProbeInterval)
        restoreArenaPromotionCriterion(from: rs)
        // The session's own interval is restored even when it lies
        // outside today's declared range (`restoreFromSession` warns).
        // The one value that cannot be restored is a non-positive
        // one: it feeds `PeriodicSaveController(interval:)`, whose
        // precondition is `interval > 0`, and the app has never
        // written one — only a corrupt or hand-edited session can
        // carry it, and the user reviewed it at load and accepted the
        // current interval in its place.
        if let pai = rs.periodicAutosaveIntervalSec, pai <= 0 {
            logReplacedAtLoad(
                PeriodicAutosaveIntervalSec.self,
                savedDescription: "\(pai)",
                currentDescription: "\(p.periodicAutosaveIntervalSec)",
                why: "is unusable (must be > 0)"
            )
        } else {
            restore(PeriodicAutosaveIntervalSec.self, saved: rs.periodicAutosaveIntervalSec, into: \.periodicAutosaveIntervalSec)
        }
        // Zero means unlimited.
        restore(MaxPeriodicAutosavesKept.self, saved: rs.maxPeriodicAutosavesKept, into: \.maxPeriodicAutosavesKept)
        // The setting only; `CheckpointPaths.automaticSavePruningForcedOff`
        // can still hold pruning off whatever is restored here. The
        // effective state is logged once the session is armed.
        restore(AutomaticSavePruningEnabled.self, saved: rs.automaticSavePruningEnabled, into: \.automaticSavePruningEnabled)
        restore(SessionSaveIncludeReplayBuffer.self, saved: rs.sessionSaveIncludeReplayBuffer, into: \.sessionSaveIncludeReplayBuffer)
        // Run-throughput knobs: operational settings the session ran with.
        restore(SelfPlayConcurrency.self, saved: rs.selfPlayWorkerCount, into: \.selfPlayConcurrency)
        restore(TrainingStepDelayMs.self, saved: rs.stepDelayMs, into: \.trainingStepDelayMs)
        restore(SelfPlayDelayMs.self, saved: rs.selfPlayDelayMs, into: \.selfPlayDelayMs)
        restore(ReplayRatioTarget.self, saved: rs.replayRatioTarget, into: \.replayRatioTarget)
        restore(ReplayRatioAutoAdjust.self, saved: rs.replayRatioAutoAdjust, into: \.replayRatioAutoAdjust)
        if let cid = rs.recordingCorpusID {
            log(
                "[RESUME-PARAM] recording_corpus_id: prior run recorded into corpus \(cid) (informational; this run starts a fresh corpus when recording is on)"
            )
        }
        // `recordingCorpusID` above is provenance only; the *intent to
        // record* lives in this boolean, which is read once at self-play
        // start (below). A transient `--parameters record_self_play_games=true`
        // run does not persist to UserDefaults (suppressPersistence), so
        // without restoring it here a resume would silently drop recording
        // back to the singleton's default. Restoring re-enables recording
        // (into a fresh corpus, matching the line above).
        restore(RecordSelfPlayGames.self, saved: rs.recordSelfPlayGames, into: \.recordSelfPlayGames)
        // LR/momentum cycling and its decay envelope, key by key. The
        // cycle and the envelope are each saved as a whole or not at
        // all; a session without them predates the feature, so the
        // enabled flags, the decay horizon and momentum following
        // resolve to their declared pre-feature values (off) while the
        // inert numbers keep the live settings for the popover. The
        // cycle's phase is a pure function of `trainingSteps`, already
        // restored above, so the schedule continues exactly where it
        // left off.
        let cycle = rs.lrMomentumCycle
        restore(LRCycleEnabled.self, saved: cycle?.lrEnabled, into: \.lrCycleEnabled)
        restore(LRCyclePeriodSteps.self, saved: cycle?.lrPeriodSteps, into: \.lrCyclePeriodSteps)
        restore(LRCycleCount.self, saved: cycle?.lrCount, into: \.lrCycleCount)
        restore(LRCycleMin.self, saved: cycle?.lrMin, into: \.lrCycleMin)
        restore(LRCycleMax.self, saved: cycle?.lrMax, into: \.lrCycleMax)
        restore(LRCycleInvert.self, saved: cycle?.lrInvert, into: \.lrCycleInvert)
        restore(MomentumCycleEnabled.self, saved: cycle?.momentumEnabled, into: \.momentumCycleEnabled)
        restore(MomentumCyclePeriodSteps.self, saved: cycle?.momentumPeriodSteps, into: \.momentumCyclePeriodSteps)
        restore(MomentumCycleCount.self, saved: cycle?.momentumCount, into: \.momentumCycleCount)
        restore(MomentumCycleMin.self, saved: cycle?.momentumMin, into: \.momentumCycleMin)
        restore(MomentumCycleMax.self, saved: cycle?.momentumMax, into: \.momentumCycleMax)
        restore(MomentumCycleInvert.self, saved: cycle?.momentumInvert, into: \.momentumCycleInvert)
        let envelope = rs.lrMomentumCycleEnvelope
        restore(LRCyclePeakEnd.self, saved: envelope?.lrPeakEnd, into: \.lrCyclePeakEnd)
        restore(LRCycleTroughEnd.self, saved: envelope?.lrTroughEnd, into: \.lrCycleTroughEnd)
        restore(LRCycleDecayHorizonSteps.self, saved: envelope?.decayHorizonSteps, into: \.lrCycleDecayHorizonSteps)
        restore(MomentumFollowsLRCycle.self, saved: envelope?.momentumFollowsLRCycle, into: \.momentumFollowsLRCycle)
        restore(MomentumFollowStartLow.self, saved: envelope?.momentumFollowStartLow, into: \.momentumFollowStartLow)
        restore(MomentumFollowStartHigh.self, saved: envelope?.momentumFollowStartHigh, into: \.momentumFollowStartHigh)
        restore(MomentumFollowEndLow.self, saved: envelope?.momentumFollowEndLow, into: \.momentumFollowEndLow)
        restore(MomentumFollowEndHigh.self, saved: envelope?.momentumFollowEndHigh, into: \.momentumFollowEndHigh)
        // Composition-aware replay-buffer sampler constraints. These
        // don't shadow on the trainer — the sampler reads them
        // straight off `TrainingParameters.shared` each
        // `sample(count:)` call (see ReplayBuffer.swift).
        restore(MaxPliesFromAnyOneGame.self, saved: rs.maxPliesFromAnyOneGame, into: \.maxPliesFromAnyOneGame)
        restore(TargetSampledGameLengthPlies.self, saved: rs.targetSampledGameLengthPlies, into: \.targetSampledGameLengthPlies)
        restore(MaxDrawPercentPerBatch.self, saved: rs.maxDrawPercentPerBatch, into: \.maxDrawPercentPerBatch)
        restore(ReplayBufferStratifyByMaterial.self, saved: rs.replayBufferStratifyByMaterial, into: \.replayBufferStratifyByMaterial)
        restore(SelfPlayDrawKeepFraction.self, saved: rs.selfPlayDrawKeepFraction, into: \.selfPlayDrawKeepFraction)
        restore(SelfPlayMaxPliesPerGame.self, saved: rs.maxPliesPerGame, into: \.selfPlayMaxPliesPerGame)
        restore(DrawWatchPDrawThreshold.self, saved: rs.drawWatchPDrawThreshold, into: \.drawWatchPDrawThreshold)
        restore(DrawWatchTerminateGames.self, saved: rs.drawWatchTerminateGames, into: \.drawWatchTerminateGames)
        restore(DrawWatchStreakLength.self, saved: rs.drawWatchStreakLength, into: \.drawWatchStreakLength)
        restore(SqrtBatchScalingLR.self, saved: rs.sqrtBatchScalingForLR, into: \.sqrtBatchScalingLR)
        restore(SignedAdvantageComplementCE.self, saved: rs.signedAdvantageComplementCE, into: \.signedAdvantageComplementCE)
        restore(LRWarmupSteps.self, saved: rs.lrWarmupSteps, into: \.lrWarmupSteps)
        // Run-management knobs that once lived only in app settings.
        restore(ReplayBufferMinPositionsBeforeTraining.self, saved: rs.replayBufferMinPositionsBeforeTraining, into: \.replayBufferMinPositionsBeforeTraining)
        restore(ArenaAutoIntervalSec.self, saved: rs.arenaAutoIntervalSec, into: \.arenaAutoIntervalSec)
        restore(
            ArenaConcurrency.self,
            saved: rs.arenaConcurrency.map { min(UpperContentView.absoluteMaxArenaConcurrency, $0) },
            into: \.arenaConcurrency
        )
        restore(CandidateProbeIntervalSec.self, saved: rs.candidateProbeIntervalSec, into: \.candidateProbeIntervalSec)
        restore(LegalMassCollapseThreshold.self, saved: rs.legalMassCollapseThreshold, into: \.legalMassCollapseThreshold)
        restore(LegalMassCollapseGraceSeconds.self, saved: rs.legalMassCollapseGraceSeconds, into: \.legalMassCollapseGraceSeconds)
        restore(LegalMassCollapseNoImprovementProbes.self, saved: rs.legalMassCollapseNoImprovementProbes, into: \.legalMassCollapseNoImprovementProbes)
        // Sampling schedules: always present in a session.
        restore(SelfPlayStartTau.self, savedFloat: rs.selfPlayTau.startTau, into: \.selfPlayStartTau)
        restore(SelfPlayTargetTau.self, savedFloat: rs.selfPlayTau.floorTau, into: \.selfPlayTargetTau)
        restore(SelfPlayTauDecayPerPly.self, savedFloat: rs.selfPlayTau.decayPerPly, into: \.selfPlayTauDecayPerPly)
        restore(ArenaStartTau.self, savedFloat: rs.arenaTau.startTau, into: \.arenaStartTau)
        restore(ArenaTargetTau.self, savedFloat: rs.arenaTau.floorTau, into: \.arenaTargetTau)
        restore(ArenaTauDecayPerPly.self, savedFloat: rs.arenaTau.decayPerPly, into: \.arenaTauDecayPerPly)
        // Saved-but-not-applied trio: persisted for the resume
        // sheet, but the resumed run deliberately reads the LIVE
        // TrainingParameters values for these. Surface the saved
        // value alongside the one actually used so a
        // batch/threshold/games divergence between save and
        // resume is never silent. (Via sqrt-batch LR scaling, a
        // batch-size divergence also shifts the effective
        // learning rate.)
        func logResumeUsesCurrent<T: Equatable>(_ id: String, saved: T, current: T) {
            let marker = saved == current ? "matches" : "DIFFERS from"
            log(
                "[RESUME-PARAM] \(id): saved=\(saved) \(marker) current=\(current) — resume uses current (saved value is informational)"
            )
        }
        logResumeUsesCurrent("batch_size", saved: rs.batchSize, current: p.trainingBatchSize)
        logResumeUsesCurrent("promote_threshold", saved: rs.promoteThreshold, current: p.arenaPromoteThreshold)
        logResumeUsesCurrent("arena_games", saved: rs.arenaGames, current: p.arenaGamesPerTournament)
        // The run-seed settings (`random_seed_mode`, `random_seed`) are
        // left alone. A resume does not use them: it continues the saved
        // run's seed together with its streams (the trainer file's
        // lineage record), or, when it cannot, draws a seed for the run —
        // `startRealTraining` decides which, and the resume's
        // `rng_sampler` / `serials` gaps report a seed not continued.
        if let notExact = notExactSummary() {
            log(notExact)
        }
    }

    // MARK: - Policy label-smoothing set restore

    /// Writes a resumed session's policy label-smoothing mode, per-move δ and
    /// per-move cap back onto `TrainingParameters.shared`.
    ///
    /// The three travel as a set. A session missing all three predates the
    /// per-move form and factually trained with fixed-total smoothing, so the
    /// mode resolves to its declared pre-feature value (held for this run)
    /// and δ and the cap — inert under fixed-total — keep the live settings.
    /// A partial set or an unknown mode only reaches here after the user
    /// reviewed it at load time and accepted the current settings in its
    /// place (`invalidSavedSettings`).
    func restorePolicyLabelSmoothingSet(from rs: SessionCheckpointState, acceptedReplacements: Set<String>) {
        let p = parameters
        if acceptedReplacements.contains(SessionCheckpointState.SavedSettingID.policyLabelSmoothing) {
            logReplacedAtLoad(
                PolicyLabelSmoothingModeParameter.self,
                savedDescription: rs.policyLabelSmoothingMode ?? "absent",
                currentDescription: p.policyLabelSmoothingMode.logToken,
                why: "set (mode, per-move δ, cap) is unusable"
            )
            logReplacedAtLoad(
                PolicyLabelSmoothingPerMove.self,
                savedDescription: rs.policyLabelSmoothingPerMove.map { "\($0)" } ?? "absent",
                currentDescription: "\(p.policyLabelSmoothingPerMove)",
                why: "set (mode, per-move δ, cap) is unusable"
            )
            logReplacedAtLoad(
                PolicyLabelSmoothingPerMoveCap.self,
                savedDescription: rs.policyLabelSmoothingPerMoveCap.map { "\($0)" } ?? "absent",
                currentDescription: "\(p.policyLabelSmoothingPerMoveCap)",
                why: "set (mode, per-move δ, cap) is unusable"
            )
            return
        }
        let savedModeRawValue: Int?
        if let token = rs.policyLabelSmoothingMode {
            guard rs.policyLabelSmoothingPerMove != nil, rs.policyLabelSmoothingPerMoveCap != nil else {
                preconditionFailure("session load reviews a partial policy label-smoothing set before resume")
            }
            guard let savedMode = PolicyLabelSmoothingMode(logToken: token) else {
                preconditionFailure("session load reviews an unknown policy_label_smoothing_mode before resume; got \(token)")
            }
            savedModeRawValue = savedMode.rawValue
        } else {
            guard rs.policyLabelSmoothingPerMove == nil, rs.policyLabelSmoothingPerMoveCap == nil else {
                preconditionFailure("session load reviews a partial policy label-smoothing set before resume")
            }
            savedModeRawValue = nil
        }
        restore(
            PolicyLabelSmoothingModeParameter.self,
            saved: savedModeRawValue,
            current: p.policyLabelSmoothingMode.rawValue,
            describe: { PolicyLabelSmoothingMode(persistedRawValue: $0).logToken },
            write: { rawValue, source in
                let mode = PolicyLabelSmoothingMode(persistedRawValue: rawValue)
                switch source {
                case .session:
                    p.policyLabelSmoothingMode = mode
                case .preFeature:
                    p.holdForThisRun(PolicyLabelSmoothingModeParameter.self) { p.policyLabelSmoothingMode = mode }
                case .currentSetting, .notExact:
                    break
                }
            }
        )
        restore(PolicyLabelSmoothingPerMove.self, savedFloat: rs.policyLabelSmoothingPerMove, into: \.policyLabelSmoothingPerMove)
        restore(PolicyLabelSmoothingPerMoveCap.self, savedFloat: rs.policyLabelSmoothingPerMoveCap, into: \.policyLabelSmoothingPerMoveCap)
    }

    // MARK: - Arena promotion criterion restore

    /// Writes a resumed session's arena promotion criterion and SPRT
    /// hypotheses back onto `TrainingParameters.shared`.
    ///
    /// The seven values are restored as a **set**, and an incomplete set is
    /// never part-applied. A session that stored `criterion = sprt` but is
    /// missing (say) `elo1` would otherwise resume running a sequential test
    /// against whatever hypotheses happen to be in `UserDefaults` today — a
    /// different experiment under the same session's name. Such a set, an
    /// unrecognised criterion token (a forward-versioned or hand-edited
    /// file), or hypotheses that fail validation reach here only after the
    /// user reviewed them at load and accepted the current settings.
    ///
    /// A session without the criterion predates it: the criterion resolves
    /// to its declared pre-feature value (score threshold, held for this run)
    /// and the SPRT hypotheses — inert under score threshold — keep the live
    /// settings.
    func restoreArenaPromotionCriterion(from rs: SessionCheckpointState) {
        let p = parameters
        let describeCriterion: (Int) -> String = { ArenaPromotionCriterion(persistedRawValue: $0).logToken }
        let writeCriterion: (Int, TrainingParameterResolution.Source) -> Void = { rawValue, source in
            let criterion = ArenaPromotionCriterion(persistedRawValue: rawValue)
            switch source {
            case .session:
                p.arenaPromotionCriterion = criterion
            case .preFeature:
                p.holdForThisRun(ArenaPromotionCriterionParameter.self) { p.arenaPromotionCriterion = criterion }
            case .currentSetting, .notExact:
                break
            }
        }

        guard let token = rs.arenaPromotionCriterion else {
            restore(
                ArenaPromotionCriterionParameter.self,
                saved: nil,
                current: p.arenaPromotionCriterion.rawValue,
                describe: describeCriterion,
                write: writeCriterion
            )
            restore(ArenaSPRTElo0.self, saved: rs.arenaSPRTElo0, into: \.arenaSPRTElo0)
            restore(ArenaSPRTElo1.self, saved: rs.arenaSPRTElo1, into: \.arenaSPRTElo1)
            restore(ArenaSPRTAlpha.self, saved: rs.arenaSPRTAlpha, into: \.arenaSPRTAlpha)
            restore(ArenaSPRTBeta.self, saved: rs.arenaSPRTBeta, into: \.arenaSPRTBeta)
            restore(ArenaSPRTMinGames.self, saved: rs.arenaSPRTMinGames, into: \.arenaSPRTMinGames)
            restore(ArenaSPRTMaxGames.self, saved: rs.arenaSPRTMaxGames, into: \.arenaSPRTMaxGames)
            return
        }

        guard let criterion = ArenaPromotionCriterion.allCases.first(where: { $0.logToken == token }) else {
            logReplacedAtLoad(
                ArenaPromotionCriterionParameter.self,
                savedDescription: "\"\(token)\"",
                currentDescription: p.arenaPromotionCriterion.logToken,
                why: "is not a known criterion"
            )
            return
        }

        guard let elo0 = rs.arenaSPRTElo0,
              let elo1 = rs.arenaSPRTElo1,
              let alpha = rs.arenaSPRTAlpha,
              let beta = rs.arenaSPRTBeta,
              let minGames = rs.arenaSPRTMinGames,
              let maxGames = rs.arenaSPRTMaxGames else {
            logReplacedAtLoad(
                ArenaPromotionCriterionParameter.self,
                savedDescription: token,
                currentDescription: "\(p.arenaPromotionCriterion.logToken) with today's hypotheses",
                why: "has an incomplete SPRT block"
            )
            return
        }

        // Validate the restored set the same way a fresh arena start would,
        // so a hand-edited file cannot install hypotheses that throw later at
        // the point where an arena is already underway.
        do {
            _ = try ArenaSPRT.SPRTConfig(
                elo0: elo0, elo1: elo1, alpha: alpha, beta: beta,
                minGames: minGames, maxGames: maxGames
            )
        } catch {
            logReplacedAtLoad(
                ArenaPromotionCriterionParameter.self,
                savedDescription: token,
                currentDescription: "\(p.arenaPromotionCriterion.logToken) with today's hypotheses",
                why: "has an invalid SPRT block (\(error))"
            )
            return
        }

        restore(
            ArenaPromotionCriterionParameter.self,
            saved: criterion.rawValue,
            current: p.arenaPromotionCriterion.rawValue,
            describe: describeCriterion,
            write: writeCriterion
        )
        restore(ArenaSPRTElo0.self, saved: elo0, into: \.arenaSPRTElo0)
        restore(ArenaSPRTElo1.self, saved: elo1, into: \.arenaSPRTElo1)
        restore(ArenaSPRTAlpha.self, saved: alpha, into: \.arenaSPRTAlpha)
        restore(ArenaSPRTBeta.self, saved: beta, into: \.arenaSPRTBeta)
        restore(ArenaSPRTMinGames.self, saved: minGames, into: \.arenaSPRTMinGames)
        restore(ArenaSPRTMaxGames.self, saved: maxGames, into: \.arenaSPRTMaxGames)
    }
}
