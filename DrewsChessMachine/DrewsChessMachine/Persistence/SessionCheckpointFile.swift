import Foundation

// MARK: - Errors

enum SessionCheckpointError: LocalizedError {
    case missingChampionFile
    case missingTrainerFile
    case missingSessionJSON
    case invalidJSON(Error, detail: String = "")
    case unsupportedVersion(Int)
    case targetDirectoryExists(URL)
    /// A session.json at a format version that requires a lineage has none.
    case missingLineage(formatVersion: Int)

    var errorDescription: String? {
        switch self {
        case .missingChampionFile:
            return "Session directory is missing champion.dcmmodel"
        case .missingTrainerFile:
            return "Session directory is missing trainer.dcmmodel"
        case .missingSessionJSON:
            return "Session directory is missing session.json"
        case .invalidJSON(let err, let detail):
            let base = "session.json could not be decoded: \(err)"
            return detail.isEmpty ? base : "\(base)\nFirst 2000 bytes of file:\n\(detail)"
        case .unsupportedVersion(let v):
            return "Unsupported session.json format version \(v)"
        case .targetDirectoryExists(let url):
            return "Refusing to overwrite existing session at \(url.lastPathComponent)"
        case .missingLineage(let version):
            return "session.json format version \(version) requires a lineage record (required from version "
                + "\(SessionCheckpointState.lineageRequiredFromFormatVersion)), and this one has none"
        }
    }
}

// MARK: - Serializable shapes for non-Codable project types

/// Codable mirror of `SamplingSchedule`. The original struct is
/// project-owned but not Codable (and making it Codable would
/// drag an import into MPSChessPlayer.swift); the checkpoint path
/// builds this wrapper on save and turns it back into the live
/// struct on load.
struct TauConfigCodable: Codable, Equatable {
    let startTau: Float
    let decayPerPly: Float
    let floorTau: Float

    init(_ schedule: SamplingSchedule) {
        self.startTau = schedule.startTau
        self.decayPerPly = schedule.decayPerPly
        self.floorTau = schedule.floorTau
    }

    var asSamplingSchedule: SamplingSchedule {
        SamplingSchedule(
            startTau: startTau,
            decayPerPly: decayPerPly,
            floorTau: floorTau
        )
    }
}

/// Codable mirror of `TournamentRecord` — saved in the session's
/// arena history for context on resume. Only the audit fields
/// (counts, score, promoted flag, duration) are persisted; live UI
/// state like `isCurrent` is not.
///
/// `gamesPlayed` and `promotionKind` are Optional for backward
/// compatibility with session files written before those fields
/// existed. A missing `gamesPlayed` is reconstructed at load time
/// as `candidateWins + championWins + draws` (same identity the
/// tournament driver uses). A missing `promotionKind` is treated
/// as `.automatic` on load when `promoted == true`, which matches
/// the only way promotions could happen before the manual Promote
/// button existed.
struct ArenaHistoryEntryCodable: Codable, Equatable {
    let finishedAtStep: Int
    let candidateWins: Int
    let championWins: Int
    let draws: Int
    let score: Double
    let promoted: Bool
    let promotedID: String?
    let durationSec: Double
    var gamesPlayed: Int?
    var promotionKind: String?
    // Per-side candidate W/L/D — optional for back-compat with
    // session files written before side tracking existed. Missing
    // values decode as `nil` and load-path code substitutes 0 so
    // the display shows "—" for the side breakdown on legacy data.
    var candidateWinsAsWhite: Int?
    var candidateWinsAsBlack: Int?
    var candidateLossesAsWhite: Int?
    var candidateLossesAsBlack: Int?
    var candidateDrawsAsWhite: Int?
    var candidateDrawsAsBlack: Int?
    /// Wall-clock time (seconds since 1970) the tournament
    /// finished. Optional for back-compat with session files
    /// written before the field existed; the Arena History UI
    /// renders "—" when nil.
    var finishedAtUnix: Int64?
    /// Candidate `ModelID` description (e.g. `20260505-3-A1B2`).
    /// Optional for back-compat. Surfaces alongside the verdict
    /// in the Arena History UI.
    var candidateID: String?
    /// Champion `ModelID` description as of arena start, before
    /// any promotion copy. Optional for back-compat.
    var championID: String?
    /// Per-arena breakdown blocks (W/D/L by game length; candidate
    /// value + arena-style score by absolute ply; same by game
    /// progress). Optional for back-compat — older session files
    /// don't carry it, and the arena-history row popover renders
    /// "no breakdown data" placeholders when it's nil. `ArenaExtendedSummary`
    /// is itself Codable, so the persisted shape is just a nested
    /// object inside this entry.
    var extendedSummary: ArenaExtendedSummary?
    /// Which rule decided this arena, as `ArenaPromotionCriterion.logToken`
    /// (`"score"` / `"sprt"`). Optional for back-compat: every session file
    /// written before SPRT existed ran the score threshold, so a missing
    /// value loads as `.scoreThreshold` rather than as "unknown".
    ///
    /// Stored as the token rather than the raw `Int` so a persisted history
    /// stays readable and survives any future renumbering of the enum.
    var promotionCriterion: String?
    /// The sequential test's latched verdict. Nil under the score threshold,
    /// and nil for an SPRT arena that was cut short before deciding.
    var sprt: ArenaSPRTVerdictCodable?
    /// Games the arena's driver had counted as finished, in any order, when
    /// the sequential test's verdict latched
    /// (`TournamentStats.sprtGamesFinishedAtDecision`). Splits the games past
    /// the verdict's start-order sample into those that had already finished
    /// and the in-flight remainder drained afterwards. Optional for
    /// back-compat: nil without a verdict, and absent from sessions saved
    /// before it was stored, where the `[ARENA]` block says the split was not
    /// recorded.
    var sprtGamesFinishedAtDecision: Int?
}

extension ArenaHistoryEntryCodable {
    /// The stored finished count at the verdict, when it can belong to
    /// `verdict` in an arena of `gamesPlayed` games: at least the verdict's
    /// sample, at most every game played. Nil otherwise — without a verdict
    /// there is nothing for it to describe, and a value outside those limits
    /// (a hand-edited or corrupt file) loses the split rather than being
    /// repaired into one the arena never had, the rule
    /// `ArenaSPRTVerdictCodable.verdict()` applies to the verdict itself.
    func validSPRTGamesFinishedAtDecision(verdict: ArenaSPRT.Verdict?, gamesPlayed: Int) -> Int? {
        guard let verdict, let finished = sprtGamesFinishedAtDecision,
              finished >= verdict.gamesAtDecision, finished <= gamesPlayed
        else { return nil }
        return finished
    }
}

/// Codable mirror of `ArenaSPRT.Verdict`, including the configuration it was
/// decided under.
///
/// The config travels with the verdict deliberately. A persisted arena
/// history spans parameter edits, so "accepted at LLR +3.1" is uninterpretable
/// without knowing which hypotheses and error rates produced that number —
/// and re-reading today's `TrainingParameters` to fill them in would silently
/// attribute the current settings to an old run.
struct ArenaSPRTVerdictCodable: Codable, Equatable {
    /// `ArenaSPRT.Decision.rawValue` — always a final one.
    let decision: String
    /// The ratio at the crossing. Nil when the verdict came from the runaway
    /// guard on a record the GSPRT could not score (see "Zero-variance
    /// records" in `documentation/arena-sprt.md`).
    let llr: Double?
    let wins: Int
    let draws: Int
    let losses: Int
    let elo0: Double
    let elo1: Double
    let alpha: Double
    let beta: Double
    let minGames: Int
    let maxGames: Int

    init(_ verdict: ArenaSPRT.Verdict) {
        self.decision = verdict.decision.rawValue
        self.llr = verdict.llr
        self.wins = verdict.wins
        self.draws = verdict.draws
        self.losses = verdict.losses
        self.elo0 = verdict.config.elo0
        self.elo1 = verdict.config.elo1
        self.alpha = verdict.config.alpha
        self.beta = verdict.config.beta
        self.minGames = verdict.config.minGames
        self.maxGames = verdict.config.maxGames
    }

    /// Rebuilds the in-memory verdict.
    ///
    /// Returns nil rather than repairing a file whose stored decision or
    /// config no longer forms a valid test — a hand-edited or
    /// forward-versioned session should lose the verdict and say so, not
    /// silently acquire a different one. `.continueTesting` is rejected for
    /// the same reason: it is not a state a verdict can be in.
    func verdict() -> ArenaSPRT.Verdict? {
        guard let decoded = ArenaSPRT.Decision(rawValue: decision), decoded.isFinal,
              let config = try? ArenaSPRT.SPRTConfig(
                  elo0: elo0, elo1: elo1, alpha: alpha, beta: beta,
                  minGames: minGames, maxGames: maxGames
              )
        else { return nil }
        return ArenaSPRT.Verdict(
            decision: decoded,
            llr: llr,
            wins: wins,
            draws: draws,
            losses: losses,
            config: config
        )
    }
}

/// Network-architecture snapshot captured at save time so the
/// resume sheet can show what shape of network produced the
/// session — e.g. `v4 · 12 blocks · 128 channels · 3.9M params`.
/// These numbers are build-time constants on `ChessNetwork`, but
/// they can change across architecture-version bumps; persisting
/// them lets the resume sheet describe the *saved* session even
/// when the user is now running a build with a different arch
/// (which the build-mismatch warning also flags). All fields are
/// non-Optional inside the struct, but the struct as a whole is
/// Optional on `SessionCheckpointState` for back-compat with
/// sessions saved before the field landed.
struct ArchitectureMetadata: Codable, Equatable {
    /// `NetworkArchitecture.architectureVersionLabel` — distinguishes
    /// topology changes (e.g. the v3 → v4 pre-activation rebuild) that
    /// pure shape constants don't capture.
    let architectureVersion: Int
    let channels: Int
    let numBlocks: Int
    let inputPlanes: Int
    let policySize: Int
    let valueHeadClasses: Int
    /// Squeeze-and-Excitation FC reduction ratio
    /// (`NetworkArchitecture.blockSeReductionRatio`). Surfaced because
    /// changing the SE width without changing channels still produces a
    /// distinct architecture for the model loader's arch hash.
    let seReductionRatio: Int
    /// Trainable + BN parameter count, computed via
    /// `NetworkArchitecture.parameterCount` at save time.
    let parameterCount: Int
}

extension ArchitectureMetadata {
    /// The snapshot of `arch` every session writer records: the legacy
    /// uniform scalars are the tower-output width and the first group's SE
    /// ratio (a mixed tower is fully described by the architecture each model
    /// file embeds).
    init(describing arch: NetworkArchitecture) {
        self.init(
            architectureVersion: arch.architectureVersionLabel,
            channels: arch.towerOutputChannels,
            numBlocks: arch.numBlocks,
            inputPlanes: arch.inputPlanes,
            policySize: arch.policySize,
            valueHeadClasses: arch.valueHeadClasses,
            seReductionRatio: arch.blockGroups[0].seReductionRatio,
            parameterCount: arch.parameterCount
        )
    }
}

// MARK: - Session State

/// Serialized form of a paused training session. Stored at
/// `session.json` inside a `.dcmsession` directory next to the
/// champion and trainer `.dcmmodel` files. Loaded at resume time
/// to re-seed counters, hyperparameter display, and network
/// identity.
struct SessionCheckpointState: Codable, Equatable {
    /// Version history: 1 — the original layout; 2 — adds `lineage`
    /// (`LineageRecord`), required from then on.
    static let currentFormatVersion: Int = 2
    /// First format version whose session.json must carry `lineage`.
    static let lineageRequiredFromFormatVersion: Int = 2

    let formatVersion: Int
    let sessionID: String
    let savedAtUnix: Int64
    let sessionStartUnix: Int64
    /// Accumulated elapsed seconds at save time, measured against
    /// the session's `sessionStart` anchor. On resume, the new
    /// session's anchor is placed at `Date() - elapsedTrainingSec`
    /// so "Total session time" picks up where it left off.
    let elapsedTrainingSec: Double

    // Counters
    let trainingSteps: Int
    let selfPlayGames: Int
    let selfPlayMoves: Int
    /// Positions trained over `trainingSteps`, each step at the batch size
    /// it trained at (`TrainedPositionsCount`). Absent (nil) when some of
    /// those steps are not recorded anywhere — never a model of them, such
    /// as the step count times one batch size. A file written before this
    /// was counted per start holds the step count times the batch size in
    /// force at its save; it is read as written.
    let trainingPositionsSeen: Int?

    // Hyperparameters (as they were in effect at save time)
    let batchSize: Int
    let learningRate: Float
    var entropyRegularizationCoeff: Float?
    /// Bootstrap-phase draw penalty (0 = disabled). See
    /// `ChessTrainer.drawPenalty`. Optional for back-compat with
    /// session files written before the field was added.
    var drawPenalty: Float?
    let promoteThreshold: Double
    let arenaGames: Int
    /// Number of arena games run concurrently per tournament.
    /// Optional for back-compat with session.json files written
    /// before parallel arena existed; absent → load-side hydrates
    /// to the user's current `effectiveArenaConcurrency`.
    var arenaConcurrency: Int?
    let selfPlayTau: TauConfigCodable
    let arenaTau: TauConfigCodable
    let selfPlayWorkerCount: Int
    var gradClipMaxNorm: Float?
    var weightDecayCoeff: Float?
    /// Channel-dropout rate (drop probability, 0 = off). Optional for
    /// back-compat with session files written before dropout existed;
    /// absent → rate 0 (`resolvedDropoutRate`).
    var dropoutRate: Float?
    /// Policy-loss coefficient applied to the policy term in
    /// `total_loss = valueLossWeight·valueLoss +
    /// policyLossWeight·policyLoss − …`. Optional for back-compat
    /// with session files written before the field became editable.
    /// Renamed from `policyScaleK` (the old K knob); old session
    /// files won't carry this key and load with the user's current
    /// `TrainingParameters.shared.policyLossWeight` instead.
    var policyLossWeight: Float?
    /// Value-loss coefficient applied to the value term in
    /// `total_loss`. Mirrors `policyLossWeight`; optional for
    /// back-compat with session files written before the value
    /// weight existed.
    var valueLossWeight: Float?
    /// Polyak momentum coefficient μ in effect at save time. Optional
    /// for back-compat with session files written before momentum
    /// landed in the schema; absent → plain SGD, μ = 0
    /// (`resolvedMomentumCoeff`), the pre-feature behavior.
    /// The optimizer's velocity buffers themselves are persisted
    /// separately in `trainer.dcmmodel` (v2 layout); this scalar
    /// controls how aggressively the saved velocity is mixed in
    /// going forward.
    var momentumCoeff: Float?
    /// Illegal-mass penalty weight in effect at save time. Multiplied
    /// into the unmasked-softmax illegal-mass term in `total_loss`,
    /// where positive values pull probability mass off illegal cells.
    /// Optional for back-compat with session files written before
    /// the term existed; absent → weight 0
    /// (`resolvedIllegalMassPenaltyWeight`), the pre-feature behavior.
    var illegalMassPenaltyWeight: Float?
    /// Policy-CE label-smoothing coefficient ε in effect at save time.
    /// ε=0 → one-hot played-move target; ε>0 → `(1−ε)·oneHot + ε·uniform(legal)`.
    /// Optional for back-compat with session files written before this
    /// term existed; absent → ε = 0
    /// (`resolvedPolicyLabelSmoothingEpsilon`), the pre-feature behavior.
    var policyLabelSmoothingEpsilon: Float?
    /// `PolicyLabelSmoothingMode.logToken` in effect at save time
    /// (`fixed_total` / `per_move`). Stored as the token, not the raw `Int`,
    /// so the file stays readable and survives any renumbering. Optional for
    /// back-compat: a session file lacking it predates the per-move form, so
    /// resume reproduces fixed-total smoothing
    /// (`resolvedPolicyLabelSmoothingMode`) rather than inheriting the live
    /// mode.
    var policyLabelSmoothingMode: String?
    /// Per-move policy label-smoothing mass δ in effect at save time
    /// (`TrainingParameters.shared.policyLabelSmoothingPerMove`). Read only in
    /// per-move mode. Optional for back-compat; absent → loader falls through
    /// to the current value (inert, since such a session resolves to
    /// fixed-total mode).
    var policyLabelSmoothingPerMove: Float?
    /// Cap on the per-move total smoothing mass in effect at save time
    /// (`TrainingParameters.shared.policyLabelSmoothingPerMoveCap`). Optional
    /// for back-compat; absent → loader falls through to the current value.
    var policyLabelSmoothingPerMoveCap: Float?
    /// Value-head W/D/L cross-entropy label-smoothing coefficient ε in
    /// effect at save time. ε=0 → hard one-hot on the game result;
    /// ε>0 → `(1−ε)·oneHot(slot) + ε·(⅓,⅓,⅓)`. Optional for back-compat
    /// with session files written before the WDL value head landed;
    /// absent → ε = 0 (`resolvedValueLabelSmoothingEpsilon`), the
    /// pre-feature behavior.
    var valueLabelSmoothingEpsilon: Float?

    // Replay-ratio controller settings. All Optional so older
    // session.json files that lack these keys still decode.
    var replayRatioTarget: Double?
    var replayRatioAutoAdjust: Bool?
    var stepDelayMs: Int?
    /// Self-play-side per-game-per-worker delay (ms) in effect at save
    /// time. Distinct from `stepDelayMs`, which is the training-side
    /// inter-batch delay. Optional for back-compat; absent → loader
    /// falls through to `TrainingParameters.shared.selfPlayDelayMs`.
    var selfPlayDelayMs: Int?
    var lastAutoComputedDelayMs: Int?

    // Training-loop parameters that previously lived only in
    // @AppStorage and so silently drifted between session-save
    // time and session-resume time. All Optional for back-compat
    // with older session.json files that pre-date the schema
    // expansion; on resume, an absent field falls through to the
    // user's current @AppStorage value, while a present field
    // overrides @AppStorage so reload is fully reproducible.
    var lrWarmupSteps: Int?
    var sqrtBatchScalingForLR: Bool?
    /// Whether the signed-advantage complement-CE branch was active at
    /// save time. When true, negative-advantage samples drive the
    /// policy via a complementary CE against a mirror-smoothed target
    /// (mass on the OTHER legal moves) instead of contributing zero
    /// gradient (the legacy clamp-on regime). Optional for back-compat
    /// with session files written before the toggle landed; resolve via
    /// `resolvedSignedAdvantageComplementCE` — never by falling through
    /// to the live `TrainingParameters` value.
    var signedAdvantageComplementCE: Bool?

    /// The complement-CE setting a resumed session should actually run
    /// with.
    ///
    /// This checkpoint field was introduced in the same commit as the
    /// complement-CE feature itself, so a session file lacking the
    /// field provably predates the feature — the run it captured
    /// factually trained in the legacy clamp-on regime. The
    /// absent-field fallback must therefore reproduce that pre-feature
    /// behavior (off), NOT the user's current default: falling through
    /// to the live setting silently switched the policy-gradient
    /// regime of old runs on resume. (Observed on the resumed KbHZ
    /// session from 2026-05: its policy loss jumped at resume because
    /// complement CE was applied to a run that never used it.)
    ///
    /// Kept as a pure static so the policy is unit-testable without
    /// constructing a full checkpoint state; the instance property is
    /// the call-site-facing form.
    static func resolvedSignedAdvantageComplementCE(savedFlag: Bool?) -> Bool {
        savedFlag ?? TrainingParameterResolution.absentValue(of: SignedAdvantageComplementCE.self)
    }

    /// Instance form of `resolvedSignedAdvantageComplementCE(savedFlag:)`.
    var resolvedSignedAdvantageComplementCE: Bool {
        Self.resolvedSignedAdvantageComplementCE(savedFlag: signedAdvantageComplementCE)
    }

    // MARK: Pre-feature fallback resolution
    //
    // The resolvers below share the complement-CE situation: each
    // checkpoint field was introduced together with the feature it
    // controls, so a session file lacking the field provably predates
    // the feature — the run it captured factually trained WITHOUT it.
    // The absent-field fallback must therefore reproduce that
    // pre-feature behavior, never the live `TrainingParameters` value
    // (which would silently change the regime of an old run on
    // resume). Each pre-feature value is declared once, on its key's
    // `@TrainingParameter(absentValue:)`, and read here through
    // `TrainingParameterResolution.absentValue(of:)`; the resume path
    // itself resolves every key through `TrainingParameterResolution`.
    // These statics stay as a typed view for the session file's own
    // callers and tests (`SessionResumeParameterFallbackTests`).

    /// Pre-feature behavior: plain SGD — no momentum term existed.
    static func resolvedMomentumCoeff(saved: Float?) -> Float {
        saved ?? Float(TrainingParameterResolution.absentValue(of: MomentumCoeff.self))
    }

    /// Pre-feature behavior: no channel dropout. A session predating the
    /// dropout feature factually trained at rate 0, so resume must NOT
    /// inherit the live `TrainingParameters.dropoutRate` (which could
    /// silently inject dropout into an old run).
    static func resolvedDropoutRate(saved: Float?) -> Float {
        saved ?? Float(TrainingParameterResolution.absentValue(of: DropoutRate.self))
    }

    /// Pre-feature behavior: no illegal-mass penalty term in the loss.
    static func resolvedIllegalMassPenaltyWeight(saved: Float?) -> Float {
        saved ?? Float(TrainingParameterResolution.absentValue(of: IllegalMassWeight.self))
    }

    /// Pre-feature behavior: one-hot policy CE — no label smoothing.
    static func resolvedPolicyLabelSmoothingEpsilon(saved: Float?) -> Float {
        saved ?? Float(TrainingParameterResolution.absentValue(of: PolicyLabelSmoothingEpsilon.self))
    }

    /// Pre-feature behavior: fixed-total policy smoothing, the only form that
    /// existed before `policy_label_smoothing_mode`. `saved` is the session's
    /// token already decoded (`PolicyLabelSmoothingMode(logToken:)`); a token
    /// no mode spells is the caller's to report, not a reason to guess.
    static func resolvedPolicyLabelSmoothingMode(saved: PolicyLabelSmoothingMode?) -> PolicyLabelSmoothingMode {
        saved ?? PolicyLabelSmoothingMode(
            persistedRawValue: TrainingParameterResolution.absentValue(of: PolicyLabelSmoothingModeParameter.self)
        )
    }

    /// Pre-feature behavior: hard one-hot W/D/L target — no value-head
    /// label smoothing. A session file lacking this field predates the
    /// term, so resume must reproduce ε=0 rather than inherit the live
    /// `TrainingParameters.valueLabelSmoothingEpsilon` (which would
    /// silently re-shape an old run's value loss on resume — the same
    /// regression the sibling resolvers exist to prevent).
    static func resolvedValueLabelSmoothingEpsilon(saved: Float?) -> Float {
        saved ?? Float(TrainingParameterResolution.absentValue(of: ValueLabelSmoothingEpsilon.self))
    }

    /// Pre-feature behavior: uncapped per-game batch sampling. The
    /// parameter's range no longer includes a disabled value, so the
    /// closest representable equivalent is the declared range maximum,
    /// where the cap essentially never binds at observed game lengths.
    /// Sourced from the parameter definition (the single source of truth)
    /// so it tracks any future change to the declared range rather than
    /// drifting from a hardcoded literal.
    static func resolvedMaxPliesFromAnyOneGame(saved: Int?) -> Int {
        saved ?? TrainingParameterResolution.absentValue(of: MaxPliesFromAnyOneGame.self)
    }

    /// Pre-feature behavior: no decay envelope and no momentum following.
    /// On nil the caller's current envelope values are preserved (so the
    /// settings popover keeps them) but the horizon is zeroed and following
    /// is turned off — the saved run cycled without either, and resume must
    /// not change its schedule.
    static func resolvedLRMomentumCycleEnvelope(
        saved: LRMomentumCycleEnvelope?,
        current: LRMomentumCycleEnvelope
    ) -> LRMomentumCycleEnvelope {
        if let saved { return saved }
        var resolved = current
        resolved.decayHorizonSteps = TrainingParameterResolution.absentValue(of: LRCycleDecayHorizonSteps.self)
        resolved.momentumFollowsLRCycle = TrainingParameterResolution.absentValue(of: MomentumFollowsLRCycle.self)
        return resolved
    }

    /// Pre-feature behavior: no LR/momentum cycling. On nil the
    /// caller's current cycle numbers (periods, bounds) are preserved
    /// so the settings popover keeps the user's values, but both
    /// enabled flags are forced off — a live cycle must never be
    /// applied to a session that predates the cycling feature.
    static func resolvedLRMomentumCycle(
        saved: LRMomentumCycle?,
        current: LRMomentumCycle
    ) -> LRMomentumCycle {
        if let saved { return saved }
        return LRMomentumCycle(
            lrEnabled: TrainingParameterResolution.absentValue(of: LRCycleEnabled.self),
            lrPeriodSteps: current.lrPeriodSteps,
            lrCount: current.lrCount,
            lrMin: current.lrMin,
            lrMax: current.lrMax,
            lrInvert: current.lrInvert,
            momentumEnabled: TrainingParameterResolution.absentValue(of: MomentumCycleEnabled.self),
            momentumPeriodSteps: current.momentumPeriodSteps,
            momentumCount: current.momentumCount,
            momentumMin: current.momentumMin,
            momentumMax: current.momentumMax,
            momentumInvert: current.momentumInvert
        )
    }
    var replayBufferMinPositionsBeforeTraining: Int?
    var arenaAutoIntervalSec: Double?
    var candidateProbeIntervalSec: Double?
    var legalMassCollapseThreshold: Double?
    var legalMassCollapseGraceSeconds: Double?
    var legalMassCollapseNoImprovementProbes: Int?
    /// Interval (in training steps) between the trainer's per-batch
    /// statistics and graph-diagnostics steps (`batch_stats_interval`; the
    /// `[BATCH-STATS]` line itself is written with the step lines).
    /// Optional for back-compat; absent → loader falls through to
    /// `TrainingParameters.shared.batchStatsInterval`.
    var batchStatsInterval: Int?
    /// KL-probe cadence in training steps (0 = off) in effect at save time.
    /// Optional for back-compat; absent → the loader falls through to the
    /// current `TrainingParameters.klProbeInterval`.
    var klProbeInterval: Int?
    /// The relative gradient cap's five settings in effect at save time
    /// (`relative_grad_clip_*`). Optional because sessions written before
    /// the cap do not state them; absent → the mode resolves to its
    /// pre-feature value (`off`, held for the run) and the other four to the
    /// current settings (`SessionParameterResume`).
    var relativeGradClipMode: Int?
    var relativeGradClipMultiple: Double?
    var relativeGradClipWindowSteps: Int?
    var relativeGradClipMinHistorySteps: Int?
    var relativeGradClipFloor: Double?
    /// Step-line time interval in seconds (`step_line_interval_sec`) in
    /// effect at save time. Logging only. Optional because sessions written
    /// before the parameter existed do not state it; absent → the resume
    /// keeps the current setting (its `absentValue` is `.currentSetting`).
    var stepLineIntervalSec: Double?
    /// Periodic-autosave cadence (seconds) in effect at save time
    /// (`TrainingParameters.shared.periodicAutosaveIntervalSec`). Optional for
    /// back-compat; absent → loader falls through to the current value.
    var periodicAutosaveIntervalSec: Double?
    /// Automatic-save (periodic + post-promotion) retention cap in effect at
    /// save time (`TrainingParameters.shared.maxPeriodicAutosavesKept`;
    /// 0 = unlimited).
    /// Optional for back-compat; absent → loader falls through to the current
    /// value.
    var maxPeriodicAutosavesKept: Int?
    /// Whether automatic-save pruning was switched on at save time
    /// (`TrainingParameters.shared.automaticSavePruningEnabled`). Records
    /// the setting only — a build can still force pruning off regardless
    /// (`CheckpointPaths.automaticSavePruningForcedOff`). Optional for
    /// back-compat; absent → loader falls through to the current value.
    var automaticSavePruningEnabled: Bool?
    /// Whether automatic saves included the replay buffer at save time
    /// (`TrainingParameters.shared.sessionSaveIncludeReplayBuffer`). The
    /// setting, not what this save did — `hasReplayBuffer` records that.
    /// Optional for back-compat; absent → the loader keeps the current
    /// value (`absentValue: .currentSetting`).
    var sessionSaveIncludeReplayBuffer: Bool?
    // --- Training health alarms (TRAINING_HEALTH_ALARMS_PLAN.md, Part K) ---
    //
    // Operational settings in effect at save time. All Optional because
    // sessions written before the alarms existed do not state them; absent →
    // the resume keeps the current setting (`absentValue: .currentSetting`).
    /// `training_health_alarms_enabled`.
    var trainingHealthAlarmsEnabled: Bool?
    /// `training_health_check_interval_steps`.
    var trainingHealthCheckIntervalSteps: Int?
    /// `training_health_learning_grace_steps`.
    var trainingHealthLearningGraceSteps: Int?
    /// `training_health_action_<rule>`, one per rule, stored as the action's
    /// name (`log`, `stop_on_critical`, `stop_on_any`) rather than its raw
    /// `Int`, so the file stays readable and survives a renumbering — the
    /// same choice as `arenaPromotionCriterion`. An unknown name is a
    /// finding of `invalidSavedSettings(current:)`.
    var trainingHealthActionNonFinite: String?
    var trainingHealthActionDeadChannels: String?
    var trainingHealthActionValueFC1ZeroVelocity: String?
    var trainingHealthActionIllegalMass: String?
    var trainingHealthActionGradientCollapse: String?
    var trainingHealthActionLossSpike: String?
    var trainingHealthActionPolicyOffsetDrift: String?
    var trainingHealthActionBatchNormRunningVarianceRunaway: String?
    var trainingHealthActionGradientSpike: String?
    var trainingHealthActionDivergence: String?
    var trainingHealthActionValueSaturation: String?
    var trainingHealthActionValueDrawSaturation: String?
    var trainingHealthActionLegalMassStall: String?
    var trainingHealthActionBatchNormRunningVarianceJump: String?
    // --- Arena promotion criterion ---
    //
    // All Optional for back-compat; absent → the loader falls through to the
    // current `TrainingParameters` value. They are saved together because a
    // sequential test is only interpretable as a set: resuming a session with
    // the criterion restored but the hypotheses defaulted would run a
    // different experiment under the same name.
    /// `ArenaPromotionCriterion.logToken` in effect at save time. Stored as
    /// the token, not the raw `Int`, so the file stays readable and survives
    /// any future renumbering.
    var arenaPromotionCriterion: String?
    var arenaSPRTElo0: Double?
    var arenaSPRTElo1: Double?
    var arenaSPRTAlpha: Double?
    var arenaSPRTBeta: Double?
    var arenaSPRTMinGames: Int?
    var arenaSPRTMaxGames: Int?
    /// Id of the standalone game corpus this run recorded self-play games into
    /// (under Corpora/), or nil when recording was off. Provenance only — the
    /// corpus lives outside the session folder. Optional + defaulted for
    /// back-compat with older session files.
    var recordingCorpusID: String? = nil
    /// Whether self-play recording was enabled at save time
    /// (`TrainingParameters.shared.recordSelfPlayGames`). `recordingCorpusID`
    /// is provenance for the corpus the prior run wrote; this boolean is the
    /// *intent to record*, which resume restores so a recording session keeps
    /// recording (into a fresh corpus). Optional + defaulted for back-compat
    /// with older session files (absent → loader falls through to current).
    var recordSelfPlayGames: Bool? = nil
    /// LR/momentum cycling configuration in effect at save time (the 12
    /// `lr_cycle_*` / `momentum_cycle_*` parameters bundled into the runtime
    /// `LRMomentumCycle` struct). Optional for back-compat with session files
    /// written before cycling landed; absent → loader falls through to the
    /// user's current `TrainingParameters.shared` cycling values. Because the
    /// schedule's phase is a pure function of the global step (which is also
    /// persisted as `trainingSteps`), restoring this is all resume needs to
    /// continue the cycle seamlessly.
    var lrMomentumCycle: LRMomentumCycle?
    /// LR-cycle decay envelope + momentum-follow configuration in effect at
    /// save time (the `lr_cycle_peak_end` / `lr_cycle_trough_end` /
    /// `lr_cycle_decay_horizon_steps` / `momentum_follow*` parameters). Kept
    /// separate from `lrMomentumCycle` so sessions written before the
    /// envelope existed still decode that field. Optional for back-compat;
    /// absent → the resumed run gets no decay and no following (the saved
    /// run had neither), with the user's other envelope values preserved.
    var lrMomentumCycleEnvelope: LRMomentumCycleEnvelope?
    /// Composition-aware replay-buffer sampler constraints in effect at
    /// save time. All Optional for back-compat with session files written
    /// before these knobs existed; absent → loader falls through to the
    /// user's current `TrainingParameters.shared` value. The sampler reads
    /// these directly off `TrainingParameters.shared` per `sample(count:)`
    /// call (see `ReplayBuffer.swift`), so resume just needs to write the
    /// saved value back onto the singleton.
    var maxPliesFromAnyOneGame: Int?
    var targetSampledGameLengthPlies: Int?
    var maxDrawPercentPerBatch: Int?
    /// Whether material-bucket-stratified replay-buffer sampling was on
    /// at save time (`TrainingParameters.shared.replayBufferStratifyByMaterial`).
    /// Optional for back-compat with session files written before this
    /// knob existed; absent → loader falls through to the user's
    /// current value. The sampler reads this directly off
    /// `TrainingParameters.shared` per `sample(count:)` call (via
    /// `SamplingConstraints.fromCurrentParameters()`), so resume just
    /// needs to write the saved value back onto the singleton.
    var replayBufferStratifyByMaterial: Bool?
    /// Fraction of drawn self-play games kept in the replay buffer
    /// at game end (the rest are dropped). 1.0 = legacy behaviour
    /// (keep everything). Optional for back-compat with sessions
    /// saved before the draw-keep filter existed; absent → loader
    /// falls through to `TrainingParameters.shared.selfPlayDrawKeepFraction`
    /// (which is 1.0 by default).
    var selfPlayDrawKeepFraction: Double?
    /// Hard cap on self-play game length (plies) before the game is
    /// dropped without emit. Optional for back-compat with sessions
    /// saved before the cap existed; absent → loader falls through
    /// to `TrainingParameters.shared.selfPlayMaxPliesPerGame`. The
    /// on-disk JSON key is still `maxPliesPerGame` — keeping the
    /// struct field name preserves loadability of pre-rename sessions.
    var maxPliesPerGame: Int?
    /// pDraw threshold the self-play draw-watch monitor uses (when a
    /// position's W/D/L draw probability clears this for N consecutive
    /// plies — N from `drawWatchStreakLength`, default 8 — the game
    /// is flagged on the Draw-watch chart tile).
    /// Mirrors `TrainingParameters.shared.drawWatchPDrawThreshold`.
    /// Optional for back-compat with sessions saved before this knob
    /// existed; absent → loader falls through to the param's default.
    var drawWatchPDrawThreshold: Double?
    /// When true, the self-play driver drops a game on the spot the
    /// moment its N-ply pDraw streak completes — same drop path as
    /// the ply-cap. Mirrors
    /// `TrainingParameters.shared.drawWatchTerminateGames`. Optional
    /// for back-compat; absent → loader falls through to the param's
    /// default (`false`, observe-only).
    var drawWatchTerminateGames: Bool?
    /// Number of consecutive plies above the pDraw threshold required
    /// to fire a draw-watch flag. Mirrors
    /// `TrainingParameters.shared.drawWatchStreakLength`. Optional
    /// for back-compat with sessions saved before this knob existed;
    /// absent → loader falls through to the param's default (8).
    var drawWatchStreakLength: Int?
    /// Lifetime self-play games that were emitted into the replay
    /// buffer (i.e. survived the draw-keep filter). `<= selfPlayGames`;
    /// equal at default keep-fraction. Optional for back-compat.
    var emittedGames: Int?
    /// Lifetime plies emitted into the replay buffer across the
    /// session. `<= selfPlayMoves`; equal at default keep-fraction.
    /// Optional for back-compat.
    var emittedPositions: Int?

    // Game-result breakdown (added v1.1 — Optional for compat)
    var whiteCheckmates: Int?
    var blackCheckmates: Int?
    var stalemates: Int?
    var fiftyMoveDraws: Int?
    var threefoldRepetitionDraws: Int?
    var insufficientMaterialDraws: Int?
    /// Lifetime count of self-play games dropped for hitting the
    /// `selfPlayMaxPliesPerGame` cap. Optional for back-compat — absent in
    /// sessions saved before the cap existed; treated as 0 on load.
    var maxPliesDropped: Int?
    var totalGameWallMs: Double?

    // Per-outcome emitted-game breakdown (added when the Results card
    // gained an Overall vs Kept layout). All Optional for back-compat
    // with sessions saved before these counters existed; loader falls
    // back to the played-side counterparts (the draw-keep filter was
    // either disabled or absent in those sessions, so emitted == played
    // at every outcome category).
    var emittedWhiteCheckmates: Int?
    var emittedBlackCheckmates: Int?
    var emittedStalemates: Int?
    var emittedFiftyMoveDraws: Int?
    var emittedThreefoldRepetitionDraws: Int?
    var emittedInsufficientMaterialDraws: Int?

    // Build metadata captured at save time. Optional for back-compat
    // with older session.json files that lack these fields.
    var buildNumber: Int?
    var buildGitHash: String?
    var buildGitBranch: String?
    var buildDate: String?
    var buildTimestamp: String?
    var buildGitDirty: Bool?

    // Replay-buffer presence (added alongside `replay_buffer.bin`).
    // `true` if the session directory contains a matching
    // replay-buffer file; nil/false for older sessions without it.
    var hasReplayBuffer: Bool?
    var replayBufferStoredCount: Int?
    var replayBufferCapacity: Int?
    var replayBufferTotalPositionsAdded: Int?

    // Chart-data presence (added alongside the optional
    // `training_chart.json` and `progress_rate_chart.json`
    // companion files). `true` iff both files exist in the session
    // directory and the per-ring sample counts agree with the
    // values in those files. Older sessions and sessions saved
    // with chart collection disabled are nil/false here. Loaded
    // by `seedFromRestoredSession` to populate the chart rings on
    // resume so the chart trajectory survives save/resume cycles.
    var hasChartData: Bool?
    var trainingChartSampleCount: Int?
    var progressRateSampleCount: Int?

    // Inline auxiliary chart state, small enough to live next to
    // the rest of session.json instead of in a side file. Both
    // Optional for back-compat; missing/nil decodes the same way
    // older sessions did (no arena bands restored,
    // `legalMassMaxAllTime` resets to 0 on session start).
    var arenaChartEvents: [ArenaChartEvent]?
    var legalMassMaxAllTime: Double?

    /// Per-run training history. Each `TrainingSegment` represents one
    /// continuous Play-and-Train period (start → stop, save, or
    /// session-quit). Cumulative status-bar metrics sum across this
    /// array plus the in-memory current run. Optional for back-compat
    /// with older session files written before segments existed; the
    /// loader treats nil/missing as "no historical segments." Each
    /// save closes the current segment with `endUnix = saveTime` and
    /// appends it; on resume, a new segment begins on the next
    /// Play-and-Train start.
    var trainingSegments: [TrainingSegment]?

    // Network identity — duplicated from the `.dcmmodel` headers so
    // a future "browse saved sessions" UI can read just
    // `session.json` and still show model IDs.
    let championID: String
    let trainerID: String

    /// Architecture-shape snapshot captured at save time. Optional for
    /// back-compat with session files written before the field landed;
    /// absent → resume sheet renders the architecture line as "unknown
    /// (session predates architecture metadata)".
    var architecture: ArchitectureMetadata?

    // Arena history (audit log — displayed in the UI on resume)
    let arenaHistory: [ArenaHistoryEntryCodable]

    /// Serialized 200-puzzle Lichess probe monitor history (OVERALL
    /// NLL/Elo series, per-theme aggregate series, and the latest
    /// per-puzzle detail rows). Optional for back-compat: sessions saved
    /// before probe-history persistence existed decode this as nil, and
    /// the loader starts the monitor empty. Positions aren't stored —
    /// the per-puzzle rows are reconstructed by name on resume (see
    /// `ProbeResultCodable`).
    var lichessProbeHistory: LichessProbeHistorySnapshot?

    /// Serialized WIDE-set (~4,435-puzzle) Lichess probe history,
    /// parallel to `lichessProbeHistory`. Optional for back-compat:
    /// sessions saved before the wide set existed decode this as nil and
    /// the wide monitor starts empty.
    var lichessProbeWideHistory: LichessProbeHistorySnapshot?

    /// Serialized tactical probe monitor history (per-probe time series
    /// of full `ProbeResult`s). Optional for back-compat, same as
    /// `lichessProbeHistory`.
    var tacticalProbeHistory: TacticalProbeHistorySnapshot?

    /// The run's lineage at the save — the same record the session's
    /// champion and trainer files carry (see `LineageRecord`). Required from
    /// format version 2; nil only in a version-1 file, written before
    /// lineage existed.
    var lineage: LineageRecord?

    /// Seconds since the last arena ended (or the run began) at the save —
    /// the arena-trigger clock a resume restores, so the next automatic arena
    /// comes due when it would have in the saved run (determinism plan C1
    /// #13). A save deferred past an arena's start predates that arena, so a
    /// resume re-runs it when due (D-6). Nil when no run was training at the
    /// save, and in sessions saved before the clock was recorded; such a
    /// resume restarts the clock and reports `clocks` NOT EXACT.
    var arenaSecondsSinceLastArena: Double?
    /// The self-play diversity window's games, oldest first
    /// (`GameDiversityTracker.windowSequences()`), so a resume continues the
    /// rolling diversity readings instead of starting them empty
    /// (determinism plan C1 #18). Nil in sessions saved before it was
    /// recorded.
    var selfPlayDiversityWindow: [[Int16]]?
    /// The training alarms' streak counters (`TrainingAlarmController.Streaks`)
    /// at save time, so a resume keeps counting toward (or out of) an alarm
    /// (determinism plan C1 #20). Nil in sessions saved before it was
    /// recorded.
    var trainingAlarmStreaks: TrainingAlarmController.Streaks?
    /// The legal-mass-collapse detector's probe window and the grace period
    /// it had used at save time (`LegalMassCollapseDetectorBox.snapshot`), so
    /// a resume neither restarts the grace period nor forgets its recent
    /// probes (determinism plan C1 #20). Nil in sessions saved before it was
    /// recorded.
    var legalMassCollapseDetector: LegalMassCollapseDetectorState?

    // MARK: - Training Segments

    /// One Play-and-Train run, bounded by start and end wall-clock
    /// times. Status-bar wall-time totals sum `durationSec` across the
    /// session's full segment array — that's "active training time"
    /// and excludes idle gaps when training was stopped. The segment
    /// also captures starting/ending counter snapshots so per-run
    /// progress can be reconstructed (e.g., "this run added 12K
    /// training steps and 3.5M positions to the buffer").
    ///
    /// The build/git fields are captured on segment-start so each
    /// segment is attributable to a specific code version — invaluable
    /// for "which build produced this entropy curve?" forensics across
    /// architecture changes.
    struct TrainingSegment: Codable, Equatable {
        let startUnix: Int64
        let endUnix: Int64
        let durationSec: Double

        let startingTrainingStep: Int
        let endingTrainingStep: Int

        let startingTotalPositions: Int
        let endingTotalPositions: Int

        let startingSelfPlayGames: Int
        let endingSelfPlayGames: Int

        let buildNumber: Int?
        let buildGitHash: String?
        let buildGitDirty: Bool?

        // Optional summary captured at segment-end (last [STATS] tick
        // values). Used by detail views; not required for cumulative
        // status-bar metrics.
        var endPolicyEntropy: Double?
        var endLossTotal: Double?
        var endGradNorm: Double?
    }

    // MARK: JSON serialization

    static func decode(_ data: Data) throws -> SessionCheckpointState {
        do {
            var state = try JSONDecoder().decode(SessionCheckpointState.self, from: data)
            // `LRMomentumCycle` leaves its envelope out of its encoded form
            // and the envelope is stored beside it, so the cycle comes back
            // with the no-decay envelope. Put the saved one back, as the
            // trainer's safetensors metadata does, so the decoded cycle is
            // the one that was saved — `CheckpointManager.saveSession`'s
            // round-trip check compares the whole struct. A session without
            // a saved envelope predates it and ran without decay, which is
            // what the cycle already holds.
            if let savedEnvelope = state.lrMomentumCycleEnvelope {
                state.lrMomentumCycle?.envelope = savedEnvelope
            }
            guard (1...Self.currentFormatVersion).contains(state.formatVersion) else {
                throw SessionCheckpointError.unsupportedVersion(state.formatVersion)
            }
            if state.formatVersion >= Self.lineageRequiredFromFormatVersion && state.lineage == nil {
                throw SessionCheckpointError.missingLineage(formatVersion: state.formatVersion)
            }
            return state
        } catch let err as SessionCheckpointError {
            throw err
        } catch {
            throw SessionCheckpointError.invalidJSON(
                error,
                detail: String(data: data.prefix(2000), encoding: .utf8) ?? "(non-utf8)"
            )
        }
    }

    /// Return a copy with `trainingSegments` replaced. Builder helper
    /// that lets `buildCurrentSessionState` construct the bulk of the
    /// state via the synthesized memberwise init (which is already at
    /// the SwiftUI/Swift type-checker complexity threshold) and then
    /// layer the segments in afterward, without forcing the init call
    /// site to grow another argument.
    func withTrainingSegments(_ segments: [TrainingSegment]?) -> SessionCheckpointState {
        var copy = self
        copy.trainingSegments = segments
        return copy
    }

    /// Return a copy with the chart-data fields filled in. Same
    /// reason as `withTrainingSegments`: keeps the memberwise init
    /// call site lean. Pass `nil` for `hasChartData` when no chart
    /// snapshot is being saved (no companion files written).
    /// Return a copy with `architecture` populated. Same builder-helper
    /// pattern as `withTrainingSegments` / `withChartData` — keeps the
    /// memberwise init call site lean.
    func withArchitecture(_ architecture: ArchitectureMetadata?) -> SessionCheckpointState {
        var copy = self
        copy.architecture = architecture
        return copy
    }

    /// The training-health settings in effect at save time — the one writer
    /// of those fields, shared by the GUI save and the train-vs-UCI save.
    func withTrainingHealthSettings(
        enabled: Bool,
        checkIntervalSteps: Int,
        learningGraceSteps: Int,
        actions: TrainingHealthActions
    ) -> SessionCheckpointState {
        var copy = self
        copy.trainingHealthAlarmsEnabled = enabled
        copy.trainingHealthCheckIntervalSteps = checkIntervalSteps
        copy.trainingHealthLearningGraceSteps = learningGraceSteps
        for rule in TrainingHealthRule.allCases {
            copy[savedTrainingHealthActionFor: rule] = actions[rule].name
        }
        return copy
    }

    /// The saved action name of one training-health rule: the one mapping
    /// from a rule to its session field.
    subscript(savedTrainingHealthActionFor rule: TrainingHealthRule) -> String? {
        get {
            switch rule {
            case .nonFinite: return trainingHealthActionNonFinite
            case .deadChannels: return trainingHealthActionDeadChannels
            case .valueFC1ZeroVelocity: return trainingHealthActionValueFC1ZeroVelocity
            case .illegalMass: return trainingHealthActionIllegalMass
            case .gradientCollapse: return trainingHealthActionGradientCollapse
            case .lossSpike: return trainingHealthActionLossSpike
            case .policyOffsetDrift: return trainingHealthActionPolicyOffsetDrift
            case .batchNormRunningVarianceRunaway: return trainingHealthActionBatchNormRunningVarianceRunaway
            case .gradientSpike: return trainingHealthActionGradientSpike
            case .divergence: return trainingHealthActionDivergence
            case .valueSaturation: return trainingHealthActionValueSaturation
            case .valueDrawSaturation: return trainingHealthActionValueDrawSaturation
            case .legalMassStall: return trainingHealthActionLegalMassStall
            case .batchNormRunningVarianceJump: return trainingHealthActionBatchNormRunningVarianceJump
            }
        }
        set {
            switch rule {
            case .nonFinite: trainingHealthActionNonFinite = newValue
            case .deadChannels: trainingHealthActionDeadChannels = newValue
            case .valueFC1ZeroVelocity: trainingHealthActionValueFC1ZeroVelocity = newValue
            case .illegalMass: trainingHealthActionIllegalMass = newValue
            case .gradientCollapse: trainingHealthActionGradientCollapse = newValue
            case .lossSpike: trainingHealthActionLossSpike = newValue
            case .policyOffsetDrift: trainingHealthActionPolicyOffsetDrift = newValue
            case .batchNormRunningVarianceRunaway: trainingHealthActionBatchNormRunningVarianceRunaway = newValue
            case .gradientSpike: trainingHealthActionGradientSpike = newValue
            case .divergence: trainingHealthActionDivergence = newValue
            case .valueSaturation: trainingHealthActionValueSaturation = newValue
            case .valueDrawSaturation: trainingHealthActionValueDrawSaturation = newValue
            case .legalMassStall: trainingHealthActionLegalMassStall = newValue
            case .batchNormRunningVarianceJump: trainingHealthActionBatchNormRunningVarianceJump = newValue
            }
        }
    }

    func withChartData(
        hasChartData: Bool?,
        trainingChartSampleCount: Int?,
        progressRateSampleCount: Int?,
        arenaChartEvents: [ArenaChartEvent]?,
        legalMassMaxAllTime: Double?
    ) -> SessionCheckpointState {
        var copy = self
        copy.hasChartData = hasChartData
        copy.trainingChartSampleCount = trainingChartSampleCount
        copy.progressRateSampleCount = progressRateSampleCount
        copy.arenaChartEvents = arenaChartEvents
        copy.legalMassMaxAllTime = legalMassMaxAllTime
        return copy
    }

    /// Return a copy with the probe-monitor histories attached. Same
    /// builder-helper pattern as `withChartData` — keeps the memberwise
    /// init call site lean.
    func withProbeHistories(
        lichess: LichessProbeHistorySnapshot?,
        wideLichess: LichessProbeHistorySnapshot?,
        tactical: TacticalProbeHistorySnapshot?
    ) -> SessionCheckpointState {
        var copy = self
        copy.lichessProbeHistory = lichess
        copy.lichessProbeWideHistory = wideLichess
        copy.tacticalProbeHistory = tactical
        return copy
    }

    /// Return a copy carrying the arena-trigger clock. Same builder-helper
    /// pattern as `withTrainingSegments`.
    func withArenaClock(secondsSinceLastArena: Double?) -> SessionCheckpointState {
        var copy = self
        copy.arenaSecondsSinceLastArena = secondsSinceLastArena
        return copy
    }

    /// Return a copy carrying the run's rolling observability state: the
    /// self-play diversity window and the alarm streak counters. Same
    /// builder-helper pattern as `withTrainingSegments`.
    func withRunObservability(diversityWindow: [[Int16]]?,
                              alarmStreaks: TrainingAlarmController.Streaks?) -> SessionCheckpointState {
        var copy = self
        copy.selfPlayDiversityWindow = diversityWindow
        copy.trainingAlarmStreaks = alarmStreaks
        return copy
    }

    /// Return a copy carrying the legal-mass-collapse detector's state. Same
    /// builder-helper pattern as `withTrainingSegments`.
    func withLegalMassCollapseDetector(_ detector: LegalMassCollapseDetectorState?) -> SessionCheckpointState {
        var copy = self
        copy.legalMassCollapseDetector = detector
        return copy
    }

    /// Return a copy carrying `lineage`. Same builder-helper pattern as
    /// `withTrainingSegments`.
    func withLineage(_ lineage: LineageRecord) -> SessionCheckpointState {
        var copy = self
        copy.lineage = lineage
        return copy
    }

    func encode() throws -> Data {
        if formatVersion >= Self.lineageRequiredFromFormatVersion && lineage == nil {
            throw SessionCheckpointError.missingLineage(formatVersion: formatVersion)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        do {
            return try encoder.encode(self)
        } catch {
            throw SessionCheckpointError.invalidJSON(error)
        }
    }
}

// MARK: - Directory layout

/// Filenames for the three items inside a `.dcmsession` directory.
/// A session is a plain directory (not a bundle) so the
/// constituent `.dcmmodel` files are immediately usable when
/// copied out in Finder.
enum SessionCheckpointLayout {
    static let championFilename = "champion.safetensors"
    static let trainerFilename = "trainer.safetensors"
    static let legacyChampionFilename = "champion.dcmmodel"
    static let legacyTrainerFilename = "trainer.dcmmodel"
    static let stateFilename = "session.json"
    static let replayBufferFilename = "replay_buffer.bin"
    static let trainingChartFilename = "training_chart.json"
    static let progressRateChartFilename = "progress_rate_chart.json"

    /// Canonical (write) path for new sessions: `champion.safetensors`.
    static func championURL(in directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(championFilename)
    }

    /// Canonical (write) path for new sessions: `trainer.safetensors`.
    static func trainerURL(in directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(trainerFilename)
    }

    /// Read path: native `.safetensors` if present, else legacy `.dcmmodel`.
    /// Falls back to the `.safetensors` path when neither exists so the
    /// caller's `fileExists` check reports the file missing as before.
    static func existingChampionURL(in directoryURL: URL) -> URL {
        resolveExisting(in: directoryURL, primary: championFilename, legacy: legacyChampionFilename)
    }

    static func existingTrainerURL(in directoryURL: URL) -> URL {
        resolveExisting(in: directoryURL, primary: trainerFilename, legacy: legacyTrainerFilename)
    }

    private static func resolveExisting(in directoryURL: URL, primary: String, legacy: String) -> URL {
        let primaryURL = directoryURL.appendingPathComponent(primary)
        if FileManager.default.fileExists(atPath: primaryURL.path) { return primaryURL }
        let legacyURL = directoryURL.appendingPathComponent(legacy)
        if FileManager.default.fileExists(atPath: legacyURL.path) { return legacyURL }
        return primaryURL
    }

    static func stateURL(in directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(stateFilename)
    }

    static func replayBufferURL(in directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(replayBufferFilename)
    }

    static func trainingChartURL(in directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(trainingChartFilename)
    }

    static func progressRateChartURL(in directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(progressRateChartFilename)
    }

    /// Read the three raw payloads out of a session directory.
    /// Parsing is deferred to the caller so errors from the two
    /// `.dcmmodel` files surface with their original
    /// `ModelCheckpointError` types.
    static func readAll(
        from directoryURL: URL
    ) throws -> (stateData: Data, championData: Data, trainerData: Data) {
        // Normalize the incoming URL to a plain file-path URL.
        // The file importer on macOS can return file-reference URLs
        // or bookmark URLs whose `appendingPathComponent` doesn't
        // resolve to the expected child path. Reconstructing via
        // `URL(fileURLWithPath:isDirectory:)` strips any of that
        // metadata and gives a clean POSIX-path-based URL whose
        // children resolve correctly.
        let normalizedDir = URL(fileURLWithPath: directoryURL.path, isDirectory: true)
        let fm = FileManager.default
        let championURL = existingChampionURL(in: normalizedDir)
        let trainerURL = existingTrainerURL(in: normalizedDir)
        let stateURL = stateURL(in: normalizedDir)

        guard fm.fileExists(atPath: championURL.path) else {
            throw SessionCheckpointError.missingChampionFile
        }
        guard fm.fileExists(atPath: trainerURL.path) else {
            throw SessionCheckpointError.missingTrainerFile
        }
        guard fm.fileExists(atPath: stateURL.path) else {
            throw SessionCheckpointError.missingSessionJSON
        }

        let stateData = try Data(contentsOf: stateURL)
        let championData = try Data(contentsOf: championURL)
        let trainerData = try Data(contentsOf: trainerURL)
        return (stateData, championData, trainerData)
    }
}

// MARK: - Saved arena promotion set

extension SessionCheckpointState {
    /// What a session saved for the arena promotion criterion and its SPRT
    /// hypotheses, resolved once. The load-time review
    /// (`invalidSavedSettings`) and the resume (`restoreArenaPromotionCriterion`)
    /// both read this, so a set the review passes is exactly a set the
    /// resume can apply.
    enum SavedArenaPromotionSet: Equatable, Sendable {
        /// None of the seven values: a session from before the criterion
        /// existed, which resumes on the declared pre-feature criterion.
        case absent
        /// The criterion and a complete SPRT block that forms a valid test.
        case valid(criterion: ArenaPromotionCriterion, sprt: ArenaSPRT.SPRTConfig)
        /// Anything else, with what is wrong: an unknown criterion, a
        /// partial set, hypotheses without a criterion, or hypotheses that
        /// do not form a valid test. A session the app wrote never has one.
        case invalid(problem: String)
    }

    /// The criterion and SPRT hypotheses this session saved. A session saves
    /// all seven or none; whatever else is found is `.invalid`, never
    /// partly applied.
    var savedArenaPromotionSet: SavedArenaPromotionSet {
        let elo0 = arenaSPRTElo0, elo1 = arenaSPRTElo1, alpha = arenaSPRTAlpha
        let beta = arenaSPRTBeta, minGames = arenaSPRTMinGames, maxGames = arenaSPRTMaxGames
        let anySPRTSaved = elo0 != nil || elo1 != nil || alpha != nil || beta != nil
            || minGames != nil || maxGames != nil
        guard let token = arenaPromotionCriterion else {
            return anySPRTSaved
                ? .invalid(problem: "SPRT hypotheses are saved without a promotion criterion; "
                    + "a session saves all seven or none")
                : .absent
        }
        guard let criterion = ArenaPromotionCriterion.allCases.first(where: { $0.logToken == token }) else {
            return .invalid(problem: "\"\(token)\" is not a known promotion criterion")
        }
        guard let elo0, let elo1, let alpha, let beta, let minGames, let maxGames else {
            return .invalid(problem: "the SPRT hypotheses are only partly saved")
        }
        do {
            let sprt = try ArenaSPRT.SPRTConfig(
                elo0: elo0, elo1: elo1, alpha: alpha, beta: beta, minGames: minGames, maxGames: maxGames)
            return .valid(criterion: criterion, sprt: sprt)
        } catch {
            return .invalid(problem: "the saved SPRT hypotheses are not a valid test (\(error))")
        }
    }
}

// MARK: - Saved settings that cannot be resumed as found

extension SessionCheckpointState {
    /// Stable ids for the saved-setting groups `invalidSavedSettings(current:)`
    /// checks; the resume block names the same ids when it applies a
    /// replacement the user accepted.
    enum SavedSettingID {
        static let policyLabelSmoothing = "policy_label_smoothing"
        static let arenaPromotionCriterion = "arena_promotion_criterion"
        static let periodicAutosaveInterval = "periodic_autosave_interval_sec"
        static let trainingHealthActions = "training_health_actions"
    }

    /// The saved settings a resume cannot use as found — a hand-edited,
    /// corrupt, or partially written `session.json` — each with the current
    /// setting offered in its place. Empty for every session the app wrote
    /// itself. A load with findings stops for the user to review them; a
    /// replacement is applied only when the user accepts it.
    func invalidSavedSettings(current: TrainingParametersSnapshot) -> [InvalidStoredSetting] {
        var findings: [InvalidStoredSetting] = []

        // Policy label smoothing: the mode, δ and cap are saved together or
        // not at all (a pre-feature session has none and resumes fixed-total).
        let smoothingFields: [(String, String?)] = [
            ("mode", policyLabelSmoothingMode),
            ("per_move", policyLabelSmoothingPerMove.map { "\($0)" }),
            ("per_move_cap", policyLabelSmoothingPerMoveCap.map { "\($0)" }),
        ]
        let presentSmoothingCount = smoothingFields.filter { $0.1 != nil }.count
        let currentSmoothing = "mode \(current.policyLabelSmoothingMode.logToken), "
            + "per_move \(current.value(for: PolicyLabelSmoothingPerMove.self)), "
            + "per_move_cap \(current.value(for: PolicyLabelSmoothingPerMoveCap.self))"
        let foundSmoothing = smoothingFields.map { "\($0.0)=\($0.1 ?? "missing")" }.joined(separator: ", ")
        if presentSmoothingCount > 0 && presentSmoothingCount < smoothingFields.count {
            findings.append(InvalidStoredSetting(
                id: SavedSettingID.policyLabelSmoothing,
                name: "Policy label smoothing (mode, per-move δ, cap)",
                found: foundSmoothing,
                problem: "only part of the set is saved; a session saves all three or none",
                replacement: currentSmoothing
            ))
        } else if let token = policyLabelSmoothingMode, PolicyLabelSmoothingMode(logToken: token) == nil {
            findings.append(InvalidStoredSetting(
                id: SavedSettingID.policyLabelSmoothing,
                name: "Policy label smoothing (mode, per-move δ, cap)",
                found: foundSmoothing,
                problem: "mode \"\(token)\" is not a known mode (fixed_total or per_move)",
                replacement: currentSmoothing
            ))
        }

        // Arena promotion criterion and its SPRT hypotheses — resolved once
        // (`savedArenaPromotionSet`), the same resolution
        // `restoreArenaPromotionCriterion` applies at resume.
        if case .invalid(let problem) = savedArenaPromotionSet {
            let currentArena = "\(current.arenaPromotionCriterion.logToken), SPRT "
                + "elo0 \(current.arenaSPRTElo0) elo1 \(current.arenaSPRTElo1) "
                + "alpha \(current.arenaSPRTAlpha) beta \(current.arenaSPRTBeta) "
                + "games \(current.arenaSPRTMinGames)…\(current.arenaSPRTMaxGames)"
            let sprtFields: [(String, String?)] = [
                ("elo0", arenaSPRTElo0.map { "\($0)" }), ("elo1", arenaSPRTElo1.map { "\($0)" }),
                ("alpha", arenaSPRTAlpha.map { "\($0)" }), ("beta", arenaSPRTBeta.map { "\($0)" }),
                ("min_games", arenaSPRTMinGames.map { "\($0)" }), ("max_games", arenaSPRTMaxGames.map { "\($0)" }),
            ]
            let foundArena = "\(arenaPromotionCriterion ?? "criterion missing"); "
                + sprtFields.map { "\($0.0)=\($0.1 ?? "missing")" }.joined(separator: ", ")
            findings.append(InvalidStoredSetting(
                id: SavedSettingID.arenaPromotionCriterion,
                name: "Arena promotion criterion",
                found: foundArena,
                problem: problem,
                replacement: currentArena
            ))
        }

        // Periodic autosave interval: a non-positive interval cannot drive the
        // periodic save timer.
        if let interval = periodicAutosaveIntervalSec, !(interval > 0) {
            findings.append(InvalidStoredSetting(
                id: SavedSettingID.periodicAutosaveInterval,
                name: "Periodic autosave interval",
                found: "\(interval) s",
                problem: "an interval must be greater than zero",
                replacement: "\(current.periodicAutosaveIntervalSec) s"
            ))
        }

        // Training-health actions: each saved name must be a known action.
        // One finding for the whole set, so accepting it replaces every
        // saved action with the current one (the actions are read together:
        // a stop decision consults every active rule's action).
        let unknownActions = TrainingHealthRule.allCases.compactMap { rule -> String? in
            guard let name = self[savedTrainingHealthActionFor: rule] else { return nil }
            do {
                _ = try TrainingHealthAction(name: name)
                return nil
            } catch {
                return "\(rule.rawValue)=\(name)"
            }
        }
        if !unknownActions.isEmpty {
            findings.append(InvalidStoredSetting(
                id: SavedSettingID.trainingHealthActions,
                name: "Training health actions",
                found: unknownActions.joined(separator: ", "),
                problem: "not a known action (\(TrainingHealthAction.allCases.map(\.name).joined(separator: ", ")))",
                replacement: TrainingHealthRule.allCases
                    .map { "\($0.rawValue)=\(current.trainingHealthAction(for: $0).name)" }
                    .joined(separator: ", ")
            ))
        }
        return findings
    }
}
