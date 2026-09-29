import Foundation

/// Every trainer-level training parameter, resolved from one
/// `TrainingParametersSnapshot` into the exact types `ChessTrainer` consumes —
/// the single path by which parameters reach a trainer.
///
/// **Why this exists.** The GUI Play-and-Train session, the offline
/// corpus-replay runner (`CorpusReplayRunner`) and the train-vs-UCI runner
/// (`TrainVsUciRunner`) each used to copy parameters onto their trainer with
/// their own hand-written list of assignments. The lists drifted: the two CLI
/// runners never set `lrMomentumCycle`, so a replay run silently trained at the
/// static learning rate and momentum while its `parameters.json` asked for a
/// cycle with a decay envelope; they also never set `batchStatsInterval` or
/// `klProbeInterval`; and the GUI's fresh-start path skipped the illegal-mass
/// weight, both label-smoothing epsilons and the cycle. Each of those was
/// invisible in the log. Funnelling every consumer through `init(_:)` +
/// `apply(to:)` makes that class of drift impossible: a parameter added here
/// reaches every trainer, and one missing here reaches none, which the
/// parity tests catch.
///
/// **What is deliberately not here.** Run-level knobs that are not trainer
/// state — batch size, replay-buffer capacity, replay ratio, prefill — stay
/// with their consumers. The per-field session-resume path in
/// `SessionController+Training.swift` still writes the trainer field by field,
/// because each field has its own "saved value vs. pre-feature fallback"
/// resolution and its own `[RESUME-PARAM]` audit line; it writes the resolved
/// values back onto `TrainingParameters.shared` as it goes, so a later
/// `apply(to:)` from the singleton reproduces the resumed configuration.
///
/// `Sendable` and `Equatable` so the CLI runners can capture it on the main
/// actor and carry it into their detached training task, and so tests can
/// compare the configuration two trainers were given.
struct TrainerHyperparameters: Sendable, Equatable {
    var learningRate: Float
    var entropyRegularizationCoeff: Float
    var drawPenalty: Float
    var weightDecayC: Float
    var dropoutRate: Float
    var gradClipMaxNorm: Float
    var policyLossWeight: Float
    var valueLossWeight: Float
    var illegalMassPenaltyWeight: Float
    var policyLabelSmoothingEpsilon: Float
    var valueLabelSmoothingEpsilon: Float
    var momentumCoeff: Float
    var useSignedAdvantageComplementCE: Bool
    var sqrtBatchScalingForLR: Bool
    var lrWarmupSteps: Int
    var batchStatsInterval: Int
    var klProbeInterval: Int
    /// The LR/momentum cycle together with its decay envelope. Both enabled
    /// flags off means the trainer uses the static `learningRate` and
    /// `momentumCoeff` — the switch that turns cycling off everywhere.
    var lrMomentumCycle: LRMomentumCycle

    /// Resolve every trainer-level parameter from `parameters`.
    init(_ parameters: TrainingParametersSnapshot) {
        learningRate = Float(parameters.learningRate)
        entropyRegularizationCoeff = Float(parameters.entropyBonus)
        drawPenalty = Float(parameters.drawPenalty)
        weightDecayC = Float(parameters.weightDecay)
        dropoutRate = Float(parameters.dropoutRate)
        gradClipMaxNorm = Float(parameters.gradClipMaxNorm)
        policyLossWeight = Float(parameters.policyLossWeight)
        valueLossWeight = Float(parameters.valueLossWeight)
        illegalMassPenaltyWeight = Float(parameters.illegalMassWeight)
        policyLabelSmoothingEpsilon = Float(parameters.policyLabelSmoothingEpsilon)
        valueLabelSmoothingEpsilon = Float(parameters.valueLabelSmoothingEpsilon)
        momentumCoeff = Float(parameters.momentumCoeff)
        useSignedAdvantageComplementCE = parameters.signedAdvantageComplementCE
        sqrtBatchScalingForLR = parameters.sqrtBatchScalingLR
        lrWarmupSteps = parameters.lrWarmupSteps
        batchStatsInterval = parameters.batchStatsInterval
        klProbeInterval = parameters.klProbeInterval
        lrMomentumCycle = parameters.lrMomentumCycle
    }

    /// Read back the configuration a trainer currently holds. Used by the
    /// parity tests to prove two construction paths produced the same trainer.
    init(currentlyAppliedTo trainer: ChessTrainer) {
        learningRate = trainer.learningRate
        entropyRegularizationCoeff = trainer.entropyRegularizationCoeff
        drawPenalty = trainer.drawPenalty
        weightDecayC = trainer.weightDecayC
        dropoutRate = trainer.dropoutRate
        gradClipMaxNorm = trainer.gradClipMaxNorm
        policyLossWeight = trainer.policyLossWeight
        valueLossWeight = trainer.valueLossWeight
        illegalMassPenaltyWeight = trainer.illegalMassPenaltyWeight
        policyLabelSmoothingEpsilon = trainer.policyLabelSmoothingEpsilon
        valueLabelSmoothingEpsilon = trainer.valueLabelSmoothingEpsilon
        momentumCoeff = trainer.momentumCoeff
        useSignedAdvantageComplementCE = trainer.useSignedAdvantageComplementCE
        sqrtBatchScalingForLR = trainer.sqrtBatchScalingForLR
        lrWarmupSteps = trainer.lrWarmupSteps
        batchStatsInterval = trainer.batchStatsInterval
        klProbeInterval = trainer.klProbeInterval
        lrMomentumCycle = trainer.lrMomentumCycle
    }

    /// Write every trainer-level parameter onto `trainer`. Idempotent, and
    /// safe on a running trainer: each property is individually lock-guarded
    /// or graph-variable backed inside `ChessTrainer` and is read by the next
    /// training step.
    func apply(to trainer: ChessTrainer) {
        trainer.learningRate = learningRate
        trainer.entropyRegularizationCoeff = entropyRegularizationCoeff
        trainer.drawPenalty = drawPenalty
        trainer.weightDecayC = weightDecayC
        trainer.dropoutRate = dropoutRate
        trainer.gradClipMaxNorm = gradClipMaxNorm
        trainer.policyLossWeight = policyLossWeight
        trainer.valueLossWeight = valueLossWeight
        trainer.illegalMassPenaltyWeight = illegalMassPenaltyWeight
        trainer.policyLabelSmoothingEpsilon = policyLabelSmoothingEpsilon
        trainer.valueLabelSmoothingEpsilon = valueLabelSmoothingEpsilon
        trainer.momentumCoeff = momentumCoeff
        trainer.useSignedAdvantageComplementCE = useSignedAdvantageComplementCE
        trainer.sqrtBatchScalingForLR = sqrtBatchScalingForLR
        trainer.lrWarmupSteps = lrWarmupSteps
        trainer.batchStatsInterval = batchStatsInterval
        trainer.klProbeInterval = klProbeInterval
        trainer.lrMomentumCycle = lrMomentumCycle
    }
}

extension ChessTrainer {
    /// Build a trainer configured from `hyperparameters` — the one trainer
    /// constructor the GUI session and the CLI runners share. The designated
    /// initializer takes most parameters as arguments (they size or seed graph
    /// state at build time); `apply(to:)` then sets the full list, including
    /// the ones the initializer does not take (dropout, the cycle, the stats
    /// and KL-probe intervals), so no caller has to remember which is which.
    convenience init(
        hyperparameters: TrainerHyperparameters,
        arch: NetworkArchitecture,
        bf16CastInForward: Bool = false
    ) throws {
        try self.init(
            learningRate: hyperparameters.learningRate,
            entropyRegularizationCoeff: hyperparameters.entropyRegularizationCoeff,
            drawPenalty: hyperparameters.drawPenalty,
            weightDecayC: hyperparameters.weightDecayC,
            gradClipMaxNorm: hyperparameters.gradClipMaxNorm,
            policyLossWeight: hyperparameters.policyLossWeight,
            valueLossWeight: hyperparameters.valueLossWeight,
            illegalMassPenaltyWeight: hyperparameters.illegalMassPenaltyWeight,
            policyLabelSmoothingEpsilon: hyperparameters.policyLabelSmoothingEpsilon,
            valueLabelSmoothingEpsilon: hyperparameters.valueLabelSmoothingEpsilon,
            momentumCoeff: hyperparameters.momentumCoeff,
            useSignedAdvantageComplementCE: hyperparameters.useSignedAdvantageComplementCE,
            sqrtBatchScalingForLR: hyperparameters.sqrtBatchScalingForLR,
            lrWarmupSteps: hyperparameters.lrWarmupSteps,
            arch: arch,
            bf16CastInForward: bf16CastInForward
        )
        hyperparameters.apply(to: self)
    }
}

extension ChessTrainer {
    /// The LR/momentum cycle evaluated at `completedSteps` (default: the
    /// trainer's own completed-step count), with the same warmup offset the
    /// SGD feed applies. A nil channel in the result means that channel is not
    /// cycling and the static value is in effect.
    func lrMomentumCycleValues(completedSteps: Int? = nil) -> LRMomentumCycle.Values {
        lrMomentumCycle.values(
            completedTrainSteps: completedSteps ?? completedTrainSteps,
            lrWarmupSteps: lrWarmupSteps
        )
    }
}
