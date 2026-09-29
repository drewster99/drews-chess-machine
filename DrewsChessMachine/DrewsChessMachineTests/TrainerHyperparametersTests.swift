//
//  TrainerHyperparametersTests.swift
//  DrewsChessMachineTests
//
//  Guards the single path by which training parameters reach a trainer
//  (`TrainerHyperparameters`) and the single validator every parameter
//  writer uses (the declared `@TrainingParameter` range).
//
//  The bug these pin: the corpus-replay and train-vs-UCI runners built their
//  trainer from a hand-copied subset of the parameters that omitted the
//  LR/momentum cycle, so a replay run silently trained at the static LR and
//  momentum whatever `lr_cycle_*` / `momentum_cycle_*` said. And the settings
//  popovers restated parameter ranges as literals that had drifted from the
//  declarations, so the UI accepted a τ floor the `--parameters` loader
//  rejects.
//
//  Every test that touches `TrainingParameters.shared` snapshots it first,
//  suppresses `UserDefaults` persistence for the duration, and restores the
//  snapshot afterwards, so neither a failure nor a pass leaks into the next
//  test or into the user's saved settings.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class TrainerHyperparametersTests: XCTestCase {

    private var savedParameterValues: [String: ParameterValue] = [:]

    override func setUp() async throws {
        try await super.setUp()
        savedParameterValues = TrainingParameters.shared.snapshot().rawValueMap()
        TrainingParameters.suppressPersistence = true
    }

    override func tearDown() async throws {
        try TrainingParameters.shared.apply(savedParameterValues)
        TrainingParameters.suppressPersistence = false
        try await super.tearDown()
    }

    /// A fully specified cycle with no warmup, no √batch scaling and no decay,
    /// so the effective values can be checked against the waveform's known
    /// trough and peak positions.
    private func configureActiveCycle() {
        let p = TrainingParameters.shared
        p.learningRate = 0.002
        p.momentumCoeff = 0.5
        p.lrWarmupSteps = 0
        p.sqrtBatchScalingLR = false
        p.lrCycleEnabled = true
        p.lrCycleMin = 0.001
        p.lrCycleMax = 0.1
        p.lrCyclePeriodSteps = 1000
        p.lrCycleCount = 0
        p.lrCycleInvert = false
        p.momentumCycleEnabled = true
        p.momentumCycleMin = 0.8
        p.momentumCycleMax = 0.95
        p.momentumCyclePeriodSteps = 1000
        p.momentumCycleCount = 0
        p.momentumCycleInvert = false
        p.lrCycleDecayHorizonSteps = 0
        p.momentumFollowsLRCycle = false
    }

    // MARK: - Resolution from parameters

    func test_init_carriesTheCycleFromParameters() {
        configureActiveCycle()
        let hyperparameters = TrainerHyperparameters(TrainingParameters.shared.snapshot())
        let cycle = hyperparameters.lrMomentumCycle
        XCTAssertTrue(cycle.lrEnabled)
        XCTAssertEqual(cycle.lrMin, 0.001)
        XCTAssertEqual(cycle.lrMax, 0.1)
        XCTAssertEqual(cycle.lrPeriodSteps, 1000)
        XCTAssertTrue(cycle.momentumEnabled)
        XCTAssertEqual(cycle.momentumMin, 0.8)
        XCTAssertEqual(cycle.momentumMax, 0.95)
        XCTAssertEqual(cycle.envelope.decayHorizonSteps, 0)
        XCTAssertFalse(cycle.envelope.momentumFollowsLRCycle)
        XCTAssertEqual(cycle, TrainingParameters.shared.lrMomentumCycle)
    }

    func test_init_carriesEveryTrainerLevelParameter() {
        let p = TrainingParameters.shared
        p.learningRate = 0.0031
        p.entropyBonus = 0.002
        p.drawPenalty = 0.3
        p.weightDecay = 0.0004
        p.dropoutRate = 0.15
        p.gradClipMaxNorm = 12
        p.policyLossWeight = 1.5
        p.valueLossWeight = 0.75
        p.illegalMassWeight = 2
        p.policyLabelSmoothingEpsilon = 0.05
        p.valueLabelSmoothingEpsilon = 0.02
        p.momentumCoeff = 0.85
        p.signedAdvantageComplementCE = false
        p.sqrtBatchScalingLR = false
        p.lrWarmupSteps = 321
        p.batchStatsInterval = 7
        p.klProbeInterval = 250
        let h = TrainerHyperparameters(p.snapshot())
        XCTAssertEqual(h.learningRate, Float(0.0031))
        XCTAssertEqual(h.entropyRegularizationCoeff, Float(0.002))
        XCTAssertEqual(h.drawPenalty, Float(0.3))
        XCTAssertEqual(h.weightDecayC, Float(0.0004))
        XCTAssertEqual(h.dropoutRate, Float(0.15))
        XCTAssertEqual(h.gradClipMaxNorm, Float(12))
        XCTAssertEqual(h.policyLossWeight, Float(1.5))
        XCTAssertEqual(h.valueLossWeight, Float(0.75))
        XCTAssertEqual(h.illegalMassPenaltyWeight, Float(2))
        XCTAssertEqual(h.policyLabelSmoothingEpsilon, Float(0.05))
        XCTAssertEqual(h.valueLabelSmoothingEpsilon, Float(0.02))
        XCTAssertEqual(h.momentumCoeff, Float(0.85))
        XCTAssertFalse(h.useSignedAdvantageComplementCE)
        XCTAssertFalse(h.sqrtBatchScalingForLR)
        XCTAssertEqual(h.lrWarmupSteps, 321)
        XCTAssertEqual(h.batchStatsInterval, 7)
        XCTAssertEqual(h.klProbeInterval, 250)
    }

    /// The CLI runners' parameter bundle must carry exactly what the GUI
    /// resolves — it used to hold its own hand-picked subset.
    func test_replayParams_carryTheSameTrainerConfigurationAsTheGUIPath() {
        configureActiveCycle()
        let snapshot = TrainingParameters.shared.snapshot()
        XCTAssertEqual(ReplayParams(snapshot).trainer, TrainerHyperparameters(snapshot))
    }

    // MARK: - Applied to a trainer

    func test_activeCycle_drivesTheTrainersEffectiveLRAndMomentum() throws {
        configureActiveCycle()
        let trainer = try ChessTrainer(
            hyperparameters: TrainerHyperparameters(TrainingParameters.shared.snapshot()),
            arch: .current
        )
        // Non-inverted cosine: trough at the period boundary, peak at the
        // midpoint — for both channels.
        XCTAssertEqual(trainer.effectiveLearningRate(forBatchSize: 4096, completedSteps: 0), Float(0.001), accuracy: 1e-9)
        XCTAssertEqual(trainer.effectiveLearningRate(forBatchSize: 4096, completedSteps: 500), Float(0.1), accuracy: 1e-7)
        XCTAssertEqual(trainer.effectiveMomentum(completedSteps: 0), Float(0.8), accuracy: 1e-6)
        XCTAssertEqual(trainer.effectiveMomentum(completedSteps: 500), Float(0.95), accuracy: 1e-6)
        // Neither equals the static value it overrides.
        XCTAssertNotEqual(trainer.effectiveLearningRate(forBatchSize: 4096, completedSteps: 250), Float(0.002))
        XCTAssertNotEqual(trainer.effectiveMomentum(completedSteps: 250), Float(0.5))
    }

    func test_disabledCycleFlags_giveTheStaticLRAndMomentum() throws {
        configureActiveCycle()
        TrainingParameters.shared.lrCycleEnabled = false
        TrainingParameters.shared.momentumCycleEnabled = false
        let trainer = try ChessTrainer(
            hyperparameters: TrainerHyperparameters(TrainingParameters.shared.snapshot()),
            arch: .current
        )
        for step in [0, 1, 250, 500, 999, 12_345] {
            XCTAssertEqual(trainer.effectiveLearningRate(forBatchSize: 4096, completedSteps: step), Float(0.002), "step \(step)")
            XCTAssertEqual(trainer.effectiveMomentum(completedSteps: step), Float(0.5), "step \(step)")
            let values = trainer.lrMomentumCycleValues(completedSteps: step)
            XCTAssertNil(values.learningRate, "step \(step)")
            XCTAssertNil(values.momentum, "step \(step)")
        }
    }

    /// A trainer built the way the CLI runners build one and a trainer
    /// configured the way the GUI session reconfigures a live one end up with
    /// the same configuration and the same effective LR and momentum at every
    /// step.
    func test_cliBuiltAndGUIConfiguredTrainers_matchStepForStep() throws {
        configureActiveCycle()
        let p = TrainingParameters.shared
        p.lrWarmupSteps = 100
        p.sqrtBatchScalingLR = true
        p.lrCycleDecayHorizonSteps = 5000
        p.lrCyclePeakEnd = 0.01
        p.lrCycleTroughEnd = 0.0001
        p.momentumFollowsLRCycle = true
        p.dropoutRate = 0.1
        p.klProbeInterval = 100
        let snapshot = p.snapshot()

        let cliTrainer = try ChessTrainer(hyperparameters: ReplayParams(snapshot).trainer, arch: .current)
        let guiTrainer = try ChessTrainer(arch: .current)
        TrainerHyperparameters(snapshot).apply(to: guiTrainer)

        XCTAssertEqual(
            TrainerHyperparameters(currentlyAppliedTo: cliTrainer),
            TrainerHyperparameters(currentlyAppliedTo: guiTrainer)
        )
        XCTAssertEqual(TrainerHyperparameters(currentlyAppliedTo: cliTrainer), TrainerHyperparameters(snapshot))
        for step in [0, 50, 100, 101, 350, 600, 1100, 4_000, 5_100, 20_000] {
            XCTAssertEqual(
                cliTrainer.effectiveLearningRate(forBatchSize: 4096, completedSteps: step),
                guiTrainer.effectiveLearningRate(forBatchSize: 4096, completedSteps: step),
                "step \(step)"
            )
            XCTAssertEqual(
                cliTrainer.effectiveMomentum(completedSteps: step),
                guiTrainer.effectiveMomentum(completedSteps: step),
                "step \(step)"
            )
        }
    }

    // MARK: - Error descriptions

    func test_trainingConfigError_localizedDescriptionIsTheRealMessage() {
        let error: Error = TrainingConfigError.outOfRange(id: "self_play_target_tau", value: "0.02")
        XCTAssertEqual(error.localizedDescription, "Value 0.02 is out of range for parameter 'self_play_target_tau'")
        XCTAssertEqual(
            (TrainingConfigError.unknownParameter(id: "no_such") as Error).localizedDescription,
            "Unknown parameter 'no_such'"
        )
        XCTAssertEqual(
            (TrainingConfigError.wrongType(id: "learning_rate") as Error).localizedDescription,
            "Wrong value type for parameter 'learning_rate'"
        )
    }

    func test_cliParametersLoad_rejectsOutOfRangeWithTheRealMessage_andAppliesNothing() throws {
        let p = TrainingParameters.shared
        let learningRateBefore = p.learningRate
        let targetTauBefore = p.selfPlayTargetTau
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-params-\(UUID().uuidString).json")
        // A valid change alongside the invalid one: all-or-nothing means the
        // valid one must not land either.
        let differentLearningRate = learningRateBefore == 0.0042 ? 0.0043 : 0.0042
        let json = "{\"learning_rate\": \(differentLearningRate), \"self_play_target_tau\": 0.005}"
        try Data(json.utf8).write(to: url)
        defer {
            do {
                try FileManager.default.removeItem(at: url)
            } catch {
                XCTFail("could not remove temp parameters file: \(error)")
            }
        }
        do {
            _ = try CliTrainingConfig.loadAndApplyTransiently(path: url.path)
            XCTFail("an out-of-range value must be rejected")
        } catch {
            XCTAssertEqual(error.localizedDescription, "Value 0.005 is out of range for parameter 'self_play_target_tau'")
        }
        XCTAssertEqual(p.learningRate, learningRateBefore)
        XCTAssertEqual(p.selfPlayTargetTau, targetTauBefore)
    }

    // MARK: - One validator for every writer

    func test_singletonSetter_revertsAnOutOfRangeAssignment() {
        let p = TrainingParameters.shared
        p.selfPlayTargetTau = 0.5
        p.selfPlayTargetTau = 0.005
        XCTAssertEqual(p.selfPlayTargetTau, 0.5)
        p.arenaTargetTau = 0.2
        p.arenaTargetTau = 0.005
        XCTAssertEqual(p.arenaTargetTau, 0.2)
    }

    func test_trainingPopover_rejectsATauFloorBelowTheDeclaredRange() {
        let p = TrainingParameters.shared
        p.selfPlayTargetTau = 0.5
        let model = TrainingSettingsPopoverModel(selfPlayDelayMaxMs: 10_000, stepDelayMaxMs: 10_000, maxSelfPlayWorkers: 8192)
        model.seedFromParams()
        model.selfPlayFloorTauText = "0.005"
        model.save()
        XCTAssertTrue(model.selfPlayFloorTauError)
        XCTAssertEqual(p.selfPlayTargetTau, 0.5)

        model.selfPlayFloorTauText = "0.01"
        model.save()
        XCTAssertFalse(model.selfPlayFloorTauError)
        XCTAssertEqual(p.selfPlayTargetTau, 0.01)
    }

    func test_arenaPopover_rejectsATauFloorBelowTheDeclaredRange() {
        let p = TrainingParameters.shared
        p.arenaTargetTau = 0.2
        let model = ArenaSettingsPopoverModel(
            maxConcurrency: 4096,
            formatDurationSpec: { "\(Int($0))s" },
            parseDurationSpec: { Double($0.replacingOccurrences(of: "s", with: "")) }
        )
        model.seedFromParams()
        model.tauFloorText = "0.005"
        model.save()
        XCTAssertTrue(model.tauFloorError)
        XCTAssertEqual(p.arenaTargetTau, 0.2)
    }

    func test_parsedInDeclaredRange_matchesTheDeclaration() {
        XCTAssertNil(SelfPlayTargetTau.parsedInDeclaredRange("0.005"))
        XCTAssertEqual(SelfPlayTargetTau.parsedInDeclaredRange(" 0.01 "), 0.01)
        XCTAssertNil(SelfPlayTargetTau.parsedInDeclaredRange("nan"))
        XCTAssertNil(SelfPlayTargetTau.parsedInDeclaredRange("abc"))
        XCTAssertNil(LRWarmupSteps.parsedInDeclaredRange("-1"))
        XCTAssertEqual(LRWarmupSteps.parsedInDeclaredRange("0"), 0)
    }
}
