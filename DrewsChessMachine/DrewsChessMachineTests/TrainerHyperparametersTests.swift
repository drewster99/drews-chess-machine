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
import TrainingParametersMacroSupport
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

    /// The one keyframe of `configureActiveCycle`: peak 0.1, trough 0.001
    /// from step 0 on (no decay). The momentum pair is inert while momentum
    /// does not follow the LR cycle.
    private static let constantKeyframes = LRKeyframeSchedule(keyframes: [
        LRKeyframe(step: 0, peak: 0.1, trough: 0.001, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
    ])

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
        p.lrCycleKeyframes = Self.constantKeyframes
        p.lrCyclePeriodSteps = 1000
        p.lrCycleCount = 0
        p.lrCycleInvert = false
        p.momentumCycleEnabled = true
        p.momentumCycleMin = 0.8
        p.momentumCycleMax = 0.95
        p.momentumCyclePeriodSteps = 1000
        p.momentumCycleCount = 0
        p.momentumCycleInvert = false
        p.momentumFollowsLRCycle = false
    }

    // MARK: - Resolution from parameters

    /// LR keyframes: the LR bounds and the decay (lrMin / lrMax, the
    /// envelope's horizon) are now the cycle's keyframes, so those are
    /// checked as the one no-decay keyframe; momentum following moved from
    /// the envelope onto the cycle.
    func test_init_carriesTheCycleFromParameters() {
        configureActiveCycle()
        let hyperparameters = TrainerHyperparameters(TrainingParameters.shared.snapshot())
        let cycle = hyperparameters.lrMomentumCycle
        XCTAssertTrue(cycle.lrEnabled)
        XCTAssertEqual(cycle.lrKeyframes, Self.constantKeyframes)
        XCTAssertEqual(cycle.lrKeyframes.keyframes.count, 1, "no decay: one keyframe")
        XCTAssertEqual(cycle.lrPeriodSteps, 1000)
        XCTAssertTrue(cycle.momentumEnabled)
        XCTAssertEqual(cycle.momentumMin, 0.8)
        XCTAssertEqual(cycle.momentumMax, 0.95)
        XCTAssertFalse(cycle.momentumFollowsLRCycle)
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
    func test_replayParams_carryTheSameTrainerConfigurationAsTheGUIPath() throws {
        configureActiveCycle()
        let snapshot = TrainingParameters.shared.snapshot()
        XCTAssertEqual(try ReplayParams(snapshot).trainer, TrainerHyperparameters(snapshot))
    }

    // MARK: - Applied to a trainer

    func test_activeCycle_drivesTheTrainersEffectiveLRAndMomentum() throws {
        configureActiveCycle()
        let trainer = try ChessTrainer(
            dropoutStream: DCMRandom(seed: 1),
            hyperparameters: TrainerHyperparameters(TrainingParameters.shared.snapshot()),
            arch: .current,
            initialization: .seeded(initSeed: 1)
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
            dropoutStream: DCMRandom(seed: 1),
            hyperparameters: TrainerHyperparameters(TrainingParameters.shared.snapshot()),
            arch: .current,
            initialization: .seeded(initSeed: 1)
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
    ///
    /// LR keyframes: the decay that was `lr_cycle_decay_horizon_steps` 5000,
    /// `lr_cycle_peak_end` 0.01, `lr_cycle_trough_end` 0.0001 after a
    /// 100-step warmup is now written as its exact keyframe form (start values
    /// held through warmup, end values at warmup + horizon; follow bounds the
    /// previous defaults).
    func test_cliBuiltAndGUIConfiguredTrainers_matchStepForStep() throws {
        configureActiveCycle()
        let p = TrainingParameters.shared
        p.lrWarmupSteps = 100
        p.sqrtBatchScalingLR = true
        p.lrCycleKeyframes = LRKeyframeSchedule(keyframes: [
            LRKeyframe(step: 0, peak: 0.1, trough: 0.001, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
            LRKeyframe(step: 100, peak: 0.1, trough: 0.001, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
            LRKeyframe(step: 5100, peak: 0.01, trough: 0.0001, momentumAtPeak: 0.90, momentumAtTrough: 0.95),
        ])
        XCTAssertEqual(p.lrCycleKeyframes.keyframes.count, 3, "the keyframes must have been accepted")
        p.momentumFollowsLRCycle = true
        p.dropoutRate = 0.1
        p.klProbeInterval = 100
        let snapshot = p.snapshot()

        let cliTrainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), hyperparameters: try ReplayParams(snapshot).trainer, arch: .current, initialization: .seeded(initSeed: 1))
        let guiTrainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), arch: .current, initialization: .seeded(initSeed: 2))
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

    // MARK: - LR keyframes: one validator for every writer

    func test_invalidKeyframes_errorDescriptionIsTheValidatorsMessage() {
        let error: Error = TrainingConfigError.invalidValue(
            id: LRCycleKeyframes.id,
            reason: LRKeyframeScheduleError.peakBelowTrough(index: 0, peak: 0.01, trough: 0.1).description
        )
        XCTAssertEqual(
            error.localizedDescription,
            "Invalid value for parameter 'lr_cycle_keyframes': keyframe 0: peak 0.01 is below trough 0.1"
        )
    }

    /// The singleton's setter refuses a schedule its validator refuses, like
    /// an out-of-range number, and keeps the previous schedule.
    func test_singletonSetter_revertsAnInvalidKeyframeSchedule() {
        let p = TrainingParameters.shared
        p.lrCycleKeyframes = Self.constantKeyframes
        XCTAssertEqual(p.lrCycleKeyframes, Self.constantKeyframes)
        p.lrCycleKeyframes = LRKeyframeSchedule(keyframes: [
            LRKeyframe(step: 0, peak: 0.01, trough: 0.1, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
        ])
        XCTAssertEqual(p.lrCycleKeyframes, Self.constantKeyframes, "peak below trough must be refused")
        p.lrCycleKeyframes = LRKeyframeSchedule(keyframes: [
            LRKeyframe(step: 0, peak: 0.1, trough: 0.001, momentumAtPeak: 0.95, momentumAtTrough: 0.85),
        ])
        XCTAssertEqual(p.lrCycleKeyframes, Self.constantKeyframes, "momentum at the peak above momentum at the trough must be refused")
        p.lrCycleKeyframes = LRKeyframeSchedule(keyframes: [])
        XCTAssertEqual(p.lrCycleKeyframes, Self.constantKeyframes, "an empty schedule must be refused")
    }

    /// A schedule written to `UserDefaults` by the one writer (`store`) reads
    /// back as the same schedule through the one reader (`inspectStored`).
    func test_storedKeyframes_roundTripThroughUserDefaults() throws {
        let defaults = try makeTemporaryDefaults()
        let schedule = LRKeyframeSchedule(keyframes: [
            LRKeyframe(step: 0, peak: 0.1, trough: 0.001, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
            LRKeyframe(step: 1000, peak: 0.1, trough: 0.001, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
            LRKeyframe(step: 1_001_000, peak: 1.0e-4, trough: 1.0e-6, momentumAtPeak: 0.90, momentumAtTrough: 0.95),
        ])
        guard case .absent = TrainingParameters.inspectStored(LRCycleKeyframes.self, in: defaults) else {
            return XCTFail("a new suite must hold no keyframes")
        }
        TrainingParameters.store(.lrKeyframes(schedule), forKey: LRCycleKeyframes.id, in: defaults)
        switch TrainingParameters.inspectStored(LRCycleKeyframes.self, in: defaults) {
        case .valid(let read):
            XCTAssertEqual(read, schedule)
        case .absent:
            XCTFail("the stored schedule was not found")
        case .invalid(let finding):
            XCTFail("the stored schedule read back as invalid: \(finding.problem)")
        }
    }

    /// A stored schedule the validator refuses is reported, with the
    /// validator's message, not used.
    func test_storedInvalidKeyframes_areReportedWithTheValidatorsMessage() throws {
        let defaults = try makeTemporaryDefaults()
        TrainingParameters.store(.lrKeyframes(LRKeyframeSchedule(keyframes: [
            LRKeyframe(step: 0, peak: 0.01, trough: 0.1, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
        ])), forKey: LRCycleKeyframes.id, in: defaults)
        guard case .invalid(let finding) = TrainingParameters.inspectStored(LRCycleKeyframes.self, in: defaults) else {
            return XCTFail("a peak below its trough must not read back as usable")
        }
        XCTAssertEqual(finding.id, LRCycleKeyframes.id)
        XCTAssertTrue(finding.problem.contains("keyframe 0: peak 0.01 is below trough 0.1"), finding.problem)
    }

    // MARK: - Owner decision Q-6: stored settings from before LR keyframes

    /// The previous schedule's nine settings, as the previous build stored
    /// them: a decay from 0.05 / 0.002 to 0.0005 / 0.00001 over 20,000 cycle
    /// steps, with momentum-follow bounds drifting from 0.8 / 0.95 to
    /// 0.88 / 0.97.
    private static let previousScheduleSettings: [String: Any] = [
        "lr_cycle_min": 0.002,
        "lr_cycle_max": 0.05,
        "lr_cycle_peak_end": 0.0005,
        "lr_cycle_trough_end": 0.00001,
        "lr_cycle_decay_horizon_steps": 20_000,
        "momentum_follow_start_low": 0.8,
        "momentum_follow_start_high": 0.95,
        "momentum_follow_end_low": 0.88,
        "momentum_follow_end_high": 0.97,
    ]

    /// Kept cycle settings stored beside the previous ones, so the conversion
    /// reads them from the suite and not from the declared defaults.
    private func storeKeptCycleSettings(in defaults: UserDefaults, lrWarmupSteps: Int) {
        TrainingParameters.store(.int(lrWarmupSteps), forKey: LRWarmupSteps.id, in: defaults)
        TrainingParameters.store(.bool(true), forKey: LRCycleEnabled.id, in: defaults)
        TrainingParameters.store(.int(1000), forKey: LRCyclePeriodSteps.id, in: defaults)
        TrainingParameters.store(.int(0), forKey: LRCycleCount.id, in: defaults)
        TrainingParameters.store(.bool(false), forKey: LRCycleInvert.id, in: defaults)
        TrainingParameters.store(.bool(true), forKey: MomentumCycleEnabled.id, in: defaults)
        TrainingParameters.store(.bool(true), forKey: MomentumFollowsLRCycle.id, in: defaults)
    }

    func test_previousScheduleSettings_areConvertedToKeyframes_andLeftInPlace() throws {
        let defaults = try makeTemporaryDefaults()
        for (id, value) in Self.previousScheduleSettings {
            defaults.set(value, forKey: id)
        }
        storeKeptCycleSettings(in: defaults, lrWarmupSteps: 300)

        TrainingParameters.convertLegacyLRScheduleSettings(in: defaults)

        var reference = LegacyLRMomentumCycleReference.disabledReference
        reference.lrEnabled = true
        reference.lrPeriodSteps = 1000
        reference.lrMin = 0.002
        reference.lrMax = 0.05
        reference.momentumEnabled = true
        reference.envelope = LegacyLRMomentumCycleEnvelopeReference(
            lrPeakEnd: 0.0005, lrTroughEnd: 0.00001, decayHorizonSteps: 20_000, momentumFollowsLRCycle: true,
            momentumFollowStartLow: 0.8, momentumFollowStartHigh: 0.95,
            momentumFollowEndLow: 0.88, momentumFollowEndHigh: 0.97)
        let expected = try LRMomentumCycle.converted(from: reference, lrWarmupSteps: 300).lrKeyframes
        XCTAssertEqual(expected, LRKeyframeSchedule(keyframes: [
            LRKeyframe(step: 0, peak: 0.05, trough: 0.002, momentumAtPeak: 0.8, momentumAtTrough: 0.95),
            LRKeyframe(step: 300, peak: 0.05, trough: 0.002, momentumAtPeak: 0.8, momentumAtTrough: 0.95),
            LRKeyframe(step: 20_300, peak: 0.0005, trough: 0.00001, momentumAtPeak: 0.88, momentumAtTrough: 0.97),
        ]))
        switch TrainingParameters.inspectStored(LRCycleKeyframes.self, in: defaults) {
        case .valid(let stored):
            XCTAssertEqual(stored, expected)
        case .absent:
            XCTFail("the previous settings were not converted")
        case .invalid(let finding):
            XCTFail("the converted schedule was stored unusable: \(finding.problem)")
        }
        guard case .valid(let lrEnabled) = TrainingParameters.inspectStored(LRCycleEnabled.self, in: defaults) else {
            return XCTFail("lr_cycle_enabled must still be stored")
        }
        XCTAssertTrue(lrEnabled, "a usable cycle stays enabled")
        for (id, value) in Self.previousScheduleSettings {
            let stored = defaults.object(forKey: id) as? NSNumber
            XCTAssertNotNil(stored, "\(id) must be left in place")
            if let stored, let number = value as? NSNumber {
                XCTAssertEqual(stored, number, "\(id) must be left unchanged")
            }
        }
    }

    private func storedKeyframes(_ defaults: UserDefaults) -> LRKeyframeSchedule? {
        guard case .valid(let stored) = TrainingParameters.inspectStored(LRCycleKeyframes.self, in: defaults) else { return nil }
        return stored
    }

    /// The declared keyframe default is the previous defaults converted at
    /// the declared warmup, so a user who never stored a schedule key keeps
    /// the schedule the previous build ran.
    func test_declaredKeyframes_areThePreviousDefaultsAtTheDeclaredWarmup() throws {
        let previous = LegacyLRScheduleKeys.previousDefaults
        let converted = try LegacyLRScheduleKeys.converting(previous) { id in
            TrainingParameters.allKeys.first { $0.id == id }?.definition.defaultValue
        }
        XCTAssertEqual(converted.values[LRCycleKeyframes.id], .lrKeyframes(LRCycleKeyframes.declaredDefault))
    }

    /// With none of the nine keys stored but another warmup stored, the
    /// previous build ran its defaults at that warmup.
    func test_aStoredWarmupAlone_convertsThePreviousDefaultsAtThatWarmup() throws {
        let defaults = try makeTemporaryDefaults()
        storeKeptCycleSettings(in: defaults, lrWarmupSteps: 500)
        TrainingParameters.convertLegacyLRScheduleSettings(in: defaults)
        let stored = try XCTUnwrap(storedKeyframes(defaults), "a stored warmup that isn't the declared one converts")
        XCTAssertEqual(stored.keyframes.map(\.step), [0, 500, 1_000_500])
        XCTAssertEqual(stored.keyframes.map(\.peak), LRCycleKeyframes.declaredDefault.keyframes.map(\.peak))

        let atDeclared = try makeTemporaryDefaults()
        storeKeptCycleSettings(in: atDeclared, lrWarmupSteps: LRWarmupSteps.declaredDefault)
        TrainingParameters.convertLegacyLRScheduleSettings(in: atDeclared)
        XCTAssertNil(atDeclared.object(forKey: LRCycleKeyframes.id), "the declared default already is that schedule")
    }

    /// Once checked, never again: a warmup changed later, in this build,
    /// doesn't rewrite the keyframes as if the previous build had run it.
    func test_theConversionRunsOnlyUntilItHasChecked() throws {
        let defaults = try makeTemporaryDefaults()
        storeKeptCycleSettings(in: defaults, lrWarmupSteps: LRWarmupSteps.declaredDefault)
        TrainingParameters.convertLegacyLRScheduleSettings(in: defaults)
        XCTAssertTrue(defaults.bool(forKey: TrainingParameters.legacyLRScheduleCheckDoneKey), "nothing to convert is a finished check")
        TrainingParameters.store(.int(500), forKey: LRWarmupSteps.id, in: defaults)
        TrainingParameters.convertLegacyLRScheduleSettings(in: defaults)
        XCTAssertNil(defaults.object(forKey: LRCycleKeyframes.id), "a later warmup edit converts nothing")
    }

    func test_aWholeNumberStoredForARealKeyReadsAsThatNumber() throws {
        let defaults = try makeTemporaryDefaults()
        for (id, value) in Self.previousScheduleSettings {
            defaults.set(value, forKey: id)
        }
        defaults.set(1, forKey: "lr_cycle_max")   // an Int, as `defaults write -int` stores it
        storeKeptCycleSettings(in: defaults, lrWarmupSteps: 300)
        TrainingParameters.convertLegacyLRScheduleSettings(in: defaults)
        let stored = try XCTUnwrap(storedKeyframes(defaults))
        XCTAssertEqual(stored.keyframes.first?.peak, 1.0)
    }

    /// An unusable stored entry was read as its default by the previous
    /// build; it converts as that default and doesn't stop the others.
    func test_unusableStoredPreviousEntries_convertAsThePreviousDefaults() throws {
        let defaults = try makeTemporaryDefaults()
        for (id, value) in Self.previousScheduleSettings {
            defaults.set(value, forKey: id)
        }
        defaults.set(50.0, forKey: "lr_cycle_max")                    // outside the previous 1e-7...10
        defaults.set("oops", forKey: "momentum_follow_end_high")      // wrong type
        storeKeptCycleSettings(in: defaults, lrWarmupSteps: 300)
        TrainingParameters.convertLegacyLRScheduleSettings(in: defaults)
        let stored = try XCTUnwrap(storedKeyframes(defaults))
        XCTAssertEqual(stored.keyframes.first?.peak, 0.1, "lr_cycle_max read as its previous default")
        XCTAssertEqual(stored.keyframes.last?.momentumAtTrough, 0.95, "momentum_follow_end_high read as its previous default")
        guard case .valid(let lrEnabled) = TrainingParameters.inspectStored(LRCycleEnabled.self, in: defaults) else {
            return XCTFail("lr_cycle_enabled must still be stored")
        }
        XCTAssertTrue(lrEnabled, "the cycle the previous build ran stays on")
    }

    func test_previousScheduleSettings_beside_storedKeyframes_areLeftAlone() throws {
        let defaults = try makeTemporaryDefaults()
        for (id, value) in Self.previousScheduleSettings {
            defaults.set(value, forKey: id)
        }
        storeKeptCycleSettings(in: defaults, lrWarmupSteps: 300)
        TrainingParameters.store(.lrKeyframes(Self.constantKeyframes), forKey: LRCycleKeyframes.id, in: defaults)

        TrainingParameters.convertLegacyLRScheduleSettings(in: defaults)

        guard case .valid(let stored) = TrainingParameters.inspectStored(LRCycleKeyframes.self, in: defaults) else {
            return XCTFail("the stored keyframes must still be there and usable")
        }
        XCTAssertEqual(stored, Self.constantKeyframes, "stored keyframes win; the previous settings are not converted over them")
        for (id, value) in Self.previousScheduleSettings {
            let stored = defaults.object(forKey: id) as? NSNumber
            XCTAssertNotNil(stored, "\(id) must be left in place")
            if let stored, let number = value as? NSNumber {
                XCTAssertEqual(stored, number, "\(id) must be left unchanged")
            }
        }
    }
}
