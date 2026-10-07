import Metal
import XCTest
@testable import DrewsChessMachine

/// The parts of the alarms plan's V-6 (GUI) the other in-process GUI tests
/// do not cover, without launching the app (team-lead decision 2026-10-07,
/// owner delegation: no app launch, focus rule): the trainer worker's
/// evaluation and value-FC1 cadence, the Health tab's settings and their
/// commit, and the alarm row's VoiceOver label. Settings are restored with
/// persistence suppressed.
@MainActor
final class TrainingHealthGuiValidationTests: XCTestCase {

    typealias S = TrainingHealthTestSupport

    private var savedParameterValues: [String: ParameterValue] = [:]

    override func setUp() async throws {
        try await super.setUp()
        savedParameterValues = TrainingParameters.shared.snapshot().rawValueMap()
        TrainingParameters.suppressPersistence = true
    }

    override func tearDown() async throws {
        TrainingParameters.suppressPersistence = true
        try TrainingParameters.shared.apply(savedParameterValues)
        TrainingParameters.suppressPersistence = false
        try await super.tearDown()
    }

    // MARK: The trainer worker's cadence

    /// V-6: a live evaluation every 50 trainer steps and one dedicated
    /// value-FC1 read every 1,000 trainer steps from the session's start
    /// (no save in the run).
    func testWorkerEvaluatesEveryFiftyStepsAndReadsValueFC1EveryThousand() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
        var arch = NetworkArchitecture.current
        arch.blockGroups = [
            BlockGroup(
                count: 1, channels: 16, conv1KernelSize: 3, conv2KernelSize: 3,
                seStyle: .none, seReductionRatio: 4, useRezero: true, rezeroAlphaInit: 0.5,
                activationFunction: .relu, activationStyle: .pre, skipMerge: .cleanAdd, dropoutMultiplier: 0)
        ]
        arch.valueHeadConvChannels = 4
        arch.valueHeadHiddenUnits = 8
        arch.valueHeadFC1HiddenActivation = .relu
        let trainer = try ChessTrainer(
            dropoutStream: DCMRandom(seed: 3), momentumCoeff: 0.9, lrWarmupSteps: 0, arch: arch,
            initialization: .seeded(initSeed: 3))
        let monitor = TrainingHealthMonitor(valueFC1Applicability: .applies, stopDecision: .byCaller)
        let config = try S.config()
        let deliveries = SyncBox(0)
        let worker = GuiTrainingHealthWorker(
            monitor: monitor, trainer: trainer, batchSize: 16, recorder: nil,
            resolveConfig: { config },
            deliver: { _ in deliveries.modify { $0 += 1 } })
        for step in 1...2000 {
            // The worker's caller passes the trainer's own clock; the
            // synthetic records stand for real steps, so advance the clock
            // the reads stamp their observations with to match.
            trainer.completedTrainSteps = step
            await worker.afterStep(S.timing(), trainerStep: step)
        }
        let summary = monitor.segmentSummary()
        XCTAssertEqual(summary.evaluations, 40 + 2, "40 live evaluations and value-FC1 reads at 1,000 and 2,000")
        XCTAssertEqual(deliveries.value, 42, "every committed evaluation is delivered to the main actor")
        // The trainer never took a real step, so its velocity is zero: the
        // reads at 1,000 and 2,000 (1,000 and 2,000 steps recorded by this
        // monitor, past rule 3's 200-step gate) judge it, and nothing else
        // raises on the healthy records.
        XCTAssertEqual(summary.raised.map(\.rule), [.valueFC1ZeroVelocity])
        XCTAssertEqual(summary.raised.first?.firstTrainerStep, 1000)
        XCTAssertFalse(worker.parkRequested)
        monitor.requestPark()
        XCTAssertTrue(worker.parkRequested, "the worker sees its own monitor's park request")
    }

    // MARK: The Health tab

    func testHealthTabShowsEveryRuleWithItsOwnName() {
        let names = TrainingHealthRule.allCases.map(\.displayName)
        XCTAssertEqual(Set(names).count, names.count)
        for rule in TrainingHealthRule.allCases {
            XCTAssertFalse(rule.meaning.isEmpty, rule.rawValue)
        }
        // Rules without a critical level cannot stop on critical (the
        // picker disables that choice).
        XCTAssertEqual(
            TrainingHealthRule.allCases.filter { !$0.hasCriticalLevel },
            [.lossSpike, .policyOffsetDrift, .batchNormRunningVarianceRunaway, .gradientSpike])
    }

    func testHealthTabSaveCommitsTheSettings() {
        let p = TrainingParameters.shared
        p.trainingHealthAlarmsEnabled = true
        p.trainingHealthCheckIntervalSteps = 1000
        p.trainingHealthLearningGraceSteps = 1000
        p.trainingHealthActionDeadChannels = .log
        let model = TrainingSettingsPopoverModel(selfPlayDelayMaxMs: 1000, stepDelayMaxMs: 1000, maxSelfPlayWorkers: 8)
        model.seedFromParams()
        XCTAssertEqual(model.trainingHealthCheckIntervalText, "1000")
        model.trainingHealthActionsValue[.deadChannels] = .stopOnCritical
        model.trainingHealthCheckIntervalText = "500"
        model.trainingHealthLearningGraceText = "250"
        model.trainingHealthAlarmsEnabledValue = false
        model.save()
        XCTAssertEqual(p.trainingHealthActionDeadChannels, .stopOnCritical)
        XCTAssertEqual(p.trainingHealthCheckIntervalSteps, 500)
        XCTAssertEqual(p.trainingHealthLearningGraceSteps, 250)
        XCTAssertFalse(p.trainingHealthAlarmsEnabled)
        XCTAssertFalse(model.trainingHealthCheckIntervalError)
    }

    func testHealthTabRefusesAnOutOfRangeInterval() {
        let p = TrainingParameters.shared
        p.trainingHealthCheckIntervalSteps = 1000
        let model = TrainingSettingsPopoverModel(selfPlayDelayMaxMs: 1000, stepDelayMaxMs: 1000, maxSelfPlayWorkers: 8)
        model.seedFromParams()
        model.trainingHealthCheckIntervalText = "10"
        model.save()
        XCTAssertTrue(model.trainingHealthCheckIntervalError)
        XCTAssertEqual(p.trainingHealthCheckIntervalSteps, 1000, "an invalid field leaves the setting alone")
    }

    // MARK: The alarm row

    func testAlarmRowVoiceOverLabel() {
        let alarm = TrainingHealthActiveAlarm(
            rule: .deadChannels, severity: .critical, since: 514, value: "dead=339/1040", detail: "", action: .log)
        XCTAssertEqual(
            TrainingHealthAlarmRow.accessibilityText(alarm: alarm, stops: true),
            "Critical: Dead channels, dead=339/1040, since trainer step 514, stops the run")
        XCTAssertEqual(
            TrainingHealthAlarmRow.accessibilityText(alarm: alarm, stops: false),
            "Critical: Dead channels, dead=339/1040, since trainer step 514, logged only")
    }
}
