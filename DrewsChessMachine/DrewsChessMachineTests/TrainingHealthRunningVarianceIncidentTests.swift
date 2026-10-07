import XCTest
@testable import DrewsChessMachine

/// Rule 14 (`bn_running_variance_jump`) on real running variances
/// (BN_RUNNING_VARIANCE_CHANGE_ALARM_PLAN, Part X3): every BN site's
/// `running_var` of the evidence runs' checkpoints, read-only by
/// `experiments/20261005-lr-schedule-ab/bn_liveness.py
/// --write-running-variance-fixture`, fed in trainer-step order through
/// `LayerHealth.runningVarianceRatios` into live digests and the real
/// evaluator (learning gate 2,000, as every evidence run).
///
/// The checkpoints are 1,000 trainer steps apart, so these tests pin the
/// rule's behavior at checkpoint spacing; the app evaluates every 50 steps
/// (validation runs V-1 … V-4 measure that). The expected steps are the
/// plan's; a difference is reported to the owner, never absorbed by changing
/// an expectation.
final class TrainingHealthRunningVarianceIncidentTests: XCTestCase {

    typealias S = TrainingHealthTestSupport
    private let rule = TrainingHealthRule.batchNormRunningVarianceJump

    private struct Fixture: Decodable {
        struct Run: Decodable {
            let run: String
            let checkpoints: [Checkpoint]
        }

        struct Checkpoint: Decodable {
            let file: String
            let fileSHA256: String
            let cumTrainerStep: Int
            let sites: [Site]

            enum CodingKeys: String, CodingKey {
                case file, sites
                case fileSHA256 = "file_sha256"
                case cumTrainerStep = "cum_trainer_step"
            }
        }

        struct Site: Decodable {
            let site: String
            let runningVariance: [Double]

            enum CodingKeys: String, CodingKey {
                case site
                case runningVariance = "running_var"
            }
        }

        let runs: [Run]
    }

    private static let fixtureRunLabels = [
        "B-silu", "B-silu clip 1.0 (from 18k)", "B-silu clip 2.0 (from 18k)", "B-silu clip 5.0 (from 18k)",
        "B-leaky (value head)", "B-leakyall", "B (ReLU)", "A (ReLU, const 0.01)", "AgG3 (GUI, SiLU)",
    ]

    private func fixture() throws -> Fixture {
        try JSONDecoder().decode(
            Fixture.self, from: try S.resourceData("TrainingHealthRunningVarianceCheckpoints", extension: "json"))
    }

    /// The live digest of one checkpoint: every site's ratios from the one
    /// ratio definition. The fixture's numbers are the shortest decimals of
    /// the stored float32 values, so `Float(_:)` restores them exactly.
    private func digest(_ checkpoint: Fixture.Checkpoint) -> LayerHealthDigest {
        let profile = BatchNormRunningVarianceProfile(sites: checkpoint.sites.map { site in
            BatchNormRunningVarianceProfile.Site(
                site: site.site, channelCount: site.runningVariance.count,
                ratios: LayerHealth.runningVarianceRatios(site.runningVariance.map { Float($0) }).ratios)
        })
        return LayerHealthDigest(
            tier: .live, deadChannels: nil, nonFiniteValueCount: nil, runningVariance: nil, valueFC1: nil,
            runningVarianceChannels: LayerHealthDigest.RunningVarianceChannels(profile: profile))
    }

    /// Rule 14's raise / escalate / worsen / clear events of one run, its
    /// checkpoints evaluated in step order by one evaluator.
    private func replay(_ label: String) throws -> [TrainingHealthEvent] {
        let run = try XCTUnwrap(try fixture().runs.first { $0.run == label }, "no fixture run \(label)")
        let config = try S.config(checkIntervalSteps: 1000, learningGraceSteps: 1000, lrWarmupSteps: 1000)
        var evaluator = TrainingHealthEvaluator(valueFC1Applicability: .applies)
        var events: [TrainingHealthEvent] = []
        for checkpoint in run.checkpoints {
            let observation = TrainingHealthObservation(
                trainerStep: checkpoint.cumTrainerStep, window: S.window([]), layerHealth: digest(checkpoint),
                stamp: S.stamp(), effectiveLearningRate: nil, effectiveMomentum: nil)
            let evaluation = evaluator.evaluate(observation, config: config)
            XCTAssertFalse(evaluation.noDataRules.contains(rule), "\(label) \(checkpoint.cumTrainerStep)")
            events += evaluation.events.filter { $0.rule == rule && $0.kind != .active }
        }
        return events
    }

    private func describe(_ events: [TrainingHealthEvent]) -> String {
        events.map { "\($0.trainerStep) \($0.kind.rawValue) \($0.severity.rawValue) \($0.value) \($0.threshold) \($0.detail)" }
            .joined(separator: "\n")
    }

    private func steps(_ events: [TrainingHealthEvent], _ kind: TrainingHealthEvent.Kind) -> [Int] {
        events.filter { $0.kind == kind }.map(\.trainerStep)
    }

    // MARK: Fixture

    func testFixtureHoldsEveryEvidenceRunInStepOrder() throws {
        let data = try fixture()
        XCTAssertEqual(data.runs.map(\.run), Self.fixtureRunLabels)
        for run in data.runs {
            let checkpointSteps = run.checkpoints.map(\.cumTrainerStep)
            XCTAssertEqual(checkpointSteps, checkpointSteps.sorted(), run.run)
            XCTAssertEqual(Set(checkpointSteps).count, checkpointSteps.count, run.run)
            for checkpoint in run.checkpoints {
                XCTAssertEqual(checkpoint.fileSHA256.count, 64, checkpoint.file)
                XCTAssertEqual(checkpoint.sites.first?.site, "stem.bn", checkpoint.file)
                XCTAssertEqual(checkpoint.sites.last?.site, "value.bn", checkpoint.file)
            }
        }
        let bSilu = try XCTUnwrap(data.runs.first { $0.run == "B-silu" })
        XCTAssertEqual(bSilu.checkpoints.map(\.cumTrainerStep), Array(stride(from: 2000, through: 22_000, by: 1000)))
    }

    // MARK: B-silu

    /// The incident: critical at 20,000 (the first checkpoint past the
    /// jump; in-app at 50-step resolution it is 19,800), naming channel 76
    /// and 34 of `blocks.2.bn1`, with the outlier count 5 → 11; nothing
    /// from 2,000 to 19,000; `worsen` at 21,000 (11 → 53).
    func testBSiluRaisesCriticalAt20000AndWorsensAt21000() throws {
        let events = try replay("B-silu")
        let raise = try XCTUnwrap(events.first, describe(events))
        XCTAssertEqual(raise.kind, .raise, describe(events))
        XCTAssertEqual(raise.trainerStep, 20_000, describe(events))
        XCTAssertEqual(raise.severity, .critical)
        XCTAssertTrue(raise.detail.contains("blocks.2.bn1[76]:0.02->980.1"), raise.detail)
        XCTAssertTrue(raise.detail.contains("blocks.2.bn1[34]:"), raise.detail)
        XCTAssertEqual(raise.threshold, "jump>=10xmin&ratio>=100|outliers>=2xmin&+5")
        XCTAssertTrue(raise.value.hasSuffix(" outliers=11/5"), raise.value)
        XCTAssertTrue(events.filter { $0.trainerStep < 20_000 }.isEmpty, describe(events))
        let worsen = try XCTUnwrap(events.first { $0.kind == .worsen }, describe(events))
        XCTAssertEqual(worsen.trainerStep, 21_000)
        XCTAssertTrue(worsen.value.hasSuffix(" outliers=53/11"), worsen.value)
        XCTAssertTrue(worsen.detail.hasPrefix("was=11 "), worsen.detail)
    }

    // MARK: Runs that survived or stayed healthy

    /// clip1 (cap 1.0): one warning at 20,000 for channel 76's reactivation
    /// (0.02 → 38.9), never critical; channel 31's slow growth to 154× never
    /// fires, nor does the count's +5 at ×1.36 at 32,000.
    func testClip1WarnsOnceAt20000AndNeverGoesCritical() throws {
        let events = try replay("B-silu clip 1.0 (from 18k)")
        XCTAssertEqual(steps(events, .raise), [20_000], describe(events))
        let raise = try XCTUnwrap(events.first { $0.kind == .raise })
        XCTAssertEqual(raise.severity, .warning)
        XCTAssertEqual(raise.detail, "channels=blocks.2.bn1[76]:0.02->38.9")
        XCTAssertTrue(events.filter { $0.severity == .critical }.isEmpty, describe(events))
        XCTAssertTrue(events.filter { $0.detail.contains("[31]:") }.isEmpty, describe(events))
    }

    func testClip2AndClip5RaiseCriticalAt20000() throws {
        for (label, level) in [("B-silu clip 2.0 (from 18k)", "->129.5"), ("B-silu clip 5.0 (from 18k)", "->513.1")] {
            let events = try replay(label)
            let raise = try XCTUnwrap(events.first { $0.kind == .raise }, "\(label)\n\(describe(events))")
            XCTAssertEqual(raise.trainerStep, 20_000, label)
            XCTAssertEqual(raise.severity, .critical, label)
            XCTAssertTrue(raise.detail.contains("blocks.2.bn1[76]:0.02\(level)"), "\(label): \(raise.detail)")
        }
    }

    func testBLeakyWarnsAt21000Only() throws {
        let events = try replay("B-leaky (value head)")
        XCTAssertEqual(steps(events, .raise), [21_000], describe(events))
        XCTAssertTrue(events.allSatisfy { $0.severity == .warning }, describe(events))
        XCTAssertTrue(events.first?.detail.contains("blocks.2.bn1[76]:0.11->35.0") == true, describe(events))
    }

    /// Two contiguous spans: each warning clears within its own span, so a
    /// hold across the gap does not keep the first alarm active.
    func testBLeakyallWarnsAndClearsInEachSpan() throws {
        let events = try replay("B-leakyall")
        XCTAssertEqual(steps(events, .raise), [13_000, 30_000], describe(events))
        XCTAssertEqual(steps(events, .clear), [15_000, 32_000], describe(events))
        XCTAssertTrue(events.allSatisfy { $0.severity == .warning }, describe(events))
    }

    /// B's 1,000 → 2,000 jump (`stem.bn[102]`) is before the gate; its count
    /// rises are at most +4.
    func testBRaisesNothing() throws {
        let events = try replay("B (ReLU)")
        XCTAssertTrue(events.isEmpty, describe(events))
    }

    func testARaisesNothing() throws {
        let events = try replay("A (ReLU, const 0.01)")
        XCTAssertTrue(events.isEmpty, describe(events))
    }

    /// AgG3's trainer saves: 1,607 is before the gate (not stored), 2,831 and
    /// 4,655 have no read in their lookback, so 5,261 against 4,655 is the
    /// one judged read — and channel 88's slow growth does not jump.
    func testAgG3RaisesNothing() throws {
        let events = try replay("AgG3 (GUI, SiLU)")
        XCTAssertTrue(events.isEmpty, describe(events))
    }
}
