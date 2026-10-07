import XCTest
@testable import DrewsChessMachine

/// The lock-protected monitor: windows, references, stamps and generations,
/// rewinds (announced and not), stale observations, the cap, the check
/// lines, the value-FC1 read schedule (D6), the segment summary and the
/// concurrency guarantees (the alarms plan, D2 and X1).
final class TrainingHealthMonitorTests: XCTestCase {

    typealias S = TrainingHealthTestSupport

    private func monitor(_ applicability: TrainingHealthValueFC1Applicability = .applies) -> TrainingHealthMonitor {
        TrainingHealthMonitor(valueFC1Applicability: applicability)
    }

    private func record(_ monitor: TrainingHealthMonitor, _ steps: ClosedRange<Int>, loss: Float = 4.0, gradient: Float = 1.0, illegal: Float = 0.01) {
        for step in steps {
            monitor.recordStep(S.record(step, loss: loss, illegal: illegal, gradient: gradient, ms: 1))
        }
    }

    @discardableResult
    private func live(
        _ monitor: TrainingHealthMonitor,
        at step: Int,
        digest: LayerHealthDigest = S.deadDigest(dead: 0),
        config: TrainingHealthConfig,
        sink: S.LineSink
    ) -> TrainingHealthEvaluation? {
        let stamp = monitor.observationStamp()
        return monitor.evaluateLive(
            stamp: stamp, layerHealth: .read(digest, trainerStep: step), learningRate: 0.1, momentum: 0.85,
            config: config, log: sink.sink)
    }

    @discardableResult
    private func checkpoint(
        _ monitor: TrainingHealthMonitor,
        at step: Int,
        digest: LayerHealthDigest,
        stamp: TrainingHealthStamp? = nil,
        config: TrainingHealthConfig,
        sink: S.LineSink
    ) -> TrainingHealthEvaluation? {
        monitor.evaluateCheckpoint(
            stamp: stamp ?? monitor.observationStamp(), layerHealth: digest, digestTrainerStep: step,
            config: config, log: sink.sink)
    }

    // MARK: Windows

    func testWindowMediansAndMaxima() throws {
        let statistics = S.window([
            S.record(1, loss: 3, illegal: 0.5, gradient: 2, offset: nil),
            S.record(2, loss: 9, illegal: 0.1, gradient: 1, offset: -4),
            S.record(3, loss: 5, illegal: 0.2, gradient: 7, offset: 2),
            S.record(4, loss: 4, illegal: 0.3, gradient: 3, offset: nil),
        ])
        XCTAssertEqual(statistics.recordCount, 4)
        XCTAssertEqual(statistics.firstTrainerStep, 1)
        XCTAssertEqual(statistics.lastTrainerStep, 4)
        XCTAssertEqual(statistics.lossMedian, 4.5)
        XCTAssertEqual(statistics.lossMax, 9)
        XCTAssertEqual(try XCTUnwrap(statistics.illegalMassMedian), 0.25, accuracy: 1e-6)
        XCTAssertEqual(statistics.gradientNormMedian, 2.5)
        XCTAssertEqual(statistics.gradientNormMax, 7)
        XCTAssertEqual(statistics.diagnosticRecordCount, 2)
        XCTAssertEqual(statistics.policyLogitMeanAbsMedian, 3)
        XCTAssertEqual(statistics.nonFiniteValueCount, 0)
    }

    func testDiagnosticFieldsOnlyFromDiagnosticSteps() throws {
        let lean = TrainingHealthStepRecord(timing: S.timing(hasDiagnostics: false, policyLogitMean: -5), trainerStep: 10)
        XCTAssertNil(lean.policyLogitMean, "a non-diagnostic step's placeholder is not a measurement")
        let diagnostic = TrainingHealthStepRecord(timing: S.timing(hasDiagnostics: true, policyLogitMean: -5), trainerStep: 10)
        XCTAssertEqual(diagnostic.policyLogitMean, -5)
        XCTAssertEqual(lean.loss, 4)
        XCTAssertEqual(lean.illegalMassPenalty, 0.01)
        XCTAssertEqual(lean.gradGlobalNorm, 1)

        let m = monitor()
        m.recordStep(S.timing(hasDiagnostics: false, policyLogitMean: -5), trainerStep: 1)
        m.recordStep(S.timing(hasDiagnostics: true, policyLogitMean: -6), trainerStep: 2)
        let result = try XCTUnwrap(live(m, at: 2, config: try S.config(), sink: S.LineSink()))
        XCTAssertEqual(result.window?.diagnosticRecordCount, 1)
        XCTAssertEqual(result.window?.policyLogitMeanAbsMedian, 6)
    }

    func testWindowStopsAtTheDigestsTrainerStep() throws {
        let m = monitor()
        let config = try S.config()
        let sink = S.LineSink()
        record(m, 1...50)
        let stamp = m.observationStamp()
        // The worker records more steps after the live read (digest at 50).
        record(m, 51...60)
        let first = try XCTUnwrap(m.evaluateLive(
            stamp: stamp, layerHealth: .read(S.deadDigest(dead: 0), trainerStep: 50), learningRate: nil, momentum: nil,
            config: config, log: sink.sink))
        XCTAssertEqual(first.window?.recordCount, 50)
        XCTAssertEqual(first.window?.lastTrainerStep, 50)
        let second = try XCTUnwrap(live(m, at: 60, config: config, sink: sink))
        XCTAssertEqual(second.window?.recordCount, 10)
        XCTAssertEqual(second.window?.firstTrainerStep, 51)
    }

    func testLiveReadFailedUsesTheLastRecordedStepAndSaysSo() throws {
        let m = monitor()
        let sink = S.LineSink()
        record(m, 1...40)
        let result = try XCTUnwrap(m.evaluateLive(
            stamp: m.observationStamp(), layerHealth: .readFailed, learningRate: nil, momentum: nil,
            config: try S.config(), log: sink.sink))
        XCTAssertEqual(result.window?.recordCount, 40)
        XCTAssertTrue(result.noDataRules.contains(.deadChannels))
        XCTAssertTrue(sink.texts.contains(
            "[HEALTH] live read failed at trainerStep=40; window boundary = last recorded step 40; layer-health rules no data"))
    }

    func testRingCapacityDropsOldest() {
        var pending = TrainingHealthMonitor.PendingRecords()
        let capacity = 5000
        for step in 1...(capacity + 10) {
            pending.append(S.record(step))
            if pending.count > capacity { pending.dropOldest() }
        }
        XCTAssertEqual(pending.count, capacity)
        let taken = pending.take(throughTrainerStep: capacity + 10)
        XCTAssertEqual(taken.first?.trainerStep, 11)
        XCTAssertEqual(taken.last?.trainerStep, capacity + 10)
        XCTAssertEqual(pending.count, 0)
    }

    func testPendingWindowCapCountsTruncation() throws {
        let m = monitor()
        let sink = S.LineSink()
        let total = TrainingHealthThresholds.pendingWindowCapacity + 10
        record(m, 1...total)
        let result = try XCTUnwrap(live(m, at: total, config: try S.config(), sink: sink))
        XCTAssertEqual(result.window?.recordCount, TrainingHealthThresholds.pendingWindowCapacity)
        XCTAssertEqual(result.window?.firstTrainerStep, 11)
        let check = try XCTUnwrap(sink.texts.first { $0.hasPrefix("[HEALTH] check ") })
        XCTAssertTrue(check.contains(" truncated=10 "), check)
    }

    func testLossReferenceNeedsMinimumHistory() {
        func entries(_ steps: [Int]) -> [(trainerStep: Int, value: Float?)] { steps.map { ($0, 4.0) } }
        XCTAssertNil(TrainingHealthReference.make(entries([100, 200, 300, 400]), windowStart: 500), "4 records")
        XCTAssertNil(TrainingHealthReference.make(entries([300, 350, 400, 450, 499]), windowStart: 500), "span 199")
        let reference = TrainingHealthReference.make(entries([299, 350, 400, 450, 499]), windowStart: 500)
        XCTAssertEqual(reference?.median, 4)
        XCTAssertEqual(reference?.spanSteps, 200)
        XCTAssertNil(TrainingHealthReference.make(entries([1, 50, 100, 150, 200, 250]), windowStart: 1251), "outside the look-back")
        XCTAssertNil(TrainingHealthReference.make([(1, nil), (50, nil), (100, nil), (150, nil), (250, nil)], windowStart: 300), "not measured")
    }

    func testLossReferenceSurvivesALongWindow() throws {
        let m = monitor()
        let config = try S.config()
        let sink = S.LineSink()
        record(m, 1...1000)
        live(m, at: 1000, config: config, sink: sink)
        record(m, 1001...4000, loss: 7.0)
        let long = try XCTUnwrap(live(m, at: 4000, config: config, sink: sink))
        XCTAssertEqual(long.window?.recordCount, 3000)
        XCTAssertEqual(long.window?.lossReference?.median, 4.0)
        XCTAssertTrue(long.events.contains { $0.rule == .lossSpike && $0.kind == .raise })
        record(m, 4001...4050, loss: 7.0)
        let next = try XCTUnwrap(live(m, at: 4050, config: config, sink: sink))
        XCTAssertEqual(next.window?.lossReference?.median, 7.0, "the 1,000 steps before the window, from the long window")
        XCTAssertEqual(next.window?.lossReference?.recordCount, 1000)
    }

    // MARK: Rewinds and generations

    func testTrainerClockRewindResetsWindowsKeepsActiveAlarms() throws {
        let m = monitor()
        let config = try S.config()
        let sink = S.LineSink()
        record(m, 1...100, illegal: 0.2)
        live(m, at: 100, digest: S.deadDigest(dead: 5), config: config, sink: sink)
        XCTAssertEqual(m.activeAlarmsSnapshot().map(\.rule), [.deadChannels])
        // One evaluation into a gradient-collapse sustain and the illegal-mass
        // running minimum (0.2) both describe the pre-rewind weights.
        record(m, 101...150, gradient: 0.01, illegal: 0.2)
        live(m, at: 150, digest: S.deadDigest(dead: 5), config: config, sink: sink)
        record(m, 151...160)
        let before = m.observationStamp()
        XCTAssertEqual(before.stepsTrainedByThisProcess, 160)

        m.noteTrainerClockRewind(to: 120, log: sink.sink)

        let after = m.observationStamp()
        XCTAssertEqual(after.generation, before.generation + 1)
        XCTAssertEqual(after.stepsTrainedByThisProcess, 120)
        XCTAssertEqual(after.lastRecordedTrainerStep, 120)
        XCTAssertEqual(sink.texts.last, "[HEALTH] trainer clock rewound 160 -> 120; generation 1; windows reset")
        XCTAssertEqual(m.activeAlarmsSnapshot().map(\.rule), [.deadChannels], "active alarms clear only by recovery")

        // Pending window discarded: the next window holds only post-rewind steps.
        record(m, 121...170, gradient: 0.01, illegal: 0.9)
        let next = try XCTUnwrap(live(m, at: 170, digest: S.deadDigest(dead: 5), config: config, sink: sink))
        XCTAssertEqual(next.window?.firstTrainerStep, 121)
        XCTAssertFalse(next.events.contains { $0.rule == .gradientCollapse }, "the pre-rewind sustain was reset")
        XCTAssertFalse(next.events.contains { $0.rule == .illegalMass }, "the pre-rewind running minimum was reset")
        XCTAssertNil(next.window?.lossReference, "the reference history was discarded")
    }

    func testUnannouncedRewindCountsExactly() {
        let m = monitor()
        record(m, 1...100)
        m.recordStep(S.record(51))
        let stamp = m.observationStamp()
        XCTAssertEqual(stamp.stepsTrainedByThisProcess, 51)
        XCTAssertEqual(stamp.generation, 1)
        XCTAssertEqual(stamp.lastRecordedTrainerStep, 51)
    }

    func testAnnouncedRewindCountsExactly() {
        let m = monitor()
        record(m, 1...100)
        m.noteTrainerClockRewind(to: 50, log: S.LineSink().sink)
        XCTAssertEqual(m.observationStamp().stepsTrainedByThisProcess, 50)
        m.recordStep(S.record(51))
        XCTAssertEqual(m.observationStamp().stepsTrainedByThisProcess, 51)
        XCTAssertEqual(m.observationStamp().generation, 1, "a record after an announced rewind is not another rewind")
    }

    func testLiveObservationFromPreviousGenerationIsIgnored() throws {
        let m = monitor()
        let sink = S.LineSink()
        record(m, 1...100)
        let stale = m.observationStamp()
        m.noteTrainerClockRewind(to: 80, log: sink.sink)
        record(m, 81...90)
        let result = m.evaluateLive(
            stamp: stale, layerHealth: .read(S.deadDigest(dead: 500), trainerStep: 100), learningRate: nil,
            momentum: nil, config: try S.config(), log: sink.sink)
        XCTAssertNil(result)
        XCTAssertTrue(m.activeAlarmsSnapshot().isEmpty)
        XCTAssertTrue(sink.texts.contains { $0.hasPrefix("[HEALTH] stale live observation ignored: trainerStep=100 generation=0 (current generation 1") })
        // The pending post-rewind records stay for the next evaluation.
        let next = try XCTUnwrap(live(m, at: 90, config: try S.config(), sink: sink))
        XCTAssertEqual(next.window?.recordCount, 10)
    }

    func testCheckpointFromPreviousGenerationIsIgnored() throws {
        let m = monitor()
        let sink = S.LineSink()
        record(m, 1...1000)
        let exportStamp = m.observationStamp()
        m.noteTrainerClockRewind(to: 900, log: sink.sink)
        let result = checkpoint(m, at: 1000, digest: S.valueFC1Digest(zero: 128), stamp: exportStamp,
                                config: try S.config(), sink: sink)
        XCTAssertNil(result)
        XCTAssertTrue(m.activeAlarmsSnapshot().isEmpty)
        XCTAssertTrue(sink.texts.contains { $0.hasPrefix("[HEALTH] stale checkpoint observation ignored: trainerStep=1000 generation=0") })
    }

    func testCheckpointArrivingFirstAfterAnUnannouncedRewindIsRejected() throws {
        let m = monitor()
        let sink = S.LineSink()
        record(m, 1...1000)
        let exportStamp = m.observationStamp()
        m.recordStep(S.record(501))   // unannounced rewind 1000 -> 500
        let result = checkpoint(m, at: 1000, digest: S.valueFC1Digest(zero: 128), stamp: exportStamp,
                                config: try S.config(), sink: sink)
        XCTAssertNil(result)
        XCTAssertTrue(m.activeAlarmsSnapshot().isEmpty)
        XCTAssertEqual(sink.texts.first,
                       "[HEALTH] trainer clock rewind detected by recordStep (not announced): 1000 -> 500; generation 1")
        XCTAssertTrue(sink.texts.contains { $0.hasPrefix("[HEALTH] stale checkpoint observation ignored:") })
    }

    func testCheckpointEvaluationNeverConsumesTheWindowOrDetectsRewind() throws {
        let m = monitor()
        let config = try S.config()
        let sink = S.LineSink()
        record(m, 1...500)
        // An old checkpoint (trainer step 200) is not a rewind and leaves the window alone.
        let result = try XCTUnwrap(checkpoint(m, at: 200, digest: S.deadDigest(dead: 0, tier: .checkpoint), config: config, sink: sink))
        XCTAssertNil(result.window)
        XCTAssertEqual(m.observationStamp().generation, 0)
        let next = try XCTUnwrap(live(m, at: 500, config: config, sink: sink))
        XCTAssertEqual(next.window?.recordCount, 500)
    }

    func testOlderCheckpointArrivingLateIsIgnored() throws {
        let m = monitor()
        let config = try S.config()
        let sink = S.LineSink()
        record(m, 1...2000)
        let damaged = checkpoint(m, at: 2000, digest: S.valueFC1Digest(zero: 128), config: config, sink: sink)
        XCTAssertEqual(damaged?.events.map(\.kind), [.raise])
        let late = try XCTUnwrap(checkpoint(m, at: 1000, digest: S.valueFC1Digest(zero: 0), config: config, sink: sink))
        XCTAssertTrue(late.events.isEmpty)
        XCTAssertEqual(late.staleRules, [.valueFC1ZeroVelocity])
        XCTAssertEqual(m.activeAlarmsSnapshot().map(\.rule), [.valueFC1ZeroVelocity])
    }

    func testOlderCheckpointsCannotClearALiveRaise() throws {
        let m = monitor()
        let config = try S.config()
        let sink = S.LineSink()
        record(m, 1...3000)
        live(m, at: 3000, digest: S.deadDigest(dead: 5), config: config, sink: sink)
        checkpoint(m, at: 2000, digest: S.deadDigest(dead: 0, tier: .checkpoint), config: config, sink: sink)
        let second = try XCTUnwrap(checkpoint(m, at: 2500, digest: S.deadDigest(dead: 0, tier: .checkpoint), config: config, sink: sink))
        XCTAssertTrue(second.staleRules.contains(.deadChannels))
        XCTAssertEqual(m.activeAlarmsSnapshot().map(\.rule), [.deadChannels])
    }

    func testRule3UsesTheStampedTrainedCount() throws {
        let m = monitor()
        let sink = S.LineSink()
        record(m, 1...199)
        let exportStamp = m.observationStamp()
        record(m, 200...200)
        let result = try XCTUnwrap(checkpoint(m, at: 199, digest: S.valueFC1Digest(zero: 128), stamp: exportStamp,
                                              config: try S.config(), sink: sink))
        XCTAssertTrue(result.noDataRules.contains(.valueFC1ZeroVelocity))
        XCTAssertTrue(m.activeAlarmsSnapshot().isEmpty)
    }

    /// Pauses the first evaluation that reaches the commit point until the
    /// test lets it continue.
    private final class CommitGate: @unchecked Sendable {
        let reached = DispatchSemaphore(value: 0)
        let proceed = DispatchSemaphore(value: 0)
        private let armed = SyncBox(true)
        func hook() {
            let pause = armed.mutate { value -> Bool in
                defer { value = false }
                return value
            }
            guard pause else { return }
            reached.signal()
            proceed.wait()
        }
    }

    func testUnannouncedRewindDuringLiveEvaluationDiscardsIt() throws {
        let gate = CommitGate()
        let m = TrainingHealthMonitor(valueFC1Applicability: .applies, beforeCommitForTesting: { gate.hook() })
        let sink = S.LineSink()
        let config = try S.config(actions: TrainingHealthActions { _ in .stopOnAny })
        record(m, 1...100)
        let stamp = m.observationStamp()
        let outcome = SyncBox<TrainingHealthEvaluation?>(nil)
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            outcome.value = m.evaluateLive(
                stamp: stamp, layerHealth: .read(S.deadDigest(dead: 500), trainerStep: 100), learningRate: nil,
                momentum: nil, config: config, log: sink.sink)
            done.signal()
        }
        gate.reached.wait()
        m.recordStep(S.record(51))   // rewind while the evaluation is between validation and commit
        gate.proceed.signal()
        done.wait()
        XCTAssertNil(outcome.value)
        XCTAssertTrue(m.activeAlarmsSnapshot().isEmpty)
        XCTAssertFalse(m.stopRequested)
        XCTAssertFalse(sink.texts.contains { $0.hasPrefix("[ALARM] health") })
        XCTAssertTrue(sink.texts.contains { $0.contains("trainer clock rewound during the evaluation") })
    }

    func testUnannouncedRewindDuringCheckpointEvaluationDiscardsIt() throws {
        let gate = CommitGate()
        let m = TrainingHealthMonitor(valueFC1Applicability: .applies, beforeCommitForTesting: { gate.hook() })
        let sink = S.LineSink()
        let config = try S.config(actions: TrainingHealthActions { _ in .stopOnAny })
        record(m, 1...1000)
        let stamp = m.observationStamp()
        let outcome = SyncBox<TrainingHealthEvaluation?>(nil)
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            outcome.value = m.evaluateCheckpoint(
                stamp: stamp, layerHealth: S.valueFC1Digest(zero: 128), digestTrainerStep: 1000, config: config,
                log: sink.sink)
            done.signal()
        }
        gate.reached.wait()
        m.recordStep(S.record(501))
        gate.proceed.signal()
        done.wait()
        XCTAssertNil(outcome.value)
        XCTAssertTrue(m.activeAlarmsSnapshot().isEmpty)
        XCTAssertFalse(m.stopRequested)
        XCTAssertFalse(sink.texts.contains { $0.hasPrefix("[ALARM] health") })
    }

    func testConcurrentRewindAndEvaluation() throws {
        let m = monitor()
        let config = try S.config()
        let sink = S.LineSink()
        record(m, 1...5000)
        DispatchQueue.concurrentPerform(iterations: 200) { index in
            switch index % 4 {
            case 0:
                m.noteTrainerClockRewind(to: 4000 + index, log: sink.sink)
            case 1:
                let stamp = m.observationStamp()
                m.evaluateLive(stamp: stamp, layerHealth: .read(S.deadDigest(dead: 1), trainerStep: 5000),
                               learningRate: nil, momentum: nil, config: config, log: sink.sink)
            case 2:
                let stamp = m.observationStamp()
                m.evaluateCheckpoint(stamp: stamp, layerHealth: S.valueFC1Digest(zero: 1), digestTrainerStep: 5000,
                                     config: config, log: sink.sink)
            default:
                m.recordStep(S.record(5001 + index))
            }
        }
        // State is consistent: a fresh stamp evaluates normally afterwards.
        let last = try XCTUnwrap(m.observationStamp().lastRecordedTrainerStep)
        m.recordStep(S.record(last + 1))
        XCTAssertNotNil(live(m, at: last + 1, config: config, sink: sink))
    }

    func testConcurrentRecordAndEvaluate() throws {
        let m = monitor()
        let config = try S.config()
        let sink = S.LineSink()
        let windows = SyncBox<[(Int, Int, Int)]>([])
        let recordingDone = SyncBox(false)
        let group = DispatchGroup()
        DispatchQueue.global().async(group: group) {
            for step in 1...10_000 {
                m.recordStep(S.record(step, ms: 1))
            }
            recordingDone.value = true
        }
        DispatchQueue.global().async(group: group) {
            while !recordingDone.value {
                let stamp = m.observationStamp()
                guard let boundary = stamp.lastRecordedTrainerStep else { continue }
                if let result = m.evaluateLive(
                    stamp: stamp, layerHealth: .read(S.deadDigest(dead: 0), trainerStep: boundary),
                    learningRate: nil, momentum: nil, config: config, log: sink.sink),
                   let window = result.window, window.recordCount > 0,
                   let first = window.firstTrainerStep, let lastStep = window.lastTrainerStep {
                    windows.modify { $0.append((first, lastStep, window.recordCount)) }
                }
            }
        }
        DispatchQueue.global().async(group: group) {
            while !recordingDone.value {
                let stamp = m.observationStamp()
                m.evaluateCheckpoint(stamp: stamp, layerHealth: S.deadDigest(dead: 0, tier: .checkpoint),
                                     digestTrainerStep: stamp.lastRecordedTrainerStep ?? 0, config: config, log: sink.sink)
            }
        }
        group.wait()
        let final = try XCTUnwrap(live(m, at: 10_000, config: config, sink: sink))
        var all = windows.value
        if let window = final.window, window.recordCount > 0,
           let first = window.firstTrainerStep, let lastStep = window.lastTrainerStep {
            all.append((first, lastStep, window.recordCount))
        }
        all.sort { $0.0 < $1.0 }
        XCTAssertEqual(all.reduce(0) { $0 + $1.2 }, 10_000, "every recorded step lands in exactly one window")
        var expectedNext = 1
        for window in all {
            XCTAssertEqual(window.0, expectedNext)
            XCTAssertEqual(window.2, window.1 - window.0 + 1)
            expectedNext = window.1 + 1
        }
        XCTAssertEqual(expectedNext, 10_001)
    }

    // MARK: Check lines

    func testCheckLineAtEachIntervalAndFinalCheckLineFlushesThePartialInterval() throws {
        let m = monitor()
        let config = try S.config()
        let sink = S.LineSink()
        for step in stride(from: 50, through: 1250, by: 50) {
            record(m, (step - 49)...step)
            live(m, at: step, config: config, sink: sink)
        }
        let checks = sink.texts.filter { $0.hasPrefix("[HEALTH] check ") }
        XCTAssertEqual(checks.count, 1)
        let first = try XCTUnwrap(checks.first)
        XCTAssertTrue(first.hasPrefix("[HEALTH] check trainerStep=1000 generation=0 evaluations=20 live=20 checkpoint=0 stale=0 truncated=0 cost_ms="), first)
        XCTAssertTrue(first.contains(" train_ms=1000.0 "), first)
        m.writeFinalCheck(log: sink.sink)
        let final = try XCTUnwrap(sink.texts.last)
        XCTAssertTrue(final.hasPrefix("[HEALTH] check trainerStep=1250 generation=0 evaluations=5 live=5 checkpoint=0"), final)
        XCTAssertTrue(final.contains(" train_ms=250.0 "), final)
        XCTAssertTrue(final.hasSuffix(" final=true"), final)
        // A checkpoint pass that finishes after the final line still logs, with its own late line.
        checkpoint(m, at: 1250, digest: S.deadDigest(dead: 0, tier: .checkpoint), config: config, sink: sink)
        XCTAssertTrue(try XCTUnwrap(sink.texts.last).hasSuffix(" late=true"))
    }

    func testFinalCheckLineFlushesThePartialInterval() throws {
        let m = monitor()
        let sink = S.LineSink()
        record(m, 1...30)
        live(m, at: 30, config: try S.config(), sink: sink)
        XCTAssertFalse(sink.texts.contains { $0.hasPrefix("[HEALTH] check ") })
        m.writeFinalCheck(log: sink.sink)
        let final = try XCTUnwrap(sink.texts.last)
        XCTAssertTrue(final.hasPrefix("[HEALTH] check trainerStep=30 generation=0 evaluations=1 live=1"), final)
        XCTAssertTrue(final.contains(" train_ms=30.0 "), final)
        XCTAssertTrue(final.hasSuffix(" final=true"))
    }

    // MARK: D6 — the value-FC1 read schedule

    private func evaluateValueFC1(_ m: TrainingHealthMonitor, at step: Int, sink: S.LineSink) throws {
        checkpoint(m, at: step, digest: S.valueFC1Digest(zero: 0), config: try S.config(), sink: sink)
    }

    func testValueFC1ReadDueAfterOneThousandStepsFromStart() throws {
        let fresh = monitor()
        XCTAssertFalse(fresh.valueFC1ReadDue(trainerStep: 5000, config: try S.config()), "nothing recorded yet")
        record(fresh, 1...999)
        XCTAssertFalse(fresh.valueFC1ReadDue(trainerStep: 999, config: try S.config()))
        record(fresh, 1000...1000)
        XCTAssertTrue(fresh.valueFC1ReadDue(trainerStep: 1000, config: try S.config()))

        let resumed = monitor()
        record(resumed, 514...1512)
        XCTAssertFalse(resumed.valueFC1ReadDue(trainerStep: 1512, config: try S.config()))
        record(resumed, 1513...1513)
        XCTAssertTrue(resumed.valueFC1ReadDue(trainerStep: 1513, config: try S.config()))

        let other = monitor(.doesNotApply(activation: .silu))
        record(other, 1...5000)
        XCTAssertFalse(other.valueFC1ReadDue(trainerStep: 5000, config: try S.config()), "rule 3 does not apply: no read is ever due")
    }

    func testValueFC1ObservationsNeverMoreThanOneThousandApart() throws {
        let m = monitor()
        let sink = S.LineSink()
        record(m, 1...1001)
        try evaluateValueFC1(m, at: 1001, sink: sink)
        record(m, 1002...2000)
        XCTAssertFalse(m.valueFC1ReadDue(trainerStep: 2000, config: try S.config()))
        record(m, 2001...2001)
        XCTAssertTrue(m.valueFC1ReadDue(trainerStep: 2001, config: try S.config()))
    }

    func testValueFC1ReadNeverDueOnCorpusReplaySaveCadence() throws {
        // Fresh and resumed at 513, saves on the segment grid and on the
        // overall trainer-step grid; each save's pass feeds rule 3 before the
        // question is asked.
        for start in [0, 513] {
            for overallGrid in [false, true] {
                let m = monitor()
                let sink = S.LineSink()
                for step in (start + 1)...(start + 5000) {
                    m.recordStep(S.record(step))
                    let segmentStep = step - start
                    let isSave = overallGrid ? step % 1000 == 0 : segmentStep % 1000 == 0
                    if isSave {
                        try evaluateValueFC1(m, at: step, sink: sink)
                    }
                    XCTAssertFalse(m.valueFC1ReadDue(trainerStep: step, config: try S.config()),
                                   "start \(start) overall grid \(overallGrid) step \(step)")
                }
            }
        }
    }

    func testValueFC1ReadDueWhenASaveProducedNoObservation() throws {
        let m = monitor()
        let sink = S.LineSink()
        for step in 1...2000 {
            m.recordStep(S.record(step))
            if step == 1000 {
                // The save failed (or its health pass did): no observation.
                XCTAssertTrue(m.valueFC1ReadDue(trainerStep: step, config: try S.config()))
                try evaluateValueFC1(m, at: step, sink: sink)   // the dedicated read
                XCTAssertFalse(m.valueFC1ReadDue(trainerStep: step, config: try S.config()))
            }
        }
        XCTAssertTrue(m.valueFC1ReadDue(trainerStep: 2000, config: try S.config()))
    }

    func testValueFC1ReadDueAnchorsOnTheRestoredClockAfterARewind() throws {
        let m = monitor()
        let sink = S.LineSink()
        record(m, 1...1500)
        try evaluateValueFC1(m, at: 1500, sink: sink)
        m.noteTrainerClockRewind(to: 1200, log: sink.sink)
        record(m, 1201...2199)
        XCTAssertFalse(m.valueFC1ReadDue(trainerStep: 2199, config: try S.config()))
        record(m, 2200...2200)
        XCTAssertTrue(m.valueFC1ReadDue(trainerStep: 2200, config: try S.config()), "1,000 after the restored clock 1,200")
    }

    // MARK: Segment summary (OD-10)

    func testSegmentSummaryOneEntryPerRaisedRule() throws {
        let m = monitor()
        let config = try S.config()
        let sink = S.LineSink()
        record(m, 1...100)
        live(m, at: 100, digest: S.deadDigest(dead: 1), config: config, sink: sink)
        record(m, 101...150)
        live(m, at: 150, digest: S.deadDigest(dead: 0), config: config, sink: sink)
        record(m, 151...200)
        live(m, at: 200, digest: S.deadDigest(dead: 0), config: config, sink: sink)   // cleared
        record(m, 201...250)
        live(m, at: 250, digest: S.deadDigest(dead: 100), config: config, sink: sink) // raised again, critical
        let summary = m.segmentSummary()
        XCTAssertEqual(summary.raised.count, 1)
        let entry = try XCTUnwrap(summary.raised.first)
        XCTAssertEqual(entry.rule, .deadChannels)
        XCTAssertEqual(entry.firstTrainerStep, 100)
        XCTAssertEqual(entry.highestSeverity, .critical)
        XCTAssertEqual(entry.raiseCount, 2)
        let json = String(decoding: try JSONEncoder().encode(summary), as: UTF8.self)
        XCTAssertTrue(json.contains("\"first_trainer_step\":100"), json)
        XCTAssertTrue(json.contains("\"highest_severity\":\"critical\""), json)
        XCTAssertTrue(json.contains("\"raise_count\":2"), json)
        XCTAssertTrue(json.contains("\"rule\":\"dead_channels\""), json)
    }

    func testSegmentSummaryCountsCommittedEvaluationsOnly() throws {
        let m = monitor()
        let config = try S.config()
        let sink = S.LineSink()
        record(m, 1...100)
        let staleStamp = m.observationStamp()
        live(m, at: 100, config: config, sink: sink)
        checkpoint(m, at: 100, digest: S.deadDigest(dead: 0, tier: .checkpoint), config: config, sink: sink)
        m.noteTrainerClockRewind(to: 90, log: sink.sink)
        checkpoint(m, at: 100, digest: S.deadDigest(dead: 0, tier: .checkpoint), stamp: staleStamp, config: config, sink: sink)
        XCTAssertEqual(m.segmentSummary().evaluations, 2)
    }

    func testSegmentSummaryZeroWhenDisabled() throws {
        let m = monitor()
        let config = try S.config(enabled: false)
        let sink = S.LineSink()
        record(m, 1...100)
        live(m, at: 100, digest: S.deadDigest(dead: 500), config: config, sink: sink)
        checkpoint(m, at: 100, digest: S.deadDigest(dead: 500, tier: .checkpoint), config: config, sink: sink)
        XCTAssertEqual(m.segmentSummary(), TrainingHealthSegmentSummary(evaluations: 0, raised: []))
        XCTAssertTrue(sink.texts.isEmpty)
    }

    // MARK: Review fixes

    /// Review M2: with alarms disabled no evaluation moves the D6 anchor, so a
    /// due test that ignored the config would ask for a 512 KB GPU read on
    /// every step past the first 1,000. Disabled means nothing is due.
    func testValueFC1ReadNeverDueWhenDisabled() throws {
        let m = monitor()
        let disabled = try S.config(enabled: false)
        let sink = S.LineSink()
        var due = 0
        for step in 1...3000 {
            m.recordStep(S.record(step))
            if m.valueFC1ReadDue(trainerStep: step, config: disabled) {
                due += 1
                checkpoint(m, at: step, digest: S.valueFC1Digest(zero: 0), config: disabled, sink: sink)
            }
        }
        XCTAssertEqual(due, 0)
        XCTAssertTrue(m.valueFC1ReadDue(trainerStep: 3000, config: try S.config()), "enabled again, the read is due")
    }

    /// Review minor 2: an unannounced rewind that no evaluation has logged yet
    /// is logged by the announced rewind that follows it, in order.
    func testUnannouncedRewindIsLoggedWhenAnAnnouncedOneFollows() throws {
        let m = monitor()
        let sink = S.LineSink()
        record(m, 1...100)
        record(m, 51...60)   // unannounced rewind 100 -> 50
        m.noteTrainerClockRewind(to: 40, log: sink.sink)
        XCTAssertEqual(sink.texts, [
            "[HEALTH] trainer clock rewind detected by recordStep (not announced): 100 -> 50; generation 1",
            "[HEALTH] trainer clock rewound 60 -> 40; generation 2; windows reset",
        ])
        record(m, 41...50)
        live(m, at: 50, config: try S.config(), sink: sink)
        XCTAssertEqual(sink.texts.filter { $0.contains("not announced") }.count, 1, "logged once")
    }

    /// Review minor 3: a disabled config judges, counts and logs nothing —
    /// not a failed live read, not a stale stamp. A rewind detected meanwhile
    /// is logged by the first enabled evaluation.
    func testDisabledConfigLogsAndCountsNothingForAFailedReadOrAStaleStamp() throws {
        let m = monitor()
        let disabled = try S.config(enabled: false)
        let sink = S.LineSink()
        record(m, 1...50)
        m.evaluateLive(stamp: m.observationStamp(), layerHealth: .readFailed, learningRate: nil, momentum: nil,
                       config: disabled, log: sink.sink)
        record(m, 51...60)
        let stale = m.observationStamp()
        m.recordStep(S.record(55))   // unannounced rewind 60 -> 54
        m.evaluateLive(stamp: stale, layerHealth: .read(S.deadDigest(dead: 0), trainerStep: 60), learningRate: nil,
                       momentum: nil, config: disabled, log: sink.sink)
        m.evaluateCheckpoint(stamp: stale, layerHealth: S.deadDigest(dead: 0, tier: .checkpoint), digestTrainerStep: 60,
                             config: disabled, log: sink.sink)
        XCTAssertEqual(sink.texts, [])

        record(m, 56...100)
        live(m, at: 100, config: try S.config(), sink: sink)
        XCTAssertEqual(sink.texts.first,
                       "[HEALTH] trainer clock rewind detected by recordStep (not announced): 60 -> 54; generation 1")
        m.writeFinalCheck(log: sink.sink)
        let final = try XCTUnwrap(sink.texts.last)
        XCTAssertTrue(final.contains(" evaluations=1 live=1 checkpoint=0 stale=0 "), final)
        XCTAssertTrue(final.contains(" liveReadFailed=0 "), final)
    }

    /// Review minor 4: `nodata=` counts a rule only on evaluations whose
    /// source is meant to feed it — never the window rules on a checkpoint
    /// pass, never rule 3 on a live read, nothing but rule 3 on a dedicated
    /// value-FC1 read — so a healthy run's check line says `nodata=none`.
    func testNoDataCountsOnlyRulesTheEvaluationsSourceFeeds() throws {
        let m = monitor()
        let config = try S.config()
        let sink = S.LineSink()
        let base = S.deadDigest(dead: 0)
        let runningVariance = LayerHealthDigest.RunningVarianceRunaway(maxOverMedian: 5, site: "blocks.0.bn1")
        let liveDigest = LayerHealthDigest(
            tier: .live, deadChannels: base.deadChannels, nonFiniteValueCount: 0, runningVariance: runningVariance,
            valueFC1: nil)
        let checkpointDigest = LayerHealthDigest(
            tier: .checkpoint, deadChannels: base.deadChannels, nonFiniteValueCount: 0, runningVariance: runningVariance,
            valueFC1: LayerHealthDigest.ValueFC1Velocity(zeroVelocityUnitCount: 0, unitCount: 128))
        for step in 1...2000 {
            m.recordStep(S.record(step, offset: 0.1, ms: 1))
            if step % 50 == 0 {
                live(m, at: step, digest: liveDigest, config: config, sink: sink)
            }
            if step % 1000 == 0 {
                checkpoint(m, at: step, digest: checkpointDigest, config: config, sink: sink)
            }
            if step == 1500 {
                checkpoint(m, at: step, digest: S.valueFC1Digest(zero: 0), config: config, sink: sink)
            }
        }
        let checks = sink.texts.filter { $0.hasPrefix("[HEALTH] check ") }
        XCTAssertEqual(checks.count, 2, sink.texts.joined(separator: "\n"))
        let first = try XCTUnwrap(checks.first)
        XCTAssertFalse(first.contains("value_fc1_zero_velocity:"), first)
        let second = try XCTUnwrap(checks.last)
        XCTAssertTrue(second.contains(" live=20 checkpoint=2 "), second)
        XCTAssertTrue(second.contains(" nodata=none "), second)
        XCTAssertTrue(sink.texts.filter { $0.hasPrefix("[ALARM] health") }.isEmpty, sink.texts.joined(separator: "\n"))
    }

    /// Review minor 5: the final check line names the newest step the
    /// monitor knows — recorded or evaluated — and `none` when it knows none;
    /// never a made-up 0.
    func testFinalCheckLineNamesTheNewestKnownStep() throws {
        let m = monitor()
        let sink = S.LineSink()
        checkpoint(m, at: 5000, digest: S.deadDigest(dead: 0, tier: .checkpoint), config: try S.config(), sink: sink)
        m.writeFinalCheck(log: sink.sink)
        let final = try XCTUnwrap(sink.texts.last)
        XCTAssertTrue(final.hasPrefix("[HEALTH] check trainerStep=5000 "), final)

        let empty = monitor()
        let emptySink = S.LineSink()
        empty.writeFinalCheck(log: emptySink.sink)
        let emptyFinal = try XCTUnwrap(emptySink.texts.last)
        XCTAssertTrue(emptyFinal.hasPrefix("[HEALTH] check trainerStep=none "), emptyFinal)
    }

    func testStaleLiveObservationWithoutAnyStepSaysNone() throws {
        let m = monitor()
        let sink = S.LineSink()
        let stale = m.observationStamp()
        m.noteTrainerClockRewind(to: 0, log: sink.sink)
        m.evaluateLive(stamp: stale, layerHealth: .readFailed, learningRate: nil, momentum: nil,
                       config: try S.config(), log: sink.sink)
        XCTAssertTrue(sink.texts.contains { $0.hasPrefix("[HEALTH] stale live observation ignored: trainerStep=none generation=0") },
                      sink.texts.joined(separator: "\n"))
    }

    /// Review minor 1: while a `dead_channels` clear is pending, the reminder
    /// and the check line still name the sites (none now), and every check
    /// field is `key=value`.
    func testCheckLineNamesTheSitesWhileAClearIsPending() throws {
        let m = monitor()
        let config = try S.config()
        let sink = S.LineSink()
        record(m, 1...950)
        live(m, at: 950, digest: S.deadDigest(dead: 5, sites: [("value.bn", 5, 16)]), config: config, sink: sink)
        record(m, 951...1000)
        live(m, at: 1000, digest: S.deadDigest(dead: 0), config: config, sink: sink)
        let check = try XCTUnwrap(sink.texts.first { $0.hasPrefix("[HEALTH] check ") })
        let fields = check.dropFirst("[HEALTH] check ".count).split(separator: " ")
        XCTAssertTrue(fields.allSatisfy { $0.contains("=") }, check)
        XCTAssertTrue(check.hasSuffix(" active=dead_channels:critical dead_channels_sites=none"), check)
        let reminder = try XCTUnwrap(sink.texts.first { $0.hasPrefix("[ALARM] health active rule=dead_channels") })
        XCTAssertTrue(reminder.hasSuffix(" value=dead=0/1040 sites=none"), reminder)
    }
}
