//
//  ChartDataRoundTripTests.swift
//  DrewsChessMachineTests
//
//  Tests for the chart-data save/restore path added alongside
//  `.dcmsession` save/load:
//
//   - `ChartFileFormat` JSON envelope round-trip on both sample
//     types, including NaN / Infinity edge cases that JSON does
//     not natively support (we carry them via the `.convertToString`
//     non-conforming float strategy).
//   - `ChartCoordinator.buildSnapshot` /
//     `seedFromRestoredSession` continuity — a saved + restored
//     session leaves the chart anchor back-dated so a new sample's
//     `elapsedSec` lands strictly after the last restored sample.
//   - Truncated / unreadable chart files surface as
//     `ChartFileError` rather than silently returning shorter
//     arrays.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class ChartDataRoundTripTests: XCTestCase {

    private var tmpDir: URL!

    override func setUpWithError() throws {
        tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-chart-test-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let tmpDir, FileManager.default.fileExists(atPath: tmpDir.path) {
            try FileManager.default.removeItem(at: tmpDir)
        }
    }

    // MARK: - Fixture builders

    private func makeTrainingSample(
        id: Int,
        elapsedSec: Double,
        thermal: ProcessInfo.ThermalState? = nil,
        gradNorm: Double? = nil
    ) -> TrainingChartSample {
        TrainingChartSample(
            id: id,
            elapsedSec: elapsedSec,
            rollingPolicyLoss: 1.5,
            rollingValueLoss: 0.25,
            rollingPolicyEntropy: 6.5,
            rollingPolicyNonNegCount: 12.0,
            rollingPolicyNonNegIllegalCount: 3.0,
            rollingGradNorm: gradNorm,
            rollingVelocityNorm: 0.4,
            rollingPolicyHeadWeightNorm: 0.15,
            replayRatio: 1.0,
            rollingPolicyLossWin: 1.4,
            rollingPolicyLossLoss: 1.6,
            rollingLegalEntropy: 3.2,
            rollingLegalMass: 0.92,
            cpuPercent: 75.5,
            gpuBusyPercent: 88.0,
            gpuMemoryMB: 512.0,
            appMemoryMB: 1024.0,
            lowPowerMode: false,
            thermalState: thermal
        )
    }

    private func makeAllNilTrainingSample(id: Int, elapsedSec: Double) -> TrainingChartSample {
        TrainingChartSample(
            id: id,
            elapsedSec: elapsedSec,
            rollingPolicyLoss: nil,
            rollingValueLoss: nil,
            rollingPolicyEntropy: nil,
            rollingPolicyNonNegCount: nil,
            rollingPolicyNonNegIllegalCount: nil,
            rollingGradNorm: nil,
            rollingVelocityNorm: nil,
            rollingPolicyHeadWeightNorm: nil,
            replayRatio: nil,
            rollingPolicyLossWin: nil,
            rollingPolicyLossLoss: nil,
            rollingLegalEntropy: nil,
            rollingLegalMass: nil,
            cpuPercent: nil,
            gpuBusyPercent: nil,
            gpuMemoryMB: nil,
            appMemoryMB: nil,
            lowPowerMode: nil,
            thermalState: nil
        )
    }

    private func makeProgressSample(id: Int, elapsedSec: Double) -> ProgressRateSample {
        ProgressRateSample(
            id: id,
            timestamp: Date(timeIntervalSince1970: 1_700_000_000 + elapsedSec),
            elapsedSec: elapsedSec,
            selfPlayCumulativeMoves: id * 100,
            trainingCumulativeMoves: id * 50,
            selfPlayMovesPerHour: 36000,
            trainingMovesPerHour: 18000
        )
    }

    /// Compare two `TrainingChartSample` instances field-by-field
    /// using bit-pattern equality on the `Double` fields. This is
    /// what the auto-synthesized `==` would do *except* it
    /// short-circuits on NaN (`Double.nan == Double.nan` is false
    /// in IEEE 754), which would falsely fail tests that
    /// deliberately verify NaN round-tripping.
    private func assertBitEqual(
        _ a: TrainingChartSample,
        _ b: TrainingChartSample,
        line: UInt = #line
    ) {
        XCTAssertEqual(a.id, b.id, "id", line: line)
        assertBitEqual(a.elapsedSec, b.elapsedSec, "elapsedSec", line: line)
        assertBitEqual(a.rollingPolicyLoss, b.rollingPolicyLoss, "rollingPolicyLoss", line: line)
        assertBitEqual(a.rollingValueLoss, b.rollingValueLoss, "rollingValueLoss", line: line)
        assertBitEqual(a.rollingPolicyEntropy, b.rollingPolicyEntropy, "rollingPolicyEntropy", line: line)
        assertBitEqual(a.rollingPolicyNonNegCount, b.rollingPolicyNonNegCount, "rollingPolicyNonNegCount", line: line)
        assertBitEqual(a.rollingPolicyNonNegIllegalCount, b.rollingPolicyNonNegIllegalCount, "rollingPolicyNonNegIllegalCount", line: line)
        assertBitEqual(a.rollingGradNorm, b.rollingGradNorm, "rollingGradNorm", line: line)
        assertBitEqual(a.rollingVelocityNorm, b.rollingVelocityNorm, "rollingVelocityNorm", line: line)
        assertBitEqual(a.rollingPolicyHeadWeightNorm, b.rollingPolicyHeadWeightNorm, "rollingPolicyHeadWeightNorm", line: line)
        assertBitEqual(a.replayRatio, b.replayRatio, "replayRatio", line: line)
        assertBitEqual(a.rollingPolicyLossWin, b.rollingPolicyLossWin, "rollingPolicyLossWin", line: line)
        assertBitEqual(a.rollingPolicyLossLoss, b.rollingPolicyLossLoss, "rollingPolicyLossLoss", line: line)
        assertBitEqual(a.rollingLegalEntropy, b.rollingLegalEntropy, "rollingLegalEntropy", line: line)
        assertBitEqual(a.rollingLegalMass, b.rollingLegalMass, "rollingLegalMass", line: line)
        assertBitEqual(a.cpuPercent, b.cpuPercent, "cpuPercent", line: line)
        assertBitEqual(a.gpuBusyPercent, b.gpuBusyPercent, "gpuBusyPercent", line: line)
        assertBitEqual(a.gpuMemoryMB, b.gpuMemoryMB, "gpuMemoryMB", line: line)
        assertBitEqual(a.appMemoryMB, b.appMemoryMB, "appMemoryMB", line: line)
        XCTAssertEqual(a.lowPowerMode, b.lowPowerMode, "lowPowerMode", line: line)
        XCTAssertEqual(a.thermalState, b.thermalState, "thermalState", line: line)
    }

    private func assertBitEqual(
        _ a: Double,
        _ b: Double,
        _ field: String,
        line: UInt = #line
    ) {
        XCTAssertEqual(
            a.bitPattern, b.bitPattern,
            "\(field): \(a) vs \(b)", line: line
        )
    }

    private func assertBitEqual(
        _ a: Double?,
        _ b: Double?,
        _ field: String,
        line: UInt = #line
    ) {
        switch (a, b) {
        case (nil, nil): break
        case (let x?, let y?):
            assertBitEqual(x, y, field, line: line)
        default:
            XCTFail("\(field) optional disagreement: \(String(describing: a)) vs \(String(describing: b))", line: line)
        }
    }

    // MARK: - Tests: TrainingChartSample round-trip

    func testTrainingChartSampleRoundTripFullyPopulated() throws {
        let url = tmpDir.appendingPathComponent("training_chart.json")
        let original = [
            makeTrainingSample(id: 0, elapsedSec: 0.0, thermal: .nominal),
            makeTrainingSample(id: 1, elapsedSec: 1.0, thermal: .fair, gradNorm: 0.42),
            makeTrainingSample(id: 2, elapsedSec: 2.0, thermal: .serious),
            makeTrainingSample(id: 3, elapsedSec: 3.0, thermal: .critical, gradNorm: 5.5),
        ]
        try writeChartFile(original, to: url)
        let restored = try readChartFile([TrainingChartSample].self, from: url)
        XCTAssertEqual(restored.count, original.count)
        for (lhs, rhs) in zip(original, restored) {
            assertBitEqual(lhs, rhs)
        }
    }

    func testTrainingChartSampleRoundTripAllNil() throws {
        let url = tmpDir.appendingPathComponent("training_chart.json")
        let original = [
            makeAllNilTrainingSample(id: 0, elapsedSec: 0.0),
            makeAllNilTrainingSample(id: 1, elapsedSec: 1.0),
        ]
        try writeChartFile(original, to: url)
        let restored = try readChartFile([TrainingChartSample].self, from: url)
        XCTAssertEqual(restored.count, original.count)
        for (lhs, rhs) in zip(original, restored) {
            assertBitEqual(lhs, rhs)
        }
    }

    /// Verifies that `Double.nan` and `±Double.infinity` survive the
    /// JSON round-trip via `.convertToString` non-conforming float
    /// strategy. `gNorm` legitimately becomes NaN if the network
    /// diverges, and the chart is the place users see that — so
    /// silently dropping NaN samples on save would erase exactly
    /// the data the user needs to debug the divergence.
    func testTrainingChartSampleRoundTripNaNAndInfinity() throws {
        let url = tmpDir.appendingPathComponent("training_chart.json")
        let original = [
            makeTrainingSample(id: 0, elapsedSec: 0.0, gradNorm: .nan),
            makeTrainingSample(id: 1, elapsedSec: 1.0, gradNorm: .infinity),
            makeTrainingSample(id: 2, elapsedSec: 2.0, gradNorm: -Double.infinity),
        ]
        try writeChartFile(original, to: url)
        let restored = try readChartFile([TrainingChartSample].self, from: url)
        XCTAssertEqual(restored.count, 3)
        XCTAssertTrue(restored[0].rollingGradNorm?.isNaN ?? false, "NaN should round-trip")
        XCTAssertEqual(restored[1].rollingGradNorm, .infinity)
        XCTAssertEqual(restored[2].rollingGradNorm, -Double.infinity)
    }

    // MARK: - Tests: ProgressRateSample round-trip

    func testProgressRateSampleRoundTrip() throws {
        let url = tmpDir.appendingPathComponent("progress_rate_chart.json")
        let original = [
            makeProgressSample(id: 0, elapsedSec: 0.0),
            makeProgressSample(id: 1, elapsedSec: 1.0),
            makeProgressSample(id: 100, elapsedSec: 100.0),
        ]
        try writeChartFile(original, to: url)
        let restored = try readChartFile([ProgressRateSample].self, from: url)
        XCTAssertEqual(restored, original)
    }

    // MARK: - Tests: file format errors

    func testTruncatedFileSurfacesAsDecodeError() throws {
        let url = tmpDir.appendingPathComponent("training_chart.json")
        let original = [
            makeTrainingSample(id: 0, elapsedSec: 0.0),
            makeTrainingSample(id: 1, elapsedSec: 1.0),
        ]
        try writeChartFile(original, to: url)
        // Truncate the file mid-array — JSON parse should fail with
        // a `decodeFailed`, not silently return a shorter array.
        let bytes = try Data(contentsOf: url)
        let truncated = bytes.prefix(bytes.count / 2)
        try truncated.write(to: url, options: [.atomic])
        XCTAssertThrowsError(try readChartFile([TrainingChartSample].self, from: url)) { error in
            guard let chartErr = error as? ChartFileError else {
                XCTFail("Expected ChartFileError, got \(error)")
                return
            }
            switch chartErr {
            case .decodeFailed: break
            default: XCTFail("Expected .decodeFailed, got \(chartErr)")
            }
        }
    }

    func testMismatchedSampleCountSurfacesAsCountError() throws {
        let url = tmpDir.appendingPathComponent("training_chart.json")
        // Build a malformed envelope where the declared sampleCount
        // (5) disagrees with the actual array length (2). The
        // checker is what guards against this on the load path
        // even when the bytes happen to JSON-parse.
        let malformed = """
        {
          "formatVersion": 1,
          "sampleCount": 5,
          "samples": [
            {"id": 0, "elapsedSec": 0},
            {"id": 1, "elapsedSec": 1}
          ]
        }
        """
        try Data(malformed.utf8).write(to: url, options: [.atomic])
        XCTAssertThrowsError(try readChartFile([TrainingChartSample].self, from: url)) { error in
            guard let chartErr = error as? ChartFileError else {
                XCTFail("Expected ChartFileError, got \(error)")
                return
            }
            switch chartErr {
            case .sampleCountMismatch(let declared, let actual):
                XCTAssertEqual(declared, 5)
                XCTAssertEqual(actual, 2)
            default:
                XCTFail("Expected .sampleCountMismatch, got \(chartErr)")
            }
        }
    }

    func testFutureFormatVersionRejected() throws {
        let url = tmpDir.appendingPathComponent("training_chart.json")
        let malformed = """
        {
          "formatVersion": 999,
          "sampleCount": 0,
          "samples": []
        }
        """
        try Data(malformed.utf8).write(to: url, options: [.atomic])
        XCTAssertThrowsError(try readChartFile([TrainingChartSample].self, from: url)) { error in
            guard let chartErr = error as? ChartFileError else {
                XCTFail("Expected ChartFileError, got \(error)")
                return
            }
            switch chartErr {
            case .unsupportedFormatVersion(let v):
                XCTAssertEqual(v, 999)
            default:
                XCTFail("Expected .unsupportedFormatVersion, got \(chartErr)")
            }
        }
    }

    // MARK: - Tests: ChartCoordinator save / restore continuity

    /// Verifies the central UX promise of the chart-restore feature:
    /// after save+restore, a new sample appended right after resume
    /// has an `elapsedSec` strictly greater than the last restored
    /// sample's, so the chart timeline continues monotonically with
    /// no overlap.
    func testChartCoordinatorRoundTripContinuesElapsedSec() async throws {
        let coord = ChartCoordinator()
        // Seed with a small synthetic trajectory mirroring what the
        // heartbeat would produce. Anchored 100 s ago so each
        // sample's elapsedSec is plausibly in the past.
        coord.chartElapsedAnchor = Date().addingTimeInterval(-100)
        coord.appendTrainingChart(
            makeTrainingSample(id: 0, elapsedSec: 50),
            totalGpuMs: 0
        )
        coord.appendTrainingChart(
            makeTrainingSample(id: 1, elapsedSec: 75),
            totalGpuMs: 0
        )
        coord.appendProgressRate(makeProgressSample(id: 0, elapsedSec: 50))
        coord.appendProgressRate(makeProgressSample(id: 1, elapsedSec: 75))

        guard let snapshot = coord.buildSnapshot() else {
            XCTFail("buildSnapshot should produce a non-nil snapshot")
            return
        }
        XCTAssertEqual(snapshot.trainingSamples.count, 2)
        XCTAssertEqual(snapshot.progressRateSamples.count, 2)
        XCTAssertEqual(snapshot.lastElapsedSec, 75)

        // Simulate resume: fresh coordinator → reset → seed.
        let resumed = ChartCoordinator()
        resumed.reset()
        resumed.seedFromRestoredSession(snapshot)
        XCTAssertEqual(resumed.trainingRing.count, 2)
        XCTAssertEqual(resumed.progressRateRing.count, 2)
        // The training-sample fixture's rollingLegalMass propagates
        // into legalMassMaxAllTime via appendTrainingChart's running-
        // max update; that value must round-trip through save/restore
        // so the LegalMassChart's tiered Y axis doesn't reset on
        // resume.
        XCTAssertEqual(resumed.legalMassMaxAllTime, 0.92)

        // Now sample as the heartbeat would, using the back-dated
        // anchor to compute elapsedSec.
        let nowElapsed = Date().timeIntervalSince(resumed.chartElapsedAnchor)
        XCTAssertGreaterThan(
            nowElapsed, snapshot.lastElapsedSec,
            "Anchor should be back-dated so 'now' is past the last restored elapsedSec"
        )
        // Slack: tests run in milliseconds, so the gap is small but
        // strictly positive. We only assert monotonicity here.
        XCTAssertLessThan(
            nowElapsed - snapshot.lastElapsedSec, 5.0,
            "Anchor back-date should land within ~5s of test-start, not jump ahead by hours"
        )
    }

    /// Regression: after `seedFromRestoredSession`, `scrollX` should
    /// already point at the window containing the latest restored
    /// sample — without this, the chart momentarily renders at
    /// `scrollX = 0` (the leftmost few minutes of restored data) on
    /// resume until the first post-resume heartbeat tick re-runs the
    /// auto-follow math inside `appendProgressRate`. The user-visible
    /// "rewind" flash that bug produced would be brief but
    /// disorienting on long restored sessions.
    func testSeedFromRestoredSessionAdvancesScrollX() async throws {
        let coord = ChartCoordinator()
        coord.chartElapsedAnchor = Date().addingTimeInterval(-7200)
        // 2-hour worth of synthetic samples, sparse so the test stays fast.
        for i in 0..<120 {
            coord.appendTrainingChart(
                makeTrainingSample(id: i, elapsedSec: Double(i) * 60),
                totalGpuMs: 0
            )
            coord.appendProgressRate(
                makeProgressSample(id: i, elapsedSec: Double(i) * 60)
            )
        }
        guard let snapshot = coord.buildSnapshot() else {
            XCTFail("buildSnapshot should succeed with non-empty rings")
            return
        }
        XCTAssertEqual(snapshot.lastElapsedSec, 7140) // 119 * 60

        let resumed = ChartCoordinator()
        resumed.reset()
        XCTAssertEqual(resumed.scrollX, 0, "Reset should zero scrollX")
        XCTAssertTrue(resumed.followLatest, "Reset should leave followLatest true")
        resumed.seedFromRestoredSession(snapshot)
        // With followLatest=true, scrollX should land at the exact
        // expression appendProgressRate uses on every new sample:
        // `max(0, lastElapsedSec - currentWindowSec)`. Asserting the
        // exact value (not just direction + bound) catches a
        // sign-flip regression that would otherwise still pass the
        // directional checks below.
        let expectedScrollX = max(0, snapshot.lastElapsedSec - ChartZoom.stops[resumed.chartZoomIdx])
        XCTAssertEqual(
            resumed.scrollX, expectedScrollX, accuracy: 0.001,
            "scrollX should land at max(0, lastElapsedSec - windowSec)"
        )
        // Belt-and-suspenders directional checks. Cheap to keep, and
        // they document the invariant in case the exact-value math
        // ever needs to be relaxed.
        XCTAssertGreaterThan(
            resumed.scrollX, 0,
            "scrollX should advance to keep the latest restored sample on screen"
        )
        XCTAssertLessThanOrEqual(
            resumed.scrollX, snapshot.lastElapsedSec,
            "scrollX should never overshoot the data span"
        )
    }

    /// Verify that an empty/disabled coordinator returns nil from
    /// `buildSnapshot` so the save path can skip writing chart files
    /// without producing a zero-sample envelope.
    func testEmptyCoordinatorBuildSnapshotReturnsNil() {
        let coord = ChartCoordinator()
        XCTAssertNil(coord.buildSnapshot(), "Empty rings should produce nil snapshot")

        coord.collectionEnabled = false
        coord.appendTrainingChart(
            makeTrainingSample(id: 0, elapsedSec: 0), totalGpuMs: 0
        )
        XCTAssertNil(
            coord.buildSnapshot(),
            "Collection-disabled gate should produce nil even if rings have stale data"
        )
    }

    // MARK: - Tests: ArenaChartEvent Codable round-trip

    func testArenaChartEventRoundTrip() throws {
        let original = ArenaChartEvent(
            id: 7,
            startElapsedSec: 100.0,
            endElapsedSec: 130.5,
            score: 0.625,
            promoted: true
        )
        let data = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(ArenaChartEvent.self, from: data)
        XCTAssertEqual(restored, original)
    }
}
