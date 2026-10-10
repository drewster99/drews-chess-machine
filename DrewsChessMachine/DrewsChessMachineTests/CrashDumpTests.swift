//
//  CrashDumpTests.swift
//  DrewsChessMachineTests
//
//  Crash dumps (GPU fault forensics plan, Part C): every part is written,
//  the batch's recorded hash is the batch's, nothing is ever overwritten (a
//  second dump with the same name gets `-2`), and a weights read or a
//  trainer-state read that fails or hangs never blocks the dump — the halt
//  behind it must always proceed.
//

import XCTest
@testable import DrewsChessMachine

final class CrashDumpTests: XCTestCase {

    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("CrashDumpTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: directory)
    }

    private static let recentBatchHashes = [
        BatchHashChain.Entry(trainerStep: 45_900, batchHash: String(repeating: "a", count: 64), chain: nil),
    ]

    private static func context(reason: CrashDumpReason = .gpuFault,
                                batchHashes: [BatchHashChain.Entry]? = recentBatchHashes) -> CrashDumpContext {
        let boards = (0..<(2 * 30 * 64)).map { Float($0 % 5) }
        let batch = CapturedTrainingBatch(batchSize: 2, floatsPerBoard: 30 * 64, boards: boards,
                                          moves: [17, 4_863], outcomes: [1, -0.013])
        return CrashDumpContext(
            reason: reason, detail: "submission stage=training step first=error", pathKind: "replay",
            modelID: "20261009-2-14oI", trainerStep: 45_974, batchTrainerStep: 45_975,
            learningRate: 0.001, momentum: 0.95, runProvenance: "[RUN] path=replay run=…",
            batch: batch,
            batchHashes: batchHashes,
            gradientNorms: nil,
            faults: [GPUFaultLedger.Fault(sequence: 1, time: Date(timeIntervalSince1970: 1_791_600_000),
                                          source: .systemLog(message: "Caused GPU Hang Error (kIOGPUCommandBufferCallbackErrorHang)"))],
            notes: [])
    }

    private static let now = Date(timeIntervalSince1970: 1_791_601_029)

    func testEveryPartIsWrittenAndTheManifestNamesTheBatch() async throws {
        let context = Self.context()
        let written = await CrashDumpWriter.write(context, in: directory, now: Self.now) {
            (weights: [[1, 2], [.nan, 3]], velocity: [[0.5]])
        }
        let folder = try XCTUnwrap(written)
        XCTAssertEqual(folder.pathExtension, "dcmcrash")
        let names = Set(try FileManager.default.contentsOfDirectory(atPath: folder.path))
        for expected in ["manifest.json", "batch.safetensors", "log-tail.txt", "system-log.txt",
                         "weights-after.safetensors", "weights-after-census.json"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected) in \(names)")
        }
        let manifest = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("manifest.json")))
                as? [String: Any])
        XCTAssertEqual(manifest["reason"] as? String, "gpu-fault")
        XCTAssertEqual(manifest["batch_trainer_step"] as? Int, 45_975)
        let batch = try XCTUnwrap(context.batch)
        XCTAssertEqual(manifest["batch_hash"] as? String,
                       BatchHashChain.batchHash(boards: batch.boards, moves: batch.moves, outcomes: batch.outcomes))
        let census = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("weights-after-census.json")))
                as? [[String: Any]])
        XCTAssertEqual(census.map { $0["non_finite"] as? Int }, [0, 1, 0])
        // Staging was renamed away, not left behind.
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: directory.path).filter { $0.hasSuffix(".tmp") }
        XCTAssertEqual(leftovers, [])
    }

    func testASecondDumpWithTheSameNameGetsASuffixAndNothingIsOverwritten() async throws {
        let firstWritten = await CrashDumpWriter.write(Self.context(), in: directory, now: Self.now, weights: nil)
        let secondWritten = await CrashDumpWriter.write(Self.context(), in: directory, now: Self.now, weights: nil)
        let first = try XCTUnwrap(firstWritten)
        let second = try XCTUnwrap(secondWritten)
        XCTAssertNotEqual(first, second)
        XCTAssertTrue(second.lastPathComponent.hasSuffix("-gpu-fault-2.dcmcrash"), second.lastPathComponent)
    }

    func testAFailingWeightsReadLeavesAnErrorFileAndTheDump() async throws {
        struct ReadFailed: Error {}
        let written = await CrashDumpWriter.write(Self.context(), in: directory, now: Self.now) {
            throw ReadFailed()
        }
        let folder = try XCTUnwrap(written)
        let names = try FileManager.default.contentsOfDirectory(atPath: folder.path)
        XCTAssertTrue(names.contains("weights-after-error.txt"))
        XCTAssertFalse(names.contains("weights-after.safetensors"))
        XCTAssertTrue(names.contains("manifest.json"))
    }

    func testAHangingWeightsReadIsAbandonedAtTheTimeLimit() async {
        let started = Date()
        let outcome = await CrashDumpWriter.withTimeLimit(seconds: 1) { () async throws -> (weights: [[Float]], velocity: [[Float]]) in
            try await Task.sleep(for: .seconds(3_600))
            return (weights: [], velocity: [])
        }
        guard case .timedOut(let limitSeconds) = outcome else {
            return XCTFail("a hung read must be abandoned")
        }
        XCTAssertEqual(limitSeconds, 1)
        XCTAssertLessThan(Date().timeIntervalSince(started), 30)
    }

    func testAHangingWeightsReadStillLeavesTheDumpAndSaysWhy() async throws {
        let started = Date()
        let written = await CrashDumpWriter.write(Self.context(), in: directory, now: Self.now,
                                                  weightsTimeLimitSeconds: 1) {
            try await Task.sleep(for: .seconds(3_600))
            return (weights: [], velocity: [])
        }
        XCTAssertLessThan(Date().timeIntervalSince(started), 30)
        let folder = try XCTUnwrap(written)
        let names = try FileManager.default.contentsOfDirectory(atPath: folder.path)
        XCTAssertTrue(names.contains("manifest.json"))
        XCTAssertFalse(names.contains("weights-after.safetensors"))
        let reason = try String(contentsOf: folder.appendingPathComponent("weights-after-error.txt"), encoding: .utf8)
        XCTAssertTrue(reason.contains("longer than 1 s"), reason)
    }

    func testWorkThatFinishesInTimeIsReturnedAndAThrowIsReported() async {
        guard case .finished(let value) = await CrashDumpWriter.withTimeLimit(seconds: 30, { 42 }) else {
            return XCTFail("work that finishes in time must be returned")
        }
        XCTAssertEqual(value, 42)
        struct ReadFailed: Error {}
        let failed = await CrashDumpWriter.withTimeLimit(seconds: 30) { () async throws -> Int in
            throw ReadFailed()
        }
        guard case .failed(let error) = failed else {
            return XCTFail("a throw must be reported, not timed out")
        }
        XCTAssertTrue(error is ReadFailed)
    }

    func testAHangingTrainerQueueReadIsAbandonedAndNotedAndTheOthersAreKept() async throws {
        let started = Date()
        let history = GradientNormHistory()
        let hashes = Self.recentBatchHashes
        let state = await CrashDumpWriter.readTrainerState(
            batch: {
                try await Task.sleep(for: .seconds(3_600))
                return nil
            },
            gradientNorms: { history },
            batchHashes: { hashes },
            limitSeconds: 1)
        XCTAssertLessThan(Date().timeIntervalSince(started), 30)
        XCTAssertNil(state.batch)
        XCTAssertEqual(state.gradientNorms, history)
        XCTAssertEqual(state.batchHashes, hashes)
        XCTAssertEqual(state.notes.count, 1, "\(state.notes)")
        let note = try XCTUnwrap(state.notes.first)
        XCTAssertTrue(note.hasPrefix("batch not captured:"), note)
        XCTAssertTrue(note.contains("within 1 s"), note)
    }

    /// The three reads run at once: three hung reads cost one limit, not
    /// three (the halt behind a dump waits for them).
    func testThreeHangingTrainerStateReadsShareOneTimeLimit() async {
        let started = Date()
        let state = await CrashDumpWriter.readTrainerState(
            batch: {
                try await Task.sleep(for: .seconds(3_600))
                return nil
            },
            gradientNorms: {
                try await Task.sleep(for: .seconds(3_600))
                return GradientNormHistory()
            },
            batchHashes: {
                // Non-throwing, like `BatchHashChain.recentEntries`. Abandoned
                // work is never cancelled, so the sleep doesn't end early.
                do {
                    try await Task.sleep(for: .seconds(3_600))
                } catch {
                    return []
                }
                return []
            },
            limitSeconds: 2)
        // Sequential reads would take at least 6 s.
        XCTAssertLessThan(Date().timeIntervalSince(started), 5.5)
        XCTAssertNil(state.batch)
        XCTAssertNil(state.gradientNorms)
        XCTAssertNil(state.batchHashes)
        XCTAssertEqual(state.notes.count, 3, "\(state.notes)")
    }

    func testAFailedTrainerStateReadIsNotedWithItsError() async {
        struct ReadFailed: LocalizedError {
            var errorDescription: String? { "queue gone" }
        }
        let state = await CrashDumpWriter.readTrainerState(
            batch: { nil },
            gradientNorms: { throw ReadFailed() },
            batchHashes: { [] },
            limitSeconds: 30)
        XCTAssertNil(state.batch)
        XCTAssertNil(state.gradientNorms)
        XCTAssertEqual(state.batchHashes, [])
        XCTAssertEqual(state.notes, ["gradient-norm history not read: queue gone"])
    }

    func testUnreadBatchHashesAreLeftOutOfTheManifestNotWrittenAsEmpty() async throws {
        let written = await CrashDumpWriter.write(Self.context(batchHashes: nil), in: directory,
                                                  now: Self.now, weights: nil)
        let folder = try XCTUnwrap(written)
        let manifest = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("manifest.json")))
                as? [String: Any])
        XCTAssertNil(manifest["recent_batch_hashes"])
        let writtenWithHashes = await CrashDumpWriter.write(Self.context(), in: directory, now: Self.now, weights: nil)
        let folderWithHashes = try XCTUnwrap(writtenWithHashes)
        let manifestWithHashes = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: folderWithHashes.appendingPathComponent("manifest.json")))
                as? [String: Any])
        XCTAssertEqual((manifestWithHashes["recent_batch_hashes"] as? [[String: Any]])?.count, 1)
    }

    func testANearMissDumpReadsNoWeights() async throws {
        let written = await CrashDumpWriter.write(Self.context(reason: .nearMiss), in: directory,
                                                  now: Self.now, weights: nil)
        let folder = try XCTUnwrap(written)
        let names = try FileManager.default.contentsOfDirectory(atPath: folder.path)
        XCTAssertFalse(names.contains { $0.hasPrefix("weights-after") })
        XCTAssertTrue(folder.lastPathComponent.contains("-near-miss"))
    }

    func testNearMissNeedsAReferenceAndAThousandTimesIt() {
        func decision(median: Double?) -> GradientCapDecision {
            GradientCapDecision(fedCap: 0.85, decidedCap: 0.85, binding: .relative, referenceMedian: median,
                                referenceCount: median == nil ? 0 : 1_000, mode: .clip, hardMax: 15,
                                multiple: 3, floor: 0.75)
        }
        // R-replay's step at 21:37:09: 10,624,371 against a median of 0.2835.
        XCTAssertTrue(CrashDumpWriter.isNearMiss(preClipNorm: 10_624_371, decision: decision(median: 0.2835)))
        XCTAssertTrue(CrashDumpWriter.isNearMiss(preClipNorm: 283.5, decision: decision(median: 0.2835)))
        XCTAssertFalse(CrashDumpWriter.isNearMiss(preClipNorm: 283.4, decision: decision(median: 0.2835)))
        // An ordinary clip (9.38×) is not one; nor is anything without a reference.
        XCTAssertFalse(CrashDumpWriter.isNearMiss(preClipNorm: 2.35, decision: decision(median: 0.2505)))
        XCTAssertFalse(CrashDumpWriter.isNearMiss(preClipNorm: 1e9, decision: decision(median: nil)))
        XCTAssertFalse(CrashDumpWriter.isNearMiss(preClipNorm: 1e9, decision: .hardMaxOnly(hardMax: 15)))
    }

    /// The shape of the three reports macOS wrote for the 2026-10-09 hangs
    /// (`gpuEvent-DrewsChessMachin-2026-10-09-225904.ips`, trimmed): a JSON
    /// header line, then a JSON body naming the process cut to 16
    /// characters, with no pid.
    private static func gpuEventReport(processName: String) -> Data {
        Data(("""
        {"bug_type":"284","timestamp":"2026-10-09 22:59:04.00 -0500","os_version":"macOS 27.2 (26B5091g)","roots_installed":0,"incident_id":"0D63083D-3260-46A7-9573-BD91CA218FCD"}
        {
          "roots_installed" : 0,
          "bug_type" : "284",
          "process_name" : "\(processName)",
          "registers" : {},
          "timestamp" : 1791604744,
          "analysis" : {"guilty_dm":3,"command_buffer_trace_id":1406631678160,"signature":579,"restart_reason_desc":"firmware-detected lockup","restart_reason":4}
        }
        """).utf8)
    }

    /// Regression (2026-10-10): the copy looked for this pid in the report,
    /// which macOS never writes, and only in the top folder, from which macOS
    /// moves reports to `Retired/` — so a dump never held one.
    func testGPUEventReportsAreFoundByProcessNameIncludingRetiredOnes() throws {
        let reports = directory.appendingPathComponent("DiagnosticReports", isDirectory: true)
        let retired = reports.appendingPathComponent("Retired", isDirectory: true)
        let otherReports = directory.appendingPathComponent("UserDiagnosticReports", isDirectory: true)
        let dump = directory.appendingPathComponent("dump", isDirectory: true)
        for folder in [retired, otherReports, dump] {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        }
        let since = Self.now.addingTimeInterval(-120)
        func place(_ name: String, in folder: URL, processName: String, modified: Date) throws {
            let url = folder.appendingPathComponent(name)
            try Self.gpuEventReport(processName: processName).write(to: url)
            try FileManager.default.setAttributes([.modificationDate: modified], ofItemAtPath: url.path)
        }
        try place("gpuEvent-DrewsChessMachin-2026-10-09-225904.ips", in: retired,
                  processName: "DrewsChessMachin", modified: Self.now.addingTimeInterval(-5))
        try place("gpuEvent-DrewsChessMachin-2026-10-09-225910.ips", in: reports,
                  processName: "DrewsChessMachin", modified: Self.now.addingTimeInterval(-1))
        try place("gpuEvent-OtherGPUApp-2026-10-09-225904.ips", in: reports,
                  processName: "OtherGPUApp", modified: Self.now.addingTimeInterval(-5))
        try place("gpuEvent-DrewsChessMachin-2026-10-09-200000.ips", in: retired,
                  processName: "DrewsChessMachin", modified: since.addingTimeInterval(-1))

        // `otherReports` has no `Retired/`, as on a Mac that never moved one:
        // not a failure to note.
        CrashDumpWriter.copyGPUEventReports(into: dump, since: since, from: [reports, otherReports],
                                            processName: "DrewsChessMachine")

        let copied = try FileManager.default.contentsOfDirectory(atPath: dump.path).sorted()
        XCTAssertEqual(copied, ["gpuEvent-DrewsChessMachin-2026-10-09-225904.ips",
                                "gpuEvent-DrewsChessMachin-2026-10-09-225910.ips"])
    }

    func testTheLaunchSweepRecognizesOnlyCrashDumpStagingFolders() {
        let kind = CheckpointPaths.OrphanStagingKind.crashDumpDirectory
        XCTAssertTrue(kind.matchesName("20261009-213709-x-step1-gpu-fault.dcmcrash.tmp"))
        XCTAssertFalse(kind.matchesName("20261009-213709-x-step1-gpu-fault.dcmcrash"))
        XCTAssertFalse(kind.matchesName(".dcmcrash.tmp"))
        XCTAssertFalse(kind.matchesName("x.dcmsession.tmp"))
    }
}
