//
//  CrashDumpTests.swift
//  DrewsChessMachineTests
//
//  Crash dumps (GPU fault forensics plan, Part C): every part is written,
//  the batch's recorded hash is the batch's, nothing is ever overwritten (a
//  second dump with the same name gets `-2`), and a weights read that fails
//  or hangs never blocks the dump — the halt behind it must always proceed.
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

    private static func context(reason: CrashDumpReason = .gpuFault) -> CrashDumpContext {
        let boards = (0..<(2 * 30 * 64)).map { Float($0 % 5) }
        let batch = CapturedTrainingBatch(batchSize: 2, floatsPerBoard: 30 * 64, boards: boards,
                                          moves: [17, 4_863], outcomes: [1, -0.013])
        return CrashDumpContext(
            reason: reason, detail: "submission stage=training step first=error", pathKind: "replay",
            modelID: "20261009-2-14oI", trainerStep: 45_974, batchTrainerStep: 45_975,
            learningRate: 0.001, momentum: 0.95, runProvenance: "[RUN] path=replay run=…",
            batch: batch,
            batchHashes: [BatchHashChain.Entry(trainerStep: 45_900, batchHash: String(repeating: "a", count: 64),
                                               chain: nil)],
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
        let outcome = await CrashDumpWriter.readWithTimeLimit({
            try await Task.sleep(for: .seconds(3_600))
            return (weights: [], velocity: [])
        }, limitSeconds: 1)
        guard case .failure(let reason) = outcome else {
            return XCTFail("a hung read must be abandoned")
        }
        XCTAssertTrue(reason.contains("longer than 1 s"), reason)
        XCTAssertLessThan(Date().timeIntervalSince(started), 30)
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

    func testTheLaunchSweepRecognizesOnlyCrashDumpStagingFolders() {
        let kind = CheckpointPaths.OrphanStagingKind.crashDumpDirectory
        XCTAssertTrue(kind.matchesName("20261009-213709-x-step1-gpu-fault.dcmcrash.tmp"))
        XCTAssertFalse(kind.matchesName("20261009-213709-x-step1-gpu-fault.dcmcrash"))
        XCTAssertFalse(kind.matchesName(".dcmcrash.tmp"))
        XCTAssertFalse(kind.matchesName("x.dcmsession.tmp"))
    }
}
