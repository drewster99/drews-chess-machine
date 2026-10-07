import XCTest
@testable import DrewsChessMachine

/// The models folder for a test whose model source never lists it (every
/// source but follow lineage): a scan fails the test.
struct LichessBotNoModelsFolderScanner: LichessBotModelFolderScanning {
    struct Unexpected: LocalizedError {
        var errorDescription: String? { "this test's model source lists no models folder" }
    }

    func scan(previous: ModelFolderHeaderCache) async throws -> ModelFolderScan {
        XCTFail("a model source other than follow lineage listed the models folder")
        throw Unexpected()
    }
}

/// A models folder for follow-lineage tests: real model files of training
/// runs whose `dcm_lineage` records `LineageTracker` makes, each save with
/// its own weights (so its own `content_sha256`), in a temporary folder
/// removed after the test.
struct LichessBotLineageModelFolder {
    let folder: URL
    let weights: [[Float]]
    let architecture: NetworkArchitecture

    static func make(for testCase: XCTestCase) async throws -> LichessBotLineageModelFolder {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotLineageModelFolder-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        testCase.addTeardownBlock {
            do {
                try FileManager.default.removeItem(at: folder)
            } catch {
                XCTFail("cleanup failed: \(error)")
            }
        }
        let network = try ChessMPSNetwork(.randomWeights(initSeed: 7))
        let weights = try await network.exportWeights()
        return LichessBotLineageModelFolder(folder: folder, weights: weights, architecture: network.arch)
    }

    /// A saved file: where, its record, and its header's content hash.
    struct SavedFile {
        let url: URL
        let record: LineageRecord
        let modelID: String
        let contentSHA256: String
        let bytes: Data
    }

    /// The bytes of `segment`'s save at `localStep`, recorded at `unix`.
    /// `variant` gives a second, different save at the same step.
    func encode(_ segment: LineageTestSegment, localStep: Int, at unix: Int64, variant: Float = 0) throws -> (bytes: Data, record: LineageRecord) {
        let record = try segment.record(localStep: localStep, at: unix)
        var distinct = weights
        distinct[0][0] += Float(localStep) * 0.001 + Float(unix % 997) * 0.000_001 + variant
        let bytes = try SafetensorsModelIO.encode(
            modelID: segment.modelID,
            createdAtUnix: unix,
            metadata: ModelCheckpointMetadata(creator: "replay", trainingStep: localStep, parentModelID: "", notes: "follow-lineage test"),
            weights: distinct,
            architecture: architecture,
            includesVelocity: false,
            lineage: record
        )
        return (bytes, record)
    }

    /// Writes `segment`'s save at `localStep` as `name` (replacing a file of
    /// that name with a new one, as a rolling save does).
    @discardableResult
    func write(_ segment: LineageTestSegment, localStep: Int, at unix: Int64, name: String, variant: Float = 0) throws -> SavedFile {
        let encoded = try encode(segment, localStep: localStep, at: unix, variant: variant)
        let url = folder.appendingPathComponent(name)
        try encoded.bytes.write(to: url, options: .atomic)
        return try saved(encoded.bytes, record: encoded.record, modelID: segment.modelID, at: url)
    }

    /// Writes `bytes` (an existing save's) as `name`.
    @discardableResult
    func copy(_ file: SavedFile, as name: String) throws -> SavedFile {
        let url = folder.appendingPathComponent(name)
        try file.bytes.write(to: url, options: .atomic)
        return try saved(file.bytes, record: file.record, modelID: file.modelID, at: url)
    }

    private func saved(_ bytes: Data, record: LineageRecord, modelID: String, at url: URL) throws -> SavedFile {
        let metadata = try ModelFileCatalog.headerMetadata(at: url)
        let sha = try XCTUnwrap(metadata[SafetensorsFile.contentHashKey])
        return SavedFile(url: url, record: record, modelID: modelID, contentSHA256: sha, bytes: bytes)
    }

    func remove(_ file: SavedFile) throws {
        try FileManager.default.removeItem(at: file.url)
    }
}

/// The models folder as a test scripts it: the real scan of a temporary
/// folder, counted, and able to fail or be held.
final class LichessBotScriptedFolderScanner: LichessBotModelFolderScanning, @unchecked Sendable {
    struct ScriptedFailure: LocalizedError {
        let reason: String
        var errorDescription: String? { reason }
    }

    private let inner: LichessBotModelsFolderScanner
    let scans = SyncBox(0)
    /// When set, every scan fails with this reason.
    let failure = SyncBox<String?>(nil)
    /// Scans that reached the hold and are waiting on it.
    let scansHeld = SyncBox(0)
    private let hold = SyncBox<LichessBotTestLatch?>(nil)

    init(directory: URL) {
        inner = LichessBotModelsFolderScanner(directory: directory)
    }

    func holdScans() {
        hold.value = LichessBotTestLatch()
    }

    func releaseScans() {
        let latch = hold.mutate { held -> LichessBotTestLatch? in
            defer { held = nil }
            return held
        }
        latch?.open()
    }

    func scan(previous: ModelFolderHeaderCache) async throws -> ModelFolderScan {
        scans.modify { $0 += 1 }
        if let latch = hold.value {
            scansHeld.modify { $0 += 1 }
            await latch.wait()
        }
        if let reason = failure.value {
            throw ScriptedFailure(reason: reason)
        }
        return try await inner.scan(previous: previous)
    }
}

/// The follow-lineage source in `LichessBotModelSlots` (follow-lineage plan
/// §3.4, §3.6): the first build before going online, checks on their own
/// cadence, never stepping back, problems that keep the last good
/// generation playing and throw (so the controller alarms), file failures
/// remembered until the file changes, and the loaded file verified against
/// what the check selected.
final class LichessBotFollowLineageSlotsTests: XCTestCase {

    private func followed(_ record: LineageRecord) -> LichessBotFollowedLineage {
        LichessBotFollowedLineage(lineageRunID: record.run.lineageRunID, anchorSegmentID: record.run.segmentID)
    }

    private func settings(following lineage: LichessBotFollowedLineage, interval: Int = 60) -> LichessBotModelSettings {
        var settings = LichessBotModelSettings.testBaseline()
        settings.source = .followLineage
        settings.followedLineage = lineage
        settings.lineageCheckIntervalSeconds = interval
        return settings
    }

    /// A loader that reads the file (or, for a scripted URL, other bytes),
    /// counting reads.
    private func countingLoader(reads: SyncBox<Int>, substitutes: SyncBox<[URL: Data]> = SyncBox([:])) -> LichessBotModelFileLoader {
        LichessBotModelFileLoader(readBytes: { url in
            reads.modify { $0 += 1 }
            if let bytes = substitutes.value[url] {
                return bytes
            }
            return try LichessBotModelFileLoader.live.readBytes(url)
        })
    }

    private func prepare(
        _ settings: LichessBotModelSettings,
        scanner: LichessBotScriptedFolderScanner,
        time: LichessBotManualTime,
        loader: LichessBotModelFileLoader = .live,
        networkFactory: LichessBotInferenceNetworkFactory = .live,
        lines: SyncBox<[String]> = SyncBox([])
    ) async throws -> LichessBotModelSlots {
        let provider = try await LichessBotHoldableModelProvider.make()
        return try await LichessBotModelSlots.prepare(
            for: settings, provider: provider, time: time, folderScanner: scanner,
            loader: loader, networkFactory: networkFactory,
            log: { line in lines.modify { $0.append(line) } })
    }

    private func assertUnavailable(_ expected: (LichessBotLineageFollowStatus.Outcome) -> Bool, _ body: () async throws -> Void, file: StaticString = #filePath, line: UInt = #line) async {
        do {
            try await body()
            XCTFail("expected the lineage to be unavailable", file: file, line: line)
        } catch LichessBotLineageFollowError.unavailable(let outcome) {
            XCTAssertTrue(expected(outcome), "unexpected outcome \(outcome)", file: file, line: line)
        } catch {
            XCTFail("expected unavailable, got \(error)", file: file, line: line)
        }
    }

    private func isNoFiles(_ outcome: LichessBotLineageFollowStatus.Outcome) -> Bool {
        if case .noFiles = outcome { return true }
        return false
    }

    private func isFork(_ outcome: LichessBotLineageFollowStatus.Outcome) -> Bool {
        if case .fork = outcome { return true }
        return false
    }

    private func isFolderUnreadable(_ outcome: LichessBotLineageFollowStatus.Outcome) -> Bool {
        if case .folderUnreadable = outcome { return true }
        return false
    }

    /// A data byte flipped: the header (and its `content_sha256`) still
    /// reads, the decode refuses the data.
    private func corrupt(_ bytes: Data) -> Data {
        var corrupted = bytes
        corrupted[corrupted.count - 1] ^= 0xFF
        return corrupted
    }

    // MARK: - Building and checking

    func testPrepareBuildsTheNewestFile() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let first = try models.write(run, localStep: 1000, at: 2_000, name: "run-replay-step1000.safetensors")
        let newest = try models.write(run, localStep: 2000, at: 3_000, name: "run-replay-step2000.safetensors")
        try models.copy(newest, as: "run-replay-latest.safetensors")
        let lines = SyncBox<[String]>([])
        let slots = try await prepare(settings(following: followed(first.record)), scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: LichessBotManualTime(), lines: lines)

        let info = await slots.current.info
        XCTAssertEqual(info.sourceKind, .followLineage)
        XCTAssertEqual(info.modelID, run.modelID)
        XCTAssertEqual(info.trainingStep, 2000)
        XCTAssertEqual(info.lineage?.segmentLocalStep, 2000)
        XCTAssertEqual(info.lineage?.contentSHA256, newest.contentSHA256)
        XCTAssertEqual(info.lineage?.followed, followed(first.record))
        let status = await slots.lineageFollowStatus
        guard case .following(let file)? = status?.outcome else {
            return XCTFail("expected following, got \(String(describing: status?.outcome))")
        }
        XCTAssertEqual(file.segmentLocalStep, 2000)
        XCTAssertTrue(lines.value.contains { $0.hasPrefix("[LICHESS-BOT] lineage follow: run=\(first.record.run.lineageRunID)") }, "\(lines.value)")
        XCTAssertTrue(lines.value.contains { $0.hasPrefix("[LICHESS-BOT] model generation 1 ready: followLineage \(run.modelID) step=2000 (going online)") && $0.contains(" seg=0 ") && $0.contains(" sha=\(newest.contentSHA256.prefix(12))") }, "\(lines.value)")
    }

    func testCheckRunsOnlyAfterTheInterval() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let file = try models.write(run, localStep: 1000, at: 2_000, name: "a.safetensors")
        let scanner = LichessBotScriptedFolderScanner(directory: models.folder)
        let time = LichessBotManualTime()
        let follow = settings(following: followed(file.record), interval: 60)
        let slots = try await prepare(follow, scanner: scanner, time: time)
        XCTAssertEqual(scanner.scans.value, 1)

        time.advance(by: .seconds(30))
        try await slots.refreshIfDue(for: follow)
        XCTAssertEqual(scanner.scans.value, 1, "not due before the interval")
        time.advance(by: .seconds(30))
        try await slots.refreshIfDue(for: follow)
        XCTAssertEqual(scanner.scans.value, 2, "due once the interval has elapsed")
        try await slots.checkLineageNow(for: follow)
        XCTAssertEqual(scanner.scans.value, 3, "Check Now checks at once")
    }

    func testNewerFileRebuildsAndOlderDoesNot() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let first = try models.write(run, localStep: 2000, at: 2_000, name: "step2000.safetensors")
        let time = LichessBotManualTime()
        let follow = settings(following: followed(first.record))
        let slots = try await prepare(follow, scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: time)

        let newer = try models.write(run, localStep: 3000, at: 3_000, name: "step3000.safetensors")
        time.advance(by: .seconds(60))
        let outcome = try await slots.refreshIfDue(for: follow)
        guard case .built(let built) = outcome else {
            return XCTFail("expected a build, got \(outcome)")
        }
        XCTAssertEqual(built.generationID, 2)
        XCTAssertEqual(built.lineage?.contentSHA256, newer.contentSHA256)

        // A later save at a lower step of the same segment is not newer.
        try models.write(run, localStep: 1500, at: 4_000, name: "step1500.safetensors")
        time.advance(by: .seconds(60))
        let again = try await slots.refreshIfDue(for: follow)
        XCTAssertEqual(again, .unchanged)
        let info = await slots.current.info
        XCTAssertEqual(info.generationID, 2)
    }

    func testSameContentDoesNotRebuild() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let file = try models.write(run, localStep: 1000, at: 2_000, name: "run-replay-step1000.safetensors")
        let time = LichessBotManualTime()
        let follow = settings(following: followed(file.record))
        let slots = try await prepare(follow, scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: time)

        try models.copy(file, as: "run-replay-latest.safetensors")
        time.advance(by: .seconds(60))
        let outcome = try await slots.refreshIfDue(for: follow)
        XCTAssertEqual(outcome, .unchanged, "the rolling copy holds the weights already playing")
    }

    func testRunEndKeepsTheLastFileAvailable() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let file = try models.write(run, localStep: 1000, at: 2_000, name: "final.safetensors")
        let time = LichessBotManualTime()
        let follow = settings(following: followed(file.record))
        let slots = try await prepare(follow, scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: time)
        for _ in 0..<3 {
            time.advance(by: .seconds(60))
            let outcome = try await slots.refreshIfDue(for: follow)
            XCTAssertEqual(outcome, .unchanged)
        }
        let status = await slots.lineageFollowStatus
        guard case .following = status?.outcome else {
            return XCTFail("a finished run's newest file is still the lineage's newest, got \(String(describing: status?.outcome))")
        }
    }

    func testDeletedNewestKeepsPlayingNeverBackward() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let older = try models.write(run, localStep: 1000, at: 2_000, name: "step1000.safetensors")
        let newest = try models.write(run, localStep: 2000, at: 3_000, name: "step2000.safetensors")
        let time = LichessBotManualTime()
        let follow = settings(following: followed(older.record))
        let lines = SyncBox<[String]>([])
        let slots = try await prepare(follow, scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: time, lines: lines)
        try models.remove(newest)
        time.advance(by: .seconds(60))
        let outcome = try await slots.refreshIfDue(for: follow)
        XCTAssertEqual(outcome, .unchanged)
        let info = await slots.current.info
        XCTAssertEqual(info.lineage?.contentSHA256, newest.contentSHA256, "never steps back to an older file")
        let status = await slots.lineageFollowStatus
        guard case .keepPlaying? = status?.outcome else {
            return XCTFail("expected keepPlaying, got \(String(describing: status?.outcome))")
        }
        XCTAssertTrue(lines.value.contains { $0.contains("ranks below the generation playing; still playing generation 1") }, "\(lines.value)")
    }

    // MARK: - Problems while online

    func testNoFilesKeepsPlayingAndThrowsOnRefresh() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let file = try models.write(run, localStep: 1000, at: 2_000, name: "a.safetensors")
        let time = LichessBotManualTime()
        let follow = settings(following: followed(file.record))
        let slots = try await prepare(follow, scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: time)
        try models.remove(file)
        time.advance(by: .seconds(60))
        for _ in 0..<2 {
            await assertUnavailable(isNoFiles) { try await slots.refreshIfDue(for: follow) }
        }
        let info = await slots.current.info
        XCTAssertEqual(info.generationID, 1, "the last good generation keeps playing")
    }

    func testFolderUnreadableKeepsPlayingAndThrowsOnRefresh() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let file = try models.write(run, localStep: 1000, at: 2_000, name: "a.safetensors")
        let scanner = LichessBotScriptedFolderScanner(directory: models.folder)
        let time = LichessBotManualTime()
        let follow = settings(following: followed(file.record))
        let slots = try await prepare(follow, scanner: scanner, time: time)
        scanner.failure.value = "permission denied"
        time.advance(by: .seconds(60))
        for _ in 0..<2 {
            await assertUnavailable(isFolderUnreadable) { try await slots.refreshIfDue(for: follow) }
        }
        let info = await slots.current.info
        XCTAssertEqual(info.generationID, 1)
    }

    func testForkKeepsPlayingAndThrowsOnRefresh() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let first = try LineageTestRuns.fresh(modelID: "20261006-1-SEG0", startedUnix: 1_000)
        let anchor = try models.write(first, localStep: 3000, at: 2_000, name: "seg0.safetensors")
        let time = LichessBotManualTime()
        let follow = settings(following: followed(anchor.record))
        let slots = try await prepare(follow, scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: time)

        let left = try LineageTestRuns.resume(from: anchor.record, parentModelID: first.modelID, parentSHA256: anchor.contentSHA256, modelID: "20261006-2-LEFT", startedUnix: 3_000)
        let right = try LineageTestRuns.resume(from: anchor.record, parentModelID: first.modelID, parentSHA256: anchor.contentSHA256, modelID: "20261006-3-RGHT", startedUnix: 3_100)
        try models.write(left, localStep: 1000, at: 4_000, name: "left.safetensors")
        try models.write(right, localStep: 1000, at: 4_100, name: "right.safetensors")
        time.advance(by: .seconds(60))
        for _ in 0..<2 {
            await assertUnavailable(isFork) { try await slots.refreshIfDue(for: follow) }
        }
        let info = await slots.current.info
        XCTAssertEqual(info.generationID, 1)
    }

    func testProblemClearingLogsAvailableAgain() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let file = try models.write(run, localStep: 1000, at: 2_000, name: "a.safetensors")
        let time = LichessBotManualTime()
        let follow = settings(following: followed(file.record))
        let lines = SyncBox<[String]>([])
        let slots = try await prepare(follow, scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: time, lines: lines)
        try models.remove(file)
        time.advance(by: .seconds(60))
        await assertUnavailable(isNoFiles) { try await slots.refreshIfDue(for: follow) }
        XCTAssertTrue(lines.value.contains { $0.hasPrefix("[LICHESS-BOT] lineage follow problem: no file of the followed lineage") && $0.hasSuffix("still playing generation 1 (\(run.modelID) step 1000)") }, "\(lines.value)")

        try models.write(run, localStep: 2000, at: 3_000, name: "b.safetensors")
        // A problem is re-checked at the next poll the backoff allows.
        let outcome = try await slots.refreshIfDue(for: follow)
        guard case .built(let built) = outcome else {
            return XCTFail("expected a build, got \(outcome)")
        }
        XCTAssertEqual(built.trainingStep, 2000)
        XCTAssertTrue(lines.value.contains { $0.hasPrefix("[LICHESS-BOT] lineage follow available again:") }, "\(lines.value)")
    }

    // MARK: - Before going online

    func testPrepareFailsWhenTheLineageHasNoFiles() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let record = try run.record(localStep: 1000, at: 2_000)
        let follow = settings(following: followed(record))
        await assertUnavailable(isNoFiles) {
            _ = try await self.prepare(follow, scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: LichessBotManualTime())
        }
    }

    func testPrepareFailsWhenTheFolderIsUnreadable() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let file = try models.write(run, localStep: 1000, at: 2_000, name: "a.safetensors")
        let follow = settings(following: followed(file.record))
        let missing = models.folder.appendingPathComponent("absent", isDirectory: true)
        await assertUnavailable(isFolderUnreadable) {
            _ = try await self.prepare(follow, scanner: LichessBotScriptedFolderScanner(directory: missing), time: LichessBotManualTime())
        }
    }

    func testPrepareFailsOnAFork() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let first = try LineageTestRuns.fresh(modelID: "20261006-1-SEG0", startedUnix: 1_000)
        let anchor = try models.write(first, localStep: 3000, at: 2_000, name: "seg0.safetensors")
        let left = try LineageTestRuns.resume(from: anchor.record, parentModelID: first.modelID, parentSHA256: anchor.contentSHA256, modelID: "20261006-2-LEFT", startedUnix: 3_000)
        let right = try LineageTestRuns.resume(from: anchor.record, parentModelID: first.modelID, parentSHA256: anchor.contentSHA256, modelID: "20261006-3-RGHT", startedUnix: 3_100)
        try models.write(left, localStep: 1000, at: 4_000, name: "left.safetensors")
        try models.write(right, localStep: 1000, at: 4_100, name: "right.safetensors")
        let follow = settings(following: followed(anchor.record))
        await assertUnavailable(isFork) {
            _ = try await self.prepare(follow, scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: LichessBotManualTime())
        }
    }

    func testPrepareFailsWhenTheNewestFileFailsToLoad() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let encoded = try models.encode(run, localStep: 1000, at: 2_000)
        let url = models.folder.appendingPathComponent("a.safetensors")
        try corrupt(encoded.bytes).write(to: url)
        let follow = settings(following: followed(encoded.record))
        do {
            _ = try await prepare(follow, scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: LichessBotManualTime())
            XCTFail("a newest file that doesn't load must keep the bot offline")
        } catch LichessBotLineageFollowError.fileFailedToLoad(let file, _) {
            XCTAssertEqual(file.lastPathComponent, "a.safetensors")
        }
    }

    // MARK: - File failures

    func testFailedNewestKeepsPlayingAndIsNotRetriedUntilItChanges() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let first = try models.write(run, localStep: 1000, at: 2_000, name: "step1000.safetensors")
        let reads = SyncBox(0)
        let time = LichessBotManualTime()
        let follow = settings(following: followed(first.record))
        let slots = try await prepare(follow, scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: time, loader: countingLoader(reads: reads))
        XCTAssertEqual(reads.value, 1)

        let newer = try models.encode(run, localStep: 2000, at: 3_000)
        let newerURL = models.folder.appendingPathComponent("step2000.safetensors")
        try corrupt(newer.bytes).write(to: newerURL)
        time.advance(by: .seconds(60))
        do {
            try await slots.refreshIfDue(for: follow)
            XCTFail("the failed load throws once")
        } catch LichessBotLineageFollowError.fileFailedToLoad {}
        XCTAssertEqual(reads.value, 2)

        for _ in 0..<2 {
            time.advance(by: .seconds(60))
            let outcome = try await slots.refreshIfDue(for: follow)
            XCTAssertEqual(outcome, .unchanged, "reported, not thrown again")
        }
        XCTAssertEqual(reads.value, 2, "not retried while the file is unchanged")
        let info = await slots.current.info
        XCTAssertEqual(info.generationID, 1, "the last good generation keeps playing")
        let status = await slots.lineageFollowStatus
        guard case .newestFailedToLoad(let file, _)? = status?.outcome else {
            return XCTFail("expected newestFailedToLoad, got \(String(describing: status?.outcome))")
        }
        XCTAssertEqual(file.segmentLocalStep, 2000)

        // Rewritten (a new file at the path): retried, and it loads.
        try newer.bytes.write(to: newerURL, options: .atomic)
        time.advance(by: .seconds(60))
        let outcome = try await slots.refreshIfDue(for: follow)
        guard case .built(let built) = outcome else {
            return XCTFail("expected a build, got \(outcome)")
        }
        XCTAssertEqual(built.trainingStep, 2000)
    }

    func testFailedNewestThrowsOnceThenReportsWithoutThrowing() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let first = try models.write(run, localStep: 1000, at: 2_000, name: "step1000.safetensors")
        let time = LichessBotManualTime()
        let follow = settings(following: followed(first.record))
        let lines = SyncBox<[String]>([])
        let slots = try await prepare(follow, scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: time, lines: lines)
        let newer = try models.encode(run, localStep: 2000, at: 3_000)
        try corrupt(newer.bytes).write(to: models.folder.appendingPathComponent("step2000.safetensors"))

        var thrown = 0
        for _ in 0..<4 {
            time.advance(by: .seconds(60))
            do {
                try await slots.refreshIfDue(for: follow)
            } catch LichessBotLineageFollowError.fileFailedToLoad {
                thrown += 1
            }
        }
        XCTAssertEqual(thrown, 1, "one alarm for the file, not one per check")
        XCTAssertEqual(lines.value.filter { $0.contains("step2000.safetensors could not be loaded") && $0.contains("not retried until it changes; still playing generation 1") }.count, 1, "\(lines.value)")
        let status = await slots.lineageFollowStatus
        guard case .newestFailedToLoad? = status?.outcome else {
            return XCTFail("expected newestFailedToLoad, got \(String(describing: status?.outcome))")
        }
    }

    func testNetworkBuildFailureDoesNotMarkTheFileFailed() async throws {
        struct NetworkBuildFailed: Error {}
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let first = try models.write(run, localStep: 1000, at: 2_000, name: "step1000.safetensors")
        let builds = SyncBox(0)
        let factory = LichessBotInferenceNetworkFactory(build: { weights, architecture in
            let attempt = builds.mutate { count -> Int in
                count += 1
                return count
            }
            if attempt == 2 {
                throw NetworkBuildFailed()
            }
            return try await InferenceNetworkFactory.build(loading: weights, arch: architecture)
        })
        let time = LichessBotManualTime()
        let follow = settings(following: followed(first.record))
        let slots = try await prepare(follow, scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: time, networkFactory: factory)

        try models.write(run, localStep: 2000, at: 3_000, name: "step2000.safetensors")
        time.advance(by: .seconds(60))
        do {
            try await slots.refreshIfDue(for: follow)
            XCTFail("the network build failure throws")
        } catch is NetworkBuildFailed {}
        // The next poll retries the build at once: the file is not marked.
        let outcome = try await slots.refreshIfDue(for: follow)
        guard case .built(let built) = outcome else {
            return XCTFail("expected a build, got \(outcome)")
        }
        XCTAssertEqual(built.trainingStep, 2000)
        XCTAssertEqual(builds.value, 3)
    }

    // MARK: - Verifying the file loaded

    func testReplacedRollingFileLoadsTheNewerSameLineageFile() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let rolling = try models.write(run, localStep: 1000, at: 2_000, name: "run-replay-latest.safetensors")
        // The next save is renamed over the rolling path between the check
        // and the load.
        let next = try models.encode(run, localStep: 2000, at: 3_000)
        let substitutes = SyncBox<[URL: Data]>([:])
        let reads = SyncBox(0)
        let lines = SyncBox<[String]>([])
        let follow = settings(following: followed(rolling.record))
        let scanner = LichessBotScriptedFolderScanner(directory: models.folder)
        let listedRolling = try await scanner.scan(previous: .empty).entries[0].url
        substitutes.value = [listedRolling: next.bytes]
        let slots = try await prepare(follow, scanner: scanner, time: LichessBotManualTime(), loader: countingLoader(reads: reads, substitutes: substitutes), lines: lines)
        let info = await slots.current.info
        XCTAssertEqual(info.trainingStep, 2000, "the generation records what was loaded")
        XCTAssertEqual(info.lineage?.segmentLocalStep, 2000)
        XCTAssertTrue(lines.value.contains { $0.contains("(file changed since the check: loaded segment step 2000)") }, "\(lines.value)")
    }

    func testFileOfAnotherLineageAtTheSamePathIsRefused() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let other = try LineageTestRuns.fresh(modelID: "20261006-2-OTHR", startedUnix: 1_000)
        let rolling = try models.write(run, localStep: 1000, at: 2_000, name: "out.safetensors")
        let intruder = try models.encode(other, localStep: 5000, at: 3_000)
        let scanner = LichessBotScriptedFolderScanner(directory: models.folder)
        let listed = try await scanner.scan(previous: .empty).entries[0].url
        let reads = SyncBox(0)
        let loader = countingLoader(reads: reads, substitutes: SyncBox([listed: intruder.bytes]))
        do {
            _ = try await prepare(settings(following: followed(rolling.record)), scanner: scanner, time: LichessBotManualTime(), loader: loader)
            XCTFail("another run's file at the selected path must be refused")
        } catch LichessBotLineageFollowError.followedFileChangedDuringLoad(_, let reason) {
            XCTAssertTrue(reason.contains("belongs to run"), reason)
        }
    }

    func testSiblingSegmentInTheRollingPathFailsVerification() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let first = try LineageTestRuns.fresh(modelID: "20261006-1-SEG0", startedUnix: 1_000)
        let anchorSave = try first.record(localStep: 3000, at: 2_000)
        let left = try LineageTestRuns.resume(from: anchorSave, parentModelID: first.modelID, parentSHA256: "s0", modelID: "20261006-2-LEFT", startedUnix: 3_000)
        let right = try LineageTestRuns.resume(from: anchorSave, parentModelID: first.modelID, parentSHA256: "s0", modelID: "20261006-3-RGHT", startedUnix: 3_100)
        // Only the left sibling's file is on disk when checked; the right
        // sibling's save then takes the shared --out-model path.
        let leftFile = try models.write(left, localStep: 1000, at: 4_000, name: "shared-out.safetensors")
        let rightSave = try models.encode(right, localStep: 1500, at: 4_200)
        let scanner = LichessBotScriptedFolderScanner(directory: models.folder)
        let listed = try await scanner.scan(previous: .empty).entries[0].url
        let loader = countingLoader(reads: SyncBox(0), substitutes: SyncBox([listed: rightSave.bytes]))
        do {
            _ = try await prepare(settings(following: followed(anchorSave)), scanner: scanner, time: LichessBotManualTime(), loader: loader)
            XCTFail("a sibling segment's file must not pass for the selected one")
        } catch LichessBotLineageFollowError.followedFileChangedDuringLoad(_, let reason) {
            XCTAssertTrue(reason.contains("not the selected segment \(leftFile.record.run.segmentID)"), reason)
        }
    }

    // MARK: - Attribution, sharing, joining

    func testGenerationRecordsLineagePosition() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let first = try LineageTestRuns.fresh(modelID: "20261006-1-SEG0", startedUnix: 1_000)
        let handoff = try models.write(first, localStep: 3000, at: 2_000, name: "seg0.safetensors")
        let second = try LineageTestRuns.resume(from: handoff.record, parentModelID: first.modelID, parentSHA256: handoff.contentSHA256, modelID: "20261006-2-SEG1", startedUnix: 3_000)
        let later = try models.write(second, localStep: 500, at: 4_000, name: "seg1.safetensors")
        let slots = try await prepare(settings(following: followed(handoff.record)), scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: LichessBotManualTime())
        let followedInfo = await slots.current.info
        let lineage = try XCTUnwrap(followedInfo.lineage)
        XCTAssertEqual(lineage.lineageRunID, later.record.run.lineageRunID)
        XCTAssertEqual(lineage.segmentID, later.record.run.segmentID)
        XCTAssertEqual(lineage.segmentIndex, 1)
        XCTAssertEqual(lineage.segmentChain, [handoff.record.run.segmentID, later.record.run.segmentID])
        XCTAssertEqual(lineage.segmentLocalStep, 500)
        XCTAssertEqual(lineage.cumTrainerStep, 3500)
        XCTAssertEqual(lineage.recordedUnix, 4_000)
        XCTAssertEqual(lineage.contentSHA256, later.contentSHA256)
        XCTAssertEqual(lineage.followed, followed(handoff.record))

        // The fixed-file source records the file's position too, following
        // nothing.
        var fileSettings = LichessBotModelSettings.testBaseline()
        fileSettings.source = .file
        fileSettings.filePath = later.url.path
        let fileSlots = try await prepare(fileSettings, scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: LichessBotManualTime())
        let fileInfo = await fileSlots.current.info
        let fileLineage = try XCTUnwrap(fileInfo.lineage)
        XCTAssertEqual(fileLineage.segmentID, later.record.run.segmentID)
        XCTAssertNil(fileLineage.followed)
    }

    func testConcurrentCallersShareOneScan() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let file = try models.write(run, localStep: 1000, at: 2_000, name: "a.safetensors")
        let scanner = LichessBotScriptedFolderScanner(directory: models.folder)
        let time = LichessBotManualTime()
        let follow = settings(following: followed(file.record))
        let slots = try await prepare(follow, scanner: scanner, time: time)
        time.advance(by: .seconds(60))
        scanner.holdScans()
        let poll = Task { try await slots.refreshIfDue(for: follow) }
        try await waitUntil("the poll's scan is held") { scanner.scansHeld.value == 1 }
        let checkNow = Task { try await slots.checkLineageNow(for: follow) }
        try await Task.sleep(for: .milliseconds(200))
        scanner.releaseScans()
        _ = try await poll.value
        _ = try await checkNow.value
        XCTAssertEqual(scanner.scans.value, 2, "the first check before going online, then one scan shared by both callers")
    }

    func testInFlightBuildIsJoinedAcrossAnIntervalEdit() async throws {
        let models = try await LichessBotLineageModelFolder.make(for: self)
        let run = try LineageTestRuns.fresh(modelID: "20261006-1-RUNA", startedUnix: 1_000)
        let first = try models.write(run, localStep: 1000, at: 2_000, name: "step1000.safetensors")
        let reads = SyncBox(0)
        let gate = DispatchSemaphore(value: 0)
        let holdNextRead = SyncBox(false)
        let loader = LichessBotModelFileLoader(readBytes: { url in
            reads.modify { $0 += 1 }
            if holdNextRead.value {
                holdNextRead.value = false
                gate.wait()
            }
            return try LichessBotModelFileLoader.live.readBytes(url)
        })
        let time = LichessBotManualTime()
        let follow = settings(following: followed(first.record), interval: 60)
        let slots = try await prepare(follow, scanner: LichessBotScriptedFolderScanner(directory: models.folder), time: time, loader: loader)
        try models.write(run, localStep: 2000, at: 3_000, name: "step2000.safetensors")
        time.advance(by: .seconds(60))
        holdNextRead.value = true
        let poll = Task { try await slots.refreshIfDue(for: follow) }
        try await waitUntil("the build's read is held") { reads.value == 2 }
        var edited = follow
        edited.lineageCheckIntervalSeconds = 90
        let second = Task { try await slots.refreshIfDue(for: edited) }
        try await Task.sleep(for: .milliseconds(200))
        gate.signal()
        let pollOutcome = try await poll.value
        let secondOutcome = try await second.value
        guard case .built(let built) = pollOutcome, case .joinedBuildInFlight(let joined) = secondOutcome else {
            return XCTFail("expected built then joined, got \(pollOutcome) and \(secondOutcome)")
        }
        XCTAssertEqual(built, joined)
        XCTAssertEqual(reads.value, 2, "one build serves both")
    }

    func testOldGenerationInfoJSONStillDecodes() throws {
        // A game record's generation as builds before `lineage` wrote it.
        let json = #"{"generationID":3,"sourceKind":"file","modelID":"20261001-1-AAAA","trainingStep":270000,"snapshotAt":781000000,"architectureSummary":"5x128","filePath":"/m/a.safetensors","fileSHA256":"abc","valueHeadRecenteredOnLoad":false}"#
        let info = try JSONDecoder().decode(LichessBotGenerationInfo.self, from: Data(json.utf8))
        XCTAssertNil(info.lineage)
        XCTAssertEqual(info.generationID, 3)

        let lineage = LichessBotGenerationLineage(
            lineageRunID: "R", segmentID: "S1", segmentIndex: 1, segmentChain: ["S0", "S1"], segmentLocalStep: 500,
            recordedUnix: 4_000, cumTrainerStep: nil, contentSHA256: "sha",
            followed: LichessBotFollowedLineage(lineageRunID: "R", anchorSegmentID: "S0"))
        var withLineage = info
        withLineage.lineage = lineage
        let decoded = try JSONDecoder().decode(LichessBotGenerationInfo.self, from: try JSONEncoder().encode(withLineage))
        XCTAssertEqual(decoded, withLineage)
    }

    private func waitUntil(_ description: String, _ condition: () async -> Bool) async throws {
        for _ in 0..<4000 {
            if await condition() { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }
}
