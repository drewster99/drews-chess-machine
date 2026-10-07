import XCTest
@testable import DrewsChessMachine

/// Champion and trainer weights from one real random-weight network, with a
/// champion and a trainer the test can take away, and snapshots it can hold
/// on a latch (one held snapshot at a time).
final class LichessBotHoldableModelProvider: LichessBotModelProvider, @unchecked Sendable {
    static let championModelID = "20261006-1-CHMP"
    static let trainerModelID = "20261006-1-TRNR"

    /// The champion's ModelID; nil means no champion exists.
    let championID = SyncBox<String?>(LichessBotHoldableModelProvider.championModelID)
    let trainerExists = SyncBox(true)
    let championSnapshots = SyncBox(0)
    let trainerSnapshots = SyncBox(0)
    /// Snapshots that have reached the hold and are waiting on it.
    let snapshotsHeld = SyncBox(0)
    private let hold = SyncBox<LichessBotTestLatch?>(nil)
    private let weights: [[Float]]
    private let architecture: NetworkArchitecture

    private init(weights: [[Float]], architecture: NetworkArchitecture) {
        self.weights = weights
        self.architecture = architecture
    }

    static func make() async throws -> LichessBotHoldableModelProvider {
        let network = try ChessMPSNetwork(.randomWeights(initSeed: 3))
        let weights = try await network.exportWeights()
        return LichessBotHoldableModelProvider(weights: weights, architecture: network.arch)
    }

    /// From now on, snapshots wait until `releaseSnapshots()`.
    func holdSnapshots() {
        hold.value = LichessBotTestLatch()
    }

    func releaseSnapshots() {
        let latch = hold.mutate { held -> LichessBotTestLatch? in
            defer { held = nil }
            return held
        }
        latch?.open()
    }

    private func waitIfHeld() async {
        guard let latch = hold.value else { return }
        snapshotsHeld.modify { $0 += 1 }
        await latch.wait()
    }

    func championModelID() async -> String? {
        championID.value
    }

    func championSnapshot() async throws -> LichessBotWeightsSnapshot {
        championSnapshots.modify { $0 += 1 }
        await waitIfHeld()
        guard let modelID = championID.value else { throw LichessBotModelError.noChampion }
        return LichessBotWeightsSnapshot(weights: weights, architecture: architecture, modelID: modelID, trainingStep: nil)
    }

    func trainerSnapshot() async throws -> LichessBotWeightsSnapshot {
        let step = trainerSnapshots.mutate { count -> Int in
            count += 1
            return count
        }
        await waitIfHeld()
        guard trainerExists.value else { throw LichessBotModelError.noTrainer }
        return LichessBotWeightsSnapshot(weights: weights, architecture: architecture, modelID: Self.trainerModelID, trainingStep: step)
    }
}

/// Model files for tests: a real random-weight network written the way the
/// app writes one.
enum LichessBotTestModelFiles {
    static let modelID = "20261006-1-FILE"

    /// Writes a model file of a random-weight network to `url`.
    static func writeRandomModel(to url: URL, trainingStep: Int = 0) async throws {
        let network = try ChessMPSNetwork(.randomWeights(initSeed: 5))
        let weights = try await network.exportWeights()
        let data = try SafetensorsModelIO.encode(
            modelID: modelID,
            createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata(creator: "manual", trainingStep: trainingStep, parentModelID: "", notes: "lichess bot test"),
            weights: weights,
            architecture: network.arch,
            includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil)
        )
        try data.write(to: url, options: .withoutOverwriting)
    }
}

/// `LichessBotModelSlots.prepare` and source switching (follow-lineage plan
/// §3.10): slots never exist without a generation, a source change builds
/// the new generation while the old one keeps playing, and a failed switch
/// or a vanished champion keeps the old one and throws.
final class LichessBotModelSlotsPrepareTests: XCTestCase {

    private func settings(_ source: LichessBotModelSourceKind, filePath: String? = nil, liveTrainerInterval: Int = 120) -> LichessBotModelSettings {
        var settings = LichessBotModelSettings.testBaseline()
        settings.source = source
        settings.filePath = filePath
        settings.liveTrainerRefreshIntervalSeconds = liveTrainerInterval
        return settings
    }

    private func temporaryFolder() throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotModelSlotsPrepareTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        addTeardownBlock {
            do {
                try FileManager.default.removeItem(at: folder)
            } catch {
                XCTFail("cleanup failed: \(error)")
            }
        }
        return folder
    }

    private func prepare(_ settings: LichessBotModelSettings, provider: any LichessBotModelProvider, time: LichessBotManualTime = LichessBotManualTime()) async throws -> LichessBotModelSlots {
        try await LichessBotModelSlots.prepare(for: settings, provider: provider, time: time, folderScanner: LichessBotNoModelsFolderScanner(), log: { _ in })
    }

    private func waitUntil(_ description: String, _ condition: () async -> Bool) async throws {
        for _ in 0..<4000 {
            if await condition() { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    func testPrepareBuildsTheFirstGeneration() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        for source in [LichessBotModelSourceKind.champion, .trainerSnapshot, .liveTrainer] {
            let slots = try await prepare(settings(source), provider: provider)
            let info = await slots.current.info
            XCTAssertEqual(info.generationID, 1, "\(source)")
            XCTAssertEqual(info.sourceKind, source)
            XCTAssertEqual(info.modelID, source == .champion ? LichessBotHoldableModelProvider.championModelID : LichessBotHoldableModelProvider.trainerModelID)
        }
        let file = try temporaryFolder().appendingPathComponent("model.safetensors")
        try await LichessBotTestModelFiles.writeRandomModel(to: file, trainingStep: 7)
        let slots = try await prepare(settings(.file, filePath: file.path), provider: provider)
        let info = await slots.current.info
        XCTAssertEqual(info.generationID, 1)
        XCTAssertEqual(info.sourceKind, .file)
        XCTAssertEqual(info.modelID, LichessBotTestModelFiles.modelID)
        XCTAssertEqual(info.trainingStep, 7)
        XCTAssertEqual(info.filePath, file.path)
        XCTAssertNotNil(info.fileSHA256)
    }

    func testPrepareThrowsWhenTheSourceCannotBuild() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        provider.championID.value = nil
        provider.trainerExists.value = false
        do {
            _ = try await prepare(settings(.champion), provider: provider)
            XCTFail("no champion: prepare must throw")
        } catch {
            XCTAssertEqual(error as? LichessBotModelError, .noChampion)
        }
        for source in [LichessBotModelSourceKind.trainerSnapshot, .liveTrainer] {
            do {
                _ = try await prepare(settings(source), provider: provider)
                XCTFail("no trainer: prepare must throw for \(source)")
            } catch {
                XCTAssertEqual(error as? LichessBotModelError, .noTrainer)
            }
        }
        do {
            _ = try await prepare(settings(.file, filePath: nil), provider: provider)
            XCTFail("no file: prepare must throw")
        } catch {
            XCTAssertEqual(error as? LichessBotModelError, .noFileSelected)
        }
        let missing = try temporaryFolder().appendingPathComponent("absent.safetensors")
        do {
            _ = try await prepare(settings(.file, filePath: missing.path), provider: provider)
            XCTFail("a missing file: prepare must throw")
        } catch CheckpointManagerError.readFailed(let url, _) {
            XCTAssertEqual(url, missing, "the loader's own error names the file")
        }
    }

    func testSourceChangeKeepsTheOldGenerationUntilTheNewOneIsBuilt() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        let slots = try await prepare(settings(.champion), provider: provider)
        provider.holdSnapshots()
        let trainer = settings(.trainerSnapshot)
        let switching = Task {
            try await slots.refreshIfDue(for: trainer)
        }
        try await waitUntil("the trainer snapshot is held") { provider.snapshotsHeld.value == 1 }
        let during = await slots.current.info
        XCTAssertEqual(during.generationID, 1)
        XCTAssertEqual(during.sourceKind, .champion, "new games keep the old generation while the new one builds")
        provider.releaseSnapshots()
        try await switching.value
        let after = await slots.current.info
        XCTAssertEqual(after.generationID, 2)
        XCTAssertEqual(after.sourceKind, .trainerSnapshot)
    }

    func testFailedSourceSwitchKeepsTheOldGenerationAndThrows() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        let slots = try await prepare(settings(.champion), provider: provider)
        provider.trainerExists.value = false
        let trainer = settings(.liveTrainer)
        do {
            try await slots.refreshIfDue(for: trainer)
            XCTFail("a switch to a source that can't build must throw")
        } catch {
            XCTAssertEqual(error as? LichessBotModelError, .noTrainer)
        }
        let kept = await slots.current.info
        XCTAssertEqual(kept.generationID, 1)
        XCTAssertEqual(kept.sourceKind, .champion)

        // The next attempt (the poll loop retries at its backoff) switches.
        provider.trainerExists.value = true
        try await slots.refreshIfDue(for: trainer)
        let switched = await slots.current.info
        XCTAssertEqual(switched.sourceKind, .liveTrainer)
        XCTAssertEqual(switched.generationID, 2, "a failed build uses up no generation number")
    }

    func testChampionGoneWhileOnlineKeepsPlayingAndThrows() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        let champion = settings(.champion)
        let slots = try await prepare(champion, provider: provider)
        provider.championID.value = nil
        do {
            try await slots.refreshIfDue(for: champion)
            XCTFail("a vanished champion must throw, so the controller alarms")
        } catch {
            XCTAssertEqual(error as? LichessBotModelError, .noChampion)
        }
        let kept = await slots.current.info
        XCTAssertEqual(kept.generationID, 1)
        XCTAssertEqual(kept.modelID, LichessBotHoldableModelProvider.championModelID)
    }

    /// An interval edit while online never stops the live trainer's
    /// refreshes (the stall half of the settings-comparison bug, plan §3.4).
    func testIntervalEditKeepsTheLiveTrainerRefreshing() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        let time = LichessBotManualTime()
        let slots = try await prepare(settings(.liveTrainer, liveTrainerInterval: 120), provider: provider, time: time)
        let edited = settings(.liveTrainer, liveTrainerInterval: 60)
        try await slots.refreshIfDue(for: edited)
        let snapshotsAfterEdit = provider.trainerSnapshots.value
        time.advance(by: .seconds(61))
        try await slots.refreshIfDue(for: edited)
        XCTAssertEqual(provider.trainerSnapshots.value, snapshotsAfterEdit + 1, "the edited interval elapsed, so the trainer is snapshotted again")
        let info = await slots.current.info
        XCTAssertEqual(info.trainingStep, provider.trainerSnapshots.value, "the newest snapshot is the one playing")
    }
}
