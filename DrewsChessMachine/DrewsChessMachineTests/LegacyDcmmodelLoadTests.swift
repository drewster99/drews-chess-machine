//
//  LegacyDcmmodelLoadTests.swift
//  DrewsChessMachineTests
//
//  End-to-end validation of Phase F: a real, on-disk legacy `.dcmmodel` (an
//  anonymous/positional historical file) must resolve its stored archHash to a
//  built-in preset, rebuild THAT architecture, and load its weights positionally
//  with the exact tensor count + per-tensor shapes the rebuilt net expects. A
//  successful load proves the historical builder's weight ORDER matches the file.
//
//  Skips when no legacy files are present (e.g. CI), so it never fails a clean
//  environment — but on the dev machine it exercises the real 8-block-v3 /
//  12-block-v4 models.
//

import XCTest
@testable import DrewsChessMachine

final class LegacyDcmmodelLoadTests: XCTestCase {

    /// The stored archHash (u32 LE at byte offset 12) of a `.dcmmodel`, used to
    /// pick ONE representative file per distinct architecture so the test stays
    /// fast (a GPU build + round-trip per file is expensive; there can be ~100
    /// sessions on disk, but only a handful of distinct arches).
    private func storedArchHash(of url: URL) -> UInt32? {
        guard let h = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? h.close() }
        guard (try? h.seek(toOffset: 12)) != nil,
              let data = try? h.read(upToCount: 4), data.count == 4 else { return nil }
        return data.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian
    }

    /// Collect real `.dcmmodel` champion files (Models/ + each session's
    /// `champion.dcmmodel`), then keep one per distinct stored archHash so each
    /// historical architecture is exercised exactly once.
    private func legacyChampionModelURLs() -> [URL] {
        let fm = FileManager.default
        var urls: [URL] = []
        if let entries = try? fm.contentsOfDirectory(at: CheckpointPaths.modelsDir, includingPropertiesForKeys: nil) {
            urls += entries.filter { $0.pathExtension == "dcmmodel" }
        }
        if let sessions = try? fm.contentsOfDirectory(at: CheckpointPaths.sessionsDir, includingPropertiesForKeys: nil) {
            for dir in sessions where dir.pathExtension == "dcmsession" {
                let champ = dir.appendingPathComponent("champion.dcmmodel")
                if fm.fileExists(atPath: champ.path) { urls.append(champ) }
            }
        }
        var seenHashes = Set<UInt32>()
        var unique: [URL] = []
        for url in urls {
            let hash = storedArchHash(of: url) ?? 0
            if seenHashes.insert(hash).inserted { unique.append(url) }
        }
        return unique
    }

    func testRealLegacyDcmmodelsResolveBuildAndLoad() async throws {
        // Opt-in only: this is slow (a GPU network build + safetensors round-trip
        // per distinct historical architecture) and depends on real files on
        // disk, so it would otherwise push the whole suite past the 10-minute
        // build+run cap. Run on demand with:
        //   DCM_RUN_LEGACY_LOAD=1  (scheme env var, or
        //   `xcodebuild test -only-testing:DrewsChessMachineTests/LegacyDcmmodelLoadTests`
        //   with that env set — see tools/verify-legacy-load.sh).
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["DCM_RUN_LEGACY_LOAD"] == "1",
            "Set DCM_RUN_LEGACY_LOAD=1 to run the legacy-load validation (slow, GPU, real-file)."
        )
        let urls = legacyChampionModelURLs()
        guard !urls.isEmpty else {
            throw XCTSkip("No legacy .dcmmodel files on disk to validate.")
        }

        var loadedCount = 0
        for url in urls {
            // decode: maps the stored archHash -> historical preset, sets
            // file.architecture to that rebuilt config.
            let file = try CheckpointManager.loadModelFile(at: url)

            // Build the resolved architecture and load the file's weights into it.
            let net = try ChessMPSNetwork(.randomWeights, arch: file.architecture)
            let base = net.network.trainableVariables.count
                + net.network.bnRunningStatsVariables.count

            XCTAssertGreaterThanOrEqual(
                file.weights.count, base,
                "\(url.lastPathComponent): file has \(file.weights.count) tensors, net needs \(base)")

            // loadWeights validates per-tensor shape against the live variable
            // list; a mismatch (wrong builder order/shape) throws here.
            try await net.network.loadWeights(Array(file.weights.prefix(base)))

            // Cross-format round-trip: original .dcmmodel weights -> encode as
            // safetensors (PyTorch layout, FC transposed) -> decode back ->
            // bit-compare vs the original load. Proves the old->new bridge is
            // lossless and that every historical tensor's WeightKind (so its
            // transpose) is classified correctly.
            let plan = file.architecture.weightTensorPlan()
            let trainablesCount = plan.filter { $0.kind != .bnRunningStat }.count
            let includesVelocity: Bool
            if file.weights.count == plan.count {
                includesVelocity = false
            } else if file.weights.count == plan.count + trainablesCount {
                includesVelocity = true
            } else {
                XCTFail("\(url.lastPathComponent): unexpected weight count \(file.weights.count) (plan \(plan.count))")
                continue
            }
            let bytes = try SafetensorsModelIO.encode(
                modelID: file.modelID,
                createdAtUnix: file.createdAtUnix,
                metadata: file.metadata,
                weights: file.weights,
                architecture: file.architecture,
                includesVelocity: includesVelocity)
            let reloaded = try SafetensorsModelIO.decode(bytes)
            XCTAssertEqual(reloaded.architecture, file.architecture,
                           "\(url.lastPathComponent): architecture changed across formats")
            XCTAssertEqual(reloaded.file.weights.count, file.weights.count,
                           "\(url.lastPathComponent): tensor count changed across formats")
            for (orig, rt) in zip(file.weights, reloaded.file.weights) {
                XCTAssertEqual(orig.map(\.bitPattern), rt.map(\.bitPattern),
                               "\(url.lastPathComponent): weight data changed across .dcmmodel->safetensors round-trip")
            }
            loadedCount += 1
        }
        XCTAssertGreaterThan(loadedCount, 0)
        print("[LegacyLoad] validated \(loadedCount) legacy .dcmmodel file(s)")
    }
}
