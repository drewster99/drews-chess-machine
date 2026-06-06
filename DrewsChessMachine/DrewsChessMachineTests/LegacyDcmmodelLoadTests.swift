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

    /// Collect real `.dcmmodel` files: the standalone Models/ folder plus each
    /// session's `champion.dcmmodel` (champion = trainables+running, no velocity).
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
        return urls
    }

    func testRealLegacyDcmmodelsResolveBuildAndLoad() async throws {
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
            loadedCount += 1
        }
        XCTAssertGreaterThan(loadedCount, 0)
        print("[LegacyLoad] validated \(loadedCount) legacy .dcmmodel file(s)")
    }
}
