import CryptoKit
import XCTest
@testable import DrewsChessMachine

/// A model file's recorded hash names exactly the bytes that were decoded
/// (follow-lineage plan §3.5). A rolling `--out-model` file is renamed over
/// at every save, so a loader that decodes one read and hashes a second can
/// record a hash of weights that were never played.
final class LichessBotModelFileLoadTests: XCTestCase {

    private func modelBytes(modelID: String, seed: UInt64) async throws -> Data {
        let network = try ChessMPSNetwork(.randomWeights(initSeed: seed))
        let weights = try await network.exportWeights()
        return try SafetensorsModelIO.encode(
            modelID: modelID,
            createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata(creator: "replay", trainingStep: 1000, parentModelID: "", notes: "file load test"),
            weights: weights,
            architecture: network.arch,
            includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil)
        )
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// The path is read twice in a row while a save renames a new file over
    /// it in between: the first read sees file A, any later read file B.
    func testRecordedHashIsOfTheBytesThatWereDecoded() async throws {
        let first = try await modelBytes(modelID: "20261006-1-AAAA", seed: 11)
        let second = try await modelBytes(modelID: "20261006-2-BBBB", seed: 12)
        XCTAssertNotEqual(sha256(first), sha256(second))
        let reads = SyncBox(0)
        let loader = LichessBotModelFileLoader(readBytes: { _ in
            let read = reads.mutate { count -> Int in
                count += 1
                return count
            }
            return read == 1 ? first : second
        })
        let loaded = try await loader.load(at: URL(fileURLWithPath: "/models/run-replay-latest.safetensors"))
        XCTAssertEqual(loaded.snapshot.modelID, "20261006-1-AAAA", "the first read is what was decoded")
        XCTAssertEqual(loaded.sha256, sha256(first), "the recorded hash names the decoded bytes")
        XCTAssertEqual(reads.value, 1, "the file is read once")
    }
}
