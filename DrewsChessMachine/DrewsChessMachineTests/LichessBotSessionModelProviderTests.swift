import Metal
import XCTest
@testable import DrewsChessMachine

/// The Lichess bot's champion snapshot is attributed only when no champion
/// weight replacement (a promotion or load) was open before the export or
/// overlapped it — the model ID alone cannot tell, since every checkpoint of
/// a run shares one.
final class LichessBotSessionModelProviderTests: XCTestCase {

    /// A small tower, so the champion builds quickly.
    private static let architecture = NetworkArchitecture(
        inputEncoding: .basic30, channels: 16, numBlocks: 1, stemConvKernelSize: 3,
        activationFunction: .relu, blockActivationStyle: .pre,
        blockSkipMerge: .cleanAdd, blockUseRezero: true, rezeroAlphaInit: 0.5,
        blockConv1KernelSize: 3, blockConv2KernelSize: 3,
        blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
        policyHeadStyle: .intermediateConv, policyPreConvChannels: 16,
        valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 16,
        computeDataType: .float32
    )

    @MainActor
    func testAnOpenChampionReplacementRefusesTheSnapshotUntilItsOriginIsRecorded() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
        let session = SessionController()
        let champion = try ChessMPSNetwork(.randomWeights(initSeed: 21), arch: Self.architecture)
        champion.identifier = ModelID(value: "20261007-1-CHMP")
        session.network = champion
        session.championOrigin = .built(initialization: .forTests, naming: .unnamedWithoutPreset)
        let provider = LichessBotSessionModelProvider()
        provider.attach(session: session)

        session.noteChampionWeightsReplaced()
        do {
            _ = try await provider.championSnapshot()
            XCTFail("a champion awaiting its origin must not be snapshotted")
        } catch let error as LichessBotSessionModelError {
            XCTAssertEqual(error, .championChangedDuringExport)
        }

        session.championOrigin = .built(initialization: .forTests, naming: .unnamedWithoutPreset)
        let snapshot = try await provider.championSnapshot()
        XCTAssertEqual(snapshot.modelID, "20261007-1-CHMP")
        let exported = try await champion.exportWeights()
        XCTAssertEqual(snapshot.weights, exported)
    }
}
